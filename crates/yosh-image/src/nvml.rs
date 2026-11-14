//! NVIDIA GPU detection through NVML (`nvml.dll` / `libnvidia-ml.so.1`, installed with the
//! driver), loaded at run time: no driver, no NVIDIA GPU, an empty list. Unlike libaji's device
//! lists this needs nothing from the AI add-on, so it can tell which GPUs an add-on lacks files
//! for. The same detection `mangajanai setup` uses to pick its runtime packs.

use std::ffi::{c_char, c_int, c_uint, c_void};

#[derive(Clone, Debug, PartialEq)]
pub struct Gpu {
    pub name: String,
    /// CUDA compute capability, e.g. (8, 6) for an RTX 30 series card.
    pub major: i32,
    pub minor: i32,
}

type Device = *mut c_void;

/// The NVIDIA GPUs, or an empty list when there is no NVIDIA driver (or NVML fails).
pub fn gpus() -> Vec<Gpu> {
    detect().unwrap_or_default()
}

fn detect() -> Option<Vec<Gpu>> {
    let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
    // SAFETY: NVML's initializers are the driver's own; nothing else runs at load.
    let lib = unsafe { libloading::Library::new(name) }.ok()?;
    // SAFETY: the symbols have these C signatures in every NVML since the _v2 entry points (2014).
    unsafe {
        let init: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlInit_v2\0").ok()?;
        let shutdown: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlShutdown\0").ok()?;
        let count: libloading::Symbol<unsafe extern "C" fn(*mut c_uint) -> c_int> =
            lib.get(b"nvmlDeviceGetCount_v2\0").ok()?;
        let handle: libloading::Symbol<unsafe extern "C" fn(c_uint, *mut Device) -> c_int> =
            lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
        let cc: libloading::Symbol<unsafe extern "C" fn(Device, *mut c_int, *mut c_int) -> c_int> =
            lib.get(b"nvmlDeviceGetCudaComputeCapability\0").ok()?;
        let dev_name: libloading::Symbol<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int> =
            lib.get(b"nvmlDeviceGetName\0").ok()?;
        if init() != 0 {
            return None;
        }
        let mut out = Vec::new();
        let mut n: c_uint = 0;
        if count(&mut n) == 0 {
            for i in 0..n {
                let mut d: Device = std::ptr::null_mut();
                if handle(i, &mut d) != 0 {
                    continue;
                }
                let (mut major, mut minor) = (0, 0);
                if cc(d, &mut major, &mut minor) != 0 {
                    continue;
                }
                let mut buf = [0 as c_char; 96];
                let name = if dev_name(d, buf.as_mut_ptr(), buf.len() as c_uint) == 0 {
                    std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
                } else {
                    "NVIDIA GPU".into()
                };
                out.push(Gpu { name, major, minor });
            }
        }
        shutdown();
        Some(out)
    }
}

/// NVML's handle on the driver, opened on first use and kept for the process.
struct Nvml {
    _lib: libloading::Library,
    by_bus_id: unsafe extern "C" fn(*const c_char, *mut Device) -> c_int,
    memory: unsafe extern "C" fn(Device, *mut [u64; 3]) -> c_int,
}

// SAFETY: NVML is thread-safe once initialized; the function pointers are plain C entry points.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

fn nvml() -> Option<&'static Nvml> {
    static NVML: std::sync::OnceLock<Option<Nvml>> = std::sync::OnceLock::new();
    NVML.get_or_init(|| {
        let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
        // SAFETY: as in `detect`; nvmlDeviceGetHandleByPciBusId_v2 and nvmlDeviceGetMemoryInfo
        // have these signatures since NVML 5.319 (2013). Initialized once, never shut down.
        unsafe {
            let lib = libloading::Library::new(name).ok()?;
            let init: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlInit_v2\0").ok()?;
            let by_bus_id = *lib.get(b"nvmlDeviceGetHandleByPciBusId_v2\0").ok()?;
            let memory = *lib.get(b"nvmlDeviceGetMemoryInfo\0").ok()?;
            (init() == 0).then_some(Nvml { _lib: lib, by_bus_id, memory })
        }
    })
    .as_ref()
}

/// The free and total memory of the NVIDIA GPU with libaji's device id `id` (its PCI location:
/// bit 63 set, domain in bits 32-62, bus in 8-15, device in 0-7), counting every process's use:
/// on Windows, libaji's own figure (CUDA's free memory, DirectML's budget) leaves out other
/// processes', which WDDM would page out to make room at a heavy cost in speed. `None` without an
/// NVIDIA driver, or for an id that isn't a PCI location (a DirectML adapter LUID).
pub fn memory(id: u64) -> Option<(u64, u64)> {
    if id >> 63 == 0 {
        return None;
    }
    let nvml = nvml()?;
    let bus_id = format!("{:08x}:{:02x}:{:02x}.0", (id >> 32) & 0x7fff_ffff, (id >> 8) & 0xff, id & 0xff);
    let bus_id = std::ffi::CString::new(bus_id).ok()?;
    let mut d: Device = std::ptr::null_mut();
    // nvmlMemory_t: total, free, used
    let mut m = [0u64; 3];
    // SAFETY: a NUL-terminated bus id and writable outputs of the documented layouts.
    unsafe { ((nvml.by_bus_id)(bus_id.as_ptr(), &mut d) == 0 && (nvml.memory)(d, &mut m) == 0).then_some((m[1], m[0])) }
}

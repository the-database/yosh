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

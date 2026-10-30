//! [`Upscaler`] backed by libaji's still-image ABI (`aji_image.h`), loaded at
//! runtime from the add-on directory — the engine never links an inference
//! runtime, so builds without the add-on carry none of it.
//!
//! The add-on directory holds `aji.dll` (or `libaji.so`) with its backends and
//! their runtime libraries, a `models/` folder of ONNX files, and a `cache/`
//! for TensorRT engines. Models load lazily, the first time a page needs one,
//! and stay resident. A TensorRT model whose engine isn't built yet comes back
//! [`SrOutcome::NotReady`]; a watcher thread polls it and bumps the epoch (and
//! wakes the shell) once it is ready, so the reader retries those pages.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libloading::Library;

use super::rules::{select, SrRule};
use super::{SrOutcome, Upscaler};
use crate::pool::Waker;

const API_VERSION: u32 = 1;
/// Build missing TensorRT engines in the background (a reader must never stall
/// on a first-use build); the watcher below picks up the result.
const LOAD_ASYNC_BUILD: c_int = 1;
const OK: c_int = 0;
const CANCELLED: c_int = -4;
const STATE_READY: c_int = 0;
const STATE_BUILDING: c_int = 1;

/// Inference backend, as `enum aji_img_backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    TensorRt = 1,
    DirectMl = 2,
}

/// A device the backend can run on (`aji_img_device`).
#[derive(Clone, Debug)]
pub struct Device {
    pub id: u64,
    pub name: String,
    pub vram_bytes: u64,
}

#[repr(C)]
struct RawDevice {
    id: u64,
    name: [c_char; 128],
    vram_bytes: u64,
    sm_major: c_int,
    sm_minor: c_int,
}

type LogFn = extern "C" fn(*mut c_void, c_int, *const c_char);
type ProgressFn = extern "C" fn(*mut c_void, c_int, c_int) -> c_int;

#[repr(C)]
struct CreateParams {
    api_version: u32,
    backend: c_int,
    device_id: u64,
    cache_dir: *const c_char,
    trtexec: *const c_char,
    trtexec_env: *const c_char,
    log: Option<LogFn>,
    log_opaque: *mut c_void,
}

#[repr(C)]
struct ModelParams {
    onnx_path: *const c_char,
    tile_w: c_int,
    tile_h: c_int,
    overlap: c_int,
    flags: c_int,
}

#[repr(C)]
#[derive(Default)]
struct ModelInfo {
    scale: c_int,
    in_channels: c_int,
    out_channels: c_int,
    tile_w: c_int,
    tile_h: c_int,
    overlap: c_int,
}

#[repr(C)]
struct Buffer {
    width: c_int,
    height: c_int,
    channels: c_int,
    stride: isize,
    data: *mut u8,
}

/// The resolved entry points. Plain `fn` pointers, so `Copy` and shareable with
/// the build watcher; valid for as long as the owning [`Library`] is loaded.
#[derive(Clone, Copy)]
struct Api {
    enum_devices: unsafe extern "C" fn(c_int, *mut RawDevice, c_int) -> c_int,
    create: unsafe extern "C" fn(*const CreateParams, *mut c_char, usize) -> *mut c_void,
    load: unsafe extern "C" fn(*mut c_void, *const ModelParams) -> *mut c_void,
    state: unsafe extern "C" fn(*mut c_void) -> c_int,
    info: unsafe extern "C" fn(*mut c_void, *mut ModelInfo) -> c_int,
    upscale:
        unsafe extern "C" fn(*mut c_void, *const Buffer, *const Buffer, Option<ProgressFn>, *mut c_void) -> c_int,
    release: unsafe extern "C" fn(*mut *mut c_void),
    last_error: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    destroy: unsafe extern "C" fn(*mut *mut c_void),
}

/// An aji handle (context or model). The library serializes calls on a model and
/// allows a context from any thread (aji_image.h "Threading"), so moving handles
/// across threads is sound.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Handle(*mut c_void);
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

fn lib_path(dir: &Path) -> PathBuf {
    if cfg!(windows) { dir.join("aji.dll") } else { dir.join("libaji.so") }
}

fn load_api(dir: &Path) -> Result<(Library, Api), String> {
    let path = lib_path(dir);
    // SAFETY: loading libaji runs only its own (trivial) initializers; the
    // backends it forwards to are loaded later, on demand.
    let lib = unsafe { Library::new(&path) }.map_err(|e| format!("{}: {e}", path.display()))?;
    macro_rules! sym {
        ($name:literal) => {
            // SAFETY: the symbol types mirror aji_image.h (API version 1).
            *unsafe { lib.get($name) }.map_err(|e| {
                format!("{}: {} ({e})", path.display(), String::from_utf8_lossy(&$name[..$name.len() - 1]))
            })?
        };
    }
    let api = Api {
        enum_devices: sym!(b"aji_img_enum_devices\0"),
        create: sym!(b"aji_img_create\0"),
        load: sym!(b"aji_img_load\0"),
        state: sym!(b"aji_img_model_state\0"),
        info: sym!(b"aji_img_model_info_get\0"),
        upscale: sym!(b"aji_img_upscale\0"),
        release: sym!(b"aji_img_release\0"),
        last_error: sym!(b"aji_img_last_error\0"),
        destroy: sym!(b"aji_img_destroy\0"),
    };
    Ok((lib, api))
}

fn cstring(p: &Path) -> Result<CString, String> {
    let s = p.to_str().ok_or_else(|| format!("non-UTF-8 path: {}", p.display()))?;
    CString::new(s).map_err(|_| format!("path contains NUL: {s}"))
}

extern "C" fn log_cb(_: *mut c_void, level: c_int, msg: *const c_char) {
    // TensorRT severities: 0 internal error, 1 error, 2 warning.
    if level <= 2 && !msg.is_null() {
        // SAFETY: aji passes a NUL-terminated message valid for this call.
        eprintln!("yosh: [upscale] {}", unsafe { CStr::from_ptr(msg) }.to_string_lossy());
    }
}

extern "C" fn progress_cb(opaque: *mut c_void, _done: c_int, _total: c_int) -> c_int {
    // SAFETY: `opaque` is the `&&dyn Fn` set up in `upscale`, alive for the call.
    let cancel = unsafe { &*(opaque as *const &dyn Fn() -> bool) };
    cancel() as c_int
}

/// Devices `backend` can run on, from the add-on in `dir`.
pub fn enum_devices(dir: &Path, backend: Backend) -> Result<Vec<Device>, String> {
    let (_lib, api) = load_api(dir)?;
    let mut raw: Vec<RawDevice> = (0..16)
        .map(|_| RawDevice { id: 0, name: [0; 128], vram_bytes: 0, sm_major: 0, sm_minor: 0 })
        .collect();
    // SAFETY: `raw` has room for the 16 entries we pass as the maximum.
    let n = unsafe { (api.enum_devices)(backend as c_int, raw.as_mut_ptr(), raw.len() as c_int) };
    if n < 0 {
        return Err("the upscaling backend could not be loaded".into());
    }
    Ok(raw
        .iter()
        .take(n.min(16) as usize)
        .map(|d| Device {
            id: d.id,
            // SAFETY: aji NUL-terminates the name within its 128 bytes.
            name: unsafe { CStr::from_ptr(d.name.as_ptr()) }.to_string_lossy().into_owned(),
            vram_bytes: d.vram_bytes,
        })
        .collect())
}

enum Slot {
    Ready { model: Handle, scale: u32 },
    Building(Handle),
    Failed,
}

/// Shared with the build watcher thread.
struct Watch {
    api: Api,
    building: Mutex<Vec<Handle>>,
    running: AtomicBool,
    epoch: AtomicU64,
    waker: Option<Waker>,
}

pub struct AjiUpscaler {
    // Field order is drop order: the context must be destroyed (see `Drop`)
    // before the library unloads.
    ctx: Handle,
    api: Api,
    models_dir: PathBuf,
    rules: Vec<SrRule>,
    slots: Mutex<HashMap<String, Slot>>,
    watch: Arc<Watch>,
    _lib: Library,
}

impl AjiUpscaler {
    /// Load the add-on in `dir` and open `backend` on `device_id` (0 = first).
    /// `waker` is called when a background engine build finishes.
    pub fn new(
        dir: &Path,
        backend: Backend,
        device_id: u64,
        rules: Vec<SrRule>,
        waker: Option<Waker>,
    ) -> Result<Self, String> {
        let (lib, api) = load_api(dir)?;
        let cache = dir.join("cache");
        let _ = std::fs::create_dir_all(&cache);
        let cache_c = cstring(&cache)?;
        let params = CreateParams {
            api_version: API_VERSION,
            backend: backend as c_int,
            device_id,
            cache_dir: cache_c.as_ptr(),
            trtexec: std::ptr::null(),
            trtexec_env: std::ptr::null(),
            log: Some(log_cb),
            log_opaque: std::ptr::null_mut(),
        };
        let mut err = [0 as c_char; 512];
        // SAFETY: `params` and its strings outlive the call; `err` is writable.
        let ctx = unsafe { (api.create)(&params, err.as_mut_ptr(), err.len()) };
        if ctx.is_null() {
            // SAFETY: aji NUL-terminates `err` (zero-initialized otherwise).
            let e = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            return Err(if e.is_empty() { "upscaler initialization failed".into() } else { e });
        }
        Ok(Self {
            ctx: Handle(ctx),
            api,
            models_dir: dir.join("models"),
            rules,
            slots: Mutex::new(HashMap::new()),
            watch: Arc::new(Watch {
                api,
                building: Mutex::new(Vec::new()),
                running: AtomicBool::new(false),
                epoch: AtomicU64::new(0),
                waker,
            }),
            _lib: lib,
        })
    }

    fn last_error(&self) -> String {
        // SAFETY: the context is live; aji returns a NUL-terminated string.
        unsafe { CStr::from_ptr((self.api.last_error)(self.ctx.0)) }.to_string_lossy().into_owned()
    }

    /// The model's handle and scale once ready; loads it on first use.
    fn model(&self, name: &str) -> Result<Option<(Handle, u32)>, String> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Slot::Building(h)) = slots.get(name) {
            let h = *h;
            // SAFETY: a live model handle.
            match unsafe { (self.api.state)(h.0) } {
                STATE_BUILDING => return Ok(None),
                STATE_READY => {
                    let scale = self.scale_of(h)?;
                    slots.insert(name.to_string(), Slot::Ready { model: h, scale });
                }
                _ => {
                    slots.insert(name.to_string(), Slot::Failed);
                    return Err(self.last_error());
                }
            }
        }
        match slots.get(name) {
            Some(Slot::Ready { model, scale }) => return Ok(Some((*model, *scale))),
            Some(Slot::Failed) => return Err(format!("model {name} failed to load")),
            _ => {}
        }
        let path = self.models_dir.join(format!("{name}.onnx"));
        if !path.is_file() {
            slots.insert(name.to_string(), Slot::Failed);
            return Err(format!("model not installed: {}", path.display()));
        }
        let path_c = cstring(&path)?;
        // Tile/overlap 0 = the library defaults (512 px, 32 px).
        let params = ModelParams {
            onnx_path: path_c.as_ptr(),
            tile_w: 0,
            tile_h: 0,
            overlap: 0,
            flags: LOAD_ASYNC_BUILD,
        };
        // SAFETY: `params` outlives the call; the context is live.
        let h = Handle(unsafe { (self.api.load)(self.ctx.0, &params) });
        if h.0.is_null() {
            slots.insert(name.to_string(), Slot::Failed);
            return Err(self.last_error());
        }
        // SAFETY: a live model handle.
        if unsafe { (self.api.state)(h.0) } == STATE_BUILDING {
            slots.insert(name.to_string(), Slot::Building(h));
            self.watch_build(h);
            return Ok(None);
        }
        let scale = self.scale_of(h)?;
        slots.insert(name.to_string(), Slot::Ready { model: h, scale });
        Ok(Some((h, scale)))
    }

    fn scale_of(&self, h: Handle) -> Result<u32, String> {
        let mut info = ModelInfo::default();
        // SAFETY: a live, READY model handle.
        if unsafe { (self.api.info)(h.0, &mut info) } != OK || info.scale < 1 {
            return Err(self.last_error());
        }
        Ok(info.scale as u32)
    }

    /// Poll a building model in the background; when it (or any other) settles,
    /// bump the epoch and wake the shell so skipped pages are retried.
    fn watch_build(&self, h: Handle) {
        let w = &self.watch;
        w.building.lock().unwrap_or_else(|e| e.into_inner()).push(h);
        if w.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let w = w.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            let mut b = w.building.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: handles stay live until the upscaler drops, which first
            // empties this list and waits for this thread to stop (see `Drop`).
            let before = b.len();
            b.retain(|h| unsafe { (w.api.state)(h.0) } == STATE_BUILDING);
            if b.len() != before {
                w.epoch.fetch_add(1, Ordering::AcqRel);
                if let Some(wk) = &w.waker {
                    wk();
                }
            }
            if b.is_empty() {
                w.running.store(false, Ordering::Release);
                return;
            }
        });
    }
}

impl Upscaler for AjiUpscaler {
    fn upscale(
        &self,
        w: u32,
        h: u32,
        channels: u32,
        pixels: &[u8],
        target_scale: f32,
        cancel: &dyn Fn() -> bool,
    ) -> SrOutcome {
        let Some(name) = select(&self.rules, h, channels == 1, target_scale) else {
            return SrOutcome::Skip;
        };
        let (model, scale) = match self.model(name) {
            Ok(Some(m)) => m,
            Ok(None) => return SrOutcome::NotReady,
            Err(e) => return SrOutcome::Failed(e),
        };
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        let inb = Buffer {
            width: w as c_int,
            height: h as c_int,
            channels: channels as c_int,
            stride: (w * channels) as isize,
            data: pixels.as_ptr() as *mut u8, // read-only on the input side
        };
        let outb = Buffer {
            width: ow as c_int,
            height: oh as c_int,
            channels: channels as c_int,
            stride: (ow * channels) as isize,
            data: out.as_mut_ptr(),
        };
        let cancel_ref: &dyn Fn() -> bool = cancel;
        let opaque = &cancel_ref as *const &dyn Fn() -> bool as *mut c_void;
        // SAFETY: both buffers describe live allocations of the stated sizes, and
        // `opaque` points at `cancel_ref`, which outlives this blocking call.
        let rc = unsafe { (self.api.upscale)(model.0, &inb, &outb, Some(progress_cb), opaque) };
        match rc {
            OK => SrOutcome::Done { scale, pixels: out, model: name.into() },
            CANCELLED => SrOutcome::Cancelled,
            _ => SrOutcome::Failed(self.last_error()),
        }
    }

    fn epoch(&self) -> u64 {
        self.watch.epoch.load(Ordering::Acquire)
    }

    fn model_for(&self, src_h: u32, gray: bool, target_scale: f32) -> Option<Arc<str>> {
        select(&self.rules, src_h, gray, target_scale).map(Arc::from)
    }
}

impl Drop for AjiUpscaler {
    fn drop(&mut self) {
        // Stop the watcher from touching handles we're about to free.
        self.watch.building.lock().unwrap_or_else(|e| e.into_inner()).clear();
        while self.watch.running.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(50));
        }
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        for (_, slot) in slots.drain() {
            if let Slot::Ready { model: Handle(mut p), .. } | Slot::Building(Handle(mut p)) = slot {
                // SAFETY: each handle is released exactly once, before the context.
                unsafe { (self.api.release)(&mut p) };
            }
        }
        let mut ctx = self.ctx.0;
        // SAFETY: last use of the context; the library is still loaded (`_lib`
        // drops after this).
        unsafe { (self.api.destroy)(&mut ctx) };
    }
}

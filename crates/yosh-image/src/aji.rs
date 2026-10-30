//! Safe bindings to libaji's still-image ABI (`aji_image.h`, API version 1, from
//! the-database/animejanai-inference ≥ v0.18.0), loaded **at runtime** with
//! `libloading` — nothing links an inference runtime, so a build carries none of
//! it until a directory with `aji.dll` / `libaji.so` (plus its backends and their
//! runtimes) is pointed at.
//!
//! Ownership follows the C API: an [`Aji`] (the loaded library) opens
//! [`Context`]s (a backend on a device), which load [`Model`]s (an ONNX file at a
//! fixed tile size). Each handle keeps its parents alive, so they can be dropped
//! in any order; the last drop releases the C object.
//!
//! ```no_run
//! # use std::path::Path;
//! use yosh_image::aji::{Aji, Backend, ContextParams, LoadParams, Pixels, PixelsMut, Upscaled};
//! let aji = Aji::load(Path::new("addon"))?;
//! let dev = aji.devices(Backend::DirectMl)?.into_iter().next().ok_or("no GPU")?;
//! let ctx = aji.context(&ContextParams { backend: Backend::DirectMl, device_id: dev.id, ..Default::default() })?;
//! let model = ctx.load_model(&LoadParams::new(Path::new("model.onnx")))?; // blocks through a TRT build
//! let s = model.info()?.scale;
//! let (w, h) = (100u32, 80u32);
//! let input = vec![0u8; (w * h) as usize];
//! let mut out = vec![0u8; (w * s * h * s) as usize];
//! let done = model.upscale(
//!     Pixels::packed(w, h, 1, &input),
//!     PixelsMut::packed(w * s, h * s, 1, &mut out),
//!     None,
//! )?;
//! assert!(matches!(done, Upscaled::Done));
//! # Ok::<(), String>(())
//! ```

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

const API_VERSION: u32 = 1;
const LOAD_ASYNC_BUILD: c_int = 1;
const OK: c_int = 0;
const CANCELLED: c_int = -4;
const STATE_READY: c_int = 0;
const STATE_BUILDING: c_int = 1;

/// Inference backend (`enum aji_img_backend`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    TensorRt = 1,
    /// Windows only.
    DirectMl = 2,
}

/// A device a backend can run on (`aji_img_device`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// Stable across runs: PCI location (TensorRT) or DXGI adapter LUID
    /// (DirectML). Pass to [`ContextParams::device_id`].
    pub id: u64,
    pub name: String,
    pub vram_bytes: u64,
    /// CUDA compute capability (TensorRT); `None` for DirectML.
    pub compute_capability: Option<(i32, i32)>,
}

/// A model's state ([`Model::state`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelState {
    Ready,
    /// A TensorRT engine is building in the background ([`LoadParams::async_build`]).
    Building,
    /// See [`Context::last_error`].
    Failed,
}

/// What a ready model does ([`Model::info`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    /// Output = `scale` × input, both axes.
    pub scale: u32,
    /// The model's own channel counts (1 or 3).
    pub in_channels: u32,
    pub out_channels: u32,
    /// The effective tile and overlap, in input pixels.
    pub tile_w: u32,
    pub tile_h: u32,
    pub overlap: u32,
}

/// How [`Model::upscale`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Upscaled {
    Done,
    /// The progress callback asked to stop.
    Cancelled,
}

// ---------------- the C ABI ----------------

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
struct RawCreateParams {
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
struct RawModelParams {
    onnx_path: *const c_char,
    tile_w: c_int,
    tile_h: c_int,
    overlap: c_int,
    flags: c_int,
}

#[repr(C)]
#[derive(Default)]
struct RawModelInfo {
    scale: c_int,
    in_channels: c_int,
    out_channels: c_int,
    tile_w: c_int,
    tile_h: c_int,
    overlap: c_int,
}

#[repr(C)]
struct RawBuffer {
    width: c_int,
    height: c_int,
    channels: c_int,
    stride: isize,
    data: *mut u8,
}

#[derive(Clone, Copy)]
struct Api {
    enum_devices: unsafe extern "C" fn(c_int, *mut RawDevice, c_int) -> c_int,
    create: unsafe extern "C" fn(*const RawCreateParams, *mut c_char, usize) -> *mut c_void,
    load: unsafe extern "C" fn(*mut c_void, *const RawModelParams) -> *mut c_void,
    state: unsafe extern "C" fn(*mut c_void) -> c_int,
    info: unsafe extern "C" fn(*mut c_void, *mut RawModelInfo) -> c_int,
    upscale: unsafe extern "C" fn(*mut c_void, *const RawBuffer, *const RawBuffer, Option<ProgressFn>, *mut c_void)
        -> c_int,
    release: unsafe extern "C" fn(*mut *mut c_void),
    last_error: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    destroy: unsafe extern "C" fn(*mut *mut c_void),
}

/// A context or model pointer. libaji allows a context from any thread and
/// serializes calls on one model (aji_image.h "Threading"), so these may move
/// and be shared across threads.
#[derive(Clone, Copy)]
struct Handle(*mut c_void);
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

fn cstring(p: &Path) -> Result<CString, String> {
    let s = p.to_str().ok_or_else(|| format!("non-UTF-8 path: {}", p.display()))?;
    CString::new(s).map_err(|_| format!("path contains NUL: {s}"))
}

// ---------------- Aji (the library) ----------------

struct LibInner {
    api: Api,
    _lib: Library,
}

/// The loaded libaji. Cheap to clone.
#[derive(Clone)]
pub struct Aji(Arc<LibInner>);

impl Aji {
    /// Load `aji.dll` (Windows) / `libaji.so` from `dir`. Its backends and their
    /// runtime libraries are found next to it, and loaded only when a context for
    /// that backend is opened.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_file(&dir.join(if cfg!(windows) { "aji.dll" } else { "libaji.so" }))
    }

    /// Load libaji from an explicit file path.
    pub fn load_file(path: &Path) -> Result<Self, String> {
        // SAFETY: loading libaji runs only its own trivial initializers.
        let lib = unsafe { Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
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
        Ok(Self(Arc::new(LibInner { api, _lib: lib })))
    }

    /// Devices `backend` can run on. An error means the backend (or its runtime)
    /// couldn't be loaded — e.g. no NVIDIA driver for TensorRT.
    pub fn devices(&self, backend: Backend) -> Result<Vec<Device>, String> {
        const MAX: usize = 16;
        let mut raw: Vec<RawDevice> = (0..MAX)
            .map(|_| RawDevice { id: 0, name: [0; 128], vram_bytes: 0, sm_major: 0, sm_minor: 0 })
            .collect();
        // SAFETY: `raw` has room for the MAX entries passed as the maximum.
        let n = unsafe { (self.0.api.enum_devices)(backend as c_int, raw.as_mut_ptr(), MAX as c_int) };
        if n < 0 {
            return Err(format!("the {backend:?} backend could not be loaded"));
        }
        Ok(raw
            .iter()
            .take((n as usize).min(MAX))
            .map(|d| Device {
                id: d.id,
                // SAFETY: aji NUL-terminates the name within its 128 bytes.
                name: unsafe { CStr::from_ptr(d.name.as_ptr()) }.to_string_lossy().into_owned(),
                vram_bytes: d.vram_bytes,
                compute_capability: (backend == Backend::TensorRt).then_some((d.sm_major, d.sm_minor)),
            })
            .collect())
    }

    /// Open `params.backend` on a device.
    pub fn context(&self, params: &ContextParams) -> Result<Context, String> {
        let cache = params.cache_dir.as_deref().map(cstring).transpose()?;
        let trtexec = params.trtexec.as_deref().map(cstring).transpose()?;
        // The log sink lives in a stable box the context owns; aji gets its address.
        let log: Option<Box<LogSink>> = params.log.clone().map(Box::new);
        let opaque = log.as_deref().map_or(std::ptr::null_mut(), |l| l as *const LogSink as *mut c_void);
        let raw = RawCreateParams {
            api_version: API_VERSION,
            backend: params.backend as c_int,
            device_id: params.device_id,
            cache_dir: cache.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            trtexec: trtexec.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            trtexec_env: std::ptr::null(),
            log: log.is_some().then_some(log_trampoline as LogFn),
            log_opaque: opaque,
        };
        let mut err = [0 as c_char; 512];
        // SAFETY: `raw` and the strings it points into outlive the call; `err` is
        // writable for its stated length; the log box outlives the context.
        let ptr = unsafe { (self.0.api.create)(&raw, err.as_mut_ptr(), err.len()) };
        if ptr.is_null() {
            // SAFETY: aji NUL-terminates `err` (zero-initialized otherwise).
            let e = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            return Err(if e.is_empty() { "upscaler initialization failed".into() } else { e });
        }
        Ok(Context(Arc::new(CtxInner { lib: self.clone(), ptr: Handle(ptr), _log: log })))
    }
}

/// A log callback: `(level, message)`, level as TensorRT's severities (0 internal
/// error, 1 error, 2 warning, 3 info, 4 verbose).
pub type LogSink = Arc<dyn Fn(i32, &str) + Send + Sync>;

extern "C" fn log_trampoline(opaque: *mut c_void, level: c_int, msg: *const c_char) {
    if opaque.is_null() || msg.is_null() {
        return;
    }
    // SAFETY: `opaque` is the context's boxed `LogSink`, alive as long as the
    // context; `msg` is NUL-terminated for the duration of the call.
    let sink = unsafe { &*(opaque as *const LogSink) };
    sink(level, &unsafe { CStr::from_ptr(msg) }.to_string_lossy());
}

/// [`Aji::context`] parameters.
#[derive(Clone, Default)]
pub struct ContextParams {
    pub backend: Backend,
    /// [`Device::id`]; 0 = the first device.
    pub device_id: u64,
    /// TensorRT engine/timing cache; `None` = each model's own directory.
    pub cache_dir: Option<PathBuf>,
    /// The `trtexec` that builds missing TensorRT engines; `None` = next to the
    /// backend library.
    pub trtexec: Option<PathBuf>,
    pub log: Option<LogSink>,
}

// ---------------- Context ----------------

struct CtxInner {
    lib: Aji,
    ptr: Handle,
    _log: Option<Box<LogSink>>,
}

impl Drop for CtxInner {
    fn drop(&mut self) {
        let mut p = self.ptr.0;
        // SAFETY: the last reference — every Model holds a Context clone, so all
        // models are already released.
        unsafe { (self.lib.0.api.destroy)(&mut p) };
    }
}

/// A backend open on a device. Cheap to clone; usable from any thread.
#[derive(Clone)]
pub struct Context(Arc<CtxInner>);

impl Context {
    /// The context's last error message (for diagnostics; may be overwritten by
    /// calls on other threads).
    pub fn last_error(&self) -> String {
        // SAFETY: a live context; aji returns a NUL-terminated string it owns.
        unsafe { CStr::from_ptr((self.0.lib.0.api.last_error)(self.0.ptr.0)) }.to_string_lossy().into_owned()
    }

    /// Load an ONNX model. By default a TensorRT model whose engine isn't cached
    /// is built here (minutes, once per model/tile/GPU/TensorRT version) and comes
    /// back ready; with [`LoadParams::async_build`] it comes back
    /// [`ModelState::Building`] instead.
    pub fn load_model(&self, params: &LoadParams) -> Result<Model, String> {
        let path = cstring(params.onnx)?;
        let (tw, th) = params.tile.unwrap_or((0, 0));
        let raw = RawModelParams {
            onnx_path: path.as_ptr(),
            tile_w: tw as c_int,
            tile_h: th as c_int,
            overlap: params.overlap.unwrap_or(0) as c_int,
            flags: if params.async_build { LOAD_ASYNC_BUILD } else { 0 },
        };
        // SAFETY: a live context; `raw` and `path` outlive the call.
        let ptr = unsafe { (self.0.lib.0.api.load)(self.0.ptr.0, &raw) };
        if ptr.is_null() {
            return Err(self.last_error());
        }
        Ok(Model { ctx: self.clone(), ptr: Handle(ptr) })
    }
}

/// [`Context::load_model`] parameters.
#[derive(Clone, Debug)]
pub struct LoadParams<'a> {
    pub onnx: &'a Path,
    /// Tile size in input pixels; `None` = 512×512. Ignored for fixed-shape
    /// models (their shape is the tile).
    pub tile: Option<(u32, u32)>,
    /// Overlap between tiles in input pixels; `None` = 32.
    pub overlap: Option<u32>,
    /// Build a missing TensorRT engine in the background (poll
    /// [`Model::state`]) instead of blocking in `load_model`.
    pub async_build: bool,
}

impl<'a> LoadParams<'a> {
    pub fn new(onnx: &'a Path) -> Self {
        Self { onnx, tile: None, overlap: None, async_build: false }
    }
}

// ---------------- Model ----------------

/// A loaded model. Calls on one model are serialized by libaji; different models
/// may run concurrently.
pub struct Model {
    ctx: Context,
    ptr: Handle,
}

impl Drop for Model {
    fn drop(&mut self) {
        let mut p = self.ptr.0;
        // SAFETY: released exactly once; the context is still alive (`self.ctx`).
        unsafe { (self.ctx.0.lib.0.api.release)(&mut p) };
    }
}

/// An input image for [`Model::upscale`]: 8-bit, `channels` 1 (gray) or 3 (RGB),
/// rows `stride` bytes apart.
#[derive(Clone, Copy)]
pub struct Pixels<'a> {
    pub w: u32,
    pub h: u32,
    pub channels: u32,
    pub stride: usize,
    pub data: &'a [u8],
}

impl<'a> Pixels<'a> {
    /// Tightly packed rows (`stride = w * channels`).
    pub fn packed(w: u32, h: u32, channels: u32, data: &'a [u8]) -> Self {
        Self { w, h, channels, stride: (w * channels) as usize, data }
    }
}

/// The output buffer for [`Model::upscale`] (`scale` × the input's size, same
/// channel count).
pub struct PixelsMut<'a> {
    pub w: u32,
    pub h: u32,
    pub channels: u32,
    pub stride: usize,
    pub data: &'a mut [u8],
}

impl<'a> PixelsMut<'a> {
    /// Tightly packed rows (`stride = w * channels`).
    pub fn packed(w: u32, h: u32, channels: u32, data: &'a mut [u8]) -> Self {
        Self { w, h, channels, stride: (w * channels) as usize, data }
    }
}

fn fits(w: u32, h: u32, channels: u32, stride: usize, len: usize) -> bool {
    let row = (w as usize) * (channels as usize);
    w > 0 && h > 0 && stride >= row && (h as usize - 1).checked_mul(stride).and_then(|v| v.checked_add(row)).is_some_and(|need| need <= len)
}

extern "C" fn progress_trampoline(opaque: *mut c_void, done: c_int, total: c_int) -> c_int {
    // SAFETY: `opaque` is the `&mut &mut dyn FnMut` set up in `upscale`, alive
    // and exclusively ours for the duration of that (blocking) call.
    let f = unsafe { &mut *(opaque as *mut &mut dyn FnMut(u32, u32) -> bool) };
    f(done.max(0) as u32, total.max(0) as u32) as c_int
}

impl Model {
    pub fn state(&self) -> ModelState {
        // SAFETY: a live model handle (polling also advances background builds).
        match unsafe { (self.ctx.0.lib.0.api.state)(self.ptr.0) } {
            STATE_READY => ModelState::Ready,
            STATE_BUILDING => ModelState::Building,
            _ => ModelState::Failed,
        }
    }

    /// Scale, channels and tile — once the model is [`ModelState::Ready`].
    pub fn info(&self) -> Result<ModelInfo, String> {
        let mut i = RawModelInfo::default();
        // SAFETY: a live model handle; `i` is writable.
        if unsafe { (self.ctx.0.lib.0.api.info)(self.ptr.0, &mut i) } != OK || i.scale < 1 {
            let e = self.ctx.last_error();
            return Err(if e.is_empty() { format!("model is not ready ({:?})", self.state()) } else { e });
        }
        Ok(ModelInfo {
            scale: i.scale as u32,
            in_channels: i.in_channels as u32,
            out_channels: i.out_channels as u32,
            tile_w: i.tile_w as u32,
            tile_h: i.tile_h as u32,
            overlap: i.overlap as u32,
        })
    }

    /// Upscale `input` into `output` (which must be `scale` × its size with the
    /// same channel count). A gray input to a 3-channel model is replicated in
    /// and channel 0 kept. `progress(done_tiles, total_tiles)` runs between tiles
    /// and returns `true` to cancel. Blocks until done.
    pub fn upscale(
        &self,
        input: Pixels<'_>,
        output: PixelsMut<'_>,
        progress: Option<&mut dyn FnMut(u32, u32) -> bool>,
    ) -> Result<Upscaled, String> {
        if !fits(input.w, input.h, input.channels, input.stride, input.data.len()) {
            return Err("input buffer is smaller than its stated size".into());
        }
        if !fits(output.w, output.h, output.channels, output.stride, output.data.len()) {
            return Err("output buffer is smaller than its stated size".into());
        }
        let inb = RawBuffer {
            width: input.w as c_int,
            height: input.h as c_int,
            channels: input.channels as c_int,
            stride: input.stride as isize,
            data: input.data.as_ptr() as *mut u8, // only read
        };
        let outb = RawBuffer {
            width: output.w as c_int,
            height: output.h as c_int,
            channels: output.channels as c_int,
            stride: output.stride as isize,
            data: output.data.as_mut_ptr(),
        };
        let mut f = progress;
        let (cb, opaque): (Option<ProgressFn>, *mut c_void) = match f.as_mut() {
            Some(f) => (Some(progress_trampoline), f as *mut &mut dyn FnMut(u32, u32) -> bool as *mut c_void),
            None => (None, std::ptr::null_mut()),
        };
        // SAFETY: both buffers were bounds-checked against their stated sizes and
        // outlive this blocking call, as does the callback `opaque` points at.
        let rc = unsafe { (self.ctx.0.lib.0.api.upscale)(self.ptr.0, &inb, &outb, cb, opaque) };
        match rc {
            OK => Ok(Upscaled::Done),
            CANCELLED => Ok(Upscaled::Cancelled),
            _ => Err(self.ctx.last_error()),
        }
    }

    /// The context this model belongs to.
    pub fn context(&self) -> &Context {
        &self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::fits;

    #[test]
    fn buffer_bounds_are_checked_before_crossing_ffi() {
        assert!(fits(4, 3, 1, 4, 12));
        assert!(!fits(4, 3, 1, 4, 11));
        assert!(fits(4, 3, 3, 16, 2 * 16 + 12), "last row needs only w*ch");
        assert!(!fits(4, 3, 3, 10, 100), "stride shorter than a row");
        assert!(!fits(0, 3, 1, 0, 100) && !fits(4, 0, 1, 4, 100));
        assert!(!fits(1, u32::MAX, 1, usize::MAX, usize::MAX), "overflow is a refusal, not a wrap");
    }
}

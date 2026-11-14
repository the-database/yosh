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
//!
//! On TensorRT-RTX (chosen, or Auto's pick), a model that fails to load or build falls back
//! to TensorRT on the same GPU: the TensorRT context opens the first time that
//! happens, and the other models stay on TensorRT-RTX. Not when it failed for lack of
//! GPU memory ([`yosh_image::aji::is_out_of_memory`]): a second CUDA context and another
//! copy of the model would need more of it, so that model fails with libaji's message.
//!
//! [`Upscaler::release_gpu`] (the shell calls it while upscaling is idle) closes the
//! models and the contexts and has libaji (0.24.0+) free the device, which on the
//! CUDA backends drops the CUDA context TensorRT's runtime would keep until exit;
//! the next page reopens the context and reloads its model from the engine cache.
//! Dropping the upscaler does the same. Opening a context and freeing a device are
//! serialized process-wide ([`GPU`]): a device reset must never overlap another
//! upscaler opening its context on the same GPU.
//!
//! **Tiles.** libaji's default 512 px square tiles suit some page sizes badly: 1200x1800 runs
//! 1.82x its pixels on them. Tiles are sized to the GPU's free memory instead (yosh-image's
//! [`yosh_image::tiles::Budget`], from what libaji 0.28.0+ reports about the loaded models,
//! recorded in the add-on's `cache/` beside the engines, as MangaJaNaiCLI does):
//!
//! - On TensorRT a model loads as one even-tiles engine (`AJI_IMG_LOAD_EVEN_TILES`), tiles of at
//!   most 1024 (less where memory is short, [`yosh_image::tiles::even_tile`]): each page is split
//!   into the fewest even tiles, as fast as a fitted engine and good for any page size.
//! - On TensorRT-RTX and DirectML, where an even-tiles engine is slow or missing, a model loads on
//!   the square, and for a page size whose best tile shape ([`yosh_image::tiles::fit`]) saves
//!   enough and fits in memory, an engine fitted to it (1216x992 there: about half the time) loads
//!   in the background from the second page of that size (a volume's pages are one size), or at
//!   once if it was built before (by yosh or MangaJaNaiCLI, [`yosh_image::tiles::fitted_known`]).
//!   The pages of that size switch to it when it's ready, and back to the square if it isn't
//!   faster on its first [`CHECK_PAGES`] pages or fails. At most [`MAX_FITTED`] stay loaded per
//!   model.
//!
//! A rule's model may be installed under another precision suffix (`_fp16`,
//! `_bf16`, `_fp32`) than the rule names, or with none (older add-ons), or with one
//! where the rule names none (an older `rules.json`): [`lookup_order`] finds
//! whichever the add-on has, so add-ons built from older and newer MangaJaNai GUI
//! model files both work with either names.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use yosh_image::aji::{Aji, Context, ContextParams, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};
pub use yosh_image::aji::{Backend, Device};
use yosh_image::tiles::{self, Budget, MemoryUse, Shape};

use super::rules::{auto_levels, select, SrRule};
use super::{SrOutcome, Upscaler};
use crate::pool::Waker;

/// Serializes opening a context, listing devices and freeing a device across the
/// process: `aji_img_release_device` resets the device for everything in it.
static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Load libaji from the add-on in `dir`. On Linux, first load the CUDA and
/// TensorRT libraries a portable add-on keeps beside it (`mangajanai setup`
/// without `--store`), by full path and globally: libaji_trt.so asks for them by
/// soname with no search path of its own, so they would otherwise only be found
/// in a system install, and TensorRT couldn't load. (A store install's packs are
/// loaded by libaji itself, from `aji-runtime.json`; a system TensorRT works as
/// before.) MangaJaNaiCLI's `load_aji` does the same.
fn load_aji(dir: &Path) -> Result<Aji, String> {
    #[cfg(unix)]
    for lib in ["libcudart.so.13", "libnvinfer.so.11", "libnvinfer_plugin.so.11", "libnvonnxparser.so.11"] {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        let path = dir.join(lib);
        if path.is_file() {
            // SAFETY: NVIDIA's libraries; their initializers only set up their own
            // state. Kept loaded for the process's life (libaji uses them until exit).
            match unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                Ok(l) => std::mem::forget(l),
                Err(e) => eprintln!("yosh: [upscale] {}: {e}", path.display()),
            }
        }
    }
    Aji::load(dir)
}

/// The `trtexec` that builds TensorRT engines, and an environment prefix for it
/// (MangaJaNaiCLI's `trtexec`): `(None, None)` on Windows (libaji runs the one
/// beside it, which finds its DLLs there). On Linux a `trtexec` beside libaji
/// (a portable add-on) gets the add-on folder on `LD_LIBRARY_PATH`, or it can't
/// find TensorRT's libraries; without one, a system TensorRT's
/// (`/usr/src/tensorrt/bin`, which isn't on `PATH`, else `PATH`'s). A store
/// install's comes from its TensorRT pack (libaji, `aji-runtime.json`).
fn trtexec(dir: &Path) -> (Option<PathBuf>, Option<String>) {
    if cfg!(windows) {
        return (None, None);
    }
    if dir.join("trtexec").is_file() {
        let dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        // single-quoted for the shell; a ' inside becomes '\''
        let quoted = dir.to_string_lossy().replace('\'', r"'\''");
        return (None, Some(format!("LD_LIBRARY_PATH='{quoted}'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}")));
    }
    let on_path = std::env::var_os("PATH")
        .and_then(|p| std::env::split_paths(&p).map(|d| d.join("trtexec")).find(|t| t.is_file()));
    let system = Some(PathBuf::from("/usr/src/tensorrt/bin/trtexec")).filter(|t| t.is_file()).or(on_path);
    (system, None)
}

/// Devices `backend` can run on, from the add-on in `dir`.
pub fn enum_devices(dir: &Path, backend: Backend) -> Result<Vec<Device>, String> {
    let aji = load_aji(dir)?;
    let _gpu = gpu_lock();
    aji.devices(backend)
}

/// libaji's default tile, which every model loads at first.
const SQUARE: u32 = 512;

/// Fitted engines kept loaded per model; past it, the least recently used is unloaded.
const MAX_FITTED: usize = 2;

/// A ready fitted engine is timed on this many pages, and kept if it's at least [`CHECK_GAIN`]
/// faster per page than the square was.
const CHECK_PAGES: u32 = 3;
const CHECK_GAIN: f64 = 0.05;

/// A fitted tile shape and the share of the square's work it saves ([`tiles::fit`]).
type Fitted = (Shape, f64);

/// A model's engine fitted to one page size (see the module docs). `square`: seconds per page on
/// the square, for the fitted engine's check.
enum Fit {
    Renting { square: Option<f64> },
    /// `bytes`: the memory it's predicted to take once loaded.
    Building { model: Arc<Model>, bytes: u64, square: Option<f64> },
    Ready { model: Arc<Model>, last_use: u64, check: Option<Check> },
    Failed,
}

/// A ready fitted engine's first pages (`runs`, which took `took` seconds), against the square's
/// `square` seconds per page.
struct Check {
    square: f64,
    runs: u32,
    took: f64,
}

enum Slot {
    /// `fell_back`: on the TensorRT fallback context (no fitted engines then).
    Ready { model: Arc<Model>, scale: u32, fell_back: bool },
    /// `fell_back`: already on the TensorRT fallback context. `path`: the file it
    /// loaded from, for that fallback.
    Building { model: Arc<Model>, fell_back: bool, path: PathBuf },
    Failed,
}

/// The TensorRT context a TensorRT-RTX upscaler falls back to, opened on first need.
struct Fallback {
    ctx: Mutex<Option<Result<Context, String>>>,
}

/// Shared with the build watcher thread.
struct Watch {
    building: Mutex<Vec<Arc<Model>>>,
    running: AtomicBool,
    epoch: AtomicU64,
    waker: Option<Waker>,
}

pub struct AjiUpscaler {
    aji: Aji,
    dir: PathBuf,
    device_id: u64,
    /// `None` once [`Upscaler::release_gpu`] closed it; the next model load reopens it.
    ctx: Mutex<Option<Context>>,
    backend: Backend,
    /// The device's CUDA compute capability (TensorRT, TensorRT-RTX), for [`lookup_order`].
    compute_capability: Option<(i32, i32)>,
    /// Present on TensorRT-RTX only.
    fallback: Option<Fallback>,
    /// The first model that fell back to TensorRT, and why ([`Upscaler::fallback`]).
    fell_back: Mutex<Option<String>>,
    /// Where models are looked up, in order: the add-on's `models/` first (the user's own
    /// models go there), then any folders [`AjiUpscaler::with_model_dirs`] added.
    models_dirs: Vec<PathBuf>,
    /// Swapped whole by [`Upscaler::set_rules`] (a profile switch).
    rules: RwLock<Arc<Vec<SrRule>>>,
    slots: Mutex<HashMap<String, Slot>>,
    /// Fitted engines, by model name and tile shape.
    fitted: Mutex<HashMap<(String, Shape), Fit>>,
    /// The fitted shape (and the share of the square's work it saves) by model name and page
    /// size, decided on the first page of that size.
    shapes: Mutex<HashMap<(String, Shape), Option<Fitted>>>,
    /// Each model's memory use, by name, measured from its engines.
    memory: Mutex<HashMap<String, MemoryUse>>,
    /// Pages upscaled, for the fitted engines' least-recently-used order.
    clock: AtomicU64,
    /// The device's name, and the largest fitted tile area by its size for a libaji that can't
    /// report memory.
    device_name: String,
    area_budget: u64,
    watch: Arc<Watch>,
    /// Upscales running now: [`Upscaler::release_gpu`] waits for none.
    active: AtomicUsize,
    /// When the last one ended (or the upscaler opened): [`Upscaler::idle_since`].
    last_used: Mutex<Instant>,
}

/// Counts an upscale as running until dropped, then stamps its end.
struct Running<'a>(&'a AjiUpscaler);

impl Drop for Running<'_> {
    fn drop(&mut self) {
        *self.0.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl AjiUpscaler {
    /// Load the add-on in `dir` and open `backend` on `device` (one of
    /// [`enum_devices`]). `waker` is called when a background engine build finishes.
    pub fn new(
        dir: &Path,
        backend: Backend,
        device: &Device,
        rules: Vec<SrRule>,
        waker: Option<Waker>,
    ) -> Result<Self, String> {
        let aji = load_aji(dir)?;
        let ctx = open_context(&aji, dir, backend, device.id)?;
        let fallback = (backend == Backend::TensorRtRtx).then(|| Fallback { ctx: Mutex::new(None) });
        Ok(Self {
            aji,
            dir: dir.to_path_buf(),
            device_id: device.id,
            ctx: Mutex::new(Some(ctx)),
            backend,
            compute_capability: device.compute_capability,
            fallback,
            fell_back: Mutex::new(None),
            models_dirs: vec![dir.join("models")],
            rules: RwLock::new(Arc::new(rules)),
            slots: Mutex::new(HashMap::new()),
            fitted: Mutex::new(HashMap::new()),
            shapes: Mutex::new(HashMap::new()),
            memory: Mutex::new(HashMap::new()),
            clock: AtomicU64::new(0),
            device_name: device.name.clone(),
            area_budget: tiles::area_budget(device.vram_bytes),
            watch: Arc::new(Watch {
                building: Mutex::new(Vec::new()),
                running: AtomicBool::new(false),
                epoch: AtomicU64::new(0),
                waker,
            }),
            active: AtomicUsize::new(0),
            last_used: Mutex::new(Instant::now()),
        })
    }

    /// The primary context, reopened if [`Upscaler::release_gpu`] closed it.
    fn primary(&self) -> Result<Context, String> {
        let mut c = self.ctx.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ctx) = &*c {
            return Ok(ctx.clone());
        }
        let ctx = open_context(&self.aji, &self.dir, self.backend, self.device_id)?;
        eprintln!("yosh: [upscale] {} reopened after a GPU release", self.backend.label());
        *c = Some(ctx.clone());
        Ok(ctx)
    }

    /// Close the models and the contexts and free the device (see
    /// [`Upscaler::release_gpu`]); `slots` is the locked slot map. Whether any
    /// context was open.
    fn close_all(&self, slots: &mut HashMap<String, Slot>) -> bool {
        slots.clear();
        self.fitted.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.shapes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.watch.building.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let primary = self.ctx.lock().unwrap_or_else(|e| e.into_inner()).take();
        let fallback = self
            .fallback
            .as_ref()
            .and_then(|f| f.ctx.lock().unwrap_or_else(|e| e.into_inner()).take())
            .and_then(Result::ok);
        if primary.is_none() && fallback.is_none() {
            return false;
        }
        let mut backends = vec![self.backend];
        if fallback.is_some() {
            backends.push(Backend::TensorRt);
        }
        let _gpu = gpu_lock();
        // The models are gone, so these were the contexts' last references.
        drop(primary);
        drop(fallback);
        for b in backends {
            match self.aji.release_device(b, self.device_id) {
                Ok(true) => {}
                // libaji before 0.24.0: the contexts closed, the CUDA context stays.
                Ok(false) => eprintln!("yosh: [upscale] this libaji can't free the device; update the add-on"),
                Err(e) => eprintln!("yosh: [upscale] {}: {e}", b.label()),
            }
        }
        true
    }

    /// Also look up models in `dirs`, after the add-on's `models/` (e.g. the model packs of a
    /// shared pack store, which the add-on's `aji-runtime.json` lists).
    pub fn with_model_dirs(mut self, dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        self.models_dirs.extend(dirs);
        self
    }

    fn rules(&self) -> Arc<Vec<SrRule>> {
        self.rules.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The installed file for a rule's model `name`: the first of [`lookup_order`]
    /// in the models folders (in order), logged when it isn't `name` itself.
    fn resolve(&self, name: &str) -> Option<PathBuf> {
        let order = lookup_order(name, self.backend, self.compute_capability);
        let (stem, path) = order
            .iter()
            .flat_map(|stem| self.models_dirs.iter().map(move |d| (stem, d.join(format!("{stem}.onnx")))))
            .find(|(_, p)| p.is_file())?;
        if stem != name {
            eprintln!("yosh: [upscale] {name}: not installed; using {}", path.display());
        }
        Some(path)
    }

    /// The model, its scale and whether it fell back to TensorRT, once ready; loads it on first
    /// use, for pages of `page` (w, h).
    fn model(&self, name: &str, page: Shape) -> Result<Option<(Arc<Model>, u32, bool)>, String> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Slot::Building { model, fell_back, path }) = slots.get(name) {
            let (m, fell_back, path) = (model.clone(), *fell_back, path.clone());
            match m.state() {
                ModelState::Building => return Ok(None),
                ModelState::Ready => {
                    let scale = m.info()?.scale;
                    self.measure(name, &path, &m);
                    slots.insert(name.to_string(), Slot::Ready { model: m, scale, fell_back });
                }
                ModelState::Failed => {
                    let e = m.context().last_error();
                    if fell_back {
                        slots.insert(name.to_string(), Slot::Failed);
                        return Err(e);
                    }
                    return self.fall_back(&mut slots, name, &path, e);
                }
            }
        }
        match slots.get(name) {
            Some(Slot::Ready { model, scale, fell_back }) => return Ok(Some((model.clone(), *scale, *fell_back))),
            Some(Slot::Failed) => return Err(format!("model {name} failed to load")),
            _ => {}
        }
        let Some(path) = self.resolve(name) else {
            slots.insert(name.to_string(), Slot::Failed);
            let path = self.models_dirs[0].join(format!("{name}.onnx"));
            return Err(format!("model not installed: {}", path.display()));
        };
        let ctx = self.primary()?;
        // TensorRT: one even-tiles engine for every page size, as large as fits (module docs)
        let even = (self.backend == Backend::TensorRt).then(|| {
            let budget = self.budget(&ctx, name, &path, 0);
            let t = tiles::even_tile(|t| budget.fits((t, t), page).0);
            eprintln!("yosh: [upscale] {name}: even tiles of up to {t}");
            t
        });
        match self.load(&mut slots, &ctx, name, &path, false, even) {
            Err(e) => self.fall_back(&mut slots, name, &path, e),
            done => done,
        }
    }

    /// Load `path` on `ctx` into `name`'s slot: the library-default tile (512 px), or with
    /// `even`, even tiles of at most that. Missing engines build in the background — a reader
    /// must never stall on a first-use build; the watcher below picks up the result.
    fn load(
        &self,
        slots: &mut HashMap<String, Slot>,
        ctx: &Context,
        name: &str,
        path: &Path,
        fell_back: bool,
        even: Option<u32>,
    ) -> Result<Option<(Arc<Model>, u32, bool)>, String> {
        let params = LoadParams {
            async_build: true,
            tile: even.map(|t| (t, t)),
            even_tiles: even.is_some(),
            ..LoadParams::new(path)
        };
        let m = match ctx.load_model(&params) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                slots.insert(name.to_string(), Slot::Failed);
                return Err(e);
            }
        };
        if m.state() == ModelState::Building {
            slots.insert(name.to_string(), Slot::Building { model: m.clone(), fell_back, path: path.to_path_buf() });
            self.watch_build(m);
            return Ok(None);
        }
        let scale = m.info()?.scale;
        self.measure(name, path, &m);
        slots.insert(name.to_string(), Slot::Ready { model: m.clone(), scale, fell_back });
        Ok(Some((m, scale, fell_back)))
    }

    /// Note `name`'s memory use from its ready engine `m` (file `path`), here and in the record
    /// in the add-on's `cache/`.
    fn measure(&self, name: &str, path: &Path, m: &Model) {
        let Ok(info) = m.info() else { return };
        let Some(use_) = m.memory().and_then(|mem| MemoryUse::measured(&mem, (info.tile_w, info.tile_h), info.scale)) else {
            return;
        };
        if let Some(file) = path.file_name() {
            tiles::record_memory(&self.dir.join("cache"), &file.to_string_lossy(), self.backend.label(), &self.device_name, &use_);
        }
        self.memory.lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string(), use_);
    }

    /// The GPU's free memory on `ctx` now, to check `name`'s engines against (its memory use as
    /// measured, recorded, or else guessed), less `reserved` for engines still building.
    fn budget(&self, ctx: &Context, name: &str, path: &Path, reserved: u64) -> Budget {
        let known = self.memory.lock().unwrap_or_else(|e| e.into_inner()).get(name).copied();
        let use_ = known.or_else(|| {
            let file = path.file_name()?.to_string_lossy().into_owned();
            tiles::known_memory(&self.dir.join("cache"), &file, self.backend.label(), &self.device_name)
        });
        Budget { mem: ctx.memory(), reserved, area_budget: self.area_budget, use_: use_.unwrap_or_else(MemoryUse::guess) }
    }

    /// `name` failed on the primary backend with `err`: on TensorRT-RTX, retry it
    /// on TensorRT (opening that context on first need); otherwise, or when it ran
    /// out of GPU memory, it stays failed.
    fn fall_back(
        &self,
        slots: &mut HashMap<String, Slot>,
        name: &str,
        path: &Path,
        err: String,
    ) -> Result<Option<(Arc<Model>, u32, bool)>, String> {
        let Some(fb) = self.fallback.as_ref().filter(|_| !yosh_image::aji::is_out_of_memory(&err)) else {
            slots.insert(name.to_string(), Slot::Failed);
            return Err(err);
        };
        eprintln!("yosh: [upscale] {} failed for {name} ({err}); using TensorRT", self.backend.label());
        self.fell_back
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(|| format!("{name}: {err}"));
        let ctx = {
            let mut c = fb.ctx.lock().unwrap_or_else(|e| e.into_inner());
            c.get_or_insert_with(|| open_context(&self.aji, &self.dir, Backend::TensorRt, self.device_id)).clone()
        };
        match ctx {
            Ok(ctx) => self.load(slots, &ctx, name, path, true, None),
            Err(e) => {
                slots.insert(name.to_string(), Slot::Failed);
                Err(format!("{err}; TensorRT fallback: {e}"))
            }
        }
    }

    /// The engine a `w`x`h` page of model `name` runs on: its fitted one if that's ready (`Some`
    /// of its shape), else `square` (the model as loaded), with the fitted shape it's renting
    /// the square for, if any (see the module docs). Starts the fitted engine's load once it's
    /// worth it. On TensorRT the model's even-tiles engine is the "square" and takes every page.
    fn pick(&self, name: &str, square: &Arc<Model>, fell_back: bool, w: u32, h: u32) -> (Arc<Model>, Option<Shape>, Option<Shape>) {
        let on_square = (square.clone(), None, None);
        if fell_back || self.backend == Backend::TensorRt {
            return on_square;
        }
        let Some(path) = self.resolve(name) else { return on_square };
        let clock = self.clock.fetch_add(1, Ordering::AcqRel) + 1;
        let mut fitted = self.fitted.lock().unwrap_or_else(|e| e.into_inner());
        let shape = *self.shapes.lock().unwrap_or_else(|e| e.into_inner()).entry((name.to_string(), (w, h))).or_insert_with(|| {
            let budget = self.budget(square.context(), name, &path, building_bytes(&fitted));
            tiles::fit(w, h, tiles::DEFAULT_OVERLAP, |s| budget.fits(s, (w, h)).0, SQUARE)
        });
        let Some((shape, _)) = shape else { return on_square };
        let key = (name.to_string(), shape);
        let cache = self.dir.join("cache");
        let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let state = match fitted.remove(&key) {
            Some(Fit::Building { model, bytes, square: sq }) => match model.state() {
                ModelState::Building => Fit::Building { model, bytes, square: sq },
                ModelState::Ready => {
                    tiles::record_fitted(&cache, &file, self.backend.label(), &self.device_name, shape);
                    self.measure(name, &path, &model);
                    eprintln!("yosh: [upscale] {name}: engine fitted to tile {} ready", tiles::label(shape));
                    Fit::Ready { model, last_use: 0, check: sq.map(|square| Check { square, runs: 0, took: 0.0 }) }
                }
                ModelState::Failed => {
                    eprintln!(
                        "yosh: [upscale] {name}: no engine fitted to tile {} ({}); staying on {SQUARE}",
                        tiles::label(shape),
                        model.context().last_error()
                    );
                    Fit::Failed
                }
            },
            // the second page of this size: a volume
            Some(Fit::Renting { square: sq }) => self.start_fitted(&mut fitted, &key, &path, square, (w, h), sq),
            Some(state) => state,
            None if tiles::fitted_known(&cache, &file, self.backend.label(), &self.device_name, shape) => {
                self.start_fitted(&mut fitted, &key, &path, square, (w, h), None)
            }
            None => Fit::Renting { square: None },
        };
        let (state, pick) = match state {
            Fit::Ready { model, check, .. } => (Fit::Ready { model: model.clone(), last_use: clock, check }, (model, None, Some(shape))),
            Fit::Renting { square: sq } => (Fit::Renting { square: sq }, (square.clone(), Some(shape), None)),
            other => (other, on_square),
        };
        fitted.insert(key, state);
        pick
    }

    /// Load `key`'s fitted engine (file `path`) on `square`'s context for `page`-size pages: at
    /// once if its engine is cached, else built in the background (the watcher picks it up).
    /// Unloads the model's least recently used fitted engines past [`MAX_FITTED`], or while it
    /// wouldn't fit in the GPU's free memory. `sq`: seconds per page on the square.
    fn start_fitted(
        &self,
        fitted: &mut HashMap<(String, Shape), Fit>,
        key: &(String, Shape),
        path: &Path,
        square: &Arc<Model>,
        page: Shape,
        sq: Option<f64>,
    ) -> Fit {
        let label = tiles::label(key.1);
        let room = |fitted: &HashMap<(String, Shape), Fit>| self.budget(square.context(), &key.0, path, building_bytes(fitted)).fits(key.1, page);
        let mut fits = room(fitted);
        loop {
            let loaded: Vec<_> = fitted
                .iter()
                .filter_map(|(k, f)| match f {
                    Fit::Ready { last_use, .. } if k.0 == key.0 => Some((*last_use, k.clone())),
                    _ => None,
                })
                .collect();
            if loaded.is_empty() || (fits.0 && loaded.len() < MAX_FITTED) {
                break;
            }
            let (_, oldest) = loaded.into_iter().min_by_key(|(t, _)| *t).expect("not empty");
            // It's known now, so seeing its size again loads it at once.
            fitted.remove(&oldest);
            fits = room(fitted);
        }
        let (fits, bytes) = fits;
        if !fits {
            eprintln!("yosh: [upscale] {}: no room in GPU memory for tile {label} (~{} MB); staying on {SQUARE}", key.0, bytes >> 20);
            return Fit::Failed;
        }
        let params = LoadParams { tile: Some(key.1), async_build: true, ..LoadParams::new(path) };
        let check = sq.map(|square| Check { square, runs: 0, took: 0.0 });
        match square.context().load_model(&params) {
            Ok(m) => {
                let m = Arc::new(m);
                match m.state() {
                    ModelState::Building => {
                        eprintln!("yosh: [upscale] {}: building an engine fitted to tile {label} in the background", key.0);
                        self.watch_build(m.clone());
                        Fit::Building { model: m, bytes, square: sq }
                    }
                    ModelState::Ready => {
                        if let Some(file) = path.file_name() {
                            let cache = self.dir.join("cache");
                            tiles::record_fitted(&cache, &file.to_string_lossy(), self.backend.label(), &self.device_name, key.1);
                        }
                        self.measure(&key.0, path, &m);
                        Fit::Ready { model: m, last_use: 0, check }
                    }
                    ModelState::Failed => {
                        eprintln!("yosh: [upscale] {}: no engine fitted to tile {label} ({})", key.0, square.context().last_error());
                        Fit::Failed
                    }
                }
            }
            Err(e) => {
                eprintln!("yosh: [upscale] {}: no engine fitted to tile {label} ({e})", key.0);
                Fit::Failed
            }
        }
    }

    /// Time a ready fitted engine's page (`secs`) against the square's, and drop it if it isn't
    /// faster on its first [`CHECK_PAGES`].
    fn check_fitted(&self, key: &(String, Shape), secs: f64) {
        let mut fitted = self.fitted.lock().unwrap_or_else(|e| e.into_inner());
        let Some(Fit::Ready { check: Some(c), .. }) = fitted.get_mut(key) else { return };
        c.runs += 1;
        c.took += secs;
        if c.runs < CHECK_PAGES {
            return;
        }
        let (mean, square) = (c.took / c.runs as f64, c.square);
        if mean > square * (1.0 - CHECK_GAIN) {
            eprintln!(
                "yosh: [upscale] {}: tile {} ran a page in {:.0} ms against {:.0} on {SQUARE}; staying on {SQUARE}",
                key.0,
                tiles::label(key.1),
                mean * 1e3,
                square * 1e3
            );
            fitted.insert(key.clone(), Fit::Failed);
        } else if let Some(Fit::Ready { check, .. }) = fitted.get_mut(key) {
            *check = None;
        }
    }

    /// Poll a building model in the background; when it (or any other) settles,
    /// bump the epoch and wake the shell so skipped pages are retried.
    fn watch_build(&self, m: Arc<Model>) {
        let w = &self.watch;
        w.building.lock().unwrap_or_else(|e| e.into_inner()).push(m);
        if w.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let w = w.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            let mut b = w.building.lock().unwrap_or_else(|e| e.into_inner());
            let before = b.len();
            b.retain(|m| m.state() == ModelState::Building);
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

/// The memory predicted for the fitted engines still building.
fn building_bytes(fitted: &HashMap<(String, Shape), Fit>) -> u64 {
    fitted.values().map(|f| if let Fit::Building { bytes, .. } = f { *bytes } else { 0 }).sum()
}

/// The precision suffixes of model file names: a file is named after its ONNX
/// precision (`4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16`).
const PRECISIONS: [&str; 3] = ["_bf16", "_fp16", "_fp32"];

/// The file stems to look for, in order, for a rule's model `name` on `backend`
/// with CUDA `compute_capability` (`None` on DirectML). The precisions go in
/// preference order: bf16, fp16, fp32 on TensorRT/TensorRT-RTX with compute
/// capability 8.0 or newer, else fp16, fp32 (never bf16, which DirectML can't run
/// and Turing and older have no support for).
///
/// - A name ending in a precision suffix: `name` itself, then the same model's
///   other precisions, then its unsuffixed name
///   (`4x_MangaJaNai_1200p_V1_ESRGAN_70k`), as older add-ons installed it. One
///   exception: a `_bf16` name on DirectML goes last, since DirectML can't run
///   it; tried only when nothing else is installed, its load error then says why.
/// - Any other name (an older `rules.json`'s unsuffixed MangaJaNai names, custom
///   models): `name` itself, then `name` with each precision suffix. For a custom
///   model those files don't exist, so it is in effect looked up exactly.
///
/// So the default rules and a `rules.json` with older or newer names find an
/// add-on's models whether it was built from older or newer MangaJaNai GUI files.
fn lookup_order(name: &str, backend: Backend, compute_capability: Option<(i32, i32)>) -> Vec<String> {
    let bf16 = backend.is_cuda() && compute_capability.is_some_and(|(major, _)| major >= 8);
    let prefs: &[&str] = if bf16 { &PRECISIONS } else { &PRECISIONS[1..] };
    let Some((base, suffix)) = PRECISIONS
        .iter()
        .find_map(|&s| name.strip_suffix(s).map(|b| (b, s)))
        .filter(|(b, _)| !b.is_empty())
    else {
        let mut order = vec![name.to_string()];
        if !name.is_empty() {
            order.extend(prefs.iter().map(|p| format!("{name}{p}")));
        }
        return order;
    };
    let bf16_on_dml = backend == Backend::DirectMl && suffix == "_bf16";
    let mut order = Vec::new();
    if !bf16_on_dml {
        order.push(name.to_string());
    }
    order.extend(prefs.iter().filter(|&&p| p != suffix).map(|p| format!("{base}{p}")));
    order.push(base.to_string());
    if bf16_on_dml {
        order.push(name.to_string());
    }
    order
}

/// Open `backend` on `device_id` with the add-on's engine cache, logging warnings
/// and errors to stderr.
fn open_context(aji: &Aji, dir: &Path, backend: Backend, device_id: u64) -> Result<Context, String> {
    let cache = dir.join("cache");
    let _ = std::fs::create_dir_all(&cache);
    let (trtexec, trtexec_env) = trtexec(dir);
    let _gpu = gpu_lock();
    aji.context(&ContextParams {
        backend,
        device_id,
        cache_dir: Some(cache),
        trtexec,
        trtexec_env,
        // TensorRT severities: 0 internal error, 1 error, 2 warning.
        log: Some(Arc::new(|level, msg| {
            if level <= 2 {
                eprintln!("yosh: [upscale] {msg}");
            }
        })),
    })
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
        self.active.fetch_add(1, Ordering::AcqRel);
        let _running = Running(self);
        let rules = self.rules();
        let Some(name) = select(&rules, h, channels == 1, target_scale) else {
            return SrOutcome::Skip;
        };
        let (square, scale, fell_back) = match self.model(name, (w, h)) {
            Ok(Some(m)) => m,
            Ok(None) => return SrOutcome::NotReady,
            Err(e) => return SrOutcome::Failed(e),
        };
        let (model, rent, on_fitted) = self.pick(name, &square, fell_back, w, h);
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        let run = |m: &Model, out: &mut [u8]| {
            let start = Instant::now();
            let rc = m.upscale(Pixels::packed(w, h, channels, pixels), PixelsMut::packed(ow, oh, channels, out), Some(&mut |_, _| cancel()));
            (rc, start.elapsed().as_secs_f64())
        };
        let (mut rc, secs) = run(&model, &mut out);
        if let (Some(shape), Err(e)) = (on_fitted, &rc) {
            // the fitted engine failed (out of memory, say): drop it, and redo the page on the square
            eprintln!("yosh: [upscale] {name}: tile {} failed ({e}); staying on {SQUARE}", tiles::label(shape));
            self.fitted.lock().unwrap_or_else(|e| e.into_inner()).insert((name.to_string(), shape), Fit::Failed);
            rc = run(&square, &mut out).0;
        } else if let (Some(shape), Ok(Upscaled::Done)) = (on_fitted, &rc) {
            self.check_fitted(&(name.to_string(), shape), secs);
        }
        if let (Some(shape), Ok(Upscaled::Done)) = (rent, &rc)
            && let Some(Fit::Renting { square }) =
                self.fitted.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&(name.to_string(), shape))
        {
            *square = Some(secs);
        }
        match rc {
            Ok(Upscaled::Done) => SrOutcome::Done { scale, pixels: out, model: name.into() },
            Ok(Upscaled::Cancelled) => SrOutcome::Cancelled,
            Err(e) => SrOutcome::Failed(e),
        }
    }

    fn epoch(&self) -> u64 {
        self.watch.epoch.load(Ordering::Acquire)
    }

    fn model_for(&self, src_h: u32, gray: bool, target_scale: f32) -> Option<Arc<str>> {
        select(&self.rules(), src_h, gray, target_scale).map(Arc::from)
    }

    fn auto_levels(&self, src_h: u32, gray: bool, target_scale: f32) -> bool {
        auto_levels(&self.rules(), src_h, gray, target_scale)
    }

    fn set_rules(&self, rules: Vec<SrRule>) {
        *self.rules.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(rules);
    }

    fn fallback(&self) -> Option<String> {
        self.fell_back.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn release_gpu(&self) -> bool {
        // Holding the slots keeps a model load out until the contexts are closed;
        // an upscale counts itself running before it asks for its model.
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let building = slots
            .values()
            .any(|s| matches!(s, Slot::Building { model, .. } if model.state() == ModelState::Building))
            || self
                .fitted
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .values()
                .any(|f| matches!(f, Fit::Building { model, .. } if model.state() == ModelState::Building));
        if self.active.load(Ordering::Acquire) > 0 || building {
            return false;
        }
        let released = self.close_all(&mut slots);
        if released {
            eprintln!("yosh: [upscale] released the GPU (idle)");
        }
        released
    }

    fn idle_since(&self) -> Option<Instant> {
        let open = self.ctx.lock().unwrap_or_else(|e| e.into_inner()).is_some()
            || self.fallback.as_ref().is_some_and(|f| f.ctx.lock().unwrap_or_else(|e| e.into_inner()).is_some());
        (open && self.active.load(Ordering::Acquire) == 0)
            .then(|| *self.last_used.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for AjiUpscaler {
    fn drop(&mut self) {
        // Close the models and contexts now (a running watcher then exits on its
        // next tick without waking anyone) and free the device, so switching
        // upscaling off gives back all of its GPU memory. A building engine's
        // context is closed too: libaji stops the build.
        let mut slots = std::mem::take(&mut *self.slots.lock().unwrap_or_else(|e| e.into_inner()));
        self.close_all(&mut slots);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(name: &str, backend: Backend, cc: Option<(i32, i32)>) -> Vec<String> {
        lookup_order(name, backend, cc)
    }

    #[test]
    fn suffixed_names_try_siblings_then_the_unsuffixed_name() {
        let m = "4x_MangaJaNai_1200p_V1_ESRGAN_70k";
        for (b, cc) in [(Backend::TensorRt, Some((8, 6))), (Backend::TensorRtRtx, Some((12, 0)))] {
            assert_eq!(
                order(&format!("{m}_fp16"), b, cc),
                [format!("{m}_fp16"), format!("{m}_bf16"), format!("{m}_fp32"), m.to_string()]
            );
            let d = "4x_IllustrationJaNai_V3detail_DAT2_28k";
            assert_eq!(
                order(&format!("{d}_bf16"), b, cc),
                [format!("{d}_bf16"), format!("{d}_fp16"), format!("{d}_fp32"), d.to_string()]
            );
            assert_eq!(
                order(&format!("{d}_fp32"), b, cc),
                [format!("{d}_fp32"), format!("{d}_bf16"), format!("{d}_fp16"), d.to_string()]
            );
        }
    }

    #[test]
    fn no_bf16_sibling_on_directml_turing_or_unknown() {
        let d = "4x_IllustrationJaNai_V3detail_DAT2_28k";
        for (b, cc) in [
            (Backend::DirectMl, None),
            (Backend::TensorRt, Some((7, 5))),
            (Backend::TensorRtRtx, Some((7, 5))),
            (Backend::TensorRt, None),
        ] {
            assert_eq!(order(&format!("{d}_fp32"), b, cc), [format!("{d}_fp32"), format!("{d}_fp16"), d.to_string()]);
            assert_eq!(order(&format!("{d}_fp16"), b, cc), [format!("{d}_fp16"), format!("{d}_fp32"), d.to_string()]);
        }
        // A bf16 name stays first where it may run (Turing: TensorRT decides) ...
        assert_eq!(
            order(&format!("{d}_bf16"), Backend::TensorRt, Some((7, 5))),
            [format!("{d}_bf16"), format!("{d}_fp16"), format!("{d}_fp32"), d.to_string()]
        );
        // ... but goes last on DirectML, which can't run it.
        assert_eq!(
            order(&format!("{d}_bf16"), Backend::DirectMl, None),
            [format!("{d}_fp16"), format!("{d}_fp32"), d.to_string(), format!("{d}_bf16")]
        );
    }

    #[test]
    fn unsuffixed_names_try_the_suffixed_files_after_the_exact_name() {
        for name in [
            "4x_MangaJaNai_1200p_V1_ESRGAN_70k",
            "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16_fp32_op20",
            "my_model_FP16",
            "_fp16",
        ] {
            for (b, cc) in [(Backend::TensorRt, Some((12, 0))), (Backend::TensorRtRtx, Some((8, 0)))] {
                assert_eq!(
                    order(name, b, cc),
                    [name.to_string(), format!("{name}_bf16"), format!("{name}_fp16"), format!("{name}_fp32")],
                    "{name} {b:?}"
                );
            }
            // bf16 only where it runs.
            for (b, cc) in [(Backend::DirectMl, None), (Backend::TensorRt, Some((7, 5))), (Backend::TensorRtRtx, None)] {
                assert_eq!(
                    order(name, b, cc),
                    [name.to_string(), format!("{name}_fp16"), format!("{name}_fp32")],
                    "{name} {b:?}"
                );
            }
        }
        assert_eq!(order("", Backend::TensorRt, Some((12, 0))), [String::new()]);
    }

    #[cfg(unix)]
    #[test]
    fn a_portable_trtexec_gets_the_add_on_on_its_library_path() {
        let dir = std::env::temp_dir().join(format!("yosh-trtexec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("trtexec"), b"").unwrap();
        let (exe, env) = trtexec(&dir);
        assert_eq!(exe, None, "libaji runs the one beside it");
        let canon = std::fs::canonicalize(&dir).unwrap();
        assert_eq!(
            env.as_deref(),
            Some(format!("LD_LIBRARY_PATH='{}'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}", canon.display()).as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn windows_leaves_trtexec_to_libaji() {
        assert_eq!(trtexec(Path::new(r"C:\nowhere")), (None, None));
    }

    /// This process's dedicated GPU memory in MiB (Windows' "GPU Process Memory"
    /// counter; nvidia-smi has no per-process numbers on WDDM).
    #[cfg(windows)]
    fn gpu_mib() -> f64 {
        std::thread::sleep(Duration::from_secs(2)); // the counter lags
        let q = format!(
            r"((Get-Counter '\GPU Process Memory(pid_{}_*)\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum",
            std::process::id()
        );
        let out = std::process::Command::new("powershell").args(["-NoProfile", "-Command", &q]).output().unwrap();
        String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap_or(0.0) / (1024.0 * 1024.0)
    }

    /// Manual (`--ignored`): `YOSH_TEST_ADDON` = an add-on folder (use a copy: this writes its
    /// `cache/`), `YOSH_TEST_MODEL` = a 2x model stem in its `models/`, `YOSH_TEST_BACKEND` as
    /// below. Upscales a 1200x1800 gray page over and over, as a volume would: the first pages
    /// run on the square, then the fitted engine builds in the background and takes over. A
    /// second upscaler on the same add-on then uses it from the first page (the fitted record),
    /// and its output matches the square's but at the tile seams. On TensorRT: the model loads
    /// as one even-tiles engine larger than the square, and nothing is fitted.
    #[test]
    #[ignore]
    fn a_fitted_engine_takes_over_a_volume() {
        let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON").expect("YOSH_TEST_ADDON"));
        let model = std::env::var("YOSH_TEST_MODEL").expect("YOSH_TEST_MODEL");
        let backend = match std::env::var("YOSH_TEST_BACKEND").as_deref() {
            Ok("tensorrt") => Backend::TensorRt,
            Ok("directml") => Backend::DirectMl,
            _ => Backend::TensorRtRtx,
        };
        let device = enum_devices(&dir, backend).unwrap().into_iter().next().expect("a device");
        let rules = vec![SrRule {
            min_h: 0,
            max_h: 0,
            gray: true,
            color: true,
            min_scale: 0.0,
            max_scale: 0.0,
            model: Some(model.clone()),
            auto_levels: false,
        }];
        let (w, h) = (1200u32, 1800u32);
        let px: Vec<u8> = (0..w * h).map(|i| ((i % w) ^ (i / w)) as u8 / 2 + (i.wrapping_mul(2_654_435_761) >> 26) as u8).collect();
        let run = |up: &AjiUpscaler| loop {
            match up.upscale(w, h, 1, &px, 2.0, &|| false) {
                SrOutcome::Done { pixels, .. } => break pixels,
                SrOutcome::NotReady => std::thread::sleep(Duration::from_millis(200)),
                SrOutcome::Failed(e) => panic!("{e}"),
                _ => panic!("no model for the page"),
            }
        };
        let on_fitted = |up: &AjiUpscaler| {
            up.fitted.lock().unwrap().values().any(|f| matches!(f, Fit::Ready { .. }))
        };

        if backend == Backend::TensorRt {
            // no fitted engines: one even-tiles engine for every page size, as large as fits
            let up = AjiUpscaler::new(&dir, backend, &device, rules, None).unwrap();
            let _ = run(&up);
            let tile = match up.slots.lock().unwrap().get(&model) {
                Some(Slot::Ready { model, .. }) => model.info().unwrap().tile_w,
                _ => 0,
            };
            let t = Instant::now();
            for _ in 0..10 {
                let _ = run(&up);
            }
            eprintln!("TensorRT: even tiles of up to {tile}, {:.0} ms a page", t.elapsed().as_secs_f64() * 100.0);
            assert!(tile > SQUARE, "an even tile larger than the square");
            assert!(!on_fitted(&up), "no fitted engines on TensorRT");
            return;
        }

        let up = AjiUpscaler::new(&dir, backend, &device, rules.clone(), None).unwrap();
        let square_out = run(&up);
        let (mut square_ms, mut fitted_ms, mut pages) = (Vec::new(), Vec::new(), 0);
        let started = Instant::now();
        while pages < 400 && fitted_ms.len() < 10 {
            let fitted = on_fitted(&up);
            let t = Instant::now();
            let out = run(&up);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if fitted && fitted_ms.is_empty() {
                let diff = out.iter().zip(&square_out).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum::<u64>() as f64
                    / out.len() as f64;
                eprintln!("fitted output: mean abs difference from the square's {diff:.3}");
                assert!(diff < 1.0, "the fitted engine upscales the same but at seams");
            }
            if fitted { fitted_ms.push(ms) } else { square_ms.push(ms) }
            pages += 1;
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let (sq, fi) = (square_ms.len(), fitted_ms.len());
        eprintln!(
            "{}: {sq} pages on {SQUARE} (median {:.0} ms), then {fi} fitted (median {:.0} ms), {:.1} s",
            backend.label(),
            med(&mut square_ms),
            med(&mut fitted_ms),
            started.elapsed().as_secs_f64()
        );
        assert!(fi > 0, "the fitted engine took over");
        drop(up);

        let again = AjiUpscaler::new(&dir, backend, &device, rules, None).unwrap();
        let _ = run(&again);
        // A cached TensorRT-RTX engine still warms up in the background (a fraction of a second).
        let loading = again.fitted.lock().unwrap().values().any(|f| matches!(f, Fit::Ready { .. } | Fit::Building { .. }));
        assert!(loading, "a known fitted engine loads with the first page");
    }

    /// Manual (`--ignored`, Windows): `YOSH_TEST_ADDON` = an installed add-on folder
    /// (`YOSH_TEST_BACKEND` = tensorrt (default), tensorrt-rtx or directml). Upscales a
    /// gray page, releases the GPU, upscales it again (same output, reloaded from the
    /// engine cache), then drops the upscaler; prints the process's GPU memory at
    /// each step and checks a release gives back what the upscaler held.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn release_gpu_frees_the_device_and_reloads() {
        let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON").expect("YOSH_TEST_ADDON"));
        let backend = match std::env::var("YOSH_TEST_BACKEND").as_deref() {
            Ok("tensorrt-rtx") => Backend::TensorRtRtx,
            Ok("directml") => Backend::DirectMl,
            _ => Backend::TensorRt,
        };
        let device = enum_devices(&dir, backend).unwrap().into_iter().next().expect("a device");
        let start = gpu_mib();
        let up = AjiUpscaler::new(&dir, backend, &device, crate::sr::rules::default_rules(), None).unwrap();
        let (w, h) = (840u32, 1200u32);
        let px: Vec<u8> = (0..w * h).map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect();
        let run = || loop {
            match up.upscale(w, h, 1, &px, 1.5, &|| false) {
                SrOutcome::Done { pixels, scale, .. } => break (pixels, scale),
                SrOutcome::NotReady => std::thread::sleep(Duration::from_millis(500)),
                SrOutcome::Failed(e) => panic!("{e}"),
                _ => panic!("no model for the page"),
            }
        };
        let (first, scale) = run();
        assert_eq!(scale, 2);
        let loaded = gpu_mib();
        assert!(up.idle_since().is_some());
        assert!(up.release_gpu());
        assert!(up.idle_since().is_none() && !up.release_gpu(), "nothing left to release");
        let released = gpu_mib();
        let t = Instant::now();
        let (again, _) = run();
        let reload_ms = t.elapsed().as_millis();
        assert!(again == first, "the reloaded model upscales the same");
        let reloaded = gpu_mib();
        drop(up);
        let dropped = gpu_mib();
        eprintln!(
            "{}: start {start:.0} MiB, loaded {loaded:.0}, released {released:.0}, reloaded {reloaded:.0} (first page after the release {reload_ms} ms), dropped {dropped:.0}",
            backend.label()
        );
        assert!(loaded - released > 0.8 * (loaded - start), "a release gives back what the upscaler held");
        assert!(reloaded - dropped > 0.8 * (reloaded - start), "dropping the upscaler does too");
    }
}

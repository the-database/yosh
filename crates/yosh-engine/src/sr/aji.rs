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
//! happens, and the other models stay on TensorRT-RTX.
//!
//! [`Upscaler::release_gpu`] (the shell calls it while upscaling is idle) closes the
//! models and the contexts and has libaji (0.24.0+) free the device, which on the
//! CUDA backends drops the CUDA context TensorRT's runtime would keep until exit;
//! the next page reopens the context and reloads its model from the engine cache.
//! Dropping the upscaler does the same. Opening a context and freeing a device are
//! serialized process-wide ([`GPU`]): a device reset must never overlap another
//! upscaler opening its context on the same GPU.
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

use super::rules::{select, SrRule};
use super::{SrOutcome, Upscaler};
use crate::pool::Waker;

/// Serializes opening a context, listing devices and freeing a device across the
/// process: `aji_img_release_device` resets the device for everything in it.
static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Devices `backend` can run on, from the add-on in `dir`.
pub fn enum_devices(dir: &Path, backend: Backend) -> Result<Vec<Device>, String> {
    let aji = Aji::load(dir)?;
    let _gpu = gpu_lock();
    aji.devices(backend)
}

enum Slot {
    Ready { model: Arc<Model>, scale: u32 },
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
        let aji = Aji::load(dir)?;
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

    /// The model and its scale once ready; loads it on first use.
    fn model(&self, name: &str) -> Result<Option<(Arc<Model>, u32)>, String> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Slot::Building { model, fell_back, path }) = slots.get(name) {
            let (m, fell_back, path) = (model.clone(), *fell_back, path.clone());
            match m.state() {
                ModelState::Building => return Ok(None),
                ModelState::Ready => {
                    let scale = m.info()?.scale;
                    slots.insert(name.to_string(), Slot::Ready { model: m, scale });
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
            Some(Slot::Ready { model, scale }) => return Ok(Some((model.clone(), *scale))),
            Some(Slot::Failed) => return Err(format!("model {name} failed to load")),
            _ => {}
        }
        let Some(path) = self.resolve(name) else {
            slots.insert(name.to_string(), Slot::Failed);
            let path = self.models_dirs[0].join(format!("{name}.onnx"));
            return Err(format!("model not installed: {}", path.display()));
        };
        let ctx = self.primary()?;
        match self.load(&mut slots, &ctx, name, &path, false) {
            Err(e) => self.fall_back(&mut slots, name, &path, e),
            done => done,
        }
    }

    /// Load `path` on `ctx` into `name`'s slot. Library-default tile/overlap
    /// (512 px, 32 px). Missing engines build in the background — a reader must
    /// never stall on a first-use build; the watcher below picks up the result.
    fn load(
        &self,
        slots: &mut HashMap<String, Slot>,
        ctx: &Context,
        name: &str,
        path: &Path,
        fell_back: bool,
    ) -> Result<Option<(Arc<Model>, u32)>, String> {
        let params = LoadParams { async_build: true, ..LoadParams::new(path) };
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
        slots.insert(name.to_string(), Slot::Ready { model: m.clone(), scale });
        Ok(Some((m, scale)))
    }

    /// `name` failed on the primary backend with `err`: on TensorRT-RTX, retry it
    /// on TensorRT (opening that context on first need); otherwise it stays failed.
    fn fall_back(
        &self,
        slots: &mut HashMap<String, Slot>,
        name: &str,
        path: &Path,
        err: String,
    ) -> Result<Option<(Arc<Model>, u32)>, String> {
        let Some(fb) = &self.fallback else {
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
            Ok(ctx) => self.load(slots, &ctx, name, path, true),
            Err(e) => {
                slots.insert(name.to_string(), Slot::Failed);
                Err(format!("{err}; TensorRT fallback: {e}"))
            }
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
    let _gpu = gpu_lock();
    aji.context(&ContextParams {
        backend,
        device_id,
        cache_dir: Some(cache),
        trtexec: None,
        trtexec_env: None,
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
        let (model, scale) = match self.model(name) {
            Ok(Some(m)) => m,
            Ok(None) => return SrOutcome::NotReady,
            Err(e) => return SrOutcome::Failed(e),
        };
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        let rc = model.upscale(
            Pixels::packed(w, h, channels, pixels),
            PixelsMut::packed(ow, oh, channels, &mut out),
            Some(&mut |_, _| cancel()),
        );
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
            .any(|s| matches!(s, Slot::Building { model, .. } if model.state() == ModelState::Building));
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

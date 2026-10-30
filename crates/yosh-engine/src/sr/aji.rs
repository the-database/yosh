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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use yosh_image::aji::{Aji, Context, ContextParams, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};
pub use yosh_image::aji::{Backend, Device};

use super::rules::{select, SrRule};
use super::{SrOutcome, Upscaler};
use crate::pool::Waker;

/// Devices `backend` can run on, from the add-on in `dir`.
pub fn enum_devices(dir: &Path, backend: Backend) -> Result<Vec<Device>, String> {
    Aji::load(dir)?.devices(backend)
}

enum Slot {
    Ready { model: Arc<Model>, scale: u32 },
    Building(Arc<Model>),
    Failed,
}

/// Shared with the build watcher thread.
struct Watch {
    building: Mutex<Vec<Arc<Model>>>,
    running: AtomicBool,
    epoch: AtomicU64,
    waker: Option<Waker>,
}

pub struct AjiUpscaler {
    ctx: Context,
    models_dir: PathBuf,
    rules: Vec<SrRule>,
    slots: Mutex<HashMap<String, Slot>>,
    watch: Arc<Watch>,
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
        let cache = dir.join("cache");
        let _ = std::fs::create_dir_all(&cache);
        let ctx = Aji::load(dir)?.context(&ContextParams {
            backend,
            device_id,
            cache_dir: Some(cache),
            trtexec: None,
            // TensorRT severities: 0 internal error, 1 error, 2 warning.
            log: Some(Arc::new(|level, msg| {
                if level <= 2 {
                    eprintln!("yosh: [upscale] {msg}");
                }
            })),
        })?;
        Ok(Self {
            ctx,
            models_dir: dir.join("models"),
            rules,
            slots: Mutex::new(HashMap::new()),
            watch: Arc::new(Watch {
                building: Mutex::new(Vec::new()),
                running: AtomicBool::new(false),
                epoch: AtomicU64::new(0),
                waker,
            }),
        })
    }

    /// The model and its scale once ready; loads it on first use.
    fn model(&self, name: &str) -> Result<Option<(Arc<Model>, u32)>, String> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Slot::Building(m)) = slots.get(name) {
            let m = m.clone();
            match m.state() {
                ModelState::Building => return Ok(None),
                ModelState::Ready => {
                    let scale = m.info()?.scale;
                    slots.insert(name.to_string(), Slot::Ready { model: m, scale });
                }
                ModelState::Failed => {
                    slots.insert(name.to_string(), Slot::Failed);
                    return Err(self.ctx.last_error());
                }
            }
        }
        match slots.get(name) {
            Some(Slot::Ready { model, scale }) => return Ok(Some((model.clone(), *scale))),
            Some(Slot::Failed) => return Err(format!("model {name} failed to load")),
            _ => {}
        }
        let path = self.models_dir.join(format!("{name}.onnx"));
        if !path.is_file() {
            slots.insert(name.to_string(), Slot::Failed);
            return Err(format!("model not installed: {}", path.display()));
        }
        // Library-default tile/overlap (512 px, 32 px). Missing TensorRT engines
        // build in the background — a reader must never stall on a first-use
        // build; the watcher below picks up the result.
        let params = LoadParams { async_build: true, ..LoadParams::new(&path) };
        let m = match self.ctx.load_model(&params) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                slots.insert(name.to_string(), Slot::Failed);
                return Err(e);
            }
        };
        if m.state() == ModelState::Building {
            slots.insert(name.to_string(), Slot::Building(m.clone()));
            self.watch_build(m);
            return Ok(None);
        }
        let scale = m.info()?.scale;
        slots.insert(name.to_string(), Slot::Ready { model: m.clone(), scale });
        Ok(Some((m, scale)))
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
        select(&self.rules, src_h, gray, target_scale).map(Arc::from)
    }
}

impl Drop for AjiUpscaler {
    fn drop(&mut self) {
        // Models hold the context and the context the library, so release order
        // takes care of itself; just let a running watcher exit on its next tick
        // (without waking anyone) instead of keeping building models alive.
        self.watch.building.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

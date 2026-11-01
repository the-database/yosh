//! The inference side: one libaji context on the chosen backend/device, the models loaded so far
//! (by file and tile), and the upscale itself — including recovering alpha for transparent images.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use yosh_image::aji::{Aji, Backend, Context, ContextParams, Device, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};

use crate::{precision, progress};

/// Which backend to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// TensorRT if an NVIDIA GPU and the TensorRT runtime are there, else DirectML.
    Auto,
    TensorRt,
    DirectMl,
    /// TensorRT for RTX (opt-in, experimental; needs aji_trtrtx.dll beside aji.dll), falling
    /// back to TensorRT when it can't start, and per model when one fails to build.
    TensorRtRtx,
}

impl BackendChoice {
    /// The backends `Upscaler::open` tries, in order.
    fn candidates(self) -> &'static [Backend] {
        match self {
            BackendChoice::Auto => &[Backend::TensorRt, Backend::DirectMl],
            BackendChoice::TensorRt => &[Backend::TensorRt],
            BackendChoice::DirectMl => &[Backend::DirectMl],
            BackendChoice::TensorRtRtx => &[Backend::TensorRtRtx, Backend::TensorRt],
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "trt" | "tensorrt" => Ok(Self::TensorRt),
            "dml" | "directml" => Ok(Self::DirectMl),
            "tensorrt-rtx" | "trt-rtx" | "trtrtx" => Ok(Self::TensorRtRtx),
            _ => Err(format!("unknown backend '{s}' (auto, tensorrt, directml, tensorrt-rtx)")),
        }
    }
}

pub fn backend_name(b: Backend) -> &'static str {
    b.label()
}

/// A chain's model file: as given if it's a path that exists, else looked up by name in the models
/// folders (the user's first, then the bundled ones). Exactly that file: no variant is swapped in.
fn resolve_in(model: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    let p = Path::new(model);
    if p.is_absolute() && p.is_file() {
        return Some(p.to_path_buf());
    }
    dirs.iter().map(|d| d.join(model)).find(|p| p.is_file())
}

pub struct Upscaler {
    ctx: Context,
    backend: Backend,
    /// The device `ctx` runs on, for the precision hints.
    device: Device,
    /// The model files whose precision hint was already given (once per file, not per tile).
    hinted: HashSet<PathBuf>,
    /// On TensorRT-RTX: where to open the TensorRT context a failed model falls back to.
    fallback: Option<Fallback>,
    model_dirs: Vec<PathBuf>,
    models: HashMap<(PathBuf, u32), Model>,
    pub description: String,
}

impl Upscaler {
    /// Open the backend: the requested one, or for Auto the first that has a usable device. A
    /// device id of 0 means the backend's first device.
    pub fn open(
        aji_dir: &Path,
        choice: BackendChoice,
        device_id: u64,
        cache_dir: Option<PathBuf>,
        model_dirs: Vec<PathBuf>,
    ) -> Result<Self, String> {
        let aji = Aji::load(aji_dir)?;
        let mut err = String::from("no compatible GPU found");
        for &backend in choice.candidates() {
            if choice == BackendChoice::TensorRtRtx && backend == Backend::TensorRt {
                progress::backend_fallback(None, &err);
            }
            let devices = match aji.devices(backend) {
                Ok(d) if !d.is_empty() => d,
                Ok(_) => continue,
                Err(e) => {
                    err = format!("{}: {e}", backend_name(backend));
                    continue;
                }
            };
            let device = devices.iter().find(|d| d.id == device_id).unwrap_or(&devices[0]);
            match open_context(&aji, backend, device.id, &cache_dir) {
                Ok(ctx) => {
                    let description = format!("{} on {}", backend_name(backend), device.name);
                    let fallback = (backend == Backend::TensorRtRtx).then(|| Fallback {
                        aji: aji.clone(),
                        device_id: device.id,
                        cache_dir: cache_dir.clone(),
                        ctx: None,
                    });
                    return Ok(Self {
                        ctx,
                        backend,
                        device: device.clone(),
                        hinted: HashSet::new(),
                        fallback,
                        model_dirs,
                        models: HashMap::new(),
                        description,
                    });
                }
                Err(e) => err = format!("{}: {e}", backend_name(backend)),
            }
        }
        Err(err)
    }

    /// A chain's model file (see [`resolve_in`]).
    pub fn resolve(&self, model: &str) -> Option<PathBuf> {
        resolve_in(model, &self.model_dirs)
    }

    /// The loaded model for `path` at `tile`, loading it (and waiting out a first-use TensorRT
    /// engine build, reported on stdout) if needed. On TensorRT-RTX, a model that fails to load
    /// or build is retried on TensorRT (that context opens the first time it's needed). The first
    /// load of a file prints its precision hint, if it has one, before anything else about it.
    fn model(&mut self, path: &Path, tile: u32) -> Result<&Model, String> {
        let key = (path.to_path_buf(), tile);
        if !self.models.contains_key(&key) {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if self.hinted.insert(path.to_path_buf()) {
                // A file that can't be read is reported by the load below.
                let p = precision::model_weights(path).map_or(precision::Precision::Unknown, |w| w.precision());
                let d = &self.device;
                if let Some(h) = precision::hint(self.backend, d.compute_capability, &d.name, p) {
                    progress::hint(&name, &h);
                }
            }
            let model = match load_model(&self.ctx, self.backend, path, tile, &name) {
                Ok(m) => m,
                Err(e) => {
                    let Some(fb) = self.fallback.as_mut() else {
                        return Err(format!("{name}: {e}"));
                    };
                    progress::backend_fallback(Some(&name), &e);
                    let ctx = fb.context()?;
                    load_model(&ctx, Backend::TensorRt, path, tile, &name).map_err(|e| format!("{name}: {e}"))?
                }
            };
            self.models.insert(key.clone(), model);
        }
        Ok(&self.models[&key])
    }

    /// Upscale `w`x`h` pixels with `channels` 1 (gray) or 3 (RGB) through the model. Returns the
    /// output and its scale.
    pub fn upscale(&mut self, path: &Path, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let model = self.model(path, tile)?;
        let scale = model.info()?.scale;
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        match model.upscale(
            Pixels::packed(w, h, channels as u32, pixels),
            PixelsMut::packed(ow, oh, channels as u32, &mut out),
            None,
        )? {
            Upscaled::Done => Ok((out, scale)),
            Upscaled::Cancelled => Err("cancelled".into()),
        }
    }

    /// Upscale an image with alpha (`channels` 2 = gray + alpha, 4 = RGBA, straight alpha): the
    /// color is composited once onto black and once onto white, both are upscaled, and alpha is
    /// recovered from how far apart they come out (1 − (white − black)); the color is the black
    /// pass un-premultiplied. The 1.x approach, which models trained without alpha handle well.
    pub fn upscale_with_alpha(&mut self, path: &Path, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let cc = (channels - 1) as usize;
        let n = channels as usize;
        let mut on_black = Vec::with_capacity(pixels.len() / n * cc);
        let mut on_white = Vec::with_capacity(on_black.capacity());
        for px in pixels.chunks_exact(n) {
            let a = px[cc] as u32;
            for &v in &px[..cc] {
                on_black.push(((v as u32 * a + 127) / 255) as u8);
                on_white.push(((v as u32 * a + 255 * (255 - a) + 127) / 255) as u8);
            }
        }
        let (black, scale) = self.upscale(path, tile, w, h, cc as u8, &on_black)?;
        let (white, _) = self.upscale(path, tile, w, h, cc as u8, &on_white)?;
        let mut out = Vec::with_capacity(black.len() / cc * n);
        for (b, wh) in black.chunks_exact(cc).zip(white.chunks_exact(cc)) {
            let diff: i32 = b.iter().zip(wh).map(|(&b, &w)| w as i32 - b as i32).sum::<i32>() / cc as i32;
            let a = (255 - diff).clamp(0, 255) as u32;
            for &v in b {
                out.push((v as u32 * 255 + a / 2).checked_div(a).map_or(0, |c| c.min(255) as u8));
            }
            out.push(a as u8);
        }
        Ok((out, scale))
    }
}

/// The TensorRT context a TensorRT-RTX upscaler falls back to, opened on first need.
struct Fallback {
    aji: Aji,
    device_id: u64,
    cache_dir: Option<PathBuf>,
    ctx: Option<Result<Context, String>>,
}

impl Fallback {
    fn context(&mut self) -> Result<Context, String> {
        let (aji, id, cache) = (&self.aji, self.device_id, &self.cache_dir);
        self.ctx
            .get_or_insert_with(|| open_context(aji, Backend::TensorRt, id, cache))
            .clone()
            .map_err(|e| format!("TensorRT fallback: {e}"))
    }
}

fn open_context(aji: &Aji, backend: Backend, device_id: u64, cache_dir: &Option<PathBuf>) -> Result<Context, String> {
    aji.context(&ContextParams {
        backend,
        device_id,
        cache_dir: cache_dir.clone(),
        trtexec: None,
        log: Some(std::sync::Arc::new(|level, msg| {
            if level <= 2 {
                eprintln!("[aji] {msg}");
            }
        })),
    })
}

/// Load `path` at `tile` on `ctx`, waiting out a background engine build (reported on stdout).
fn load_model(ctx: &Context, backend: Backend, path: &Path, tile: u32, name: &str) -> Result<Model, String> {
    let model = ctx.load_model(&LoadParams { onnx: path, tile: Some((tile, tile)), overlap: None, async_build: true })?;
    if model.state() == ModelState::Building {
        progress::engine_build(name, tile, backend.label());
        while model.state() == ModelState::Building {
            std::thread::sleep(Duration::from_millis(250));
        }
        if model.state() == ModelState::Ready {
            progress::engine_ready(name);
        }
    }
    if model.state() == ModelState::Failed {
        return Err(ctx.last_error());
    }
    Ok(model)
}

/// The devices of one backend, as JSON for the GUI's device list:
/// `[{"id", "name", "vram", "compute_capability"}]` (`id` as a decimal string — it's a full 64-bit
/// value; `compute_capability` as "major.minor", null on DirectML). An unloadable backend is an
/// empty list.
pub fn devices_json(aji_dir: &Path, backend: Backend) -> Result<String, String> {
    let aji = Aji::load(aji_dir)?;
    let devices = aji.devices(backend).unwrap_or_default();
    let list: Vec<_> = devices
        .iter()
        .map(|d| {
            serde_json::json!({
                "id": d.id.to_string(),
                "name": d.name,
                "vram": d.vram_bytes,
                "compute_capability": d.compute_capability.map(|(major, minor)| format!("{major}.{minor}")),
            })
        })
        .collect();
    serde_json::to_string(&list).map_err(|e| e.to_string())
}

/// `--model-info`'s JSON object, fields in the documented order.
#[derive(serde::Serialize)]
struct ModelInfo {
    precision: &'static str,
    weights: precision::Weights,
    hint: Option<String>,
}

/// The precision of the ONNX model at `path`, as JSON:
/// `{"precision", "weights": {"fp32", "fp16", "bf16"}, "hint"}`. With a `backend`, `hint` is
/// [`precision::hint`] for the device an upscale would run on (see [`hint_device`]); without one,
/// or when no device is found, it is null.
pub fn model_info_json(path: &Path, aji_dir: &Path, backend: Option<BackendChoice>, device_id: u64) -> Result<String, String> {
    let weights = precision::model_weights(path)?;
    let p = weights.precision();
    let hint = backend.and_then(|choice| {
        let (backend, device) = hint_device(aji_dir, choice, device_id)?;
        precision::hint(backend, device.compute_capability, &device.name, p)
    });
    serde_json::to_string(&ModelInfo { precision: p.as_str(), weights, hint }).map_err(|e| e.to_string())
}

/// The backend and device an upscale with `choice` and `device_id` would run on, as far as the
/// device lists tell (no context is opened): the first of [`BackendChoice::candidates`] that lists
/// a device, and on it the device with that id, else its first.
fn hint_device(aji_dir: &Path, choice: BackendChoice, device_id: u64) -> Option<(Backend, Device)> {
    let aji = match Aji::load(aji_dir) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no hint: {e}");
            return None;
        }
    };
    choice.candidates().iter().find_map(|&b| {
        let devices = aji.devices(b).ok()?;
        let device = devices.iter().find(|d| d.id == device_id).or(devices.first())?;
        Some((b, device.clone()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_takes_the_first_models_folder_that_has_the_file() {
        let root = std::env::temp_dir().join(format!("mangajanai-cli-resolve-test-{}", std::process::id()));
        let (user, bundled) = (root.join("user"), root.join("bundled"));
        for d in [&user, &bundled] {
            std::fs::create_dir_all(d).unwrap();
        }
        let touch = |p: PathBuf| {
            std::fs::write(&p, b"").unwrap();
            p
        };
        let a = touch(bundled.join("a.onnx"));
        let b = touch(user.join("b.onnx"));
        touch(bundled.join("b.onnx"));
        let dirs = [user.clone(), bundled.clone()];

        assert_eq!(resolve_in("a.onnx", &dirs), Some(a.clone()));
        // The user's folder comes first.
        assert_eq!(resolve_in("b.onnx", &dirs), Some(b));
        // A path the user picked is used as is.
        assert_eq!(resolve_in(a.to_str().unwrap(), &dirs), Some(a));
        assert_eq!(resolve_in("missing.onnx", &dirs), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}

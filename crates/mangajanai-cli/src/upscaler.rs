//! The inference side: one libaji context on the chosen backend/device, the models loaded so far
//! (by file and tile), and the upscale itself — including recovering alpha for transparent images.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use yosh_image::aji::{Aji, Backend, Context, ContextParams, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};

use crate::progress;

/// Which backend to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// TensorRT if an NVIDIA GPU and the TensorRT runtime are there, else DirectML.
    Auto,
    TensorRt,
    DirectMl,
}

impl BackendChoice {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "trt" | "tensorrt" => Ok(Self::TensorRt),
            "dml" | "directml" => Ok(Self::DirectMl),
            _ => Err(format!("unknown backend '{s}' (auto, tensorrt, directml)")),
        }
    }
}

pub fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::TensorRt => "TensorRT",
        Backend::DirectMl => "DirectML",
    }
}

pub struct Upscaler {
    ctx: Context,
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
        let candidates: &[Backend] = match choice {
            BackendChoice::Auto => &[Backend::TensorRt, Backend::DirectMl],
            BackendChoice::TensorRt => &[Backend::TensorRt],
            BackendChoice::DirectMl => &[Backend::DirectMl],
        };
        let mut err = String::from("no compatible GPU found");
        for &backend in candidates {
            let devices = match aji.devices(backend) {
                Ok(d) if !d.is_empty() => d,
                Ok(_) => continue,
                Err(e) => {
                    err = format!("{}: {e}", backend_name(backend));
                    continue;
                }
            };
            let device = devices.iter().find(|d| d.id == device_id).unwrap_or(&devices[0]);
            let params = ContextParams {
                backend,
                device_id: device.id,
                cache_dir: cache_dir.clone(),
                trtexec: None,
                log: Some(std::sync::Arc::new(|level, msg| {
                    if level <= 2 {
                        eprintln!("[aji] {msg}");
                    }
                })),
            };
            match aji.context(&params) {
                Ok(ctx) => {
                    let description = format!("{} on {}", backend_name(backend), device.name);
                    return Ok(Self { ctx, model_dirs, models: HashMap::new(), description });
                }
                Err(e) => err = format!("{}: {e}", backend_name(backend)),
            }
        }
        Err(err)
    }

    /// A chain's model file: as given if it's a path that exists, else looked up by name in the
    /// models folders (the user's first, then the bundled ones).
    pub fn resolve(&self, model: &str) -> Option<PathBuf> {
        let p = Path::new(model);
        if p.is_absolute() && p.is_file() {
            return Some(p.to_path_buf());
        }
        self.model_dirs.iter().map(|d| d.join(model)).find(|p| p.is_file())
    }

    /// The loaded model for `path` at `tile`, loading it (and waiting out a first-use TensorRT
    /// engine build, reported on stdout) if needed.
    fn model(&mut self, path: &Path, tile: u32) -> Result<&Model, String> {
        let key = (path.to_path_buf(), tile);
        if !self.models.contains_key(&key) {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let model = self.ctx.load_model(&LoadParams {
                onnx: path,
                tile: Some((tile, tile)),
                overlap: None,
                async_build: true,
            })?;
            if model.state() == ModelState::Building {
                progress::engine_build(&name, tile);
                while model.state() == ModelState::Building {
                    std::thread::sleep(Duration::from_millis(250));
                }
                if model.state() == ModelState::Ready {
                    progress::engine_ready(&name);
                }
            }
            if model.state() == ModelState::Failed {
                return Err(format!("{name}: {}", self.ctx.last_error()));
            }
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

/// The devices of one backend, as JSON for the GUI's device list: `[{"id", "name", "vram"}]`
/// (`id` as a decimal string — it's a full 64-bit value). An unloadable backend is an empty list.
pub fn devices_json(aji_dir: &Path, backend: Backend) -> Result<String, String> {
    let aji = Aji::load(aji_dir)?;
    let devices = aji.devices(backend).unwrap_or_default();
    let list: Vec<_> = devices
        .iter()
        .map(|d| serde_json::json!({ "id": d.id.to_string(), "name": d.name, "vram": d.vram_bytes }))
        .collect();
    serde_json::to_string(&list).map_err(|e| e.to_string())
}

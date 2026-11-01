//! The inference side: one libaji context on the chosen backend/device, the models loaded so far
//! (by file and tile), and the upscale itself — including recovering alpha for transparent images.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
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
    /// TensorRT for RTX (opt-in, experimental; needs aji_trtrtx.dll beside aji.dll), falling
    /// back to TensorRT when it can't start, and per model when one fails to build.
    TensorRtRtx,
}

impl BackendChoice {
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

/// The subfolder of a models folder that holds bf16 variants of fp32 models, under the same file
/// names. MangaJaNai GUI ships them there so its model list (top-level *.onnx only) never shows
/// them: DirectML can't run bf16, nor can Turing and older.
const BF16_DIR: &str = "bf16";

/// Whether bf16 variants are preferred: on TensorRT or TensorRT-RTX with CUDA compute capability
/// 8.0 or newer (Ampere and newer, data-center parts included). Turing (7.5) and older have no
/// bf16, and DirectML can't run it. bf16 runs the fp32-only IllustrationJaNai DAT2/HAT_L models
/// ~40% faster there, visually identical (fp16 overflows on them).
fn prefers_bf16(backend: Backend, compute_capability: Option<(i32, i32)>) -> bool {
    backend.is_cuda() && compute_capability.is_some_and(|(major, _)| major >= 8)
}

/// Where a chain's model may have a bf16 variant, in lookup order: `<dir>/bf16/<model>` for each
/// models folder, in `resolve_in`'s order. Only for a bare file name: a path the user picked is
/// used as is.
fn bf16_candidates(model: &str, dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut parts = Path::new(model).components();
    if !matches!((parts.next(), parts.next()), (Some(Component::Normal(_)), None)) {
        return Vec::new();
    }
    dirs.iter().map(|d| d.join(BF16_DIR).join(model)).collect()
}

/// A chain's model file: as given if it's a path that exists, else looked up by name in the models
/// folders (the user's first, then the bundled ones). With `bf16`, also its bf16 variant if one is
/// there (the first [`bf16_candidates`] entry that exists); a bf16 file without the plain one
/// isn't used.
fn resolve_in(model: &str, dirs: &[PathBuf], bf16: bool) -> Option<ModelFile> {
    let p = Path::new(model);
    let path = if p.is_absolute() && p.is_file() {
        p.to_path_buf()
    } else {
        dirs.iter().map(|d| d.join(model)).find(|p| p.is_file())?
    };
    let bf16 = if bf16 { bf16_candidates(model, dirs).into_iter().find(|p| p.is_file()) } else { None };
    Some(ModelFile { path, bf16 })
}

/// A chain's model, resolved: the plain file, and the bf16 variant to try first, if any.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelFile {
    pub path: PathBuf,
    pub bf16: Option<PathBuf>,
}

pub struct Upscaler {
    aji: Aji,
    device_id: u64,
    cache_dir: Option<PathBuf>,
    ctx: Context,
    backend: Backend,
    /// Whether `resolve` looks for bf16 variants ([`prefers_bf16`]).
    prefer_bf16: bool,
    /// The context bf16 variants load on, opened on first need. With a `--cache-dir` it caches
    /// engines in its bf16 subfolder: aji names a cached engine after the model's file name, which
    /// the two variants share. Without one, engines sit beside each model, so `ctx` serves.
    bf16_ctx: Option<Result<Context, String>>,
    /// On TensorRT-RTX: the TensorRT context a failed model falls back to, opened on first need.
    fallback_ctx: Option<Result<Context, String>>,
    model_dirs: Vec<PathBuf>,
    /// By file (both variants, so a model named by path and the same one by name never mix) and
    /// tile: whichever variant loaded.
    models: HashMap<(ModelFile, u32), Model>,
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
            BackendChoice::TensorRtRtx => &[Backend::TensorRtRtx, Backend::TensorRt],
        };
        let mut err = String::from("no compatible GPU found");
        for &backend in candidates {
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
                    return Ok(Self {
                        aji,
                        device_id: device.id,
                        cache_dir,
                        ctx,
                        backend,
                        prefer_bf16: prefers_bf16(backend, device.compute_capability),
                        bf16_ctx: None,
                        fallback_ctx: None,
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

    /// A chain's model file (see [`resolve_in`]), with its bf16 variant when this backend and GPU
    /// prefer one.
    pub fn resolve(&self, model: &str) -> Option<ModelFile> {
        resolve_in(model, &self.model_dirs, self.prefer_bf16)
    }

    /// The loaded model for `file` at `tile`, loading it (and waiting out a first-use engine
    /// build, reported on stdout) if needed.
    fn model(&mut self, file: &ModelFile, tile: u32) -> Result<&Model, String> {
        let key = (file.clone(), tile);
        if !self.models.contains_key(&key) {
            let model = self.load(file, tile)?;
            self.models.insert(key.clone(), model);
        }
        Ok(&self.models[&key])
    }

    /// Load `file` at `tile`, trying in this order and stopping at the first that works:
    ///
    /// 1. its bf16 variant (if any) on this backend;
    /// 2. the plain file on this backend;
    /// 3. on TensorRT-RTX, the plain file on TensorRT (that context opens the first time it's
    ///    needed), as before bf16 variants existed.
    ///
    /// A bf16 failure is most likely the variant's own (an export the backend can't build), so the
    /// plain file is the next try, on the same backend. Only if that fails too does TensorRT-RTX
    /// fall back, and then with the plain file, the one the GUI lists: a bf16 retry on TensorRT
    /// would add another, possibly minutes-long, build to a rare double failure.
    fn load(&mut self, file: &ModelFile, tile: u32) -> Result<Model, String> {
        let name = file.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(bf16) = &file.bf16 {
            match self.bf16_context().and_then(|ctx| load_model(&ctx, self.backend, bf16, tile, &name, true)) {
                Ok(m) => {
                    progress::log(&format!("{name}: using the bf16 variant ({})", bf16.display()));
                    return Ok(m);
                }
                Err(e) => progress::log(&format!("bf16 variant of {name} failed ({e}); using the fp32 model")),
            }
        }
        match load_model(&self.ctx, self.backend, &file.path, tile, &name, false) {
            Ok(m) => Ok(m),
            Err(e) if self.backend == Backend::TensorRtRtx => {
                progress::backend_fallback(Some(&name), &e);
                let ctx = self.fallback_context()?;
                load_model(&ctx, Backend::TensorRt, &file.path, tile, &name, false).map_err(|e| format!("{name}: {e}"))
            }
            Err(e) => Err(format!("{name}: {e}")),
        }
    }

    fn bf16_context(&mut self) -> Result<Context, String> {
        let Some(dir) = &self.cache_dir else { return Ok(self.ctx.clone()) };
        let (aji, backend, id, dir) = (&self.aji, self.backend, self.device_id, Some(dir.join(BF16_DIR)));
        self.bf16_ctx
            .get_or_insert_with(|| open_context(aji, backend, id, &dir))
            .clone()
            .map_err(|e| format!("bf16 context: {e}"))
    }

    /// The TensorRT context a TensorRT-RTX upscaler falls back to.
    fn fallback_context(&mut self) -> Result<Context, String> {
        let (aji, id, cache) = (&self.aji, self.device_id, &self.cache_dir);
        self.fallback_ctx
            .get_or_insert_with(|| open_context(aji, Backend::TensorRt, id, cache))
            .clone()
            .map_err(|e| format!("TensorRT fallback: {e}"))
    }

    /// Upscale `w`x`h` pixels with `channels` 1 (gray) or 3 (RGB) through the model. Returns the
    /// output and its scale.
    pub fn upscale(&mut self, file: &ModelFile, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let model = self.model(file, tile)?;
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
    pub fn upscale_with_alpha(&mut self, file: &ModelFile, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
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
        let (black, scale) = self.upscale(file, tile, w, h, cc as u8, &on_black)?;
        let (white, _) = self.upscale(file, tile, w, h, cc as u8, &on_white)?;
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

/// Load `path` at `tile` on `ctx`, waiting out a background engine build (reported on stdout;
/// `bf16` marks the bf16 variant there).
fn load_model(ctx: &Context, backend: Backend, path: &Path, tile: u32, name: &str, bf16: bool) -> Result<Model, String> {
    let model = ctx.load_model(&LoadParams { onnx: path, tile: Some((tile, tile)), overlap: None, async_build: true })?;
    if model.state() == ModelState::Building {
        progress::engine_build(name, tile, backend.label(), bf16);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_only_on_cuda_backends_from_ampere() {
        for cc in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)] {
            assert!(prefers_bf16(Backend::TensorRt, Some(cc)), "{cc:?}");
            assert!(prefers_bf16(Backend::TensorRtRtx, Some(cc)), "{cc:?}");
        }
        for cc in [(7, 5), (7, 0), (6, 1)] {
            assert!(!prefers_bf16(Backend::TensorRt, Some(cc)), "{cc:?}");
            assert!(!prefers_bf16(Backend::TensorRtRtx, Some(cc)), "{cc:?}");
        }
        assert!(!prefers_bf16(Backend::TensorRt, None));
        assert!(!prefers_bf16(Backend::DirectMl, None));
        assert!(!prefers_bf16(Backend::DirectMl, Some((12, 0))));
    }

    #[test]
    fn bf16_candidates_follow_the_models_folders_for_bare_names_only() {
        let dirs = [PathBuf::from("user"), PathBuf::from("bundled")];
        assert_eq!(
            bf16_candidates("4x_a.onnx", &dirs),
            [Path::new("user").join("bf16").join("4x_a.onnx"), Path::new("bundled").join("bf16").join("4x_a.onnx")]
        );
        let abs = std::env::temp_dir().join("4x_a.onnx");
        for model in [abs.to_str().unwrap(), "sub/4x_a.onnx", "../4x_a.onnx", "", "."] {
            assert!(bf16_candidates(model, &dirs).is_empty(), "{model}");
        }
    }

    #[test]
    fn resolve_finds_the_bf16_variant_only_when_preferred() {
        let root = std::env::temp_dir().join(format!("mangajanai-cli-bf16-test-{}", std::process::id()));
        let (user, bundled) = (root.join("user"), root.join("bundled"));
        for d in [user.join("bf16"), bundled.join("bf16")] {
            std::fs::create_dir_all(d).unwrap();
        }
        let touch = |p: PathBuf| {
            std::fs::write(&p, b"").unwrap();
            p
        };
        let a = touch(bundled.join("a.onnx"));
        let a_bf16 = touch(bundled.join("bf16").join("a.onnx"));
        let b = touch(user.join("b.onnx"));
        let b_bf16 = touch(user.join("bf16").join("b.onnx"));
        touch(bundled.join("bf16").join("b.onnx"));
        let c = touch(user.join("c.onnx"));
        touch(bundled.join("bf16").join("only_bf16.onnx"));
        let dirs = [user.clone(), bundled.clone()];

        assert_eq!(resolve_in("a.onnx", &dirs, true), Some(ModelFile { path: a.clone(), bf16: Some(a_bf16) }));
        assert_eq!(resolve_in("a.onnx", &dirs, false), Some(ModelFile { path: a.clone(), bf16: None }));
        // The first models folder that has one wins, as for the plain file.
        assert_eq!(resolve_in("b.onnx", &dirs, true), Some(ModelFile { path: b, bf16: Some(b_bf16) }));
        assert_eq!(resolve_in("c.onnx", &dirs, true), Some(ModelFile { path: c, bf16: None }));
        // A path the user picked is used as is.
        assert_eq!(resolve_in(a.to_str().unwrap(), &dirs, true), Some(ModelFile { path: a, bf16: None }));
        // No plain file, no model.
        assert_eq!(resolve_in("only_bf16.onnx", &dirs, true), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}

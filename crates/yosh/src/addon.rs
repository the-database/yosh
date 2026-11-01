//! The AI-upscaling add-on (see `yosh_engine::sr`): where it lives, whether it's
//! installed, and starting it.
//!
//! The add-on is a directory the app never ships: libaji (`aji.dll` with its
//! DirectML/TensorRT (and optionally TensorRT-RTX) backends and their runtime libraries), a `models/` folder of
//! ONNX models, optionally a `rules.json` overriding which model handles which
//! page, and a `cache/` the backend fills (TensorRT engines). Until it exists the
//! setting is inert and decoding is exactly as without the feature.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use yosh_engine::pool::Waker;
use yosh_engine::sr::Upscaler;
use yosh_engine::sr::rules::{SrRule, default_rules};

use crate::config::UpscaleBackend;

/// What the Settings panel shows for the add-on.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Status {
    /// Setting off (nothing loaded).
    #[default]
    Off,
    NotInstalled,
    Starting,
    /// Running on the named device.
    Ready(String),
    Error(String),
}

/// A started upscaler.
pub struct Running {
    pub upscaler: Arc<dyn Upscaler>,
    /// What it runs on ("<GPU> · <backend>").
    pub device: String,
    /// TensorRT-RTX was tried first (chosen, or Auto's pick) but couldn't start:
    /// why, and the backend running instead ("TensorRT", or "DirectML" under Auto).
    pub fallback: Option<(String, &'static str)>,
}

pub type Started = Result<Running, String>;

/// Whether the add-on's library is present in `dir`.
pub fn installed(dir: &Path) -> bool {
    dir.join(if cfg!(windows) { "aji.dll" } else { "libaji.so" }).is_file()
}

/// The rule table: `rules.json` from the add-on if present and valid, else the
/// built-in MangaJaNaiConverterGui defaults. A `rules.json` naming older model
/// files keeps working: a model found under its exact name is used as is, else the
/// same model under another precision suffix, or without one (`sr::aji`'s lookup
/// order), so older unsuffixed names also find the `_fp16` files.
pub fn rules(dir: &Path) -> Vec<SrRule> {
    #[derive(serde::Deserialize)]
    struct Rule {
        #[serde(default)]
        min_h: u32,
        #[serde(default)]
        max_h: u32,
        #[serde(default)]
        gray: bool,
        #[serde(default)]
        color: bool,
        #[serde(default)]
        min_scale: f32,
        #[serde(default)]
        max_scale: f32,
        model: Option<String>,
    }
    let path = dir.join("rules.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return default_rules();
    };
    match serde_json::from_slice::<Vec<Rule>>(&bytes) {
        Ok(rs) => rs
            .into_iter()
            .map(|r| SrRule {
                min_h: r.min_h,
                max_h: r.max_h,
                gray: r.gray,
                color: r.color,
                min_scale: r.min_scale,
                max_scale: r.max_scale,
                model: r.model,
            })
            .collect(),
        Err(e) => {
            eprintln!("yosh: ignoring {}: {e}", path.display());
            default_rules()
        }
    }
}

/// Start the upscaler on a background thread (loading the runtime and opening the
/// device take a moment); the result arrives on the returned channel, after which
/// `wake` is called (send-then-wake).
pub fn start(dir: PathBuf, backend: UpscaleBackend, wake: Waker) -> Receiver<Started> {
    let (tx, rx) = std::sync::mpsc::channel();
    let wake2 = wake.clone();
    std::thread::spawn(move || {
        let _ = tx.send(start_blocking(&dir, backend, wake2));
        wake();
    });
    rx
}

/// Whether Auto prefers TensorRT-RTX on a GPU of this CUDA compute capability:
/// the consumer GPUs TensorRT-RTX supports (7.5 RTX 20 / GTX 16, 8.6 RTX 30,
/// 8.9 RTX 40, 12.x RTX 50 and newer). On Turing (7.5) TensorRT-RTX has no FP32
/// GEMMs or 3D convolutions, but none of the models the apps ship by default
/// hit those gaps: MangaJaNai ESRGAN is fp16 convolutions only, IllustrationJaNai
/// FDAT M/XL and SPAN use fp16 MatMuls. Only the optional fp32 IllustrationJaNai
/// DAT2/HAT_L exports use FP32 GEMMs, and a model that fails on TensorRT-RTX
/// falls back to TensorRT on its own (`AjiUpscaler`). The data-center parts
/// (8.0, 9.0, 10.0) stay excluded: TensorRT-RTX 1.7 doesn't support them. The
/// explicit choice runs it anywhere.
#[cfg(feature = "sr")]
fn auto_prefers_tensorrt_rtx((major, minor): (i32, i32)) -> bool {
    matches!((major, minor), (7, 5) | (8, 6) | (8, 9)) || major >= 12
}

/// The backends to try, in order, for `choice`. `rtx_cc` is the compute
/// capability of TensorRT-RTX's first device when the add-on has TensorRT-RTX
/// (only probed for Auto on Windows).
///
/// Auto prefers TensorRT-RTX on a GPU it suits (faster engine builds and pages
/// than TensorRT), then TensorRT — on an NVIDIA GPU with the TensorRT runtime
/// installed it is far faster than DirectML (transformer models especially), at
/// the cost of a one-time engine build per model — then DirectML, which runs on
/// any DirectX 12 GPU. Off Windows there is only TensorRT. The TensorRT-RTX
/// choice falls back to TensorRT when it can't start; on TensorRT-RTX a model
/// that fails to build falls back to TensorRT on its own (`AjiUpscaler`).
#[cfg(feature = "sr")]
fn candidates(
    choice: UpscaleBackend,
    windows: bool,
    rtx_cc: Option<(i32, i32)>,
) -> &'static [yosh_engine::sr::aji::Backend] {
    use yosh_engine::sr::aji::Backend;
    match choice {
        UpscaleBackend::TensorRt => &[Backend::TensorRt],
        UpscaleBackend::TensorRtRtx => &[Backend::TensorRtRtx, Backend::TensorRt],
        UpscaleBackend::DirectMl => &[Backend::DirectMl],
        UpscaleBackend::Auto if !windows => &[Backend::TensorRt],
        UpscaleBackend::Auto if rtx_cc.is_some_and(auto_prefers_tensorrt_rtx) => {
            &[Backend::TensorRtRtx, Backend::TensorRt, Backend::DirectMl]
        }
        UpscaleBackend::Auto => &[Backend::TensorRt, Backend::DirectMl],
    }
}

#[cfg(feature = "sr")]
fn start_blocking(dir: &Path, backend: UpscaleBackend, wake: Waker) -> Started {
    use yosh_engine::sr::aji::{AjiUpscaler, Backend, enum_devices};
    // Auto looks at TensorRT-RTX only when the add-on has it. A device list that
    // fails or comes back empty just leaves it out: no fallback to report.
    let mut rtx_device = (backend == UpscaleBackend::Auto
        && cfg!(windows)
        && dir.join("aji_trtrtx.dll").is_file())
    .then(|| enum_devices(dir, Backend::TensorRtRtx).ok()?.into_iter().next())
    .flatten();
    let candidates = candidates(backend, cfg!(windows), rtx_device.as_ref().and_then(|d| d.compute_capability));
    let mut err = String::from("no compatible GPU");
    // Why TensorRT-RTX, listed first, didn't start: whatever starts after it is
    // the fallback (kept apart from `err`, which a later candidate overwrites).
    let mut rtx_failed: Option<String> = None;
    for &b in candidates {
        if candidates[0] == Backend::TensorRtRtx && b != Backend::TensorRtRtx {
            rtx_failed.get_or_insert_with(|| err.clone());
        }
        let listed = match rtx_device.take().filter(|_| b == Backend::TensorRtRtx) {
            Some(d) => Ok(Some(d)),
            None => enum_devices(dir, b).map(|d| d.into_iter().next()),
        };
        let device = match listed {
            Ok(Some(d)) => d,
            Ok(None) => continue,
            Err(e) => {
                err = e;
                continue;
            }
        };
        match AjiUpscaler::new(dir, b, &device, rules(dir), Some(wake.clone())) {
            Ok(up) => {
                if let Some(why) = &rtx_failed {
                    eprintln!("yosh: [upscale] TensorRT-RTX failed ({why}); using {}", b.label());
                }
                let device = format!("{} · {}", device.name, b.label());
                let fallback = rtx_failed.map(|why| (why, b.label()));
                return Ok(Running { upscaler: Arc::new(up), device, fallback });
            }
            Err(e) => err = e,
        }
    }
    Err(err)
}

#[cfg(not(feature = "sr"))]
fn start_blocking(_dir: &Path, _backend: UpscaleBackend, _wake: Waker) -> Started {
    Err("this build has no upscaling support".into())
}

/// Open the add-on folder in the file browser, creating it first so there is
/// somewhere to put the files.
pub fn open_folder(dir: &Path) {
    let _ = std::fs::create_dir_all(dir.join("models"));
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer").arg(dir).spawn();
    #[cfg(not(windows))]
    let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
}

#[cfg(all(test, feature = "sr"))]
mod tests {
    use super::*;
    use yosh_engine::sr::aji::Backend::{DirectMl, TensorRt, TensorRtRtx};

    #[test]
    fn auto_prefers_tensorrt_rtx_on_supported_consumer_gpus() {
        for cc in [(7, 5), (8, 6), (8, 9), (12, 0), (12, 1), (13, 0)] {
            assert!(auto_prefers_tensorrt_rtx(cc), "{cc:?}");
        }
        // Pre-Turing, and the data-center parts TensorRT-RTX doesn't support.
        for cc in [(7, 0), (8, 0), (8, 7), (9, 0), (10, 0), (11, 0), (6, 1)] {
            assert!(!auto_prefers_tensorrt_rtx(cc), "{cc:?}");
        }
    }

    #[test]
    fn auto_tries_tensorrt_rtx_first_only_when_installed_on_a_suited_gpu() {
        let auto = UpscaleBackend::Auto;
        assert_eq!(candidates(auto, true, Some((12, 0))), [TensorRtRtx, TensorRt, DirectMl]);
        assert_eq!(candidates(auto, true, Some((8, 6))), [TensorRtRtx, TensorRt, DirectMl]);
        assert_eq!(candidates(auto, true, Some((7, 5))), [TensorRtRtx, TensorRt, DirectMl]);
        assert_eq!(candidates(auto, true, Some((6, 1))), [TensorRt, DirectMl]);
        assert_eq!(candidates(auto, true, Some((9, 0))), [TensorRt, DirectMl]);
        // Not installed (or no device listed).
        assert_eq!(candidates(auto, true, None), [TensorRt, DirectMl]);
        // Off Windows there is only TensorRT.
        assert_eq!(candidates(auto, false, Some((12, 0))), [TensorRt]);
        assert_eq!(candidates(auto, false, None), [TensorRt]);
    }

    #[test]
    fn explicit_choices_ignore_the_gpu() {
        for (windows, cc) in [(true, None), (true, Some((7, 5))), (true, Some((12, 0))), (false, None)] {
            assert_eq!(candidates(UpscaleBackend::TensorRt, windows, cc), [TensorRt]);
            assert_eq!(candidates(UpscaleBackend::DirectMl, windows, cc), [DirectMl]);
            assert_eq!(candidates(UpscaleBackend::TensorRtRtx, windows, cc), [TensorRtRtx, TensorRt]);
        }
    }
}

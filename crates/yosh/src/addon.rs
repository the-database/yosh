//! The AI-upscaling add-on (see `yosh_engine::sr`): where it lives, whether it's
//! installed, and starting it.
//!
//! The add-on is a directory the app never ships: libaji (`aji.dll` with its
//! DirectML/TensorRT backends and their runtime libraries), a `models/` folder of
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

/// A started upscaler and what it runs on ("<GPU> · <backend>").
pub type Started = Result<(Arc<dyn Upscaler>, String), String>;

/// Whether the add-on's library is present in `dir`.
pub fn installed(dir: &Path) -> bool {
    dir.join(if cfg!(windows) { "aji.dll" } else { "libaji.so" }).is_file()
}

/// The rule table: `rules.json` from the add-on if present and valid, else the
/// built-in MangaJaNaiConverterGui defaults.
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

#[cfg(feature = "sr")]
fn start_blocking(dir: &Path, backend: UpscaleBackend, wake: Waker) -> Started {
    use yosh_engine::sr::aji::{AjiUpscaler, Backend, enum_devices};
    // Auto prefers TensorRT — on an NVIDIA GPU with the TensorRT runtime installed
    // it is far faster (transformer models especially), at the cost of a one-time
    // engine build per model — and falls back to DirectML, which runs on any
    // DirectX 12 GPU. Off Windows there is only TensorRT.
    let candidates: &[Backend] = match backend {
        UpscaleBackend::TensorRt => &[Backend::TensorRt],
        UpscaleBackend::DirectMl => &[Backend::DirectMl],
        UpscaleBackend::Auto if cfg!(windows) => &[Backend::TensorRt, Backend::DirectMl],
        UpscaleBackend::Auto => &[Backend::TensorRt],
    };
    let mut err = String::from("no compatible GPU");
    for &b in candidates {
        let device = match enum_devices(dir, b).map(|d| d.into_iter().next()) {
            Ok(Some(d)) => d,
            Ok(None) => continue,
            Err(e) => {
                err = e;
                continue;
            }
        };
        match AjiUpscaler::new(dir, b, device.id, rules(dir), Some(wake.clone())) {
            Ok(up) => {
                let name = if b == Backend::TensorRt { "TensorRT" } else { "DirectML" };
                return Ok((Arc::new(up), format!("{} · {name}", device.name)));
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

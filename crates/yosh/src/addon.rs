//! The AI-upscaling add-on (see `yosh_engine::sr`): where it lives, whether it's
//! installed, and starting it.
//!
//! The add-on is a directory the app never ships: libaji (`aji.dll` with its
//! DirectML/TensorRT (and optionally TensorRT-RTX) backends and their runtime libraries), a `models/` folder of
//! ONNX models, optionally a `rules.json` with the custom upscaling profiles
//! (`profiles.rs`), and a `cache/` the backend fills (TensorRT engines). Until it exists the
//! setting is inert and decoding is exactly as without the feature.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use yosh_engine::pool::Waker;
use yosh_engine::sr::Upscaler;
use yosh_engine::sr::native::{NativeReport, NativeUse};
use yosh_engine::sr::rules::SrRule;

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
    /// The shared pack store was moved to this folder, which is missing now (an unplugged
    /// drive): not started (`addon_store::missing_folder`).
    StoreMissing(String),
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

/// What Settings and the AI chip say about libaji's native GPU kernels ([`native_text`]).
#[derive(Debug, Default, PartialEq)]
pub struct NativeText {
    /// After the backend on the status line ("native kernels", "... for 1 of 2 models",
    /// "... not used", "... off"); empty on DirectML or before a model loaded.
    pub summary: String,
    /// One line per model loaded, for the status line's hover.
    pub models: Vec<String>,
    /// The models that don't run on them, and why, shown under the status line.
    pub not_used: Vec<String>,
}

/// [`NativeText`] for an upscaler's [`Upscaler::native_kernels`] report (`None`: DirectML).
pub fn native_text(t: &yosh_engine::i18n::Strings, report: Option<&NativeReport>) -> NativeText {
    let Some(r) = report else { return NativeText::default() };
    if r.off {
        return NativeText { summary: t.upscale_native_off.to_string(), ..Default::default() };
    }
    let (native, total) = (r.native(), r.models.len());
    let summary = match native {
        _ if total == 0 => String::new(),
        n if n == total => t.upscale_native_on.to_string(),
        0 => t.upscale_native_none.to_string(),
        n => (t.upscale_native_some)(n, total),
    };
    let mut text = NativeText { summary, ..Default::default() };
    for (model, use_) in &r.models {
        match use_ {
            NativeUse::Native(what) => text.models.push((t.upscale_native_model)(model, what)),
            NativeUse::NotUsed(why) => {
                let line = (t.upscale_native_not_used)(model, why);
                text.models.push(line.clone());
                text.not_used.push(line);
            }
        }
    }
    text
}

/// Where the add-on's models are: its `models/` (the user's own models go there), then, when
/// setup put the add-on's packs in the shared pack store, the model pack folders the add-on's
/// `aji-runtime.json` lists (its `models` key; MangaJaNaiCLI `setup --store`).
pub fn models_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![dir.join("models")];
    let json = std::fs::read(dir.join("aji-runtime.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    if let Some(list) = json.as_ref().and_then(|j| j.get("models")).and_then(|m| m.as_array()) {
        dirs.extend(list.iter().filter_map(|v| v.as_str()).map(PathBuf::from));
    }
    dirs
}

/// [`set_native_kernels`] last turned them off (`AJI_NATIVE=off`).
static SET_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// libaji's native GPU kernels on or off for the models loaded from now on: libaji reads
/// `AJI_NATIVE` on each model load and runs TensorRT engines for every model when it says
/// `off`. Called before each upscaler start. With them on, only an `AJI_NATIVE` this
/// function set is removed, so one the user set themselves still applies.
pub fn set_native_kernels(on: bool) {
    use std::sync::atomic::Ordering;
    if on {
        if SET_OFF.swap(false, Ordering::SeqCst) {
            set_env("AJI_NATIVE", None);
        }
    } else {
        SET_OFF.store(true, Ordering::SeqCst);
        set_env("AJI_NATIVE", Some("off"));
    }
}

/// Set or remove a variable where libaji's `getenv` sees it.
fn set_env(name: &str, value: Option<&str>) {
    // SAFETY: called on the UI thread before the upscaler thread starts loading models,
    // which is where libaji reads the variable.
    unsafe {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
    // Windows: libaji reads it with getenv from ucrtbase.dll, whose copy of the environment is
    // taken once and isn't updated by SetEnvironmentVariableW (what set_var calls). yosh's
    // release build links its own C runtime statically (+crt-static), so its own _putenv_s
    // wouldn't reach that copy either: call ucrtbase's. An empty value removes the variable.
    #[cfg(windows)]
    {
        use std::ffi::{CString, c_char, c_int};
        type PutenvS = unsafe extern "C" fn(*const c_char, *const c_char) -> c_int;
        let (Ok(n), Ok(v)) = (CString::new(name), CString::new(value.unwrap_or(""))) else {
            return;
        };
        if let Some(f) = ucrt_proc(c"_putenv_s") {
            // SAFETY: ucrtbase's _putenv_s has this signature (cdecl is the x64/arm64 ABI);
            // both strings are NUL-terminated and outlive the call, and the CRT copies them.
            unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, PutenvS>(f)(n.as_ptr(), v.as_ptr()) };
        }
    }
}

/// A function exported by ucrtbase.dll, the C runtime libaji uses (loaded and kept loaded).
#[cfg(windows)]
fn ucrt_proc(name: &std::ffi::CStr) -> Option<unsafe extern "system" fn() -> isize> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let dll: Vec<u16> = "ucrtbase.dll\0".encode_utf16().collect();
    // SAFETY: a NUL-terminated name of a system DLL.
    let h = unsafe { LoadLibraryW(dll.as_ptr()) };
    if h.is_null() {
        return None;
    }
    // SAFETY: a live module handle and a NUL-terminated export name.
    unsafe { GetProcAddress(h, name.as_ptr().cast()) }
}

/// Whether the add-on's library is present in `dir`.
pub fn installed(dir: &Path) -> bool {
    dir.join(if cfg!(windows) { "aji.dll" } else { "libaji.so" }).is_file()
}

/// Start the upscaler on a background thread (loading the runtime and opening the
/// device take a moment); the result arrives on the returned channel, after which
/// `wake` is called (send-then-wake).
/// `rules` are the active profile's ([`crate::profiles`]).
pub fn start(dir: PathBuf, backend: UpscaleBackend, rules: Vec<SrRule>, wake: Waker) -> Receiver<Started> {
    let (tx, rx) = std::sync::mpsc::channel();
    let wake2 = wake.clone();
    std::thread::spawn(move || {
        let _ = tx.send(start_blocking(&dir, backend, rules, wake2));
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
fn start_blocking(dir: &Path, backend: UpscaleBackend, rules: Vec<SrRule>, wake: Waker) -> Started {
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
        match AjiUpscaler::new(dir, b, &device, rules.clone(), Some(wake.clone())) {
            Ok(up) => {
                let native = !SET_OFF.load(std::sync::atomic::Ordering::SeqCst);
                let up = up.with_model_dirs(models_dirs(dir).into_iter().skip(1)).with_native_kernels(native);
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
fn start_blocking(_dir: &Path, _backend: UpscaleBackend, _rules: Vec<SrRule>, _wake: Waker) -> Started {
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
    fn models_dirs_add_the_store_packs_from_aji_runtime_json() {
        let dir = std::env::temp_dir().join(format!("yosh-addon-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(models_dirs(&dir), vec![dir.join("models")]);
        std::fs::write(dir.join("aji-runtime.json"), br#"{"format":1,"trt":"t","models":["/store/m1","/store/m2"]}"#).unwrap();
        assert_eq!(models_dirs(&dir), vec![dir.join("models"), PathBuf::from("/store/m1"), PathBuf::from("/store/m2")]);
        std::fs::write(dir.join("aji-runtime.json"), b"not json").unwrap();
        assert_eq!(models_dirs(&dir), vec![dir.join("models")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

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

#[cfg(test)]
mod native_text_tests {
    use super::{NativeText, native_text};
    use yosh_engine::sr::native::{NativeReport, NativeUse};

    fn report(models: &[(&str, Option<&str>, &str)]) -> NativeReport {
        let models = models
            .iter()
            .map(|&(m, native, why)| {
                (m.to_string(), native.map_or_else(|| NativeUse::NotUsed(why.into()), |w| NativeUse::Native(w.into())))
            })
            .collect();
        NativeReport { off: false, models }
    }

    #[test]
    fn says_whether_the_models_run_on_native_kernels() {
        let t = yosh_engine::i18n::tr();
        // DirectML, and before a model loaded: nothing.
        assert_eq!(native_text(t, None), NativeText::default());
        assert_eq!(native_text(t, Some(&report(&[]))), NativeText::default());
        let off = native_text(t, Some(&NativeReport { off: true, models: Vec::new() }));
        assert_eq!((off.summary.as_str(), off.models.len(), off.not_used.len()), ("native kernels off", 0, 0));

        let all = native_text(t, Some(&report(&[("a", Some("ESRGAN 2x"), ""), ("b", Some("ESRGAN 4x"), "")])));
        assert_eq!(all.summary, "native kernels");
        assert_eq!(all.models, ["a: native kernels (ESRGAN 2x)", "b: native kernels (ESRGAN 4x)"]);
        assert!(all.not_used.is_empty());

        let some = native_text(t, Some(&report(&[("a", Some("ESRGAN 2x"), ""), ("b", None, "not 3-channel")])));
        assert_eq!(some.summary, "native kernels for 1 of 2 models");
        assert_eq!(some.models, ["a: native kernels (ESRGAN 2x)", "Native kernels not used for b: not 3-channel"]);
        assert_eq!(some.not_used, ["Native kernels not used for b: not 3-channel"]);

        let none = native_text(t, Some(&report(&[("b", None, "needs an RTX 30 or newer")])));
        assert_eq!(none.summary, "native kernels not used");
        assert_eq!(none.not_used, ["Native kernels not used for b: needs an RTX 30 or newer"]);
    }
}

#[cfg(test)]
mod native_kernels_tests {
    use super::set_native_kernels;

    /// What libaji's getenv sees: ucrtbase.dll's environment on Windows (yosh's release build
    /// links its own C runtime statically), the C library's elsewhere.
    fn c_getenv(name: &str) -> Option<String> {
        use std::ffi::{CStr, CString, c_char};
        type Getenv = unsafe extern "C" fn(*const c_char) -> *const c_char;
        #[cfg(windows)]
        let getenv: Getenv = {
            let f = super::ucrt_proc(c"getenv").expect("ucrtbase getenv");
            // SAFETY: ucrtbase's getenv has this signature.
            unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, Getenv>(f) }
        };
        #[cfg(not(windows))]
        let getenv: Getenv = {
            unsafe extern "C" {
                fn getenv(name: *const c_char) -> *const c_char;
            }
            getenv
        };
        let n = CString::new(name).unwrap();
        // SAFETY: a NUL-terminated name; the result is copied before the environment changes.
        let v = unsafe { getenv(n.as_ptr()) };
        (!v.is_null()).then(|| unsafe { CStr::from_ptr(v) }.to_string_lossy().into_owned())
    }

    #[test]
    fn native_off_reaches_libajis_c_runtime_and_on_removes_only_ours() {
        set_native_kernels(true);
        assert_eq!(c_getenv("AJI_NATIVE"), None);
        set_native_kernels(false);
        assert_eq!(c_getenv("AJI_NATIVE").as_deref(), Some("off"));
        set_native_kernels(true);
        assert_eq!(c_getenv("AJI_NATIVE"), None);
    }
}

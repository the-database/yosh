//! Whether libaji ran the upscaler's models on its native GPU kernels (libaji 0.29.0+ on the
//! TensorRT backends; DirectML has none), for the shell to show.
//!
//! libaji's image path says so through the context's log callback, at info level, on every model
//! load: `native kernels: <model> (<what>)` when the model runs on them, and `native kernels not
//! used for <model>: <why>; using TensorRT` (or TensorRT-RTX) when it builds an engine instead. A
//! model it doesn't consider at all (1-channel, a fixed input size, or `AJI_NATIVE=off`) gets no
//! line; its native probe then says why ([`probe_reason`]). `<model>` is the ONNX file's stem.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// How one model runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeUse {
    /// On native kernels; libaji's description of the build (family, scale, tile).
    Native(String),
    /// On a TensorRT engine, and why.
    NotUsed(String),
}

/// What [`super::Upscaler::native_kernels`] reports.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeReport {
    /// The setting is off: every model runs on TensorRT engines (nothing per model then).
    pub off: bool,
    /// The models loaded so far, by file stem, in name order; the latest load of each.
    pub models: Vec<(String, NativeUse)>,
}

impl NativeReport {
    /// How many of [`NativeReport::models`] run on native kernels.
    pub fn native(&self) -> usize {
        self.models.iter().filter(|(_, u)| matches!(u, NativeUse::Native(_))).count()
    }
}

/// Whether a log line is one of libaji's native-kernel lines (the only info lines kept).
pub fn is_native_line(msg: &str) -> bool {
    msg.starts_with("native kernels: ") || msg.starts_with("native kernels not used for ")
}

/// What libaji's line `msg` says about the model of file stem `model`; `None` for any other line.
pub fn parse(msg: &str, model: &str) -> Option<NativeUse> {
    if let Some(what) = msg
        .strip_prefix("native kernels: ")
        .and_then(|r| r.strip_prefix(model))
        .and_then(|r| r.strip_prefix(" ("))
        .and_then(|r| r.strip_suffix(')'))
    {
        return Some(NativeUse::Native(what.to_string()));
    }
    let rest = msg.strip_prefix("native kernels not used for ")?.strip_prefix(model)?.strip_prefix(": ")?;
    // "...; using TensorRT" / "; using TensorRT-RTX": the backend is shown elsewhere.
    let why = rest.rsplit_once("; using ").map_or(rest, |(why, _)| why);
    Some(NativeUse::NotUsed(why.to_string()))
}

/// Why a model that got no native-kernel line at its load doesn't run on them, from the error of
/// its native probe: `<model>: not for native kernels (<why>)` or `<model>: native kernels not
/// used: <why>`. `None` when the error isn't about this model (a libaji without the probe).
pub fn probe_reason(err: &str, model: &str) -> Option<String> {
    let rest = err.strip_prefix(model)?.strip_prefix(": ")?;
    let why = rest
        .strip_prefix("not for native kernels (")
        .and_then(|r| r.strip_suffix(')'))
        .or_else(|| rest.strip_prefix("native kernels not used: "))
        .unwrap_or(rest);
    Some(why.to_string())
}

/// The latest [`NativeUse`] of each model loaded, by file stem; shared by the upscaler's
/// contexts (a TensorRT-RTX model that falls back to TensorRT loads again there).
#[derive(Default)]
pub struct NativeLog {
    models: Mutex<BTreeMap<String, NativeUse>>,
}

impl NativeLog {
    pub fn set(&self, model: &str, use_: NativeUse) {
        self.models.lock().unwrap_or_else(|e| e.into_inner()).insert(model.to_string(), use_);
    }

    pub fn report(&self, off: bool) -> NativeReport {
        let models = if off {
            Vec::new()
        } else {
            self.models.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        NativeReport { off, models }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16";

    #[test]
    fn parses_libajis_native_lines() {
        let on = format!("native kernels: {M} (ESRGAN 2x, 512x512 tile, 64 MB)");
        assert!(is_native_line(&on));
        assert_eq!(parse(&on, M), Some(NativeUse::Native("ESRGAN 2x, 512x512 tile, 64 MB".into())));
        for backend in ["TensorRT", "TensorRT-RTX"] {
            let off = format!("native kernels not used for {M}: needs compute capability 8.0 (this GPU is 7.5); using {backend}");
            assert!(is_native_line(&off));
            assert_eq!(parse(&off, M), Some(NativeUse::NotUsed("needs compute capability 8.0 (this GPU is 7.5)".into())));
        }
        // A reason with "; " of its own keeps it.
        let off = format!("native kernels not used for {M}: a; b; using TensorRT");
        assert_eq!(parse(&off, M), Some(NativeUse::NotUsed("a; b".into())));
    }

    #[test]
    fn lines_about_other_models_or_other_things_are_not_this_models() {
        let on = format!("native kernels: {M} (ESRGAN 2x)");
        assert_eq!(parse(&on, "4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16"), None);
        // A name that only starts like this one's.
        assert_eq!(parse(&format!("native kernels: {M}_v2 (ESRGAN 2x)"), M), None);
        for msg in ["Loaded engine size: 32 MiB", "native kernels failed: out of memory", ""] {
            assert!(!is_native_line(msg), "{msg}");
            assert_eq!(parse(msg, M), None, "{msg}");
        }
    }

    #[test]
    fn probe_errors_give_the_reason() {
        assert_eq!(probe_reason(&format!("{M}: not for native kernels (not 3-channel)"), M).as_deref(), Some("not 3-channel"));
        assert_eq!(probe_reason(&format!("{M}: not for native kernels (fixed input size)"), M).as_deref(), Some("fixed input size"));
        assert_eq!(probe_reason(&format!("{M}: native kernels not used: unsupported op Gemm"), M).as_deref(), Some("unsupported op Gemm"));
        assert_eq!(probe_reason("aji_img_load: no model path", M), None);
        assert_eq!(probe_reason("", M), None);
    }

    #[test]
    fn the_log_keeps_each_models_latest_load_and_reports_off_as_off() {
        let log = NativeLog::default();
        assert_eq!(log.report(false), NativeReport::default());
        log.set("b", NativeUse::NotUsed("tile size not accepted".into()));
        log.set("a", NativeUse::Native("ESRGAN 2x".into()));
        let r = log.report(false);
        assert_eq!(r.native(), 1);
        assert_eq!(r.models.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        // A fallback's load of the same model replaces the first.
        log.set("b", NativeUse::Native("ESRGAN 2x".into()));
        assert_eq!(log.report(false).native(), 2);
        assert_eq!(log.report(true), NativeReport { off: true, models: Vec::new() });
    }
}

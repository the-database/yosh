//! Model selection: an ordered rule list, first match wins — the same shape as
//! MangaJaNaiConverterGui's workflow "chains", so a model pack's rule table means
//! the same thing in both apps. Only the fields a reader needs are modelled:
//! source-height range, gray/color, and target-scale range.

/// One rule. A bound of 0 means "unbounded" (MangaJaNaiConverterGui's convention).
#[derive(Clone, Debug, PartialEq)]
pub struct SrRule {
    /// Source height range, inclusive.
    pub min_h: u32,
    pub max_h: u32,
    /// Which pages the rule applies to (visually gray vs color).
    pub gray: bool,
    pub color: bool,
    /// Target scale (displayed / source height) range, inclusive.
    pub min_scale: f32,
    pub max_scale: f32,
    /// Model file stem (`<models dir>/<model>.onnx`), or `None` for "don't
    /// upscale pages matching this rule".
    pub model: Option<String>,
}

impl SrRule {
    fn matches(&self, src_h: u32, gray: bool, scale: f32) -> bool {
        (self.min_h == 0 || src_h >= self.min_h)
            && (self.max_h == 0 || src_h <= self.max_h)
            && (if gray { self.gray } else { self.color })
            && (self.min_scale == 0.0 || scale >= self.min_scale)
            && (self.max_scale == 0.0 || scale <= self.max_scale)
    }
}

/// The model for a page, if any: the first matching rule's model.
pub fn select(rules: &[SrRule], src_h: u32, gray: bool, scale: f32) -> Option<&str> {
    rules
        .iter()
        .find(|r| r.matches(src_h, gray, scale))
        .and_then(|r| r.model.as_deref())
}

/// MangaJaNaiConverterGui's default workflow: IllustrationJaNai for color pages;
/// for gray pages the MangaJaNai model trained nearest the source height. Within
/// each group the 2x model comes first, so a target of exactly 2x picks it, and
/// anything above 2x takes the 4x model. Model stems are the ONNX exports of the
/// same checkpoints, named after their ONNX precision as MangaJaNai GUI 2.0 ships
/// them (`_fp16`). `sr::aji` also finds an add-on's other precisions of a model and
/// the older unsuffixed MangaJaNai names.
pub fn default_rules() -> Vec<SrRule> {
    let mut rules = vec![
        color(0.0, 2.0, "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16"),
        color(2.0, 0.0, "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16"),
    ];
    // (min_h, max_h, bucket, 2x checkpoint, 4x checkpoint)
    const GRAY: [(u32, u32, &str, &str, &str); 7] = [
        (0, 1250, "1200p", "70k", "70k"),
        (1251, 1350, "1300p", "75k", "75k"),
        (1351, 1450, "1400p", "70k", "105k"),
        (1451, 1550, "1500p", "90k", "105k"),
        (1551, 1760, "1600p", "90k", "70k"),
        (1761, 1984, "1920p", "70k", "105k"),
        (1985, 0, "2048p", "95k", "70k"),
    ];
    for (lo, hi, bucket, k2, k4) in GRAY {
        rules.push(gray(lo, hi, 0.0, 2.0, format!("2x_MangaJaNai_{bucket}_V1_ESRGAN_{k2}_fp16")));
        rules.push(gray(lo, hi, 2.0, 0.0, format!("4x_MangaJaNai_{bucket}_V1_ESRGAN_{k4}_fp16")));
    }
    rules
}

/// The "Quality" profile: the default's gray rules, and IllustrationJaNai DAT2 on
/// colour pages at every scale (only a 4x export exists). The DAT2/HAT pack is
/// bf16 on NVIDIA compute capability 8.0+ and fp32 elsewhere; `sr::aji`'s lookup
/// finds the `_fp32` file from this `_bf16` name.
pub fn quality_rules() -> Vec<SrRule> {
    let mut rules = vec![color(0.0, 0.0, QUALITY_COLOR)];
    rules.extend(default_rules().into_iter().filter(|r| r.gray));
    rules
}

/// The colour model of [`quality_rules`].
pub const QUALITY_COLOR: &str = "4x_IllustrationJaNai_V3denoise_DAT2_27k_bf16";

/// The "Performance" profile: the 2x models only, at every scale — the 2x
/// MangaJaNai of the page's height bucket on gray pages and IllustrationJaNai
/// SPAN on colour pages. Above 2x the GPU magnifies the rest (the one allowed
/// GPU resample).
pub fn performance_rules() -> Vec<SrRule> {
    let mut rules = vec![color(0.0, 0.0, "2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16")];
    rules.extend(default_rules().into_iter().filter(|r| r.gray && r.max_scale == 2.0).map(|mut r| {
        r.max_scale = 0.0;
        r
    }));
    rules
}

/// A model name without its ONNX precision suffix (`_fp16`, `_bf16`, `_fp32`):
/// the name the precision variants of one model share. `sr::aji`'s lookup tries
/// every precision for an unsuffixed name, so a rule may name either.
pub fn model_base(name: &str) -> &str {
    ["_fp16", "_bf16", "_fp32"]
        .iter()
        .find_map(|s| name.strip_suffix(s))
        .unwrap_or(name)
}

fn color(min_scale: f32, max_scale: f32, model: &str) -> SrRule {
    SrRule {
        min_h: 0,
        max_h: 0,
        gray: false,
        color: true,
        min_scale,
        max_scale,
        model: Some(model.to_string()),
    }
}

fn gray(min_h: u32, max_h: u32, min_scale: f32, max_scale: f32, model: String) -> SrRule {
    SrRule { min_h, max_h, gray: true, color: false, min_scale, max_scale, model: Some(model) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rules_pick_like_mangajanai_converter() {
        let r = default_rules();
        // Gray: the bucket follows the source height, the factor the target scale.
        assert_eq!(select(&r, 1200, true, 1.8), Some("2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16"));
        assert_eq!(select(&r, 1250, true, 3.0), Some("4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16"));
        assert_eq!(select(&r, 1251, true, 1.5), Some("2x_MangaJaNai_1300p_V1_ESRGAN_75k_fp16"));
        assert_eq!(select(&r, 1700, true, 1.27), Some("2x_MangaJaNai_1600p_V1_ESRGAN_90k_fp16"));
        assert_eq!(select(&r, 1984, true, 2.5), Some("4x_MangaJaNai_1920p_V1_ESRGAN_105k_fp16"));
        assert_eq!(select(&r, 3000, true, 1.1), Some("2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16"));
        // Exactly 2x matches both rules of a group; the 2x one comes first.
        assert_eq!(select(&r, 1400, true, 2.0), Some("2x_MangaJaNai_1400p_V1_ESRGAN_70k_fp16"));
        // Color ignores height.
        assert_eq!(
            select(&r, 800, false, 2.7),
            Some("4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16")
        );
        assert_eq!(
            select(&r, 2400, false, 1.2),
            Some("2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16")
        );
    }

    #[test]
    fn default_models_are_the_shipped_fp16_files() {
        let r = default_rules();
        assert_eq!(r.len(), 16);
        let names: Vec<&str> = r.iter().filter_map(|r| r.model.as_deref()).collect();
        assert_eq!(names.iter().filter(|n| n.contains("_MangaJaNai_")).count(), 14);
        assert!(names.iter().all(|n| n.ends_with("_fp16")), "{names:?}");
    }

    #[test]
    fn quality_and_performance_profiles() {
        let q = quality_rules();
        assert_eq!(select(&q, 1500, false, 1.3), Some(QUALITY_COLOR));
        assert_eq!(select(&q, 1500, false, 3.0), Some(QUALITY_COLOR));
        assert_eq!(select(&q, 1500, true, 1.3), Some("2x_MangaJaNai_1500p_V1_ESRGAN_90k_fp16"));
        assert_eq!(select(&q, 1500, true, 3.0), Some("4x_MangaJaNai_1500p_V1_ESRGAN_105k_fp16"));
        let p = performance_rules();
        assert_eq!(p.len(), 8);
        assert_eq!(select(&p, 900, false, 3.5), Some("2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16"));
        assert_eq!(select(&p, 1500, true, 1.3), Some("2x_MangaJaNai_1500p_V1_ESRGAN_90k_fp16"));
        assert_eq!(select(&p, 1500, true, 3.5), Some("2x_MangaJaNai_1500p_V1_ESRGAN_90k_fp16"));
        assert_eq!(select(&p, 3000, true, 4.0), Some("2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16"));
    }

    #[test]
    fn model_base_strips_only_a_precision_suffix() {
        assert_eq!(model_base("4x_X_DAT2_27k_bf16"), "4x_X_DAT2_27k");
        assert_eq!(model_base("2x_X_fp16_fp32"), "2x_X_fp16");
        assert_eq!(model_base("2x_X_fp32_op17"), "2x_X_fp32_op17");
        assert_eq!(model_base("custom"), "custom");
    }

    #[test]
    fn first_match_wins_and_none_means_skip() {
        let rules = vec![
            SrRule { min_h: 0, max_h: 1000, gray: true, color: true, min_scale: 0.0, max_scale: 0.0, model: None },
            SrRule { min_h: 0, max_h: 0, gray: true, color: true, min_scale: 0.0, max_scale: 0.0, model: Some("any".into()) },
        ];
        assert_eq!(select(&rules, 900, true, 2.0), None, "an explicit no-model rule wins");
        assert_eq!(select(&rules, 1001, false, 2.0), Some("any"));
        assert_eq!(select(&[], 1001, false, 2.0), None);
    }
}

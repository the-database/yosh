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
/// same checkpoints.
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
        rules.push(gray(lo, hi, 0.0, 2.0, format!("2x_MangaJaNai_{bucket}_V1_ESRGAN_{k2}")));
        rules.push(gray(lo, hi, 2.0, 0.0, format!("4x_MangaJaNai_{bucket}_V1_ESRGAN_{k4}")));
    }
    rules
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
        assert_eq!(select(&r, 1200, true, 1.8), Some("2x_MangaJaNai_1200p_V1_ESRGAN_70k"));
        assert_eq!(select(&r, 1250, true, 3.0), Some("4x_MangaJaNai_1200p_V1_ESRGAN_70k"));
        assert_eq!(select(&r, 1251, true, 1.5), Some("2x_MangaJaNai_1300p_V1_ESRGAN_75k"));
        assert_eq!(select(&r, 1700, true, 1.27), Some("2x_MangaJaNai_1600p_V1_ESRGAN_90k"));
        assert_eq!(select(&r, 1984, true, 2.5), Some("4x_MangaJaNai_1920p_V1_ESRGAN_105k"));
        assert_eq!(select(&r, 3000, true, 1.1), Some("2x_MangaJaNai_2048p_V1_ESRGAN_95k"));
        // Exactly 2x matches both rules of a group; the 2x one comes first.
        assert_eq!(select(&r, 1400, true, 2.0), Some("2x_MangaJaNai_1400p_V1_ESRGAN_70k"));
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

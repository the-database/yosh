//! Import a MangaJaNai GUI workflow export (`.mwf`) as a profile's rules.
//!
//! The reading is MangaJaNaiCLI's (`src/settings.rs` `Workflow`/`Chain`, `src/chain.rs`
//! `parse_resolution`/`activates`), trimmed to what picks a model: a `.mwf` is one
//! Newtonsoft-serialized workflow whose `Chains` is `{"$values": [...]}`; a chain's
//! `MinResolution`/`MaxResolution` are `"WxH"` and its scale factors whole numbers,
//! 0 meaning unbounded, and the first chain that matches an image is used — the
//! same as [`SrRule`]. A chain's "Auto Adjust Levels on Grayscale Images" carries
//! over ([`SrRule::auto_levels`]). What yosh has no equivalent for — width limits,
//! resizing before the model, a tile size other than the 512 px it always uses,
//! a grayscale detection threshold other than its own — is reported back
//! ([`Import::ignored`], [`Import::gray_threshold`]) instead of silently dropped.

use std::path::Path;

use serde::{Deserialize, Deserializer};
use yosh_engine::decode::GRAYSCALE_THRESHOLD;
use yosh_engine::sr::rules::SrRule;

/// Newtonsoft writes an unset string as `null`; read it as empty.
fn null_as_empty<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

#[derive(Deserialize, Default)]
struct Values<T> {
    #[serde(rename = "$values", default = "Vec::new")]
    values: Vec<T>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "PascalCase", default)]
struct Workflow {
    #[serde(deserialize_with = "null_as_empty")]
    workflow_name: String,
    /// The workflow's gray detection threshold (the GUI's default is 12).
    grayscale_detection_threshold: Option<i32>,
    chains: Values<Chain>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", default)]
struct Chain {
    #[serde(deserialize_with = "null_as_empty")]
    min_resolution: String,
    #[serde(deserialize_with = "null_as_empty")]
    max_resolution: String,
    is_grayscale: bool,
    is_color: bool,
    min_scale_factor: Option<i32>,
    max_scale_factor: Option<i32>,
    #[serde(deserialize_with = "null_as_empty")]
    model_file_path: String,
    /// "Auto (Estimate)", "Maximum", "No Tiling" or a number of pixels.
    #[serde(deserialize_with = "null_as_empty")]
    model_tile_size: String,
    auto_adjust_levels: bool,
    resize_height_before_upscale: Option<i32>,
    resize_width_before_upscale: Option<i32>,
    resize_factor_before_upscale: Option<f64>,
}

impl Default for Chain {
    fn default() -> Self {
        Self {
            min_resolution: "0x0".into(),
            max_resolution: "0x0".into(),
            is_grayscale: false,
            is_color: false,
            min_scale_factor: Some(0),
            max_scale_factor: Some(0),
            model_file_path: String::new(),
            model_tile_size: "Auto (Estimate)".into(),
            auto_adjust_levels: false,
            resize_height_before_upscale: Some(0),
            resize_width_before_upscale: Some(0),
            resize_factor_before_upscale: Some(100.0),
        }
    }
}

/// `"WxH"` → (w, h); 0 means unbounded. Unparseable parts count as 0.
fn parse_resolution(s: &str) -> (u32, u32) {
    let mut it = s.split('x').map(|p| p.trim().parse::<u32>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

/// The tile size libaji runs every model of yosh's at (its default).
const YOSH_TILE: u32 = 512;

/// The tile (input pixels) a chain's "Model Tile Size" runs at in MangaJaNaiCLI
/// (`chain::tile_size`): "Auto (Estimate)" and anything unrecognized 512,
/// "Maximum"/"No Tiling" 1024, a number snapped to a multiple of 16.
fn tile_size(setting: &str) -> u32 {
    match setting.trim() {
        "Maximum" | "No Tiling" => 1024,
        s => match s.parse::<u32>() {
            Ok(n) => (n / 16 * 16).clamp(64, 4096),
            Err(_) => YOSH_TILE,
        },
    }
}

/// What yosh can't carry over from a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ignored {
    /// A minimum or maximum width (yosh's rules match on height only).
    Width,
    /// "Resize before upscale" (yosh always gives the model the whole page).
    PreResize,
    /// A model tile size other than 512 px, as MangaJaNaiCLI would run it (yosh
    /// runs every model at 512).
    TileSize(u32),
}

/// An imported workflow.
#[derive(Debug, PartialEq)]
pub struct Import {
    /// The workflow's name (else the file's).
    pub name: String,
    pub rules: Vec<SrRule>,
    /// `(chain number, what)`, 1-based like the GUI's chain list.
    pub ignored: Vec<(usize, Ignored)>,
    /// The workflow's grayscale detection threshold, when it isn't yosh's own
    /// ([`GRAYSCALE_THRESHOLD`], the GUI's default): not carried over.
    pub gray_threshold: Option<i32>,
}

/// The model a chain names, as a rule's model: the file stem (the GUI stores
/// a file name in its models folder; a `.pth` of GUI 1.x names the same model's
/// ONNX export), or `None` for "No Model".
fn model(path: &str) -> Option<String> {
    let p = path.trim();
    if p.is_empty() || p.eq_ignore_ascii_case("No Model") {
        return None;
    }
    let file = p.rsplit(['/', '\\']).next().unwrap_or(p);
    let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or(file);
    Some(stem.to_string())
}

/// Read `.mwf` text; `fallback_name` names the profile if the workflow has none.
pub fn parse(text: &str, fallback_name: &str) -> Result<Import, String> {
    let wf: Workflow = serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    let mut ignored = Vec::new();
    let rules = wf
        .chains
        .values
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (min_w, min_h) = parse_resolution(&c.min_resolution);
            let (max_w, max_h) = parse_resolution(&c.max_resolution);
            if min_w != 0 || max_w != 0 {
                ignored.push((i + 1, Ignored::Width));
            }
            let pre = c.resize_height_before_upscale.unwrap_or(0) > 0
                || c.resize_width_before_upscale.unwrap_or(0) > 0
                || c.resize_factor_before_upscale.is_some_and(|f| f > 0.0 && f != 100.0);
            if pre {
                ignored.push((i + 1, Ignored::PreResize));
            }
            let tile = tile_size(&c.model_tile_size);
            // Only where a model runs: a "No Model" chain's tile means nothing.
            if tile != YOSH_TILE && model(&c.model_file_path).is_some() {
                ignored.push((i + 1, Ignored::TileSize(tile)));
            }
            SrRule {
                min_h,
                max_h,
                gray: c.is_grayscale,
                color: c.is_color,
                min_scale: c.min_scale_factor.unwrap_or(0).max(0) as f32,
                max_scale: c.max_scale_factor.unwrap_or(0).max(0) as f32,
                model: model(&c.model_file_path),
                auto_levels: c.auto_adjust_levels,
            }
        })
        .collect();
    let name = if wf.workflow_name.trim().is_empty() { fallback_name } else { wf.workflow_name.trim() };
    let gray_threshold = wf.grayscale_detection_threshold.filter(|&t| t != GRAYSCALE_THRESHOLD);
    Ok(Import { name: name.to_string(), rules, ignored, gray_threshold })
}

/// Read a `.mwf` file.
pub fn load(path: &Path) -> Result<Import, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    parse(&text, stem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yosh_engine::sr::rules::{default_rules, select};

    #[test]
    fn reads_a_gui_export() {
        let mwf = r#"{"$type":"W","WorkflowName":"Upscale Manga","ModeScaleSelected":true,
            "Chains":{"$type":"L","$values":[
              {"$type":"C","ChainNumber":"1","MinResolution":"0x0","MaxResolution":"0x1250","IsGrayscale":true,
               "IsColor":false,"MinScaleFactor":0,"MaxScaleFactor":2,
               "ModelFilePath":"2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx","ResizeFactorBeforeUpscale":100.0},
              {"$type":"C","MinResolution":"0x0","MaxResolution":"0x0","IsColor":true,"MinScaleFactor":2,
               "ModelFilePath":"C:\\models\\4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.pth"},
              {"$type":"C","IsGrayscale":true,"IsColor":true,"ModelFilePath":"No Model"},
              {"$type":"C","IsGrayscale":true,"ModelFilePath":null,"MinResolution":null}
            ]}}"#;
        let imp = parse(mwf, "file").unwrap();
        assert_eq!(imp.name, "Upscale Manga");
        assert!(imp.ignored.is_empty());
        assert_eq!(imp.gray_threshold, None);
        let r = &imp.rules;
        assert_eq!(r.len(), 4);
        assert_eq!((r[0].min_h, r[0].max_h, r[0].gray, r[0].color), (0, 1250, true, false));
        assert_eq!((r[0].min_scale, r[0].max_scale), (0.0, 2.0));
        assert_eq!(r[0].model.as_deref(), Some("2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16"));
        assert_eq!(r[1].min_scale, 2.0);
        assert_eq!(r[1].model.as_deref(), Some("4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16"));
        assert_eq!(r[2].model, None);
        assert_eq!(r[3].model, None);
        assert_eq!(r[3].min_h, 0, "null resolution reads as unbounded");
        // First match wins, as in the GUI.
        assert_eq!(select(r, 1100, true, 1.5), Some("2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16"));
        assert_eq!(select(r, 1100, false, 3.0), Some("4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16"));
        assert_eq!(select(r, 1400, true, 1.5), None, "the No Model chain");
    }

    #[test]
    fn reports_what_yosh_ignores_and_names_from_the_file() {
        let mwf = r#"{"Chains":{"$values":[
            {"MinResolution":"800x0","IsColor":true,"ModelFilePath":"a.onnx"},
            {"IsColor":true,"ResizeHeightBeforeUpscale":1000,"ModelFilePath":"b.onnx"},
            {"IsColor":true,"ResizeFactorBeforeUpscale":50.0,"MaxResolution":"0x2000","ModelFilePath":"c.onnx"}
        ]}}"#;
        let imp = parse(mwf, "mine").unwrap();
        assert_eq!(imp.name, "mine");
        assert_eq!(imp.ignored, vec![(1, Ignored::Width), (2, Ignored::PreResize), (3, Ignored::PreResize)]);
        assert_eq!(imp.rules[2].max_h, 2000);
    }

    #[test]
    fn auto_levels_carry_over_and_tile_size_and_threshold_are_reported() {
        let mwf = r#"{"GrayscaleDetectionThreshold":20,"Chains":{"$values":[
            {"IsGrayscale":true,"AutoAdjustLevels":true,"ModelTileSize":"Auto (Estimate)","ModelFilePath":"a.onnx"},
            {"IsGrayscale":true,"AutoAdjustLevels":false,"ModelTileSize":"256","ModelFilePath":"b.onnx"},
            {"IsColor":true,"ModelTileSize":"Maximum","ModelFilePath":"c.onnx"},
            {"IsColor":true,"ModelTileSize":"No Tiling","ModelFilePath":"No Model"},
            {"IsColor":true,"ModelTileSize":"520","ModelFilePath":"d.onnx"},
            {"IsColor":true,"ModelTileSize":null,"ModelFilePath":"e.onnx"}
        ]}}"#;
        let imp = parse(mwf, "x").unwrap();
        let levels: Vec<bool> = imp.rules.iter().map(|r| r.auto_levels).collect();
        assert_eq!(levels, [true, false, false, false, false, false]);
        // 520 snaps to 512, which yosh runs; a null size is "Auto (Estimate)".
        assert_eq!(imp.ignored, vec![(2, Ignored::TileSize(256)), (3, Ignored::TileSize(1024))]);
        assert_eq!(imp.gray_threshold, Some(20));
        let same = parse(r#"{"GrayscaleDetectionThreshold":12,"Chains":{"$values":[]}}"#, "x").unwrap();
        assert_eq!(same.gray_threshold, None);
    }

    #[test]
    fn the_default_workflow_imports_as_the_default_rules() {
        // MangaJaNai GUI's default chains (MangaJaNaiCLI's `default_chains`), as the GUI exports them.
        let mut chains = vec![
            ("0x0", "0x0", false, 0, 2, "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16".to_string()),
            ("0x0", "0x0", false, 2, 0, "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16".to_string()),
        ];
        for (lo, hi, b, k2, k4) in [
            ("0x0", "0x1250", "1200p", "70k", "70k"),
            ("0x1251", "0x1350", "1300p", "75k", "75k"),
            ("0x1351", "0x1450", "1400p", "70k", "105k"),
            ("0x1451", "0x1550", "1500p", "90k", "105k"),
            ("0x1551", "0x1760", "1600p", "90k", "70k"),
            ("0x1761", "0x1984", "1920p", "70k", "105k"),
            ("0x1985", "0x0", "2048p", "95k", "70k"),
        ] {
            chains.push((lo, hi, true, 0, 2, format!("2x_MangaJaNai_{b}_V1_ESRGAN_{k2}_fp16")));
            chains.push((lo, hi, true, 2, 0, format!("4x_MangaJaNai_{b}_V1_ESRGAN_{k4}_fp16")));
        }
        let values: Vec<String> = chains
            .iter()
            .map(|(lo, hi, g, mn, mx, m)| {
                format!(
                    r#"{{"MinResolution":"{lo}","MaxResolution":"{hi}","IsGrayscale":{g},"IsColor":{c},
                       "MinScaleFactor":{mn},"MaxScaleFactor":{mx},"ModelFilePath":"{m}.onnx",
                       "ModelTileSize":"Auto (Estimate)","AutoAdjustLevels":{g}}}"#,
                    c = !g
                )
            })
            .collect();
        let mwf = format!(r#"{{"GrayscaleDetectionThreshold":12,"Chains":{{"$values":[{}]}}}}"#, values.join(","));
        let imp = parse(&mwf, "x").unwrap();
        assert_eq!(imp.rules, default_rules());
        assert!(imp.ignored.is_empty() && imp.gray_threshold.is_none());
    }
}

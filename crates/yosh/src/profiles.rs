//! AI-upscaling profiles: which rule table picks the model for each page.
//!
//! Laid out like mpv-AnimeJaNai's slots: three built-in profiles (Quality,
//! Balanced, Performance — Shift+1/2/3) that can't be edited, and nine custom
//! slots (Ctrl+1..9) the profile editor fills; Shift/Ctrl+0 switches upscaling
//! off. Balanced is MangaJaNai GUI's default workflow, and the default profile.
//!
//! The custom slots live in the add-on folder's `rules.json`:
//! `{"profiles": [{"slot": 1, "name": "...", "rules": [rule, ...]}]}`, each rule
//! with the fields of [`SrRule`]. The file used to be a bare rule array that
//! replaced the built-in rules; such a file still loads, as custom slot 1
//! ([`Loaded::legacy`]: the app then makes slot 1 the default and rewrites the
//! file in the new form). Which profile yosh starts on is `Settings::upscale_profile`.

use std::path::{Path, PathBuf};

use yosh_engine::i18n::tr;
use yosh_engine::sr::rules::{
    QUALITY_COLOR, SrRule, default_rules, model_base, performance_rules, quality_rules,
};

/// How many custom slots there are (Ctrl+1..9).
pub const SLOTS: usize = 9;

/// A profile: one of the built-ins or a custom slot (1-based, as on the keys).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ProfileId {
    Quality,
    #[default]
    Balanced,
    Performance,
    Custom(u8),
}

impl ProfileId {
    pub const BUILTIN: [ProfileId; 3] = [ProfileId::Quality, ProfileId::Balanced, ProfileId::Performance];

    /// The token `Settings::upscale_profile` stores (not UI text).
    pub fn token(self) -> String {
        match self {
            ProfileId::Quality => "quality".into(),
            ProfileId::Balanced => "balanced".into(),
            ProfileId::Performance => "performance".into(),
            ProfileId::Custom(n) => format!("custom{n}"),
        }
    }

    /// The profile a stored token names; anything unknown is `None` (the caller
    /// falls back to Balanced), so a token from a newer yosh can't break this one.
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "quality" => Some(ProfileId::Quality),
            "balanced" => Some(ProfileId::Balanced),
            "performance" => Some(ProfileId::Performance),
            _ => {
                let n: u8 = s.strip_prefix("custom")?.parse().ok()?;
                (1..=SLOTS as u8).contains(&n).then_some(ProfileId::Custom(n))
            }
        }
    }

    /// Its key, for menus and help ("Shift+1", "Ctrl+4").
    pub fn key(self) -> String {
        match self {
            ProfileId::Quality => "Shift+1".into(),
            ProfileId::Balanced => "Shift+2".into(),
            ProfileId::Performance => "Shift+3".into(),
            ProfileId::Custom(n) => format!("Ctrl+{n}"),
        }
    }

    pub fn builtin(self) -> bool {
        !matches!(self, ProfileId::Custom(_))
    }
}

/// A filled custom slot.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Custom {
    /// Empty = "Custom <n>".
    pub name: String,
    pub rules: Vec<SrRule>,
}

/// The nine custom slots.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Profiles {
    pub slots: [Option<Custom>; SLOTS],
}

impl Profiles {
    pub fn slot(&self, n: u8) -> Option<&Custom> {
        self.slots.get((n as usize).wrapping_sub(1))?.as_ref()
    }

    /// Whether `id` can be used: a built-in, or a filled slot.
    pub fn exists(&self, id: ProfileId) -> bool {
        match id {
            ProfileId::Custom(n) => self.slot(n).is_some(),
            _ => true,
        }
    }

    /// Its name as shown.
    pub fn name(&self, id: ProfileId) -> String {
        let t = tr();
        match id {
            ProfileId::Quality => t.profile_quality.into(),
            ProfileId::Balanced => t.profile_balanced.into(),
            ProfileId::Performance => t.profile_performance.into(),
            ProfileId::Custom(n) => match self.slot(n) {
                Some(c) if !c.name.trim().is_empty() => c.name.trim().to_string(),
                _ => (t.profile_custom)(n),
            },
        }
    }

    /// The rules `id` runs with the models in `models_dirs`, or `None` for an
    /// empty slot. Quality uses Balanced's colour rules when the add-on has no
    /// DAT2 model (one made by hand, without the DAT2/HAT pack).
    pub fn rules(&self, id: ProfileId, models_dirs: &[PathBuf]) -> Option<Vec<SrRule>> {
        Some(match id {
            ProfileId::Quality if !model_installed(models_dirs, QUALITY_COLOR) => {
                let mut r = default_rules();
                r.retain(|r| !r.gray);
                r.extend(quality_rules().into_iter().filter(|r| r.gray));
                r
            }
            ProfileId::Quality => quality_rules(),
            ProfileId::Balanced => default_rules(),
            ProfileId::Performance => performance_rules(),
            ProfileId::Custom(n) => self.slot(n)?.rules.clone(),
        })
    }

    /// Built-ins first, then the filled slots: what a profile menu offers.
    pub fn usable(&self) -> Vec<ProfileId> {
        let custom = (1..=SLOTS as u8).map(ProfileId::Custom).filter(|&id| self.exists(id));
        ProfileId::BUILTIN.into_iter().chain(custom).collect()
    }
}

/// The built-in rules of `id` as the editor shows them (no fallback applied).
pub fn builtin_rules(id: ProfileId) -> Vec<SrRule> {
    match id {
        ProfileId::Quality => quality_rules(),
        ProfileId::Performance => performance_rules(),
        _ => default_rules(),
    }
}

// ---- rules.json ---------------------------------------------------------------------

#[derive(serde::Deserialize, serde::Serialize)]
struct RuleJson {
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
    #[serde(default)]
    model: Option<String>,
}

impl From<RuleJson> for SrRule {
    fn from(r: RuleJson) -> Self {
        SrRule {
            min_h: r.min_h,
            max_h: r.max_h,
            gray: r.gray,
            color: r.color,
            min_scale: r.min_scale,
            max_scale: r.max_scale,
            model: r.model,
        }
    }
}

impl From<&SrRule> for RuleJson {
    fn from(r: &SrRule) -> Self {
        RuleJson {
            min_h: r.min_h,
            max_h: r.max_h,
            gray: r.gray,
            color: r.color,
            min_scale: r.min_scale,
            max_scale: r.max_scale,
            model: r.model.clone(),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ProfileJson {
    slot: u8,
    #[serde(default)]
    name: String,
    #[serde(default)]
    rules: Vec<RuleJson>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct FileJson {
    #[serde(default)]
    profiles: Vec<ProfileJson>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum AnyJson {
    Profiles(FileJson),
    Legacy(Vec<RuleJson>),
}

/// What [`load`] found.
#[derive(Debug, Default)]
pub struct Loaded {
    pub profiles: Profiles,
    /// `rules.json` was the older bare rule array (now custom slot 1).
    pub legacy: bool,
}

fn rules_path(addon_dir: &Path) -> PathBuf {
    addon_dir.join("rules.json")
}

/// Read the custom slots from `rules.json` text. Errors name what's wrong.
pub fn parse(text: &str) -> Result<Loaded, String> {
    let text = text.trim_start_matches('\u{feff}');
    match serde_json::from_str::<AnyJson>(text).map_err(|e| e.to_string())? {
        AnyJson::Legacy(rules) => {
            let mut profiles = Profiles::default();
            profiles.slots[0] = Some(Custom {
                name: "rules.json".into(),
                rules: rules.into_iter().map(SrRule::from).collect(),
            });
            Ok(Loaded { profiles, legacy: true })
        }
        AnyJson::Profiles(f) => {
            let mut profiles = Profiles::default();
            for p in f.profiles {
                if (1..=SLOTS as u8).contains(&p.slot) {
                    profiles.slots[p.slot as usize - 1] = Some(Custom {
                        name: p.name,
                        rules: p.rules.into_iter().map(SrRule::from).collect(),
                    });
                }
            }
            Ok(Loaded { profiles, legacy: false })
        }
    }
}

/// The custom slots as `rules.json` text.
pub fn to_json(profiles: &Profiles) -> String {
    let f = FileJson {
        profiles: profiles
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let c = c.as_ref()?;
                Some(ProfileJson {
                    slot: i as u8 + 1,
                    name: c.name.clone(),
                    rules: c.rules.iter().map(RuleJson::from).collect(),
                })
            })
            .collect(),
    };
    serde_json::to_string_pretty(&f).unwrap_or_default()
}

/// The add-on's custom slots; none if there's no (readable) `rules.json`.
pub fn load(addon_dir: &Path) -> Loaded {
    let path = rules_path(addon_dir);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Loaded::default();
    };
    parse(&text).unwrap_or_else(|e| {
        eprintln!("yosh: ignoring {}: {e}", path.display());
        Loaded::default()
    })
}

/// Write the custom slots to `rules.json` (through a temporary file, so a crash
/// mid-write can't lose them).
pub fn save(addon_dir: &Path, profiles: &Profiles) -> Result<(), String> {
    std::fs::create_dir_all(addon_dir).map_err(|e| e.to_string())?;
    let path = rules_path(addon_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, to_json(profiles)).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

// ---- models -------------------------------------------------------------------------

/// Whether a rule's model `name` is in one of `models_dirs` under any precision
/// (the files `sr::aji`'s lookup would find for it).
pub fn model_installed(models_dirs: &[PathBuf], name: &str) -> bool {
    let base = model_base(name);
    std::iter::once(name.to_string())
        .chain(std::iter::once(base.to_string()))
        .chain(["_fp16", "_bf16", "_fp32"].iter().map(|s| format!("{base}{s}")))
        .any(|stem| models_dirs.iter().any(|d| d.join(format!("{stem}.onnx")).is_file()))
}

/// The models in `models_dirs`, one entry per model (its precision variants
/// share a name, [`model_base`]: the lookup picks the best one installed),
/// sorted.
pub fn installed_models(models_dirs: &[PathBuf]) -> Vec<String> {
    let mut names: Vec<String> = models_dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let ext = p.extension()?.to_str()?;
            if !ext.eq_ignore_ascii_case("onnx") {
                return None;
            }
            Some(model_base(p.file_stem()?.to_str()?).to_string())
        })
        .collect();
    names.sort_by_key(|n| n.to_lowercase());
    names.dedup();
    names
}

/// Copy a model file into `models_dir`; the name it's under (its stem).
pub fn add_model(models_dir: &Path, file: &Path) -> Result<String, String> {
    let name = file.file_name().ok_or("not a file")?;
    std::fs::create_dir_all(models_dir).map_err(|e| e.to_string())?;
    std::fs::copy(file, models_dir.join(name)).map_err(|e| e.to_string())?;
    Ok(Path::new(name).file_stem().and_then(|s| s.to_str()).map(model_base).unwrap_or_default().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_round_trip_and_unknown_is_none() {
        for id in [
            ProfileId::Quality,
            ProfileId::Balanced,
            ProfileId::Performance,
            ProfileId::Custom(1),
            ProfileId::Custom(9),
        ] {
            assert_eq!(ProfileId::from_token(&id.token()), Some(id));
        }
        assert_eq!(ProfileId::from_token("custom0"), None);
        assert_eq!(ProfileId::from_token("custom10"), None);
        assert_eq!(ProfileId::from_token("auto"), None);
    }

    #[test]
    fn legacy_rules_json_becomes_slot_1() {
        let l = parse(r#"[{"max_h":1250,"gray":true,"model":"a"},{"color":true,"model":null}]"#).unwrap();
        assert!(l.legacy);
        let c = l.profiles.slot(1).unwrap();
        assert_eq!(c.rules.len(), 2);
        assert_eq!(c.rules[0].max_h, 1250);
        assert_eq!(c.rules[0].model.as_deref(), Some("a"));
        assert_eq!(c.rules[1].model, None);
        assert!(l.profiles.slot(2).is_none());
    }

    #[test]
    fn profiles_round_trip_through_json() {
        let mut p = Profiles::default();
        p.slots[0] = Some(Custom { name: "Photos".into(), rules: default_rules() });
        p.slots[4] = Some(Custom {
            name: String::new(),
            rules: vec![SrRule {
                min_h: 0,
                max_h: 0,
                gray: true,
                color: true,
                min_scale: 0.0,
                max_scale: 0.0,
                model: Some("4x_general".into()),
            }],
        });
        let back = parse(&to_json(&p)).unwrap();
        assert!(!back.legacy);
        assert_eq!(back.profiles, p);
        assert!(back.profiles.exists(ProfileId::Custom(5)));
        assert!(!back.profiles.exists(ProfileId::Custom(2)));
        assert_eq!(
            back.profiles.usable(),
            vec![
                ProfileId::Quality,
                ProfileId::Balanced,
                ProfileId::Performance,
                ProfileId::Custom(1),
                ProfileId::Custom(5)
            ]
        );
    }

    #[test]
    fn slots_out_of_range_are_dropped() {
        let l = parse(r#"{"profiles":[{"slot":0,"rules":[]},{"slot":10,"rules":[]},{"slot":9,"rules":[]}]}"#).unwrap();
        assert_eq!(l.profiles.usable().len(), 4);
        assert!(l.profiles.exists(ProfileId::Custom(9)));
    }

    #[test]
    fn quality_falls_back_without_dat2_and_models_are_listed_once() {
        let dir = std::env::temp_dir().join(format!("yosh-profiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["b_fp16.onnx", "b_fp32.onnx", "a.onnx", "notes.txt"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let dirs = [dir.clone()];
        assert_eq!(installed_models(&dirs), vec!["a".to_string(), "b".to_string()]);
        assert!(model_installed(&dirs, "b_bf16"));
        assert!(model_installed(&dirs, "b"));
        assert!(!model_installed(&dirs, "c_fp16"));
        let p = Profiles::default();
        let q = p.rules(ProfileId::Quality, &dirs).unwrap();
        assert!(q.iter().all(|r| r.model.as_deref() != Some(QUALITY_COLOR)));
        assert_eq!(q.iter().filter(|r| r.color).count(), 2, "Balanced's colour rules");
        std::fs::write(dir.join(format!("{}.onnx", model_base(QUALITY_COLOR).to_owned() + "_fp32")), b"x").unwrap();
        assert_eq!(p.rules(ProfileId::Quality, &dirs).unwrap(), quality_rules());
        assert_eq!(p.rules(ProfileId::Custom(3), &dirs), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

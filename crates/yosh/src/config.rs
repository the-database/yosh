//! Persisted settings + per-volume last-read page (per-OS config dir via
//! `directories`, stored as JSON).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Cap on the most-recently-read list (`Settings::recents`). Bounds the persisted
/// JSON and the future recents shelf; the head is the resume target.
pub const RECENTS_CAP: usize = 32;

/// Persisted window geometry so size/position/maximized survive a restart.
/// Coordinates are physical pixels: `x`/`y` are the window's outer top-left
/// (incl. decorations), `w`/`h` are the inner (client-area) size. When
/// `maximized`, the geometry is the *restored* rect to return to on un-maximize.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct WindowState {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub maximized: bool,
}

/// Chrome theme preference. `System` follows the OS day/night setting (read via
/// `winit::window::Theme`); `Light`/`Dark` force it. `Light` is the e-ink-friendly
/// mode — a white page letterbox + egui's light visuals; the dark default suits a
/// backlit monitor. The Android shell carries the same three-way choice.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemePref {
    #[default]
    System,
    Light,
    Dark,
}

impl ThemePref {
    /// Resolve to dark-vs-light, consulting the cached OS night-mode flag for `System`.
    /// The Settings panel selects a variant directly, so no label/cycle helpers needed.
    pub fn is_dark(self, system_dark: bool) -> bool {
        match self {
            ThemePref::System => system_dark,
            ThemePref::Light => false,
            ThemePref::Dark => true,
        }
    }
}

/// Performance profile: how hard the reader is allowed to work this machine.
/// A single picker rather than individual knobs — the [`yosh_engine::reader::Budget`]
/// fields are interdependent (a wider prefetch window with a small cache just evicts
/// itself), so exposing them separately would only let a user build a worse
/// configuration. `Auto` (the default) follows the power source: the full desktop
/// budget on AC, the `Mid` tier on battery. The rest pin a tier, which is how a
/// machine can be held at full aggression while unplugged (`High`) or kept quiet
/// even on mains (`Low`/`Mid`). Same four-way choice — and the same wording in the
/// UI — as the Android shell's `PerfPref`.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "lowercase")]
pub enum PerfPref {
    #[default]
    Auto,
    Low,
    Mid,
    High,
}

impl PerfPref {
    /// The tier this pins, or `None` for `Auto` (resolve it from the power source).
    pub fn tier(self) -> Option<yosh_engine::reader::DeviceTier> {
        use yosh_engine::reader::DeviceTier;
        match self {
            PerfPref::Auto => None,
            PerfPref::Low => Some(DeviceTier::Low),
            PerfPref::Mid => Some(DeviceTier::Mid),
            PerfPref::High => Some(DeviceTier::High),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub direction_rtl: bool,
    /// 0 = fit window, 1 = fit width, 2 = fit height.
    pub fit: u8,
    pub layout_spread: bool,
    /// Continuous vertical-strip scroll mode (overrides page-flip layout).
    pub scroll: bool,
    /// Downscale pages on the GPU instead of on the CPU.
    pub gpu: bool,
    /// Last library root folder (browse grid).
    pub library_root: Option<String>,
    /// Most-recently-read volume paths, newest first, deduped, capped at
    /// `RECENTS_CAP`. `recents[0]` is the resume target on a no-arg launch (the
    /// first entry that still exists on disk); the same list backs the future
    /// "recently read" shelf. Same key form as `last_pages`.
    pub recents: Vec<String>,
    /// Resume the most recent volume on a no-arg launch (default on). Toggled from
    /// the top bar.
    pub resume_on_startup: bool,
    /// Volume path (folder or archive) → last-read page index.
    pub last_pages: HashMap<String, usize>,
    /// Volume path → read-tracking pair `(furthest_page_count, total_pages)`, where
    /// `furthest_page_count` is the 1-based count of the furthest page ever shown
    /// (the far page of a spread counts). Drives the library's read-state visuals:
    /// faded "finished" covers, the in-progress bar, and the per-series status. A
    /// volume with a `last_pages` entry but no `progress` entry counts as started.
    pub progress: HashMap<String, (usize, usize)>,
    /// Series folders (by path string) the user collapsed in the library. Absent ⇒
    /// expanded, so the default (everything expanded) stores nothing.
    pub collapsed: HashSet<String>,
    /// Volume path → spread pairing parity offset (0 or 1).
    pub spread_offsets: HashMap<String, u8>,
    /// Whether the keys overlay has been shown once (first-launch onboarding).
    pub help_seen: bool,
    /// Show the bottom seekbar (auto-hides; reveals when the cursor nears the
    /// bottom edge). Toggled with key `B`.
    pub seekbar_enabled: bool,
    /// Animate page flips with a quick slide + soft fade on the outgoing page
    /// (page-flip mode only). Toggled with key `T`. Off by default on desktop — a
    /// fast sweep across a large monitor is more fatiguing than on a phone (Android
    /// enables it unconditionally in its own shell).
    pub page_transition_enabled: bool,
    /// Book-gutter shading along the inner edges of an un-joined two-page spread.
    /// Settings-panel only (no key); off by default.
    pub spine_shadow_enabled: bool,
    /// Peak darkening at the seam, 0..1.
    pub spine_shadow_strength: f32,
    /// Chrome theme: System (follow OS) / Light (e-ink) / Dark. Cycled from the top bar.
    pub theme: ThemePref,
    /// Performance profile (Auto = full budget on AC, throttled on battery).
    pub perf: PerfPref,
    /// Last window geometry (size/position/maximized). None until first saved.
    pub window: Option<WindowState>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            direction_rtl: true, // manga default
            fit: 0,
            layout_spread: false,
            scroll: false,
            gpu: false,
            library_root: None,
            recents: Vec::new(),
            resume_on_startup: true,
            last_pages: HashMap::new(),
            progress: HashMap::new(),
            collapsed: HashSet::new(),
            spread_offsets: HashMap::new(),
            help_seen: false,
            seekbar_enabled: true,
            page_transition_enabled: false, // desktop default off; Android shell forces on
            spine_shadow_enabled: false,
            spine_shadow_strength: 0.35,
            // Dark by default on desktop (a backlit monitor); intentionally unlike the
            // Android shell, which defaults to System so e-ink panels get Light.
            theme: ThemePref::Dark,
            perf: PerfPref::Auto,
            window: None,
        }
    }
}

fn config_file() -> Option<PathBuf> {
    // Portable mode: a `yosh-portable.txt` marker next to the exe keeps config
    // beside the exe (travels with the app, leaves nothing in %APPDATA%). The
    // portable zip ships the marker; the installer does not.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("yosh-portable.txt").exists()
    {
        return Some(dir.join("yosh-state.json"));
    }
    directories::ProjectDirs::from("", "the-database", "yosh")
        .map(|d| d.config_dir().join("state.json"))
}

/// Directory for the cover-thumbnail cache. Portable mode keeps it beside the exe
/// (so it travels with the app and leaves nothing in the user profile); otherwise
/// it's the per-OS cache dir. `None` if neither can be resolved.
pub fn cache_dir() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("yosh-portable.txt").exists()
    {
        return Some(dir.join("yosh-thumbs"));
    }
    directories::ProjectDirs::from("", "the-database", "yosh")
        .map(|d| d.cache_dir().join("thumbs"))
}

pub fn load() -> Settings {
    config_file()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Persist the settings. One file holds every setting, reading position and
/// recent, so the write is atomic — a temp file plus a rename over the target
/// (same trick as `thumbcache::store`). A plain `fs::write` killed mid-flight
/// leaves truncated JSON, which `load()` silently discards for defaults; the
/// rename either lands whole or not at all. Both `config_file()` branches keep
/// the temp beside the target, so it's a same-volume atomic replace on Windows
/// and POSIX alike. Best-effort throughout: a failed write leaves the previous
/// state file intact.
pub fn save(settings: &Settings) {
    let Some(path) = config_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_vec_pretty(settings) {
        // The UI thread is the only writer, so a fixed temp name can't collide.
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state file written by an older build has no `perf` key at all. The
    /// container-level `#[serde(default)]` must fill it from `Settings::default()`
    /// rather than failing the whole parse — a failed parse is silently swallowed by
    /// `load()`, which would reset every setting, position and recent the user had.
    #[test]
    fn old_state_json_without_perf_still_loads() {
        let old = r#"{
            "direction_rtl": false,
            "fit": 2,
            "theme": "light",
            "recents": ["C:\\books\\vol1.cbz"],
            "help_seen": true
        }"#;
        let s: Settings = serde_json::from_str(old).expect("old state.json must still parse");
        assert_eq!(s.perf, PerfPref::Auto, "missing perf defaults to Auto");
        assert!(!s.spine_shadow_enabled, "missing spine shadow defaults to off");
        assert_eq!(s.spine_shadow_strength, 0.35);
        // The keys that *were* present survive, so this isn't a silent reset.
        assert!(!s.direction_rtl);
        assert_eq!(s.fit, 2);
        assert!(matches!(s.theme, ThemePref::Light));
        assert_eq!(s.recents.len(), 1);
    }

    /// The persisted tokens are the lowercase variant names, and they round-trip.
    #[test]
    fn perf_pref_round_trips_as_lowercase_tokens() {
        for (pref, token) in [
            (PerfPref::Auto, "\"auto\""),
            (PerfPref::Low, "\"low\""),
            (PerfPref::Mid, "\"mid\""),
            (PerfPref::High, "\"high\""),
        ] {
            assert_eq!(serde_json::to_string(&pref).unwrap(), token);
            assert_eq!(serde_json::from_str::<PerfPref>(token).unwrap(), pref);
        }
    }

    /// `Auto` defers to the power source; every other choice pins its tier.
    #[test]
    fn perf_pref_tier_mapping() {
        use yosh_engine::reader::DeviceTier;
        assert_eq!(PerfPref::Auto.tier(), None);
        assert_eq!(PerfPref::Low.tier(), Some(DeviceTier::Low));
        assert_eq!(PerfPref::Mid.tier(), Some(DeviceTier::Mid));
        assert_eq!(PerfPref::High.tier(), Some(DeviceTier::High));
    }
}

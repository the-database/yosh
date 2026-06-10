//! Persisted settings + per-volume last-read page (per-OS config dir via
//! `directories`, stored as JSON).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
            last_pages: HashMap::new(),
            progress: HashMap::new(),
            collapsed: HashSet::new(),
            spread_offsets: HashMap::new(),
            help_seen: false,
            seekbar_enabled: true,
            page_transition_enabled: false, // desktop default off; Android shell forces on
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

pub fn save(settings: &Settings) {
    let Some(path) = config_file() else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_vec_pretty(settings) {
        let _ = std::fs::write(path, json);
    }
}

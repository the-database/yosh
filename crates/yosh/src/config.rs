//! Persisted settings + per-volume last-read page (per-OS config dir via
//! `directories`, stored as JSON).

use std::collections::HashMap;
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
    /// Uncapped present (Immediate) instead of vsync (Fifo).
    pub turbo: bool,
    /// Downscale pages on the GPU instead of on the CPU.
    pub gpu: bool,
    /// Last library root folder (browse grid).
    pub library_root: Option<String>,
    /// Volume path (folder or archive) → last-read page index.
    pub last_pages: HashMap<String, usize>,
    /// Volume path → spread pairing parity offset (0 or 1).
    pub spread_offsets: HashMap<String, u8>,
    /// Whether the keys overlay has been shown once (first-launch onboarding).
    pub help_seen: bool,
    /// "Jump" seek mode (key J): skip ahead past not-yet-decoded pages for fast
    /// long-distance seeks. Default off = "step" (hold on each page; see them all).
    pub jump: bool,
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
            turbo: false,
            gpu: false,
            library_root: None,
            last_pages: HashMap::new(),
            spread_offsets: HashMap::new(),
            help_seen: false,
            jump: false,
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

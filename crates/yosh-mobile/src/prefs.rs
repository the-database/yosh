//! Viewing options the shell persists (layout, theme, volume keys, performance).

use super::*;

/// Spine-shadow peak darkening when the user hasn't picked one (matches desktop).
pub(crate) const DEFAULT_SPINE_STRENGTH: f32 = 0.55;

/// The user's page-layout *choice*. The engine's `Layout` is binary
/// (`Single`/`Spread`); this adds `Auto`, which the shell resolves to a concrete
/// `Layout` from the live viewport each frame: portrait → single, landscape →
/// two-page spread. So physically rotating the tablet switches modes — single while
/// upright, spread when turned sideways for a double-page.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutMode {
    Single,
    Spread,
    Auto,
}

impl LayoutMode {
    /// Resolve to a concrete engine `Layout` for a viewport of `w × h` px.
    /// Landscape is strictly `w > h`; portrait and square fall to single.
    pub(crate) fn resolve(self, w: u32, h: u32) -> Layout {
        match self {
            LayoutMode::Single => Layout::Single,
            LayoutMode::Spread => Layout::Spread,
            LayoutMode::Auto => {
                if w > h {
                    Layout::Spread
                } else {
                    Layout::Single
                }
            }
        }
    }

    /// Persistence / info-overlay token.
    pub(crate) fn label(self) -> &'static str {
        match self {
            LayoutMode::Single => "single",
            LayoutMode::Spread => "spread",
            LayoutMode::Auto => "auto",
        }
    }
}

/// Chrome theme preference. `System` follows the OS day/night setting (read via
/// JNI on Android — winit doesn't surface it); `Light`/`Dark` force it. Light is
/// the e-ink-friendly mode (white background, dark text, opaque popups); the dark
/// default is unusable on a reflective e-ink panel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThemePref {
    System,
    Light,
    Dark,
}

impl ThemePref {
    /// Resolve to dark-vs-light, consulting the cached system night-mode flag for `System`.
    pub(crate) fn is_dark(self, system_dark: bool) -> bool {
        match self {
            ThemePref::System => system_dark,
            ThemePref::Light => false,
            ThemePref::Dark => true,
        }
    }
    /// Persistence token (view.txt slot 7).
    pub(crate) fn label(self) -> &'static str {
        match self {
            ThemePref::System => "system",
            ThemePref::Light => "light",
            ThemePref::Dark => "dark",
        }
    }
    pub(crate) fn parse(tok: Option<&&str>) -> Self {
        match tok {
            Some(&"light") => ThemePref::Light,
            Some(&"dark") => ThemePref::Dark,
            _ => ThemePref::System,
        }
    }
}

/// Whether (and which way) the volume rocker turns pages while a book is open.
/// Off by default: it takes the buttons away from the volume. Both directions are
/// offered because "down = forward" and "up = forward" each feel natural to
/// someone (and depend on which way the device is held).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VolKeys {
    Off,
    /// Volume Down = next page, Volume Up = previous.
    DownNext,
    /// Volume Up = next page, Volume Down = previous.
    UpNext,
}

impl VolKeys {
    /// Persistence token (view.txt slot 12).
    pub(crate) fn label(self) -> &'static str {
        match self {
            VolKeys::Off => "novolkeys",
            VolKeys::DownNext => "volkeys",
            VolKeys::UpNext => "volkeysup",
        }
    }
    pub(crate) fn parse(tok: Option<&&str>) -> Self {
        match tok {
            Some(&"volkeys") => VolKeys::DownNext,
            Some(&"volkeysup") => VolKeys::UpNext,
            _ => VolKeys::Off,
        }
    }
}

/// Performance profile: how hard the reader is allowed to work this device.
/// A single picker rather than individual knobs — the `Budget` fields are
/// interdependent (a wider prefetch window with a small cache just evicts itself),
/// so exposing them separately would only let a user build a worse configuration.
/// `Auto` (the default) uses the probed [`DeviceTier`]; the rest pin a tier, which
/// is how a flagship keeps today's aggressive behavior (`Performance` = the tier
/// that applies no ceilings) and a device the heuristic misjudged can be corrected.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PerfPref {
    Auto,
    Low,
    Mid,
    High,
}

impl PerfPref {
    /// The tier this pins, or `None` for `Auto` (use the probed one).
    pub(crate) fn tier(self) -> Option<DeviceTier> {
        match self {
            PerfPref::Auto => None,
            PerfPref::Low => Some(DeviceTier::Low),
            PerfPref::Mid => Some(DeviceTier::Mid),
            PerfPref::High => Some(DeviceTier::High),
        }
    }
    /// Persistence token (view.txt slot 8).
    pub(crate) fn label(self) -> &'static str {
        match self {
            PerfPref::Auto => "auto",
            PerfPref::Low => "low",
            PerfPref::Mid => "mid",
            PerfPref::High => "high",
        }
    }
    pub(crate) fn parse(tok: Option<&&str>) -> Self {
        match tok {
            Some(&"low") => PerfPref::Low,
            Some(&"mid") => PerfPref::Mid,
            Some(&"high") => PerfPref::High,
            _ => PerfPref::Auto,
        }
    }
}

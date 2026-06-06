//! The platform-agnostic reading-state machine.
//!
//! This module will own the reader's view model — navigation, zoom/pan, fit and
//! layout, the continuous-scroll anchor, the decode-view debounce, and the
//! single-resize-invariant draw math — so a shell only has to supply a surface,
//! input, and storage. It is filled in across Phase 2; for now it carries the
//! [`Viewport`], the one piece the shell hands in every frame.

use crate::page::{fit_scale, FitMode};

/// The drawable surface size in physical pixels, as the reading math sees it.
///
/// The shell mirrors its `wgpu` surface config into this once per frame (and on
/// resize), so the reader computes exact per-page decode targets and quad
/// placement without depending on any windowing type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Viewport {
    pub w: u32,
    pub h: u32,
}

/// Zoom limits, measured against the image's *native* resolution (1 image px :
/// 1 screen px = 100%), matching BandiView. The reader's zoom is a fit-multiplier,
/// so these convert to per-page multiplier bounds via [`clamp_zoom_multiplier`].
pub const MIN_ZOOM_PCT: f32 = 0.05; // 5% of native
pub const MAX_ZOOM_PCT: f32 = 200.0; // 20000% of native

/// Reading direction — page-turn order and spread pairing (LTR vs RTL/manga).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Ltr,
    Rtl,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Direction::Ltr => "LTR",
            Direction::Rtl => "RTL",
        }
    }
}

/// Fixed zoom ladder in native percent (BandiView): 5, then 10..300 by 10,
/// 320..500 by 20, 550..20000 by 50. The endpoints equal the clamp range
/// (`MIN_ZOOM_PCT*100` .. `MAX_ZOOM_PCT*100`), so snapping never fights the clamp.
pub fn zoom_presets() -> Vec<f32> {
    let mut v = vec![5.0];
    let mut p = 10;
    while p <= 300 {
        v.push(p as f32);
        p += 10;
    }
    p = 320;
    while p <= 500 {
        v.push(p as f32);
        p += 20;
    }
    p = 550;
    while p <= 20000 {
        v.push(p as f32);
        p += 50;
    }
    v
}

/// The next stop above (`zoom_in`) or below `current_pct` in `ladder` (which must
/// be sorted ascending: the fixed presets plus any spliced fit stops). A 0.1%
/// relative guard — tighter than the smallest step (~0.25% at the top) — keeps
/// float noise from sticking on or skipping a level. Clamps at the ends.
pub fn next_zoom_preset(ladder: &[f32], current_pct: f32, zoom_in: bool) -> f32 {
    if zoom_in {
        ladder
            .iter()
            .copied()
            .find(|&p| p > current_pct * 1.001)
            .unwrap_or_else(|| *ladder.last().unwrap())
    } else {
        ladder
            .iter()
            .copied()
            .rev()
            .find(|&p| p < current_pct * 0.999)
            .unwrap_or(ladder[0])
    }
}

/// On-screen scale (device-px per *native* source-px) for a page, mirroring the
/// draw scale in `build_quads`. `content` feeds `fit_scale` (a single page: its
/// own dims; a facing pair: the combined width and shared height); `decoded_h` is
/// the anchor's displayed decoded height; `src_h` its native.
pub fn anchor_native_scale(
    fit: FitMode,
    screen: (f32, f32),
    content: (f32, f32),
    decoded_h: f32,
    src_h: f32,
    zoom: f32,
) -> f32 {
    // 1:1 now draws at native × zoom regardless of the decoded texture size (see
    // single_quad), so its native scale is exactly `zoom` — the decoded_h/src_h
    // correction below applies only to the fit-scaled modes.
    if fit == FitMode::Actual {
        return zoom;
    }
    let ((sw, sh), (fit_w, fit_h)) = (screen, content);
    fit_scale(fit, sw, sh, fit_w, fit_h) * zoom * decoded_h / src_h.max(1.0)
}

/// Clamp a fit-multiplier `zoom` so the *effective native* zoom stays within
/// [`MIN_ZOOM_PCT`, `MAX_ZOOM_PCT`]. `base` is the page's native scale at zoom = 1.
pub fn clamp_zoom_multiplier(zoom: f32, base: f32) -> f32 {
    let (lo, hi) = (MIN_ZOOM_PCT / base, MAX_ZOOM_PCT / base);
    zoom.clamp(lo.min(hi), lo.max(hi))
}

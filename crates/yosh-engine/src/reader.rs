//! The platform-agnostic reading-state machine.
//!
//! This module will own the reader's view model — navigation, zoom/pan, fit and
//! layout, the continuous-scroll anchor, the decode-view debounce, and the
//! single-resize-invariant draw math — so a shell only has to supply a surface,
//! input, and storage. It is filled in across Phase 2; for now it carries the
//! [`Viewport`], the one piece the shell hands in every frame.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use crate::cache::PageCache;
use crate::layout::Layout;
use crate::page::{fit_scale, FitMode};
use crate::pool::DecodePool;
use crate::source::PageSource;
use crate::texpool::TexturePool;

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

/// Decode-target height floor. Each page's decode target tracks its exact
/// on-screen size (see `page_target_h`) so the HQ linear-light CPU resize does the
/// *full* reduction in one pass and the GPU samples 1:1 — the single-resize
/// invariant. Below it, extreme zoom-out would otherwise decode a sub-pixel page.
pub const MIN_TARGET: u32 = 32;

/// A quad to draw this frame (NDC scale + top-left offset), referencing a cached page.
pub struct Quad {
    pub slot: usize,
    pub page_index: usize,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub rot: u32, // 0/1/2/3 = 0/90/180/270° CW (single-page draws only; 0 for spreads)
}

/// Build a [`Quad`] from pixel-space placement — top-left `(x_px, y_px)` and size
/// `(dw, dh)` within a `(sw, sh)` surface — converting to NDC scale + offset.
pub fn quad_from_px(
    slot: usize,
    page_index: usize,
    x_px: f32,
    y_px: f32,
    dw: f32,
    dh: f32,
    sw: f32,
    sh: f32,
    rot: u32,
) -> Quad {
    Quad {
        slot,
        page_index,
        scale: [2.0 * dw / sw, 2.0 * dh / sh],
        offset: [-1.0 + 2.0 * x_px / sw, 1.0 - 2.0 * y_px / sh],
        rot,
    }
}

/// Default h/w aspect estimate for not-yet-decoded pages in the scroll strip.
pub const DEFAULT_ASPECT: f32 = 1.5;

/// The platform-agnostic reading-state machine: navigation, zoom/pan, fit/layout,
/// the continuous-scroll anchor, the decode-view debounce, and the engine
/// resources (page source, decode pool, cache, texture pool) it drives. A shell
/// owns a `Reader`, feeds it a [`Viewport`] and input each frame, and renders the
/// draw list it produces. Fields are `pub` while Phase 2 migrates logic in; they
/// tighten as the reading methods land on `impl Reader`.
pub struct Reader {
    // --- Engine resources ---
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub tex_pool: Arc<TexturePool>,
    pub source: Option<Arc<dyn PageSource>>,
    pub pool: Option<DecodePool>,
    pub cache: PageCache,
    /// Worker count for the decode pool, kept so `set_source` can rebuild it.
    pub workers: usize,
    /// Pages whose decode errored, mapped to the error message (shown to the user).
    pub failed: HashMap<usize, String>,

    // --- Reading model ---
    pub index: usize,
    pub start_index: usize,
    pub last_drawn: Option<usize>,
    pub fit: FitMode,
    pub layout: Layout,
    pub spread_offset: usize, // spread pairing parity (0 or 1), per-volume
    pub rotation: u8,         // 90° CW steps (0..=3); single-page draws only
    pub zoom: f32,            // page-flip zoom factor (1.0 = fit)
    pub pan_x: f32,           // page-flip pan offset in screen px (from centered)
    pub pan_y: f32,
    pub direction: Direction,
    pub jump: bool, // seek mode (key J): true = skip ahead, false = step every page
    pub nav_times: VecDeque<Instant>,
    pub scroll_mode: bool,
    pub top_offset: f32, // px the anchor page is scrolled above the viewport top
    pub est_aspect: f32, // h/w estimate for undecoded pages in the strip

    // --- Surface + decode-view debounce ---
    pub viewport: Viewport,
    /// Last-seen `(surface_w, surface_h, zoom)`; once it holds across a frame the
    /// view is "settled" and target-change re-decodes are allowed.
    pub pending_view: (u32, u32, f32),
    pub view_settled: bool,
    /// True while a "settled view is GPU-downscaling" warning has already fired for
    /// the current episode, so the tripwire logs once, not per frame.
    pub gpu_downscale_warned: bool,
    /// When the zoomed-page wheel-pan first parked at the top/bottom edge (gates
    /// the hard-stop dwell before flipping).
    pub pan_edge_at: Option<Instant>,
}

impl Reader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tex_pool: Arc<TexturePool>,
        cache_cap: usize,
        workers: usize,
        fit: FitMode,
        layout: Layout,
        scroll_mode: bool,
        jump: bool,
        direction: Direction,
        start_index: usize,
    ) -> Self {
        Self {
            cache: PageCache::new(cache_cap, tex_pool.clone()),
            device,
            queue,
            tex_pool,
            source: None,
            pool: None,
            workers,
            failed: HashMap::new(),
            index: 0,
            start_index,
            last_drawn: None,
            fit,
            layout,
            spread_offset: 0,
            rotation: 0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            direction,
            jump,
            nav_times: VecDeque::new(),
            scroll_mode,
            top_offset: 0.0,
            est_aspect: DEFAULT_ASPECT,
            viewport: Viewport::default(),
            pending_view: (0, 0, 1.0),
            view_settled: false,
            gpu_downscale_warned: false,
            pan_edge_at: None,
        }
    }
}

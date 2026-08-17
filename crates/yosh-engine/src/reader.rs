//! The platform-agnostic reading-state machine.
//!
//! This module will own the reader's view model — navigation, zoom/pan, fit and
//! layout, the continuous-scroll anchor, the decode-view debounce, and the
//! single-resize-invariant draw math — so a shell only has to supply a surface,
//! input, and storage. It is filled in across Phase 2; for now it carries the
//! [`Viewport`], the one piece the shell hands in every frame.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cache::PageCache;
use crate::decode::MAX_TEX_DIM;
use crate::layout::{self, Layout};
use crate::page::{fit_scale, FitMode, PageTexture, MAX_QUADS};
use crate::pool::{DecodePool, Msg, Waker};
use crate::prefetch::desired_window;
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

/// How much of a volume the whole-volume LQ thumbnail tier covers. The tail is
/// pure background filler, so a constrained device can shrink or drop it without
/// touching the zero-hitch HQ path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LqTier {
    /// Every page of the volume gets a thumbnail (desktop / flagship behavior).
    Full,
    /// Only pages within ±n of the read position — the pivot restride re-centres
    /// the window as the reader travels, so scrubbing nearby still previews.
    Windowed(u32),
    /// No thumbnail tier at all.
    Off,
}

/// Coarse device class, used to cap [`Budget`] on hardware that can't sustain the
/// desktop configuration. Probed by the shell (RAM, CPU clock); a user-facing
/// performance setting can override it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceTier {
    Low,
    Mid,
    /// Flagship / desktop: **no ceilings at all** — `for_tier` returns exactly what
    /// `derive` computed, so today's behavior is bit-identical.
    High,
}

/// Per-device resource budget, derived from the memory the app may spend on
/// decoded pages + GPU textures and the CPU count. Desktop-class inputs
/// (≥ ~384 MB, ≥ 8 cores) reproduce the historical fixed budgets exactly; small
/// Android heaps scale every dimension down to stay under the OOM killer. A shell
/// supplies the inputs (`available_parallelism`, a RAM/heap probe).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Decode worker threads.
    pub workers: usize,
    /// Bounded decoded-page cache size.
    pub cache_cap: usize,
    /// Global cap on recycled GPU textures.
    pub texpool_max: usize,
    /// Base forward prefetch window (pages ahead of the read position).
    pub fwd: usize,
    /// Forward window when widened by flip velocity.
    pub fwd_max: usize,
    /// Backward prefetch window (pages behind).
    pub back: usize,
    /// Eviction ceiling for the whole-volume LQ thumbnail cache (`lq_cache`). Large
    /// enough to hold a normal volume's worth of tiny previews; for huge volumes the
    /// cache's distance-eviction keeps the nearest pages.
    pub lq_cap: usize,
    /// Ceiling on an uncached 1:1 (`FitMode::Actual`) decode, as a multiple of the
    /// viewport height. `None` (desktop / High) keeps the historical behavior: an
    /// uncached page decodes full-native, since its source size isn't known yet.
    /// `Some(m)` bounds that first decode to `m × viewport.h` — a page can then be
    /// GPU-*up*scaled (the one sanctioned resample) instead of spending ~96 MB of
    /// RGBA on a page most of which is off-screen. Converges through the normal
    /// `target_stale` re-decode once the source dims are known.
    pub actual_cap_vh: Option<u32>,
    /// Decoded height for whole-volume LQ thumbnails (see [`LQ_THUMB_H`]).
    pub lq_thumb_h: u32,
    /// How much of the volume the thumbnail tier covers (see [`LqTier`]).
    pub lq_tier: LqTier,
}

impl Budget {
    /// Derive the budget. `mem_budget_mb` is the memory the reader may use for its
    /// page cache + textures (desktop: a fraction of system RAM; Android: roughly
    /// the per-app heap). Clamps keep desktop at the long-standing values.
    pub fn derive(mem_budget_mb: u64, cpus: usize) -> Self {
        let workers = cpus.clamp(2, 8);
        let cache_cap = ((mem_budget_mb / 8) as usize).clamp(16, 48);
        let texpool_max = (cache_cap / 2).clamp(8, 24);
        let fwd = (cache_cap / 3).clamp(6, 16);
        let fwd_max = (fwd * 5 / 2).clamp(12, 40);
        let back = (fwd / 2).clamp(3, 6);
        let lq_cap = cache_cap.saturating_mul(16);
        Self {
            workers,
            cache_cap,
            texpool_max,
            fwd,
            fwd_max,
            back,
            lq_cap,
            actual_cap_vh: None,
            lq_thumb_h: LQ_THUMB_H,
            lq_tier: LqTier::Full,
        }
    }

    /// [`Budget::derive`], then per-tier **ceilings** for hardware that can't
    /// sustain the desktop configuration.
    ///
    /// Every numeric ceiling is a `min`, never a raise: a device whose derived
    /// budget is already smaller keeps it. [`DeviceTier::High`] applies nothing at
    /// all, so a flagship (and every desktop caller, which goes on using `derive`
    /// directly) is bit-identical to before this existed — that equality is pinned
    /// by a unit test.
    pub fn for_tier(tier: DeviceTier, mem_budget_mb: u64, cpus: usize) -> Self {
        let mut b = Self::derive(mem_budget_mb, cpus);
        let (workers, cache_cap, texpool_max, fwd, fwd_max, back, lq_cap, actual, thumb_h, lq) =
            match tier {
                DeviceTier::Low => (3, 20, 10, 6, 12, 3, 320, Some(2), 360, LqTier::Windowed(64)),
                DeviceTier::Mid => (6, 32, 16, 10, 24, 4, 512, Some(4), 540, LqTier::Full),
                DeviceTier::High => return b,
            };
        b.workers = b.workers.min(workers);
        b.cache_cap = b.cache_cap.min(cache_cap);
        b.texpool_max = b.texpool_max.min(texpool_max);
        b.fwd = b.fwd.min(fwd);
        b.fwd_max = b.fwd_max.min(fwd_max);
        b.back = b.back.min(back);
        b.lq_cap = b.lq_cap.min(lq_cap);
        b.actual_cap_vh = actual;
        b.lq_thumb_h = b.lq_thumb_h.min(thumb_h);
        b.lq_tier = lq;
        b
    }
}

/// A quad to draw this frame (NDC scale + top-left offset), referencing a cached page.
pub struct Quad {
    pub slot: usize,
    pub page_index: usize,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub rot: u32, // 0/1/2/3 = 0/90/180/270° CW (single-page draws only; 0 for spreads)
    /// Opacity multiplier (1.0 = opaque). < 1.0 only for a fading page-turn overlay.
    pub alpha: f32,
    /// Horizontal motion-blur smear in UV (0.0 = none; symmetric). Only the
    /// outgoing page of a page-turn transition sets this — it streaks the page so it
    /// doesn't clash with the sharp incoming page beneath. The slide carries the
    /// direction; the smear is along the (horizontal) slide axis.
    pub blur: f32,
    /// Signed spine-shadow width in UV (0.0 = none). Set only on the two pages of
    /// an un-joined spread pair: >0 shades the page's right edge (screen-left page),
    /// <0 its left edge (screen-right page). Singles, wide joined spreads, and
    /// scroll always carry 0.0.
    pub spine: f32,
    /// Spine-shadow peak darkening, 0..1 (0.0 = off).
    pub spine_strength: f32,
}

/// Build a [`Quad`] from pixel-space placement — top-left `(x_px, y_px)` and size
/// `(dw, dh)` within a `(sw, sh)` surface — converting to NDC scale + offset.
#[allow(clippy::too_many_arguments)]
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
        alpha: 1.0,
        blur: 0.0,
        spine: 0.0,
        spine_strength: 0.0,
    }
}

/// Spine-shadow width as a fraction of the page's on-screen width, clamped in
/// device px so it stays proportionate at any zoom. The fraction was measured
/// from the CDisplayEx reference in issue #11: ~132 px of gradient on a
/// ~1630 px page ≈ 8% of the page width.
const SPINE_FRAC: f32 = 0.08;
const SPINE_MIN_PX: f32 = 6.0;
const SPINE_MAX_PX: f32 = 256.0;

/// Signed spine-shadow widths in UV for the (screen-left, screen-right) pages of
/// a spread, from their on-screen pixel widths. Left page shades its right
/// edge (+), right its left edge (−). 0.0 for degenerate widths.
pub fn spine_uv(dwl: f32, dwr: f32) -> (f32, f32) {
    let w = |d: f32| {
        if d <= 1.0 {
            return 0.0;
        }
        (d * SPINE_FRAC).clamp(SPINE_MIN_PX, SPINE_MAX_PX).min(d) / d
    };
    (w(dwl), -w(dwr))
}

/// Duration of the page-flip transition animation (outgoing page blur + fade).
pub const TRANSITION_MS: u64 = 140;
/// Peak horizontal motion-blur smear (UV half-width) on the fading outgoing page,
/// so it streaks along the slide axis instead of clashing as a second sharp image
/// over the incoming page. 0.0 disables the blur (crisp slide + fade).
const TRANSITION_MAX_BLUR: f32 = 0.06;
/// How far the outgoing page slides toward the exit edge by the end of the
/// transition, as a fraction of the viewport width. The slide is what reads as
/// "sweeping away" in the tapped direction; the blur + fade are the polish.
const TRANSITION_SLIDE_FRAC: f32 = 0.12;

/// A live page-flip animation: the previous view's pages sliding out, fading +
/// motion-smearing, over [`TRANSITION_MS`] while the new view draws underneath.
/// Only armed in discrete page-flip mode (not scroll). `from_frac` is where the
/// slide starts: 0 for a tap flip; the dragged offset for a committed
/// interactive drag — same animation, just picking up where the finger left it.
struct PageTransition {
    start: Instant,
    /// The outgoing view's page indices (1 single, 2 spread).
    out_pages: Vec<usize>,
    /// True ⇒ the outgoing page slides off toward +x (screen right).
    exit_right: bool,
    /// Slide start, as a fraction of the viewport width (0.0 for tap flips).
    from_frac: f32,
}

// --- Interactive page drag (Chunky-style: the page follows the finger) ---
/// Release commits the flip past this fraction of *finger travel*…
const DRAG_COMMIT_FRAC: f32 = 0.25;
/// …or on a flick: at least this fast (px/s, same sign as the drag)…
const DRAG_FLICK_PX_S: f32 = 600.0;
/// …with at least this much travel (so a stray touch can't flip).
const DRAG_FLICK_MIN_FRAC: f32 = 0.04;
/// Releasing while moving back *against* the drag faster than this cancels the
/// flip regardless of how far the page was pulled — a deliberate reversal means
/// "changed my mind" (Chunky behavior). Gentler than the flick threshold so a
/// gentle-but-real backtrack cancels, while sensor jitter doesn't.
const DRAG_CANCEL_PX_S: f32 = 250.0;
/// Extra damping when dragging against the first/last page (rubber-band).
const DRAG_RUBBER: f32 = 0.35;
/// Snap-back duration when a drag is released without committing.
const DRAG_SETTLE_MS: u64 = 150;
/// The page's displacement saturates here (fraction of the viewport width):
/// near-1:1 tracking for small drags, then growing resistance — a full-width
/// finger travel lands essentially at this cap (Chunky-style), and the commit
/// animation takes over from wherever the page actually sits.
const DRAG_MAX_FRAC: f32 = 0.33;

/// Inertial scroll (fling) physics for continuous scroll mode. Velocity decays
/// exponentially (`v *= exp(-FRICTION*dt)`), so a flick's total glide ≈ `v0/FRICTION`.
const SCROLL_FLING_FRICTION: f32 = 1.5; // per second; lower = longer glide
const SCROLL_FLING_MIN_V: f32 = 50.0; // px/s; below this the fling stops
const SCROLL_FLING_MAX_V: f32 = 60000.0; // px/s; cap on release velocity (hard flick)

/// A live one-finger page drag: the current view follows the finger while the
/// neighbor view it is being dragged toward shows underneath (both views are
/// normally already prefetched). On release the drag either commits — `step` +
/// a [`PageTransition`] continuing from the dragged offset — or snap-backs here.
struct PageDrag {
    /// Signed horizontal finger displacement, px (+ = finger moved right).
    dx: f32,
    /// Released without committing: animate the displacement `.0 → 0` from `.1`.
    settle: Option<(f32, Instant)>,
}

impl PageDrag {
    /// The displacement to draw this frame (raw while tracking; easing toward 0
    /// during a snap-back).
    fn current_dx(&self) -> f32 {
        match self.settle {
            None => self.dx,
            Some((from, start)) => {
                let p = (start.elapsed().as_secs_f32() / (DRAG_SETTLE_MS as f32 / 1000.0))
                    .clamp(0.0, 1.0);
                from * (1.0 - p) * (1.0 - p) // ease-out back to rest
            }
        }
    }

    /// Still tracking, or mid snap-back?
    fn live(&self) -> bool {
        match self.settle {
            None => true,
            Some((_, start)) => start.elapsed() < Duration::from_millis(DRAG_SETTLE_MS),
        }
    }
}

/// Flip direction a horizontal drag is asking for: dragging the page toward the
/// "previous" edge advances (drag metaphor — LTR swipe-left = next; RTL mirrors).
/// Matches the shell's historical swipe mapping.
pub fn drag_dir(direction: Direction, dx: f32) -> i64 {
    let next = match direction {
        Direction::Rtl => dx > 0.0,
        Direction::Ltr => dx < 0.0,
    };
    if next { 1 } else { -1 }
}

/// Should a released drag commit the flip? Far enough, or a deliberate flick
/// (fast + same direction + non-trivial travel) — but a deliberate *reversal*
/// at release cancels no matter how far the page was pulled. Measured on raw
/// *finger* travel, not the resistance-damped page displacement.
pub fn drag_commits(dx_px: f32, velocity_px_s: f32, viewport_w: f32) -> bool {
    // Moving back against the drag when the finger lifts = "changed my mind".
    if velocity_px_s.abs() > DRAG_CANCEL_PX_S && velocity_px_s.signum() != dx_px.signum() {
        return false;
    }
    let frac = dx_px.abs() / viewport_w.max(1.0);
    frac > DRAG_COMMIT_FRAC
        || (frac > DRAG_FLICK_MIN_FRAC
            && velocity_px_s.abs() > DRAG_FLICK_PX_S
            && velocity_px_s.signum() == dx_px.signum())
}

/// Finger travel → page displacement (both px, sign preserved): ~1:1 for small
/// drags, smoothly saturating at `DRAG_MAX_FRAC` of the viewport width
/// (`d = MAX·(1 − e^(−|dx|/MAX))`, slope 1 at rest — no kink when the drag
/// starts; a full-width travel reaches ~95% of the cap).
pub fn drag_resist(dx_px: f32, viewport_w: f32) -> f32 {
    let max = DRAG_MAX_FRAC * viewport_w.max(1.0);
    dx_px.signum() * max * (1.0 - (-dx_px.abs() / max).exp())
}

/// Default h/w aspect estimate for not-yet-decoded pages in the scroll strip.
pub const DEFAULT_ASPECT: f32 = 1.5;

/// Default decoded height (px) for whole-volume LQ-tier thumbnails — small enough
/// that a full volume of previews is cheap (~0.2 MB/page gray), large enough to read
/// which page you're on. Shown only transiently until the full-res page lands. The
/// live value is [`Budget::lq_thumb_h`] (the Low device tier shrinks it).
pub const LQ_THUMB_H: u32 = 540;

/// How far (in pages) the reading position must travel before the whole-volume
/// thumbnail tail is rebuilt around the new position. The tail is only *ordered*
/// by distance from the reader — every entry is work worth doing wherever you
/// are — so re-centring it is a scheduling nicety, not a correctness need, and
/// doing it once per stride keeps the O(volume log volume) sort off the frame
/// path. Small enough that a few page turns' worth of travel still re-aims it.
pub const LQ_TAIL_RESTRIDE: usize = 32;

/// Does the whole-volume thumbnail tail need rebuilding for reading position
/// `index`, given the position it was last built around (`None` = never built, or
/// invalidated by a source/pool swap)? Split out of `prefetch` so the stride rule
/// is testable on its own.
fn tail_needs_restride(pivot: Option<usize>, index: usize) -> bool {
    match pivot {
        None => true,
        Some(p) => index.abs_diff(p) >= LQ_TAIL_RESTRIDE,
    }
}

/// The 1:1 (`FitMode::Actual`) decode target: the page's displayed height —
/// `src_h × zoom`, since 1:1 draws at native × zoom regardless of texture size —
/// bounded by the GPU's aspect-derived ceiling `max_h` and, on a constrained
/// device (`cap_vh = Some(m)`), by `m` viewport heights.
///
/// `src_h = None` is a page that isn't decoded yet, so its native size is unknown:
/// an unclamped device (desktop / [`DeviceTier::High`]) asks for everything and
/// lets `target_dims` cap at the source, then re-decodes exactly once the real dims
/// land; a clamped device asks only for what the screen can show, since a
/// full-native RGBA decode of a mostly-off-screen page is exactly the ~96 MB spike
/// the cap exists to prevent.
///
/// Clamping *below* the displayed height is the one place the single-resize
/// invariant is deliberately relaxed — the GPU then **up**scales, the sanctioned
/// resample direction (no screentone moiré), never downscales. Pure so the clamp
/// is unit-testable: a real `Reader` needs a wgpu `Device`.
fn actual_target(
    src_h: Option<f32>,
    zoom: f32,
    max_h: u32,
    viewport_h: u32,
    cap_vh: Option<u32>,
) -> u32 {
    let cap = match cap_vh {
        Some(m) => max_h.min(viewport_h.max(1).saturating_mul(m).max(MIN_TARGET)),
        None => max_h,
    };
    match src_h {
        Some(h) => ((h * zoom).round() as u32).clamp(MIN_TARGET, cap),
        None if cap_vh.is_some() => cap,
        None => u32::MAX,
    }
}

/// Which page indices the thumbnail tail may cover for reading position `index`,
/// under the device's [`LqTier`]: the whole volume on `Full`, a ±n band on
/// `Windowed` (the pivot restride re-centres it as the reader travels), and
/// nothing on `Off`. `None` ⇒ skip the tail build entirely. Split out of
/// `prefetch` so the range math is testable without a GPU-backed `Reader`.
fn lq_tail_range(tier: LqTier, index: usize, len: usize) -> Option<std::ops::RangeInclusive<usize>> {
    if len == 0 {
        return None;
    }
    match tier {
        LqTier::Off => None,
        LqTier::Full => Some(0..=len - 1),
        LqTier::Windowed(n) => {
            let n = n as usize;
            Some(index.saturating_sub(n)..=index.saturating_add(n).min(len - 1))
        }
    }
}

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
    /// Whole-volume LQ preview tier: one tiny `LQ_THUMB_H`-tall thumbnail per page,
    /// filled at lowest priority and drawn as a transient fallback when the full-res
    /// page isn't cached yet (instant seek/jump). Always on; bounded by its own cap
    /// (`Budget::lq_cap`). Separate from the `two_tier` fast-filter flag (same-res).
    pub lq_cache: PageCache,
    /// Worker count for the decode pool, kept so `set_source` can rebuild it.
    pub workers: usize,
    /// [`Budget::actual_cap_vh`] — the 1:1 uncached-decode ceiling, in viewport
    /// heights. `None` on desktop / the High tier (no clamp at all).
    pub actual_cap_vh: Option<u32>,
    /// [`Budget::lq_thumb_h`] — decoded height of whole-volume thumbnails.
    pub lq_thumb_h: u32,
    /// [`Budget::lq_tier`] — how much of the volume the thumbnail tier covers.
    pub lq_tier: LqTier,
    /// The shell's frame-wake callback, remembered so every pool the reader
    /// (re)builds gets it. Set once per window via [`Reader::set_waker`].
    pub waker: Option<Waker>,
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
    /// Two-tier decode: LQ (fast) while seeking → HQ on settle. Off = always HQ
    /// (the desktop's behavior).
    two_tier: bool,
    pub nav_times: VecDeque<Instant>,
    pub scroll_mode: bool,
    pub top_offset: f32, // px the anchor page is scrolled above the viewport top
    /// Inertial scroll velocity (px/sec applied to `top_offset`; +ve scrolls the
    /// strip forward). 0 when not flinging; driven by `fling_tick`.
    pub scroll_velocity: f32,
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
    /// Transient messages (boundary hit, zoom level) emitted by nav/zoom commands;
    /// the shell drains these each frame into its timed on-screen toast.
    pub pending_toasts: Vec<String>,
    // Prefetch window, from the device `Budget` (replaces the old fixed consts).
    pub fwd: usize,
    pub fwd_max: usize,
    pub back: usize,

    // --- Page-turn transition (in-book flip animation) ---
    /// Animate page flips with an outgoing-page blur+fade. Set by the shell from
    /// settings; off by default (each shell opts in after construction).
    pub transition_enabled: bool,
    /// Effective spine-shadow strength, 0..1 (`0.0` = disabled). Set by the shell,
    /// which owns the enabled × strength settings; the engine sees one number.
    pub spine_strength: f32,
    /// The live flip animation, if one is in flight. Cleared once it expires.
    transition: Option<PageTransition>,
    /// Live interactive page drag (page-flip mode only), if one is in progress.
    drag: Option<PageDrag>,
    /// Did the last `build_quads` draw a mid-animation frame (transition overlay
    /// or live drag)? The shells' redraw guards read this instead of re-checking
    /// the clock: a re-check can land just *after* the animation expired even
    /// though the frame that was drawn was mid-fade — freezing a half-faded
    /// ghost of the outgoing page on screen. Deciding once, at draw time,
    /// guarantees one more frame after any animation frame (which then draws
    /// clean and clears this).
    anim_drawn: std::cell::Cell<bool>,
    /// During a page-flip drag, the leading-edge seam for the shell's drop shadow:
    /// `(seam_fraction 0..1, signed_intensity)` — sign = which side the revealed page
    /// is on. `None` when no drag is live. Set by `build_quads`, read via `drag_seam`.
    drag_seam: std::cell::Cell<Option<(f32, f32)>>,

    /// Inputs of the last `prefetch()` job-list rebuild. Shells call `prefetch()`
    /// every frame; when neither the view nor the caches changed since the last
    /// call the desired job list is identical, so it skips the rebuild (and the
    /// `set_jobs` lock + wakeups it would cost).
    last_jobs_key: Option<JobsKey>,
    /// Reading position the whole-volume thumbnail tail was last built around, or
    /// `None` when the tail needs (re)building — see [`LQ_TAIL_RESTRIDE`].
    lq_tail_pivot: Option<usize>,
}

/// Everything the `prefetch()` job list depends on. Cache/failed contents are
/// captured via change counters (`PageCache::epoch`) rather than the sets
/// themselves; `zoom` as bits so the key is `Eq`-comparable.
///
/// Note what is deliberately *absent*: the `lq_cache` epoch. The job list this key
/// guards is the **HQ window only** — the whole-volume thumbnail tail lives in the
/// pool now and is maintained on its own stride (`lq_tail_pivot`), so a landing
/// thumbnail changes nothing here. Including it made every one of the hundreds of
/// thumbnails in a volume fill force a full job-list rebuild + `set_jobs`.
#[derive(PartialEq, Clone, Copy)]
struct JobsKey {
    index: usize,
    len: usize,
    fwd: usize,
    viewport: (u32, u32),
    zoom_bits: u32,
    settled: bool,
    fit: FitMode,
    layout: Layout,
    spread_offset: usize,
    rotation: u8,
    scroll_mode: bool,
    cache_epoch: u64,
    failed_len: usize,
}

impl Reader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tex_pool: Arc<TexturePool>,
        budget: Budget,
        fit: FitMode,
        layout: Layout,
        scroll_mode: bool,
        direction: Direction,
        start_index: usize,
        two_tier: bool,
    ) -> Self {
        Self {
            cache: PageCache::new(budget.cache_cap, tex_pool.clone()),
            lq_cache: PageCache::new(budget.lq_cap, tex_pool.clone()),
            device,
            queue,
            tex_pool,
            source: None,
            pool: None,
            workers: budget.workers,
            actual_cap_vh: budget.actual_cap_vh,
            lq_thumb_h: budget.lq_thumb_h,
            lq_tier: budget.lq_tier,
            waker: None,
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
            nav_times: VecDeque::new(),
            scroll_mode,
            top_offset: 0.0,
            scroll_velocity: 0.0,
            est_aspect: DEFAULT_ASPECT,
            viewport: Viewport::default(),
            pending_view: (0, 0, 1.0),
            view_settled: false,
            gpu_downscale_warned: false,
            pan_edge_at: None,
            pending_toasts: Vec::new(),
            fwd: budget.fwd,
            fwd_max: budget.fwd_max,
            back: budget.back,
            two_tier,
            transition_enabled: false,
            spine_strength: 0.0,
            transition: None,
            drag: None,
            anim_drawn: std::cell::Cell::new(false),
            drag_seam: std::cell::Cell::new(None),
            last_jobs_key: None,
            lq_tail_pivot: None,
        }
    }

    /// Force the next `prefetch()` to rebuild everything it caches across frames:
    /// the HQ job list *and* the whole-volume thumbnail tail. Call it whenever the
    /// pool or the source is replaced — a fresh pool has an empty tail, and a stale
    /// pivot would suppress the rebuild that refills it until the reader happened to
    /// travel `LQ_TAIL_RESTRIDE` pages.
    pub fn invalidate_jobs(&mut self) {
        self.last_jobs_key = None;
        self.lq_tail_pivot = None;
    }

    /// Queue a transient message; the shell drains it into its timed toast.
    pub fn toast(&mut self, msg: impl Into<String>) {
        self.pending_toasts.push(msg.into());
    }

    /// Install the shell's frame-wake callback (see [`Waker`]): remembered on the
    /// reader *and* pushed to the live pool, so a landing decode schedules the
    /// frame that draws it instead of the shell polling for it. Every site that
    /// rebuilds `self.pool` must re-apply it (`apply_refreshed_source` does; the
    /// shells' own rebuilds call `DecodePool::set_waker` with `reader.waker`).
    pub fn set_waker(&mut self, w: Option<Waker>) {
        self.waker = w;
        if let Some(pool) = &self.pool {
            pool.set_waker(self.waker.clone());
        }
    }

    /// Drain finished decodes from the pool into the right cache — full-res pages to
    /// `cache`, whole-volume LQ thumbnails to `lq_cache` — and record failures. Both
    /// shells call this once per frame (replaces a duplicated poll loop).
    pub fn drain_pool(&mut self) {
        let msgs = match &self.pool {
            Some(pool) => pool.poll(),
            None => return,
        };
        for msg in msgs {
            match msg {
                Msg::Done { index, page, thumb } => {
                    if thumb {
                        self.lq_cache.insert(index, page, self.index);
                    } else {
                        self.est_aspect = page.h as f32 / page.w as f32;
                        self.cache.insert(index, page, self.index);
                        // The page has a real texture now, so any thumbnail still
                        // queued for it is wasted work. Cancelling is a set insert;
                        // the tail itself is only rebuilt on the pivot stride.
                        if let Some(pool) = &self.pool {
                            pool.cancel_lq(index);
                        }
                    }
                }
                Msg::Failed { index, error } => {
                    self.failed.insert(index, error);
                }
            }
        }
    }

    /// The texture to draw for page `i`: the full-res page if cached, else the
    /// whole-volume LQ thumbnail — a transient upscaled preview shown until the
    /// full-res decode lands and snaps in (the decode wakes the frame loop itself,
    /// so nothing has to keep redrawing on the chance that it did).
    pub fn page_texture(&self, i: usize) -> Option<&PageTexture> {
        self.cache.get(i).or_else(|| self.lq_cache.get(i))
    }
}

impl Reader {
    /// Does the current page overflow the window vertically under the active fit?
    pub fn current_overflows(&self) -> bool {
        let Some(pt) = self.cache.get(self.index) else {
            return false;
        };
        let (sw, sh) = (self.viewport.w.max(1) as f32, self.viewport.h.max(1) as f32);
        let s = fit_scale(self.fit, sw, sh, pt.w as f32, pt.h as f32) * self.zoom;
        pt.h as f32 * s > sh + 0.5
    }

    /// Top edge (screen px): centered, then panned by `pan_y`, clamped so the
    /// page can't pull away from the viewport edge when larger than it.
    pub fn vertical_top(&self, dh: f32, sh: f32) -> f32 {
        let maxp = ((dh - sh) / 2.0).max(0.0);
        (sh - dh) / 2.0 + self.pan_y.clamp(-maxp, maxp)
    }

    /// Left edge (screen px): centered, then panned by `pan_x`, clamped.
    pub fn horizontal_left(&self, dw: f32, sw: f32) -> f32 {
        let maxp = ((dw - sw) / 2.0).max(0.0);
        (sw - dw) / 2.0 + self.pan_x.clamp(-maxp, maxp)
    }

    /// Displayed height of the current page under the active fit + zoom.
    pub fn current_display_h(&self) -> f32 {
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        match self.cache.get(self.index) {
            Some(t) => {
                t.h as f32 * fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32) * self.zoom
            }
            None => sh,
        }
    }

    /// Flip-mode anchor metrics `(sw, sh, fit_w, fit_h, dec_h, src_h)` — the inputs
    /// `anchor_native_scale` needs, computed once and shared by `anchor_scale`
    /// (current fit/zoom) and `fit_native_pct` (an arbitrary fit at zoom 1). `None`
    /// in scroll mode (no facing-pair layout) or before the anchor is decoded.
    pub fn anchor_metrics(&self) -> Option<(f32, f32, f32, f32, f32, f32)> {
        if self.scroll_mode {
            return None;
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        let len = self.source.as_ref()?.len();
        if len == 0 {
            return None;
        }
        let (a, b) = layout::view_pages(self.layout, self.index, len, self.spread_offset);
        let ta = self.cache.get(a)?;
        // Wide (landscape) page is shown alone; otherwise pair with `b` if ready.
        let force_single = ta.w > ta.h;
        let tb = if force_single { None } else { b.and_then(|bi| self.cache.get(bi)) };
        let (fit_w, fit_h, dec_h) = match tb {
            Some(tb) => {
                let h_ref = ta.h.max(tb.h) as f32;
                let wa = ta.w as f32 * h_ref / ta.h.max(1) as f32;
                let wb = tb.w as f32 * h_ref / tb.h.max(1) as f32;
                (wa + wb, h_ref, h_ref)
            }
            None => (ta.w as f32, ta.h as f32, ta.h as f32),
        };
        Some((sw, sh, fit_w, fit_h, dec_h, ta.src_h.max(1) as f32))
    }

    /// The page(s) actually on screen right now, as `(anchor, facing?)`.
    ///
    /// [`layout::view_pages`] gives the *pairing* for an index, but two reader-side
    /// rules decide what's really drawn, and shells must not re-derive them: a wide
    /// (landscape) page is a pre-joined double-spread and shows alone, and continuous
    /// scroll has no pairing at all. Both mirror `place_view`.
    ///
    /// This also normalizes the anchor: `goto` sets `index` directly, so after a
    /// seekbar jump `index` can be the *second* page of a pair — callers that use it
    /// raw end up describing a different page than the title does.
    ///
    /// Deliberately independent of decode state, so the answer doesn't flicker while
    /// the facing page is still decoding. Don't read `build_quads()` for this: during
    /// a page-turn transition or a live drag it also carries the outgoing view.
    pub fn visible_pages(&self) -> (usize, Option<usize>) {
        let len = self.source.as_ref().map_or(0, |s| s.len());
        if len == 0 {
            return (self.index, None);
        }
        if self.scroll_mode {
            return (self.index.min(len - 1), None);
        }
        let (a, b) = layout::view_pages(self.layout, self.index, len, self.spread_offset);
        // Wide (landscape) anchor is a double-page image → it shows alone.
        let wide = self.cache.get(a).is_some_and(|t| t.w > t.h);
        (a, if wide { None } else { b })
    }

    /// device-px-per-native-px of the in-view anchor page, matching exactly what
    /// `build_quads` draws (single vs. facing-pair dims). `None` while the anchor
    /// isn't decoded yet.
    pub fn anchor_scale(&self) -> Option<f32> {
        if self.scroll_mode {
            // Strip pages are laid out at width = sw * zoom (height follows aspect).
            let sw = self.viewport.w.max(1) as f32;
            let t = self.cache.get(self.index)?;
            return Some(sw * self.zoom / t.src_w.max(1) as f32);
        }
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        Some(anchor_native_scale(
            self.fit,
            (sw, sh),
            (fit_w, fit_h),
            dec_h,
            src_h,
            self.zoom,
        ))
    }

    /// The native zoom % the current anchor would display at under `fit` at zoom 1
    /// — used to splice fit-to-window / fit-to-width stops into the zoom ladder.
    /// `None` in scroll mode (handled inline in `zoom_ladder`) or before decode.
    pub fn fit_native_pct(&self, fit: FitMode) -> Option<f32> {
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        Some(anchor_native_scale(fit, (sw, sh), (fit_w, fit_h), dec_h, src_h, 1.0) * 100.0)
    }

    /// Zoom relative to the *original* image resolution (1 image px : 1 screen px
    /// = 100%), for the toast + info overlay. Derived from the same scale the
    /// renderer draws, so it tracks fit-to-window upscaling and facing pairs
    /// exactly. Falls back to the raw factor while the anchor isn't decoded.
    pub fn effective_zoom_pct(&self) -> f32 {
        let scale = self.anchor_scale().unwrap_or(self.zoom);
        (scale * 100.0).max(0.0)
    }

    /// Device-px per *decoded texel* for the in-view anchor — i.e. exactly how the
    /// GPU sampler scales the texture at draw time. `1.0` = sampling 1:1 (the HQ CPU
    /// resize did all the work); `>1` = GPU upscale (zoom-past-native magnification,
    /// the one allowed GPU resample); `<1` = GPU downscale (the soft/moiré path —
    /// only ever valid as a transient while a re-decode is in flight). `None` before
    /// the anchor is decoded.
    pub fn gpu_sample_scale(&self) -> Option<f32> {
        if self.scroll_mode {
            let t = self.cache.get(self.index)?;
            let sw = self.viewport.w.max(1) as f32;
            return Some(sw * self.zoom / t.w.max(1) as f32); // strip drawn at width sw*zoom
        }
        // Equals single_quad's draw scale `s`: native scale × (src_h / decoded_h).
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        let native = anchor_native_scale(self.fit, (sw, sh), (fit_w, fit_h), dec_h, src_h, self.zoom);
        Some(native * src_h / dec_h.max(1.0))
    }

    /// The in-view anchor's full resize pipeline for the info overlay:
    /// `"<CPU resize path>  →  <GPU sampling state>"`. Empty until decoded.
    /// `(CPU resize-path label, GPU sample scale, re-decode-pending)` for the
    /// in-view anchor — `None` until it's decoded. `pending` means the texture's
    /// decode target no longer matches the *current* desired target, so a re-decode
    /// is due: any GPU downscale right now is transient and will converge. So
    /// `!pending && scale < 1` is the only genuine single-resize-invariant violation
    /// (decoded at the intended target, yet the GPU still has to shrink it).
    pub fn anchor_resize_state(&self) -> Option<(&'static str, f32, bool)> {
        let src = self.source.as_ref()?;
        let len = src.len();
        if len == 0 {
            return None;
        }
        let anchor = if self.scroll_mode {
            self.index
        } else {
            layout::view_pages(self.layout, self.index, len, self.spread_offset).0
        };
        let t = self.cache.get(anchor)?;
        let s = self.gpu_sample_scale()?;
        let pending = t.target_h != self.page_target_h(anchor);
        Some((t.path.label(), s, pending))
    }

    /// The in-view anchor's full resize pipeline for the info overlay:
    /// `"<CPU resize path>  →  <GPU sampling state>"`. Empty until decoded.
    pub fn resize_path_label(&self) -> String {
        let Some((cpu, s, pending)) = self.anchor_resize_state() else {
            return String::new();
        };
        let gpu = if (s - 1.0).abs() <= 0.01 {
            "GPU 1:1".to_string()
        } else if s > 1.0 {
            format!("GPU \u{2191}{s:.2}\u{d7} (magnify)")
        } else if pending {
            format!("GPU \u{2193}{s:.2}\u{d7} (re-decoding\u{2026})")
        } else {
            format!("GPU \u{2193}{s:.2}\u{d7} (LQ \u{2014} STUCK)")
        };
        format!("{cpu}  \u{2192}  {gpu}")
    }

    /// Refresh the live resize readout (`ui.resize_path`) and fire a one-shot debug
    /// warning only on a *genuine* violation: the anchor is decoded at its intended
    /// target (no re-decode pending) yet the GPU is still downscaling it. Re-decode
    /// transients (a fresh page still at its prefetch-guessed size, a zoom/resize not
    /// yet settled) are expected and are not warned.
    pub fn update_resize_readout(&mut self) {
        let stuck = !self.scroll_mode
            && matches!(self.anchor_resize_state(), Some((_, s, pending)) if !pending && s < 0.99);
        if stuck && !self.gpu_downscale_warned {
            eprintln!(
                "yosh: WARNING — view at its decode target is still GPU-downscaling (single-resize invariant violated): {}",
                self.resize_path_label()
            );
            self.gpu_downscale_warned = true;
        } else if !stuck {
            self.gpu_downscale_warned = false;
        }
    }

    /// The active zoom ladder: the fixed presets plus the current page's
    /// fit-to-window and fit-to-width stops (which depend on its resolution),
    /// in-range, sorted, and de-duplicated. In scroll mode the two fit stops are
    /// the page's width-fit (the zoom-1 strip) and height-fit native percents.
    pub fn zoom_ladder(&self) -> Vec<f32> {
        let mut ladder = zoom_presets();
        let (lo, hi) = (MIN_ZOOM_PCT * 100.0, MAX_ZOOM_PCT * 100.0);
        let mut stops: Vec<f32> = Vec::new();
        if self.scroll_mode {
            if let Some(t) = self.cache.get(self.index) {
                let sw = self.viewport.w.max(1) as f32;
                let sh = self.viewport.h.max(1) as f32;
                stops.push(sw / t.src_w.max(1) as f32 * 100.0); // fit width (strip @ zoom 1)
                stops.push(sh / t.src_h.max(1) as f32 * 100.0); // fit window (height fills)
            }
        } else {
            for f in [FitMode::Window, FitMode::Width] {
                if let Some(p) = self.fit_native_pct(f) {
                    stops.push(p);
                }
            }
        }
        for p in stops {
            // Splice a fit stop only if it isn't essentially on a value already in
            // the ladder. Otherwise a fit level a hair off a round preset (e.g. a
            // fit-window of 69.99% next to the 70% preset) shadows the preset, and
            // zoom snaps to 69.99% / 70.01% instead of a clean 70%. Keeping the
            // round preset still gets the "(Fit window/width)" toast label via the
            // `near()` check in `zoom_to_preset`.
            if (lo..=hi).contains(&p) && !ladder.iter().any(|&q| (q - p).abs() <= q * 1e-3) {
                ladder.push(p);
            }
        }
        ladder.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ladder
    }

    /// Clamp the page-flip zoom so the page's *effective native* zoom stays within
    /// [`MIN_ZOOM_PCT`, `MAX_ZOOM_PCT`]. No-op until the anchor page is decoded.
    pub fn clamp_zoom_native(&mut self) {
        if self.zoom > 0.0
            && let Some(s) = self.anchor_scale()
        {
            let base = s / self.zoom;
            if base > 0.0 {
                self.zoom = clamp_zoom_multiplier(self.zoom, base);
            }
        }
    }

    /// Clamp stored pan to the current page's overflow so dragging/zooming can't
    /// strand the view in an empty region.
    pub fn clamp_pan(&mut self) {
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        if self.scroll_mode {
            let cw = sw * self.zoom;
            let mx = ((cw - sw) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            return;
        }
        if let Some(t) = self.cache.get(self.index) {
            // Match single_quad's rotated bounding box so pan clamps to the
            // displayed (possibly turned) page, not the source orientation.
            let single = self.layout == Layout::Single || t.w > t.h;
            let (ew, eh) = if single && self.rotation % 2 == 1 {
                (t.h as f32, t.w as f32)
            } else {
                (t.w as f32, t.h as f32)
            };
            let s = fit_scale(self.fit, sw, sh, ew, eh) * self.zoom;
            let mx = ((ew * s - sw) / 2.0).max(0.0);
            let my = ((eh * s - sh) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            self.pan_y = self.pan_y.clamp(-my, my);
        }
    }

    pub fn single_quad(&self, idx: usize, t: &PageTexture, sw: f32, sh: f32) -> Quad {
        // A 90°/270° turn swaps the page's effective width/height for fitting; the
        // shader then turns the texture inside this (rotated) bounding box. The box
        // dimensions stay whole texels at the fit scale, so 1:1 sampling holds (the
        // decode target in `page_target_h` is rotation-aware to match).
        let (dw, dh) = if self.fit == FitMode::Actual {
            // 1:1: size from the *source* dims × zoom, not the decoded dims, so the
            // displayed box is the same native size whether the texture is full res
            // (zoom ≥ 1) or re-decoded smaller for zoom-out — the latter then samples
            // 1:1 instead of the GPU bilinear-downscaling a full-res texture.
            let (nw, nh) = if self.rotation % 2 == 1 {
                (t.src_h as f32, t.src_w as f32)
            } else {
                (t.src_w as f32, t.src_h as f32)
            };
            (nw * self.zoom, nh * self.zoom)
        } else {
            let (ew, eh) = if self.rotation % 2 == 1 {
                (t.h as f32, t.w as f32)
            } else {
                (t.w as f32, t.h as f32)
            };
            let s = fit_scale(self.fit, sw, sh, ew, eh) * self.zoom;
            (ew * s, eh * s)
        };
        // Snap the page to the device-pixel grid. At 1:1 (fit-to-window) a
        // fractional offset would make the bilinear sampler blend every column
        // 50/50 with its neighbour — a horizontal smear that also beats against
        // halftone screentones. Whole-pixel placement samples texel centers 1:1.
        quad_from_px(
            0,
            idx,
            self.horizontal_left(dw, sw).round(),
            self.vertical_top(dh, sh).round(),
            dw.round(),
            dh.round(),
            sw,
            sh,
            self.rotation as u32,
        )
    }

    /// Compute the quads to draw this frame (1 for single/last-held, 2 for a
    /// ready spread). Only includes pages present in the cache. When a page-flip
    /// transition is in flight, the outgoing view is appended on top, fading out.
    pub fn build_quads(&self) -> Vec<Quad> {
        self.anim_drawn.set(false);
        self.drag_seam.set(None);
        let Some(src) = &self.source else {
            return Vec::new();
        };
        let len = src.len();
        if len == 0 {
            return Vec::new();
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;

        let (a, b) = layout::view_pages(self.layout, self.index, len, self.spread_offset);

        // Live interactive drag: the current view follows the finger; the neighbor
        // view it's being dragged toward shows underneath (drawn first — pages are
        // opaque, so quad order is the reveal). Mutually exclusive with the
        // transition overlay (`drag_update` snaps it; a commit replaces the drag).
        if let Some(d) = &self.drag
            && d.live()
        {
            self.anim_drawn.set(true);
            let raw = d.current_dx();
            let toward = drag_dir(self.direction, raw);
            let incoming = if toward > 0 {
                layout::next_view(self.layout, self.index, len, self.spread_offset)
            } else {
                layout::prev_view(self.layout, self.index, len, self.spread_offset)
            };
            let mut quads = Vec::new();
            let finger = if incoming != self.index {
                let (ia, ib) = layout::view_pages(self.layout, incoming, len, self.spread_offset);
                quads = self.place_view(ia, ib, sw, sh, 0, false);
                raw
            } else {
                // First/last page: nothing to reveal — extra-stiff pull.
                raw * DRAG_RUBBER
            };
            // Progressive resistance: the page tracks the finger near rest but
            // saturates at DRAG_MAX_FRAC of the width (Chunky-style). The
            // fractional offset breaks pixel snapping only while the drag is
            // live (same accepted transient as scroll), re-snapping on rest.
            let dx = drag_resist(finger, sw);
            // Drop-shadow seam for the shell — only when a neighbor is actually
            // revealed (no shadow on a first/last-page rubber-band).
            if incoming != self.index && dx.abs() > 0.5 {
                let seam_x = if dx > 0.0 { dx } else { sw + dx };
                let si = (dx.abs() / 220.0).clamp(0.0, 1.0) * dx.signum();
                self.drag_seam.set(Some((seam_x / sw, si)));
            }
            let base = quads.len().max(2);
            for mut q in self.place_view(a, b, sw, sh, base, true) {
                q.offset[0] += 2.0 * dx / sw;
                quads.push(q);
            }
            return quads;
        }

        let mut quads = self.place_view(a, b, sw, sh, 0, true);

        // Page-turn transition: overlay the previous view, fading + smearing out.
        if let Some(t) = &self.transition {
            let p = (t.start.elapsed().as_secs_f32() / (TRANSITION_MS as f32 / 1000.0)).clamp(0.0, 1.0);
            if p < 1.0 {
                self.anim_drawn.set(true);
                let eased = 1.0 - (1.0 - p) * (1.0 - p); // ease-out (slide + defocus)
                // Fade faster than the slide (cubic) so the faint outgoing ghost
                // clears early. A fade that tracks the slide leaves a dim page
                // creeping through the back half — that lingering tail is what reads
                // as "sluggish" even when the slide speed already matches.
                let fade = 1.0 - p;
                let alpha = fade * fade * fade;
                let blur = TRANSITION_MAX_BLUR * eased; // horizontal motion blur, grows as it fades
                // Slide the outgoing page toward the exit edge (NDC: full viewport
                // width = 2.0) — from rest for a tap flip, or from the dragged
                // offset for a committed drag (the same animation, continued).
                let exit = if t.exit_right { 1.0 } else { -1.0 };
                let slide_frac = t.from_frac + TRANSITION_SLIDE_FRAC * eased;
                let slide = exit * slide_frac * 2.0;
                let oa = t.out_pages[0];
                let ob = t.out_pages.get(1).copied();
                // Outgoing quads draw last (on top); slots after the current view's.
                let base = quads.len().max(2);
                for mut q in self.place_view(oa, ob, sw, sh, base, false) {
                    q.alpha = alpha;
                    q.blur = blur;
                    q.offset[0] += slide;
                    quads.push(q);
                }
            }
        }
        quads
    }

    /// Did the most recent `build_quads` draw a mid-animation frame (page-turn
    /// overlay or interactive drag)? This is what an end-of-frame redraw guard
    /// must use — a clock-based "is the animation still running?" check there
    /// would re-sample time *after* the draw, and an animation expiring in that
    /// gap freezes its last mid-fade frame on screen. Decided at draw time, so
    /// the frame after any animation frame always renders (clean).
    pub fn animation_drawn(&self) -> bool {
        self.anim_drawn.get()
    }

    /// During a live page-flip drag, `(seam_fraction 0..1, signed_intensity)` for the
    /// shell's edge drop shadow; `None` otherwise. See the `drag_seam` field.
    pub fn drag_seam(&self) -> Option<(f32, f32)> {
        self.drag_seam.get()
    }

    /// Place one view's pages into draw quads — 1 for a single page (or a wide
    /// double-spread image shown alone), 2 for a ready facing-page spread.
    /// `slot_base` is the first GPU quad slot to use. `hold_last` falls back to the
    /// last-drawn page when the anchor isn't cached yet (current view only; the
    /// transition overlay passes `false` so a missing outgoing page just snaps).
    fn place_view(&self, a: usize, b: Option<usize>, sw: f32, sh: f32, slot_base: usize, hold_last: bool) -> Vec<Quad> {
        let ta = self.page_texture(a);
        // Wide (landscape) page is a double-spread image → show it alone.
        let force_single = ta.is_some_and(|t| t.w > t.h);
        let b = if force_single { None } else { b };
        let tb = b.and_then(|bi| self.page_texture(bi).map(|t| (bi, t)));

        match (ta, tb) {
            (Some(ta), Some((bi, tb))) => {
                // Facing pages share a display height. Size each to a common
                // reference height (its width following its own aspect ratio)
                // before fitting the pair to the window. Aspect ratios are
                // stable across decode resolutions, so if the two pages are
                // momentarily decoded at different heights — e.g. mid re-decode
                // after a fullscreen toggle / resize, where one updates a frame
                // before the other — neither page jumps size. (Identical to
                // per-pixel sizing when both heights already match.)
                let h_ref = ta.h.max(tb.h) as f32;
                let wa = ta.w as f32 * h_ref / ta.h.max(1) as f32;
                let wb = tb.w as f32 * h_ref / tb.h.max(1) as f32;
                let combined_w = wa + wb;
                let s = fit_scale(self.fit, sw, sh, combined_w, h_ref) * self.zoom;
                let x0 = self.horizontal_left(combined_w * s, sw);
                let dh = h_ref * s;
                // Screen order: LTR puts the lower index on the left; RTL reverses.
                let (l_idx, wl, r_idx, wr) = match self.direction {
                    Direction::Ltr => (a, wa, bi, wb),
                    Direction::Rtl => (bi, wb, a, wa),
                };
                let (dwl, dwr) = (wl * s, wr * s);
                // Snap to the pixel grid (see single_quad). The right page starts
                // at the left's snapped right edge, so there's no sub-pixel seam.
                let yt = self.vertical_top(dh, sh).round();
                let dhr = dh.round();
                let xl = x0.round();
                let dwl_r = dwl.round();
                let mut v = vec![
                    quad_from_px(slot_base, l_idx, xl, yt, dwl_r, dhr, sw, sh, 0),
                    quad_from_px(slot_base + 1, r_idx, xl + dwl_r, yt, dwr.round(), dhr, sw, sh, 0),
                ];
                // Only an un-joined pair gets the gutter shading; the wide-spread and
                // single arms below never touch these fields.
                if self.spine_strength > 0.0 {
                    let (sl, sr) = spine_uv(dwl_r, dwr.round());
                    v[0].spine = sl;
                    v[1].spine = sr;
                    v[0].spine_strength = self.spine_strength;
                    v[1].spine_strength = self.spine_strength;
                }
                v
            }
            (Some(ta), None) => {
                let mut q = self.single_quad(a, ta, sw, sh);
                q.slot = slot_base;
                vec![q]
            }
            _ => {
                // Anchor has no texture yet (not even a thumbnail): hold the
                // last-drawn page if it still has one.
                if hold_last
                    && let Some(li) = self.last_drawn
                    && let Some(t) = self.page_texture(li)
                {
                    let mut q = self.single_quad(li, t, sw, sh);
                    q.slot = slot_base;
                    return vec![q];
                }
                Vec::new()
            }
        }
    }

    pub fn page_display_h(&self, i: usize, sw: f32) -> f32 {
        let cw = sw * self.zoom; // strip content width (zoomable)
        match self.page_texture(i) {
            Some(t) => cw * (t.h as f32 / t.w as f32),
            None => cw * self.est_aspect,
        }
    }

    /// Keep `(index, top_offset)` in range using best-known page heights, so the
    /// scroll anchor stays valid as nearby pages decode (and their real heights land).
    pub fn normalize(&mut self) {
        let len = match &self.source {
            Some(s) => s.len(),
            None => return,
        };
        if len == 0 {
            return;
        }
        let sw = self.viewport.w.max(1) as f32;
        while self.index + 1 < len {
            let h = self.page_display_h(self.index, sw);
            if self.top_offset >= h {
                self.top_offset -= h;
                self.index += 1;
            } else {
                break;
            }
        }
        while self.top_offset < 0.0 && self.index > 0 {
            self.index -= 1;
            self.top_offset += self.page_display_h(self.index, sw);
        }
        if self.index == 0 && self.top_offset < 0.0 {
            self.top_offset = 0.0;
        }
        if self.index + 1 >= len {
            let vh = self.viewport.h as f32;
            let max_off = (self.page_display_h(len - 1, sw) - vh).max(0.0);
            if self.top_offset > max_off {
                self.top_offset = max_off;
            }
        }
    }

    /// Build the visible vertical-strip quads (width-fit, stacked top to bottom).
    pub fn build_scroll_quads(&self) -> Vec<Quad> {
        // No flip animations in scroll mode — clear the draw-time flag so a
        // mode switch can't leave a stale `animation_drawn` pinning the loop.
        self.anim_drawn.set(false);
        let Some(src) = &self.source else {
            return Vec::new();
        };
        let len = src.len();
        if len == 0 {
            return Vec::new();
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        let mut quads = Vec::new();
        let cw = sw * self.zoom; // strip width (zoom); centered with horizontal pan
        let x = self.horizontal_left(cw, sw);
        let mut y = -self.top_offset;
        let mut i = self.index;
        let mut slot = 0;
        while i < len && y < sh && slot < MAX_QUADS {
            let dh = self.page_display_h(i, sw);
            if y + dh > 0.0
                && self.page_texture(i).is_some() {
                    quads.push(quad_from_px(slot, i, x, y, cw, dh, sw, sh, 0));
                    slot += 1;
                }
            y += dh;
            i += 1;
        }
        quads
    }

    /// Forward prefetch-window width: base `FWD`, widened by recent flip velocity
    /// (flips in the last ~0.8 s) up to `FWD_MAX`, so fast seeking buffers further
    /// ahead.
    pub fn dynamic_fwd(&mut self) -> usize {
        let now = Instant::now();
        while let Some(&t) = self.nav_times.front() {
            if now.duration_since(t) > Duration::from_millis(800) {
                self.nav_times.pop_front();
            } else {
                break;
            }
        }
        (self.fwd + self.nav_times.len() * 4).min(self.fwd_max)
    }

    /// Source aspect (w / h) for page `index`: from its decoded texture if present,
    /// else the in-view anchor's, else the running estimate. Used to size the decode
    /// target before the page itself is decoded (exact for the usual uniform-size
    /// volume; corrected in place once the page's own dimensions are known).
    pub fn page_aspect(&self, index: usize) -> f32 {
        if let Some(t) = self.cache.get(index) {
            return t.src_w as f32 / t.src_h.max(1) as f32;
        }
        if let Some(t) = self.cache.get(self.index) {
            return t.src_w as f32 / t.src_h.max(1) as f32;
        }
        1.0 / self.est_aspect.max(0.01) // est_aspect is h / w
    }

    /// The *exact* decode target (on-screen displayed pixel height) for page
    /// `index` under the active fit/zoom/layout. Decoding each page to this height
    /// makes the HQ CPU resize the only resample and the GPU sample 1:1 — the
    /// single-resize invariant. `target_dims` later caps it at the source height
    /// (so a display larger than native means full-res + GPU upscale, the one
    /// allowed exception). 1:1 keeps full source res (it draws at `zoom` directly).
    pub fn page_target_h(&self, index: usize) -> u32 {
        let aspect = self.page_aspect(index).max(0.001);
        // Cap the decode target so neither the texture height nor its aspect-derived
        // width exceeds the GPU's real max texture size. This replaces a former fixed
        // 3840 cap, which forced the GPU to *upscale* (and thus moiré) any page taller
        // than 3840 px viewed near native — the texture couldn't be decoded to the
        // shown size, so the GPU resampled it. Now the HQ CPU resize hits the display
        // size and the GPU stays 1:1 below native, all the way up to what the GPU can
        // hold. (`target_dims` still caps at the source height, so it never upscales.)
        let max_dim = MAX_TEX_DIM.load(std::sync::atomic::Ordering::Relaxed);
        let max_h = ((max_dim as f32 / aspect.max(1.0)).floor() as u32).max(MIN_TARGET);
        if self.fit == FitMode::Actual && !self.scroll_mode {
            // 1:1 displays at native × zoom. Target that height so the page decodes
            // to its *shown* size: `target_dims` caps at the source height, so
            // zoom ≥ 1 keeps full res (magnification GPU-upscales, the one allowed
            // GPU resample) while zoom < 1 decodes smaller → the HQ CPU resize does
            // the reduction and the GPU samples 1:1 (no bilinear-downscale moiré).
            // (Rotation-independent: a 90° turn swaps which screen edge the texture
            // height maps to, but the target works out to src_h × zoom either way.)
            return actual_target(
                self.cache.get(index).map(|t| t.src_h as f32),
                self.zoom,
                max_h,
                self.viewport.h,
                self.actual_cap_vh,
            );
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        let target = if self.scroll_mode {
            // Continuous strip: width-fit at width sw*zoom, height follows aspect.
            sw * self.zoom / aspect
        } else {
            // A page is drawn alone when layout is Single or it's a wide
            // (landscape) page that force-shows alone — only then does rotation
            // apply. `content_aspect` is the on-screen box's width/height.
            let single = self.layout == Layout::Single || aspect > 1.0;
            let rotated = single && self.rotation % 2 == 1;
            let content_aspect = if rotated {
                1.0 / aspect // rotated single page: box is the inverse of the source
            } else if self.layout == Layout::Spread && aspect <= 1.0 {
                // Pair two non-wide pages (assume a same-size facing page — exact
                // for uniform volumes; wide pages always show alone).
                aspect * 2.0
            } else {
                aspect
            };
            let box_h = fit_scale(self.fit, sw, sh, content_aspect, 1.0) * self.zoom;
            // Decode target = the texture height that draws 1:1. For a rotated
            // single page the texture's height lands along the screen *width*, so
            // the target is the box width (box_h * content_aspect); else box height.
            if rotated { box_h * content_aspect } else { box_h }
        };
        (target.round() as u32).clamp(MIN_TARGET, max_h)
    }

    /// Debounce the decode view. While the surface size or zoom is changing (a
    /// resize/zoom drag) the view is "unsettled" and `prefetch` won't re-decode
    /// cached pages for a target change — it just keeps showing the old textures.
    /// Once the value holds for a frame the view settles and stale pages re-decode
    /// in place (no black frame). Page-flipping leaves the view settled, so it
    /// never re-decodes.
    pub fn update_decode_view(&mut self) {
        let desired = (self.viewport.w, self.viewport.h, self.zoom);
        self.view_settled = desired == self.pending_view;
        self.pending_view = desired;
    }

    /// Recompute the prefetch window and hand it to the pool with each page's exact
    /// decode target. A page is queued if it's missing, or (once the view has
    /// settled) if its decoded target no longer matches its current exact target —
    /// then it re-decodes at the new resolution and overwrites in place.
    pub fn prefetch(&mut self) {
        let fwd = self.dynamic_fwd();
        let settled = self.view_settled;
        let Some(src) = &self.source else {
            return;
        };
        let len = src.len();
        // Shells call this every frame; skip the whole job-list rebuild when none
        // of its inputs changed since the last call (the queued jobs are still the
        // right desired set — workers drain them in place). Cache/failed changes
        // are visible through the epochs / length, so a landing page, a failure,
        // or a clear always recomputes.
        let key = JobsKey {
            index: self.index,
            len,
            fwd,
            viewport: (self.viewport.w, self.viewport.h),
            zoom_bits: self.zoom.to_bits(),
            settled,
            fit: self.fit,
            layout: self.layout,
            spread_offset: self.spread_offset,
            rotation: self.rotation,
            scroll_mode: self.scroll_mode,
            cache_epoch: self.cache.epoch(),
            failed_len: self.failed.len(),
        };
        if self.last_jobs_key == Some(key) {
            return;
        }
        self.last_jobs_key = Some(key);
        // Two-tier: if the page we're on isn't decoded yet (we've outrun the HQ
        // decode), decode this window with the cheap LQ resize so it appears
        // immediately. Once the anchor is cached the next prefetch wants HQ and
        // upgrades the LQ pages in place. (Pages prefetched HQ-ahead stay HQ — no
        // LQ flash when the buffer keeps up.)
        let anchor = layout::view_pages(self.layout, self.index, len, self.spread_offset).0;
        let lq = self.two_tier && !self.cache.contains(anchor);
        let window = desired_window(self.index, len, fwd, self.back);
        let jobs: Vec<(usize, u32, bool, bool)> = window
            .iter()
            .copied()
            .filter(|i| !self.failed.contains_key(i))
            .filter_map(|i| {
                let want = self.page_target_h(i);
                match self.cache.get(i) {
                    None => Some((i, want, lq, false)),
                    Some(p) => {
                        let target_stale = settled && p.target_h != want;
                        let quality_stale = !lq && p.lq; // have LQ, now want HQ
                        (target_stale || quality_stale).then_some((i, want, lq, false))
                    }
                }
            })
            .collect();
        if let Some(pool) = &self.pool {
            pool.set_jobs(jobs);
        }
        // Whole-volume LQ tier: a lowest-priority queue of tiny thumbnail jobs for
        // every page not in the HQ window, not already full-res cached, and not
        // already thumbnailed — nearest-first so a scrub finds previews sooner. It
        // lives in the pool (a second queue the workers only touch when the window
        // queue is empty), so it does *not* need rebuilding whenever the window
        // changes: pages that land a real texture are cancelled individually
        // (`drain_pool` → `cancel_lq`) and the list self-empties as it is consumed.
        //
        // The O(volume log volume) rebuild therefore runs once per `LQ_TAIL_RESTRIDE`
        // pages of travel — enough to re-centre "nearest-first" on where the reader
        // actually is — instead of once per landed page, which is what made prefetch
        // O(volume) per frame during a fill. Stops once `lq_cache` is full, so a
        // volume larger than `lq_cap` doesn't churn.
        //
        // The device's `lq_tier` decides how much of the volume is eligible at all
        // (`lq_tail_range`): the whole thing on desktop/High, a ±n band around the
        // reader on the Low tier — which the restride re-centres as it travels — or
        // nothing when the tier is off.
        if tail_needs_restride(self.lq_tail_pivot, self.index)
            && self.lq_cache.len() < self.lq_cache.cap()
            && let Some(range) = lq_tail_range(self.lq_tier, self.index, len)
        {
            let in_window: std::collections::HashSet<usize> = window.iter().copied().collect();
            let mut tail: Vec<usize> = range
                .filter(|i| {
                    !in_window.contains(i)
                        && !self.cache.contains(*i)
                        && !self.lq_cache.contains(*i)
                        && !self.failed.contains_key(i)
                })
                .collect();
            let cur = self.index as i64;
            tail.sort_by_key(|&i| (i as i64 - cur).abs());
            // Hand the pool no more thumbs than the cache has room for: the tail is
            // consumed to completion once queued, so on a volume larger than `lq_cap`
            // an untruncated tail would keep decoding thumbs whose inserts just evict
            // each other. Nearest-first order means the truncation keeps exactly the
            // pages a scrub is most likely to preview; a later restride re-fills from
            // whatever room the cache has then.
            tail.truncate(self.lq_cache.cap().saturating_sub(self.lq_cache.len()));
            if let Some(pool) = &self.pool {
                pool.set_lq_tail(tail.into_iter().map(|i| (i, self.lq_thumb_h)).collect());
                self.lq_tail_pivot = Some(self.index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Budget;
    use super::{
        drag_commits, drag_dir, drag_resist, Direction, DRAG_COMMIT_FRAC, DRAG_FLICK_MIN_FRAC,
        DRAG_MAX_FRAC,
    };

    // Drag metaphor: pulling the page toward the "previous" edge advances.
    // LTR: swipe/drag left = next; RTL mirrors. Must match the shell's
    // historical swipe mapping (lib.rs handle_gesture).
    #[test]
    fn drag_dir_matches_swipe_mapping() {
        assert_eq!(drag_dir(Direction::Ltr, -100.0), 1, "LTR drag left = next");
        assert_eq!(drag_dir(Direction::Ltr, 100.0), -1, "LTR drag right = prev");
        assert_eq!(drag_dir(Direction::Rtl, 100.0), 1, "RTL drag right = next");
        assert_eq!(drag_dir(Direction::Rtl, -100.0), -1, "RTL drag left = prev");
    }

    #[test]
    fn drag_commit_thresholds() {
        let w = 1000.0;
        // Past the distance threshold commits regardless of velocity.
        assert!(drag_commits(w * (DRAG_COMMIT_FRAC + 0.01), 0.0, w));
        assert!(drag_commits(-w * (DRAG_COMMIT_FRAC + 0.01), 0.0, w));
        // Short of it, a slow release snaps back…
        assert!(!drag_commits(w * 0.10, 0.0, w));
        // …but a deliberate flick commits (fast, same sign, non-trivial travel).
        assert!(drag_commits(w * 0.10, 900.0, w));
        assert!(drag_commits(-w * 0.10, -900.0, w));
        // A flick *against* the drag direction must not commit.
        assert!(!drag_commits(w * 0.10, -900.0, w));
        // Micro-travel never commits, however fast (stray touches).
        assert!(!drag_commits(w * (DRAG_FLICK_MIN_FRAC - 0.01), 5000.0, w));
        // A deliberate reversal cancels even PAST the distance threshold —
        // pulling far right then backtracking left means "changed my mind".
        assert!(!drag_commits(w * 0.60, -400.0, w));
        assert!(!drag_commits(-w * 0.60, 400.0, w));
        // …but sub-threshold opposing jitter at release doesn't kill a real commit.
        assert!(drag_commits(w * 0.60, -100.0, w));
        // And a far pull released while still (or coasting forward) commits.
        assert!(drag_commits(w * 0.60, 0.0, w));
        assert!(drag_commits(w * 0.60, 300.0, w));
    }

    // The resistance curve: ~1:1 tracking at rest (no kink when the drag
    // starts), strictly increasing, saturating at DRAG_MAX_FRAC of the width —
    // a full-width finger travel lands essentially at the cap (Chunky feel).
    #[test]
    fn drag_resistance_saturates() {
        let w = 1000.0;
        let max = DRAG_MAX_FRAC * w;
        // Near rest the page tracks the finger almost exactly.
        assert!((drag_resist(1.0, w) - 1.0).abs() < 0.01);
        // Sign is preserved.
        assert!(drag_resist(-200.0, w) < 0.0);
        assert_eq!(drag_resist(200.0, w), -drag_resist(-200.0, w));
        // Strictly increasing but always under the cap…
        let mut prev = 0.0;
        for i in 1..=20 {
            let d = drag_resist(i as f32 * 100.0, w);
            assert!(d > prev && d < max, "d={d} prev={prev}");
            prev = d;
        }
        // …and a full-width travel gets within ~5% of it.
        assert!(drag_resist(w, w) > max * 0.94);
    }

    // The whole-volume thumbnail tail is rebuilt on a *stride*, not per landed
    // page: the O(volume log volume) scan runs once per `LQ_TAIL_RESTRIDE` pages of
    // travel in either direction. Travelling less than a stride must reuse the tail
    // already queued in the pool (that reuse is the whole point of Phase 2); a
    // cleared pivot (source/pool swap) always rebuilds.
    #[test]
    fn tail_rebuilds_once_per_stride_of_travel() {
        use super::{tail_needs_restride, LQ_TAIL_RESTRIDE};
        let pivot = Some(100);
        assert!(tail_needs_restride(None, 100), "no tail yet → build one");
        assert!(!tail_needs_restride(pivot, 100), "standing still never rebuilds");
        assert!(
            !tail_needs_restride(pivot, 100 + LQ_TAIL_RESTRIDE - 1),
            "31 pages of travel keeps the queued tail"
        );
        assert!(
            tail_needs_restride(pivot, 100 + LQ_TAIL_RESTRIDE),
            "32 pages of travel re-centres it"
        );
        // Symmetric: reading backwards re-centres the same way (and can't underflow).
        assert!(!tail_needs_restride(pivot, 100 - (LQ_TAIL_RESTRIDE - 1)));
        assert!(tail_needs_restride(pivot, 100 - LQ_TAIL_RESTRIDE));
        assert!(tail_needs_restride(Some(0), LQ_TAIL_RESTRIDE));
        assert!(!tail_needs_restride(Some(LQ_TAIL_RESTRIDE), 1));
    }

    // `JobsKey` guards the **HQ window** job list only. A landing whole-volume
    // thumbnail must not change it: the tail lives in the pool and is maintained on
    // its own stride, so rebuilding the job list for each of a volume's hundreds of
    // thumbnails was pure O(volume) waste per landed page. A landing full-res page
    // still must rebuild (it feeds target_stale / quality_stale / the two-tier
    // anchor). The unused `lq_epoch` parameter is the trip-wire: re-adding that
    // field to the key breaks this test.
    #[test]
    fn jobs_key_ignores_the_thumbnail_tier() {
        use super::{FitMode, JobsKey, Layout};
        let key = |cache_epoch: u64, _lq_epoch: u64| JobsKey {
            index: 12,
            len: 500,
            fwd: 16,
            viewport: (3840, 2160),
            zoom_bits: 1.0f32.to_bits(),
            settled: true,
            fit: FitMode::Window,
            layout: Layout::Single,
            spread_offset: 0,
            rotation: 0,
            scroll_mode: false,
            cache_epoch,
            failed_len: 0,
        };
        // `Layout` has no `Debug`, so compare with `==` rather than assert_eq!.
        assert!(key(7, 1) == key(7, 99), "a landing thumbnail must not rebuild the job list");
        assert!(key(7, 1) != key(8, 1), "a landing full-res page still rebuilds it");
    }

    // Desktop-class inputs reproduce the historical fixed budget exactly.
    #[test]
    fn budget_desktop_matches_legacy_fixed_values() {
        let b = Budget::derive(8192 / 16, 8); // 512 MB slice, 8 cores
        assert_eq!(b.workers, 8);
        assert_eq!(b.cache_cap, 48);
        assert_eq!(b.texpool_max, 24);
        assert_eq!(b.fwd, 16);
        assert_eq!(b.fwd_max, 40);
        assert_eq!(b.back, 6);
    }

    // A small Android-class heap scales every dimension down, but never below the
    // floors (so seeking still works on a tiny device).
    #[test]
    fn budget_small_heap_scales_down_with_floors() {
        let small = Budget::derive(192, 8); // ~192 MB app heap
        assert!(small.cache_cap < 48 && small.cache_cap >= 16, "{}", small.cache_cap);
        assert!(small.texpool_max < 24 && small.texpool_max >= 8);
        assert!(small.fwd < 16 && small.fwd >= 6);
        // Extreme floor: a tiny budget + few cores still yields a usable reader.
        let tiny = Budget::derive(8, 1);
        assert_eq!(tiny.workers, 2, "at least 2 workers");
        assert_eq!(tiny.cache_cap, 16, "cache floor");
        assert_eq!(tiny.texpool_max, 8, "texpool floor");
        assert_eq!(tiny.fwd, 6, "fwd floor");
        assert_eq!(tiny.back, 3, "back floor");
    }

    // The flagship rule, pinned: `DeviceTier::High` applies **no** ceilings, so a
    // High-tier budget is bit-identical to plain `derive` for every input — that is
    // what makes this whole feature a no-op on desktop and on fast phones.
    #[test]
    fn for_tier_high_is_exactly_derive() {
        use super::{Budget, DeviceTier};
        for mem in [8u64, 64, 192, 256, 384, 512, 1024, 8192] {
            for cpus in [1usize, 2, 4, 6, 8, 16] {
                assert_eq!(
                    Budget::for_tier(DeviceTier::High, mem, cpus),
                    Budget::derive(mem, cpus),
                    "High must not touch derive (mem {mem}, cpus {cpus})"
                );
            }
        }
    }

    // Low/Mid apply the ceiling table to a device whose *derived* budget is bigger
    // (the broken-probe case that started this: a 6 GB phone saturating every
    // desktop clamp). Note what a ceiling is not: a floor.
    #[test]
    fn for_tier_low_and_mid_cap_a_desktop_class_budget() {
        use super::{Budget, DeviceTier, LqTier};
        let desktop = Budget::derive(1024, 8); // saturates every `derive` clamp
        assert_eq!(desktop.workers, 8);
        assert_eq!(desktop.cache_cap, 48);

        let low = Budget::for_tier(DeviceTier::Low, 1024, 8);
        assert_eq!(
            (low.workers, low.cache_cap, low.texpool_max, low.fwd, low.fwd_max, low.back, low.lq_cap),
            (3, 20, 10, 6, 12, 3, 320)
        );
        assert_eq!(low.actual_cap_vh, Some(2));
        assert_eq!(low.lq_thumb_h, 360);
        assert_eq!(low.lq_tier, LqTier::Windowed(64));

        let mid = Budget::for_tier(DeviceTier::Mid, 1024, 8);
        assert_eq!(
            (mid.workers, mid.cache_cap, mid.texpool_max, mid.fwd, mid.fwd_max, mid.back, mid.lq_cap),
            (6, 32, 16, 10, 24, 4, 512)
        );
        assert_eq!(mid.actual_cap_vh, Some(4));
        assert_eq!(mid.lq_thumb_h, 540);
        assert_eq!(mid.lq_tier, LqTier::Full);
    }

    // Ceilings are `min`s: a device whose derived budget is already below the tier
    // ceiling keeps its (smaller) derived values — the tier must never *raise* a
    // budget onto hardware that couldn't derive it.
    #[test]
    fn for_tier_never_raises_a_smaller_derived_budget() {
        use super::{Budget, DeviceTier};
        for tier in [DeviceTier::Low, DeviceTier::Mid] {
            let tiny = Budget::derive(8, 2); // every floor: 2 workers, cache 16, fwd 6…
            let capped = Budget::for_tier(tier, 8, 2);
            assert_eq!(capped.workers, tiny.workers, "{tier:?}");
            assert_eq!(capped.cache_cap, tiny.cache_cap, "{tier:?}");
            assert_eq!(capped.texpool_max, tiny.texpool_max, "{tier:?}");
            assert_eq!(capped.fwd, tiny.fwd, "{tier:?}");
            assert_eq!(capped.back, tiny.back, "{tier:?}");
            assert!(capped.lq_cap <= tiny.lq_cap, "{tier:?}");
        }
    }

    // The mobile 1:1 clamp. Desktop (`None`) must stay exactly what the
    // `decode_target_matches_drawn_size*` invariant tests model: displayed height
    // when the page is decoded, `u32::MAX` (→ full native) when it isn't. A capped
    // device bounds both arms at `m × viewport.h`, which is the whole point — the
    // uncached arm's full-native decode is the ~96 MB spike on a 4K-ish page.
    #[test]
    fn actual_target_clamps_only_on_capped_devices() {
        use super::actual_target;
        let (max_h, vh) = (8192u32, 2400u32);

        // Desktop / High: no cap anywhere.
        assert_eq!(actual_target(None, 1.0, max_h, vh, None), u32::MAX, "uncached decodes full");
        assert_eq!(actual_target(Some(5200.0), 1.0, max_h, vh, None), 5200);
        assert_eq!(actual_target(Some(5200.0), 0.5, max_h, vh, None), 2600);

        // Low tier (2× viewport heights = 4800): the uncached full-native decode
        // becomes a bounded one, and a tall page at 1:1 is capped (GPU upscales).
        assert_eq!(actual_target(None, 1.0, max_h, vh, Some(2)), 4800);
        assert_eq!(actual_target(Some(5200.0), 1.0, max_h, vh, Some(2)), 4800);
        // …but a page that already fits under the cap is untouched: no clamp, so
        // the GPU still samples it 1:1.
        assert_eq!(actual_target(Some(5200.0), 0.5, max_h, vh, Some(2)), 2600);
        // Mid (4× = 9600) is above the GPU ceiling here, so `max_h` still wins.
        assert_eq!(actual_target(Some(9000.0), 1.0, max_h, vh, Some(4)), max_h);

        // Degenerate inputs can't produce an inverted clamp range (a panic).
        assert_eq!(actual_target(Some(1000.0), 0.0, max_h, 0, Some(2)), super::MIN_TARGET);
        assert_eq!(actual_target(None, 1.0, max_h, 1, Some(2)), super::MIN_TARGET);
    }

    // `LqTier` decides how much of a volume the thumbnail tail may cover: all of it
    // (desktop / Mid / High), a ±n band that the pivot restride re-centres as the
    // reader travels (Low), or none at all. The band is clamped to the volume at
    // both ends — the low end must not underflow, the high end must not run past
    // the last page.
    #[test]
    fn lq_tail_range_windows_the_thumbnail_tier() {
        use super::{lq_tail_range, LqTier};
        assert_eq!(lq_tail_range(LqTier::Full, 300, 500), Some(0..=499), "whole volume");
        assert_eq!(lq_tail_range(LqTier::Off, 300, 500), None, "no tail at all");

        let w = LqTier::Windowed(64);
        assert_eq!(lq_tail_range(w, 300, 500), Some(236..=364));
        assert_eq!(lq_tail_range(w, 10, 500), Some(0..=74), "clamped at the start");
        assert_eq!(lq_tail_range(w, 480, 500), Some(416..=499), "clamped at the end");
        assert_eq!(lq_tail_range(w, 5, 20), Some(0..=19), "a short volume is fully covered");
        // Travelling a restride (32 pages) shifts the band, so pages that were
        // outside it become eligible — this is what keeps a scrub previewable.
        let before = lq_tail_range(w, 300, 500).unwrap();
        let after = lq_tail_range(w, 332, 500).unwrap();
        assert!(!before.contains(&396) && after.contains(&396));

        // An empty volume has nothing to thumbnail under any tier (and `len - 1`
        // must not underflow).
        for t in [LqTier::Full, LqTier::Windowed(64), LqTier::Off] {
            assert_eq!(lq_tail_range(t, 0, 0), None, "{t:?}");
        }
    }

    // The cache cap is monotonic in the memory budget (more RAM never shrinks it).
    #[test]
    fn budget_cache_monotonic_in_memory() {
        let caps: Vec<usize> = [64u64, 128, 256, 384, 512, 4096]
            .iter()
            .map(|&m| Budget::derive(m, 8).cache_cap)
            .collect();
        assert!(caps.windows(2).all(|w| w[1] >= w[0]), "{caps:?}");
    }

    // A 2048-tall portrait page on a 4K (2160-tall) screen, fit-to-window and
    // height-constrained, is displayed at 2160 → ~105% of native, not 100%.
    #[test]
    pub fn anchor_native_scale_fit_to_window_reports_upscale() {
        let s = super::anchor_native_scale(
            super::FitMode::Window,
            (3840.0, 2160.0),
            (1448.0, 2048.0),
            2048.0,
            2048.0,
            1.0,
        );
        assert!((s - 2160.0 / 2048.0).abs() < 1e-4, "got {s}");
    }

    // 1:1 (Actual) at zoom 1 is exactly native: 100%.
    #[test]
    pub fn anchor_native_scale_actual_is_unity() {
        let s = super::anchor_native_scale(
            super::FitMode::Actual,
            (3840.0, 2160.0),
            (1448.0, 2048.0),
            2048.0,
            2048.0,
            1.0,
        );
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    // Proof of the single-resize invariant (page-flip path): a page decoded to
    // its per-page target (the displayed height `page_target_h` computes) is drawn
    // by `build_quads` at that *same* height, so the GPU sampler maps 1 texel : 1
    // pixel and adds no second resize. Checks the decode target and the draw size
    // agree across fit modes, aspects, zooms, and surface sizes.
    #[test]
    pub fn decode_target_matches_drawn_size() {
        use crate::page::{fit_scale, FitMode};
        for (sw, sh) in [(3840.0_f32, 2160.0_f32), (1920.0, 1080.0), (1600.0, 2560.0)] {
            for fit in [FitMode::Window, FitMode::Width, FitMode::Height] {
                for aspect in [0.5_f32, 0.69, 1.0, 1.5] {
                    for zoom in [0.1_f32, 0.5, 1.0] {
                        // Decode target = the page's displayed height (page_target_h).
                        let th = (fit_scale(fit, sw, sh, aspect, 1.0) * zoom).round().max(1.0);
                        let tw = (th * aspect).round().max(1.0);
                        // build_quads draws that decoded (tw x th) texture at height:
                        let drawn = th * fit_scale(fit, sw, sh, tw, th) * zoom;
                        assert!(
                            (drawn - th).abs() <= 2.0,
                            "fit {} a {aspect} z {zoom} {sw}x{sh}: drawn {drawn} vs texture {th}",
                            fit.label(),
                        );
                    }
                }
            }
        }
    }

    // Single-resize invariant under a 90°/270° turn: `page_target_h` swaps the
    // aspect and returns the on-screen box *width* as the decode target (texture
    // height), then `single_quad` turns the texture inside the rotated box. The
    // turned texture's height must still map 1 texel : 1 pixel along the screen
    // width — i.e. the rotated-draw fit scale stays ~1, so no second GPU resize.
    #[test]
    pub fn decode_target_matches_drawn_size_rotated() {
        use crate::page::{fit_scale, FitMode};
        for (sw, sh) in [(3840.0_f32, 2160.0_f32), (1920.0, 1080.0), (1600.0, 2560.0)] {
            for fit in [FitMode::Window, FitMode::Width, FitMode::Height] {
                for aspect in [0.5_f32, 0.69, 1.0, 1.5] {
                    for zoom in [0.1_f32, 0.5, 1.0] {
                        // page_target_h (rotated): content_aspect = 1/aspect, the
                        // target is the box width = box_h * content_aspect.
                        let box_h = fit_scale(fit, sw, sh, 1.0 / aspect, 1.0) * zoom;
                        let th = (box_h / aspect).round().max(1.0); // texture height (target)
                        let tw = (th * aspect).round().max(1.0); // texture width (source aspect)
                        // single_quad swaps (w,h) for the odd rotation: ew = th, eh = tw.
                        let s = fit_scale(fit, sw, sh, th, tw) * zoom;
                        let drawn_w = th * s; // screen width the turned texture's height fills
                        assert!(
                            (drawn_w - th).abs() <= 2.0,
                            "rot fit {} a {aspect} z {zoom} {sw}x{sh}: drawn_w {drawn_w} vs texture-h {th}",
                            fit.label(),
                        );
                    }
                }
            }
        }
    }

    // Single-resize invariant in 1:1 (Actual) fit when zoomed *out* (the fixed
    // path): the page targets its displayed native×zoom height, decodes to that,
    // and is drawn at the same size — so the GPU samples 1:1 and never bilinear-
    // downscales a full-res texture. (Surface size is irrelevant in 1:1.)
    #[test]
    pub fn decode_target_matches_drawn_size_actual_zoomed_out() {
        for (src_w, src_h) in [(1500.0_f32, 5200.0), (5200.0, 1500.0), (2048.0, 2048.0)] {
            let _ = src_w; // 1:1 sizes off src_h × zoom; width follows the same scale
            for zoom in [0.1_f32, 0.27, 0.5, 0.99] {
                // page_target_h (Actual): displayed height = src_h × zoom; target_dims
                // caps at the source height (no cap here since zoom < 1).
                let target = (src_h * zoom).round().max(1.0);
                let th = target.min(src_h); // decoded texture height
                let drawn = src_h * zoom; // single_quad draws the box at src_h × zoom
                let gpu_scale = drawn / th; // displayed ÷ decoded — must be ~1 (no resize)
                assert!(
                    (gpu_scale - 1.0).abs() <= 0.01,
                    "actual {src_w}x{src_h} z {zoom}: gpu_scale {gpu_scale} (drawn {drawn}, texture {th})",
                );
            }
        }
    }

    // A source taller than the *former* fixed 3840 cap, viewed below native, must
    // decode to its displayed height — capped only by the GPU's real max texture
    // size, aspect-aware so the width fits too — so the GPU samples 1:1. The old
    // 3840 cap forced a GPU upscale (e.g. a 5207px page at 80–90% → ↑1.08–1.22×),
    // which beats against the screentone → moiré. This models `page_target_h`'s cap.
    #[test]
    pub fn large_page_decodes_to_display_not_a_fixed_cap() {
        let max_dim = 8192u32; // default MAX_TEX_DIM (the GPU's real limit)
        for (src_w, src_h) in [(3600.0_f32, 5207.0), (5207.0, 3600.0), (4000.0, 6000.0)] {
            let aspect = src_w / src_h;
            let max_h = ((max_dim as f32 / aspect.max(1.0)).floor() as u32).max(super::MIN_TARGET);
            for zoom in [0.5_f32, 0.74, 0.8, 0.9, 1.0] {
                // page_target_h (Actual): displayed height = src_h × zoom, clamped to max_h.
                let target = ((src_h * zoom).round() as u32).clamp(super::MIN_TARGET, max_h);
                let th = (target as f32).min(src_h); // target_dims caps at source
                let tw = (th * aspect).round() as u32; // width follows source aspect
                let display = src_h * zoom; // single_quad (Actual) draws at src_h × zoom
                let gpu_scale = display / th;
                assert!(
                    (gpu_scale - 1.0).abs() <= 0.01,
                    "src {src_w}x{src_h} z {zoom}: gpu_scale {gpu_scale} (display {display}, tex {th})"
                );
                assert!(
                    th as u32 <= max_dim && tw <= max_dim,
                    "src {src_w}x{src_h} z {zoom}: texture {tw}x{th} exceeds GPU max {max_dim}"
                );
            }
        }
    }

    // Spine-shadow sign convention: the screen-left page shades its right edge
    // (positive), the screen-right page its left edge (negative). The shader's
    // `select` reads the sign, so a flip here would put both shadows on the outer
    // edges. Equal widths ⇒ a symmetric seam; degenerate widths ⇒ off.
    #[test]
    fn spine_uv_signs_and_sides() {
        let (l, r) = super::spine_uv(800.0, 800.0);
        assert!(l > 0.0, "screen-left page shades its right edge (+)");
        assert!(r < 0.0, "screen-right page shades its left edge (−)");
        assert!((l + r).abs() < 1e-6, "equal widths ⇒ symmetric seam");
        let (l, r) = super::spine_uv(0.0, 1.0);
        assert!(l == 0.0 && r == 0.0, "degenerate widths ⇒ no shadow");
    }

    // The width is a fraction of the page, clamped in device px so it neither
    // vanishes on a small page nor balloons when zoomed. Always ≤ the page (UV ≤ 1).
    #[test]
    fn spine_uv_px_clamps() {
        let (l, _) = super::spine_uv(4000.0, 4000.0);
        assert!((l - super::SPINE_MAX_PX / 4000.0).abs() < 1e-6, "wide page hits the px cap");
        let (l, _) = super::spine_uv(60.0, 60.0);
        assert!((l - super::SPINE_MIN_PX / 60.0).abs() < 1e-6, "narrow page hits the px floor");
        let (l, _) = super::spine_uv(800.0, 800.0);
        assert!((l - super::SPINE_FRAC).abs() < 1e-6, "mid page uses the fraction");
        for w in [2.0_f32, 50.0, 240.0, 800.0, 4000.0] {
            let (l, r) = super::spine_uv(w, w);
            assert!(l <= 1.0 && -r <= 1.0, "w {w}: uv width {l} exceeds the page");
        }
    }

    // Every quad is built by `quad_from_px`, which leaves the shadow off — so the
    // single, wide-joined-spread, scroll and hold-last paths are shadow-free by
    // construction (only `place_view`'s pair arm ever stamps the fields).
    #[test]
    fn quad_from_px_defaults_no_spine() {
        let q = super::quad_from_px(0, 7, 100.0, 20.0, 800.0, 1200.0, 1920.0, 1080.0, 0);
        assert_eq!(q.spine, 0.0);
        assert_eq!(q.spine_strength, 0.0);
    }

    // The fit-multiplier clamp maps to native bounds: far zoom-out hits the 5%
    // floor, far zoom-in hits the 20000% ceiling, mid values pass through.
    #[test]
    pub fn zoom_multiplier_clamps_to_native_bounds() {
        let base = 2160.0_f32 / 2048.0; // native scale at zoom = 1 (fit-to-window)
        let lo = super::clamp_zoom_multiplier(1e-6, base);
        assert!((lo * base - super::MIN_ZOOM_PCT).abs() < 1e-4, "lo eff {}", lo * base);
        let hi = super::clamp_zoom_multiplier(1e9, base);
        assert!((hi * base - super::MAX_ZOOM_PCT).abs() < 1e-2, "hi eff {}", hi * base);
        let mid = super::clamp_zoom_multiplier(1.0, base);
        assert!((mid - 1.0).abs() < 1e-6, "mid {mid}");
    }

    // The BandiView ladder: 5, 10..300 by 10, 320..500 by 20, 550..20000 by 50.
    #[test]
    pub fn zoom_ladder_shape() {
        let p = super::zoom_presets();
        assert_eq!(p.first().copied(), Some(5.0));
        assert_eq!(p.last().copied(), Some(20000.0));
        assert!(p.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
        for v in [10.0, 100.0, 300.0, 320.0, 500.0, 550.0, 20000.0] {
            assert!(p.contains(&v), "ladder missing {v}");
        }
        let idx = |v: f32| p.iter().position(|&x| x == v).unwrap();
        assert_eq!(p[idx(300.0) + 1], 320.0, "300 -> 320 (step 20)");
        assert_eq!(p[idx(500.0) + 1], 550.0, "500 -> 550 (step 50)");
    }

    // +/- step to the neighbouring fixed stop, clamping at the ends.
    #[test]
    pub fn zoom_stepping_fixed() {
        let p = super::zoom_presets();
        let up = |c: f32| super::next_zoom_preset(&p, c, true);
        let dn = |c: f32| super::next_zoom_preset(&p, c, false);
        assert_eq!(up(71.0), 80.0);
        assert_eq!(up(80.0), 90.0);
        assert_eq!(up(300.0), 320.0);
        assert_eq!(up(500.0), 550.0);
        assert_eq!(up(20000.0), 20000.0, "clamps at the top");
        assert_eq!(dn(5.0), 5.0, "clamps at the bottom");
        assert_eq!(dn(95.0), 90.0);
        assert_eq!(dn(320.0), 300.0);
        assert_eq!(dn(550.0), 500.0);
    }

    // A spliced fit-% (e.g. 71.34) becomes a reachable stop between fixed presets.
    #[test]
    pub fn zoom_stepping_dynamic_stop() {
        let ladder = vec![70.0, 71.34, 80.0, 90.0];
        assert_eq!(super::next_zoom_preset(&ladder, 70.0, true), 71.34);
        assert_eq!(super::next_zoom_preset(&ladder, 71.34, true), 80.0);
        assert_eq!(super::next_zoom_preset(&ladder, 71.34, false), 70.0);
    }

    // A name-only stub source: exercises the live-folder-refresh classifier without a
    // GPU (the real sources read pixels; here only the name list matters).
    struct NamesSource(Vec<String>);
    impl crate::source::PageSource for NamesSource {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn name(&self, i: usize) -> &str {
            &self.0[i]
        }
        fn read_page(&self, _: usize) -> std::io::Result<std::sync::Arc<Vec<u8>>> {
            unreachable!("classifier never reads pixels")
        }
    }

    // The classifier separates the cheap index-preserving cases (append / tail-trim)
    // from a mid-list change that shifts indices and needs a name-based remap.
    #[test]
    fn classify_refresh_distinguishes_append_trim_and_reorder() {
        use super::{classify_refresh, Refresh};
        let src = |names: &[&str]| NamesSource(names.iter().map(|s| s.to_string()).collect());
        let base = src(&["1.png", "2.png", "3.png"]);
        // Identical listing → nothing to do.
        assert_eq!(classify_refresh(&base, &src(&["1.png", "2.png", "3.png"])), Refresh::Same);
        // Pages appended at the end → existing indices unchanged.
        assert_eq!(
            classify_refresh(&base, &src(&["1.png", "2.png", "3.png", "4.png"])),
            Refresh::Prefix
        );
        // Pages trimmed from the end → still a prefix.
        assert_eq!(classify_refresh(&base, &src(&["1.png", "2.png"])), Refresh::Prefix);
        // A file inserted mid-list (natural sort places "1a" between "1" and "2").
        assert_eq!(
            classify_refresh(&base, &src(&["1.png", "1a.png", "2.png", "3.png"])),
            Refresh::Reorder
        );
        // A middle file removed → indices after it shift.
        assert_eq!(classify_refresh(&base, &src(&["1.png", "3.png"])), Refresh::Reorder);
    }
}

impl Reader {
    /// Flip one view in `dir`. Returns `true` if the position actually changed; at
    /// the first/last page it queues a toast and returns `false`.
    ///
    /// Seeking gates on the *preview*, not on HQ: while the current page has no
    /// texture at all — not even the whole-volume LQ thumbnail — and that thumbnail
    /// could still arrive, the flip is held, so a fast seek shows every page (as LQ)
    /// rather than skipping past undecoded ones. Once the LQ cache is warm this
    /// effectively never blocks (the replacement for the old step/jump toggle: LQ
    /// makes "see every page" automatic at LQ speed). Two escapes keep nav unstuck:
    /// a *failed* page (never cached) is allowed past, and a volume larger than the
    /// LQ cap (cache full → its tail can't be thumbnailed) advances freely there.
    pub fn step(&mut self, dir: i64) -> bool {
        self.step_styled(dir, 0.0)
    }

    /// Is the reader at the volume boundary in `dir` (last view for `dir > 0`,
    /// first view for `dir < 0`)? Distinguishes a real boundary from the other
    /// reasons `step` can return false (e.g. the LQ warm-up gate) — a shell uses
    /// this to offer "next/previous book" when a flip runs out of pages.
    pub fn at_edge(&self, dir: i64) -> bool {
        let Some(src) = &self.source else { return false };
        let len = src.len();
        if len == 0 {
            return false;
        }
        let next = if dir > 0 {
            layout::next_view(self.layout, self.index, len, self.spread_offset)
        } else {
            layout::prev_view(self.layout, self.index, len, self.spread_offset)
        };
        next == self.index
    }

    /// `step` with an explicit transition start offset: 0 for a tap flip; the
    /// dragged page displacement for a committed interactive drag, so the same
    /// slide+fade+blur animation picks up where the finger left the page.
    fn step_styled(&mut self, dir: i64, from_frac: f32) -> bool {
        let Some(src) = &self.source else { return false };
        let len = src.len();
        if len == 0 {
            return false;
        }
        let cur = layout::view_pages(self.layout, self.index, len, self.spread_offset).0;
        if self.page_texture(cur).is_none()
            && !self.failed.contains_key(&cur)
            && self.lq_cache.len() < self.lq_cache.cap()
        {
            return false;
        }
        let next = if dir > 0 {
            layout::next_view(self.layout, self.index, len, self.spread_offset)
        } else {
            layout::prev_view(self.layout, self.index, len, self.spread_offset)
        };
        if next != self.index {
            // Arm the page-flip animation: the current view slides+fades out over
            // the new one. Page-flip mode only (scroll has no discrete flip).
            if self.transition_enabled && !self.scroll_mode {
                // Suppress during rapid seeking: if the previous flip was within one
                // transition-length, snap instead. Overlapping animations look busy,
                // and a fast flurry of flips shouldn't surface the blur at all.
                let rapid = self
                    .nav_times
                    .back()
                    .is_some_and(|t| t.elapsed() < Duration::from_millis(TRANSITION_MS));
                if rapid {
                    self.transition = None; // clear any in-flight one too
                } else {
                    let (oa, ob) = layout::view_pages(self.layout, self.index, len, self.spread_offset);
                    let mut out_pages = vec![oa];
                    if let Some(b) = ob {
                        out_pages.push(b);
                    }
                    // Slide the outgoing page away from the tapped edge. Tapping left
                    // (or its keyboard equivalent) and tapping right go opposite ways;
                    // RTL inverts the on-screen sense of "forward", so XOR it in. Net:
                    // tap-left ⇒ slide right, tap-right ⇒ slide left, both directions.
                    let exit_right = (dir > 0) == (self.direction == Direction::Rtl);
                    self.transition = Some(PageTransition {
                        start: Instant::now(),
                        out_pages,
                        exit_right,
                        from_frac,
                    });
                }
            }
            self.nav_times.push_back(Instant::now());
            self.goto(next);
            true
        } else {
            // Nowhere to go — let the reader know why seeking did nothing.
            self.toast(if dir > 0 { "Last page" } else { "First page" });
            false
        }
    }

    /// Start or refresh an interactive page drag (the shell feeds the signed
    /// horizontal finger displacement each move). Page-flip mode only. A new
    /// drag snaps any in-flight flip animation — same rule as rapid `step`s.
    pub fn drag_update(&mut self, dx_px: f32) {
        if self.scroll_mode || self.source.is_none() {
            return;
        }
        self.transition = None;
        match &mut self.drag {
            Some(d) if d.settle.is_none() => d.dx = dx_px,
            _ => self.drag = Some(PageDrag { dx: dx_px, settle: None }),
        }
    }

    /// The finger lifted: commit the flip (far enough, or a deliberate flick —
    /// the outgoing page then continues from the dragged offset off-screen), or
    /// snap back. Returns whether the flip committed.
    pub fn drag_release(&mut self, velocity_px_s: f32) -> bool {
        let Some(d) = &self.drag else { return false };
        if d.settle.is_some() {
            return false;
        }
        let dx = d.dx;
        let w = self.viewport.w.max(1) as f32;
        let committed = drag_commits(dx, velocity_px_s, w) && {
            let dir = drag_dir(self.direction, dx);
            // The usual page-turn animation continues from where the page
            // actually sits — the resistance-damped displacement, not the raw
            // finger travel.
            let from_frac = drag_resist(dx, w).abs() / w;
            // At the first/last page (or while the step-gate holds) this returns
            // false and the drag falls through to the snap-back below.
            self.step_styled(dir, from_frac)
        };
        if committed {
            self.drag = None; // the armed transition takes over the animation
        } else if let Some(d) = &mut self.drag {
            d.settle = Some((d.dx, Instant::now()));
        }
        committed
    }

    /// Abort a drag without committing (e.g. a second finger landed — the
    /// gesture became a pinch): snap the page back to rest.
    pub fn drag_cancel(&mut self) {
        if let Some(d) = &mut self.drag
            && d.settle.is_none()
        {
            d.settle = Some((d.dx, Instant::now()));
        }
    }

    pub fn goto(&mut self, index: usize) {
        self.index = index;
        self.pan_x = 0.0;
        self.pan_y = 0.0; // start new page centered
        self.top_offset = 0.0;
        // The shell persists the read position (it owns the volume key + settings).
        self.prefetch();
    }

    /// Apply a freshly-rescanned listing for the *same* volume (live folder refresh).
    /// The new source replaces the current one while preserving the read position and
    /// the decoded caches **by filename**, so files added/removed/reordered on disk
    /// don't flash the page or jump the reader. Called by the shell when the folder
    /// watcher's debounced rebuild lands. No-ops if nothing actually changed.
    pub fn apply_refreshed_source(&mut self, new: Arc<dyn PageSource>) {
        let Some(old) = self.source.clone() else {
            self.source = Some(new); // nothing open yet — just adopt it
            self.invalidate_jobs();
            self.prefetch();
            return;
        };
        let (old_len, new_len) = (old.len(), new.len());
        // A rescan that finds no images (everything deleted) is ignored — keep the last
        // good listing on screen rather than emptying the reader (and guards `len - 1`).
        if new_len == 0 {
            return;
        }
        match classify_refresh(old.as_ref(), new.as_ref()) {
            // Watcher fired on an unrelated touch (mtime/attrs of a file we already
            // have) — the listing is identical, so there is nothing to do.
            Refresh::Same => {}
            // Append or tail-trim: existing indices are unchanged, so keep the cache and
            // any in-flight decodes and swap the listing in place — no pool rebuild, no
            // hitch while pages are landing during a live download.
            Refresh::Prefix => {
                if new_len < old_len {
                    self.cache.remap(|i| (i < new_len).then_some(i));
                    self.lq_cache.remap(|i| (i < new_len).then_some(i));
                    if self.index >= new_len {
                        self.index = new_len - 1;
                    }
                }
                if let Some(pool) = &self.pool {
                    pool.set_source(new.clone());
                }
                self.source = Some(new);
                // Appended/trimmed pages change what the thumbnail tail should hold,
                // and the old tail's indices were built against the old listing.
                self.invalidate_jobs();
                self.prefetch();
            }
            // A file landed/left mid-list, so indices shift. Re-key the read position and
            // both decoded caches by filename so the same page stays on screen, then
            // rebuild the pool so stale in-flight `Done{old_index}` results from the old
            // listing are dropped instead of landing at a now-different index.
            Refresh::Reorder => {
                let new_pos: HashMap<&str, usize> =
                    (0..new_len).map(|i| (new.name(i), i)).collect();
                let anchor =
                    layout::view_pages(self.layout, self.index, old_len, self.spread_offset).0;
                let anchor_name = old.name(anchor).to_string();
                self.cache.remap(|i| new_pos.get(old.name(i)).copied());
                self.lq_cache.remap(|i| new_pos.get(old.name(i)).copied());
                self.index = new_pos
                    .get(anchor_name.as_str())
                    .copied()
                    .unwrap_or_else(|| self.index.min(new_len - 1));
                self.failed.clear();
                let pool = DecodePool::new(
                    new.clone(),
                    self.device.clone(),
                    self.queue.clone(),
                    self.tex_pool.clone(),
                    self.workers,
                );
                // The rebuilt pool starts wakerless — hand it the shell's callback
                // or landings from here on would never schedule a frame.
                pool.set_waker(self.waker.clone());
                self.pool = Some(pool);
                self.source = Some(new);
                // The new pool's queues are empty (and indices shifted), so both the
                // job list and the thumbnail tail must be rebuilt from scratch.
                self.invalidate_jobs();
                self.prefetch();
            }
        }
    }

    /// Snap zoom to the next ladder stop above/below the current native %. The
    /// ladder mixes the fixed BandiView presets with this page's fit stops, so a
    /// step can land exactly on fit-to-window / fit-to-width.
    pub fn zoom_to_preset(&mut self, zoom_in: bool) {
        let cur = self.effective_zoom_pct();
        let mut label: Option<&'static str> = None;
        if self.anchor_scale().is_some() && cur > 0.0 {
            let ladder = self.zoom_ladder();
            let target = next_zoom_preset(&ladder, cur, zoom_in);
            // Tag the stop if it is this page's fit-to-window / fit-to-width level.
            if !self.scroll_mode {
                let near = |p: Option<f32>| p.is_some_and(|p| (p - target).abs() <= target * 1e-3);
                if near(self.fit_native_pct(FitMode::Window)) {
                    label = Some("Fit window");
                } else if near(self.fit_native_pct(FitMode::Width)) {
                    label = Some("Fit width");
                }
            }
            self.zoom *= target / cur; // rescale the fit-multiplier to hit target %
        } else {
            // Anchor not decoded yet: coarse step; the next press snaps once it lands.
            self.zoom *= if zoom_in { 1.25 } else { 1.0 / 1.25 };
        }
        self.clamp_zoom_native();
        self.clamp_pan();
        let pct = self.effective_zoom_pct();
        match label {
            // Fit label on its own line so the "Zoom %" line stays centered
            // (the toast is center-aligned), aligned across zoom levels.
            Some(l) => self.toast(format!("Zoom {pct:.2}%\n({l})")),
            None => self.toast(format!("Zoom {pct:.2}%")),
        }
    }

    pub fn scroll_by(&mut self, dy: f32) {
        let len = match &self.source {
            Some(s) => s.len(),
            None => return,
        };
        let before = self.index;
        let before_off = self.top_offset;
        self.top_offset += dy;
        self.normalize();
        if self.index != before {
            self.nav_times.push_back(Instant::now());
        } else if dy.abs() > 0.5 && (self.top_offset - before_off).abs() < 0.5 {
            // The strip didn't move despite a scroll — clamped at an end.
            if dy < 0.0 && self.index == 0 && self.top_offset <= 0.5 {
                self.toast("First page");
            } else if dy > 0.0 && self.index + 1 >= len {
                self.toast("Last page");
            }
        }
        self.prefetch();
    }

    /// Begin an inertial scroll glide at `velocity` px/sec (clamped). Driven by
    /// `fling_tick` each frame until it decays below `SCROLL_FLING_MIN_V` or hits an end.
    pub fn start_fling(&mut self, velocity: f32) {
        self.scroll_velocity = velocity.clamp(-SCROLL_FLING_MAX_V, SCROLL_FLING_MAX_V);
    }

    /// Stop any in-flight scroll glide (e.g. a finger touched down to catch it).
    pub fn stop_fling(&mut self) {
        self.scroll_velocity = 0.0;
    }

    /// Whether a scroll glide is currently active.
    pub fn flinging(&self) -> bool {
        self.scroll_velocity.abs() >= SCROLL_FLING_MIN_V
    }

    /// Advance the inertial scroll by `dt` seconds: move the strip, decay the
    /// velocity, and stop at the volume's ends. Returns whether the glide continues
    /// (so the shell knows to schedule another frame). No prefetch/toast here — the
    /// shell's per-frame render prefetches, and a per-frame toast would spam.
    pub fn fling_tick(&mut self, dt: f32) -> bool {
        if self.scroll_velocity.abs() < SCROLL_FLING_MIN_V {
            self.scroll_velocity = 0.0;
            return false;
        }
        let before = self.index;
        let before_off = self.top_offset;
        let dy = self.scroll_velocity * dt;
        self.top_offset += dy;
        self.normalize();
        self.scroll_velocity *= (-SCROLL_FLING_FRICTION * dt).exp();
        // Clamped at an end (the strip didn't move despite a push) → stop dead.
        if self.index == before && dy.abs() > 0.5 && (self.top_offset - before_off).abs() < 0.5 {
            self.scroll_velocity = 0.0;
            return false;
        }
        self.scroll_velocity.abs() >= SCROLL_FLING_MIN_V
    }
}

/// How a rescanned listing differs from the one currently open, compared by entry
/// name. Drives [`Reader::apply_refreshed_source`]'s cheap (index-preserving) vs.
/// careful (index-remapping) paths.
#[derive(PartialEq, Eq, Debug)]
enum Refresh {
    /// Identical set in the same order — nothing changed.
    Same,
    /// One name list is a prefix of the other: pages were only appended at, or
    /// trimmed from, the end, so existing indices are unchanged.
    Prefix,
    /// Names diverge before the shorter length — a file was inserted, removed, or
    /// renamed mid-list, shifting the indices after it.
    Reorder,
}

fn classify_refresh(old: &dyn PageSource, new: &dyn PageSource) -> Refresh {
    let (o, n) = (old.len(), new.len());
    for i in 0..o.min(n) {
        if old.name(i) != new.name(i) {
            return Refresh::Reorder;
        }
    }
    if o == n {
        Refresh::Same
    } else {
        Refresh::Prefix
    }
}

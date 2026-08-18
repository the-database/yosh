//! Touch gesture state machine — the one implementation both shells drive.
//!
//! Touch input used to be transliterated per platform (desktop `app.rs`, Android
//! `lib.rs`), which meant every physics fix had to be ported by hand and the two
//! drifted: the phone lacked the pan fling, the lift-off bounce guard, the floored
//! velocities and the zoomed-scroll routing. This module owns the whole machine —
//! finger bookkeeping, the lock thresholds, pinch-zoom, the release velocities and
//! the glide starts — mutating a `&mut Reader` directly, so a shell is a ~20-line
//! wrapper that maps its own event type in and its own side effects out.
//!
//! **Shell-specific decisions leave as events, never as branches in here.** A tap,
//! a committed flip and a commit-strength drag into the volume boundary mean
//! different things on each platform (the phone resets zoom and offers the next
//! book; the desktop keeps zoom on a flip and has no book prompt), so the machine
//! reports them and the shell decides. Everything else — what the reader does — is
//! identical by construction.
//!
//! The engine carries no windowing types, so [`Phase`] is our own; shells map
//! `winit::event::TouchPhase` onto it. `Instant`s are *injected* (`now`) rather
//! than sampled inside, so tests can drive a synthetic timeline.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::reader::{drag_commits, drag_dir, Reader};

/// A touch landing within this window after a drag release is the digitizer's
/// lift-off bounce (a phantom contact as the finger peels off the glass), not
/// deliberate input: a human re-tap takes ≳200 ms. Without this, panels that
/// bounce kill every fling at birth — "touch has no physics" (issue #9).
pub const BOUNCE_WINDOW: Duration = Duration::from_millis(150);
/// How long a suspected bounce contact may linger before it counts as a real
/// grab: past this the glides it caught stay caught (the user meant to stop the
/// strip), under it they are handed back.
const BOUNCE_REARM: Duration = Duration::from_millis(120);
/// Width of the release-velocity window: only samples this recent describe the
/// throw the finger actually made.
const SAMPLE_WINDOW_MS: u128 = 100;
/// Floor on the sample window's dt when computing a release velocity. Samples
/// processed in a burst (a stalled frame) compress the window's timestamps, and a
/// zeroed velocity would silently kill the fling — so clamp, never gate to zero.
const DT_FLOOR: f64 = 0.016;
/// Horizontal travel (fraction of the surface width) that locks a single-finger
/// move into a real drag rather than a tap's jitter. Doubles as the release's
/// micro-tap radius, on **both** axes — a tap is judged against one circle, not
/// against the (taller) lock rectangle, so a vertical scrub is never a tap.
const LOCK_FRAC_W: f64 = 0.015;
/// Vertical counterpart of [`LOCK_FRAC_W`], as a fraction of the surface height.
/// Smaller because a strip scroll should engage sooner than a page flip.
const LOCK_FRAC_H: f64 = 0.01;
/// Per-frame dt ceiling for the inertial glides: a stalled frame must not
/// teleport the strip by a whole second of velocity.
const TICK_DT_MAX: f32 = 0.05;

/// Touch phase, mirroring `winit::event::TouchPhase` without depending on winit
/// (the engine carries no windowing types). Shells map their own enum onto this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Start,
    Move,
    End,
    Cancel,
}

/// The per-event shell facts the machine can't know for itself.
pub struct GestureCtx {
    /// Raw surface width (physical px) — the desktop's `gpu.config`, Android's
    /// `app.config`. Deliberately **not** `reader.viewport`: the desktop viewport
    /// is inset by the pinned top bar, and the lock/tap thresholds below are
    /// calibrated against the whole screen.
    pub surface_w: f64,
    /// Raw surface height (physical px), same source as `surface_w`.
    pub surface_h: f64,
    /// Height (px) of chrome the page is drawn *below*, so the pinch focal math
    /// can run in content space. Desktop: `top_inset_px()`. Android: `0.0` (the
    /// page is full-screen there), which makes the two identical.
    pub top_inset: f64,
    /// egui claimed this event (a chrome widget is under the finger).
    pub egui_consumed: bool,
    /// The library grid is open. It is egui's: its own kinetic scrolling handles
    /// the finger, and `egui_consumed` lags a frame on a press, so gate on the
    /// view itself rather than on the flag.
    pub library_view: bool,
}

/// Something the shell — not the engine — has to decide.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GestureEvent {
    /// A near-stationary release at `(x, y)`: route it through the shell's own
    /// tap zones (edge flips, chrome toggles, double-tap fullscreen).
    Tap { x: f64, y: f64 },
    /// An interactive page drag committed its flip. The reader has already
    /// stepped; this is for post-flip shell bookkeeping (Android resets zoom and
    /// persists the position; the desktop deliberately ignores it, keeping the
    /// zoom/pan a click flip would keep).
    FlipCommitted,
    /// A commit-strength drag in `dir` that ran into the volume boundary: intent
    /// to leave the book. Android arms its next/prev-book prompt; the desktop has
    /// none and ignores it.
    EdgeDragRelease { dir: i64 },
}

/// What one touch event produced.
#[derive(Default)]
pub struct GestureResponse {
    /// Shell-side decisions, in the order they occurred (empty for all but tap /
    /// flip / boundary releases — the hot path allocates nothing).
    pub events: Vec<GestureEvent>,
    /// The reader moved (or a glide started) and the frame on screen is stale.
    /// **Load-bearing on Android**, whose event loop buys no frame per event; the
    /// desktop's redraw guard already covers everything, so it ignores this.
    pub redraw: bool,
}

/// Recent `(time, x, y)` pointer samples, for the release velocity of a throw.
/// Shared by the touch machine and the shells' mouse-drag arms so a strip thrown
/// with the mouse glides exactly like one thrown with a finger.
#[derive(Default)]
pub struct VelocityTracker {
    samples: VecDeque<(Instant, f64, f64)>,
}

impl VelocityTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the window (a fresh press starts a fresh throw).
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Record a position, dropping anything older than [`SAMPLE_WINDOW_MS`].
    pub fn push(&mut self, now: Instant, x: f64, y: f64) {
        self.samples.push_back((now, x, y));
        while self
            .samples
            .front()
            .is_some_and(|(t, ..)| now.duration_since(*t).as_millis() > SAMPLE_WINDOW_MS)
        {
            self.samples.pop_front();
        }
    }

    /// Velocity (px/s) of the release at `(x, y)`, measured against the *oldest*
    /// surviving sample — i.e. across the whole ~100 ms window, which smooths the
    /// digitizer's per-event noise. `(0, 0)` when nothing was sampled.
    pub fn velocity(&self, now: Instant, x: f64, y: f64) -> (f64, f64) {
        self.samples.front().map_or((0.0, 0.0), |(t0, x0, y0)| {
            let dt = now.duration_since(*t0).as_secs_f64().max(DT_FLOOR);
            ((x - x0) / dt, (y - y0) / dt)
        })
    }

    /// Number of live samples (diagnostics + tests).
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Active two-finger pinch, captured at start so each move can both scale
/// about — and pan with — the finger midpoint (zoom-to-focal-point).
#[derive(Clone, Copy)]
struct Pinch {
    dist0: f64,       // finger separation when the pinch began
    zoom0: f32,       // reader.zoom when it began
    pan0: (f32, f32), // reader.pan_x / pan_y when it began
    mid0: (f64, f64), // finger midpoint (screen px) when it began
}

/// The touch state machine. One per shell; feed it every touch event and tick it
/// once per frame.
pub struct TouchGestures {
    /// Active touch points by finger id, for swipe / pinch-zoom / pan.
    touches: HashMap<u64, (f64, f64)>,
    /// Single-finger gesture start (for swipe-vs-tap on release).
    gesture_start: Option<(f64, f64)>,
    /// A single-finger move locked in as an interactive page drag (the page
    /// follows the finger; the engine renders it via `Reader::drag_update`).
    /// Locks once the motion is clearly horizontal; cleared on release/pinch.
    page_drag: bool,
    /// A single-finger drag locked in as continuous scroll (scroll mode).
    scroll_drag: bool,
    /// A single-finger drag locked in as a pan of an overflowing/zoomed page
    /// (page-flip mode) — release throws the page (`Reader::start_pan_fling`).
    pan_drag: bool,
    /// Recent samples of the dragging finger (or, from a shell's mouse arm, the
    /// pointer): x for the page-flip flick-to-commit and horizontal pan flings,
    /// y for the scroll and vertical pan flings.
    pub samples: VelocityTracker,
    /// Active pinch, captured at its start (see [`Pinch`]).
    pinch: Option<Pinch>,
    /// When the last drag (scroll, page or pan) released — the reference point
    /// for [`BOUNCE_WINDOW`] lift-off bounce detection.
    last_drag_release: Option<Instant>,
    /// The glides a suspected bounce contact interrupted: `(when it landed,
    /// scroll velocity, pan velocity)` at that moment. If the contact lifts
    /// again without becoming anything, the release hands them back.
    caught_fling: Option<(Instant, f32, (f32, f32))>,
    /// Last glide tick, for the per-frame dt of the inertial glides.
    last_fling_tick: Instant,
    /// Velocity (px/s) measured at the last touch release that started a glide,
    /// surfaced in the desktop info overlay so touch physics are diagnosable
    /// from a screenshot.
    pub last_touch_vy: Option<f32>,
}

impl Default for TouchGestures {
    fn default() -> Self {
        Self {
            touches: HashMap::new(),
            gesture_start: None,
            page_drag: false,
            scroll_drag: false,
            pan_drag: false,
            samples: VelocityTracker::new(),
            pinch: None,
            last_drag_release: None,
            caught_fling: None,
            last_fling_tick: Instant::now(),
            last_touch_vy: None,
        }
    }
}

impl TouchGestures {
    pub fn new() -> Self {
        Self::default()
    }

    /// Is any finger currently on the glass? Shells gate their mouse arms on this:
    /// Windows (and some Linux stacks) also deliver OS-synthesized *mouse* events
    /// for a touch, and acting on both would pan twice and fake a click on release.
    pub fn touch_active(&self) -> bool {
        !self.touches.is_empty()
    }

    /// Reset the tick clock so the next [`tick`](Self::tick) measures dt from
    /// *now*, not from whenever the last glide happened to stop. Called before
    /// every fling start in here, and by the shells' mouse-drag release arms.
    pub fn mark_fling_start(&mut self, now: Instant) {
        self.last_fling_tick = now;
    }

    /// Advance both inertial glides (strip scroll and 2-D pan) one frame.
    /// Returns whether either continues, i.e. whether the shell should schedule
    /// another frame. A no-op — two float compares — when nothing is gliding.
    pub fn tick(&mut self, reader: &mut Reader, now: Instant) -> bool {
        if !reader.flinging() && !reader.pan_flinging() {
            return false;
        }
        let dt = now.duration_since(self.last_fling_tick).as_secs_f32().clamp(0.0, TICK_DT_MAX);
        self.last_fling_tick = now;
        // Both are evaluated, never short-circuited: a diagonal throw in a zoomed
        // strip runs a scroll glide *and* a pan glide, and each has to be ticked.
        let scroll_on = reader.fling_tick(dt);
        let pan_on = reader.pan_fling_tick(dt);
        scroll_on || pan_on
    }

    /// Distance and midpoint (screen px) of the first two active touch points.
    /// Returning both from one call keeps them on the same finger pair.
    fn two_finger_metrics(&self) -> Option<(f64, f64, f64)> {
        let mut it = self.touches.values();
        let a = it.next()?;
        let b = it.next()?;
        let dist = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        Some((dist, (a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0))
    }

    /// Route a touch event into swipe/tap (one finger) or pinch-zoom/pan (two).
    // The arity is the event itself (phase, id, x, y, now) plus the two things it
    // acts on (reader, ctx); bundling them into a struct would only move the same
    // fields one level down at every call site.
    #[allow(clippy::too_many_arguments)]
    pub fn on_touch(
        &mut self,
        reader: &mut Reader,
        ctx: &GestureCtx,
        phase: Phase,
        id: u64,
        x: f64,
        y: f64,
        now: Instant,
    ) -> GestureResponse {
        let mut resp = GestureResponse::default();
        match phase {
            Phase::Start => {
                self.touches.insert(id, (x, y));
                // The library grid is egui's (see `GestureCtx::library_view`).
                // Only the touch map is kept up to date there.
                if ctx.library_view {
                    return resp;
                }
                if self.touches.len() == 1 {
                    self.gesture_start = Some((x, y));
                    self.page_drag = false;
                    self.scroll_drag = false;
                    self.pan_drag = false;
                    // A finger down catches any in-flight glide (strip or pan). But
                    // one landing faster after a drag release than a human can re-tap
                    // may be the digitizer's lift-off bounce — remember the glides
                    // so the release can hand them back if the contact comes to
                    // nothing (a press that lingers or travels is a real grab).
                    self.caught_fling = ((reader.flinging() || reader.pan_flinging())
                        && self
                            .last_drag_release
                            .is_some_and(|t| now.duration_since(t) < BOUNCE_WINDOW))
                    .then_some((now, reader.scroll_velocity, reader.pan_velocity));
                    reader.stop_fling();
                    reader.stop_pan_fling();
                    self.samples.clear();
                    // No redraw: the glide's own last tick already drew the position
                    // this stops at, so catching it changes nothing on screen.
                } else if self.touches.len() == 2 {
                    // Begin a pinch; cancel the single-finger gesture — including
                    // a live page drag, which snaps back.
                    self.gesture_start = None;
                    self.caught_fling = None; // two fingers are never a bounce
                    if self.page_drag {
                        self.page_drag = false;
                        reader.drag_cancel();
                        resp.redraw = true;
                    }
                    if let Some((d, mx, my)) = self.two_finger_metrics() {
                        self.pinch = Some(Pinch {
                            dist0: d,
                            zoom0: reader.zoom,
                            pan0: (reader.pan_x, reader.pan_y),
                            mid0: (mx, my),
                        });
                    }
                }
            }
            Phase::Move => {
                let prev = self.touches.insert(id, (x, y));
                if ctx.library_view {
                    return resp;
                }
                if let Some(p) = self.pinch {
                    // Pinch → zoom (engine re-decodes HQ once it settles), anchored
                    // to the finger midpoint so the content under the fingers stays
                    // put (and follows a two-finger drag). Deliberately *not* gated
                    // on `egui_consumed`: two fingers are never chrome interaction.
                    if p.dist0 > 1.0
                        && let Some((d, mx, my)) = self.two_finger_metrics()
                    {
                        // The page may be drawn below a top bar, so the focal math
                        // runs in content space: the reading viewport's size, and
                        // finger y measured from under the bar. A shell whose page
                        // is full-screen passes inset 0, which reduces this to the
                        // plain screen-space version.
                        let inset = ctx.top_inset;
                        let (sw, sh) = (reader.viewport.w as f32, reader.viewport.h as f32);
                        // Raw target from finger spread, then a fit "detent": within
                        // ONE gesture the zoom can approach the fit scale (zoom == 1.0)
                        // but not cross it, so a single max zoom-out from above — or
                        // zoom-in from below — lands exactly on fit. Crossing requires
                        // releasing and re-pinching (then zoom0 ~= 1.0, so no barrier).
                        // Barrier side comes from the immutable zoom0; we hard-clamp (no
                        // dist0 re-baseline) so the barrier holds — at the cost of a
                        // small dead zone if the user over-pinches past fit and reverses
                        // mid-gesture.
                        const FIT: f32 = 1.0;
                        const EPS: f32 = 0.001; // matches the fit-reset button's "fitted" test
                        let raw = p.zoom0 * (d / p.dist0) as f32;
                        reader.zoom = if p.zoom0 > FIT + EPS {
                            raw.max(FIT) // started above fit: can't drop below it this gesture
                        } else if p.zoom0 < FIT - EPS {
                            raw.min(FIT) // started below fit: can't rise above it this gesture
                        } else {
                            raw // started at fit: free to cross either way (re-pinch path)
                        };
                        reader.clamp_zoom_native();
                        // Actual (post-clamp) scale ratio: keep the content point
                        // under the initial midpoint pinned to the current one.
                        let k = reader.zoom / p.zoom0;
                        reader.pan_x =
                            mx as f32 - sw / 2.0 - k * (p.mid0.0 as f32 - sw / 2.0 - p.pan0.0);
                        reader.pan_y = (my - inset) as f32
                            - sh / 2.0
                            - k * ((p.mid0.1 - inset) as f32 - sh / 2.0 - p.pan0.1);
                        reader.clamp_pan();
                        resp.redraw = true;
                    }
                } else if self.touches.len() == 1 && !ctx.egui_consumed {
                    // Single finger over an overflowing/zoomed page in page-flip mode
                    // → pan, exactly like the mouse drag. The cost is that swipe-to-
                    // flip can't lock in while a page overflows — flips are edge taps
                    // there, same as with the mouse. (Scroll mode never pans here:
                    // vertical is the strip, horizontal is handled in its branch.)
                    if !reader.scroll_mode && (reader.zoom > 1.001 || reader.current_overflows()) {
                        // The pan itself sticks to the finger from the first move
                        // (micro-jitter pans invisibly, so taps survive the release's
                        // radius test); the lock only marks the gesture as a real pan,
                        // deciding fling-vs-tap at release.
                        if !self.pan_drag && let Some((sx, sy)) = self.gesture_start {
                            let w = ctx.surface_w;
                            let h = ctx.surface_h;
                            self.pan_drag = (x - sx).abs() > w * LOCK_FRAC_W
                                || (y - sy).abs() > h * LOCK_FRAC_H;
                        }
                        if self.pan_drag {
                            self.samples.push(now, x, y);
                        }
                        if let Some((px, py)) = prev {
                            reader.pan_x += (x - px) as f32;
                            reader.pan_y += (y - py) as f32;
                            reader.clamp_pan();
                            resp.redraw = true;
                        }
                    } else if let Some((sx, sy)) = self.gesture_start {
                        let (dx, dy) = (x - sx, y - sy);
                        if reader.scroll_mode {
                            // Scroll mode: a vertical drag scrolls the strip
                            // continuously (incremental, finger-tracking); when the
                            // strip is zoomed past the window width, horizontal motion
                            // pans it too (mirroring the mouse drag). Locks once the
                            // motion is clearly a drag so taps/seekbar survive.
                            if !self.scroll_drag {
                                let w = ctx.surface_w;
                                let h = ctx.surface_h;
                                self.scroll_drag = (dy.abs() > h * LOCK_FRAC_H
                                    && dy.abs() > dx.abs())
                                    || (reader.zoom > 1.001 && dx.abs() > w * LOCK_FRAC_W);
                            }
                            if self.scroll_drag {
                                // Sample for the release fling velocity (page-flip
                                // uses the same buffer for x, but the two modes are
                                // mutually exclusive per gesture).
                                self.samples.push(now, x, y);
                                if let Some((px, py)) = prev {
                                    reader.pan_x += (x - px) as f32;
                                    reader.top_offset -= (y - py) as f32;
                                    reader.clamp_pan();
                                    reader.normalize();
                                    resp.redraw = true;
                                }
                            }
                        } else {
                            // Page-flip: the page follows the finger (Chunky-style),
                            // the neighbor revealed underneath. Locks once the motion
                            // is clearly horizontal, so taps and the seekbar stay intact.
                            if !self.page_drag {
                                let w = ctx.surface_w;
                                self.page_drag =
                                    dx.abs() > w * LOCK_FRAC_W && dx.abs() > dy.abs();
                            }
                            if self.page_drag {
                                self.samples.push(now, x, y);
                                reader.drag_update(dx as f32);
                                resp.redraw = true;
                            }
                        }
                    }
                }
            }
            Phase::End | Phase::Cancel => {
                self.touches.remove(&id);
                if self.touches.len() < 2 {
                    self.pinch = None;
                }
                if self.touches.is_empty() {
                    let was_drag = std::mem::take(&mut self.page_drag);
                    let was_scroll = std::mem::take(&mut self.scroll_drag);
                    let was_pan = std::mem::take(&mut self.pan_drag);
                    let start = self.gesture_start.take();
                    if was_scroll || was_drag || was_pan {
                        // Reference point for lift-off bounce detection: a contact
                        // landing within BOUNCE_WINDOW of this is digitizer noise.
                        self.last_drag_release = Some(now);
                    }
                    if was_scroll {
                        // Scroll release → inertial fling from the recent velocity
                        // (vertical drives the strip; horizontal keeps a zoomed
                        // strip's sideways throw gliding too).
                        if phase != Phase::Cancel {
                            let (vx, vy) = self.samples.velocity(now, x, y);
                            self.mark_fling_start(now);
                            self.last_touch_vy = Some(vy as f32);
                            // strip velocity = −finger velocity (flick up → forward)
                            reader.start_fling(-vy as f32);
                            if reader.zoom > 1.001 {
                                reader.start_pan_fling(vx as f32, 0.0);
                            }
                            resp.redraw = true;
                        }
                        self.samples.clear();
                    } else if was_drag {
                        // The interactive drag owns this gesture end-to-end; the
                        // old end-of-gesture swipe must not also fire.
                        if phase == Phase::Cancel {
                            reader.drag_cancel();
                        } else {
                            // Release velocity from the ~100 ms sample window —
                            // decides flick-to-commit on short drags.
                            let (v, _) = self.samples.velocity(now, x, y);
                            if reader.drag_release(v as f32) {
                                // Committed. What that *means* is the shell's call:
                                // Android resets zoom + persists, the desktop keeps
                                // the view exactly as a click flip would.
                                resp.events.push(GestureEvent::FlipCommitted);
                            } else if let Some((sx, _)) = start {
                                // A commit-strength swipe into the volume boundary
                                // counts as intent to leave the book. (A reversal or
                                // a sub-threshold release does not — same rules as a
                                // real commit, via the shared `drag_commits`.)
                                let dxf = (x - sx) as f32;
                                let dir = drag_dir(reader.direction, dxf);
                                if drag_commits(dxf, v as f32, ctx.surface_w as f32)
                                    && reader.at_edge(dir)
                                {
                                    resp.events.push(GestureEvent::EdgeDragRelease { dir });
                                }
                            }
                        }
                        // Either way the page moves: a committed flip animates, a
                        // rejected one snaps back.
                        resp.redraw = true;
                        self.samples.clear();
                    } else if was_pan {
                        // Pan release → throw the page: a 2-D glide that clamps at
                        // the page edges. This is what makes panning around a huge
                        // page feel like the strip does (issue #9 follow-up).
                        if phase != Phase::Cancel {
                            let (vx, vy) = self.samples.velocity(now, x, y);
                            self.mark_fling_start(now);
                            self.last_touch_vy = Some(vx.hypot(vy) as f32);
                            // The page sticks to the finger, so the glide continues
                            // in the finger's own direction (no strip-style negation).
                            reader.start_pan_fling(vx as f32, vy as f32);
                            resp.redraw = true;
                        }
                        self.samples.clear();
                    } else if let Some((sx, sy)) = start {
                        // Nothing locked in: a near-stationary release is a tap;
                        // anything that travelled was a zoomed pan (or a vertical
                        // scrub) and is not. Unzoomed horizontal motion locks into the
                        // interactive drag long before reaching here, so flipping by
                        // swipe is handled by `drag_release` above.
                        let w = ctx.surface_w;
                        let micro = (x - sx).abs() < w * LOCK_FRAC_W
                            && (y - sy).abs() < w * LOCK_FRAC_W;
                        if let Some((t0, v, (pvx, pvy))) = self.caught_fling.take() {
                            // The contact landed moments after a drag release (see
                            // Phase::Start): a micro-contact that lifts right off again
                            // is the digitizer's lift-off bounce — hand back the glides
                            // it interrupted, and never treat it as a tap. A press
                            // that lingered is a real grab: the glides stay caught.
                            if micro && now.duration_since(t0) < BOUNCE_REARM {
                                self.mark_fling_start(now);
                                reader.start_fling(v);
                                reader.start_pan_fling(pvx, pvy);
                                resp.redraw = true;
                            }
                        } else if micro
                            && !ctx.library_view
                            && !ctx.egui_consumed
                            && self
                                .last_drag_release
                                .is_none_or(|t| now.duration_since(t) >= BOUNCE_WINDOW)
                        {
                            // The bounce guard doesn't slow rapid tap-tap flipping:
                            // a tap's own release never arms `last_drag_release`.
                            // No redraw — the shell's tap handler draws its own effect.
                            resp.events.push(GestureEvent::Tap { x: sx, y: sy });
                        }
                    }
                }
            }
        }
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::FitMode;
    use crate::reader::{Budget, DeviceTier, Direction, Viewport};
    use crate::texpool::TexturePool;
    use std::sync::Arc;

    const W: f64 = 1000.0;
    const H: f64 = 2000.0;

    // A name-only stub source: the gesture machine only ever needs `len()` (bounds
    // for `normalize`/`at_edge`), never pixels. Mirrors the one in reader.rs.
    struct NamesSource(Vec<String>);
    impl crate::source::PageSource for NamesSource {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn name(&self, i: usize) -> &str {
            &self.0[i]
        }
        fn read_page(&self, _: usize) -> std::io::Result<Arc<Vec<u8>>> {
            unreachable!("the gesture machine never reads pixels")
        }
    }

    /// A `Reader` on wgpu's `noop` backend: real device/queue handles (so the
    /// engine's types are constructed exactly as in production) with no GPU work
    /// behind them, which is all the input path needs — it moves numbers, never
    /// textures. `lq_cap = 0` disables the LQ warm-up gate in `step_styled`, which
    /// would otherwise refuse every flip here (no page ever decodes).
    fn test_reader(scroll_mode: bool, pages: usize) -> Reader {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut budget = Budget::for_tier(DeviceTier::High, 512, 4);
        budget.lq_cap = 0;
        let pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));
        let mut r = Reader::new(
            Arc::new(device),
            Arc::new(queue),
            pool,
            budget,
            FitMode::Window,
            crate::layout::Layout::Single,
            scroll_mode,
            Direction::Ltr,
            0,
            false,
        );
        r.source = Some(Arc::new(NamesSource(
            (0..pages).map(|i| format!("{i:03}.png")).collect(),
        )));
        r.viewport = Viewport { w: W as u32, h: H as u32 };
        r
    }

    fn ctx() -> GestureCtx {
        GestureCtx {
            surface_w: W,
            surface_h: H,
            top_inset: 0.0,
            egui_consumed: false,
            library_view: false,
        }
    }

    fn ms(t: Instant, n: u64) -> Instant {
        t + Duration::from_millis(n)
    }

    // --- 1. lock thresholds -------------------------------------------------

    // A horizontal move past 1.5% of the width locks the interactive page drag;
    // one under it stays a tap and surfaces as a Tap event on release.
    #[test]
    fn horizontal_lock_threshold_and_micro_tap() {
        let mut r = test_reader(false, 5);
        let mut g = TouchGestures::new();
        let (c, t0) = (ctx(), Instant::now());

        g.on_touch(&mut r, &c, Phase::Start, 1, 500.0, 1000.0, t0);
        // 14 px < 1000 × 0.015 = 15 px → no lock yet.
        let a = g.on_touch(&mut r, &c, Phase::Move, 1, 486.0, 1000.0, ms(t0, 10));
        assert!(!g.page_drag, "14 px must not lock the drag");
        assert!(!a.redraw);
        // 20 px > 15 px, and dominantly horizontal → locked.
        let b = g.on_touch(&mut r, &c, Phase::Move, 1, 480.0, 1000.0, ms(t0, 20));
        assert!(g.page_drag, "20 px of horizontal travel must lock the drag");
        assert!(b.redraw, "a live drag repaints");

        // A separate gesture that never leaves the tap radius is a tap.
        let mut g = TouchGestures::new();
        g.on_touch(&mut r, &c, Phase::Start, 2, 500.0, 1000.0, ms(t0, 500));
        g.on_touch(&mut r, &c, Phase::Move, 2, 505.0, 1004.0, ms(t0, 510));
        let up = g.on_touch(&mut r, &c, Phase::End, 2, 505.0, 1004.0, ms(t0, 520));
        assert_eq!(up.events, vec![GestureEvent::Tap { x: 500.0, y: 1000.0 }]);
        assert!(!up.redraw, "a tap leaves the redraw to the shell's tap handler");
    }

    // --- 2. zoomed-scroll routing -------------------------------------------

    // The Android bug this module fixes: while a scroll strip is zoomed, vertical
    // motion must still scroll it and horizontal motion must pan it — and the
    // release must throw both.
    #[test]
    fn zoomed_scroll_strip_scrolls_and_pans() {
        let mut r = test_reader(true, 10);
        r.zoom = 1.5;
        let mut g = TouchGestures::new();
        let (c, t0) = (ctx(), Instant::now());

        // Vertical: finger up → strip forward.
        g.on_touch(&mut r, &c, Phase::Start, 1, 500.0, 1000.0, t0);
        g.on_touch(&mut r, &c, Phase::Move, 1, 500.0, 940.0, ms(t0, 16));
        assert!(g.scroll_drag, "60 px of vertical travel locks the strip");
        assert!(r.top_offset > 0.0, "top_offset {}", r.top_offset);
        g.on_touch(&mut r, &c, Phase::Move, 1, 500.0, 880.0, ms(t0, 32));
        // Horizontal: pans the over-wide strip (this is what page-flip-only pan
        // handling used to swallow into `pan_y`, which scroll mode ignores).
        g.on_touch(&mut r, &c, Phase::Move, 1, 560.0, 880.0, ms(t0, 48));
        assert!(r.pan_x > 0.0, "pan_x {}", r.pan_x);

        let up = g.on_touch(&mut r, &c, Phase::End, 1, 560.0, 880.0, ms(t0, 64));
        assert!(up.redraw);
        assert!(r.flinging(), "the strip keeps gliding");
        assert!(r.pan_flinging(), "and so does the sideways throw");
        assert!(r.scroll_velocity > 0.0, "flick up glides forward");
    }

    // --- 3. pan release velocity + dt floor ---------------------------------

    // A pan thrown inside one frame's worth of time still flings: the dt floor
    // keeps the velocity finite instead of the old `dt > 0.005 → else 0.0` gate
    // silently zeroing it. The glide follows the finger, unnegated.
    #[test]
    fn pan_release_flings_finger_signed_with_dt_floor() {
        let mut r = test_reader(false, 5);
        r.zoom = 2.0; // page-flip + zoomed → the pan arm
        let mut g = TouchGestures::new();
        let (c, t0) = (ctx(), Instant::now());

        g.on_touch(&mut r, &c, Phase::Start, 1, 500.0, 1000.0, t0);
        // A burst: all three events land within 4 ms of each other.
        g.on_touch(&mut r, &c, Phase::Move, 1, 560.0, 1030.0, ms(t0, 2));
        assert!(g.pan_drag, "60 px locks the pan");
        g.on_touch(&mut r, &c, Phase::Move, 1, 620.0, 1060.0, ms(t0, 4));
        let up = g.on_touch(&mut r, &c, Phase::End, 1, 620.0, 1060.0, ms(t0, 6));

        assert!(up.redraw);
        let (vx, vy) = r.pan_velocity;
        assert!(vx.is_finite() && vy.is_finite());
        // (620 − 560)/0.016 = 3750 px/s, (1060 − 1030)/0.016 = 1875 px/s.
        assert!((vx - 3750.0).abs() < 1.0, "vx {vx}");
        assert!((vy - 1875.0).abs() < 1.0, "vy {vy}");
        assert!(r.pan_flinging());
        assert_eq!(g.last_touch_vy, Some(vx.hypot(vy)));
    }

    // --- 4. lift-off bounce -------------------------------------------------

    /// Drive a scroll drag + release, leaving the strip gliding and
    /// `last_drag_release` armed. Returns the release time.
    fn throw_strip(g: &mut TouchGestures, r: &mut Reader, c: &GestureCtx, t0: Instant) -> Instant {
        g.on_touch(r, c, Phase::Start, 1, 500.0, 1000.0, t0);
        g.on_touch(r, c, Phase::Move, 1, 500.0, 900.0, ms(t0, 16));
        g.on_touch(r, c, Phase::Move, 1, 500.0, 800.0, ms(t0, 32));
        g.on_touch(r, c, Phase::End, 1, 500.0, 800.0, ms(t0, 48));
        assert!(r.flinging(), "setup: the strip must be gliding");
        ms(t0, 48)
    }

    // A phantom contact right after a release hands the glide back and is never a tap.
    #[test]
    fn bounce_contact_restores_the_glide() {
        let mut r = test_reader(true, 10);
        let mut g = TouchGestures::new();
        let c = ctx();
        let rel = throw_strip(&mut g, &mut r, &c, Instant::now());
        let v = r.scroll_velocity;

        // Lands 20 ms later (inside BOUNCE_WINDOW) → the glide is caught, not lost.
        g.on_touch(&mut r, &c, Phase::Start, 2, 500.0, 800.0, ms(rel, 20));
        assert!(!r.flinging(), "the finger stops the strip while it is down");
        assert!(g.caught_fling.is_some());
        // …and lifts 10 ms after that without moving → bounce: restore, no tap.
        let up = g.on_touch(&mut r, &c, Phase::End, 2, 501.0, 800.0, ms(rel, 30));
        assert!(up.redraw);
        assert!(up.events.is_empty(), "a bounce is never a tap");
        assert_eq!(r.scroll_velocity, v, "the interrupted glide is handed back");
    }

    // A press that lingers past the re-arm window, or that travels, is a real grab.
    #[test]
    fn lingering_or_travelled_contact_keeps_the_glide_caught() {
        for (label, lift_at, x) in
            [("lingered", 200u64, 500.0f64), ("travelled", 30, 700.0)]
        {
            let mut r = test_reader(true, 10);
            let mut g = TouchGestures::new();
            let c = ctx();
            let rel = throw_strip(&mut g, &mut r, &c, Instant::now());

            g.on_touch(&mut r, &c, Phase::Start, 2, 500.0, 800.0, ms(rel, 20));
            let up = g.on_touch(&mut r, &c, Phase::End, 2, x, 800.0, ms(rel, lift_at));
            assert!(!r.flinging(), "{label}: the glide stays stopped");
            assert!(up.events.is_empty(), "{label}: and it is not a tap either");
        }
    }

    // --- 5. tap suppression -------------------------------------------------

    // A tap inside BOUNCE_WINDOW of a drag release is swallowed; one after it is
    // honoured; and taps themselves never arm the guard, so tap-tap flipping is
    // as fast as ever.
    #[test]
    fn tap_suppression_window_and_rapid_taps() {
        let mut r = test_reader(false, 5);
        let mut g = TouchGestures::new();
        let (c, t0) = (ctx(), Instant::now());

        // A page drag that snaps back (no glide, so no bounce capture).
        g.on_touch(&mut r, &c, Phase::Start, 1, 500.0, 1000.0, t0);
        g.on_touch(&mut r, &c, Phase::Move, 1, 470.0, 1000.0, ms(t0, 16));
        g.on_touch(&mut r, &c, Phase::End, 1, 470.0, 1000.0, ms(t0, 32));
        assert!(!r.flinging() && !r.pan_flinging());

        // 100 ms later: still inside the window → suppressed.
        g.on_touch(&mut r, &c, Phase::Start, 2, 200.0, 1000.0, ms(t0, 132));
        let early = g.on_touch(&mut r, &c, Phase::End, 2, 200.0, 1000.0, ms(t0, 140));
        assert!(early.events.is_empty(), "a bounce-window tap is digitizer noise");

        // 200 ms later: outside it → a real tap.
        g.on_touch(&mut r, &c, Phase::Start, 3, 200.0, 1000.0, ms(t0, 240));
        let late = g.on_touch(&mut r, &c, Phase::End, 3, 200.0, 1000.0, ms(t0, 248));
        assert_eq!(late.events, vec![GestureEvent::Tap { x: 200.0, y: 1000.0 }]);

        // Immediately again: a tap's own release never armed the guard.
        g.on_touch(&mut r, &c, Phase::Start, 4, 200.0, 1000.0, ms(t0, 260));
        let again = g.on_touch(&mut r, &c, Phase::End, 4, 200.0, 1000.0, ms(t0, 268));
        assert_eq!(again.events, vec![GestureEvent::Tap { x: 200.0, y: 1000.0 }]);
    }

    // --- 6. flip / boundary events ------------------------------------------

    /// Drag from mid-screen by `dx` px and release; returns the release response.
    fn swipe(g: &mut TouchGestures, r: &mut Reader, c: &GestureCtx, dx: f64) -> GestureResponse {
        let t0 = Instant::now();
        g.on_touch(r, c, Phase::Start, 1, 500.0, 1000.0, t0);
        g.on_touch(r, c, Phase::Move, 1, 500.0 + dx / 2.0, 1000.0, ms(t0, 40));
        g.on_touch(r, c, Phase::Move, 1, 500.0 + dx, 1000.0, ms(t0, 80));
        g.on_touch(r, c, Phase::End, 1, 500.0 + dx, 1000.0, ms(t0, 120))
    }

    // A far-enough drag commits, and says so exactly once.
    #[test]
    fn committed_drag_reports_flip_committed() {
        let mut r = test_reader(false, 5);
        let mut g = TouchGestures::new();
        let c = ctx();
        // LTR: dragging left (dx < 0) asks for the next page. 300 px = 30% > 25%.
        let up = swipe(&mut g, &mut r, &c, -300.0);
        assert_eq!(up.events, vec![GestureEvent::FlipCommitted]);
        assert_eq!(r.index, 1, "the reader actually stepped");
        assert!(up.redraw);
    }

    // Running the same swipe into the volume boundary reports the direction
    // instead — in both reading directions.
    #[test]
    fn edge_drag_release_reports_direction() {
        // LTR at the last page: swipe left = forward.
        let mut r = test_reader(false, 3);
        r.index = 2;
        let mut g = TouchGestures::new();
        let c = ctx();
        let up = swipe(&mut g, &mut r, &c, -300.0);
        assert_eq!(up.events, vec![GestureEvent::EdgeDragRelease { dir: 1 }]);
        assert_eq!(r.index, 2, "and the reader did not move");

        // RTL mirrors: forward is a swipe right.
        let mut r = test_reader(false, 3);
        r.direction = Direction::Rtl;
        r.index = 2;
        let mut g = TouchGestures::new();
        let up = swipe(&mut g, &mut r, &c, 300.0);
        assert_eq!(up.events, vec![GestureEvent::EdgeDragRelease { dir: 1 }]);

        // Backwards at the first page.
        let mut r = test_reader(false, 3);
        let mut g = TouchGestures::new();
        let up = swipe(&mut g, &mut r, &c, 300.0);
        assert_eq!(up.events, vec![GestureEvent::EdgeDragRelease { dir: -1 }]);
    }

    // A short, slow drag mid-volume just snaps back: no flip, no boundary.
    #[test]
    fn sub_threshold_drag_mid_volume_reports_nothing() {
        let mut r = test_reader(false, 5);
        r.index = 2;
        let mut g = TouchGestures::new();
        let c = ctx();
        // 40 px over 120 ms = 333 px/s: past the 15 px lock, under both the 25%
        // commit fraction and the 600 px/s flick.
        let up = swipe(&mut g, &mut r, &c, -40.0);
        assert!(up.events.is_empty(), "{:?}", up.events);
        assert_eq!(r.index, 2);
        assert!(up.redraw, "the snap-back still animates");
    }

    // --- 7. idle tick --------------------------------------------------------

    #[test]
    fn idle_tick_is_a_noop() {
        let mut r = test_reader(true, 5);
        r.index = 2;
        r.top_offset = 123.0;
        let mut g = TouchGestures::new();
        assert!(!g.tick(&mut r, Instant::now()));
        assert_eq!((r.index, r.top_offset), (2, 123.0));
        assert_eq!(r.pan_velocity, (0.0, 0.0));
    }

    // --- 8. VelocityTracker --------------------------------------------------

    #[test]
    fn velocity_tracker_trims_and_measures_from_the_oldest_sample() {
        let t0 = Instant::now();
        let mut v = VelocityTracker::new();
        assert_eq!(v.velocity(t0, 10.0, 10.0), (0.0, 0.0), "empty → no throw");

        v.push(t0, 0.0, 0.0);
        v.push(ms(t0, 50), 50.0, 0.0);
        assert_eq!(v.len(), 2);
        // 150 ms in, the t0 sample is older than the 100 ms window and is dropped.
        v.push(ms(t0, 150), 150.0, 0.0);
        assert_eq!(v.len(), 2, "the stale sample was trimmed");

        // Basis is the oldest survivor (t0+50 ms at x=50), not the previous push:
        // (250 − 50) / 0.150 s.
        let (vx, vy) = v.velocity(ms(t0, 200), 250.0, 0.0);
        assert!((vx - 200.0 / 0.150).abs() < 0.01, "vx {vx}");
        assert_eq!(vy, 0.0);

        v.clear();
        assert!(v.is_empty());
    }
}

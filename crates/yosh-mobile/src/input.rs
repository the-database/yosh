//! Touch, tap zones and hardware keys.

use super::*;

/// Tap-zone fractions, shared with `on_tap`: top strip opens the library, the side
/// edges flip.
pub(crate) const TOP_ZONE: f32 = 0.10;
pub(crate) const EDGE_ZONE: f32 = 0.20;

impl Shell {
    /// Route a touch event into the shared gesture machine, then apply the
    /// shell-specific decisions it hands back. Everything the reader does — locks,
    /// pinch, drag, the release velocities and glides — lives in the engine now;
    /// what a tap, a committed flip or a boundary swipe *mean* is decided here.
    ///
    /// Events that arrive before the first `resumed` (no `App` yet) are dropped:
    /// there is no reader to drive, and the finger can't have started a gesture on
    /// a window that doesn't exist.
    pub(crate) fn handle_touch(&mut self, phase: TouchPhase, id: u64, x: f64, y: f64, egui_consumed: bool) {
        let Some(app) = self.app.as_mut() else { return };
        let ctx = GestureCtx {
            surface_w: app.config.width as f64,
            surface_h: app.config.height as f64,
            // The page is drawn full-screen here (chrome floats over it), so the
            // pinch focal math needs no inset — which is exactly the desktop's
            // fullscreen case.
            top_inset: 0.0,
            egui_consumed,
            library_view: app.library_view,
        };
        let phase = match phase {
            TouchPhase::Started => Phase::Start,
            TouchPhase::Moved => Phase::Move,
            TouchPhase::Ended => Phase::End,
            TouchPhase::Cancelled => Phase::Cancel,
        };
        let resp =
            self.gestures.on_touch(&mut app.reader, &ctx, phase, id, x, y, Instant::now());
        // Load-bearing: this loop buys no frame per event, so a gesture that moved
        // the reader has to ask for one explicitly. Done inside the `app` borrow,
        // before the events (whose handlers need all of `self`).
        if resp.redraw {
            app.window.request_redraw();
        }
        for ev in resp.events {
            match ev {
                GestureEvent::Tap { x, y } => self.on_tap(x, y),
                GestureEvent::FlipCommitted => self.after_flip(),
                GestureEvent::EdgeDragRelease { dir } => self.boundary_hit(dir),
            }
        }
    }

    /// Flip `dir` pages and persist the new position. Running out of pages at a
    /// volume boundary arms (or confirms) the next/prev-book prompt instead.
    pub(crate) fn flip(&mut self, dir: i64) {
        let Some(app) = self.app.as_mut() else { return };
        if app.reader.step(dir) {
            self.after_flip();
        } else if self.app.as_ref().is_some_and(|a| a.reader.at_edge(dir)) {
            self.boundary_hit(dir);
        }
    }

    /// Hardware keys: an external keyboard, a Bluetooth page-turner remote (most
    /// send arrows, PageUp/PageDown or media next/previous; camera-shutter style
    /// ones send the volume keys) or the device's own volume rocker. Mirrors the
    /// desktop keymap: Down/PageDown/Space = forward, Up/PageUp = back,
    /// Left/Right follow the reading direction (page-flip only). Volume keys are
    /// always delivered here (`handle_volume_keys`), so any press that doesn't
    /// turn a page is handed back to the system volume.
    pub(crate) fn handle_key(&mut self, ev: &KeyEvent, egui_consumed: bool) {
        if ev.state != ElementState::Pressed {
            return;
        }
        let PhysicalKey::Code(code) = ev.physical_key else { return };
        let Some(app) = self.app.as_ref() else {
            // No reader yet: still let the rocker work.
            self.platform.unused_key(code);
            return;
        };
        let reading = !egui_consumed && self.current_key.is_some() && !app.library_view;
        let scroll = app.reader.scroll_mode;
        let rtl = app.reader.direction == Direction::Rtl;
        // `dir` is in story order (+1 = next page), like `flip`.
        let dir = match code {
            KeyCode::AudioVolumeDown | KeyCode::AudioVolumeUp if !reading => None,
            KeyCode::AudioVolumeDown => match app.vol_keys {
                VolKeys::Off => None,
                VolKeys::DownNext => Some(1),
                VolKeys::UpNext => Some(-1),
            },
            KeyCode::AudioVolumeUp => match app.vol_keys {
                VolKeys::Off => None,
                VolKeys::DownNext => Some(-1),
                VolKeys::UpNext => Some(1),
            },
            KeyCode::ArrowDown | KeyCode::PageDown | KeyCode::Space | KeyCode::MediaTrackNext => {
                Some(1)
            }
            KeyCode::ArrowUp | KeyCode::PageUp | KeyCode::MediaTrackPrevious => Some(-1),
            KeyCode::ArrowRight if !scroll => Some(if rtl { -1 } else { 1 }),
            KeyCode::ArrowLeft if !scroll => Some(if rtl { 1 } else { -1 }),
            _ => None,
        };
        let Some(dir) = dir.filter(|_| reading) else {
            self.platform.unused_key(code);
            return;
        };
        if scroll {
            if let Some(app) = self.app.as_mut() {
                let vh = app.reader.viewport.h as f32;
                app.reader.scroll_by(dir as f32 * vh * 0.9);
                app.window.request_redraw();
            }
            return;
        }
        // An armed next/prev-book prompt is modal, as for taps: the same direction
        // goes through `flip` (which runs into the edge and confirms it); any other
        // key just dismisses it.
        if let Some(app) = self.app.as_mut()
            && app.book_prompt.as_ref().is_some_and(|p| p.dir != dir)
        {
            app.book_prompt = None;
            app.window.request_redraw();
            return;
        }
        self.flip(dir);
    }

    /// Post-flip bookkeeping shared by tap flips and committed drags (the step
    /// already happened): reset the view to fit and persist the position.
    fn after_flip(&mut self) {
        if let Some(app) = self.app.as_mut() {
            app.reader.zoom = 1.0;
            app.reader.pan_x = 0.0;
            app.reader.pan_y = 0.0;
            app.window.request_redraw();
            let idx = app.reader.index;
            if let Some(k) = self.current_key.clone() {
                self.positions.insert(k, idx);
                self.mark_positions();
            }
        }
    }

    /// Tap-zones: top strip opens the library; thin side edges flip (direction-aware);
    /// the large center toggles the reading chrome (seekbar + hints).
    pub(crate) fn on_tap(&mut self, x: f64, y: f64) {
        let Some((w, h)) = self
            .app
            .as_ref()
            .map(|a| (a.config.width as f64, a.config.height as f64))
        else {
            return;
        };
        // When the library is open, every tap belongs to egui (the ✕ button closes
        // it). egui's "consumed" flag lags a frame on a touch press, so without this
        // a tap near the top would fall through here and toggle the library shut
        // before egui could register the row's click.
        if self.app.as_ref().map(|a| a.library_view).unwrap_or(false) {
            return;
        }
        // While the next/prev-book prompt is up it acts like a lightweight modal.
        // Tapping the card — or the same next/prev-page edge zone that armed it
        // (the one the card anchors to) — opens the sibling: the "tap again to
        // open" gesture. Any *other* tap just dismisses the prompt and is
        // swallowed (e.g. tapping the opposite edge dismisses instead of paging
        // back; center/top neither toggle chrome nor open the library).
        if self.app.as_ref().is_some_and(|a| a.book_prompt.is_some()) {
            let confirm_dir = self.app.as_ref().and_then(|app| {
                let p = app.book_prompt.as_ref()?;
                let rtl = app.reader.direction == Direction::Rtl;
                let on_left = book_prompt_on_left(p.dir, rtl);
                // The confirm edge is the one whose page-flip matches p.dir (left
                // in RTL for next / LTR for prev, mirrored) — the side the card
                // anchors to. Its whole vertical strip confirms, card or not.
                let on_edge = if on_left {
                    x < w * EDGE_ZONE as f64
                } else {
                    x > w * (1.0 - EDGE_ZONE as f64)
                };
                // Card rect: anchored 16pt in from that edge, ~centered vertically
                // (covers the card's center-side half that's past the edge zone).
                let ppp = app.egui_ctx.pixels_per_point() as f64;
                let half_w = (BOOK_PROMPT_W_PT as f64 / 2.0 + 12.0) * ppp;
                let half_h = 200.0 * ppp;
                let cx = if on_left {
                    (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                } else {
                    w - (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                };
                let on_card = (x - cx).abs() < half_w && (y - h / 2.0).abs() < half_h;
                ((on_card || on_edge)
                    && p.sibling.is_some()
                    && p.at.elapsed().as_millis() < BOOK_PROMPT_MS as u128)
                    .then_some(p.dir)
            });
            if let Some(dir) = confirm_dir {
                if let Some(app) = self.app.as_mut() {
                    app.book_prompt = None;
                }
                self.open_sibling_book(dir);
            } else if let Some(app) = self.app.as_mut() {
                app.book_prompt = None; // any other tap dismisses, swallowed
                app.window.request_redraw();
            }
            return;
        }
        if y < h * TOP_ZONE as f64 {
            // Top strip opens the library browser.
            self.open_library();
            return;
        }
        let rtl = self
            .app
            .as_ref()
            .map(|a| a.reader.direction == Direction::Rtl)
            .unwrap_or(false);
        // In scroll mode there are no discrete pages to flip, so the side edges fall
        // through to the center action (toggle the reading chrome).
        let scroll_mode = self.app.as_ref().map(|a| a.reader.scroll_mode).unwrap_or(false);
        if !scroll_mode && x < w * EDGE_ZONE as f64 {
            // Left edge: next in RTL, previous in LTR.
            self.flip(if rtl { 1 } else { -1 });
        } else if !scroll_mode && x > w * (1.0 - EDGE_ZONE as f64) {
            // Right edge: previous in RTL, next in LTR.
            self.flip(if rtl { -1 } else { 1 });
        } else if let Some(app) = self.app.as_mut() {
            // Center: toggle the reading chrome (also closes the options popup).
            // (An armed book prompt is handled by the gate above, so it never
            // reaches here.)
            app.controls = !app.controls;
            if app.controls {
                app.controls_shown_at = Instant::now(); // re-show the zone hints
            } else {
                app.show_options = false;
                app.show_info = false;
            }
            app.window.request_redraw();
        }
    }

    /// A next/prev action ran out of pages (`Reader::at_edge`). First hit arms
    /// the "Next/Previous book" prompt (resolving the sibling once and kicking
    /// its cover decode); a repeat in the same direction within the window opens
    /// it. (Tapping the card directly is handled by its egui button / `on_tap`.)
    fn boundary_hit(&mut self, dir: i64) {
        let confirm = self
            .app
            .as_ref()
            .and_then(|a| a.book_prompt.as_ref())
            .is_some_and(|p| {
                p.dir == dir
                    && p.sibling.is_some()
                    && p.at.elapsed().as_millis() < BOOK_PROMPT_MS as u128
            });
        if confirm {
            if let Some(app) = self.app.as_mut() {
                app.book_prompt = None;
            }
            self.open_sibling_book(dir);
            return;
        }
        // Cache-only lookup: a cold cache arms the card with no neighbour yet and
        // `poll_background` fills it in when the folder listing lands (the prompt is
        // on screen instantly either way).
        let sibling = match self.sibling_book_path(dir) {
            SibLookup::Found(p) => Some(p),
            SibLookup::Cold | SibLookup::Missing => None,
        };
        let title = sibling.as_deref().map(name_of).unwrap_or_default();
        if let Some(p) = &sibling {
            // Cover preview: same off-thread pipeline as the library; the result
            // lands via the per-frame `thumb_rx` drain (woken by the worker) and the
            // card picks it up while the prompt is showing.
            self.queue_cover(p);
        }
        if let Some(app) = self.app.as_mut() {
            app.book_prompt = Some(BookPrompt {
                dir,
                at: Instant::now(),
                sibling,
                title,
            });
            app.window.request_redraw();
        }
    }
}

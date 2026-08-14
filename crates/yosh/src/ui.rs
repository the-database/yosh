//! egui chrome: open buttons, reading-mode toggles, page indicator, and the
//! library browse grid. Toggle buttons + grid clicks set request flags the app
//! consumes after the egui frame.

use std::path::PathBuf;

use crate::library::{series_status, vol_state, LibCtx, Library, VolState};

#[derive(Default)]
pub struct UiState {
    /// Path the user requested to open this frame (consumed by the app).
    pub pending_open: Option<PathBuf>,
    /// Library root the user requested to open/scan this frame.
    pub pending_library: Option<PathBuf>,
    /// Currently-open volume path, for display.
    pub opened: Option<PathBuf>,
    pub status: String,

    // Current mode labels, set by the app each frame for the toggle buttons.
    pub dir_label: &'static str,
    pub fit_label: &'static str,
    pub layout_label: &'static str,
    /// Whether the page-turn transition is on (for the "Turn:" button label).
    pub transition_on: bool,
    /// Whether resume-last-book-on-startup is on. Set each frame; drives the
    /// Settings panel's Resume toggle.
    pub resume_on_startup: bool,
    // Current view state, set by the app each frame so the Settings panel can
    // highlight the active value in each control group.
    pub scroll_on: bool,
    pub dir_rtl: bool,
    pub layout_spread: bool,
    pub fit_mode: u8,
    pub rotation: u8,
    pub theme: crate::config::ThemePref,
    /// Whether a volume is currently loaded. Distinguishes the onboarding panel
    /// (nothing open) from the library grid; set by the app each frame.
    pub reader_open: bool,
    /// Whether a library root is configured (a grid is reachable). Drives the
    /// onboarding card's primary CTA; set by the app each frame.
    pub has_library_root: bool,

    // Requests raised by clicks, consumed by the app after the frame.
    pub req_toggle_dir: bool,
    pub req_cycle_fit: bool,
    pub req_toggle_layout: bool,
    pub req_toggle_transition: bool,
    pub req_toggle_resume: bool,
    pub req_toggle_library: bool,
    /// Whether the Settings panel window is open (toggled by the top-bar gear).
    pub settings_open: bool,
    // Settings-panel requests, drained by the app after the frame.
    pub req_toggle_scroll: bool,
    pub req_toggle_pairing: bool,
    pub req_rotate: bool,
    /// Direct fit selection (0..3) from the panel's radio group.
    pub req_set_fit: Option<u8>,
    /// Direct theme selection from the panel's radio group.
    pub req_set_theme: Option<crate::config::ThemePref>,
    /// Rescan the current library root (toolbar ⟳). Drained by the app.
    pub rescan: bool,
    /// Series whose section the user clicked to expand/collapse this frame
    /// (its `dir`). Drained by the app, which flips the persisted collapsed set.
    pub toggle_series: Option<PathBuf>,
    /// Volume paths of the series sections that were expanded *and* on screen this
    /// frame — the app decodes only these covers (lazy) and LRU-evicts the rest.
    pub visible_covers: Vec<PathBuf>,
    /// A library scan is in flight (off-thread) — show "Scanning library…". Set by
    /// the app each frame.
    pub scanning: bool,
    pub help_open: bool,
    /// Whether to draw the top chrome bar (hidden in fullscreen unless the
    /// cursor is at the top edge). Set by the app each frame.
    pub show_bar: bool,
    /// Measured height of the top bar in *physical* pixels, reported back by
    /// `chrome` each frame (0 when the bar isn't drawn). The app reserves this
    /// much space above the page so an opaque bar never covers it — the page is
    /// drawn by wgpu, not egui, so the panel's layout doesn't inset it on its own.
    pub bar_px: f32,
    /// Tab info overlay: whether it's shown, and its (label, value) lines (built
    /// by the app for the current page).
    pub info_open: bool,
    pub info: Vec<(String, String)>,
    /// Current zoom percent (native-relative), appended live to the info overlay
    /// (refreshed every frame so it tracks zooming without rebuilding page info).
    pub zoom_pct: f32,
    /// Live resize-pipeline readout for the in-view page (CPU path → GPU state),
    /// appended to the info overlay each frame so HQ vs. a stray GPU resize is
    /// always visible. Empty when nothing is decoded yet.
    pub resize_path: String,
    /// Transient toast message (boundary reached, zoom level); None when idle.
    pub toast: Option<String>,
    /// Whether to show the centered loading spinner (the current page's decode
    /// has been pending long enough to warrant feedback). Set by the app.
    pub loading: bool,
    /// `Some((file name, error reason))` when the current page's decode failed
    /// (show a notice instead of the spinner). Set by the app.
    pub failed: Option<(String, String)>,
    /// Edge navigation arrows: set true while the cursor hovers the left/right
    /// page-flip strip (page-flip reader mode only). Set by the app.
    pub hover_left: bool,
    pub hover_right: bool,
    /// Auto-update: the available newer version (None if up to date), whether an
    /// update is in progress / failed, and the click request. Set by the app.
    pub update_version: Option<String>,
    pub updating: bool,
    pub update_failed: bool,
    pub req_update: bool,

    /// Seekbar (bottom progress scrubber). Display fields set by the app each
    /// frame; `seek_request` is the page the user clicked/dragged to, drained
    /// by the app after the frame.
    pub seek_show: bool,
    pub seek_index: usize,
    pub seek_total: usize,
    pub seek_rtl: bool,
    pub seek_style: SeekbarStyle,
    pub seek_request: Option<usize>,
    /// Page indices currently buffered (decode-ahead ready set), painted as an
    /// mpv-style cache bar under the seekbar track. Refilled by the app each frame.
    pub seek_buffered: Vec<usize>,
    /// Page indices with an LQ preview thumbnail (the whole volume once warm),
    /// painted as a fainter wash beneath the HQ cache bar. Refilled each frame.
    pub seek_lq_buffered: Vec<usize>,

    /// Animation control panel (bottom-left), shown for animated GIF/WebP pages.
    /// Display fields set by the app each frame; the `anim_req_*` fields are
    /// drained by the app after the frame.
    pub anim_show: bool,
    /// True for an auto-playing animation (GIF/WebP) → show the play/pause button.
    /// False for `.ico` layers → step-only (no play/pause).
    pub anim_is_animation: bool,
    pub anim_playing: bool,
    pub anim_frame: usize,
    pub anim_total: usize,
    /// Set when the user clicks play/pause.
    pub anim_req_toggle_play: bool,
    /// −1 / +1 when the user clicks the step buttons (drained).
    pub anim_req_step: i32,
    /// Frame the user clicked on the animation progress bar (drained).
    pub anim_req_seek: Option<usize>,
    /// Set when the user hides the panel via its close button.
    pub anim_req_hide: bool,

    /// The yosh mascot logo (embedded PNG), lazily decoded the first time the
    /// onboarding panel renders and cached for the rest of the session.
    pub logo: Option<egui::TextureHandle>,
}

/// Which seekbar to draw. Only `Bar` is implemented today; the variant exists so
/// a richer (e.g. thumbnail) style can slot in behind a single `match` later.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SeekbarStyle {
    #[default]
    Bar, // BandiView-style track + handle
         // Thumbnails, // YACReader-style — future
}

fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}

/// Draw a translucent navigation chevron in a non-interactable, foreground Area
/// anchored to a window edge. Painted with line segments over a soft backdrop so
/// it stays visible on any page and doesn't depend on font glyph coverage.
fn nav_arrow(ctx: &egui::Context, id: &str, align: egui::Align2, offset: egui::Vec2, left: bool) {
    egui::Area::new(egui::Id::new(id))
        .anchor(align, offset)
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(54.0, 84.0), egui::Sense::hover());
            let p = ui.painter();
            let c = rect.center();
            p.circle_filled(c, 27.0, egui::Color32::from_black_alpha(96));
            let stroke = egui::Stroke::new(5.0_f32, egui::Color32::from_white_alpha(205));
            let (dx, dy) = (10.0, 18.0);
            let (tip, back) = if left { (-dx, dx) } else { (dx, -dx) };
            p.line_segment([c + egui::vec2(back, -dy), c + egui::vec2(tip, 0.0)], stroke);
            p.line_segment([c + egui::vec2(tip, 0.0), c + egui::vec2(back, dy)], stroke);
        });
}

/// BandiView-style seekbar: a translucent pill floating above the bottom edge
/// with a track, a filled progress segment, and a circular handle. Click or drag
/// anywhere to jump; hovering previews the target page as "page / total". The
/// horizontal axis follows the reading direction — in RTL, page 0 is at the right
/// edge and the last page at the left (progress flows right-to-left).
fn seekbar_bar(ctx: &egui::Context, st: &mut UiState) {
    if st.seek_total <= 1 {
        return; // nothing to scrub
    }
    let total = st.seek_total;
    let last = (total - 1) as f32; // >= 1, so no divide-by-zero
    let rtl = st.seek_rtl;
    let index = st.seek_index.min(total - 1);

    egui::Area::new(egui::Id::new("seekbar"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -16.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let bar_w = (ctx.content_rect().width() * 0.5).clamp(280.0, 640.0);
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(150))
                .inner_margin(egui::Margin::symmetric(16, 9))
                .corner_radius(egui::CornerRadius::same(10))
                .show(ui, |ui| {
                    // A tall hit-rect: the whole bar is clickable/draggable, so
                    // height is the vertical click target — keep it generous.
                    let (rect, resp) = ui
                        .allocate_exact_size(egui::vec2(bar_w, 30.0), egui::Sense::click_and_drag());

                    let r = 11.0_f32; // handle radius
                    let x0 = rect.left() + r; // handle-center span, inset by the radius
                    let x1 = rect.right() - r;
                    let span = (x1 - x0).max(1.0);
                    let cy = rect.center().y;

                    // Direction-aware mapping between a page fraction and an x.
                    let x_of = |frac: f32| if rtl { x1 - frac * span } else { x0 + frac * span };
                    let page_at = |px: f32| {
                        let t = ((px - x0) / span).clamp(0.0, 1.0); // LTR fraction
                        let frac = if rtl { 1.0 - t } else { t };
                        (frac * last).round() as usize
                    };
                    let handle_x = x_of(index as f32 / last);

                    let p = ui.painter();
                    p.line_segment(
                        [egui::pos2(x0, cy), egui::pos2(x1, cy)],
                        egui::Stroke::new(8.0_f32, egui::Color32::from_white_alpha(60)),
                    );
                    // mpv-style cache bar: a thin strip just under the track marking
                    // which pages are buffered (the decode-ahead ready set). The
                    // pipeline keeps more pages ahead than behind, so the band sits
                    // asymmetrically around the handle.
                    let half = (span / last * 0.5).max(0.75); // half a page-step wide
                    let (yt, yb) = (cy + 6.0, cy + 8.0);
                    // Faint wash: pages with an LQ preview thumbnail (the whole volume
                    // once warm). Same green as the HQ bar but barely-there, so it
                    // reads as a single cache bar with two intensities; the brighter
                    // HQ ticks below draw on top.
                    let lq_tick = egui::Color32::from_rgba_unmultiplied(120, 165, 140, 55);
                    for &i in &st.seek_lq_buffered {
                        if i >= total {
                            continue;
                        }
                        let xc = x_of(i as f32 / last);
                        let a = (xc - half).clamp(x0, x1);
                        let b = (xc + half).clamp(x0, x1);
                        p.rect_filled(
                            egui::Rect::from_min_max(egui::pos2(a, yt), egui::pos2(b, yb)),
                            0.0,
                            lq_tick,
                        );
                    }
                    if !st.seek_buffered.is_empty() {
                        // Muted + translucent so it reads as secondary info — never
                        // louder than the progress fill (alpha 190) or the handle.
                        let buf = egui::Color32::from_rgba_unmultiplied(120, 165, 140, 150);
                        for &i in &st.seek_buffered {
                            if i >= total {
                                continue;
                            }
                            let xc = x_of(i as f32 / last);
                            let a = (xc - half).clamp(x0, x1);
                            let b = (xc + half).clamp(x0, x1);
                            p.rect_filled(
                                egui::Rect::from_min_max(egui::pos2(a, yt), egui::pos2(b, yb)),
                                0.0,
                                buf,
                            );
                        }
                    }
                    let start_x = if rtl { x1 } else { x0 };
                    p.line_segment(
                        [egui::pos2(start_x, cy), egui::pos2(handle_x, cy)],
                        egui::Stroke::new(8.0_f32, egui::Color32::from_white_alpha(190)),
                    );
                    p.circle_filled(egui::pos2(handle_x, cy), r, egui::Color32::WHITE);
                    p.circle_stroke(
                        egui::pos2(handle_x, cy),
                        r,
                        egui::Stroke::new(2.0_f32, egui::Color32::from_black_alpha(120)),
                    );

                    // Click or drag (live scrub) jumps to the page under the pointer.
                    if (resp.clicked() || resp.dragged())
                        && let Some(pos) = resp.interact_pointer_pos()
                    {
                        st.seek_request = Some(page_at(pos.x));
                    }
                    // Hover previews the *target* page (what a click would jump
                    // to). Drawn as our own overlay rather than egui's hover
                    // tooltip so it appears instantly (no hover delay) and the
                    // "N / total" text always stays on a single line.
                    if let Some(pos) = resp.hover_pos() {
                        let text = format!("{} / {}", page_at(pos.x) + 1, total);
                        egui::Area::new(egui::Id::new("seekbar_tip"))
                            .order(egui::Order::Tooltip)
                            .interactable(false)
                            .fixed_pos(pos + egui::vec2(10.0, -30.0))
                            .show(ctx, |ui| {
                                egui::Frame::new()
                                    .fill(egui::Color32::from_black_alpha(220))
                                    .inner_margin(egui::Margin::symmetric(8, 4))
                                    .corner_radius(egui::CornerRadius::same(5))
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(text)
                                                    .color(egui::Color32::WHITE),
                                            )
                                            .wrap_mode(egui::TextWrapMode::Extend),
                                        );
                                    });
                            });
                    }
                });
        });
}

/// BandiView-style mini controls for the animation in view (GIF or WebP): a small
/// translucent panel in the bottom-left with play/pause, frame stepping, a
/// `frame / total` readout, a click-to-seek progress bar, and a close button.
/// Playback continues while hidden; the `G` key (or this button) toggles it.
fn anim_panel(ctx: &egui::Context, st: &mut UiState) {
    if st.anim_total <= 1 {
        return; // a single-frame page has nothing to control
    }
    let total = st.anim_total;
    let frame = st.anim_frame.min(total - 1);
    let last = (total - 1) as f32; // >= 1

    egui::Area::new(egui::Id::new("anim_panel"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(16.0, -16.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(160))
                .inner_margin(egui::Margin::symmetric(10, 6))
                .corner_radius(egui::CornerRadius::same(9))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.visuals_mut().override_text_color = Some(egui::Color32::from_white_alpha(235));
                    ui.horizontal(|ui| {
                        // Play/pause — only for an actual animation (GIF/WebP).
                        // `.ico` layers are static, so the button is omitted and the
                        // user just steps with < / > and the seek bar.
                        if st.anim_is_animation {
                            // A fixed-size button with a hand-painted icon (play
                            // triangle / pause bars). Same width in both states — no
                            // reflow on toggle — and independent of glyph fonts.
                            let bh = ui.spacing().interact_size.y;
                            let btn = ui.add_sized(egui::vec2(bh * 1.4, bh), egui::Button::new(""));
                            {
                                let c = btn.rect.center();
                                let col = egui::Color32::from_white_alpha(235);
                                let p = ui.painter();
                                if st.anim_playing {
                                    // Pause — two equal vertical bars.
                                    let bw = (bh * 0.15).max(2.5);
                                    let bar = egui::vec2(bw, bh * 0.5);
                                    let off = bw * 0.5 + bh * 0.07;
                                    for s in [-1.0_f32, 1.0] {
                                        p.rect_filled(
                                            egui::Rect::from_center_size(
                                                egui::pos2(c.x + s * off, c.y),
                                                bar,
                                            ),
                                            egui::CornerRadius::same(1),
                                            col,
                                        );
                                    }
                                } else {
                                    // Play — right-pointing triangle.
                                    let r = bh * 0.28;
                                    p.add(egui::Shape::convex_polygon(
                                        vec![
                                            egui::pos2(c.x - r * 0.7, c.y - r),
                                            egui::pos2(c.x - r * 0.7, c.y + r),
                                            egui::pos2(c.x + r, c.y),
                                        ],
                                        col,
                                        egui::Stroke::NONE,
                                    ));
                                }
                            }
                            if btn.on_hover_text(if st.anim_playing { "Pause" } else { "Play" }).clicked() {
                                st.anim_req_toggle_play = true;
                            }
                        }
                        if ui.button("<").on_hover_text("Previous frame").clicked() {
                            st.anim_req_step = -1;
                        }
                        ui.label(
                            egui::RichText::new(format!("{} / {}", frame + 1, total))
                                .monospace()
                                .color(egui::Color32::WHITE),
                        );
                        if ui.button(">").on_hover_text("Next frame").clicked() {
                            st.anim_req_step = 1;
                        }

                        // Click/drag the progress track to seek to a frame.
                        let (rect, resp) = ui
                            .allocate_exact_size(egui::vec2(96.0, 14.0), egui::Sense::click_and_drag());
                        let x0 = rect.left() + 3.0;
                        let x1 = rect.right() - 3.0;
                        let span = (x1 - x0).max(1.0);
                        let cy = rect.center().y;
                        let hx = x0 + (frame as f32 / last) * span;
                        {
                            let p = ui.painter();
                            p.line_segment(
                                [egui::pos2(x0, cy), egui::pos2(x1, cy)],
                                egui::Stroke::new(5.0_f32, egui::Color32::from_white_alpha(55)),
                            );
                            p.line_segment(
                                [egui::pos2(x0, cy), egui::pos2(hx, cy)],
                                egui::Stroke::new(5.0_f32, egui::Color32::from_white_alpha(200)),
                            );
                            p.circle_filled(egui::pos2(hx, cy), 5.0, egui::Color32::WHITE);
                        }
                        if (resp.clicked() || resp.dragged())
                            && let Some(pos) = resp.interact_pointer_pos()
                        {
                            let t = ((pos.x - x0) / span).clamp(0.0, 1.0);
                            st.anim_req_seek = Some((t * last).round() as usize);
                        }

                        if ui.button("×").on_hover_text("Hide (G)").clicked() {
                            st.anim_req_hide = true;
                        }
                    });
                });
        });
}

#[allow(deprecated)]
pub fn chrome(
    ctx: &egui::Context,
    st: &mut UiState,
    lib: &Library,
    libctx: &LibCtx,
    library_view: bool,
) {
    let show_bar = st.show_bar;
    let bar = egui::TopBottomPanel::top("top_bar").show_animated(ctx, show_bar, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Open folder…").clicked()
                && let Some(p) = rfd::FileDialog::new().set_title("Open page folder").pick_folder()
            {
                st.pending_open = Some(p);
            }
            if ui.button("Open file…").clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .set_title("Open comic archive or image")
                    .add_filter(
                        "Comics & images",
                        &[
                            "cbz", "cbr", "zip", "rar", "7z", "cb7", "png", "jpg", "jpeg", "webp",
                            "gif", "bmp", "avif", "jxl", "psd", "ico",
                        ],
                    )
                    .pick_file()
            {
                st.pending_open = Some(p);
            }
            // Going *to* the library only makes sense with a warm book; the way *back*
            // is the descriptive "Resume <book>" button on the right (added below),
            // mirroring the Android shell. (Setting the library lives in the library
            // view's "Change library…", not here — it's rare.)
            if st.reader_open && !library_view && ui.button("Library").clicked() {
                st.req_toggle_library = true;
            }
            ui.separator();
            // Frequently-changed, per-book view controls keep quick top-bar buttons;
            // set-and-forget options (theme / resume / page-turn / …) live behind ⚙ Settings.
            // They act on the open book, so they're hidden on the library grid.
            if !library_view {
                if ui.button(format!("Dir: {}", st.dir_label)).clicked() {
                    st.req_toggle_dir = true;
                }
                if ui.button(format!("Fit: {}", st.fit_label)).clicked() {
                    st.req_cycle_fit = true;
                }
                if ui.button(format!("Layout: {}", st.layout_label)).clicked() {
                    st.req_toggle_layout = true;
                }
            }
            if ui
                .button("⚙ Settings")
                .on_hover_text("Reading mode, direction, layout, fit, rotation, page-turn, resume, theme")
                .clicked()
            {
                st.settings_open = !st.settings_open;
            }
            if ui.button("? Help").clicked() {
                st.help_open = !st.help_open;
            }
            if let Some(v) = st.update_version.clone() {
                if st.updating {
                    ui.label(
                        egui::RichText::new("Updating…")
                            .color(egui::Color32::from_rgb(130, 200, 130)),
                    );
                } else if st.update_failed {
                    ui.label(
                        egui::RichText::new("Update failed")
                            .color(egui::Color32::from_rgb(220, 130, 130)),
                    )
                    .on_hover_text("Couldn't install the update — grab it from the releases page.");
                } else if ui
                    .button(
                        egui::RichText::new(format!("Update to v{v}"))
                            .color(egui::Color32::from_rgb(140, 225, 140)),
                    )
                    .on_hover_text(format!("Download yosh v{v} and restart"))
                    .clicked()
                {
                    st.req_update = true;
                }
            }
            ui.separator();
            if library_view {
                // The page indicator / volume name describe the open book, not the grid,
                // so on the library they're replaced by a one-click "Resume <book>" back
                // to the reader (shown only with a warm book — the library is the home,
                // so there's nothing to return to otherwise). Mirrors the Android shell.
                if st.reader_open {
                    let label = match &st.opened {
                        Some(p) => {
                            let name = p
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or_else(|| p.to_str().unwrap_or(""));
                            // Clip by char count (titles are often CJK) so a long name
                            // can't push the other header buttons off-screen.
                            let total = name.chars().count();
                            let clip: String = name.chars().take(24).collect();
                            let clip = if total > 24 { format!("{clip}…") } else { clip };
                            format!("{}  Resume {clip}", egui_phosphor::fill::BOOK_OPEN)
                        }
                        None => format!("{}  Resume", egui_phosphor::fill::BOOK_OPEN),
                    };
                    if ui.button(label).clicked() {
                        st.req_toggle_library = true;
                    }
                }
            } else {
                if !st.status.is_empty() {
                    ui.label(&st.status);
                    ui.separator();
                }
                match &st.opened {
                    Some(p) => {
                        let short = p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or_else(|| p.to_str().unwrap_or(""));
                        ui.label(short)
                    }
                    None => ui.label("no volume open"),
                };
            }
        });
    });
    // Report the bar's measured height back to the app, which reserves that much
    // space above the page. `show_animated` returns None while the panel is fully
    // collapsed, and animates the height while it opens/closes — the app only
    // *uses* this when the bar is pinned (windowed), so the animating value never
    // reaches the decode target. Physical px: the reader's viewport is physical.
    st.bar_px = bar.map_or(0.0, |b| b.response.rect.height() * ctx.pixels_per_point());

    if st.help_open {
        egui::Window::new("yosh — keys")
            .collapsible(false)
            .resizable(false)
            .open(&mut st.help_open)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(concat!("yosh ", env!("CARGO_PKG_VERSION")))
                        .color(egui::Color32::from_gray(140)),
                );
                ui.heading("Navigate");
                ui.label("← →   flip (reading-direction aware)");
                ui.label("↑ ↓ / Space / PgUp PgDn   flip");
                ui.label("Home / End   first / last page");
                ui.label("[  ]   previous / next book (folder or archive)");
                ui.label("click left/right edge — flip;   wheel — flip or pan");
                ui.label("double-click the middle — fullscreen");
                ui.separator();
                ui.heading("View presets");
                ui.label("9  fit window      8  fit width      0  100% (1:1)");
                ui.label("7  two-page  L→R   6  two-page  R→L (manga)");
                ui.separator();
                ui.heading("Layout / direction");
                ui.label("S   single ↔ two-page spread");
                ui.label("D   reading direction  RTL ↔ LTR");
                ui.label("C   continuous vertical scroll");
                ui.label("O   shift spread pairing (fix wrong pairing)");
                ui.separator();
                ui.heading("View");
                ui.label("+ / −   zoom;   drag — pan;   a preset key resets zoom");
                ui.label("R   rotate 90° (clockwise)");
                ui.label("I   show image info overlay");
                ui.label("B   toggle bottom seekbar");
                ui.label("T   page-turn transition (slide + fade on flip)");
                ui.label("G   show/hide the animation panel (animated GIF / WebP)");
                ui.label("F11   fullscreen      Esc   quit");
                ui.separator();
                ui.heading("Files");
                ui.label("E   show in Explorer (open the folder & select the file)");
                ui.label("Open folder / Open file;  Library ↔ Reader;  ⚙ Settings");
                ui.label("drag a folder, archive, or image onto the window");
                ui.label("F1   toggle this help");
            });
    }

    settings_window(ctx, st);

    if st.info_open && !library_view && !st.info.is_empty() {
        // Sit just under the bar, whatever height it actually is (it was a
        // hardcoded 44.0, which drifts with theme/DPI/font changes). `bar_px` is
        // physical; egui positions in points.
        let below_bar = st.bar_px / ctx.pixels_per_point() + 4.0;
        egui::Area::new(egui::Id::new("info_overlay"))
            .fixed_pos(egui::pos2(12.0, below_bar))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(190))
                    .inner_margin(egui::Margin::same(8))
                    .corner_radius(egui::CornerRadius::same(6))
                    .show(ui, |ui| {
                        egui::Grid::new("info_grid")
                            .num_columns(2)
                            .spacing([14.0, 3.0])
                            .show(ui, |ui| {
                                for (k, v) in &st.info {
                                    // An empty pair separates the two halves of a
                                    // spread — draw a rule rather than a blank row.
                                    if k.is_empty() && v.is_empty() {
                                        ui.separator();
                                        ui.separator();
                                        ui.end_row();
                                        continue;
                                    }
                                    ui.label(
                                        egui::RichText::new(k).color(egui::Color32::from_gray(150)),
                                    );
                                    ui.label(
                                        egui::RichText::new(v)
                                            .color(egui::Color32::WHITE)
                                            .monospace(),
                                    );
                                    ui.end_row();
                                }
                                // Live view state (refreshed every frame, not cached with the page).
                                ui.label(
                                    egui::RichText::new("Zoom").color(egui::Color32::from_gray(150)),
                                );
                                ui.label(
                                    egui::RichText::new(format!("{:.2}%", st.zoom_pct))
                                        .color(egui::Color32::WHITE)
                                        .monospace(),
                                );
                                ui.end_row();
                                if !st.resize_path.is_empty() {
                                    ui.label(
                                        egui::RichText::new("Resize")
                                            .color(egui::Color32::from_gray(150)),
                                    );
                                    ui.label(
                                        egui::RichText::new(&st.resize_path)
                                            .color(egui::Color32::WHITE)
                                            .monospace(),
                                    );
                                    ui.end_row();
                                }
                            });
                    });
            });
    }

    // Centered loading indicator: shown while the current page is still decoding
    // (e.g. seeking quickly through very high-resolution pages). The previous
    // page stays on screen beneath it; this just signals the next one is coming.
    if st.loading && !library_view {
        egui::Area::new(egui::Id::new("loading_overlay"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(180))
                    .inner_margin(egui::Margin::same(16))
                    .corner_radius(egui::CornerRadius::same(10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new().size(22.0));
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new("Loading…")
                                    .color(egui::Color32::WHITE)
                                    .size(15.0),
                            );
                        });
                    });
            });
    }

    // Centered failure notice: the current page's decode errored (unsupported or
    // corrupt). Replaces the spinner so a bad page doesn't appear to load forever.
    // Names the file and shows the decoder's error so it's clear which page / why.
    if let Some((name, reason)) = &st.failed
        && !library_view
    {
        egui::Area::new(egui::Id::new("failed_overlay"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(190))
                    .inner_margin(egui::Margin::same(16))
                    .corner_radius(egui::CornerRadius::same(10))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new(format!("Couldn't open {name}"))
                                    .color(egui::Color32::WHITE)
                                    .size(15.0),
                            );
                            ui.label(
                                egui::RichText::new(reason)
                                    .color(egui::Color32::from_gray(170))
                                    .size(12.0),
                            );
                        });
                    });
            });
    }

    // Transient toast (boundary reached, zoom level). Floats near the bottom,
    // above the seekbar's reveal zone; auto-cleared by the app after a moment.
    if let Some(msg) = &st.toast
        && !library_view
    {
        // Bottom-anchored, so extra lines (the "(Fit window)" suffix) grow upward
        // and would shove the first "Zoom %" line higher than a one-line toast.
        // Nudge the bubble down by one row per extra line so the first line stays
        // at a fixed height as you step through zoom levels.
        let extra = msg.matches('\n').count() as f32;
        let row_h = ctx.fonts_mut(|f| f.row_height(&egui::FontId::proportional(15.0)));
        egui::Area::new(egui::Id::new("toast"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -96.0 + extra * row_h))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(205))
                    .inner_margin(egui::Margin::symmetric(16, 9))
                    .corner_radius(egui::CornerRadius::same(8))
                    .show(ui, |ui| {
                        // Extend = never auto-wrap (only explicit '\n' breaks lines,
                        // e.g. the "(Fit window)" suffix), so egui can't newline the
                        // toast inconsistently. Center-align so the "Zoom %" line
                        // stays put across zoom levels.
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(msg).color(egui::Color32::WHITE).size(15.0),
                            )
                            .wrap_mode(egui::TextWrapMode::Extend)
                            .halign(egui::Align::Center),
                        );
                    });
            });
    }

    // Page-flip affordance: a chevron at whichever edge the cursor hovers.
    if st.hover_left {
        nav_arrow(ctx, "nav_arrow_left", egui::Align2::LEFT_CENTER, egui::vec2(20.0, 0.0), true);
    }
    if st.hover_right {
        nav_arrow(ctx, "nav_arrow_right", egui::Align2::RIGHT_CENTER, egui::vec2(-20.0, 0.0), false);
    }

    if st.seek_show {
        match st.seek_style {
            SeekbarStyle::Bar => seekbar_bar(ctx, st),
        }
    }

    if st.anim_show && !library_view {
        anim_panel(ctx, st);
    }

    if library_view {
        egui::CentralPanel::default().show(ctx, |ui| {
            library_sections(ui, st, lib, libctx);
        });
    }
}

/// The Settings panel (top-bar ⚙ button): a full mirror of the Android view-options
/// popup (`yosh-android` `options_popup`). Frequently-changed view controls
/// (direction/layout/fit) also have quick top-bar buttons; the set-and-forget ones
/// (page-turn / resume / theme) live only here. Each control sets a `req_*` flag that
/// the app drains after the frame. `.open` takes a local copy so the body can still
/// borrow `st` for the controls.
fn settings_window(ctx: &egui::Context, st: &mut UiState) {
    if !st.settings_open {
        return;
    }
    let mut open = true;
    egui::Window::new("⚙ Settings")
        .collapsible(false)
        .resizable(false)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);

            ui.label(egui::RichText::new("Reading mode").strong());
            ui.horizontal(|ui| {
                if ui.selectable_label(!st.scroll_on, "Page-flip").clicked() && st.scroll_on {
                    st.req_toggle_scroll = true;
                }
                if ui.selectable_label(st.scroll_on, "Scroll").clicked() && !st.scroll_on {
                    st.req_toggle_scroll = true;
                }
            });

            ui.label(egui::RichText::new("Reading direction").strong());
            ui.horizontal(|ui| {
                if ui.selectable_label(!st.dir_rtl, "→ LTR").clicked() && st.dir_rtl {
                    st.req_toggle_dir = true;
                }
                if ui.selectable_label(st.dir_rtl, "← RTL").clicked() && !st.dir_rtl {
                    st.req_toggle_dir = true;
                }
            });

            ui.label(egui::RichText::new("Page layout").strong());
            ui.horizontal(|ui| {
                if ui.selectable_label(!st.layout_spread, "Single").clicked() && st.layout_spread {
                    st.req_toggle_layout = true;
                }
                if ui.selectable_label(st.layout_spread, "Two-page").clicked() && !st.layout_spread {
                    st.req_toggle_layout = true;
                }
            });
            // Pairing parity only matters in a spread.
            if st.layout_spread
                && ui
                    .button("Shift page pairing")
                    .on_hover_text("Fix a mis-paired spread (key O)")
                    .clicked()
            {
                st.req_toggle_pairing = true;
            }

            ui.label(egui::RichText::new("Fit").strong());
            ui.horizontal(|ui| {
                for (i, text) in ["Window", "Width", "Height", "1:1"].iter().enumerate() {
                    if ui.selectable_label(st.fit_mode == i as u8, *text).clicked() {
                        st.req_set_fit = Some(i as u8);
                    }
                }
            });

            ui.label(egui::RichText::new("Rotation").strong());
            if ui
                .button(format!("Rotate 90° ⟳  ·  now {}°", st.rotation as u32 * 90))
                .clicked()
            {
                st.req_rotate = true;
            }

            ui.separator();

            ui.label(egui::RichText::new("Page-turn animation").strong());
            ui.horizontal(|ui| {
                if ui.selectable_label(st.transition_on, "On").clicked() && !st.transition_on {
                    st.req_toggle_transition = true;
                }
                if ui.selectable_label(!st.transition_on, "Off").clicked() && st.transition_on {
                    st.req_toggle_transition = true;
                }
            });

            ui.label(egui::RichText::new("Resume on startup").strong());
            ui.horizontal(|ui| {
                if ui.selectable_label(st.resume_on_startup, "On").clicked() && !st.resume_on_startup {
                    st.req_toggle_resume = true;
                }
                if ui.selectable_label(!st.resume_on_startup, "Off").clicked() && st.resume_on_startup {
                    st.req_toggle_resume = true;
                }
            });

            ui.label(egui::RichText::new("Theme").strong());
            ui.horizontal(|ui| {
                for (t, text) in [
                    (crate::config::ThemePref::System, "System"),
                    (crate::config::ThemePref::Light, "Light"),
                    (crate::config::ThemePref::Dark, "Dark"),
                ] {
                    if ui.selectable_label(st.theme == t, text).clicked() {
                        st.req_set_theme = Some(t);
                    }
                }
            });
        });
    st.settings_open = open;
}

/// First-run onboarding, rendered *inside* the library view when no library folder
/// is configured yet (the library is the home screen, so there's no separate "no
/// comic open" page). Centered mascot + the two ways to get going: pick a library
/// folder, or open a one-off file/folder. Once a library is set, the library shows
/// its grid / scanning / empty states instead (see `library_sections`).
fn onboarding(ui: &mut egui::Ui, st: &mut UiState) {
    // Lazily decode the mascot once, then reuse the texture for the session.
    if st.logo.is_none()
        && let Some(img) = decode_logo()
    {
        st.logo = Some(ui.ctx().load_texture("yosh_logo", img, egui::TextureOptions::LINEAR));
    }
    ui.add_space(ui.available_height() * 0.16);
    ui.vertical_centered(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 12.0);
        // The yosh mascot if it decoded; otherwise a book glyph.
        if let Some(tex) = &st.logo {
            let [w, h] = tex.size();
            let scale = 120.0 / h as f32;
            ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(
                w as f32 * scale,
                h as f32 * scale,
            )));
        } else {
            ui.label(egui::RichText::new("📖").size(56.0));
        }
        ui.label(egui::RichText::new("No comic open").strong().size(22.0));
        ui.label(
            egui::RichText::new("Set up your library, or open a file to start reading.")
                .size(14.0)
                .color(egui::Color32::from_white_alpha(160)),
        );
        ui.add_space(4.0);
        ui.spacing_mut().button_padding = egui::vec2(18.0, 10.0);
        if ui
            .add(egui::Button::new(
                egui::RichText::new("📚 Pick your comics folder…").size(18.0),
            ))
            .on_hover_text("Choose a folder of comics to browse as a library")
            .clicked()
            && let Some(p) = rfd::FileDialog::new()
                .set_title("Choose a library folder")
                .pick_folder()
        {
            st.pending_library = Some(p);
        }
        if ui
            .add(egui::Button::new(egui::RichText::new("Open file…").size(18.0)))
            .clicked()
            && let Some(p) = rfd::FileDialog::new()
                .set_title("Open comic archive or image")
                .add_filter(
                    "Comics & images",
                    &[
                        "cbz", "cbr", "zip", "rar", "7z", "cb7", "png", "jpg", "jpeg", "webp",
                        "gif", "bmp", "avif", "jxl", "psd", "ico",
                    ],
                )
                .pick_file()
        {
            st.pending_open = Some(p);
        }
        if ui
            .add(egui::Button::new(egui::RichText::new("Open folder…").size(18.0)))
            .clicked()
            && let Some(p) = rfd::FileDialog::new()
                .set_title("Open page folder")
                .pick_folder()
        {
            st.pending_open = Some(p);
        }
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("Press F1 for keys.")
                .size(12.0)
                .color(egui::Color32::from_white_alpha(140)),
        );
    });
}

/// Decode the embedded yosh mascot logo for the onboarding card. The PNG is
/// transparent; the engine's decode premultiplies its alpha, so build the egui
/// image from premultiplied bytes (`from_rgba_unmultiplied` would double-multiply
/// and darken the anti-aliased edges).
fn decode_logo() -> Option<egui::ColorImage> {
    const LOGO: &[u8] = include_bytes!("../assets/yosh.png");
    let mut resizer = fast_image_resize::Resizer::new();
    let decoded = yosh_engine::decode::decode_and_downscale(LOGO, 256, &mut resizer).ok()?;
    let rgba = yosh_engine::decode::to_rgba_image(decoded);
    Some(egui::ColorImage::from_rgba_premultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

/// Width of a cover cell and its image in the sectioned library row.
const CELL_W: f32 = 150.0;
const COVER_H: f32 = 210.0;

/// How many covers the "Recently read" shelf shows at most (newest first).
const RECENTS_SHELF: usize = 12;

/// The Chunky-style library: a vertical list of collapsible series sections, each
/// a horizontal, scrollbar/drag-scrollable row of cover thumbnails with read-state
/// visuals (faded "finished" covers, an in-progress bar, the open volume's stroke).
fn library_sections(ui: &mut egui::Ui, st: &mut UiState, lib: &Library, libctx: &LibCtx) {
    // No library folder configured yet → this home view IS the first-run onboarding
    // (no Rescan header, nothing to scan). Use the mirrored flag, not `lib.root`,
    // which is None mid-first-scan right after a folder is picked.
    if !st.has_library_root {
        onboarding(ui, st);
        return;
    }
    // Solid (non-floating) scrollbars so the horizontal bar under an overflowing row
    // is clearly visible and the vertical bar reserves a real strip on the right
    // (the default floating bars reserve 0 width and overlay the rows). Scoped to the
    // library central panel.
    ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
    // egui paints a fade-out gradient at scroll edges (default strength 0.5); on the
    // cover grid it dims the thumbnails at the screen bottom, reading as a stray drop
    // shadow anchored to the scrolling content. Turn it off. (Matches the Android fix.)
    ui.spacing_mut().scroll.fade.strength = 0.0;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui.button("⟳ Rescan").on_hover_text("Re-scan the library folder").clicked() {
            st.rescan = true;
        }
        // Set/change the library root lives here (rare action), not in the top bar —
        // mirrors the Android library's "Change library…".
        if ui.button("📂 Change library…").on_hover_text("Pick a different comics folder").clicked()
            && let Some(p) = rfd::FileDialog::new()
                .set_title("Choose a library folder")
                .pick_folder()
        {
            st.pending_library = Some(p);
        }
        if let Some(root) = &lib.root {
            ui.label(egui::RichText::new(root.to_string_lossy()).weak());
        }
    });
    ui.separator();

    if lib.is_empty() {
        ui.add_space(40.0);
        ui.vertical_centered(|ui| {
            if st.scanning {
                ui.add(egui::Spinner::new());
                ui.label("Scanning library…");
            } else {
                ui.label("No comics found here — use “Library…” to pick your comics folder.");
            }
        });
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        recents_row(ui, st, lib, libctx);
        for series in &lib.series {
            let key = series.dir.to_string_lossy();
            let expanded = !libctx.collapsed.contains(key.as_ref());

            // Per-volume read state + the series' aggregate status label.
            let states: Vec<VolState> = series
                .volumes
                .iter()
                .map(|v| {
                    vol_state(
                        libctx.progress,
                        libctx.last_pages,
                        v.path.to_string_lossy().as_ref(),
                    )
                })
                .collect();
            let status = series_status(&states);

            // Full-width clickable header: caret + name (left), status (right).
            // Built with a normal horizontal layout (left label + a right-to-left
            // sub-layout) so both ends lay out and clip reliably; the row response
            // is re-interacted for the click.
            let caret = if expanded {
                egui_phosphor::fill::CARET_DOWN
            } else {
                egui_phosphor::fill::CARET_RIGHT
            };
            let full_w = ui.available_width();
            let header = ui.allocate_ui_with_layout(
                egui::vec2(full_w, 36.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("{caret}  {}", series.name))
                            .size(18.0)
                            .strong(),
                    );
                    // Status sits just after the title (a dim "· N unread" / "·
                    // Reading" / "· Finished"). Kept left-aligned with the title
                    // rather than pushed to the far right: text near the right edge
                    // of this nested scroll-area layout doesn't paint reliably.
                    ui.add_space(10.0);
                    ui.label(
                        egui::RichText::new(format!("· {status}"))
                            .size(13.0)
                            .weak(),
                    );
                },
            );
            let rect = header.response.rect;
            if header.response.interact(egui::Sense::click()).clicked() {
                st.toggle_series = Some(series.dir.clone());
            }

            if expanded {
                // Lazy covers: only sections actually on screen queue decodes.
                if ui.is_rect_visible(rect) {
                    st.visible_covers
                        .extend(series.volumes.iter().map(|v| v.path.clone()));
                }
                // Horizontal cover row, full width. The wheel is intentionally NOT a
                // scroll source here, so a plain wheel over a row falls through to the
                // outer vertical list (reliable vertical scrolling with the cursor
                // anywhere). The row scrolls sideways via its scrollbar or by
                // click-dragging the covers; a quick click still opens a volume.
                egui::ScrollArea::horizontal()
                    .id_salt(&series.dir)
                    .scroll_source(egui::scroll_area::ScrollSource {
                        scroll_bar: true,
                        drag: true,
                        mouse_wheel: false,
                    })
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (v, state) in series.volumes.iter().zip(&states) {
                                let is_current = libctx.current_key
                                    == Some(v.path.to_string_lossy().as_ref());
                                volume_cell(ui, st, &v.name, v.path.clone(), v.thumb, *state, is_current);
                            }
                        });
                    });
            }
            ui.add_space(10.0);
        }
    });
}

/// The "Recently read" shelf at the top of the library: the most-recently opened
/// volumes (newest first) that still exist in the current library, drawn with the
/// same cover cells as the series rows. Mirrors the Android library's recents row.
/// No-op when there are no resolvable recents (e.g. a fresh library, or recents that
/// all point outside the current root).
fn recents_row(ui: &mut egui::Ui, st: &mut UiState, lib: &Library, libctx: &LibCtx) {
    if libctx.recents.is_empty() {
        return;
    }
    // Index volumes by their string key (matching how recents/last_pages are keyed:
    // `path.to_string_lossy()`) so a recent resolves to its cover/name/state.
    let mut by_key: std::collections::HashMap<String, &crate::library::Volume> =
        std::collections::HashMap::new();
    for series in &lib.series {
        for v in &series.volumes {
            by_key.insert(v.path.to_string_lossy().into_owned(), v);
        }
    }
    let recents: Vec<&crate::library::Volume> = libctx
        .recents
        .iter()
        .filter_map(|p| by_key.get(p).copied())
        .take(RECENTS_SHELF)
        .collect();
    if recents.is_empty() {
        return;
    }
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Recently read").size(18.0).strong());
    egui::ScrollArea::horizontal()
        .id_salt("recents_row")
        .scroll_source(egui::scroll_area::ScrollSource {
            scroll_bar: true,
            drag: true,
            mouse_wheel: false,
        })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                for v in &recents {
                    let key = v.path.to_string_lossy();
                    let state = vol_state(libctx.progress, libctx.last_pages, key.as_ref());
                    let is_current = libctx.current_key == Some(key.as_ref());
                    // Recents may belong to a collapsed or off-screen series; queue
                    // their covers so the shelf isn't left blank.
                    st.visible_covers.push(v.path.clone());
                    volume_cell(ui, st, &v.name, v.path.clone(), v.thumb, state, is_current);
                }
            });
        });
    ui.add_space(10.0);
    ui.separator();
}

/// One volume in a series row: cover (faded when finished, thin progress bar when
/// started, highlight stroke when currently open) + truncated name. Clicking sets
/// `pending_open` so the app opens it next frame.
#[allow(deprecated)] // egui ImageButton — matches the rest of this module
fn volume_cell(
    ui: &mut egui::Ui,
    st: &mut UiState,
    name: &str,
    path: PathBuf,
    thumb: Option<egui::TextureId>,
    state: VolState,
    is_current: bool,
) {
    ui.allocate_ui(egui::vec2(CELL_W, COVER_H + 40.0), |ui| {
        ui.vertical(|ui| {
            let finished = state == VolState::Finished;
            let r = match thumb {
                Some(tid) => {
                    let mut img = egui::Image::new(egui::load::SizedTexture::new(
                        tid,
                        egui::vec2(CELL_W, COVER_H),
                    ));
                    if finished {
                        // Read volumes fade out (Chunky-style).
                        img = img.tint(egui::Color32::from_white_alpha(72));
                    }
                    ui.add(egui::ImageButton::new(img))
                }
                None => ui.add_sized(
                    [CELL_W, COVER_H],
                    egui::Button::new(egui::RichText::new("…").size(24.0)),
                ),
            };
            let accent = ui.visuals().selection.bg_fill;
            if is_current {
                ui.painter().rect_stroke(
                    r.rect,
                    3.0,
                    egui::Stroke::new(2.0_f32, accent),
                    egui::StrokeKind::Outside,
                );
            }
            if let VolState::InProgress(frac) = state {
                let y = r.rect.bottom() - 2.0;
                let w = r.rect.width() * frac.clamp(0.02, 1.0);
                ui.painter().line_segment(
                    [
                        egui::pos2(r.rect.left(), y),
                        egui::pos2(r.rect.left() + w, y),
                    ],
                    egui::Stroke::new(3.0_f32, accent),
                );
            }
            let mut label = egui::RichText::new(elide(name, 22)).size(12.0);
            if finished {
                label = label.weak();
            }
            ui.add_sized([CELL_W, 32.0], egui::Label::new(label).truncate());
            if r.clicked() {
                st.pending_open = Some(path);
            }
        });
    });
}

//! egui chrome: open buttons, reading-mode toggles, page indicator, and the
//! library browse grid. Toggle buttons + grid clicks set request flags the app
//! consumes after the egui frame.

use std::path::PathBuf;

use crate::library::Library;

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

    // Requests raised by clicks, consumed by the app after the frame.
    pub req_toggle_dir: bool,
    pub req_cycle_fit: bool,
    pub req_toggle_layout: bool,
    pub req_toggle_library: bool,
    pub clicked_volume: Option<usize>,
    pub help_open: bool,
    /// Whether to draw the top chrome bar (hidden in fullscreen unless the
    /// cursor is at the top edge). Set by the app each frame.
    pub show_bar: bool,
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
    /// Pointer is over the seekbar this frame. Lets the app keep routing the
    /// mouse wheel to the reader (the bar isn't scrollable) while egui still
    /// gets clicks/drags for seeking.
    pub seek_hovered: bool,

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
            let stroke = egui::Stroke::new(5.0, egui::Color32::from_white_alpha(205));
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
                    // Let the wheel keep scrolling the reader while over the bar.
                    st.seek_hovered = resp.contains_pointer();

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
                        egui::Stroke::new(8.0, egui::Color32::from_white_alpha(60)),
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
                        egui::Stroke::new(8.0, egui::Color32::from_white_alpha(190)),
                    );
                    p.circle_filled(egui::pos2(handle_x, cy), r, egui::Color32::WHITE);
                    p.circle_stroke(
                        egui::pos2(handle_x, cy),
                        r,
                        egui::Stroke::new(2.0, egui::Color32::from_black_alpha(120)),
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
                                egui::Stroke::new(5.0, egui::Color32::from_white_alpha(55)),
                            );
                            p.line_segment(
                                [egui::pos2(x0, cy), egui::pos2(hx, cy)],
                                egui::Stroke::new(5.0, egui::Color32::from_white_alpha(200)),
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
pub fn chrome(ctx: &egui::Context, st: &mut UiState, lib: &Library, library_view: bool) {
    let show_bar = st.show_bar;
    egui::TopBottomPanel::top("top_bar").show_animated(ctx, show_bar, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Open folder…").clicked() {
                if let Some(p) = rfd::FileDialog::new().set_title("Open page folder").pick_folder()
                {
                    st.pending_open = Some(p);
                }
            }
            if ui.button("Open file…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
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
            }
            if ui.button("Library…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Choose a library folder")
                    .pick_folder()
                {
                    st.pending_library = Some(p);
                }
            }
            if !lib.volumes.is_empty() && ui.button(if library_view { "Reader" } else { "Grid" }).clicked() {
                st.req_toggle_library = true;
            }
            ui.separator();
            if ui.button(format!("Dir: {}", st.dir_label)).clicked() {
                st.req_toggle_dir = true;
            }
            if ui.button(format!("Fit: {}", st.fit_label)).clicked() {
                st.req_cycle_fit = true;
            }
            if ui.button(format!("Layout: {}", st.layout_label)).clicked() {
                st.req_toggle_layout = true;
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
        });
    });

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
                ui.label("G   show/hide the animation panel (animated GIF / WebP)");
                ui.label("F11   fullscreen      Esc   quit");
                ui.separator();
                ui.heading("Files");
                ui.label("E   show in Explorer (open the folder & select the file)");
                ui.label("Open folder / Open file / Library;  Grid ↔ Reader");
                ui.label("drag a folder, archive, or image onto the window");
                ui.label("F1   toggle this help");
            });
    }

    if st.info_open && !library_view && !st.info.is_empty() {
        egui::Area::new(egui::Id::new("info_overlay"))
            .fixed_pos(egui::pos2(12.0, 44.0))
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
            if lib.volumes.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No volumes found — pick a library folder.");
                });
                return;
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (i, v) in lib.volumes.iter().enumerate() {
                        ui.vertical(|ui| {
                            ui.set_width(184.0);
                            let clicked = match v.thumb {
                                Some(tid) => ui
                                    .add(egui::ImageButton::new(egui::Image::new(
                                        egui::load::SizedTexture::new(
                                            tid,
                                            egui::vec2(174.0, 246.0),
                                        ),
                                    )))
                                    .clicked(),
                                None => ui
                                    .add_sized(
                                        egui::vec2(174.0, 246.0),
                                        egui::Button::new("(no preview)"),
                                    )
                                    .clicked(),
                            };
                            if clicked {
                                st.clicked_volume = Some(i);
                            }
                            ui.label(elide(&v.name, 24));
                        });
                    }
                });
            });
        });
    }
}

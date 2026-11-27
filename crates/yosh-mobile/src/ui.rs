//! egui widgets for the touch chrome: seekbar, options, prompts, library cells.

use super::*;

/// How long the next/prev-book boundary prompt stays armed and on screen.
pub(crate) const BOOK_PROMPT_MS: u64 = 3000;

/// Which screen edge the boundary-prompt card sits on: the edge whose tap zone
/// triggers `dir` (next = left in RTL / right in LTR; prev mirrors). Shared by
/// the egui draw and `on_tap`'s fallback hit-rect so they can't disagree.
pub(crate) fn book_prompt_on_left(dir: i64, rtl: bool) -> bool {
    (dir > 0) == rtl
}

/// Width of the boundary prompt card, egui points.
pub(crate) const BOOK_PROMPT_W_PT: f32 = 260.0;

/// Bottom seekbar: "page / total" + a draggable slider that requests a jump.
/// Hidden for a single-page source.
/// Image-info overlay as a closable centered popup (Android take on the "I"
/// overlay): check it, then ✕ to dismiss — not a persistent overlay.
/// A framed, clickable image — egui 0.34's `ImageButton` (removed in 0.36), ported
/// so the covers keep their exact look: a `button_padding.x` border on every side,
/// the interact-state fill and stroke, and the image at its own size. Mirrors the
/// desktop shell's `ui::image_button`.
fn image_button(ui: &mut egui::Ui, image: egui::Image<'_>) -> egui::Response {
    let padding = egui::Vec2::splat(ui.spacing().button_padding.x);
    let avail = ui.available_size() - 2.0 * padding;
    // Covers are already-uploaded `SizedTexture`s, so the size is known up front.
    let size = image.calc_size(avail, image.size());
    let (rect, response) = ui.allocate_exact_size(size + 2.0 * padding, egui::Sense::click());
    response.widget_info(|| egui::WidgetInfo::new(egui::WidgetType::Button));
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        let frame = rect.expand(visuals.expansion);
        let radius = image.image_options().corner_radius;
        ui.painter().rect_filled(frame, radius, visuals.weak_bg_fill);
        image.paint_at(ui, ui.layout().align_size_within_rect(size, rect.shrink2(padding)));
        ui.painter()
            .rect_stroke(frame, radius, visuals.bg_stroke, egui::StrokeKind::Inside);
    }
    response
}

pub(crate) fn info_popup(ctx: &egui::Context, lines: &[(String, String)], close: &mut bool) {
    egui::Area::new(egui::Id::new("info_popup"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -20.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                ui.label(egui::RichText::new(tr().info).strong().size(18.0));
                egui::Grid::new("info_grid")
                    .num_columns(2)
                    .spacing([18.0, 6.0])
                    .show(ui, |ui| {
                        for (k, v) in lines {
                            ui.label(egui::RichText::new(k.as_str()).strong());
                            ui.label(v.as_str());
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                if ui
                    .add_sized(
                        [120.0, 40.0],
                        egui::Button::new(egui::RichText::new(tr().close).size(16.0)),
                    )
                    .clicked()
                {
                    *close = true;
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn seekbar(
    ctx: &egui::Context,
    title: &str,
    cur: usize,
    len: usize,
    rtl: bool,
    spread: bool,
    fit: FitMode,
    zoomed: bool,
    buffered: &[usize],
    lq_buffered: &[usize],
    seek_to: &mut Option<usize>,
    open_options: &mut bool,
    open_info: &mut bool,
    page_nav: &mut i64,
    book_nav: &mut i64,
    toggle_offset: &mut bool,
    cycle_fit: &mut bool,
    to_library: &mut bool,
) {
    use egui_phosphor::fill as ph;
    if len <= 1 {
        return;
    }
    // A floating pill lifted off the bottom edge (the very edge collides with the
    // system gesture bar + is awkward to grab), fattened for touch.
    let sw = ctx.content_rect().width();
    // The seekbar hand-paints the slider/buttons (egui can't style them per-widget
    // here), so pick high-contrast tones for light/e-ink off the active theme.
    let light = !ctx.global_style().visuals.dark_mode;
    // Translucent panel so the page shows through behind the controls. Keep the
    // popup's color/stroke/rounding, just drop the fill's alpha.
    let frame = translucent_popup(&ctx.global_style());
    egui::Area::new(egui::Id::new("seekbar"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -40.0))
        .show(ctx, |ui| {
            frame.show(ui, |ui| {
                ui.set_width((sw * 0.85).min(960.0));
                ui.spacing_mut().interact_size.y = 30.0;
                // Book filename atop the pill, centered and filling the width;
                // `.truncate()` ellipsizes a long name instead of widening the
                // (fixed-width) pill.
                if !title.is_empty() {
                    ui.vertical_centered_justified(|ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new(title).size(15.0).strong())
                                .truncate(),
                        );
                    });
                    ui.add_space(2.0);
                }
                ui.horizontal(|ui| {
                    // Page indicator split across the slider — current page on the
                    // reading-start side, total on the far side (LTR: cur | slider | total;
                    // RTL mirrored, matching the slider's value transform below). Both are
                    // monospace boxes sized to the total's digit width, so the two ends are
                    // equal width and the slider — and thus the whole pill — stays centered.
                    let digits = len.to_string().len();
                    let num_w = ui
                        .painter()
                        .layout_no_wrap(
                            "0".repeat(digits),
                            egui::FontId::monospace(18.0),
                            egui::Color32::WHITE,
                        )
                        .size()
                        .x
                        .ceil()
                        + 6.0;
                    let sp = ui.spacing().item_spacing.x;
                    ui.spacing_mut().slider_width =
                        (ui.available_width() - 2.0 * (num_w + sp) - 2.0).max(120.0);
                    let num = |ui: &mut egui::Ui, n: usize| {
                        ui.add_sized(
                            [num_w, 24.0],
                            egui::Label::new(
                                egui::RichText::new(n.to_string()).size(18.0).monospace(),
                            ),
                        );
                    };
                    // Slider colors, fully opaque (the handle must never be translucent).
                    // egui ties the idle handle's fill to the rail-behind color
                    // (widgets.inactive.bg_fill), so keep that bright blue unconditionally
                    // — the handle is then always solid blue. The trailing portion (rail
                    // start → handle) is selection.bg_fill; in RTL (manga order) that's the
                    // UNSEEKED side, so paint it neutral grey so the unread part reads as
                    // clearly not-there. (In LTR the trailing side is the read side, so the
                    // track reads inverted there — we optimise for RTL; the handle stays
                    // correctly blue.) A white ring keeps the handle legible against blue.
                    let (seeked, unseeked, ring_col) = if light {
                        // On white: dark handle/rail, light-grey track, dark ring.
                        (
                            egui::Color32::from_rgb(70, 70, 70),
                            egui::Color32::from_rgb(200, 200, 200),
                            egui::Color32::from_rgb(30, 30, 30),
                        )
                    } else {
                        (
                            egui::Color32::from_rgb(96, 185, 255), // bright blue: handle + seeked rail
                            egui::Color32::from_rgb(120, 124, 130), // neutral grey: unseeked track
                            egui::Color32::WHITE,
                        )
                    };
                    let ring = egui::Stroke::new(2.0, ring_col);
                    let v = ui.visuals_mut();
                    v.selection.bg_fill = unseeked;
                    v.widgets.inactive.bg_fill = seeked;
                    v.widgets.hovered.bg_fill = seeked;
                    v.widgets.active.bg_fill = seeked;
                    v.widgets.inactive.fg_stroke = ring;
                    v.widgets.hovered.fg_stroke = ring;
                    v.widgets.active.fg_stroke = ring;
                    // Leading number: total in RTL (page 1 is on the right), current in LTR.
                    num(ui, if rtl { len } else { cur + 1 });
                    // RTL: map so page 1 sits on the right and progress runs leftward.
                    let mut sv = if rtl { len - 1 - cur } else { cur };
                    let resp = ui.add(
                        egui::Slider::new(&mut sv, 0..=len - 1)
                            .show_value(false)
                            .trailing_fill(true),
                    );
                    if resp.changed() {
                        *seek_to = Some(if rtl { len - 1 - sv } else { sv });
                    }
                    // Trailing number: current in RTL, total in LTR.
                    num(ui, if rtl { cur + 1 } else { len });
                    // mpv-style cache bar: thin ticks along the bottom of the rail.
                    // Faint wash = LQ preview thumbnails (whole volume once warm);
                    // brighter ticks = decode-ahead HQ pages near the handle, drawn on
                    // top. Both stay subordinate to the blue handle/rail.
                    if !buffered.is_empty() || !lq_buffered.is_empty() {
                        let track = resp.rect;
                        let r = track.height() / 2.5; // egui's slider handle radius
                        let x0 = track.left() + r;
                        let x1 = track.right() - r;
                        let span = (x1 - x0).max(1.0);
                        let last = (len - 1) as f32;
                        let half = (span / last * 0.5).max(0.75); // half a page-step wide
                        let yb = track.bottom() - 1.0;
                        let yt = yb - 2.0;
                        let (lq_tick, hq_tick) = if light {
                            // Darker green so the cache wash reads on white.
                            (
                                egui::Color32::from_rgba_unmultiplied(70, 130, 90, 90),
                                egui::Color32::from_rgba_unmultiplied(50, 110, 70, 180),
                            )
                        } else {
                            (
                                egui::Color32::from_rgba_unmultiplied(120, 165, 140, 55),
                                egui::Color32::from_rgba_unmultiplied(120, 165, 140, 150),
                            )
                        };
                        let p = ui.painter();
                        // LQ wash first, then HQ on top. Mirror the slider's own RTL
                        // value transform so ticks line up with where the handle sits.
                        for (set, color) in [(lq_buffered, lq_tick), (buffered, hq_tick)] {
                            for &i in set {
                                if i >= len {
                                    continue;
                                }
                                let sv_i = if rtl { last - i as f32 } else { i as f32 };
                                let xc = x0 + sv_i / last * span;
                                let a = (xc - half).clamp(x0, x1);
                                let b = (xc + half).clamp(x0, x1);
                                p.rect_filled(
                                    egui::Rect::from_min_max(egui::pos2(a, yt), egui::pos2(b, yb)),
                                    0.0,
                                    color,
                                );
                            }
                        }
                    }
                });
                // One centered row of big touch icon buttons. The nav arrows are
                // POSITIONAL (left = leftward in reading); the action flips with
                // direction. Outer = book (a book glyph with an arrow over it), inner
                // = page; gear = options, info, ↔ = pairing (spread only), × = close.
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let count = if spread { 9 } else { 8 };
                    let sp = ui.spacing().item_spacing.x;
                    let avail = ui.available_width();
                    // Fit every button across the available width (shrinking on narrow
                    // screens), but cap the size so they don't balloon on wide ones.
                    let bw = (((avail - (count - 1) as f32 * sp) / count as f32).floor())
                        .clamp(40.0, 58.0);
                    let total = count as f32 * bw + (count - 1) as f32 * sp;
                    ui.add_space(((avail - total) * 0.5).max(0.0));
                    // PerfectViewer-style palette: every button gets its own vivid hue
                    // (no plain white). Green page arrows + orange book stay; each utility
                    // icon takes a distinct colour so they read apart at a glance. On
                    // light/e-ink the hues wash out to mid-grey, so use one near-black for
                    // every glyph (max contrast — colour-coding is lost on grayscale anyway).
                    let (green, orange, amber, cyan, violet, azure, rose) = if light {
                        let k = egui::Color32::from_rgb(40, 40, 40);
                        (k, k, k, k, k, k, k)
                    } else {
                        (
                            egui::Color32::from_rgb(124, 200, 80),
                            egui::Color32::from_rgb(212, 140, 56),
                            egui::Color32::from_rgb(224, 176, 74), // gear / options
                            egui::Color32::from_rgb(96, 196, 200), // fit
                            egui::Color32::from_rgb(176, 138, 232), // spread toggle
                            egui::Color32::from_rgb(104, 170, 236), // info
                            egui::Color32::from_rgb(232, 128, 150), // return to library
                        )
                    };
                    let big = |ui: &mut egui::Ui, txt: &str, color: egui::Color32| {
                        ui.add_sized(
                            [bw, 46.0],
                            egui::Button::new(egui::RichText::new(txt).size(22.0).color(color)),
                        )
                        .clicked()
                    };
                    // A solid orange book with a glyph poking past its edge — used for
                    // prev/next-book (a green arrow off the left/right edge). Offsetting
                    // the glyph past the edge (rather than dead centre) keeps both the
                    // book and the glyph legible; the glyph's side is positional (fixed),
                    // and the action flips with direction at the call.
                    let book_overlay = |ui: &mut egui::Ui,
                                        glyph: &str,
                                        off: egui::Vec2,
                                        gsize: f32,
                                        glyph_color: egui::Color32|
                     -> bool {
                        let resp = ui.add_sized([bw, 46.0], egui::Button::new(""));
                        let c = resp.rect.center();
                        let p = ui.painter_at(resp.rect);
                        p.text(
                            c + egui::vec2(0.0, 1.0),
                            egui::Align2::CENTER_CENTER,
                            ph::BOOK,
                            egui::FontId::proportional(28.0),
                            orange,
                        );
                        p.text(
                            c + off,
                            egui::Align2::CENTER_CENTER,
                            glyph,
                            egui::FontId::proportional(gsize),
                            glyph_color,
                        );
                        resp.clicked()
                    };
                    if big(ui, ph::GEAR, amber) {
                        *open_options = true;
                    }
                    // Fit-mode button: when zoomed (in or out of the fit scale), the
                    // first tap drops the zoom to restore the active fit (icon dimmed
                    // to show no fit is active); otherwise it cycles fit modes. The icon
                    // reflects the current fit (1:1 stays as text).
                    let fit_color = if zoomed {
                        cyan.gamma_multiply(0.5)
                    } else {
                        cyan
                    };
                    let fit_rich = match fit {
                        FitMode::Window => egui::RichText::new(ph::ARROWS_OUT).size(22.0),
                        FitMode::Width => egui::RichText::new(ph::ARROWS_OUT_LINE_HORIZONTAL).size(22.0),
                        FitMode::Height => egui::RichText::new(ph::ARROWS_OUT_LINE_VERTICAL).size(22.0),
                        FitMode::Actual => egui::RichText::new("1:1").size(16.0),
                    }
                    .color(fit_color);
                    if ui
                        .add_sized([bw, 46.0], egui::Button::new(fit_rich))
                        .clicked()
                    {
                        *cycle_fit = true;
                    }
                    if book_overlay(ui, ph::ARROW_FAT_LEFT, egui::vec2(-11.0, 1.0), 16.0, green) {
                        *book_nav = if rtl { 1 } else { -1 };
                    }
                    if big(ui, ph::ARROW_FAT_LEFT, green) {
                        *page_nav = if rtl { 1 } else { -1 };
                    }
                    if spread && big(ui, ph::ARROWS_LEFT_RIGHT, violet) {
                        *toggle_offset = true;
                    }
                    if big(ui, ph::ARROW_FAT_RIGHT, green) {
                        *page_nav = if rtl { -1 } else { 1 };
                    }
                    if book_overlay(ui, ph::ARROW_FAT_RIGHT, egui::vec2(11.0, 1.0), 16.0, green) {
                        *book_nav = if rtl { -1 } else { 1 };
                    }
                    if big(ui, ph::INFO, azure) {
                        *open_info = true;
                    }
                    // Return to the bookshelf, leaving the book warm (instant resume).
                    // Redundant with the top-strip tap, but a visible, discoverable button.
                    if big(ui, ph::BOOKS, rose) {
                        *to_library = true;
                    }
                });
            });
        });
}

/// Nothing-open helper: with no comic loaded the screen is otherwise just the dark
/// clear color, so explain how to open one. Centered card with the two ways in —
/// browse the library or pick a single file — plus the tap-the-top tip.
pub(crate) fn empty_state(
    ctx: &egui::Context,
    logo: Option<&egui::TextureHandle>,
    open_library: &mut bool,
    open_picker: &mut bool,
) {
    egui::Area::new(egui::Id::new("empty_state"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width((ctx.content_rect().width() * 0.8).min(420.0));
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 12.0);
                    // Secondary text color from the active theme (was a hardcoded
                    // white alpha — invisible on the light card).
                    let weak = ui.visuals().weak_text_color();
                    // The yosh mascot if it decoded; otherwise a book glyph.
                    if let Some(tex) = logo {
                        let [w, h] = tex.size();
                        let scale = 84.0 / h as f32;
                        ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(
                            w as f32 * scale,
                            h as f32 * scale,
                        )));
                    } else {
                        ui.label(egui::RichText::new("📖").size(56.0));
                    }
                    ui.label(egui::RichText::new(tr().no_comic_open).strong().size(22.0));
                    ui.label(
                        egui::RichText::new(tr().open_comic_to_start)
                            .size(15.0)
                            .color(weak),
                    );
                    ui.add_space(4.0);
                    ui.spacing_mut().button_padding = egui::vec2(18.0, 12.0);
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(format!(
                                "{}  {}",
                                egui_phosphor::fill::BOOKS,
                                tr().browse_library
                            ))
                            .size(18.0),
                        ))
                        .clicked()
                    {
                        *open_library = true;
                    }
                    if ui
                        .add(egui::Button::new(egui::RichText::new(tr().open_file).size(18.0)))
                        .clicked()
                    {
                        *open_picker = true;
                    }
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(tr().tap_top_tip)
                            .size(12.0)
                            .color(weak),
                    );
                });
            });
        });
}

/// "Opening…" card: a spinner plus the book's name, shown while a background open
/// is in flight (see [`Shell::start_open`]). `name` is empty when the open is the
/// launch resume, which is still deciding *which* book to reopen.
pub(crate) fn opening_card(ctx: &egui::Context, name: &str) {
    egui::Area::new(egui::Id::new("opening_card"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.set_max_width((ctx.content_rect().width() * 0.8).min(360.0));
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 10.0);
                    ui.add(egui::Spinner::new().size(40.0));
                    ui.label(egui::RichText::new(tr().opening).strong().size(18.0));
                    if !name.is_empty() {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(name)
                                    .size(14.0)
                                    .color(ui.visuals().weak_text_color()),
                            )
                            .truncate(),
                        );
                    }
                });
            });
        });
}

/// One volume in a series row: cover (faded when finished, thin progress bar
/// when started, highlight stroke when currently open) + truncated name.
/// Clicking opens it.
pub(crate) fn volume_cell(ui: &mut egui::Ui, v: &VolCell, reqs: &mut FrameReqs, close_lib: &mut bool) {
    const CELL_W: f32 = 140.0;
    const COVER_H: f32 = 186.0;
    // The caption strip under the cover: two 12pt rows.
    const CAPTION_H: f32 = 34.0;
    ui.allocate_ui(egui::vec2(CELL_W, COVER_H + CAPTION_H + 8.0), |ui| {
        ui.vertical(|ui| {
            let finished = v.state == VolState::Finished;
            let r = if let Some(t) = &v.thumb {
                let mut img = egui::Image::new(t).fit_to_exact_size(egui::vec2(CELL_W, COVER_H));
                if finished {
                    // Read volumes fade out (Chunky-style).
                    img = img.tint(egui::Color32::from_white_alpha(72));
                }
                image_button(ui, img)
            } else {
                ui.add_sized(
                    [CELL_W, COVER_H],
                    egui::Button::new(egui::RichText::new("…").size(24.0)),
                )
            };
            let accent = ui.visuals().selection.bg_fill;
            if v.is_current {
                ui.painter().rect_stroke(
                    r.rect,
                    3.0,
                    egui::Stroke::new(2.0, accent),
                    egui::StrokeKind::Outside,
                );
            }
            if let VolState::InProgress(frac) = v.state {
                let y = r.rect.bottom() - 2.0;
                let w = r.rect.width() * frac.clamp(0.02, 1.0);
                ui.painter().line_segment(
                    [
                        egui::pos2(r.rect.left(), y),
                        egui::pos2(r.rect.left() + w, y),
                    ],
                    egui::Stroke::new(3.0, accent),
                );
            }
            // Wrapped to two rows, then ellipsized. Not `add_sized`: that justifies
            // the label, and epaint stretches every row but the last.
            let color = if finished {
                ui.visuals().weak_text_color()
            } else {
                ui.visuals().text_color()
            };
            let mut job = egui::text::LayoutJob::simple(
                v.label.clone(),
                egui::FontId::proportional(12.0),
                color,
                CELL_W,
            );
            job.wrap.max_rows = 2;
            ui.allocate_ui_with_layout(
                egui::vec2(CELL_W, CAPTION_H),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.add(egui::Label::new(job).wrap().show_tooltip_when_elided(false));
                },
            );
            if r.clicked() {
                reqs.open = Some(v.path.clone());
                *close_lib = true;
            }
        });
    });
}

/// Centered "Next/Previous book" card, shown after a flip ran out of pages:
/// the sibling's cover (once its background decode lands) + title; tapping the
/// cover — or repeating the boundary action — opens it (`book_nav`, the same
/// request the seekbar's book buttons use). With no sibling (last book in the
/// folder, or a SAF-picked comic) it just names the boundary.
pub(crate) fn book_prompt_card(
    ctx: &egui::Context,
    dir: i64,
    rtl: bool,
    title: &str,
    thumb: Option<&egui::TextureHandle>,
    has_sibling: bool,
    book_nav: &mut i64,
) {
    // Anchor to the edge whose tap zone fired it (next = left in RTL, right in
    // LTR; prev mirrors) so the card appears under the tapping thumb.
    let anchor = if book_prompt_on_left(dir, rtl) {
        (egui::Align2::LEFT_CENTER, [16.0, 0.0])
    } else {
        (egui::Align2::RIGHT_CENTER, [-16.0, 0.0])
    };
    egui::Area::new(egui::Id::new("book_prompt"))
        .anchor(anchor.0, anchor.1)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(BOOK_PROMPT_W_PT);
                ui.vertical_centered(|ui| {
                    if !has_sibling {
                        ui.label(
                            egui::RichText::new(if dir > 0 { tr().last_page } else { tr().first_page })
                                .size(18.0)
                                .strong(),
                        );
                        return;
                    }
                    ui.label(
                        egui::RichText::new(if dir > 0 { tr().next_book } else { tr().previous_book })
                            .size(18.0)
                            .strong(),
                    );
                    ui.add_space(8.0);
                    let cover_w = BOOK_PROMPT_W_PT - 24.0;
                    let clicked = if let Some(t) = thumb {
                        image_button(
                            ui,
                            egui::Image::new(t).fit_to_exact_size(egui::vec2(cover_w, 240.0)),
                        )
                        .clicked()
                    } else {
                        // Cover still decoding: placeholder button, same action.
                        ui.add_sized(
                            [cover_w, 120.0],
                            egui::Button::new(egui::RichText::new("📖").size(48.0)),
                        )
                        .clicked()
                    };
                    ui.add_space(6.0);
                    ui.add(
                        egui::Label::new(egui::RichText::new(title).size(14.0)).truncate(),
                    );
                    ui.label(egui::RichText::new(tr().tap_again_to_open).size(12.0).weak());
                    if clicked {
                        *book_nav = dir;
                    }
                });
            });
        });
}

/// A popup frame with the fill alpha dropped so the page shows through (matches the
/// seekbar). Shared by the seekbar + the options/info popups.
fn translucent_popup(style: &egui::Style) -> egui::Frame {
    let frame = egui::Frame::popup(style);
    // Light / e-ink mode: keep it fully opaque (translucency over the page looks
    // muddy on a reflective panel). Only the dark theme lets the page show through.
    if !style.visuals.dark_mode {
        return frame;
    }
    let f = frame.fill;
    frame.fill(egui::Color32::from_rgba_unmultiplied(f.r(), f.g(), f.b(), 200))
}

/// A soft drop shadow riding the dragged page's leading edge (page-flip swipe), cast
/// onto the revealed page beneath. `frac` is the seam x (0..1 of width); `si`'s sign
/// is the revealed side, magnitude the intensity. Painted on the background layer so
/// it sits over the page but under the chrome.
pub(crate) fn drag_shadow(ctx: &egui::Context, frac: f32, si: f32) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("drag_shadow"),
    ));
    let rect = ctx.content_rect();
    let seam_x = rect.left() + frac * rect.width();
    let w = 22.0;
    let far_x = if si > 0.0 { seam_x - w } else { seam_x + w };
    let dark = egui::Color32::from_black_alpha((si.abs() * 60.0).clamp(0.0, 255.0) as u8);
    let clear = egui::Color32::TRANSPARENT;
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(egui::pos2(seam_x, rect.top()), dark);
    mesh.colored_vertex(egui::pos2(seam_x, rect.bottom()), dark);
    mesh.colored_vertex(egui::pos2(far_x, rect.top()), clear);
    mesh.colored_vertex(egui::pos2(far_x, rect.bottom()), clear);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 3, 2);
    painter.add(mesh);
}

/// Faint labels + outlines over the tap zones (shown briefly with the controls) so
/// the layout is discoverable. Painted, NOT laid out as widgets: an egui Area
/// registers under the pointer and makes the tap report "consumed", blocking the
/// edge/top zones. A background-layer painter does no hit-testing, so taps fall
/// through to nav.
pub(crate) fn zone_hints(ctx: &egui::Context, rtl: bool, scroll: bool) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("zone_hints"),
    ));
    let rect = ctx.content_rect();
    // Outline the tap zones. Two-tone — a dark stroke under a lighter one — so the
    // lines read on both white and dark manga pages.
    let dark = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(120));
    let light = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(200));
    let ty = rect.top() + rect.height() * TOP_ZONE;
    // Top strip (library) — both modes.
    for s in [dark, light] {
        painter.hline(rect.x_range(), ty, s);
    }
    // Side flip zones only in page-flip; scroll mode has no side flips (any tap
    // below the top strip toggles the controls).
    if !scroll {
        let lx = rect.left() + rect.width() * EDGE_ZONE;
        let rx = rect.right() - rect.width() * EDGE_ZONE;
        let yr = egui::Rangef::new(ty, rect.bottom());
        for s in [dark, light] {
            painter.vline(lx, yr, s);
            painter.vline(rx, yr, s);
        }
    }
    let font = egui::FontId::proportional(18.0);
    let draw = |pos: egui::Pos2, anchor: egui::Align2, text: &str| {
        // A dark rounded backing keeps the label legible over light pages.
        let galley =
            painter.layout_no_wrap(text.to_owned(), font.clone(), egui::Color32::from_white_alpha(235));
        let r = anchor.anchor_size(pos, galley.size());
        painter.rect_filled(
            r.expand2(egui::vec2(10.0, 6.0)),
            6.0,
            egui::Color32::from_black_alpha(180),
        );
        painter.galley(r.min, galley, egui::Color32::from_white_alpha(235));
    };
    draw(
        egui::pos2(rect.center().x, rect.top() + 56.0),
        egui::Align2::CENTER_CENTER,
        &format!("{}  {}", egui_phosphor::fill::BOOKS, tr().library),
    );
    // Side flip labels only in page-flip.
    if !scroll {
        let (next, prev) = (tr().zone_next, tr().zone_prev);
        let (left, right) = if rtl { (next, prev) } else { (prev, next) };
        draw(
            egui::pos2(rect.left() + 40.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            left,
        );
        draw(
            egui::pos2(rect.right() - 40.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            right,
        );
    }
}

/// Viewing-options popup (from the seekbar gear): reading direction, page layout,
/// fit. Records the chosen value into the `set_*` outs; the caller applies it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn options_popup(
    ctx: &egui::Context,
    dir: Direction,
    layout: LayoutMode,
    effective_spread: bool,
    fit: FitMode,
    // Whether fits may stretch a page past 100% native — the *inverse* of the
    // reader's `fit_no_upscale`, because the row reads "Stretch small pages"
    // (stretching is what yosh has always done). Mirrors the desktop panel.
    stretch_on: bool,
    transition_on: bool,
    spine_on: bool,
    spine_strength: f32,
    resume_on: bool,
    vol_keys: VolKeys,
    lang_pref: Option<Lang>,
    scroll_on: bool,
    theme: ThemePref,
    perf: PerfPref,
    rotation: u8,
    set_dir: &mut Option<Direction>,
    set_layout: &mut Option<LayoutMode>,
    set_fit: &mut Option<FitMode>,
    // Chosen `fit_no_upscale` (the inverse of the row's On/Off).
    set_no_upscale: &mut Option<bool>,
    set_transition: &mut Option<bool>,
    set_spine_on: &mut Option<bool>,
    set_spine_strength: &mut Option<f32>,
    set_resume: &mut Option<bool>,
    set_vol_keys: &mut Option<VolKeys>,
    set_lang: &mut Option<Option<Lang>>,
    set_scroll: &mut Option<bool>,
    set_theme: &mut Option<ThemePref>,
    set_perf: &mut Option<PerfPref>,
    toggle_offset: &mut bool,
    rotate: &mut bool,
    // Status-bar height in points (the bars show while the popup is up).
    top_inset: f32,
) {
    let t = tr();
    egui::Area::new(egui::Id::new("view_options"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -40.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().button_padding = egui::vec2(16.0, 10.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 10.0);

                // Cap the height and scroll the rest: in landscape on a tablet the
                // full list is taller than the screen, and an unscrolled Area just
                // runs off both edges (the top rows were unreachable). The cap keeps
                // the card clear of the status bar given the -40pt anchor offset.
                let max_h = ctx.viewport_rect().height() - 80.0 - 2.0 * (top_inset + 16.0);
                egui::ScrollArea::vertical().max_height(max_h.max(160.0)).show(ui, |ui| {
                    ui.label(egui::RichText::new(t.reading_mode).strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(!scroll_on, egui::RichText::new(t.page_flip).size(16.0))
                            .clicked()
                        {
                            *set_scroll = Some(false);
                        }
                        if ui
                            .selectable_label(scroll_on, egui::RichText::new(t.scroll).size(16.0))
                            .clicked()
                        {
                            *set_scroll = Some(true);
                        }
                    });

                    ui.label(egui::RichText::new(t.reading_direction).strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(dir == Direction::Ltr, egui::RichText::new(t.dir_ltr).size(16.0))
                            .clicked()
                        {
                            *set_dir = Some(Direction::Ltr);
                        }
                        if ui
                            .selectable_label(dir == Direction::Rtl, egui::RichText::new(t.dir_rtl).size(16.0))
                            .clicked()
                        {
                            *set_dir = Some(Direction::Rtl);
                        }
                    });

                    ui.label(egui::RichText::new(t.page_layout).strong());
                    ui.horizontal(|ui| {
                        for (m, text) in [
                            (LayoutMode::Single, t.single),
                            (LayoutMode::Spread, t.two_page),
                            (LayoutMode::Auto, t.auto),
                        ] {
                            if ui
                                .selectable_label(layout == m, egui::RichText::new(text).size(16.0))
                                .clicked()
                            {
                                *set_layout = Some(m);
                            }
                        }
                    });
                    // Auto resolves to single in portrait and two-page in landscape.
                    if layout == LayoutMode::Auto {
                        ui.label(
                            egui::RichText::new(t.layout_auto_hint)
                                .size(12.0)
                                .color(egui::Color32::from_white_alpha(150)),
                        );
                    }
                    // Shift which pages are paired together (e.g. so a cover sits alone).
                    // Shown whenever the *effective* layout is a spread (incl. Auto-landscape).
                    if effective_spread
                        && ui
                            .button(egui::RichText::new(t.shift_page_pairing).size(16.0))
                            .clicked()
                    {
                        *toggle_offset = true;
                    }

                    ui.label(egui::RichText::new(t.fit).strong());
                    ui.horizontal(|ui| {
                        for (f, text) in [
                            (FitMode::Window, t.fit_window_opt),
                            (FitMode::Width, t.fit_width_opt),
                            (FitMode::Height, t.fit_height_opt),
                            (FitMode::Actual, "1:1"),
                        ] {
                            if ui
                                .selectable_label(fit == f, egui::RichText::new(text).size(16.0))
                                .clicked()
                            {
                                *set_fit = Some(f);
                            }
                        }
                    });

                    // Sits directly under the fit choices because it *modifies* them:
                    // it bounds every fit at 100% native rather than being a fit of its
                    // own (issue #13). Same row and wording as the desktop panel.
                    ui.label(egui::RichText::new(t.stretch_small_pages).strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(stretch_on, egui::RichText::new(t.on).size(16.0))
                            .clicked()
                        {
                            *set_no_upscale = Some(false);
                        }
                        if ui
                            .selectable_label(!stretch_on, egui::RichText::new(t.off).size(16.0))
                            .clicked()
                        {
                            *set_no_upscale = Some(true);
                        }
                    });
                    ui.label(
                        egui::RichText::new(t.stretch_hint)
                            .size(12.0)
                            .color(egui::Color32::from_white_alpha(150)),
                    );

                    // Manual page rotation (single-page draws only — the engine ignores
                    // it for spreads). Per-comic and transient: not persisted, reset to
                    // upright on the next open, exactly like the desktop `R` key.
                    ui.label(egui::RichText::new(t.rotation).strong());
                    if ui
                        .button(egui::RichText::new((t.rotate_now)(rotation as u32 * 90)).size(16.0))
                        .clicked()
                    {
                        *rotate = true;
                    }

                    ui.label(egui::RichText::new(t.page_turn_animation).strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(transition_on, egui::RichText::new(t.on).size(16.0))
                            .clicked()
                        {
                            *set_transition = Some(true);
                        }
                        if ui
                            .selectable_label(!transition_on, egui::RichText::new(t.off).size(16.0))
                            .clicked()
                        {
                            *set_transition = Some(false);
                        }
                    });

                    // Book-gutter shading only means anything on an un-joined pair, so
                    // show it with the other spread-only controls' gate.
                    if effective_spread {
                        ui.label(egui::RichText::new(t.spine_shadow).strong());
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(spine_on, egui::RichText::new(t.on).size(16.0))
                                .clicked()
                            {
                                *set_spine_on = Some(true);
                            }
                            if ui
                                .selectable_label(!spine_on, egui::RichText::new(t.off).size(16.0))
                                .clicked()
                            {
                                *set_spine_on = Some(false);
                            }
                        });
                        let mut strength = spine_strength;
                        if ui
                            .add_enabled(
                                spine_on,
                                egui::Slider::new(&mut strength, 0.0..=1.0)
                                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                    .text(t.strength),
                            )
                            .changed()
                        {
                            *set_spine_strength = Some(strength);
                        }
                    }

                    ui.label(egui::RichText::new(t.resume_on_startup).strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(resume_on, egui::RichText::new(t.on).size(16.0))
                            .clicked()
                        {
                            *set_resume = Some(true);
                        }
                        if ui
                            .selectable_label(!resume_on, egui::RichText::new(t.off).size(16.0))
                            .clicked()
                        {
                            *set_resume = Some(false);
                        }
                    });

                    // Hardware page turning (issue #16). Arrow / PageUp / PageDown /
                    // media keys always flip; the volume rocker is opt-in because
                    // it takes the buttons away from the volume while a book is open.
                    ui.label(egui::RichText::new(t.vol_keys).strong());
                    ui.horizontal(|ui| {
                        for (v, text) in [
                            (VolKeys::Off, t.off),
                            (VolKeys::DownNext, t.vol_down_next),
                            (VolKeys::UpNext, t.vol_up_next),
                        ] {
                            if ui
                                .selectable_label(vol_keys == v, egui::RichText::new(text).size(16.0))
                                .clicked()
                            {
                                *set_vol_keys = Some(v);
                            }
                        }
                    });
                    ui.label(
                        egui::RichText::new(t.vol_keys_hint)
                            .size(12.0)
                            .color(egui::Color32::from_white_alpha(150)),
                    );

                    // A dropdown, not a row of buttons: the list only grows. Each
                    // language is listed in its own script, so it's findable whatever
                    // the UI is currently showing. Rows are padded for touch.
                    ui.label(egui::RichText::new(t.language).strong());
                    egui::ComboBox::from_id_salt("language")
                        .selected_text(
                            egui::RichText::new(i18n::lang_pref_label(lang_pref)).size(16.0),
                        )
                        .width(260.0)
                        .show_ui(ui, |ui| {
                            ui.spacing_mut().button_padding = egui::vec2(16.0, 12.0);
                            let item = |text: &str| egui::RichText::new(text).size(18.0);
                            if ui.selectable_label(lang_pref.is_none(), item(t.lang_system)).clicked() {
                                *set_lang = Some(None);
                            }
                            for l in Lang::ALL {
                                if ui.selectable_label(lang_pref == Some(l), item(l.native_name())).clicked() {
                                    *set_lang = Some(Some(l));
                                }
                            }
                        });

                    ui.label(egui::RichText::new(t.theme).strong());
                    ui.horizontal(|ui| {
                        for (t, text) in [
                            (ThemePref::System, t.theme_system),
                            (ThemePref::Light, t.theme_light),
                            (ThemePref::Dark, t.theme_dark),
                        ] {
                            if ui
                                .selectable_label(theme == t, egui::RichText::new(text).size(16.0))
                                .clicked()
                            {
                                *set_theme = Some(t);
                            }
                        }
                    });

                    // One profile, not individual knobs: the budget's fields are
                    // interdependent. Auto = the hardware probe; the rest pin a tier
                    // (Performance = no ceilings, i.e. the desktop configuration).
                    ui.label(egui::RichText::new(t.performance).strong());
                    ui.horizontal(|ui| {
                        for (p, text) in [
                            (PerfPref::Auto, t.perf_auto),
                            (PerfPref::Low, t.perf_low),
                            (PerfPref::Mid, t.perf_mid),
                            (PerfPref::High, t.perf_high),
                        ] {
                            if ui
                                .selectable_label(perf == p, egui::RichText::new(text).size(16.0))
                                .clicked()
                            {
                                *set_perf = Some(p);
                            }
                        }
                    });
                    ui.label(
                        egui::RichText::new(t.perf_hint)
                        .size(12.0)
                        .color(egui::Color32::from_white_alpha(150)),
                    );
                });
            });
        });
}

/// Decode the embedded yosh mascot logo for the "no comic open" card. The PNG is
/// transparent; the engine's decode premultiplies its alpha, so build the egui
/// image from premultiplied bytes (`from_rgba_unmultiplied` would double-multiply
/// and darken the anti-aliased edges).
pub(crate) fn decode_logo() -> Option<egui::ColorImage> {
    const LOGO: &[u8] = include_bytes!("../../yosh/assets/yosh.png");
    let mut resizer = fast_image_resize::Resizer::new();
    let decoded = yosh_engine::decode::decode_and_downscale(LOGO, 256, &mut resizer).ok()?;
    let rgba = yosh_engine::decode::to_rgba_image(decoded);
    Some(egui::ColorImage::from_rgba_premultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

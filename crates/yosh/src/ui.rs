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
    pub turbo_label: &'static str,

    // Requests raised by clicks, consumed by the app after the frame.
    pub req_toggle_dir: bool,
    pub req_cycle_fit: bool,
    pub req_toggle_layout: bool,
    pub req_toggle_present: bool,
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
    /// Whether to show the centered loading spinner (the current page's decode
    /// has been pending long enough to warrant feedback). Set by the app.
    pub loading: bool,
    /// Edge navigation arrows: set true while the cursor hovers the left/right
    /// page-flip strip (page-flip reader mode only). Set by the app.
    pub hover_left: bool,
    pub hover_right: bool,
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
                            "gif", "bmp", "avif",
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
            if ui.button(format!("Present: {}", st.turbo_label)).clicked() {
                st.req_toggle_present = true;
            }
            if ui.button("? Help").clicked() {
                st.help_open = !st.help_open;
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
                ui.label("I   show image info overlay");
                ui.label("T   present  vsync ↔ turbo");
                ui.label("F11   fullscreen      Esc   quit");
                ui.separator();
                ui.heading("Files");
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

    // Page-flip affordance: a chevron at whichever edge the cursor hovers.
    if st.hover_left {
        nav_arrow(ctx, "nav_arrow_left", egui::Align2::LEFT_CENTER, egui::vec2(20.0, 0.0), true);
    }
    if st.hover_right {
        nav_arrow(ctx, "nav_arrow_right", egui::Align2::RIGHT_CENTER, egui::vec2(-20.0, 0.0), false);
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

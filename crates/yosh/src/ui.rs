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

#[allow(deprecated)]
pub fn chrome(ctx: &egui::Context, st: &mut UiState, lib: &Library, library_view: bool) {
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
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
                ui.label("click left·right half — flip;   wheel — flip or pan");
                ui.separator();
                ui.heading("Layout");
                ui.label("S   single ↔ two-page spread");
                ui.label("O   shift spread pairing (fix wrong pairing)");
                ui.label("D   reading direction  RTL ↔ LTR");
                ui.label("C   continuous vertical scroll");
                ui.separator();
                ui.heading("View");
                ui.label("F   fit mode  (window / width / height)");
                ui.label("+ / − / 0   zoom in / out / reset;   drag — pan");
                ui.label("T   present  vsync ↔ turbo");
                ui.label("G   GPU downscale on/off");
                ui.label("F11   fullscreen");
                ui.separator();
                ui.heading("Files");
                ui.label("Open folder / Open file / Library;  Grid ↔ Reader");
                ui.label("drag a folder, archive, or image onto the window");
                ui.label("F1   toggle this help");
            });
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

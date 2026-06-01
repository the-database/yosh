//! egui chrome: open buttons, reading-mode toggles, page indicator.
//! The toggle buttons set request flags the app consumes after the egui frame.

use std::path::PathBuf;

#[derive(Default)]
pub struct UiState {
    /// Path the user requested to open this frame (consumed by the app).
    pub pending_open: Option<PathBuf>,
    /// Currently-open volume path, for display.
    pub opened: Option<PathBuf>,
    pub status: String,

    // Current mode labels, set by the app each frame for the toggle buttons.
    pub dir_label: &'static str,
    pub fit_label: &'static str,
    pub layout_label: &'static str,

    // Requests raised by button clicks, consumed by the app after the frame.
    pub req_toggle_dir: bool,
    pub req_cycle_fit: bool,
    pub req_toggle_layout: bool,
}

#[allow(deprecated)]
pub fn chrome(ctx: &egui::Context, st: &mut UiState) {
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Open folder…").clicked() {
                if let Some(p) = rfd::FileDialog::new().set_title("Open page folder").pick_folder()
                {
                    st.pending_open = Some(p);
                }
            }
            if ui.button("Open archive…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .set_title("Open comic archive")
                    .add_filter("Comic archives", &["cbz", "cbr", "zip", "rar", "7z", "cb7"])
                    .pick_file()
                {
                    st.pending_open = Some(p);
                }
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
}

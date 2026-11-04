//! The AI-upscaling profile editor (Settings → AI upscaling → Edit profiles…):
//! the three built-in profiles to look at or copy, and the nine custom slots to
//! fill — each an ordered rule list (first match wins, as `sr::rules`) of source
//! height, gray/colour, target scale and model. Edits go to a draft; Save hands
//! it to the app (`UiState::req_save_profiles`), which writes `rules.json` and
//! applies it to the running upscaler.

use std::path::PathBuf;

use yosh_engine::i18n::tr;
use yosh_engine::sr::rules::{SrRule, model_base, select};

use crate::mwf;
use crate::profiles::{self, Custom, ProfileId, Profiles, SLOTS};
use crate::ui::UiState;

/// The editor's state, kept in `UiState` while the window is open (and after it
/// closes with unsaved changes, so reopening resumes them).
pub struct Editor {
    pub open: bool,
    /// What `rules.json` holds, and the edited copy.
    pub saved: Profiles,
    pub draft: Profiles,
    pub selected: ProfileId,
    /// `models/`, and the models in it (precision variants under one name).
    pub models_dir: PathBuf,
    pub models: Vec<String>,
    /// The "try a page" inputs.
    pub try_h: u32,
    pub try_gray: bool,
    pub try_scale: f32,
    /// The last import/save/add report, and whether it's a problem.
    pub note: Option<(String, bool)>,
}

impl Editor {
    pub fn new(profiles: &Profiles, selected: ProfileId, models_dir: PathBuf) -> Self {
        let models = profiles::installed_models(&models_dir);
        Self {
            open: true,
            saved: profiles.clone(),
            draft: profiles.clone(),
            selected,
            models_dir,
            models,
            try_h: 1400,
            try_gray: true,
            try_scale: 1.5,
            note: None,
        }
    }

    /// The rules shown for the selection (a built-in's, a slot's, or none).
    fn rules(&self) -> Option<Vec<SrRule>> {
        match self.selected {
            ProfileId::Custom(n) => self.draft.slot(n).map(|c| c.rules.clone()),
            id => Some(profiles::builtin_rules(id)),
        }
    }

    fn first_empty_slot(&self) -> Option<u8> {
        (1..=SLOTS as u8).find(|&n| self.draft.slot(n).is_none())
    }

    fn fill(&mut self, n: u8, custom: Custom) {
        self.draft.slots[n as usize - 1] = Some(custom);
        self.selected = ProfileId::Custom(n);
    }

    fn import_into(&mut self, n: u8) {
        let t = tr();
        let Some(path) = rfd::FileDialog::new()
            .set_title(t.dlg_import_mwf)
            .add_filter("MangaJaNai workflow", &["mwf", "json"])
            .pick_file()
        else {
            return;
        };
        match mwf::load(&path) {
            Ok(imp) => {
                let mut msg = (t.mwf_imported)(&imp.name, imp.rules.len());
                let ignored: Vec<String> = imp
                    .ignored
                    .iter()
                    .map(|&(c, what)| match what {
                        mwf::Ignored::Width => (t.mwf_ignored_width)(c),
                        mwf::Ignored::PreResize => (t.mwf_ignored_resize)(c),
                    })
                    .collect();
                if !ignored.is_empty() {
                    msg = format!("{msg}\n{}", (t.mwf_ignored)(&ignored.join(", ")));
                }
                let missing = imp
                    .rules
                    .iter()
                    .filter_map(|r| r.model.as_deref())
                    .filter(|m| !profiles::model_installed(&self.models_dir, m))
                    .count();
                if missing > 0 {
                    msg = format!("{msg}\n{}", (t.mwf_missing_models)(missing));
                }
                self.fill(n, Custom { name: imp.name, rules: imp.rules });
                self.note = Some((msg, missing > 0));
            }
            Err(e) => self.note = Some(((t.mwf_failed)(&e), true)),
        }
    }

    fn add_model(&mut self) {
        let t = tr();
        let Some(path) = rfd::FileDialog::new()
            .set_title(t.dlg_add_model)
            .add_filter("ONNX", &["onnx"])
            .pick_file()
        else {
            return;
        };
        match profiles::add_model(&self.models_dir, &path) {
            Ok(name) => {
                self.models = profiles::installed_models(&self.models_dir);
                self.note = Some(((t.model_added)(&name), false));
            }
            Err(e) => self.note = Some(((t.model_add_failed)(&e), true)),
        }
    }
}

/// A rule matching every page, with `model`.
fn every_page(model: Option<String>) -> SrRule {
    SrRule { min_h: 0, max_h: 0, gray: true, color: true, min_scale: 0.0, max_scale: 0.0, model }
}

/// Draw the editor if it's open.
pub fn show(ctx: &egui::Context, st: &mut UiState) {
    let Some(ed) = st.profile_editor.as_mut() else {
        return;
    };
    if !ed.open {
        return;
    }
    let t = tr();
    let mut open = true;
    let mut save = false;
    egui::Window::new(t.profiles_title)
        .id(egui::Id::new("profile_editor"))
        .open(&mut open)
        .default_width(900.0)
        .default_height(560.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(200.0);
                    profile_list(ui, ed);
                });
                ui.separator();
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, true]).max_height(440.0).show(ui, |ui| {
                        profile_body(ui, ed);
                    });
                });
            });
            ui.separator();
            if let Some((msg, bad)) = &ed.note {
                let text = egui::RichText::new(msg).small();
                ui.label(if *bad { text.color(egui::Color32::from_rgb(220, 180, 100)) } else { text });
            }
            ui.horizontal(|ui| {
                if ui.button(t.add_model).on_hover_text(t.add_model_hint).clicked() {
                    ed.add_model();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dirty = ed.draft != ed.saved;
                    if ui.add_enabled(dirty, egui::Button::new(t.profiles_save)).clicked() {
                        save = true;
                    }
                    if ui.add_enabled(dirty, egui::Button::new(t.profiles_revert)).clicked() {
                        ed.draft = ed.saved.clone();
                        ed.note = None;
                    }
                    if dirty {
                        ui.label(egui::RichText::new(t.profiles_unsaved).weak().small());
                    }
                });
            });
        });
    if save {
        st.req_save_profiles = Some(ed.draft.clone());
    }
    if !open {
        if ed.draft == ed.saved {
            st.profile_editor = None;
        } else {
            ed.open = false;
        }
    }
}

fn profile_list(ui: &mut egui::Ui, ed: &mut Editor) {
    let t = tr();
    ui.label(egui::RichText::new(t.profiles_builtin).strong());
    for id in ProfileId::BUILTIN {
        let text = format!("{}  {}", id.key(), ed.draft.name(id));
        if ui.selectable_label(ed.selected == id, text).clicked() {
            ed.selected = id;
        }
    }
    ui.add_space(6.0);
    ui.label(egui::RichText::new(t.profiles_custom).strong());
    for n in 1..=SLOTS as u8 {
        let id = ProfileId::Custom(n);
        let text = if ed.draft.exists(id) {
            egui::RichText::new(format!("{}  {}", id.key(), ed.draft.name(id)))
        } else {
            egui::RichText::new(format!("{}  {}", id.key(), t.profile_empty)).weak()
        };
        if ui.selectable_label(ed.selected == id, text).clicked() {
            ed.selected = id;
        }
    }
}

fn profile_body(ui: &mut egui::Ui, ed: &mut Editor) {
    let t = tr();
    match ed.selected {
        ProfileId::Custom(n) if ed.draft.slot(n).is_none() => {
            ui.label((t.profile_empty_slot)(n));
            ui.horizontal_wrapped(|ui| {
                for id in ProfileId::BUILTIN {
                    if ui.button(ed.draft.name(id)).clicked() {
                        let name = (t.profile_copy_of)(&ed.draft.name(id));
                        ed.fill(n, Custom { name, rules: profiles::builtin_rules(id) });
                    }
                }
                if ui.button(t.profile_one_model).clicked() {
                    let model = ed.models.first().cloned();
                    ed.fill(n, Custom { name: String::new(), rules: vec![every_page(model)] });
                }
                if ui.button(t.profile_import).on_hover_text(t.profile_import_hint).clicked() {
                    ed.import_into(n);
                }
            });
            return;
        }
        ProfileId::Custom(n) => {
            ui.horizontal(|ui| {
                ui.label(t.profile_name);
                let hint = (t.profile_custom)(n);
                if let Some(c) = ed.draft.slots[n as usize - 1].as_mut() {
                    ui.add(egui::TextEdit::singleline(&mut c.name).hint_text(hint).desired_width(220.0));
                }
                if ui.button(t.profile_import).on_hover_text(t.profile_import_hint).clicked() {
                    ed.import_into(n);
                }
                if ui.button(t.profile_clear).clicked() {
                    ed.draft.slots[n as usize - 1] = None;
                }
            });
            if ed.draft.slot(n).is_none() {
                return;
            }
        }
        id => {
            ui.label(egui::RichText::new(ed.draft.name(id)).strong());
            ui.label(egui::RichText::new(t.profile_builtin_note).weak().small());
            let slot = ed.first_empty_slot();
            let r = ui.add_enabled(slot.is_some(), egui::Button::new(t.profile_copy));
            let r = if slot.is_none() { r.on_disabled_hover_text(t.profile_slots_full) } else { r };
            if r.clicked()
                && let Some(n) = slot
            {
                let name = (t.profile_copy_of)(&ed.draft.name(id));
                ed.fill(n, Custom { name, rules: profiles::builtin_rules(id) });
                return;
            }
        }
    }
    ui.add_space(4.0);
    ui.label(egui::RichText::new(t.rules_hint).weak().small());
    rules_table(ui, ed);
    ui.add_space(6.0);
    try_line(ui, ed);
}

/// "0" shows as "any"; typing "any" (or nothing) gives 0.
fn any_fmt(v: f64, unit: &str) -> String {
    if v == 0.0 {
        tr().any.to_string()
    } else if unit.is_empty() {
        format!("{v:.0}")
    } else {
        format!("{v:.2}{unit}")
    }
}

fn any_parse(s: &str) -> Option<f64> {
    let s = s.trim().trim_end_matches(['×', 'x', 'X']).trim();
    if s.is_empty() || s.eq_ignore_ascii_case(tr().any) || s.eq_ignore_ascii_case("any") {
        Some(0.0)
    } else {
        s.replace(',', ".").parse().ok()
    }
}

fn rules_table(ui: &mut egui::Ui, ed: &mut Editor) {
    let t = tr();
    let editable = !ed.selected.builtin();
    let mut shown;
    let rules = match ed.selected {
        ProfileId::Custom(n) => match ed.draft.slots[n as usize - 1].as_mut() {
            Some(c) => &mut c.rules,
            None => return,
        },
        _ => {
            shown = profiles::builtin_rules(ed.selected);
            &mut shown
        }
    };
    let models = &ed.models;
    let models_dir = &ed.models_dir;
    let mut action: Option<(usize, i32)> = None; // (row, -1 up / 1 down / 0 remove)
    egui::Grid::new(("rules", ed.selected.token())).striped(true).num_columns(6).spacing([10.0, 4.0]).show(
        ui,
        |ui| {
            for h in ["#", t.col_pages, t.col_height, t.col_scale, t.col_model, ""] {
                ui.label(egui::RichText::new(h).weak().small());
            }
            ui.end_row();
            let len = rules.len();
            for (i, r) in rules.iter_mut().enumerate() {
                ui.label(format!("{}", i + 1));
                ui.add_enabled_ui(editable, |ui| {
                    let pages = match (r.gray, r.color) {
                        (true, true) => t.pages_both,
                        (true, false) => t.pages_gray,
                        (false, true) => t.pages_color,
                        (false, false) => t.pages_none,
                    };
                    egui::ComboBox::from_id_salt(("pages", i)).selected_text(pages).width(90.0).show_ui(ui, |ui| {
                        for (g, c, text) in
                            [(true, false, t.pages_gray), (false, true, t.pages_color), (true, true, t.pages_both)]
                        {
                            if ui.selectable_label(r.gray == g && r.color == c, text).clicked() {
                                (r.gray, r.color) = (g, c);
                            }
                        }
                    });
                });
                ui.add_enabled_ui(editable, |ui| {
                    ui.horizontal(|ui| {
                        for v in [&mut r.min_h, &mut r.max_h] {
                            ui.add(
                                egui::DragValue::new(v)
                                    .range(0..=100_000)
                                    .speed(10.0)
                                    .custom_formatter(|v, _| any_fmt(v, ""))
                                    .custom_parser(any_parse),
                            );
                        }
                    });
                });
                ui.add_enabled_ui(editable, |ui| {
                    ui.horizontal(|ui| {
                        for v in [&mut r.min_scale, &mut r.max_scale] {
                            ui.add(
                                egui::DragValue::new(v)
                                    .range(0.0..=16.0)
                                    .speed(0.05)
                                    .custom_formatter(|v, _| any_fmt(v, "×"))
                                    .custom_parser(any_parse),
                            );
                        }
                    });
                });
                ui.add_enabled_ui(editable, |ui| model_combo(ui, i, &mut r.model, models, models_dir));
                if editable {
                    ui.horizontal(|ui| {
                        if ui.add_enabled(i > 0, egui::Button::new("⏶").small()).on_hover_text(t.move_up).clicked() {
                            action = Some((i, -1));
                        }
                        if ui
                            .add_enabled(i + 1 < len, egui::Button::new("⏷").small())
                            .on_hover_text(t.move_down)
                            .clicked()
                        {
                            action = Some((i, 1));
                        }
                        if ui.small_button("✕").on_hover_text(t.remove_rule).clicked() {
                            action = Some((i, 0));
                        }
                    });
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        },
    );
    match action {
        Some((i, 0)) => {
            rules.remove(i);
        }
        Some((i, d)) => rules.swap(i, (i as i32 + d) as usize),
        None => {}
    }
    if editable && ui.button(format!("+ {}", t.add_rule)).clicked() {
        let model = rules.last().and_then(|r| r.model.clone()).or_else(|| models.first().cloned());
        rules.push(every_page(model));
    }
}

fn model_combo(
    ui: &mut egui::Ui,
    i: usize,
    model: &mut Option<String>,
    models: &[String],
    models_dir: &std::path::Path,
) {
    let t = tr();
    let missing = model.as_deref().is_some_and(|m| !profiles::model_installed(models_dir, m));
    let shown = match model.as_deref() {
        None => egui::RichText::new(t.model_none).italics(),
        Some(m) if missing => egui::RichText::new(format!("⚠ {m}")).color(egui::Color32::from_rgb(220, 180, 100)),
        Some(m) => egui::RichText::new(m),
    };
    let r = egui::ComboBox::from_id_salt(("model", i)).selected_text(shown).width(380.0).show_ui(ui, |ui| {
        if ui.selectable_label(model.is_none(), egui::RichText::new(t.model_none).italics()).clicked() {
            *model = None;
        }
        let current = model.as_deref().map(|m| model_base(m).to_string());
        for m in models {
            if ui.selectable_label(current.as_deref() == Some(m.as_str()), m).clicked() {
                *model = Some(m.clone());
            }
        }
    });
    if missing {
        r.response.on_hover_text(t.model_missing);
    }
}

fn try_line(ui: &mut egui::Ui, ed: &mut Editor) {
    let t = tr();
    let Some(rules) = ed.rules() else {
        return;
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(t.try_page);
        ui.add(egui::DragValue::new(&mut ed.try_h).range(1..=100_000).speed(10.0).suffix(" px"));
        egui::ComboBox::from_id_salt("try_gray")
            .selected_text(if ed.try_gray { t.pages_gray } else { t.pages_color })
            .width(80.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut ed.try_gray, true, t.pages_gray);
                ui.selectable_value(&mut ed.try_gray, false, t.pages_color);
            });
        ui.label(t.try_shown_at);
        ui.add(egui::DragValue::new(&mut ed.try_scale).range(1.01..=16.0).speed(0.05).suffix("×"));
        let picked = select(&rules, ed.try_h, ed.try_gray, ed.try_scale);
        ui.label(egui::RichText::new(format!("→ {}", picked.unwrap_or(t.try_none))).strong());
    });
}

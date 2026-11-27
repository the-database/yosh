//! The chrome's fonts: Phosphor icon glyphs plus the platform's CJK fallback.

use super::*;

/// The system CJK fallback font (`Noto Sans CJK`), loaded **once per process**.
///
/// egui's bundled fonts have no CJK glyphs, so Japanese comic and file names would
/// render as tofu squares without this. The catch is that the file is ~40 MB and a
/// resume rebuilds the whole egui context: re-reading and re-copying it on every
/// foreground was one of the biggest single costs on the resume path. `FontData` is
/// immutable and egui takes it behind an `Arc`, so every egui context this process
/// builds can be handed the same one. `None` ⇒ the device has no such font (older /
/// non-CJK builds), and the fallback is simply not installed.
///
/// The bytes are leaked once (process lifetime anyway) so each egui context — and
/// each language switch — can borrow them as a `FontData` for whichever face of
/// the collection the UI language wants, with no 31 MB copy.
fn cjk_font(platform: &dyn Platform, face: u32) -> Option<Arc<egui::FontData>> {
    static CJK_BYTES: std::sync::OnceLock<Option<&'static [u8]>> = std::sync::OnceLock::new();
    let bytes = (*CJK_BYTES.get_or_init(|| {
        let bytes = platform.cjk_font_bytes()?;
        log::info!("loaded CJK fallback font ({} KB)", bytes.len() / 1024);
        Some(&*Box::leak(bytes.into_boxed_slice()))
    }))?;
    let mut data = egui::FontData::from_static(bytes);
    data.index = face;
    Some(Arc::new(data))
}

/// Install the chrome fonts: Phosphor icon glyphs (seekbar buttons) plus the CJK
/// fallback in face `cjk_face`. Phosphor is installed unconditionally; egui's
/// bundled fonts have no CJK glyphs, so the system Noto Sans CJK goes in as a
/// fallback when present, for the UI's own CJK text and Japanese comic / file
/// names. Re-run on a language switch that changes the face.
pub(crate) fn install_fonts(ctx: &egui::Context, platform: &dyn Platform, cjk_face: u32) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Fill);
    if let Some(cjk) = cjk_font(platform, cjk_face) {
        fonts.font_data.insert("cjk".to_owned(), cjk);
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(fam).or_default().push("cjk".to_owned());
        }
    }
    ctx.set_fonts(fonts);
}

//! UI strings, one typed table per language.
//!
//! Every user-visible string in both shells (and the engine's toasts) is a field
//! of [`Strings`]; each language is a single `static` struct literal in its own
//! file (`en.rs`, …). That makes a missing translation a **compile error** rather
//! than a silent English fallback, and keeps the per-frame cost of a label to an
//! atomic load and a pointer — no hashing, no allocation for plain text.
//!
//! Text with values in it is a `fn` field, so each language owns its word order
//! (none of the shipped languages inflect for number, so a plain function covers
//! plurals too). Persistence tokens (`FitMode::label()` and friends) are *not*
//! UI text and stay English; display code goes through [`tr`] instead.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::layout::Layout;
use crate::page::FitMode;
use crate::reader::Direction;

mod de;
mod en;
mod es;
mod fr;
mod ja;
mod ko;
mod pt;
mod ru;
mod zh_hans;
mod zh_hant;

/// A UI language. `En` is the reference; the rest are community-correctable
/// drafts (see the header of each language file).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    Es,
    Pt,
    Fr,
    De,
    Ru,
    Ja,
    ZhHans,
    ZhHant,
    Ko,
}

impl Lang {
    /// Picker order.
    pub const ALL: [Lang; 10] = [
        Lang::En,
        Lang::Es,
        Lang::Pt,
        Lang::Fr,
        Lang::De,
        Lang::Ru,
        Lang::Ja,
        Lang::ZhHans,
        Lang::ZhHant,
        Lang::Ko,
    ];

    /// The language's own name — what the picker shows, whatever the UI language.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Es => "Español",
            Lang::Pt => "Português",
            Lang::Fr => "Français",
            Lang::De => "Deutsch",
            Lang::Ru => "Русский",
            Lang::Ja => "日本語",
            Lang::ZhHans => "简体中文",
            Lang::ZhHant => "繁體中文",
            Lang::Ko => "한국어",
        }
    }

    /// Persistence token (a BCP-47 tag).
    pub fn token(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Es => "es",
            Lang::Pt => "pt",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::Ru => "ru",
            Lang::Ja => "ja",
            Lang::ZhHans => "zh-Hans",
            Lang::ZhHant => "zh-Hant",
            Lang::Ko => "ko",
        }
    }

    /// Inverse of [`token`](Self::token); `None` for anything else (e.g. "system").
    pub fn from_token(tok: &str) -> Option<Lang> {
        Lang::ALL.into_iter().find(|l| l.token() == tok)
    }

    /// Best supported language for an OS locale tag, BCP-47 (`zh-Hant-TW`,
    /// `ja-JP`) or POSIX (`zh_TW.UTF-8`). Chinese splits on script, else region:
    /// Hant for an explicit `Hant` script or TW/HK/MO, Hans otherwise. Anything
    /// unsupported is English.
    pub fn from_tag(tag: &str) -> Lang {
        // POSIX → BCP-47-ish: drop the codeset/modifier, `_` → `-`.
        let tag = tag.split(['.', '@']).next().unwrap_or("").replace('_', "-");
        let mut parts = tag.split('-').map(|p| p.to_ascii_lowercase());
        match parts.next().as_deref() {
            Some("es") => Lang::Es,
            Some("pt") => Lang::Pt,
            Some("fr") => Lang::Fr,
            Some("de") => Lang::De,
            Some("ru") => Lang::Ru,
            Some("ja") => Lang::Ja,
            Some("ko") => Lang::Ko,
            Some("zh") => {
                let rest: Vec<String> = parts.collect();
                let has = |x: &str| rest.iter().any(|p| p == x);
                if has("hant") || (!has("hans") && (has("tw") || has("hk") || has("mo"))) {
                    Lang::ZhHant
                } else {
                    Lang::ZhHans
                }
            }
            _ => Lang::En,
        }
    }

    /// Face of the system Noto Sans CJK collection (`NotoSansCJK-Regular.ttc`) that
    /// draws this language's glyph forms — the same index Android's fonts.xml uses.
    /// The non-CJK languages show CJK only in file names, most often Japanese.
    pub fn cjk_face(self) -> u32 {
        match self {
            Lang::En | Lang::Es | Lang::Pt | Lang::Fr | Lang::De | Lang::Ru | Lang::Ja => 0,
            Lang::Ko => 1,
            Lang::ZhHans => 2,
            Lang::ZhHant => 3,
        }
    }

    fn strings(self) -> &'static Strings {
        match self {
            Lang::En => &en::EN,
            Lang::Es => &es::ES,
            Lang::Pt => &pt::PT,
            Lang::Fr => &fr::FR,
            Lang::De => &de::DE,
            Lang::Ru => &ru::RU,
            Lang::Ja => &ja::JA,
            Lang::ZhHans => &zh_hans::ZH_HANS,
            Lang::ZhHant => &zh_hant::ZH_HANT,
            Lang::Ko => &ko::KO,
        }
    }
}

/// The active language, as `Lang as u8`. A global because every corner of both
/// shells (and the engine's toasts, raised deep in the reader) draws text.
static CURRENT: AtomicU8 = AtomicU8::new(Lang::En as u8);

/// Switch the UI language. Takes effect on the next string lookup — text drawn
/// every frame follows immediately; text a shell caches (e.g. an info overlay)
/// is the shell's to rebuild.
pub fn set_lang(lang: Lang) {
    CURRENT.store(lang as u8, Ordering::Relaxed);
}

/// What the language picker shows for a preference (`None` = follow the OS):
/// the language's own name, or "System (日本語)" — naming what the OS resolved
/// to, so following the system never hides which language you actually got.
pub fn lang_pref_label(pref: Option<Lang>) -> String {
    match pref {
        Some(l) => l.native_name().to_string(),
        None => format!("{} ({})", tr().lang_system, lang().native_name()),
    }
}

/// The active UI language.
pub fn lang() -> Lang {
    let v = CURRENT.load(Ordering::Relaxed);
    Lang::ALL.into_iter().find(|l| *l as u8 == v).unwrap_or(Lang::En)
}

/// One section of the F1 key help: a heading and its `(keys, what it does)` rows.
pub struct HelpSection {
    pub title: &'static str,
    pub rows: &'static [(&'static str, &'static str)],
}

/// Every user-visible UI string. Grouped by where it shows up; the shell-specific
/// groups are simply unused by the other shell.
pub struct Strings {
    // --- common: toggles, reading options (desktop Settings + Android options) ---
    pub on: &'static str,
    pub off: &'static str,
    pub reading_mode: &'static str,
    pub page_flip: &'static str,
    pub scroll: &'static str,
    pub reading_direction: &'static str,
    /// Direction picker choices, arrow included.
    pub dir_ltr: &'static str,
    pub dir_rtl: &'static str,
    pub page_layout: &'static str,
    pub single: &'static str,
    pub two_page: &'static str,
    pub auto: &'static str,
    pub shift_page_pairing: &'static str,
    pub fit: &'static str,
    pub fit_window_opt: &'static str,
    pub fit_width_opt: &'static str,
    pub fit_height_opt: &'static str,
    pub stretch_small_pages: &'static str,
    pub rotation: &'static str,
    /// The rotate button: "Rotate 90° ⟳ · now {deg}°".
    pub rotate_now: fn(u32) -> String,
    pub page_turn_animation: &'static str,
    pub strength: &'static str,
    pub resume_on_startup: &'static str,
    pub theme: &'static str,
    pub theme_system: &'static str,
    pub theme_light: &'static str,
    pub theme_dark: &'static str,
    pub performance: &'static str,
    pub perf_auto: &'static str,
    pub perf_low: &'static str,
    pub perf_mid: &'static str,
    pub perf_high: &'static str,
    /// The SR add-on section of desktop Settings.
    pub ai_upscaling: &'static str,
    pub upscale_pages: &'static str,
    pub upscale_hint: &'static str,
    pub upscale_not_installed: &'static str,
    pub upscale_starting: &'static str,
    /// "Ready · {device}".
    pub upscale_ready: fn(&str) -> String,
    /// "Unavailable: {error}".
    pub upscale_error: fn(&str) -> String,
    /// "TensorRT-RTX failed ({why}); using {backend} instead" (TensorRT, or
    /// DirectML when Auto tried TensorRT-RTX and TensorRT didn't start either).
    pub upscale_fallback: fn(&str, &str) -> String,
    pub open_addon_folder: &'static str,
    /// Settings → AI upscaling: the button that installs the add-on (`mangajanai setup`), and
    /// the line under it.
    pub addon_install: &'static str,
    pub addon_install_hint: &'static str,
    /// The button when a newer add-on release is pinned; `{v}` is its version ("v0.3.1").
    pub addon_update: fn(&str) -> String,
    /// After Update while the add-on is loaded (its files can't be replaced until a restart).
    pub addon_update_restart: &'static str,
    /// The add-on lacks files for a GPU here (set up on another PC or GPU): the button that
    /// re-runs setup, and the line under it.
    pub addon_install_gpu: &'static str,
    pub addon_install_gpu_hint: &'static str,
    /// The install's progress bar, by step.
    pub addon_preparing: &'static str,
    pub addon_dl_installer: &'static str,
    pub addon_dl_runtime: &'static str,
    pub addon_dl_models: &'static str,
    pub addon_unpacking: &'static str,
    pub addon_failed: fn(&str) -> String,
    pub addon_cancel: &'static str,
    /// Remove the add-on: the button, its confirmation (`gb` = the folder's size), the
    /// confirming button, and the note when it waits for the next start.
    pub addon_remove: &'static str,
    pub addon_remove_confirm: fn(f32) -> String,
    pub addon_remove_yes: &'static str,
    pub addon_remove_restart: &'static str,
    /// Settings → AI upscaling → Storage, shown when the add-on's files are in the shared pack
    /// store (MangaJaNaiCLI `store`): the heading and the line under it.
    pub storage: &'static str,
    pub storage_hint: &'static str,
    /// Under the folder's path: in the default place, moved from `{root}`, or missing.
    pub storage_default: &'static str,
    pub storage_moved: fn(&str) -> String,
    pub storage_missing: &'static str,
    /// The folder's size, after its path.
    pub storage_size: fn(f32) -> String,
    /// The buttons: Move…, Move back, and (folder missing) Try again / Reset.
    pub storage_move: &'static str,
    pub storage_move_back: &'static str,
    pub storage_retry: &'static str,
    pub storage_reset: &'static str,
    /// The folder picker's title.
    pub dlg_storage: &'static str,
    /// The confirmations: Move (`gb`, the new folder), Move back (`gb`, the default folder),
    /// Reset (the missing folder); and the confirming buttons.
    pub storage_move_confirm: fn(f32, &str) -> String,
    pub storage_back_confirm: fn(f32, &str) -> String,
    pub storage_reset_confirm: fn(&str) -> String,
    pub storage_move_yes: &'static str,
    pub storage_reset_yes: &'static str,
    /// A move's progress bar (`done`, `total` GB), before its copy starts, and a reset.
    pub storage_moving: fn(f32, f32) -> String,
    pub storage_starting: &'static str,
    pub storage_resetting: &'static str,
    /// A move asked for while the add-on is loaded: it runs at the next start.
    pub storage_restart: &'static str,
    pub storage_failed: fn(&str) -> String,
    /// AI upscaling can't start: the shared folder `{folder}` is missing.
    pub upscale_store_missing: fn(&str) -> String,
    // --- desktop: AI-upscaling profiles (Settings, the profile editor, the AI chip) ---
    /// The profile editor window's title.
    pub profiles_title: &'static str,
    /// Built-in profile names (Shift+1/2/3).
    pub profile_quality: &'static str,
    pub profile_balanced: &'static str,
    pub profile_performance: &'static str,
    /// An unnamed custom slot: "Custom {n}".
    pub profile_custom: fn(u8) -> String,
    /// The editor's list headings.
    pub profiles_builtin: &'static str,
    pub profiles_custom: &'static str,
    /// An empty custom slot in the list.
    pub profile_empty: &'static str,
    /// "Custom slot {n} is empty. Start it from:" (then buttons).
    pub profile_empty_slot: fn(u8) -> String,
    /// A copied profile's name: "Copy of {name}".
    pub profile_copy_of: fn(&str) -> String,
    pub profile_one_model: &'static str,
    pub profile_import: &'static str,
    pub profile_import_hint: &'static str,
    pub profile_name: &'static str,
    pub profile_clear: &'static str,
    pub profile_builtin_note: &'static str,
    pub profile_copy: &'static str,
    pub profile_slots_full: &'static str,
    /// How the rule table reads.
    pub rules_hint: &'static str,
    /// A rule bound of 0: no limit (short, shown in a number field).
    pub any: &'static str,
    /// Rule table column headings.
    pub col_pages: &'static str,
    pub col_height: &'static str,
    pub col_scale: &'static str,
    pub col_model: &'static str,
    /// The rule table's auto-levels column, and its hover text.
    pub col_levels: &'static str,
    pub levels_hint: &'static str,
    /// Which pages a rule applies to.
    pub pages_gray: &'static str,
    pub pages_color: &'static str,
    pub pages_both: &'static str,
    pub pages_none: &'static str,
    /// Rule row buttons.
    pub move_up: &'static str,
    pub move_down: &'static str,
    pub remove_rule: &'static str,
    pub add_rule: &'static str,
    /// A rule's model choice that skips upscaling.
    pub model_none: &'static str,
    pub model_missing: &'static str,
    /// The "try a page" line: "Try a page: [1400 px] [Gray] shown at [1.5×] → model".
    pub try_page: &'static str,
    pub try_shown_at: &'static str,
    pub try_none: &'static str,
    pub add_model: &'static str,
    pub add_model_hint: &'static str,
    pub profiles_save: &'static str,
    pub profiles_revert: &'static str,
    pub profiles_unsaved: &'static str,
    pub profiles_save_failed: fn(&str) -> String,
    /// File dialog titles.
    pub dlg_import_mwf: &'static str,
    pub dlg_add_model: &'static str,
    /// "Imported “{name}”: {count} rules."
    pub mwf_imported: fn(&str, usize) -> String,
    /// One chain setting yosh doesn't carry over ("chain {n}: …").
    pub mwf_ignored_width: fn(usize) -> String,
    pub mwf_ignored_resize: fn(usize) -> String,
    /// "chain {n}: tile size {px} px (yosh uses 512 px)".
    pub mwf_ignored_tile: fn(usize, u32) -> String,
    /// The workflow's grayscale detection threshold, when it isn't yosh's 12.
    pub mwf_ignored_threshold: fn(i32) -> String,
    /// "Not carried over (yosh has no such setting): {list}".
    pub mwf_ignored: fn(&str) -> String,
    pub mwf_missing_models: fn(usize) -> String,
    pub mwf_failed: fn(&str) -> String,
    pub model_added: fn(&str) -> String,
    pub model_add_failed: fn(&str) -> String,
    /// Settings → AI upscaling: the profile yosh starts on.
    pub upscale_profile: &'static str,
    pub edit_profiles: &'static str,
    pub upscale_profile_hint: &'static str,
    /// The AI chip's tooltip: "Profile: {name}".
    pub sr_profile: fn(&str) -> String,
    pub sr_profile_off: &'static str,
    pub sr_chip_menu_hint: &'static str,
    /// Toasts for a profile switch.
    pub toast_profile: fn(&str) -> String,
    pub toast_profile_empty: fn(u8) -> String,
    /// The reader's "AI" status chip and busy indicator (what upscaling is doing
    /// for the page in view).
    pub sr_working: &'static str,
    pub sr_preparing: &'static str,
    pub sr_preparing_tip: &'static str,
    pub sr_done: &'static str,
    pub sr_not_needed: &'static str,
    pub sr_unsupported: &'static str,
    pub sr_no_model: &'static str,
    /// "AI upscaling failed: {error}".
    pub sr_failed: fn(&str) -> String,
    pub sr_book_off: &'static str,

    // --- common: library, empty states, navigation ---
    pub open_file: &'static str,
    pub no_comic_open: &'static str,
    pub library: &'static str,
    pub change_library: &'static str,
    pub scanning_library: &'static str,
    pub no_comics_found: &'static str,
    pub recently_read: &'static str,
    /// Series status: some volumes started, none unread / some unread / all read.
    pub reading: &'static str,
    pub reading_unread: fn(usize) -> String,
    pub unread: fn(usize) -> String,
    pub finished: &'static str,
    /// The way back to the open book from the library ("Resume" / "Resume {title}").
    pub resume: &'static str,
    pub resume_book: fn(&str) -> String,

    // --- common: engine toasts ---
    pub first_page: &'static str,
    pub last_page: &'static str,
    pub fit_window: &'static str,
    pub fit_width: &'static str,
    pub fit_height: &'static str,
    /// Zoom toast: the percentage, plus the fit level it landed on (if any) on a
    /// second line — keep the `\n`, the toast layout depends on it.
    pub zoom_toast: fn(f32, Option<&str>) -> String,

    // --- common: info overlay row keys (the values stay technical / English) ---
    pub info_file: &'static str,
    pub info_page: &'static str,
    pub info_size: &'static str,
    pub info_modified: &'static str,
    pub info_resolution: &'static str,
    pub info_format: &'static str,
    pub info_color: &'static str,
    pub info_zoom: &'static str,
    pub info_resize: &'static str,
    pub info_ai_model: &'static str,
    pub info_lq_tier: &'static str,

    // --- desktop: top bar ---
    pub open_folder: &'static str,
    /// Short current-state names on the top-bar buttons ("Dir: RTL", "Fit: width").
    pub dir_short_ltr: &'static str,
    pub dir_short_rtl: &'static str,
    pub fit_short_window: &'static str,
    pub fit_short_width: &'static str,
    pub fit_short_height: &'static str,
    pub layout_short_single: &'static str,
    pub layout_short_spread: &'static str,
    pub layout_short_scroll: &'static str,
    pub dir_button: fn(&str) -> String,
    pub fit_button: fn(&str) -> String,
    pub layout_button: fn(&str) -> String,
    pub settings: &'static str,
    pub settings_tip: &'static str,
    pub help: &'static str,
    pub updating: &'static str,
    pub update_failed: &'static str,
    pub update_failed_tip: &'static str,
    pub update_to: fn(&str) -> String,
    pub update_tip: fn(&str) -> String,
    pub no_volume_open: &'static str,
    /// Page-indicator suffix when the page failed to decode.
    pub status_failed: &'static str,
    pub no_images_found: &'static str,
    pub open_failed: fn(&str) -> String,
    pub title_library: &'static str,

    // --- desktop: F1 help ---
    pub help_title: &'static str,
    pub help_sections: &'static [HelpSection],

    // --- desktop: Settings window ---
    pub scroll_speed: &'static str,
    pub scroll_speed_hint: &'static str,
    pub pairing_tip: &'static str,
    pub stretch_hint_key: &'static str,
    pub spine_shadow_two_page: &'static str,
    pub spine_hint: &'static str,
    /// What `Auto` resolves to right now: tier name, and whether on battery.
    pub perf_auto_now: fn(&str, bool) -> String,

    // --- desktop: onboarding, library header, file dialogs ---
    pub onboarding_sub: &'static str,
    pub pick_comics_folder: &'static str,
    pub pick_comics_folder_tip: &'static str,
    pub press_f1: &'static str,
    pub rescan: &'static str,
    pub rescan_tip: &'static str,
    pub change_library_tip: &'static str,
    pub dlg_open_folder: &'static str,
    pub dlg_open_file: &'static str,
    pub dlg_filter: &'static str,
    pub dlg_library: &'static str,

    // --- desktop: overlays + animation panel ---
    pub loading: &'static str,
    pub couldnt_open: fn(&str) -> String,
    pub info_touch: &'static str,
    pub play: &'static str,
    pub pause: &'static str,
    pub prev_frame: &'static str,
    pub next_frame: &'static str,
    pub hide_anim: &'static str,

    // --- desktop: toasts ---
    pub toast_direction: fn(&str) -> String,
    pub toast_layout: fn(&str) -> String,
    pub toast_scroll_mode: &'static str,
    pub toast_page_flip_mode: &'static str,
    pub toast_seekbar_on: &'static str,
    pub toast_seekbar_off: &'static str,
    pub toast_transition_on: &'static str,
    pub toast_transition_off: &'static str,
    pub toast_spread_offset: fn(usize) -> String,
    pub toast_rotation: fn(u32) -> String,
    pub toast_stretch_on: &'static str,
    pub toast_stretch_off: &'static str,
    pub toast_upscale_on: &'static str,
    pub toast_upscale_off: &'static str,
    pub toast_upscale_unavailable: &'static str,
    pub toast_spine_on: &'static str,
    pub toast_spine_off: &'static str,
    pub toast_nothing_open: &'static str,
    pub toast_shown_in_explorer: &'static str,
    pub toast_first_book: &'static str,
    pub toast_last_book: &'static str,
    /// Preset toasts: "Spread, {dir}" and "{fit} (single)".
    pub toast_spread: fn(&str) -> String,
    pub toast_single: fn(&str) -> String,
    pub toast_partial_archive: fn(usize) -> String,
    pub toast_performance_restored: &'static str,

    // --- android ---
    pub info: &'static str,
    pub close: &'static str,
    pub open_comic_to_start: &'static str,
    pub browse_library: &'static str,
    pub tap_top_tip: &'static str,
    pub opening: &'static str,
    pub next_book: &'static str,
    pub previous_book: &'static str,
    pub tap_again_to_open: &'static str,
    pub zone_next: &'static str,
    pub zone_prev: &'static str,
    pub layout_auto_hint: &'static str,
    pub stretch_hint: &'static str,
    pub spine_shadow: &'static str,
    pub vol_keys: &'static str,
    pub vol_down_next: &'static str,
    pub vol_up_next: &'static str,
    pub vol_keys_hint: &'static str,
    pub perf_hint: &'static str,
    pub grant_prompt: &'static str,
    pub grant_button: &'static str,
    pub back: &'static str,
    pub up: &'static str,
    pub set_as_library: &'static str,
    pub scanning: &'static str,
    pub no_subfolders: &'static str,
    pub info_book: &'static str,
    pub info_screen: &'static str,
    pub info_source: &'static str,
    pub info_decoded: &'static str,
    pub info_gpu: &'static str,
    pub info_layout: &'static str,
    pub info_direction: &'static str,

    // --- language picker (both shells) ---
    pub language: &'static str,
    /// The "follow the OS language" choice.
    pub lang_system: &'static str,
}

/// The active language's strings.
#[inline]
pub fn tr() -> &'static Strings {
    lang().strings()
}

/// Display columns of a char: 2 for East Asian wide/fullwidth (CJK, kana, Hangul,
/// fullwidth forms), else 1. An approximation of UAX #11 that covers what shows
/// up in comic titles.
fn char_cols(c: char) -> usize {
    match c as u32 {
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

/// Clip `s` to at most `cols` display columns (wide chars count 2), appending
/// "…" when anything was cut. Keeps a header button sized the same whether the
/// title is Latin or CJK — a char count lets a CJK title run twice as wide.
pub fn truncate_cols(s: &str, cols: usize) -> String {
    let total: usize = s.chars().map(char_cols).sum();
    if total <= cols {
        return s.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in s.chars() {
        let w = char_cols(c);
        if used + w > cols {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push('…');
    out
}

/// Short display name of a reading direction (top bar, toasts).
pub fn dir_short(d: Direction) -> &'static str {
    match d {
        Direction::Ltr => tr().dir_short_ltr,
        Direction::Rtl => tr().dir_short_rtl,
    }
}

/// Short display name of a fit mode (top bar).
pub fn fit_short(f: FitMode) -> &'static str {
    let t = tr();
    match f {
        FitMode::Window => t.fit_short_window,
        FitMode::Width => t.fit_short_width,
        FitMode::Height => t.fit_short_height,
        FitMode::Actual => "1:1",
    }
}

/// Short display name of a concrete page layout (top bar, toasts).
pub fn layout_short(l: Layout) -> &'static str {
    match l {
        Layout::Single => tr().layout_short_single,
        Layout::Spread => tr().layout_short_spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_map_to_languages() {
        for (tag, want) in [
            ("ja-JP", Lang::Ja),
            ("ja", Lang::Ja),
            ("ko-KR", Lang::Ko),
            ("zh-CN", Lang::ZhHans),
            ("zh", Lang::ZhHans),
            ("zh-Hans-CN", Lang::ZhHans),
            ("zh-SG", Lang::ZhHans),
            ("zh-TW", Lang::ZhHant),
            ("zh-HK", Lang::ZhHant),
            ("zh-MO", Lang::ZhHant),
            ("zh-Hant", Lang::ZhHant),
            ("zh-Hant-HK", Lang::ZhHant),
            ("zh-Hans-HK", Lang::ZhHans),
            ("zh_TW.UTF-8", Lang::ZhHant),
            ("zh_CN.UTF-8", Lang::ZhHans),
            ("ja_JP.eucJP@x", Lang::Ja),
            ("es-ES", Lang::Es),
            ("es-419", Lang::Es),
            ("pt-BR", Lang::Pt),
            ("pt_PT.UTF-8", Lang::Pt),
            ("fr-FR", Lang::Fr),
            ("fr-CA", Lang::Fr),
            ("de-DE", Lang::De),
            ("de-AT", Lang::De),
            ("ru-RU", Lang::Ru),
            ("en-US", Lang::En),
            ("it-IT", Lang::En),
            ("C", Lang::En),
            ("", Lang::En),
        ] {
            assert_eq!(Lang::from_tag(tag), want, "{tag}");
        }
    }

    #[test]
    fn tokens_round_trip() {
        for l in Lang::ALL {
            assert_eq!(Lang::from_token(l.token()), Some(l));
        }
        assert_eq!(Lang::from_token("system"), None);
    }

    #[test]
    fn every_language_interpolates_its_values() {
        // Read each table directly (not via `tr()`), so this can't race other
        // tests over the global language.
        for l in Lang::ALL {
            let t = l.strings();
            assert!((t.unread)(7).contains('7'), "{l:?}");
            assert!((t.reading_unread)(7).contains('7'), "{l:?}");
            assert!((t.resume_book)("TITLE").contains("TITLE"), "{l:?}");
            assert!((t.rotate_now)(270).contains("270"), "{l:?}");
            assert!((t.toast_partial_archive)(42).contains("42"), "{l:?}");
            assert!((t.couldnt_open)("x.png").contains("x.png"), "{l:?}");
            assert!((t.update_to)("9.9.9").contains("9.9.9"), "{l:?}");
            assert!((t.perf_auto_now)(t.perf_high, true).contains(t.perf_high), "{l:?}");
            // The zoom toast keeps its line break: the toast layout counts it.
            let z = (t.zoom_toast)(125.0, Some(t.fit_width));
            assert!(z.contains("125.00") && z.contains('\n') && z.contains(t.fit_width), "{l:?}");
            assert!(!(t.zoom_toast)(125.0, None).contains('\n'), "{l:?}");
            // Storage: every folder and size shows up (the reset one names it twice).
            assert!((t.storage_moved)("C:/ROOT").contains("C:/ROOT"), "{l:?}");
            assert!((t.storage_size)(4.25).contains("4.2"), "{l:?}");
            let m = (t.storage_move_confirm)(4.25, "D:/TO");
            assert!(m.contains("4.2") && m.contains("D:/TO"), "{l:?}");
            let b = (t.storage_back_confirm)(4.25, "C:/ROOT");
            assert!(b.contains("4.2") && b.contains("C:/ROOT"), "{l:?}");
            assert_eq!((t.storage_reset_confirm)("D:/GONE").matches("D:/GONE").count(), 2, "{l:?}");
            let p = (t.storage_moving)(1.25, 4.25);
            assert!(p.contains("1.2") && p.contains("4.2"), "{l:?}");
            assert!((t.storage_failed)("ERR").contains("ERR"), "{l:?}");
            assert!((t.upscale_store_missing)("D:/GONE").contains("D:/GONE"), "{l:?}");
            // Same help shape as English, so no section silently goes missing.
            assert_eq!(t.help_sections.len(), en::EN.help_sections.len(), "{l:?}");
        }
    }

    #[test]
    fn truncate_counts_wide_chars_double() {
        assert_eq!(truncate_cols("short", 24), "short");
        assert_eq!(truncate_cols("abcdefgh", 5), "abcde…");
        // 6 CJK chars = 12 columns.
        assert_eq!(truncate_cols("かぐや様は告らせたい", 12), "かぐや様は告…");
        assert_eq!(truncate_cols("かぐや", 6), "かぐや");
        // Mixed: "ab" (2) + "漫画" (4) = 6; the next wide char would make 8.
        assert_eq!(truncate_cols("ab漫画です", 7), "ab漫画…");
    }
}

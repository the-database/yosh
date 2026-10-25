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

use crate::layout::Layout;
use crate::page::FitMode;
use crate::reader::Direction;

mod en;

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
}

/// The active language's strings.
#[inline]
pub fn tr() -> &'static Strings {
    &en::EN
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

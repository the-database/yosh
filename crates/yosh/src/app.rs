//! Application: winit `ApplicationHandler`, owns GPU + egui + reader state.
//! M1.3: async decode pool + bounded cache + forward prefetch → hitch-free
//! navigation. The current page is drawn from the cache; if a target isn't ready
//! yet the last-drawn page is held (no flicker).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{
    ElementState, KeyEvent, MouseButton, MouseScrollDelta, Touch, TouchPhase, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use fast_image_resize::Resizer;
use notify::Watcher as _; // brings the `.watch()` method into scope

use crate::config;
use crate::gpu::Gpu;
use crate::library::{cover_bytes, Library, VolKind};
use yosh_engine::gesture::{GestureCtx, GestureEvent, Phase, TouchGestures};
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::{DecodePool, Waker};
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, RarSource, SevenzSource, ZipSource};
use yosh_engine::layout::Layout;
use yosh_engine::reader::{Budget, DeviceTier, Direction, Reader, Viewport};
use yosh_engine::texpool::TexturePool;
use crate::ui::{self, UiState};
use crate::update;

/// Memory (MB) the reader may spend on its page cache + GPU textures. A slice of
/// system RAM on the desktop; an Android shell would pass its per-app heap class.
/// Falls back to a generous value (→ the full desktop budget) when RAM is unknown.
fn detect_mem_budget_mb() -> u64 {
    (total_ram_mb().unwrap_or(8192) / 16).max(64)
}

#[cfg(windows)]
fn total_ram_mb() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `s` is a correctly-sized, zeroed MEMORYSTATUSEX with dwLength set.
    (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some(s.ullTotalPhys / (1024 * 1024))
}

#[cfg(target_os = "linux")]
fn total_ram_mb() -> Option<u64> {
    // /proc/meminfo line: "MemTotal:   16384000 kB"
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = info
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb / 1024)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn total_ram_mb() -> Option<u64> {
    None
}

/// Is the machine running on battery right now? `None` = can't tell (no probe on
/// this platform, or the OS reports "unknown"), which every caller treats as AC —
/// the conservative choice, since guessing "battery" would silently throttle a
/// desktop. A machine with no battery at all answers `Some(false)`, so a tower
/// never pays for this beyond one cheap syscall.
#[cfg(windows)]
fn on_battery() -> Option<bool> {
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut s: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
    // SAFETY: `s` is a correctly-sized, zeroed SYSTEM_POWER_STATUS.
    if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
        return None;
    }
    // BatteryFlag bit 7 (128) = "no system battery" — a desktop PC, always on mains,
    // whatever `ACLineStatus` claims.
    if s.BatteryFlag & 128 != 0 {
        return Some(false);
    }
    match s.ACLineStatus {
        0 => Some(true),  // offline → battery
        1 => Some(false), // online → AC
        _ => None,        // 255 = unknown
    }
}

#[cfg(target_os = "linux")]
fn on_battery() -> Option<bool> {
    // /sys/class/power_supply/<name>/type == "Mains" is the AC adapter; its
    // `online` is 1 when plugged in. Same sysfs walk shape as the Android shell's
    // CPU-clock probe. A machine with no Mains node at all (desktop, container,
    // VM) counts as plugged in.
    let mut found_mains = false;
    let mut any_online = false;
    for entry in std::fs::read_dir("/sys/class/power_supply").ok()?.flatten() {
        let dir = entry.path();
        if !std::fs::read_to_string(dir.join("type")).is_ok_and(|t| t.trim() == "Mains") {
            continue;
        }
        found_mains = true;
        any_online |= std::fs::read_to_string(dir.join("online")).is_ok_and(|o| o.trim() == "1");
    }
    Some(found_mains && !any_online)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn on_battery() -> Option<bool> {
    None
}

/// The tier to actually run at: the pinned one if the user chose a profile, else
/// `Auto`'s rule — the uncapped desktop budget on mains, the `Mid` ceilings on
/// battery (fewer workers, smaller cache/prefetch, a bounded 1:1 decode), which is
/// the difference between reading and heating the laptop. A free function because
/// `resumed` needs it before a `State` exists.
fn effective_tier(perf: config::PerfPref, on_battery: bool) -> DeviceTier {
    perf.tier().unwrap_or(if on_battery {
        DeviceTier::Mid
    } else {
        DeviceTier::High
    })
}

/// How often `Auto` re-checks the power source. The probe is one syscall, so the
/// interval is belt-and-braces rather than a cost concern: it keeps the check off
/// the per-frame path during a fast seek, and a plug/unplug taking up to this long
/// to register is imperceptible for something that only changes decode aggression.
const POWER_RECHECK: Duration = Duration::from_secs(20);

/// Pixels scrolled per mouse-wheel line in continuous-scroll mode.
const SCROLL_WHEEL_PX: f32 = 110.0;
/// Library cover thumbnail height (decoded off-thread; see `pump_covers`).
const THUMB_H: u32 = 360;
/// Grace period before the centered loading spinner appears, so quick page
/// loads don't flash an indicator on every flip — only genuinely slow decodes
/// (e.g. seeking fast through very high-res pages) cross it.
const LOADING_INDICATOR_DELAY: Duration = Duration::from_millis(150);
/// Fraction of the window width on each side that flips pages on click (and
/// shows a hover arrow). The middle is reserved for double-click → fullscreen.
const EDGE_FRAC: f32 = 0.15;
/// Max gap between two middle-zone clicks to count as a double-click.
const DOUBLE_CLICK: Duration = Duration::from_millis(350);
/// How long a transient toast (boundary hit, zoom level) stays on screen.
const TOAST_DURATION: Duration = Duration::from_millis(1500);
/// Minimum time a zoomed-page scroll must dwell parked at the top/bottom edge
/// before a further scroll flips the page — turns the edge into a perceptible
/// hard stop instead of an instant jump to the next page.
const EDGE_DWELL: Duration = Duration::from_millis(350);
/// Live-refresh debounce: how quiet the watched folder/archive must go after a
/// change before the volume is rebuilt (see `poll_folder_watch`).
const WATCH_QUIET: Duration = Duration::from_millis(250);
/// Cap on that debounce: a *continuously* written archive never goes quiet, so
/// rebuild at least this often after the first event of a burst.
const WATCH_MAX_WAIT: Duration = Duration::from_millis(1000);

pub struct App {
    start_index: usize,
    state: Option<State>,
    /// Background-open channel, created before the event loop so a CLI-arg open
    /// can be spawned in `App::new` (pre-window, pre-GPU). `State` adopts both
    /// ends in `resumed`; every later `open()` reuses the same channel.
    open_tx: std::sync::mpsc::Sender<(u64, Built)>,
    open_rx: Option<std::sync::mpsc::Receiver<(u64, Built)>>,
    /// Some(path) when a CLI open is already in flight as generation 1.
    cli_open: Option<PathBuf>,
    /// Waker slot the pre-spawned open thread fires; filled by `resumed` the
    /// moment the window exists.
    early_waker: Arc<std::sync::OnceLock<Waker>>,
    /// Instrumentation stamp for the CLI open (handed to `State::open_t0`).
    open_t0: Option<Instant>,
}

/// Playback state for the animation (GIF/WebP) currently in view (BandiView-style
/// mini controls). `page` binds the controls to one page; when the viewed
/// animation changes it rebinds and resets. Non-controlled animations keep
/// free-running on the wall clock.
struct Playback {
    /// User hid the control panel (toggle with the key); playback continues.
    hidden: bool,
    /// Play/pause. Paused (or stepping) freezes on `frame`.
    playing: bool,
    /// The page index these controls govern (None when no animation is in view).
    page: Option<usize>,
    /// Current frame index (0-based).
    frame: usize,
    /// When `frame` was last advanced — drives play timing against frame delays.
    last: Instant,
}

impl Default for Playback {
    fn default() -> Self {
        Playback { hidden: false, playing: true, page: None, frame: 0, last: Instant::now() }
    }
}

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    /// The reading-state machine + the engine resources it drives (page source,
    /// decode pool, cache, texture pool, wgpu device/queue). The shell mirrors the
    /// surface into `reader.viewport` and feeds it input each frame.
    reader: Reader,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui: UiState,

    page_pipeline: PagePipeline,
    cursor_x: f64,
    cursor_y: f64,
    /// Ctrl down right now — the modifier that turns the wheel into a focal zoom.
    /// Tracked from `ModifiersChanged` and cleared on focus loss, since a Ctrl-held
    /// alt-tab releases the key while another window has focus and we'd never see it.
    ctrl_held: bool,
    /// Fractional wheel notches banked toward the next zoom-ladder step. A ladder
    /// step is inherently a whole thing, but a precision touchpad's Ctrl+two-finger
    /// scroll arrives as a stream of small `PixelDelta`s — without banking them the
    /// zoom would tear through the ladder. Reset whenever the wheel reverses.
    zoom_notches: f32,
    mouse_down: bool,
    drag_dist: f32, // accumulated drag distance, to distinguish click from pan
    cursor_in_window: bool, // gates the edge-hover navigation arrows
    last_mid_click: Option<Instant>, // middle-zone double-click → fullscreen

    /// The shared touch state machine (finger bookkeeping, lock thresholds,
    /// pinch, release velocities, glide starts) — the same one the Android shell
    /// drives. Its `samples` tracker doubles as the mouse drag's velocity window,
    /// so a strip thrown with the pointer glides exactly like one thrown with a
    /// finger, and its `tick` runs both inertial glides.
    gestures: TouchGestures,

    settings: config::Settings,
    /// Cached OS day/night flag (from `winit::window::Theme`), consulted when the
    /// theme preference is `System`. Seeded at construction and refreshed on
    /// `WindowEvent::ThemeChanged`.
    system_dark: bool,
    /// Last observed window geometry (outer x/y, inner w/h, physical px) while in
    /// the normal — not maximized, not fullscreen — state. Seeded from the saved
    /// settings so a session spent entirely maximized still persists the prior
    /// restored rect; updated on move/resize; written back on exit.
    win_geom: Option<(i32, i32, u32, u32)>,
    /// Whether the window was maximized, sampled alongside `win_geom` while not
    /// fullscreen. Persisted instead of a live `is_maximized()` so quitting from
    /// fullscreen still restores the maximized state underneath it.
    win_maximized: bool,
    /// Whether the decode pool is currently frozen because the window can't be
    /// seen (minimized / fully occluded). Purely for idempotence: the park and
    /// unpark signals both repeat, so `park`/`unpark` act only on a transition.
    parked: bool,

    // Performance profile. The two probe results are captured once at startup
    // (they can't change while the process runs) so the budget can be rebuilt at
    // any time — a picker change, or the power source flipping under `Auto`.
    /// Memory (MB) the reader may spend on decoded pages + GPU textures.
    mem_budget_mb: u64,
    /// Logical CPUs, as `Budget` wants them.
    cpus: usize,
    /// Last probed power source (`on_battery()`, `None` → treated as AC).
    on_battery: bool,
    /// When that probe last ran — `Auto` re-checks at most every `POWER_RECHECK`.
    power_checked: Instant,
    /// The tier the live `Reader` is currently configured for, so `apply_perf` can
    /// skip the work when the effective tier hasn't actually moved (re-clicking the
    /// selected option, or a power re-check that found no change).
    applied_tier: DeviceTier,
    volume_key: Option<String>,
    /// Visible page(s) the Tab info overlay text was built for, as
    /// `Reader::visible_pages` reports them (None = rebuild needed).
    info_for: Option<(usize, Option<usize>)>,
    /// The full info-overlay rows are gathered off-thread: they need `read_page` +
    /// `modified`, which block on a spun-down disk (and, on a RAR still
    /// decompressing, until that entry is extracted). The overlay shows no-I/O
    /// placeholder rows meanwhile, replaced when a result lands in `poll_background`.
    info_tx: std::sync::mpsc::Sender<InfoRows>,
    info_rx: std::sync::mpsc::Receiver<InfoRows>,
    /// The anchor page currently waiting to decode and when that wait began,
    /// used to delay the loading spinner *per page* (so a fast page reached at
    /// the end of a slow-seek streak still gets its own grace period). None as
    /// soon as the anchor is ready.
    loading_pending: Option<(usize, Instant)>,
    /// Transient on-screen message (boundary reached, zoom level) + when it was
    /// raised; cleared after `TOAST_DURATION`.
    toast: Option<(String, Instant)>,
    /// Fixed origin for animation timing. A single shared clock is correct —
    /// every animated page derives its current frame from the same wall time, so
    /// all animations loop in step and the render loop stays stateless per page.
    anim_origin: Instant,
    /// Mini play/pause/step controls for the animation (GIF/WebP) currently in view.
    playback: Playback,
    /// Last window title pushed to the OS, so `render()` only calls `set_title`
    /// when the computed title actually changes (it's recomputed every frame).
    last_title: String,

    library: Library,
    library_view: bool,
    /// Monotonic per-frame counter used to stamp cover recency for LRU eviction
    /// (the sectioned library only decodes/keeps the covers it has shown).
    cover_clock: u64,
    /// Off-thread library scan (recursive dir walk): generation-tagged so a newer
    /// folder pick / rescan supersedes an in-flight scan; `scanning` drives the
    /// "Scanning library…" state.
    scanning: bool,
    scan_gen: u64,
    scan_tx: std::sync::mpsc::Sender<(u64, Library)>,
    scan_rx: std::sync::mpsc::Receiver<(u64, Library)>,
    /// Off-thread cover decode: workers send back decoded RGBA thumbnails keyed by
    /// path; the main thread uploads + registers them with egui.
    cover_tx: std::sync::mpsc::Sender<(PathBuf, yosh_engine::decode::DecodedImage)>,
    cover_rx: std::sync::mpsc::Receiver<(PathBuf, yosh_engine::decode::DecodedImage)>,
    /// Cover paths queued or in flight on a worker (dedup; also the "already tried"
    /// marker, cleared on LRU eviction / library replace so a re-scroll re-queues).
    queued_covers: std::collections::HashSet<PathBuf>,
    /// Volume location by path, rebuilt when the library is replaced, so a drained
    /// cover result maps back to its volume in O(1).
    cover_loc: std::collections::HashMap<PathBuf, (usize, usize)>,

    // Auto-update: a background thread checks the latest public release on
    // launch; the UI offers a one-click in-place update.
    update_rx: Option<std::sync::mpsc::Receiver<update::Update>>,
    update: Option<update::Update>,
    update_apply_rx: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
    updating: bool,
    update_error: Option<String>,

    // Async open: source construction (the I/O-bound part of opening a volume)
    // runs on a background thread so a slow network-share open never freezes the
    // UI. Each `open` bumps `open_gen`; `render` applies only the newest result,
    // so rapid `[`/`]` supersede in-flight opens. `opening_key` tracks the
    // most-recently-targeted volume so successive `[`/`]` advance from the pending
    // target rather than repeating the same neighbor.
    open_gen: u64,
    opening: bool,
    opening_key: Option<PathBuf>,
    open_tx: std::sync::mpsc::Sender<(u64, Built)>,
    open_rx: std::sync::mpsc::Receiver<(u64, Built)>,
    /// Instrumentation: when the in-flight open was initiated (debug logging only;
    /// cleared when the first page of the new volume is drawn or the open fails).
    open_t0: Option<Instant>,
    /// Startup resume scan: which recent still exists on disk is decided on a
    /// background thread (a stale network share or a spun-down HDD can make
    /// `Path::exists` take seconds, and this runs before the first frame). Some
    /// while the scan is in flight — the window is already up, showing the spinner
    /// — then None once the answer has been applied (or something else opened first).
    resume_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>>,
    /// Cached parent-dir scan for `[`/`]` — (parent dir, want_folder, natural-sort
    /// paths). Warmed in the background on open so the first jump doesn't pay it.
    sib_cache: Option<(PathBuf, bool, Vec<PathBuf>)>,
    sib_tx: std::sync::mpsc::Sender<(PathBuf, bool, Vec<PathBuf>)>,
    sib_rx: std::sync::mpsc::Receiver<(PathBuf, bool, Vec<PathBuf>)>,
    /// A `[`/`]` press that arrived while `sib_cache` was cold, held until the
    /// background scan lands (then replayed in `poll_background`). Presses
    /// accumulate, so mashing `]` three times before the scan finishes jumps three
    /// volumes rather than one.
    pending_sib_jump: Option<i64>,

    // Live folder refresh: an OS filesystem watch on the open folder, so images
    // added/removed on disk appear without reopening. `watcher` holds the notify
    // handle (dropping it unwatches; recreated per folder open in `set_source`);
    // its events arrive on `watch_rx` from notify's own thread. A change sets
    // `watch_dirty` (a debounce stamp); once it's been quiet briefly, an off-thread
    // `FolderSource` rebuild is spawned (gated by `rescanning`) and handed back on
    // `rescan_rx`, tagged with the `open_gen` it started under so a volume switch
    // discards a stale result — mirroring the `open_gen` guard on `open_rx`.
    watcher: Option<notify::RecommendedWatcher>,
    /// When watching a folder, `None` — every change in the watched dir is relevant.
    /// When watching a growing `.cbz`/`.zip`, we watch its *parent directory* (notify's
    /// Windows backend is directory-oriented; a lone-file watch is unreliable) and keep
    /// the archive's path here so sibling-file events in that dir are filtered out.
    watch_filter: Option<PathBuf>,
    watch_tx: std::sync::mpsc::Sender<notify::Result<notify::Event>>,
    watch_rx: std::sync::mpsc::Receiver<notify::Result<notify::Event>>,
    /// Finished off-thread watcher installs (`install_folder_watch` spawns them —
    /// `is_dir`/`canonicalize`/`watch()` are filesystem calls that can stall for
    /// seconds on a network share, and they used to run inside `set_source` on the
    /// main thread). Drained in `poll_background`, generation-tagged like `open_rx`.
    watch_install_tx: std::sync::mpsc::Sender<WatchInstall>,
    watch_install_rx: std::sync::mpsc::Receiver<WatchInstall>,
    watch_dirty: Option<Instant>,
    /// When the current burst of change events began (a folder copy or a streaming
    /// archive write fires many). `watch_dirty` (last event) drives the settle delay;
    /// this caps it so a *continuous* writer still refreshes at least every `MAX_WAIT`.
    watch_dirty_since: Option<Instant>,
    rescanning: bool,
    rescan_tx: std::sync::mpsc::Sender<(u64, Option<Arc<dyn PageSource>>)>,
    rescan_rx: std::sync::mpsc::Receiver<(u64, Option<Arc<dyn PageSource>>)>,
}

/// Result of constructing a page source: `(source, volume-key path, explicit
/// start index)`, or an error message. Built off-thread by `build_source`.
type Built = Result<(Arc<dyn PageSource>, PathBuf, Option<usize>), String>;

/// One finished off-thread watcher install: the open generation it was built
/// for, and the registered watcher + archive-path filter (None = this volume
/// isn't watchable, or notify failed — the feature is simply inactive).
type WatchInstall = (u64, Option<(notify::RecommendedWatcher, Option<PathBuf>)>);

/// One off-thread info-overlay result: the open generation, the visible-pages key
/// it was gathered for, and one `(page index, rows)` block per visible page. The
/// two tags let the main thread drop a result it has already moved past — a volume
/// switch bumps the generation, a page turn changes the key.
type InfoRows = (u64, (usize, Option<usize>), Vec<(usize, Vec<(String, String)>)>);

/// Bump `key` to the front of the most-recently-read list (newest first), drop any
/// older duplicate, and cap the length. Generic over the key string so the resume
/// path and the future recents shelf share one definition.
fn push_recent(settings: &mut config::Settings, key: &str) {
    settings.recents.retain(|p| p != key);
    settings.recents.insert(0, key.to_string());
    settings.recents.truncate(config::RECENTS_CAP);
}

/// Construct a page source for `path` (folder / archive / single image). Pure
/// I/O, touches no app state, so it runs on a background thread — a slow
/// (e.g. network-share) open never blocks the UI. The result is handed back to
/// the main thread and applied via `set_source`.
///
/// `resume` is where reading is *expected* to start (a saved position or a CLI
/// start index). Only the sequential archives use it, and only to order their
/// extraction so the resume page comes out first — see `RarSource::with_start` /
/// `SevenzSource::with_start`. It is advisory: `set_source` re-resolves the real
/// start index authoritatively at apply time, and a mismatch costs extraction
/// order, never correctness. The random-access formats ignore it entirely.
fn build_source(path: &Path, resume: Option<usize>) -> Built {
    if path.is_dir() {
        return FolderSource::new(path)
            .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
            .map_err(|e| e.to_string());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("cbz") | Some("zip") => ZipSource::new(path)
            .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
            .map_err(|e| e.to_string()),
        Some("cbr") | Some("rar") => RarSource::with_start(path, resume)
            .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
            .map_err(|e| e.to_string()),
        Some("7z") | Some("cb7") => SevenzSource::with_start(path, resume)
            .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
            .map_err(|e| e.to_string()),
        // A single image opens its containing folder, positioned at that image,
        // so you can seek forward/back within the folder.
        _ if is_image_ext(path) => {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            FolderSource::new(parent)
                .map(|s| {
                    let start = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .and_then(|n| s.index_of_name(n));
                    (Arc::new(s) as Arc<dyn PageSource>, parent.to_path_buf(), start)
                })
                .map_err(|e| e.to_string())
        }
        _ => Err("unsupported file type (open a folder, image, CBZ, CBR, or 7z)".to_string()),
    }
}

/// Can this archive grow while open, so live-refresh should watch it? A `.cbz`/`.zip`
/// being written has no central directory yet, so `ZipSource` opens it via local-header
/// recovery and a rebuild picks up newly-completed pages. `.cbr`/`.rar`/`.7z` can't be
/// partially read (they need end-of-file structures), so they aren't watched.
fn is_growable_archive(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("cbz" | "zip")
    )
}

/// True if the saved window rect overlaps any currently-connected monitor, so
/// we don't restore a window onto a display that's been unplugged (which would
/// strand it off-screen and unreachable). On overlap failure we drop only the
/// saved position; the size is still honored and the OS places the window.
fn geometry_on_screen(event_loop: &ActiveEventLoop, w: &config::WindowState) -> bool {
    let (wx0, wy0) = (w.x, w.y);
    let (wx1, wy1) = (w.x + w.w as i32, w.y + w.h as i32);
    event_loop.available_monitors().any(|m| {
        let mp = m.position();
        let ms = m.size();
        let (mx1, my1) = (mp.x + ms.width as i32, mp.y + ms.height as i32);
        wx0 < mx1 && wx1 > mp.x && wy0 < my1 && wy1 > mp.y
    })
}

fn fit_from_u8(v: u8) -> FitMode {
    match v {
        1 => FitMode::Width,
        2 => FitMode::Height,
        3 => FitMode::Actual,
        _ => FitMode::Window,
    }
}

fn fit_to_u8(f: FitMode) -> u8 {
    match f {
        FitMode::Window => 0,
        FitMode::Width => 1,
        FitMode::Height => 2,
        FitMode::Actual => 3,
    }
}

/// Spine-shadow strength the engine sees: the shell owns the enabled × strength
/// pair, the `Reader` takes one number (0.0 = off).
fn effective_spine(s: &config::Settings) -> f32 {
    if s.spine_shadow_enabled {
        s.spine_shadow_strength
    } else {
        0.0
    }
}

/// Bank `dy` wheel notches into `bank` and return the whole zoom-ladder steps it is
/// now worth, keeping the fraction for the next event.
///
/// A ladder step is inherently a whole thing, but a precision touchpad's Ctrl +
/// two-finger scroll arrives as a stream of small `PixelDelta`s worth a fraction of a
/// notch each; stepping per *event* would tear through the ladder. A reversal drops
/// the bank instead of letting a half-notch of the old direction eat into the new one.
fn bank_notches(bank: &mut f32, dy: f32) -> i32 {
    if *bank != 0.0 && bank.signum() != dy.signum() {
        *bank = 0.0;
    }
    *bank += dy;
    let steps = bank.trunc();
    *bank -= steps;
    steps as i32
}

/// System CJK fallback fonts, in preference order — the first that reads wins.
/// At module scope so the (tens-of-MB) read can be started on a background thread
/// before `install_fonts` is reached; see the `cjk_font` spawn in `resumed`.
const CJK_FONT_CANDIDATES: &[&str] = &[
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    "/Library/Fonts/Hiragino Sans GB.ttc",
];

/// Install the egui chrome fonts: the Phosphor icon set (for the library section
/// carets) is added unconditionally, and a best-effort system CJK font — read
/// ahead of time and passed in as `cjk` — is appended as a fallback so
/// Japanese/Chinese/Korean text (paths, filenames, library titles) renders. Both
/// are *fallbacks* layered after the default fonts, so Latin keeps its default
/// look. A single `set_fonts` applies the lot.
fn install_fonts(ctx: &egui::Context, cjk: Option<Vec<u8>>) {
    let mut fonts = egui::FontDefinitions::default();
    // Phosphor carets etc. — pure-Rust glyph set, always available (matches the
    // Android shell, so the library headers never fall back to tofu boxes).
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Fill);

    if let Some(bytes) = cjk {
        fonts.font_data.insert(
            "cjk".to_owned(),
            // The CJK fallback's glyphs render above the Latin baseline; nudge them
            // down so mixed JP/EN lines (top bar, info overlay, titles) line up.
            std::sync::Arc::new(egui::FontData::from_owned(bytes).tweak(egui::FontTweak {
                y_offset_factor: 0.18,
                ..Default::default()
            })),
        );
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push("cjk".to_owned());
        }
    }
    ctx.set_fonts(fonts);
}

/// The engine's "something new to draw" callback for this window: it only calls
/// `Window::request_redraw`, which winit documents as thread-safe, so a decode
/// worker can schedule the next frame the instant a page lands. The closure owns
/// an `Arc<Window>` clone, so even a straggler worker's call lands on a live window.
///
/// Every background producer in this shell clones one and calls it *after* its
/// `tx.send` — send-then-wake, the same order the pool uses (`pool.rs`): the
/// woken frame drains the channel, so a result that lands while the app is idle
/// is applied immediately. This is *the* delivery mechanism for background work
/// — the loop polls nothing on its own (see `about_to_wait`), so a producer that
/// forgets to wake would strand its result until the next unrelated event.
fn frame_waker(window: &Arc<Window>) -> Waker {
    let w = window.clone();
    Arc::new(move || w.request_redraw())
}

/// The window / taskbar / title-bar icon, decoded from the embedded logo PNG.
/// (The icon embedded in the .exe via the build script is used by Explorer and
/// shortcuts, but winit needs the window icon set explicitly at runtime.)
fn window_icon() -> Option<winit::window::Icon> {
    let img = image::load_from_memory(include_bytes!("../assets/yosh.png"))
        .ok()?
        .to_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}

/// Bind the exe's own embedded icon to the window on Windows — the real taskbar fix.
///
/// `with_window_icon` / `with_taskbar_icon` build the icon from `yosh.png`, which
/// is 256x255 (non-square) and shares one HICON for both the small (title-bar) and
/// big (taskbar) slots. The small slot tolerates it; the taskbar's large slot does
/// not and falls back to the generic app icon. The `.ico` the build script embeds
/// in the exe is square and multi-resolution — the exact icon Explorer, the title
/// bar, and the installer already render correctly — so we pull it straight out of
/// the running exe with `ExtractIconExW` (large + small at the system icon sizes)
/// and set it as ICON_BIG / ICON_SMALL, plus the window-class icons as a backstop.
#[cfg(windows)]
fn bind_exe_icon(window: &Window) {
    use std::os::windows::ffi::OsStrExt;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::Shell::ExtractIconExW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GCLP_HICON, GCLP_HICONSM, ICON_BIG, ICON_SMALL, SendMessageW, SetClassLongPtrW, WM_SETICON,
    };

    let hwnd = match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Win32(h)) => h.hwnd.get() as *mut core::ffi::c_void,
        _ => return,
    };
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let wide: Vec<u16> = exe.as_os_str().encode_wide().chain(std::iter::once(0)).collect();

    let mut large: *mut core::ffi::c_void = std::ptr::null_mut();
    let mut small: *mut core::ffi::c_void = std::ptr::null_mut();
    // Icon group 0 = the application icon embedded by the build script.
    if unsafe { ExtractIconExW(wide.as_ptr(), 0, &mut large, &mut small, 1) } == 0 {
        return;
    }
    // The extracted HICONs are handed to the window (WM_SETICON / class icon) and
    // must outlive bind_exe_icon — the window references them, it doesn't copy. So
    // they are intentionally not destroyed here; this single, process-lifetime
    // window owns them and the OS reclaims them at exit.
    unsafe {
        if !large.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_BIG as usize, large as isize);
            SetClassLongPtrW(hwnd, GCLP_HICON, large as isize);
        }
        if !small.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL as usize, small as isize);
            SetClassLongPtrW(hwnd, GCLP_HICONSM, small as isize);
        }
    }
}

/// Open the OS file browser at `path`'s containing folder and select `path`.
/// On Windows this uses `SHOpenFolderAndSelectItems`, which reuses an existing
/// Explorer window showing that folder instead of spawning a new one each time
/// (unlike `explorer.exe /select,`). Runs on a short-lived thread because it
/// initializes COM and may launch/raise Explorer — neither belongs on the UI thread.
#[cfg(windows)]
fn reveal_in_explorer(path: &std::path::Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, CoInitializeEx, CoUninitialize,
    };
    use windows_sys::Win32::UI::Shell::{ILCreateFromPathW, ILFree, SHOpenFolderAndSelectItems};

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    std::thread::spawn(move || unsafe {
        let _ = CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32);
        let pidl = ILCreateFromPathW(wide.as_ptr());
        if !pidl.is_null() {
            // cidl = 0 + apidl = null → open the item's parent and select the item.
            let _ = SHOpenFolderAndSelectItems(pidl, 0, std::ptr::null(), 0);
            ILFree(pidl);
        }
        CoUninitialize();
    });
}

/// Non-Windows fallback: best-effort open the containing folder (no selection),
/// so the Linux build stays functional and green.
#[cfg(not(windows))]
fn reveal_in_explorer(path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

impl App {
    pub fn new(initial_path: Option<PathBuf>, start_index: usize) -> Self {
        // A CLI / file-association open is known before the event loop even starts,
        // so start reading the disk *here* — the listing then overlaps window
        // creation, `Gpu::new` (the dominant startup term) and the CJK font read
        // instead of queuing behind them. `State` adopts this channel wholesale in
        // `resumed`, so the result arrives on the same path as any later open; the
        // pre-spawn is hardcoded generation 1 and `State` starts at `open_gen: 1`.
        let (open_tx, open_rx) = std::sync::mpsc::channel();
        // The waker needs a `Window`, which doesn't exist yet: `resumed` fills this
        // slot the moment it does. Both race orders deliver — see the spawn below.
        let early_waker: Arc<std::sync::OnceLock<Waker>> = Arc::new(std::sync::OnceLock::new());
        let mut open_t0 = None;
        if let Some(path) = initial_path.clone() {
            open_t0 = Some(Instant::now());
            if cfg!(debug_assertions) {
                eprintln!("[open] initiated (cli): {}", path.display());
            }
            let tx = open_tx.clone();
            let waker = early_waker.clone();
            std::thread::spawn(move || {
                // Resume hint, resolved here rather than passed in: `config::load()` is a
                // small local JSON read that overlaps the archive open on this same
                // thread, so it costs nothing at startup, and `resumed` loads the
                // settings again for real (idempotent). Mirrors `set_source`'s
                // precedence — an explicit CLI page beats the saved position.
                let hint = if start_index > 0 {
                    Some(start_index)
                } else {
                    config::load()
                        .last_pages
                        .get(path.to_string_lossy().as_ref())
                        .copied()
                        .filter(|&i| i > 0)
                };
                let _ = tx.send((1, build_source(&path, hint)));
                // Send-then-wake, as every producer does — except there may be no
                // window to wake yet. If the slot is still empty the result simply
                // waits in the channel and is drained by the first `render()`,
                // which `resumed`'s unconditional `request_redraw` guarantees.
                if let Some(wake) = waker.get() {
                    wake();
                }
            });
        }
        Self {
            start_index,
            state: None,
            open_tx,
            open_rx: Some(open_rx),
            cli_open: initial_path,
            early_waker,
            open_t0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            // Resuming from background: the OS tore the surface down (Android does
            // this on every background), but the device, decode pool, cache and
            // reader state all survive. Make a fresh window + surface and carry on
            // with no re-decode. Desktop fires `resumed` only once, so this branch
            // is Android's; re-binding egui to the new window is the future Android
            // shell's concern.
            let attrs = Window::default_attributes()
                .with_title("yosh")
                .with_window_icon(window_icon());
            let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
            state.gpu.recreate_surface(window.clone());
            state.window = window;
            // The pool survives the surface rebuild, but its waker points at the
            // window that just died — re-aim it at the new one.
            let waker = frame_waker(&state.window);
            state.reader.set_waker(Some(waker));
            state.window.request_redraw();
            return;
        }
        let mut settings = config::load();
        // The CJK fallback font is a tens-of-MB .ttc; read it while the adapter/device
        // request runs instead of after it. Joined (not polled) at install time.
        let cjk_font = std::thread::spawn(|| {
            CJK_FONT_CANDIDATES
                .iter()
                .find_map(|p| std::fs::read(p).ok())
        });
        let attrs = Window::default_attributes()
            .with_title("yosh")
            .with_window_icon(window_icon());
        // Restore the last window geometry (size/position/maximized) if we saved
        // one; otherwise fall back to the default launch size. The saved position
        // is only reapplied when it still lands on a connected monitor, so a
        // window saved on a now-disconnected display doesn't open off-screen.
        let attrs = match settings.window {
            Some(w) => {
                let mut a = attrs.with_inner_size(PhysicalSize::new(w.w.max(1), w.h.max(1)));
                if geometry_on_screen(event_loop, &w) {
                    a = a.with_position(PhysicalPosition::new(w.x, w.y));
                }
                a.with_maximized(w.maximized)
            }
            None => attrs.with_inner_size(winit::dpi::LogicalSize::new(1100.0, 1500.0)),
        };
        // `with_window_icon` sets only the small (title-bar) icon; the taskbar
        // reads ICON_BIG, which winit exposes separately on Windows.
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_taskbar_icon(window_icon())
        };
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        // A CLI open spawned back in `App::new` may still be running: hand it a real
        // waker the instant a window exists, so a listing that finishes during the
        // GPU/font work below still schedules its own frame. Filled here rather than
        // later because everything after this point can block for a while.
        let _ = self.early_waker.set(frame_waker(&window));
        // Override winit's PNG-derived icon with the exe's square embedded .ico so
        // the taskbar (ICON_BIG) shows the logo instead of the generic app icon.
        #[cfg(windows)]
        bind_exe_icon(&window);
        let gpu = Gpu::new(window.clone());

        let egui_ctx = egui::Context::default();
        // A panicked reader thread degrades to "no CJK fallback" — already the
        // behavior on a system with none of the candidate fonts installed.
        install_fonts(&egui_ctx, cjk_font.join().unwrap_or(None));
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            Some(8192),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let page_pipeline = PagePipeline::new(&gpu.device, gpu.config.format);
        // Size the reader's resource budget to the device: CPU count + a slice of
        // system RAM. Desktop reproduces the historical fixed budget; a constrained
        // device (e.g. Android) scales cache / textures / prefetch down.
        // …then to the performance profile: `Auto` (the default) resolves to the
        // uncapped `High` tier on mains and throttles to `Mid` on battery, and a
        // pinned choice overrides both. `for_tier(High, ..)` *is* `derive(..)` —
        // pinned by the engine's `for_tier_high_is_exactly_derive` — so a plugged-in
        // machine on `Auto` gets exactly the budget it always got.
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let mem_budget_mb = detect_mem_budget_mb();
        let on_battery = on_battery().unwrap_or(false);
        let tier = effective_tier(settings.perf, on_battery);
        let budget = Budget::for_tier(tier, mem_budget_mb, cpus);
        let tex_pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));

        let mut ui = UiState {
            status: format!("{} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend),
            ..UiState::default()
        };
        // Resume the last book on a no-arg launch: open the most recent volume that
        // still exists on disk (a stale head is skipped so the list stays useful).
        // `set_source` resumes its saved page automatically. The `exists()` probes run
        // off-thread — they hit the filesystem, and a dead network share or sleeping
        // HDD would otherwise delay the *window itself* by seconds. The answer is
        // picked up in `poll_background`.
        // A CLI open is already in flight (spawned in `App::new` as generation 1),
        // so it needs no `pending_open` — only the resume scan is decided here.
        let cli_open = self.cli_open.take();
        let mut resume_rx = None;
        if cli_open.is_none() && settings.resume_on_startup && !settings.recents.is_empty() {
            let (tx, rx) = std::sync::mpsc::channel();
            let recents = settings.recents.clone();
            let wake = frame_waker(&window);
            std::thread::spawn(move || {
                let found = recents
                    .into_iter()
                    .find(|p| std::path::Path::new(p).exists())
                    .map(PathBuf::from);
                let _ = tx.send(found);
                wake(); // send-then-wake
            });
            resume_rx = Some(rx);
        }

        // The recursive library scan can be slow on a big tree, so it runs
        // off-thread (kicked off below, after the channels exist); the grid shows
        // "Scanning library…" until it lands. The library is the home screen: with
        // nothing to open (no CLI arg, no resume) we land in the library view — which
        // shows the configured grid, or the onboarding when no library is set yet.
        let library = Library::empty();
        let has_root = settings.library_root.is_some();
        // A pending resume scan counts as "something is opening": start on the reader
        // background (with the spinner, via `opening` below) rather than flashing the
        // library grid for the frames the scan takes.
        let library_view = cli_open.is_none() && ui.pending_open.is_none() && resume_rx.is_none();
        // Show the keys overlay once, on the first launch ever, then persist so it
        // never auto-opens again (F1 / "? Help" reopen it on demand).
        if !settings.help_seen {
            ui.help_open = true;
            settings.help_seen = true;
            config::save(&settings);
        }

        window.request_redraw();
        // Kick off a background update check against the public GitHub releases.
        let (update_tx, update_rx) = std::sync::mpsc::channel();
        let wake = frame_waker(&window);
        std::thread::spawn(move || {
            if let Some(u) = update::check() {
                let _ = update_tx.send(u);
                wake(); // send-then-wake (nothing to show when already current)
            }
        });
        // Adopt the open channel built in `App::new` (a pre-spawned CLI open may
        // already have sent on it, or still be running). Taking the receiver is safe:
        // the `state.as_mut()` early-return above handles a re-entered `resumed`, and
        // the desktop shell builds `State` exactly once.
        let open_tx = self.open_tx.clone();
        let open_rx = self.open_rx.take().expect("desktop resumed builds State once");
        // Channels for sibling-volume prescans and the off-thread Tab info overlay.
        let (sib_tx, sib_rx) = std::sync::mpsc::channel();
        let (info_tx, info_rx) = std::sync::mpsc::channel();
        // Channels for the live-folder-refresh watcher: raw notify events in, and
        // off-thread `FolderSource` rebuilds out.
        let (watch_tx, watch_rx) = std::sync::mpsc::channel();
        let (watch_install_tx, watch_install_rx) = std::sync::mpsc::channel();
        let (rescan_tx, rescan_rx) = std::sync::mpsc::channel();
        // Off-thread library scan + cover decode.
        let (scan_tx, scan_rx) = std::sync::mpsc::channel();
        let (cover_tx, cover_rx) = std::sync::mpsc::channel();
        let mut scan_gen = 0u64;
        let scanning = has_root;
        if let Some(root) = settings.library_root.clone() {
            scan_gen = 1;
            let tx = scan_tx.clone();
            let wake = frame_waker(&window);
            std::thread::spawn(move || {
                let _ = tx.send((1, Library::scan(std::path::Path::new(&root))));
                wake(); // send-then-wake
            });
        }
        let mut reader = Reader::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            tex_pool,
            budget,
            fit_from_u8(settings.fit),
            if settings.layout_spread {
                Layout::Spread
            } else {
                Layout::Single
            },
            settings.scroll,
            if settings.direction_rtl {
                Direction::Rtl
            } else {
                Direction::Ltr
            },
            self.start_index,
            false, // two_tier: desktop keeps the always-HQ pipeline
        );
        reader.transition_enabled = settings.page_transition_enabled;
        reader.fit_no_upscale = settings.no_stretch;
        reader.spine_strength = effective_spine(&settings);
        // Decode→UI wakeup: a worker that finishes a page schedules the frame that
        // draws it (winit's `request_redraw` is thread-safe), so the loop doesn't
        // have to keep drawing on the chance that one landed. Set once here and
        // remembered by the `Reader`, which re-applies it to every pool it builds;
        // shell-side rebuilds (`set_source`) re-apply it themselves.
        reader.set_waker(Some(frame_waker(&window)));
        // Seed the OS day/night flag from the window (None on platforms that don't
        // report it → assume dark, egui's default look).
        let system_dark = window
            .theme()
            .map(|t| t == winit::window::Theme::Dark)
            .unwrap_or(true);
        self.state = Some(State {
            window,
            gpu,
            reader,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            page_pipeline,
            cursor_x: 0.0,
            cursor_y: 0.0,
            ctrl_held: false,
            zoom_notches: 0.0,
            mouse_down: false,
            drag_dist: 0.0,
            cursor_in_window: false,
            last_mid_click: None,
            gestures: TouchGestures::new(),
            win_geom: settings.window.map(|w| (w.x, w.y, w.w, w.h)),
            win_maximized: settings.window.is_some_and(|w| w.maximized),
            parked: false,
            mem_budget_mb,
            cpus,
            on_battery,
            power_checked: Instant::now(),
            applied_tier: tier,
            settings,
            system_dark,
            volume_key: None,
            info_for: None,
            info_tx,
            info_rx,
            loading_pending: None,
            toast: None,
            anim_origin: Instant::now(),
            playback: Playback::default(),
            last_title: String::new(),
            library,
            library_view,
            cover_clock: 0,
            scanning,
            scan_gen,
            scan_tx,
            scan_rx,
            cover_tx,
            cover_rx,
            queued_covers: std::collections::HashSet::new(),
            cover_loc: std::collections::HashMap::new(),
            update_rx: Some(update_rx),
            update: None,
            update_apply_rx: None,
            updating: false,
            update_error: None,
            // The pre-spawned CLI open is generation 1, so `State` must start there
            // or its result would be discarded as stale by the drain in `render`.
            open_gen: if cli_open.is_some() { 1 } else { 0 },
            // Spinner while the resume scan — or the pre-spawned CLI open — runs.
            opening: resume_rx.is_some() || cli_open.is_some(),
            // Keeps rapid `[`/`]` advancing from the pending CLI target rather than
            // from nothing, exactly as a `State::open`-initiated open would.
            opening_key: cli_open,
            open_tx,
            open_rx,
            open_t0: self.open_t0.take(),
            resume_rx,
            sib_cache: None,
            sib_tx,
            sib_rx,
            pending_sib_jump: None,
            watcher: None,
            watch_filter: None,
            watch_tx,
            watch_rx,
            watch_install_tx,
            watch_install_rx,
            watch_dirty: None,
            watch_dirty_since: None,
            rescanning: false,
            rescan_tx,
            rescan_rx,
        });
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let response = state.egui_state.on_window_event(&state.window, &event);
        // With `ControlFlow::Wait` the loop no longer free-runs: any input/window
        // event may have changed reading state, so each one buys a frame (render
        // then re-requests itself while anything is still in flight). Requesting
        // from *within* RedrawRequested would loop forever, so that one is exempt —
        // render's own end-of-frame conditions decide there. (For the same reason
        // `response.repaint` must NOT be honored here: egui-winit reports
        // `repaint: true` for RedrawRequested itself — it means "paint now", not
        // "schedule another frame" — which would re-arm every frame, silently
        // restoring the continuous loop. egui's real "keep animating" signal is
        // the `repaint_delay == 0` check on `full_output` at the end of render.)
        let buys_frame = !matches!(event, WindowEvent::RedrawRequested);

        match event {
            // Quit by process exit rather than `event_loop.exit()`: the orderly path
            // runs every destructor on the way out (wgpu/driver teardown, the notify
            // watcher, threads still finishing a slow read), which can hang the window
            // on screen for seconds. `persist()` has already flushed everything durable
            // — the config JSON — and the thumb cache writes temp-file + rename, so a
            // killed write can at worst strand a stray temp file. Same pattern as
            // `relaunch()`.
            WindowEvent::CloseRequested => {
                state.persist();
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                // Keep the reading viewport in lock-step with the surface so an
                // input event between renders sees the new size (as it did when
                // these reads came straight from `gpu.config`).
                state.reader.viewport = state.content_viewport();
                // A minimize arrives on Windows as a resize to 0×0 — and often as
                // *only* that, since `Occluded` isn't guaranteed to be delivered
                // there. Treat it as the park signal, and any real size as the
                // unpark; both are idempotent, so the pair of signals can overlap
                // freely with `Occluded` on platforms that send both.
                if size.width == 0 || size.height == 0 {
                    state.park();
                } else {
                    state.unpark();
                }
            }
            // Geometry is sampled in `render`, not here — see
            // `record_window_geometry` for why the event-time state is unreliable.
            WindowEvent::Moved(_) => {}
            // OS switched day/night: refresh the cached flag so a `System` theme
            // follows it (buys_frame below repaints the chrome).
            WindowEvent::ThemeChanged(theme) => {
                state.system_dark = theme == winit::window::Theme::Dark;
            }
            WindowEvent::DroppedFile(path) => state.ui.pending_open = Some(path),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(action) = action_from(&event) && !response.consumed {
                    if matches!(action, Action::Quit) {
                        // Immediate exit, skipping destructor teardown — see
                        // `CloseRequested` above.
                        state.persist();
                        std::process::exit(0);
                    } else {
                        state.apply_action(action);
                    }
                }
            }
            // Touchscreen gestures (swipe-to-flip, pinch-zoom, finger scroll + fling,
            // tap) — the reader's own layer, since egui only ever sees the primary
            // finger. `consumed` is what keeps a drag begun on the chrome off the page.
            WindowEvent::Touch(Touch {
                phase,
                location,
                id,
                ..
            }) => state.handle_touch(phase, id, location.x, location.y, response.consumed),
            // The mouse arms below stand down while any finger is on the glass:
            // Windows can also deliver OS-synthesized mouse events for a touch, and
            // acting on both would pan twice (and fake a click on release).
            WindowEvent::CursorMoved { position, .. } if !state.gestures.touch_active() => {
                state.on_cursor_moved(position.x, position.y)
            }
            // Non-scrollable chrome (seekbar, anim panel) isn't allowed to swallow
            // the wheel — keep routing it to the reader (clicks/drags still work).
            WindowEvent::MouseWheel { delta, .. }
                if !response.consumed || state.wheel_passthrough() =>
            {
                state.on_wheel(delta)
            }
            WindowEvent::MouseInput {
                state: btn,
                button: MouseButton::Left,
                ..
            } if !response.consumed && !state.gestures.touch_active() => {
                state.on_left_button(btn == ElementState::Pressed)
            }
            WindowEvent::CursorEntered { .. } => state.cursor_in_window = true,
            WindowEvent::CursorLeft { .. } => state.cursor_in_window = false,
            // CursorLeft isn't guaranteed on focus loss / occlusion (alt-tab, a
            // fast monitor switch), which would otherwise strand a hover arrow
            // on screen. A later CursorMoved / CursorEntered re-arms the flag.
            //
            // Focus loss deliberately does *not* park: reading with another window
            // focused (notes, a browser) is normal use, and freezing the decode
            // pool on every alt-tab would stall the read-ahead the app exists for.
            // Only "can't be seen at all" — occluded or minimized — parks.
            // Ctrl arms the wheel's focal zoom. Not routed through `action_from`: this
            // is modifier *state*, not a keypress, and the wheel needs it on the frame
            // the notch arrives.
            WindowEvent::ModifiersChanged(m) => state.ctrl_held = m.state().control_key(),
            WindowEvent::Focused(false) => {
                state.cursor_in_window = false;
                // A Ctrl-held alt-tab releases the key while we're unfocused, so the
                // flag would stay stuck on until the next ModifiersChanged.
                state.ctrl_held = false;
            }
            WindowEvent::Occluded(true) => {
                state.cursor_in_window = false;
                state.park();
            }
            // Visible again: thaw and let the next frame re-queue the work the park
            // abandoned. `Focused(true)` is a backstop for a restore that arrives
            // without an `Occluded(false)` or a resize.
            WindowEvent::Occluded(false) | WindowEvent::Focused(true) => state.unpark(),
            _ => {}
        }

        if buys_frame {
            state.window.request_redraw();
        }
    }

    /// The shell's **only** clock: a pure deadline scheduler, run once after every
    /// batch of events. Nothing here polls — every background producer wakes the
    /// loop itself right after its `tx.send` (see `frame_waker`), so results land
    /// on a frame the instant they arrive rather than on a heartbeat; everything
    /// else is either event-driven or animation-guarded (`render`'s redraw guard
    /// re-requests frames while something is moving).
    ///
    /// What is left is the handful of *timed* transitions — state set at T that
    /// must become visible at T + D with no further input — enumerated by
    /// `State::deadlines`: the folder-watch debounce, the loading-spinner grace
    /// period, and toast expiry.
    ///
    /// A deadline that has come due buys exactly one frame (`request_redraw`) and
    /// is **not** re-armed: the frame's `poll_background` / `render` is what
    /// consumes the ripe state, and the redraw wakes the loop by itself. Only
    /// still-future deadlines go into `WaitUntil`, and the control flow is re-set
    /// on *every* pass — explicitly back to `Wait` when nothing is armed — because
    /// a fired `WaitUntil` left in place would spin the loop at full speed.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let mut next: Option<Instant> = None;
        // `about_to_wait` also runs before `resumed` has built the window/state.
        if let Some(state) = self.state.as_mut() {
            let now = Instant::now();
            let mut due = false;
            for deadline in state.deadlines().into_iter().flatten() {
                if deadline <= now {
                    due = true;
                } else {
                    next = Some(next.map_or(deadline, |n: Instant| n.min(deadline)));
                }
            }
            if due {
                state.window.request_redraw();
            }
        }
        event_loop.set_control_flow(match next {
            Some(deadline) => ControlFlow::WaitUntil(deadline),
            None => ControlFlow::Wait,
        });
    }
}

enum Action {
    Forward,
    Backward,
    Left,
    Right,
    First,
    Last,
    CycleFit,
    ToggleDir,
    ToggleLayout,
    ToggleScroll,
    ZoomIn,
    ZoomOut,
    // View presets (number keys): each sets a complete page-flip view at once.
    PresetWindow,
    PresetWidth,
    PresetActual,
    PresetSpreadLtr,
    PresetSpreadRtl,
    ToggleHelp,
    ToggleFullscreen,
    ToggleSpreadOffset,
    ToggleInfo,
    ToggleSeekbar,
    TogglePageTransition,
    ToggleStretch,
    ToggleAnimBar,
    PrevVolume,
    NextVolume,
    Rotate,
    ShowInExplorer,
    Quit,
}

/// Map a key event to an action, preferring the physical key but falling back to
/// the logical key (covers injected events without scancodes). Vertical keys are
/// absolute (forward/backward); horizontal keys are resolved by reading direction.
fn action_from(ev: &KeyEvent) -> Option<Action> {
    if let PhysicalKey::Code(c) = ev.physical_key {
        match c {
            KeyCode::ArrowDown | KeyCode::Space | KeyCode::PageDown => return Some(Action::Forward),
            KeyCode::ArrowUp | KeyCode::PageUp => return Some(Action::Backward),
            KeyCode::ArrowRight => return Some(Action::Right),
            KeyCode::ArrowLeft => return Some(Action::Left),
            KeyCode::Home => return Some(Action::First),
            KeyCode::End => return Some(Action::Last),
            KeyCode::KeyD => return Some(Action::ToggleDir),
            KeyCode::KeyS => return Some(Action::ToggleLayout),
            KeyCode::KeyO => return Some(Action::ToggleSpreadOffset),
            KeyCode::KeyC => return Some(Action::ToggleScroll),
            KeyCode::KeyB => return Some(Action::ToggleSeekbar),
            KeyCode::KeyT => return Some(Action::TogglePageTransition),
            KeyCode::KeyZ => return Some(Action::ToggleStretch),
            KeyCode::KeyG => return Some(Action::ToggleAnimBar),
            KeyCode::KeyE => return Some(Action::ShowInExplorer),
            KeyCode::KeyR => return Some(Action::Rotate),
            KeyCode::Equal | KeyCode::NumpadAdd => return Some(Action::ZoomIn),
            KeyCode::Minus | KeyCode::NumpadSubtract => return Some(Action::ZoomOut),
            KeyCode::Digit9 | KeyCode::Numpad9 => return Some(Action::PresetWindow),
            KeyCode::Digit8 | KeyCode::Numpad8 => return Some(Action::PresetWidth),
            KeyCode::Digit7 | KeyCode::Numpad7 => return Some(Action::PresetSpreadLtr),
            KeyCode::Digit6 | KeyCode::Numpad6 => return Some(Action::PresetSpreadRtl),
            KeyCode::Digit0 | KeyCode::Numpad0 => return Some(Action::PresetActual),
            KeyCode::F1 => return Some(Action::ToggleHelp),
            KeyCode::KeyI => return Some(Action::ToggleInfo),
            KeyCode::F11 => return Some(Action::ToggleFullscreen),
            KeyCode::Escape => return Some(Action::Quit),
            KeyCode::BracketLeft => return Some(Action::PrevVolume),
            KeyCode::BracketRight => return Some(Action::NextVolume),
            _ => {}
        }
    }
    if let Key::Named(n) = &ev.logical_key {
        match n {
            NamedKey::ArrowDown | NamedKey::PageDown => return Some(Action::Forward),
            NamedKey::ArrowUp | NamedKey::PageUp => return Some(Action::Backward),
            NamedKey::ArrowRight => return Some(Action::Right),
            NamedKey::ArrowLeft => return Some(Action::Left),
            NamedKey::Home => return Some(Action::First),
            NamedKey::End => return Some(Action::Last),
            NamedKey::F1 => return Some(Action::ToggleHelp),
            NamedKey::Escape => return Some(Action::Quit),
            _ => {}
        }
    }
    None
}

/// Gather display info for page `index` (Tab overlay): reads the page bytes once
/// and probes the header for resolution + format. A free function over the bare
/// source, because it runs on a **background** thread — `read_page` and `modified`
/// are disk I/O, and on a RAR still decompressing `read_page` blocks until that
/// entry lands, which would freeze the UI for as long as the extraction takes.
/// The cache-derived "LQ tier" row needs no I/O and is appended by the main
/// thread when this result is applied.
fn page_info(src: &dyn PageSource, index: usize) -> Vec<(String, String)> {
    let name = src.name(index).to_string();
    let bytes = src.read_page(index).ok();
    let (res, fmt, size, color) = match &bytes {
        Some(b) => {
            let (w, h, detail) = yosh_engine::meta::probe(b);
            let res = if w == 0 || h == 0 {
                "—".to_string()
            } else {
                format!("{w} × {h}")
            };
            let color = yosh_engine::icc::extract_icc(b)
                .as_deref()
                .and_then(yosh_engine::icc::describe)
                .unwrap_or_else(|| "—".to_string());
            (res, detail, yosh_engine::meta::human_size(b.len() as u64), color)
        }
        None => (
            "—".to_string(),
            "—".to_string(),
            "—".to_string(),
            "—".to_string(),
        ),
    };
    let modified = src.modified(index).unwrap_or_else(|| "—".to_string());
    vec![
        ("File".to_string(), name),
        ("Page".to_string(), format!("{} / {}", index + 1, src.len())),
        ("Size".to_string(), size),
        ("Modified".to_string(), modified),
        ("Resolution".to_string(), res),
        ("Format".to_string(), fmt),
        ("Color".to_string(), color),
    ]
}

impl State {
    fn apply_action(&mut self, action: Action) {
        match action {
            Action::Forward => {
                if self.reader.scroll_mode {
                    let vh = self.reader.viewport.h as f32;
                    self.reader.scroll_by(vh * 0.9);
                } else {
                    self.reader.step(1);
                }
            }
            Action::Backward => {
                if self.reader.scroll_mode {
                    let vh = self.reader.viewport.h as f32;
                    self.reader.scroll_by(-vh * 0.9);
                } else {
                    self.reader.step(-1);
                }
            }
            // In RTL, "left" advances the story; in LTR, "right" does. (Page-flip only.)
            Action::Right if !self.reader.scroll_mode => {
                self.reader.step(if self.reader.direction == Direction::Ltr { 1 } else { -1 });
            }
            Action::Left if !self.reader.scroll_mode => {
                self.reader.step(if self.reader.direction == Direction::Ltr { -1 } else { 1 });
            }
            Action::Right | Action::Left => {}
            Action::First => self.reader.goto(0),
            Action::Last => {
                if let Some(s) = &self.reader.source {
                    self.reader.goto(s.len().saturating_sub(1));
                }
            }
            Action::CycleFit if self.reader.scroll_mode => {
                // In scroll: toggle width-fit (zoom 1) vs height-fit (a typical
                // page ~fills the viewport height).
                self.reader.pan_x = 0.0;
                if (self.reader.zoom - 1.0).abs() < 0.01 {
                    let sw = self.reader.viewport.w.max(1) as f32;
                    let sh = self.reader.viewport.h.max(1) as f32;
                    let cw = sh / self.reader.est_aspect.max(0.1);
                    self.reader.zoom = (cw / sw).clamp(0.2, 8.0);
                } else {
                    self.reader.zoom = 1.0;
                }
            }
            Action::CycleFit => {
                self.reader.fit = self.reader.fit.cycle();
                self.reader.zoom = 1.0;
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
                self.settings.fit = fit_to_u8(self.reader.fit);
                config::save(&self.settings);
            }
            Action::ZoomIn => self.reader.zoom_to_preset(true),
            Action::ZoomOut => self.reader.zoom_to_preset(false),
            Action::PresetWindow => self.apply_view(FitMode::Window, false, None),
            Action::PresetWidth => self.apply_view(FitMode::Width, false, None),
            Action::PresetActual => self.apply_view(FitMode::Actual, false, None),
            Action::PresetSpreadLtr => {
                self.apply_view(FitMode::Window, true, Some(Direction::Ltr))
            }
            Action::PresetSpreadRtl => {
                self.apply_view(FitMode::Window, true, Some(Direction::Rtl))
            }
            Action::ToggleDir => {
                self.reader.direction = match self.reader.direction {
                    Direction::Ltr => Direction::Rtl,
                    Direction::Rtl => Direction::Ltr,
                };
                self.settings.direction_rtl = self.reader.direction == Direction::Rtl;
                config::save(&self.settings);
                self.toast(format!("Direction: {}", self.reader.direction.label()));
            }
            Action::ToggleLayout => {
                self.reader.layout = self.reader.layout.toggled();
                // Snap to the current view's anchor so pairing is consistent.
                self.reader.snap_to_view_start();
                self.reader.pan_y = 0.0;
                self.settings.layout_spread = self.reader.layout == Layout::Spread;
                config::save(&self.settings);
                self.reader.prefetch();
                self.toast(format!("Layout: {}", self.reader.layout.label()));
            }
            Action::ToggleScroll => {
                self.reader.scroll_mode = !self.reader.scroll_mode;
                self.reader.top_offset = 0.0;
                self.settings.scroll = self.reader.scroll_mode;
                config::save(&self.settings);
                self.reader.prefetch();
                self.toast(if self.reader.scroll_mode {
                    "Scroll mode"
                } else {
                    "Page-flip mode"
                });
            }
            Action::ToggleHelp => self.ui.help_open = !self.ui.help_open,
            Action::ToggleInfo => {
                self.ui.info_open = !self.ui.info_open;
                self.info_for = None; // rebuild the overlay text next render
            }
            Action::ToggleSeekbar => {
                self.settings.seekbar_enabled = !self.settings.seekbar_enabled;
                config::save(&self.settings);
                // Announce the change like the other toggles — a silent flip left
                // the seekbar disabled with no clue it had been turned off.
                self.toast(if self.settings.seekbar_enabled {
                    "Seekbar: on"
                } else {
                    "Seekbar: off"
                });
            }
            Action::TogglePageTransition => {
                self.settings.page_transition_enabled = !self.settings.page_transition_enabled;
                self.reader.transition_enabled = self.settings.page_transition_enabled;
                config::save(&self.settings);
                self.toast(if self.settings.page_transition_enabled {
                    "Page transition: on"
                } else {
                    "Page transition: off"
                });
            }
            Action::ToggleStretch => self.toggle_stretch(),
            Action::ToggleAnimBar => self.playback.hidden = !self.playback.hidden,
            Action::ToggleFullscreen => {
                let fs = match self.window.fullscreen() {
                    Some(_) => None,
                    None => Some(Fullscreen::Borderless(None)),
                };
                self.window.set_fullscreen(fs);
            }
            Action::ToggleSpreadOffset => {
                self.reader.spread_offset ^= 1;
                if let Some(k) = &self.volume_key {
                    self.settings
                        .spread_offsets
                        .insert(k.clone(), self.reader.spread_offset as u8);
                    config::save(&self.settings);
                }
                // Re-anchor so the current view re-pairs with the new parity.
                self.reader.snap_to_view_start();
                self.reader.prefetch();
                self.toast(format!("Spread offset: {}", self.reader.spread_offset));
            }
            Action::PrevVolume => self.jump_volume(-1),
            Action::NextVolume => self.jump_volume(1),
            Action::Rotate => {
                self.reader.rotation = (self.reader.rotation + 1) % 4;
                // Recenter: the rotated box has different bounds, so any prior pan
                // would now be out of range.
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
                self.reader.prefetch(); // re-decode at the rotation-aware target (1:1)
                self.toast(format!("Rotation: {}\u{00b0}", self.reader.rotation as u32 * 90));
            }
            Action::ShowInExplorer => self.reveal_current(),
            // Esc → quit is intercepted in `window_event` (needs the event loop),
            // so it never reaches here.
            Action::Quit => {}
        }
    }

    /// Raise a transient on-screen toast (boundary reached, zoom level).
    fn toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    /// Flip "stretch small pages" (key `Z` and the Settings panel row) — the shell
    /// side of [`yosh_engine::reader::Reader::fit_no_upscale`]. Shared by both entry
    /// points so they can't drift.
    ///
    /// Whether a fit may upscale changes every small page's *displayed* size, so the
    /// derived view state has to be re-derived with it: the zoom clamp is measured
    /// against the native scale, the pan clamp against the drawn box, and the scroll
    /// anchor against the strip's page heights. `prefetch` then re-queues the pages
    /// whose decode target moved — `JobsKey` carries the flag, so the job list is
    /// rebuilt exactly once rather than on the next unrelated change.
    fn toggle_stretch(&mut self) {
        self.settings.no_stretch = !self.settings.no_stretch;
        self.reader.fit_no_upscale = self.settings.no_stretch;
        self.reader.clamp_zoom_native();
        self.reader.clamp_pan();
        if self.reader.scroll_mode {
            self.reader.normalize();
        }
        self.reader.prefetch();
        config::save(&self.settings);
        // Phrased as the user-facing setting (stretching is the default), so the
        // toast matches the panel row rather than inverting it.
        self.toast(if self.settings.no_stretch {
            "Stretch small pages: off"
        } else {
            "Stretch small pages: on"
        });
    }

    /// "Show in Explorer" (key `E`): open the containing folder of the current
    /// item and select it. For a folder/single-image volume that's the current
    /// page's image file; for an archive it's the archive file itself.
    fn reveal_current(&mut self) {
        let Some(key) = self.volume_key.clone() else {
            self.toast("Nothing open");
            return;
        };
        let base = PathBuf::from(&key);
        let target = if base.is_dir() {
            // Folder (incl. single-image opens, which open the parent folder):
            // select the current page's file. `name(index)` is a flat file name.
            match self.reader.source.as_ref() {
                Some(s) if !s.is_empty() => base.join(s.name(self.reader.index)),
                _ => base,
            }
        } else {
            base // archive (or single file): select it in its folder
        };
        reveal_in_explorer(&target);
        self.toast("Shown in Explorer");
    }

    /// Open the previous (`delta < 0`) or next (`delta > 0`) sibling volume of
    /// the same kind — folder ↔ folder, archive ↔ archive — in natural-sort
    /// order within the current volume's parent directory (`[` / `]`). The
    /// reading mode/position of the current volume is persisted by `open`; the
    /// new one resumes its own saved page. No-op at the ends or with nothing open.
    fn jump_volume(&mut self, delta: i64) {
        // Base "current" on the pending target if an open is in flight, so a
        // second `[`/`]` advances from the not-yet-loaded neighbor instead of
        // repeating it.
        let cur = match self.opening_key.clone() {
            Some(p) => p,
            None => match &self.volume_key {
                Some(k) => PathBuf::from(k),
                None => return,
            },
        };
        let Some(parent) = cur.parent().map(|p| p.to_path_buf()) else {
            return;
        };
        let want_folder = cur.is_dir();
        // Use the background-warmed cache when it matches this folder. On a miss
        // (cold cache, e.g. a `[`/`]` in the first moments after launch) don't scan
        // here — a parent-dir `read_dir` + per-entry stat on a network share would
        // freeze the UI. Park the jump, warm the cache off-thread, and replay it in
        // `poll_background` once the listing lands.
        let hit = matches!(&self.sib_cache, Some((p, wf, _)) if *p == parent && *wf == want_folder);
        if !hit {
            self.pending_sib_jump = Some(self.pending_sib_jump.take().unwrap_or(0) + delta);
            self.warm_sib_cache(&cur);
            return;
        }
        let sibs = &self.sib_cache.as_ref().unwrap().2;
        let cur_name = cur.file_name();
        let Some(idx) = sibs.iter().position(|p| p.file_name() == cur_name) else {
            return;
        };
        let target = idx as i64 + delta;
        if target < 0 {
            self.toast("First book");
            return;
        }
        let next = sibs.get(target as usize).cloned();
        match next {
            Some(path) => self.open(&path),
            None => self.toast("Last book"),
        }
    }

    /// Snapshot the window's restored (non-maximized, non-fullscreen) geometry and
    /// whether it is maximized. `win_geom` keeps the last *normal* rect, which is
    /// what an un-maximize should return to, and what we persist.
    ///
    /// **Called once per frame from `render`, never from the event handlers.** On
    /// Windows a maximize delivers `WM_MOVE` before `WM_SIZE`, and winit only sets
    /// its `MAXIMIZED` flag while handling `WM_SIZE` — so a `Moved` handler sees
    /// `is_maximized() == false` alongside the already-maximized rect and records
    /// the filled-screen rect as the restore target. (That poisoned the saved
    /// geometry: the window reopened correctly maximized, but un-maximizing landed
    /// on a near-fullscreen rect, and exiting from fullscreen reopened *windowed*
    /// at that size.) By render time both messages have been processed, so the
    /// maximized flag and the rect agree.
    fn record_window_geometry(&mut self) {
        // Fullscreen reports the filled-screen rect and hides the underlying
        // maximized state, so leave both snapshots alone and keep what we had.
        if self.window.fullscreen().is_some() {
            return;
        }
        self.win_maximized = self.window.is_maximized();
        if self.win_maximized {
            return; // maximized rect isn't the restore target
        }
        let Ok(pos) = self.window.outer_position() else {
            return;
        };
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.win_geom = Some((pos.x, pos.y, size.width, size.height));
        }
    }

    fn persist(&mut self) {
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), self.reader.index);
        }
        // Save geometry + the maximized flag. Both come from the per-frame snapshot:
        // `win_geom` holds the last *normal* rect (so an un-maximize after restart
        // returns to the right size/position) and `win_maximized` is the last state
        // seen outside fullscreen (so quitting from fullscreen still reopens
        // maximized rather than windowed at the filled-screen size).
        if let Some((x, y, w, h)) = self.win_geom {
            self.settings.window = Some(config::WindowState {
                x,
                y,
                w,
                h,
                maximized: self.win_maximized,
            });
        }
        config::save(&self.settings);
    }

    /// After a successful self-update: persist, launch the freshly-replaced exe
    /// (reopening the current volume), and exit.
    fn relaunch(&mut self) -> ! {
        self.persist();
        if let Ok(exe) = std::env::current_exe() {
            let mut cmd = std::process::Command::new(exe);
            if let Some(k) = &self.volume_key {
                cmd.arg(k);
            }
            let _ = cmd.spawn();
        }
        std::process::exit(0);
    }

    /// Foreground chrome with nothing scrollable in it: hovering these must not
    /// swallow the wheel — the reader keeps navigating (clicks/drags on the
    /// widget still work normally).
    const WHEEL_PASSTHROUGH: [&'static str; 2] = ["seekbar", "anim_panel"];

    /// Is the pointer over one of those layers?
    ///
    /// Hit-tested against the *layer* rect via `layer_id_at` — which is exactly
    /// what `egui_wants_pointer_input()` (the `consumed` flag this cancels) uses,
    /// read from the same completed frame. A per-widget `contains_pointer()` used
    /// to stand in for this and covered a smaller rect than the layer, leaving a
    /// dead strip over the seekbar frame's margin — painted as part of the pill,
    /// but outside the widget — where neither side of the guard fired.
    fn wheel_passthrough(&self) -> bool {
        let Some(pos) = self.egui_ctx.input(|i| i.pointer.interact_pos()) else {
            return false;
        };
        self.egui_ctx
            .layer_id_at(pos)
            .is_some_and(|l| Self::WHEEL_PASSTHROUGH.iter().any(|&id| l.id == egui::Id::new(id)))
    }

    /// Wheel notches this delta is worth. A mouse's detent is one `LineDelta`; a
    /// precision touchpad reports `PixelDelta`, where the same physical gesture arrives
    /// as many small deltas — `WHEEL_PX_PER_NOTCH` is what makes a two-finger swipe
    /// worth a comparable number of notches.
    fn wheel_notches(delta: MouseScrollDelta) -> f32 {
        const WHEEL_PX_PER_NOTCH: f32 = 50.0;
        match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => p.y as f32 / WHEEL_PX_PER_NOTCH,
        }
    }

    /// Mouse wheel: zoom at the cursor with Ctrl held; otherwise pan within an
    /// overflowing page, or flip at the edges / when the page already fits.
    fn on_wheel(&mut self, delta: MouseScrollDelta) {
        if self.ctrl_held {
            // Zoom about the cursor in *content* space: `reader.viewport` excludes the
            // pinned top bar, so the cursor's window y has to lose that inset or the
            // anchored point drifts by the bar's height.
            let steps = bank_notches(&mut self.zoom_notches, Self::wheel_notches(delta));
            if steps == 0 {
                return;
            }
            let focal = (self.cursor_x as f32, self.cursor_y as f32 - self.top_inset_px());
            for _ in 0..steps.abs() {
                self.reader.zoom_to_preset_about(steps > 0, Some(focal));
            }
            return;
        }
        if self.reader.scroll_mode {
            // The wheel is the one abstract ratchet in the app — its px-per-line
            // constant is a made-up number — so it is what the speed setting scales.
            // A finger drag is direct manipulation (stays 1:1) and a release fling
            // continues the gesture that was actually measured; neither is scaled.
            let dy_px = match delta {
                MouseScrollDelta::LineDelta(_, y) => y * SCROLL_WHEEL_PX,
                MouseScrollDelta::PixelDelta(p) => p.y as f32,
            } * self.settings.scroll_speed;
            self.reader.scroll_by(-dy_px); // wheel down (y<0) scrolls the strip down
            return;
        }
        let dy = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
        };
        if dy == 0.0 {
            return;
        }
        let overflow = self.reader.current_overflows();
        if !overflow {
            // Page fits: wheel flips (down = forward).
            self.reader.step(if dy < 0.0 { 1 } else { -1 });
            return;
        }
        // Vertical pan in px. At the top/bottom edge, hard-stop first: a scroll
        // that would overshoot the edge just parks at it; only a *further* scroll
        // once already parked flips the page. So scrolling to the end of a zoomed
        // page stops there instead of immediately jumping to the next page — you
        // have to keep scrolling past the stop to advance. Only reset the pan when
        // a flip actually happened (else the first/last page snaps to its edge).
        let sh = self.reader.viewport.h.max(1) as f32;
        let maxp = ((self.reader.current_display_h() - sh) / 2.0).max(0.0);
        let cur = self.reader.pan_y.clamp(-maxp, maxp);
        let next = cur + dy * 80.0;
        let now = Instant::now();
        // True once we've been parked at an edge long enough that a further
        // scroll should flip (the hard stop the user has to keep scrolling past).
        let dwelt = self.reader.pan_edge_at.is_some_and(|t| now.duration_since(t) >= EDGE_DWELL);
        if next > maxp + 0.5 {
            if cur >= maxp - 0.5 {
                // Parked at the top: flip to the previous page only after dwelling.
                if dwelt {
                    self.reader.pan_edge_at = None;
                    self.reader.pan_y = if self.reader.step(-1) { -1.0e6 } else { maxp };
                } else {
                    self.reader.pan_y = maxp; // hold the stop
                    self.reader.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.reader.pan_y = maxp; // just reached the top edge -> park + start dwell
                self.reader.pan_edge_at = Some(now);
            }
        } else if next < -maxp - 0.5 {
            if cur <= -maxp + 0.5 {
                // Parked at the bottom: flip to the next page only after dwelling.
                if dwelt {
                    self.reader.pan_edge_at = None;
                    self.reader.pan_y = if self.reader.step(1) { 1.0e6 } else { -maxp };
                } else {
                    self.reader.pan_y = -maxp; // hold the stop
                    self.reader.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.reader.pan_y = -maxp; // just reached the bottom edge -> park + start dwell
                self.reader.pan_edge_at = Some(now);
            }
        } else {
            self.reader.pan_y = next;
            self.reader.pan_edge_at = None; // panning within the page
        }
    }

    /// A clean click: the left/right edge strips flip pages; the wide middle
    /// does nothing on a single click but toggles fullscreen on a double-click.
    fn on_click(&mut self) {
        let w = self.reader.viewport.w.max(1) as f64;
        let edge = (w * EDGE_FRAC as f64).max(1.0);
        if self.cursor_x < edge {
            self.last_mid_click = None;
            self.apply_action(Action::Left);
        } else if self.cursor_x > w - edge {
            self.last_mid_click = None;
            self.apply_action(Action::Right);
        } else {
            let now = Instant::now();
            let is_double = self
                .last_mid_click
                .is_some_and(|t| now.duration_since(t) < DOUBLE_CLICK);
            if is_double {
                self.last_mid_click = None;
                self.apply_action(Action::ToggleFullscreen);
            } else {
                self.last_mid_click = Some(now);
            }
        }
    }

    fn on_cursor_moved(&mut self, x: f64, y: f64) {
        let dx = (x - self.cursor_x) as f32;
        let dy = (y - self.cursor_y) as f32;
        self.cursor_x = x;
        self.cursor_y = y;
        self.cursor_in_window = true;
        if !self.mouse_down {
            return;
        }
        self.drag_dist += dx.abs() + dy.abs();
        // Sample (time, x, y) so the release can glide on from the recent pointer
        // velocity — the *same* tracker (and ~100 ms window) the touch path uses,
        // so a strip or page thrown with the pointer behaves identically either way.
        self.gestures.samples.push(Instant::now(), x, y);
        if self.reader.scroll_mode {
            // Grab the strip: pan horizontally, scroll vertically.
            self.reader.pan_x += dx;
            self.reader.top_offset -= dy;
            self.reader.clamp_pan();
            self.reader.normalize();
        } else {
            self.reader.pan_x += dx;
            self.reader.pan_y += dy;
            self.reader.clamp_pan();
        }
    }

    /// Left button: a clean press/release (little movement) flips; a press +
    /// drag pans instead, and a thrown strip — or a thrown zoomed page — keeps
    /// gliding, on exactly the physics a finger gets.
    fn on_left_button(&mut self, pressed: bool) {
        if pressed {
            self.mouse_down = true;
            self.drag_dist = 0.0;
            // A press catches any in-flight glide (strip or pan) and starts a
            // fresh velocity window.
            self.reader.stop_fling();
            self.reader.stop_pan_fling();
            self.gestures.samples.clear();
        } else {
            if self.mouse_down && self.drag_dist >= 6.0 {
                // Grab-and-throw: release velocity from the recent samples, on the
                // same rule (and dt floor) as a finger release.
                let now = Instant::now();
                let (vx, vy) = self.gestures.samples.velocity(now, self.cursor_x, self.cursor_y);
                self.gestures.mark_fling_start(now);
                if self.reader.scroll_mode {
                    self.reader.start_fling(-vy as f32); // strip velocity = −pointer velocity
                    if self.reader.zoom > 1.001 {
                        self.reader.start_pan_fling(vx as f32, 0.0);
                    }
                } else {
                    // Page-flip pan: the page sticks to the pointer, so the glide
                    // continues in the pointer's own direction (no negation). On a
                    // page that doesn't overflow, `clamp_pan` pins it at once and
                    // `pan_fling_tick` kills the velocity on its first frame.
                    self.reader.start_pan_fling(vx as f32, vy as f32);
                }
            }
            if self.mouse_down && self.drag_dist < 6.0 {
                self.on_click();
            }
            self.mouse_down = false;
        }
    }

    /// Route a touch event into the shared gesture machine, then apply the one
    /// decision it leaves to the shell. `FlipCommitted` and `EdgeDragRelease` are
    /// deliberately ignored here: a desktop flip keeps the zoom/pan it had (same
    /// as a click flip), and there is no next/prev-book prompt to arm.
    fn handle_touch(&mut self, phase: TouchPhase, id: u64, x: f64, y: f64, egui_consumed: bool) {
        // Windows may also synthesize mouse events for this finger. The mouse arms
        // already stand down while a touch is live; dropping any drag they had
        // begun keeps a stale one from resuming.
        if phase == TouchPhase::Started {
            self.mouse_down = false;
        }
        let ctx = GestureCtx {
            // Raw surface, not `reader.viewport`: the thresholds are calibrated
            // against the whole window, which the pinned top bar doesn't shrink.
            surface_w: self.gpu.config.width as f64,
            surface_h: self.gpu.config.height as f64,
            top_inset: self.top_inset_px() as f64,
            egui_consumed,
            library_view: self.library_view,
        };
        let phase = match phase {
            TouchPhase::Started => Phase::Start,
            TouchPhase::Moved => Phase::Move,
            TouchPhase::Ended => Phase::End,
            TouchPhase::Cancelled => Phase::Cancel,
        };
        // `redraw` is ignored: the desktop event loop's redraw guard already buys a
        // frame for every input event.
        let resp =
            self.gestures.on_touch(&mut self.reader, &ctx, phase, id, x, y, Instant::now());
        for ev in resp.events {
            if let GestureEvent::Tap { x, y } = ev {
                self.on_tap(x, y);
            }
        }
    }

    /// A touch tap routes through the existing click semantics: edge strips flip,
    /// the middle double-taps to fullscreen. `on_click` reads the cursor fields.
    fn on_tap(&mut self, x: f64, y: f64) {
        self.cursor_x = x;
        self.cursor_y = y;
        self.on_click();
    }

    /// Advance the inertial scroll glide one frame (a no-op unless flinging): apply
    /// the velocity to the strip, decay it, and schedule the next frame while it
    /// lasts. Self-sustaining, and deliberately *not* a `deadlines()` entry — a
    /// glide is an animation, so it belongs to the redraw guard, not the scheduler.
    fn tick_scroll_fling(&mut self) {
        if self.gestures.tick(&mut self.reader, Instant::now()) {
            self.window.request_redraw();
        }
    }

    /// Decode up to `budget` not-yet-tried library cover thumbnails this frame
    /// and register them with egui.
    /// Update the open volume's read-tracking entry: the furthest page ever shown
    /// (1-based count; the far page of a spread counts) and the volume's total.
    /// Called every rendered frame while reading — cheap map math; the file write
    /// rides along with the existing `config::save`. Mirrors the Android shell.
    fn note_progress(&mut self) {
        let Some(key) = self.volume_key.clone() else {
            return;
        };
        let len = match &self.reader.source {
            Some(s) => s.len(),
            None => return,
        };
        if len == 0 {
            return;
        }
        let (a, b) = self.reader.visible_pages();
        let seen = b.unwrap_or(a) + 1;
        let e = self.settings.progress.entry(key).or_insert((0, 0));
        e.0 = e.0.max(seen);
        e.1 = len;
    }

    /// Free every registered library cover texture (before a rescan / new library,
    /// which then replaces the volumes and their backing `PageTexture`s).
    fn free_cover_textures(&mut self) {
        for v in self.library.all_volumes() {
            if let Some(id) = v.thumb {
                self.egui_renderer.free_texture(&id);
            }
        }
    }

    /// Kick off a recursive library scan off the main thread (generation-tagged so a
    /// newer pick/rescan wins). The result is applied in `poll_background`.
    fn start_scan(&mut self, root: PathBuf) {
        self.scan_gen = self.scan_gen.wrapping_add(1);
        self.scanning = true;
        let generation = self.scan_gen;
        let tx = self.scan_tx.clone();
        let wake = frame_waker(&self.window);
        std::thread::spawn(move || {
            let _ = tx.send((generation, Library::scan(&root)));
            wake(); // send-then-wake
        });
        self.window.request_redraw();
    }

    /// Replace the displayed library: free the old cover textures, reset the cover
    /// queue, and rebuild the path→volume index used to route decoded covers.
    fn set_library(&mut self, lib: Library) {
        self.free_cover_textures();
        self.queued_covers.clear();
        self.cover_loc.clear();
        for (si, s) in lib.series.iter().enumerate() {
            for (vi, v) in s.volumes.iter().enumerate() {
                self.cover_loc.insert(v.path.clone(), (si, vi));
            }
        }
        self.library = lib;
    }

    /// Queue the covers the sectioned library reported on screen this frame
    /// (`ui.visible_covers`) for OFF-THREAD decode, and evict the least-recently-seen
    /// resident textures past a cap so a deep library can't pin hundreds of MB. The
    /// heavy read+decode+downscale (and the disk-cache lookup) runs on a worker; the
    /// main thread only uploads + registers finished thumbnails in `poll_background`.
    fn pump_covers(&mut self) {
        let visible = std::mem::take(&mut self.ui.visible_covers);
        if !visible.is_empty() {
            self.cover_clock = self.cover_clock.wrapping_add(1);
            let clock = self.cover_clock;
            // Stamp recency for visible covers; gather undecoded, not-yet-queued ones.
            let mut batch: Vec<(PathBuf, VolKind)> = Vec::new();
            for path in &visible {
                if let Some(&(si, vi)) = self.cover_loc.get(path) {
                    let v = &mut self.library.series[si].volumes[vi];
                    v.last_seen = clock;
                    if v.thumb.is_none() && self.queued_covers.insert(path.clone()) {
                        batch.push((path.clone(), v.kind));
                    }
                }
            }
            if !batch.is_empty() {
                let tx = self.cover_tx.clone();
                let cache_dir = config::cache_dir();
                let wake = frame_waker(&self.window);
                std::thread::spawn(move || {
                    let mut resizer = Resizer::new();
                    for (path, kind) in batch {
                        if let Some(img) = yosh_engine::thumbcache::load_or_decode(
                            cache_dir.as_deref(),
                            &path,
                            THUMB_H,
                            &mut resizer,
                            || cover_bytes(&path, kind),
                        ) {
                            if tx.send((path, img)).is_err() {
                                break; // receiver gone (app closing)
                            }
                            wake(); // send-then-wake, per cover — they stream in
                        }
                    }
                });
            }
        }

        // LRU eviction: keep only the most-recently-seen covers resident.
        const THUMB_CAP: usize = 192;
        let live = self.library.all_volumes().filter(|v| v.thumb.is_some()).count();
        if live > THUMB_CAP {
            let mut ages: Vec<(u64, usize, usize)> = Vec::new();
            for (si, s) in self.library.series.iter().enumerate() {
                for (vi, v) in s.volumes.iter().enumerate() {
                    if v.thumb.is_some() {
                        ages.push((v.last_seen, si, vi));
                    }
                }
            }
            ages.sort_by_key(|&(age, _, _)| age);
            for (_, si, vi) in ages.into_iter().take(live - THUMB_CAP) {
                let path = self.library.series[si].volumes[vi].path.clone();
                let v = &mut self.library.series[si].volumes[vi];
                if let Some(id) = v.thumb.take() {
                    self.egui_renderer.free_texture(&id);
                }
                v.thumb_tex = None; // dropping the texture frees it
                self.queued_covers.remove(&path); // allow a re-decode when scrolled back
            }
        }
    }

    /// Begin opening `path`. The source is built on a background thread (see
    /// `build_source`) so a slow network-share open never freezes the UI — the
    /// current page stays on screen under the spinner until the new source lands
    /// in `render`. Each call bumps `open_gen`; only the newest result is applied,
    /// so rapid `[`/`]` supersede in-flight opens instead of queuing stale swaps.
    fn open(&mut self, path: &Path) {
        self.open_t0 = Some(Instant::now());
        if cfg!(debug_assertions) {
            eprintln!("[open] initiated: {}", path.display());
        }
        self.open_gen = self.open_gen.wrapping_add(1);
        let generation = self.open_gen;
        let tx = self.open_tx.clone();
        let path = path.to_path_buf();
        self.opening = true;
        self.opening_key = Some(path.clone());
        let wake = frame_waker(&self.window);
        // Where reading will most likely resume, mirroring `set_source`'s precedence
        // (a pending CLI start index beats the saved position). Computed here because
        // the closure can't touch `self`. Advisory only — `set_source` re-resolves it
        // authoritatively when the source lands, so a mismatch costs extraction order,
        // never correctness.
        let hint = if self.reader.start_index > 0 {
            Some(self.reader.start_index)
        } else {
            self.settings
                .last_pages
                .get(path.to_string_lossy().as_ref())
                .copied()
                .filter(|&i| i > 0)
        };
        std::thread::spawn(move || {
            let _ = tx.send((generation, build_source(&path, hint)));
            wake(); // send-then-wake (the drain for this one lives in `render`)
        });
    }

    fn set_source(&mut self, source: Arc<dyn PageSource>, path: &Path, start: Option<usize>) {
        // Persist the previous volume's position before switching.
        if let Some(k) = self.volume_key.take() {
            self.settings.last_pages.insert(k, self.reader.index);
        }
        let key = path.to_string_lossy().into_owned();
        self.reader.spread_offset = self.settings.spread_offsets.get(&key).copied().unwrap_or(0) as usize;
        // Explicit start (e.g. a specific dropped image) wins; else CLI start
        // index; else the saved position.
        let idx = match start {
            Some(i) => i,
            None => {
                let resume = self.settings.last_pages.get(&key).copied().unwrap_or(0);
                if self.reader.start_index > 0 {
                    self.reader.start_index
                } else {
                    resume
                }
            }
        };
        self.reader.start_index = 0;

        let pool = DecodePool::new(
            source.clone(),
            self.gpu.device.clone(),
            self.gpu.queue.clone(),
            self.reader.tex_pool.clone(),
            self.reader.workers,
        );
        // A fresh pool starts wakerless: hand it the window's frame waker (kept on
        // the reader) or this volume's decodes would land without scheduling the
        // frame that draws them.
        pool.set_waker(self.reader.waker.clone());
        self.reader.pool = Some(pool);
        self.reader.reset_volume_state();
        self.info_for = None;
        self.reader.nav_times.clear();
        self.reader.rotation = 0; // each volume opens upright
        self.reader.index = idx.min(source.len() - 1);
        push_recent(&mut self.settings, &key);
        self.volume_key = Some(key);
        self.ui.opened = Some(path.to_path_buf());
        self.reader.source = Some(source);
        self.library_view = false; // opening anything switches to the reader
        // Fresh pool, fresh caches: drop the cross-frame prefetch memo *and* the
        // thumbnail-tail pivot, or the new pool's empty tail would stay empty until
        // the reader happened to travel a full stride.
        self.reader.invalidate_jobs();
        self.reader.prefetch();
        // Live refresh: watch a folder (added/removed images) or a still-being-written
        // .cbz/.zip (pages appended on disk). RAR/7z can't be partially read, so this
        // tears the watcher down for them.
        self.install_folder_watch(path);
        // Warm the sibling-volume list for this folder in the background, so the
        // first `[`/`]` press doesn't pay the parent-dir scan on the main thread.
        self.warm_sib_cache(path);
    }

    /// (Re)install the live-refresh watcher for `path`, or tear it down for a volume
    /// that can't grow. We watch a **folder** (`build_source` hands `set_source` the
    /// *directory* for both folder and single-image opens, so `is_dir()` also covers
    /// "opened one image, then siblings appear") or a still-being-written **`.cbz`/`.zip`**
    /// file (recovered via `ZipSource`'s local-header scan). RAR/7z/complete files aren't
    /// watched. A `notify` failure is non-fatal — the feature is simply inactive for that volume.
    ///
    /// Only the *reset* runs here: picking the target (`is_dir`, `canonicalize`) and
    /// registering the watch are filesystem calls that can stall for seconds on a
    /// network share, and this sits on `set_source`'s main-thread path — right where
    /// the first page is trying to reach the screen. So the install runs on a worker
    /// and is adopted in `poll_background`, tagged with the open generation so a
    /// volume switch during the gap discards it. The few unwatched milliseconds cost
    /// nothing: the feature is already debounced/best-effort, and the listing
    /// `build_source` just produced is by definition current.
    fn install_folder_watch(&mut self, path: &Path) {
        // Reset any pending refresh carried over from the previous volume, and drop
        // the old watch (dropping the watcher unregisters it). Synchronous: it must
        // be true the moment the volume changes, whatever the install thread does.
        self.watch_dirty = None;
        self.watch_dirty_since = None;
        self.rescanning = false;
        self.watcher = None;
        self.watch_filter = None;
        // Everything captured here is cheap; the filesystem work is all inside the
        // thread. `watch_tx`'s handler waker and the install waker are separate
        // clones — the handler outlives this call, held by the watcher itself.
        let generation = self.open_gen;
        let install_tx = self.watch_install_tx.clone();
        let event_tx = self.watch_tx.clone();
        let wake = frame_waker(&self.window);
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let handler_wake = wake.clone();
            let built = (|| {
                // Pick what to watch: a folder directly, or — for a growing archive — its
                // parent directory (notify's Windows backend watches directories; a lone-file
                // watch is unreliable, especially for a writer holding the file open and
                // appending). For the archive case we keep the file path to filter the dir's
                // events down to it.
                let (target, filter): (PathBuf, Option<PathBuf>) = if path.is_dir() {
                    (path.clone(), None)
                } else if is_growable_archive(&path) {
                    // Canonicalize so the parent dir resolves even for a relative/bare-filename
                    // path (whose `parent()` would be ""); fall back to the path as given.
                    let abs = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    let parent = abs.parent()?.to_path_buf(); // no parent dir to watch
                    (parent, Some(abs))
                } else {
                    return None; // RAR/7z/complete file: can't grow, not watched
                };
                // notify calls the handler from its own thread, so the event only reaches
                // the debounce in `poll_folder_watch` when a frame runs: send-then-wake.
                let handler = move |res| {
                    let _ = event_tx.send(res);
                    handler_wake();
                };
                let mut w = notify::recommended_watcher(handler).ok()?;
                w.watch(&target, notify::RecursiveMode::NonRecursive).ok()?;
                Some((w, filter))
            })();
            let _ = install_tx.send((generation, built));
            wake(); // send-then-wake (adopted by `poll_background`)
        });
    }

    /// Drive the live-refresh watcher each frame: coalesce filesystem-change events into
    /// a debounce stamp, kick off a debounced off-thread rebuild of the open volume once
    /// it has settled (or at least every `MAX_WAIT` under a continuous writer), and apply
    /// a finished rebuild to the reader (which preserves position + decoded cache by name).
    /// Returns true when a rebuild was applied (the view changed — worth a frame).
    fn poll_folder_watch(&mut self) -> bool {
        // Coalesce incoming change events. A bulk copy / streaming archive write fires
        // many — collapse them. `watch_dirty` tracks the latest event (settle delay);
        // `watch_dirty_since` the first of the streak (max-wait cap). Pure access events
        // (reads) don't change the listing, so they're ignored.
        while let Ok(ev) = self.watch_rx.try_recv() {
            let Ok(e) = ev else { continue };
            if e.kind.is_access() {
                continue; // a read, not a listing change
            }
            // Archive watch: the parent dir is watched, so ignore events for sibling
            // files (compare by file name — the dir watch is flat, so names are unique).
            if let Some(target) = &self.watch_filter
                && !e.paths.iter().any(|p| p.file_name() == target.file_name())
            {
                continue;
            }
            let now = Instant::now();
            self.watch_dirty = Some(now);
            self.watch_dirty_since.get_or_insert(now);
        }
        // Fire once the burst has settled for `WATCH_QUIET`, OR it's been churning for
        // `WATCH_MAX_WAIT` without a gap (a continuously-streamed archive never goes quiet,
        // so the settle alone would starve). The rebuild runs off-thread (a `read_dir` or a
        // full archive local-header scan shouldn't touch the UI thread); `rescanning`
        // prevents overlap; the watcher is only present for watched volumes, so this never
        // fires otherwise. `watch_deadline` mirrors this test for the frame scheduler —
        // keep the two in step.
        let ready = self.watch_dirty.is_some_and(|t| t.elapsed() >= WATCH_QUIET)
            || self.watch_dirty_since.is_some_and(|t| t.elapsed() >= WATCH_MAX_WAIT);
        if ready
            && !self.rescanning
            && self.watcher.is_some()
            && let Some(path) = self.ui.opened.clone()
        {
            self.watch_dirty = None;
            self.watch_dirty_since = None;
            self.rescanning = true;
            let generation = self.open_gen;
            let tx = self.rescan_tx.clone();
            let wake = frame_waker(&self.window);
            std::thread::spawn(move || {
                // Reuse the open-path dispatcher: a directory rebuilds a `FolderSource`, a
                // still-growing `.cbz`/`.zip` a `ZipSource` (local-header recovery). On a
                // transient error (mid-write, no complete pages yet) it sends `None` and
                // the next debounced refresh retries; `apply_refreshed_source` also no-ops
                // on an empty listing.
                // No resume hint: only folders and still-growing zips are watched, and
                // neither is a sequential archive, so nothing would read it.
                let built = build_source(&path, None).ok().map(|(src, _, _)| src);
                // Always send (even on error) so `rescanning` clears.
                let _ = tx.send((generation, built));
                wake(); // send-then-wake
            });
        }
        // Apply finished rebuilds — newest generation only (a `[`/`]` volume switch
        // bumped `open_gen`, exactly like the `open_rx` guard above).
        let mut refreshed = false;
        while let Ok((generation, built)) = self.rescan_rx.try_recv() {
            self.rescanning = false;
            if generation == self.open_gen
                && let Some(source) = built
            {
                self.reader.apply_refreshed_source(source);
                refreshed = true;
            }
        }
        refreshed
    }

    /// When the pending watch burst is due to rebuild the volume — exactly the
    /// instant `poll_folder_watch`'s `ready` test above flips true: the settle
    /// delay measured from the **last** event (`watch_dirty`) or the max-wait cap
    /// measured from the **first** (`watch_dirty_since`), whichever comes first.
    ///
    /// `None` when no burst is pending — and also in the cases where `ready` would
    /// be true but the rebuild is gated off (one already in flight, or the volume
    /// isn't watched at all, e.g. stale events drained after switching to a RAR):
    /// there the stamps deliberately stay standing, so a scheduled frame could not
    /// consume them and the scheduler would re-fire forever. A finishing rescan
    /// wakes the loop itself (`rescan_tx` + waker), which re-opens the question.
    fn watch_deadline(&self) -> Option<Instant> {
        if self.rescanning || self.watcher.is_none() || self.ui.opened.is_none() {
            return None;
        }
        let settle = self.watch_dirty.map(|t| t + WATCH_QUIET);
        let cap = self.watch_dirty_since.map(|t| t + WATCH_MAX_WAIT);
        match (settle, cap) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Every timer the shell has armed, for `about_to_wait`'s scheduler: state that
    /// was set at some T and must become visible at T + delay with no further input.
    /// Each is armed **only while its transition is still pending**, so the frame a
    /// ripe deadline buys always consumes it — a deadline the frame couldn't clear
    /// would be re-scheduled every pass and spin the loop.
    fn deadlines(&self) -> [Option<Instant>; 3] {
        [
            // Live folder refresh: the debounced rebuild (mirrors `poll_folder_watch`).
            self.watch_deadline(),
            // Loading spinner grace: the moment the centered spinner is due to appear.
            // Dropped once it *has* appeared (`ui.loading`) — from then on the page
            // either lands (its decode wakes a frame) or egui's own `Spinner` keeps
            // repainting, and re-arming a deadline the frame can't clear would spin.
            match self.loading_pending {
                Some((_, t)) if !self.ui.loading => Some(t + LOADING_INDICATOR_DELAY),
                _ => None,
            },
            // Toast expiry: `render` drops the message once this passes.
            self.toast.as_ref().map(|&(_, t)| t + TOAST_DURATION),
        ]
    }

    /// Freeze the decode pool because the window can't be seen (minimized, or fully
    /// occluded). A hidden window otherwise keeps every worker busy — the prefetch
    /// window, and behind it the whole-volume thumbnail tail, which on a long book
    /// runs for a good while — all of it real CPU spent behind a surface nobody is
    /// looking at. So parking is worth it the moment the pixels stop being visible,
    /// and costs nothing when the pool happened to be idle already.
    ///
    /// Park is a *pause*, not a cancel: the queues survive, and the caches are
    /// deliberately kept (desktop RAM is plentiful and restoring should be
    /// instant — no `lq_cache` clear, unlike the memory-pressed Android shell).
    /// A no-op when nothing is open: the pool is `None` until a book is.
    fn park(&mut self) {
        if self.parked {
            return;
        }
        self.parked = true;
        // An inertial glide would otherwise keep asking for frames behind a surface
        // nobody can see — and would land the strip somewhere the user never watched
        // it travel to. Kill it here; the position it had is what comes back.
        self.reader.stop_fling();
        self.reader.stop_pan_fling();
        if let Some(pool) = &self.reader.pool {
            pool.park();
        }
    }

    /// Thaw after [`Self::park`]. Decodes abandoned mid-flight during the park are
    /// simply re-queued: `invalidate_jobs` drops the cross-frame prefetch memo (the
    /// `JobsKey` is unchanged by a park, so without this the next `prefetch` would
    /// no-op and the pool would sit idle), and the redraw buys the frame that runs
    /// that prefetch — same pairing as a fresh pool in `set_source`.
    fn unpark(&mut self) {
        if !self.parked {
            return;
        }
        self.parked = false;
        if let Some(pool) = &self.reader.pool {
            pool.unpark();
        }
        self.reader.invalidate_jobs();
        self.window.request_redraw();
    }

    /// The tier this machine should be running at right now (the setting, or
    /// `Auto`'s power-source rule).
    fn effective_tier(&self) -> DeviceTier {
        effective_tier(self.settings.perf, self.on_battery)
    }

    /// Apply the performance profile to the live reader — **no book reopen**. Every
    /// budget field is runtime-settable: the window sizes are plain fields, both
    /// caches and the texture pool re-cap in place (evicting down keeps the pages
    /// nearest the read position, so nothing on screen flashes), and the worker count
    /// is the one thing that needs a new pool. Rebuilding the pool is cheap and
    /// hitch-free because its teardown is signal-only and the caches are deliberately
    /// **kept** — the current page stays decoded and on screen while the new pool
    /// refills around it. Port of the Android shell's `apply_perf`.
    ///
    /// A no-op when the effective tier hasn't moved, so re-picking the selected
    /// option (or a power re-check that found nothing new) costs nothing.
    fn apply_perf(&mut self) {
        let tier = self.effective_tier();
        if tier == self.applied_tier {
            return;
        }
        self.applied_tier = tier;
        let b = Budget::for_tier(tier, self.mem_budget_mb, self.cpus);
        let current = self.reader.index;
        self.reader.workers = b.workers;
        self.reader.fwd = b.fwd;
        self.reader.fwd_max = b.fwd_max;
        self.reader.back = b.back;
        self.reader.actual_cap_vh = b.actual_cap_vh;
        self.reader.lq_thumb_h = b.lq_thumb_h;
        self.reader.lq_tier = b.lq_tier;
        self.reader.cache.set_cap(b.cache_cap, current);
        self.reader.lq_cache.set_cap(b.lq_cap, current);
        self.reader.tex_pool.set_max_total(b.texpool_max);
        // Nothing open (library view): the fields above are enough — the next book's
        // pool is built with `reader.workers`.
        let Some(src) = self.reader.source.clone() else {
            return;
        };
        let pool = DecodePool::new(
            src,
            self.gpu.device.clone(),
            self.gpu.queue.clone(),
            self.reader.tex_pool.clone(),
            b.workers,
        );
        // A fresh pool starts wakerless and with empty queues: re-install the wake
        // callback, then force a full rebuild of both the job list and the thumbnail
        // tail (`JobsKey` doesn't capture budget fields, so the memoized key would
        // otherwise suppress the very prefetch that refills the new pool).
        pool.set_waker(self.reader.waker.clone());
        self.reader.pool = Some(pool);
        // A profile change while minimized must not un-park the reader.
        if self.parked
            && let Some(pool) = &self.reader.pool
        {
            pool.park();
        }
        self.reader.invalidate_jobs();
        self.reader.prefetch();
    }

    /// Poll the channels that background threads deliver on — called once per frame
    /// from `render`. Every producer wakes the loop right after its send, so a frame
    /// is always coming when there is something here to drain. Returns true when
    /// something user-visible landed and is worth one frame: an update-check result,
    /// an update-apply failure, or a live-refreshed volume listing.
    fn poll_background(&mut self) -> bool {
        let mut changed = false;
        // Startup resume scan: the first recent that still exists on disk (or None).
        // Apply it only if nothing has been opened in the meantime — a dropped file,
        // a library pick or a `[`/`]` during the scan wins over the stale resume.
        if let Some(rx) = self.resume_rx.take() {
            match rx.try_recv() {
                Err(std::sync::mpsc::TryRecvError::Empty) => self.resume_rx = Some(rx),
                // A `Disconnected` scan thread answered nothing and never will, so
                // fold it into the "nothing to resume" case — the spinner must not stick.
                res => {
                    changed = true;
                    if self.open_gen == 0
                        && self.ui.pending_open.is_none()
                        && self.reader.source.is_none()
                    {
                        match res.ok().flatten() {
                            Some(p) => self.ui.pending_open = Some(p),
                            // Nothing resumable left: drop the spinner and land on the
                            // library grid (the no-arg home screen).
                            None => {
                                self.opening = false;
                                self.library_view = true;
                            }
                        }
                    }
                }
            }
        }
        // Adopt a finished off-thread watcher install — newest generation only; a
        // stale one (volume already switched) is dropped here, which unregisters it.
        // No `changed` flag: an installed watcher isn't user-visible, and until it
        // lands `poll_folder_watch`'s rebuild gate and `watch_deadline` both see
        // `watcher: None` and stay quiescent — so no deadline is armed that the
        // frame it buys couldn't consume.
        while let Ok((generation, built)) = self.watch_install_rx.try_recv() {
            if generation == self.open_gen
                && let Some((w, filter)) = built
            {
                self.watcher = Some(w);
                self.watch_filter = filter;
            }
        }
        // Sibling-volume scans only warm the `[`/`]` cache — nothing to show.
        let mut sibs_landed = false;
        while let Ok(entry) = self.sib_rx.try_recv() {
            self.sib_cache = Some(entry);
            sibs_landed = true;
        }
        // …unless a `[`/`]` was parked waiting for exactly that listing: replay it now
        // that the cache is warm. If the volume moved folders meanwhile this re-warms
        // and re-parks instead — still terminating, since each pass needs a fresh scan.
        if sibs_landed
            && let Some(d) = self.pending_sib_jump.take()
        {
            self.jump_volume(d);
            changed = true;
        }
        // Off-thread library scan: apply the newest result (frees the old cover
        // textures and rebuilds the path→volume index).
        while let Ok((generation, lib)) = self.scan_rx.try_recv() {
            if generation == self.scan_gen {
                self.set_library(lib);
                self.scanning = false;
                changed = true;
            }
        }
        // Off-thread cover decodes: upload + register finished thumbnails (the only
        // steps that must run on the main thread).
        while let Ok((path, img)) = self.cover_rx.try_recv() {
            self.queued_covers.remove(&path);
            if let Some(&(si, vi)) = self.cover_loc.get(&path) {
                let pt = PagePipeline::upload(
                    &self.gpu.device,
                    &self.gpu.queue,
                    &img,
                    &self.reader.tex_pool,
                    0,
                );
                let id = self.egui_renderer.register_native_texture(
                    &self.gpu.device,
                    &pt.view,
                    wgpu::FilterMode::Linear,
                );
                let v = &mut self.library.series[si].volumes[vi];
                v.thumb = Some(id);
                v.thumb_tex = Some(pt);
                v.last_seen = self.cover_clock;
                changed = true;
            }
        }
        // Off-thread Tab info overlay: replace the placeholder rows with the full
        // ones. Both tags must still match — a result for a volume that has since
        // been switched (generation) or for pages already turned past (key) would
        // describe something other than what's on screen, so it's dropped.
        while let Ok((generation, key, blocks)) = self.info_rx.try_recv() {
            if generation != self.open_gen || self.info_for != Some(key) {
                continue;
            }
            let len = self.reader.source.as_ref().map_or(0, |s| s.len());
            let mut rows = Vec::new();
            for (n, (index, block)) in blocks.into_iter().enumerate() {
                if n > 0 {
                    rows.push((String::new(), String::new())); // spread separator
                }
                rows.extend(block);
                // LQ preview tier: fill progress + what's on screen for this page
                // (HQ full-res, the soft LQ thumbnail, or neither yet). Cache reads,
                // no I/O — which is why this row is added here and not off-thread.
                let showing = if self.reader.cache.contains(index) {
                    "HQ"
                } else if self.reader.lq_cache.contains(index) {
                    "LQ preview"
                } else {
                    "—"
                };
                rows.push((
                    "LQ tier".to_string(),
                    format!("{}/{} · {}", self.reader.lq_cache.len(), len, showing),
                ));
            }
            self.ui.info = rows;
            changed = true;
        }
        // Live folder refresh: react to images added/removed in the open folder.
        changed |= self.poll_folder_watch();
        // Auto-update: pick up the background check result and any apply result.
        if let Some(rx) = self.update_rx.take() {
            match rx.try_recv() {
                Ok(u) => {
                    self.update = Some(u);
                    changed = true; // show the "Update" button
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => self.update_rx = Some(rx),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
        if let Some(rx) = self.update_apply_rx.take() {
            match rx.try_recv() {
                Ok(Ok(())) => self.relaunch(),
                Ok(Err(e)) => {
                    self.updating = false;
                    self.update_error = Some(e);
                    changed = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => self.update_apply_rx = Some(rx),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.updating = false;
                    self.update_error = Some("update interrupted".into());
                    changed = true;
                }
            }
        }
        changed
    }

    /// Scan the current volume's folder for sibling volumes on a background
    /// thread and hand the result back for `sib_cache` (consumed in `render`).
    /// Keeps the parent-dir `read_dir` + per-entry stat off the UI thread so
    /// `[`/`]` stays responsive even on a network share.
    fn warm_sib_cache(&self, vol: &Path) {
        let tx = self.sib_tx.clone();
        let of = vol.to_path_buf();
        let wake = frame_waker(&self.window);
        std::thread::spawn(move || {
            let Some(parent) = of.parent().map(|p| p.to_path_buf()) else {
                return;
            };
            let want_folder = of.is_dir();
            let _ = tx.send((parent, want_folder, crate::library::sibling_volumes(&of)));
            wake(); // send-then-wake (a parked `[`/`]` jump replays on that frame)
        });
    }

    /// Desired decode height: the page's on-screen height (window height × zoom),
    /// quantized to avoid churn and clamped. Per-page it's further capped to the
    /// source height in `decode_and_downscale` (never upscale a page).
    /// Apply a view preset (number keys): set fit + layout (+ optional reading
    /// direction) at once, leave scroll, reset zoom/pan, re-anchor the spread
    /// pairing, persist, and refresh the prefetch window.
    fn apply_view(&mut self, fit: FitMode, spread: bool, dir: Option<Direction>) {
        self.reader.scroll_mode = false;
        self.reader.fit = fit;
        self.reader.layout = if spread { Layout::Spread } else { Layout::Single };
        if let Some(d) = dir {
            self.reader.direction = d;
        }
        self.reader.snap_to_view_start();
        self.reader.zoom = 1.0;
        self.reader.pan_x = 0.0;
        self.reader.pan_y = 0.0;
        self.settings.fit = fit_to_u8(self.reader.fit);
        self.settings.layout_spread = self.reader.layout == Layout::Spread;
        self.settings.direction_rtl = self.reader.direction == Direction::Rtl;
        self.settings.scroll = false;
        config::save(&self.settings);
        self.reader.prefetch();
        // Tell the user what the preset just switched to (presets change fit +
        // layout + maybe direction at once, so summarize the resulting view).
        let view_label = if spread {
            format!("Spread, {}", self.reader.direction.label())
        } else {
            let f = match fit {
                FitMode::Window => "Fit window",
                FitMode::Width => "Fit width",
                FitMode::Height => "Fit height",
                FitMode::Actual => "1:1",
            };
            format!("{f} (single)")
        };
        self.toast(view_label);
    }

    /// The in-view anchor page if it is an animated (GIF/WebP) page with its texture
    /// decoded — the page the mini playback controls govern.
    fn anim_anchor(&self) -> Option<usize> {
        if self.library_view {
            return None;
        }
        self.reader.source.as_ref()?;
        let anchor = self.reader.visible_pages().0;
        (self.reader.cache.get(anchor)?.frame_count() > 1).then_some(anchor)
    }

    /// Advance/refresh playback for the in-view animation and publish the panel's
    /// display state to `self.ui`. Driven every frame by the continuous render
    /// loop.
    fn update_playback(&mut self) {
        let Some(anchor) = self.anim_anchor() else {
            self.playback.page = None;
            self.ui.anim_show = false;
            return;
        };
        let frames = self.reader.cache.get(anchor).map_or(1, |t| t.frame_count());
        // GIF/WebP auto-play; `.ico` layers are stepped manually (no play/pause).
        let is_anim = self.reader.cache.get(anchor).is_some_and(|t| t.is_animation());
        // Rebind (and reset) when the viewed page changes.
        if self.playback.page != Some(anchor) {
            self.playback.page = Some(anchor);
            self.playback.frame = 0;
            self.playback.playing = is_anim;
            self.playback.last = Instant::now();
        }
        if self.playback.playing {
            let now = Instant::now();
            // Advance through as many frames as real time has passed (respecting
            // each frame's own delay), so playback tracks wall time even if we
            // briefly lagged.
            loop {
                let d = self
                    .reader
                    .cache
                    .get(anchor)
                    .map_or(100, |t| t.frame_delay_ms(self.playback.frame))
                    .max(1) as u64;
                if now.duration_since(self.playback.last) >= Duration::from_millis(d) {
                    self.playback.last += Duration::from_millis(d);
                    self.playback.frame = (self.playback.frame + 1) % frames;
                } else {
                    break;
                }
            }
        }
        self.playback.frame = self.playback.frame.min(frames - 1);
        self.ui.anim_show = !self.playback.hidden;
        self.ui.anim_is_animation = is_anim;
        self.ui.anim_playing = self.playback.playing;
        self.ui.anim_frame = self.playback.frame;
        self.ui.anim_total = frames;
    }

    fn playback_toggle(&mut self) {
        self.playback.playing = !self.playback.playing;
        if self.playback.playing {
            self.playback.last = Instant::now(); // resume from now, no time-warp jump
        }
    }

    fn playback_frame_count(&self) -> usize {
        self.playback.page.and_then(|p| self.reader.cache.get(p)).map_or(1, |t| t.frame_count())
    }

    /// Step the animation by `d` frames (pauses playback; wraps around).
    fn playback_step(&mut self, d: i32) {
        let frames = self.playback_frame_count();
        if frames <= 1 {
            return;
        }
        self.playback.playing = false;
        self.playback.frame = (self.playback.frame as i32 + d).rem_euclid(frames as i32) as usize;
    }

    /// Jump the animation to a specific frame (pauses playback).
    fn playback_seek(&mut self, frame: usize) {
        let frames = self.playback_frame_count();
        if frames <= 1 {
            return;
        }
        self.playback.playing = false;
        self.playback.frame = frame.min(frames - 1);
    }

    /// The dynamic window title: `{book} > {file} (W × H) [ pg / total ] - yosh`
    /// in the reader (resolution Firefox-tab style, shown once the page decodes),
    /// `Library - yosh` in the grid, plain `yosh` when nothing is open.
    fn title(&self) -> String {
        if self.library_view {
            return "Library - yosh".to_string();
        }
        let Some(src) = &self.reader.source else {
            return "yosh".to_string();
        };
        let len = src.len();
        if len == 0 {
            return "yosh".to_string();
        }
        // The page(s) actually shown — both halves of a two-page spread, not just
        // the anchor (issue #12).
        let (anchor, facing) = self.reader.visible_pages();
        let anchor = anchor.min(len - 1);
        let facing = facing.filter(|b| *b < len);
        // Just the basename — archive entries can carry a subfolder path.
        let base = |i: usize| {
            let name = src.name(i);
            std::path::Path::new(name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(name)
                .to_string()
        };
        // Native resolution, Firefox-tab style. Pulled from the decoded texture
        // (`src_w`/`src_h` are pre-downscale source dims), so it appears once the
        // page lands and is empty while it's still decoding.
        let res = |i: usize| match self.reader.cache.get(i) {
            Some(t) if t.src_w > 0 && t.src_h > 0 => format!(" ({} × {})", t.src_w, t.src_h),
            _ => String::new(),
        };
        let (file, pos) = match facing {
            // Reading order, so the pair reads the way it's drawn.
            Some(b) if self.reader.direction == Direction::Rtl => (
                format!("{}{} | {}{}", base(b), res(b), base(anchor), res(anchor)),
                format!("[ {}-{} / {} ] - yosh", anchor + 1, b + 1, len),
            ),
            Some(b) => (
                format!("{}{} | {}{}", base(anchor), res(anchor), base(b), res(b)),
                format!("[ {}-{} / {} ] - yosh", anchor + 1, b + 1, len),
            ),
            None => (
                format!("{}{}", base(anchor), res(anchor)),
                format!("[ {} / {} ] - yosh", anchor + 1, len),
            ),
        };
        match self.ui.opened.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()) {
            Some(book) => format!("{book} > {file} {pos}"),
            None => format!("{file} {pos}"),
        }
    }

    /// Physical pixels of surface reserved at the top for the chrome bar.
    ///
    /// The page is drawn by wgpu straight onto the surface, while the top bar is an
    /// egui `TopBottomPanel` painted over it afterwards — so unless we inset the
    /// page ourselves, an opaque bar simply covers its top (issue #10).
    ///
    /// Only the *pinned* (windowed) bar reserves space. In fullscreen the bar is a
    /// hover-reveal overlay whose height egui **animates**, and the viewport feeds
    /// `page_target_h` → `JobsKey`, so insetting by an animating height would churn
    /// the decode target every frame of the reveal. Treating it as a pure overlay
    /// keeps the height constant except across a real windowed/fullscreen change.
    fn top_inset_px(&self) -> f32 {
        if self.window.fullscreen().is_some() {
            0.0
        } else {
            self.ui.bar_px.max(0.0).min(self.gpu.config.height as f32 * 0.5)
        }
    }

    /// The reading viewport: the surface minus the space reserved for the top bar.
    fn content_viewport(&self) -> Viewport {
        Viewport {
            w: self.gpu.config.width,
            h: (self.gpu.config.height as f32 - self.top_inset_px()).max(1.0) as u32,
        }
    }

    #[allow(deprecated)]
    fn render(&mut self) {
        // Mirror the live surface size into the reading viewport (the value the
        // reading math reads instead of `gpu.config`), less the top-bar inset so
        // the chrome never covers the page.
        self.reader.viewport = self.content_viewport();
        // Sample window geometry here, where a move/resize has fully settled.
        self.record_window_geometry();
        // Advance an inertial scroll glide (if any) before drawing this frame.
        self.tick_scroll_fling();
        if let Some(p) = self.ui.pending_open.take() {
            self.open(&p);
        }

        // Drain finished decodes into the right cache (full-res → cache, LQ-tier
        // thumbnails → lq_cache) and record failures.
        self.reader.drain_pool();
        // Was the centered spinner on screen last frame? `ui.loading` isn't recomputed
        // until later this frame, so it still holds the previous frame's answer.
        let spinner_was_visible = self.ui.loading;
        // Apply finished background opens, newest generation only — a later
        // `[`/`]` bumps `open_gen`, so a stale in-flight result is discarded.
        while let Ok((generation, built)) = self.open_rx.try_recv() {
            if generation != self.open_gen {
                continue; // superseded by a newer open()
            }
            self.opening = false;
            self.opening_key = None;
            if let Some(t0) = self.open_t0
                && cfg!(debug_assertions)
            {
                eprintln!("[open] source ready in {:?}", t0.elapsed());
            }
            match built {
                Ok((source, key, start)) if !source.is_empty() => {
                    let partial = source.is_partial();
                    let n = source.len();
                    self.set_source(source, &key, start);
                    // Carry the opening spinner straight into the first-page grace
                    // instead of restarting it: `opening` has just gone false, so the
                    // per-page block below would stamp `loading_pending` fresh and blank
                    // the spinner for LOADING_INDICATOR_DELAY before bringing it back.
                    // Backdating the stamp keeps it continuously on screen until the
                    // first page actually lands.
                    if spinner_was_visible {
                        let anchor = self.reader.visible_pages().0;
                        if !self.reader.cache.contains(anchor) {
                            let now = Instant::now();
                            let backdated = now.checked_sub(LOADING_INDICATOR_DELAY).unwrap_or(now);
                            self.loading_pending = Some((anchor, backdated));
                        }
                    }
                    if partial {
                        self.toast(format!("Partial archive — {n} pages recovered"));
                    }
                }
                // A failed open never draws a first page, so drop the stamp here —
                // a stale one would mislabel a later cache-hit on the old volume.
                Ok(_) => {
                    self.ui.status = "no images found".into();
                    self.open_t0 = None;
                }
                Err(e) => {
                    self.ui.status = format!("open failed: {e}");
                    self.open_t0 = None;
                }
            }
        }
        // Background channels (sibling scans, folder watch, update check). Their
        // producers wake the loop on send, so this frame is the one that lands them.
        self.poll_background();
        // Battery-aware `Auto`: re-probe the power source now and then, and re-tier
        // if it flipped. Rendered frames are the right clock for this — the budget
        // only matters while decode work is producing frames — so it needs no timer
        // of its own; the `Instant` gate just keeps even a cheap syscall off the
        // per-frame path of a fast seek. A pinned profile never probes at all.
        if self.settings.perf == config::PerfPref::Auto
            && self.power_checked.elapsed() > POWER_RECHECK
        {
            self.power_checked = Instant::now();
            self.on_battery = on_battery().unwrap_or(false);
            if self.effective_tier() != self.applied_tier {
                self.apply_perf();
                // Auto only ever moves between High (mains) and Mid (battery), so
                // the power source names the direction.
                self.toast(if self.on_battery {
                    "Battery saver"
                } else {
                    "Performance restored"
                });
            }
        }
        self.ui.update_version = self.update.as_ref().map(|u| u.version.clone());
        self.ui.updating = self.updating;
        self.ui.update_failed = self.update_error.is_some();
        // Debounce the decode view, so resize/zoom drags re-decode once on settle.
        self.reader.update_decode_view();
        // Keep the scroll anchor valid as page heights resolve, then refresh work.
        if self.reader.scroll_mode {
            self.reader.normalize();
        }
        self.reader.prefetch();
        // Advance the in-view animation's frame and refresh its control panel.
        self.update_playback();
        // Keep the OS titlebar in sync with the open book + page (change-only).
        let title = self.title();
        if title != self.last_title {
            self.window.set_title(&title);
            self.last_title = title;
        }

        // Decide what to draw this frame (library grid hides the page).
        let quads = if self.library_view {
            Vec::new()
        } else if self.reader.scroll_mode {
            self.reader.build_scroll_quads()
        } else {
            self.reader.build_quads()
        };
        self.ui.dir_label = self.reader.direction.label();
        self.ui.fit_label = self.reader.fit.label();
        self.ui.layout_label = if self.reader.scroll_mode {
            "scroll"
        } else {
            self.reader.layout.label()
        };
        self.ui.transition_on = self.settings.page_transition_enabled;
        // Shown inverted in the panel: the row is "Stretch small pages".
        self.ui.stretch_on = !self.settings.no_stretch;
        self.ui.spine_shadow_on = self.settings.spine_shadow_enabled;
        self.ui.spine_shadow_strength = self.settings.spine_shadow_strength;
        self.ui.scroll_speed = self.settings.scroll_speed;
        self.ui.resume_on_startup = self.settings.resume_on_startup;
        // Current view state for the Settings panel's active-value highlighting.
        self.ui.scroll_on = self.reader.scroll_mode;
        self.ui.dir_rtl = self.reader.direction == Direction::Rtl;
        self.ui.layout_spread = !self.reader.scroll_mode && self.reader.layout == Layout::Spread;
        self.ui.fit_mode = fit_to_u8(self.reader.fit);
        self.ui.rotation = self.reader.rotation;
        self.ui.theme = self.settings.theme;
        self.ui.perf = self.settings.perf;
        // What `Auto` currently resolves to, so the picker shows the live answer
        // instead of leaving the user to guess.
        self.ui.perf_auto = format!(
            "Auto — currently {} ({})",
            match self.effective_tier() {
                DeviceTier::Low => "Battery saver",
                DeviceTier::Mid => "Balanced",
                DeviceTier::High => "Performance",
            },
            if self.on_battery {
                "on battery"
            } else {
                "plugged in"
            }
        );
        // Lets the chrome tell "nothing open" (→ onboarding panel) apart from the
        // library grid; `ui.opened` is sticky once set, so it can't.
        self.ui.reader_open = self.reader.source.is_some();
        self.ui.has_library_root = self.settings.library_root.is_some();
        // Build the Tab info overlay text, reading the source once per page change.
        // Keyed on the *visible* pages, not the raw index: `goto` doesn't normalize
        // to the pair anchor, so a seekbar jump could land `index` on the second
        // page and describe a different page than the title (issue #12).
        let visible = self.reader.visible_pages();
        if self.ui.info_open && !self.library_view && self.info_for != Some(visible) {
            let (a, b) = visible;
            self.info_for = Some(visible);
            match self.reader.source.clone() {
                Some(src) => {
                    // Fill the overlay *this* frame with what costs nothing — the
                    // name and page number the source already knows. Everything
                    // else (size, modified, resolution, format, color) needs the
                    // page bytes, so `page_info` gathers it off-thread and
                    // `poll_background` swaps the full rows in when they land.
                    let head = |i: usize| {
                        vec![
                            ("File".to_string(), src.name(i).to_string()),
                            ("Page".to_string(), format!("{} / {}", i + 1, src.len())),
                        ]
                    };
                    let mut rows = head(a);
                    // Both halves of a spread get described, separated by a blank row.
                    if let Some(b) = b {
                        rows.push((String::new(), String::new()));
                        rows.extend(head(b));
                    }
                    self.ui.info = rows;
                    let tx = self.info_tx.clone();
                    let generation = self.open_gen;
                    let wake = frame_waker(&self.window);
                    std::thread::spawn(move || {
                        let mut blocks = vec![(a, page_info(src.as_ref(), a))];
                        if let Some(b) = b {
                            blocks.push((b, page_info(src.as_ref(), b)));
                        }
                        let _ = tx.send((generation, visible, blocks));
                        wake(); // send-then-wake
                    });
                }
                None => self.ui.info = Vec::new(),
            }
        }
        // Live view state for the overlays: current zoom % (shown in the info
        // overlay, refreshed every frame so it tracks zooming without a rebuild)
        // and the active toast (dropped once it expires).
        self.ui.zoom_pct = self.reader.effective_zoom_pct();
        self.ui.touch_fling = self.gestures.last_touch_vy.map(|vy| {
            let (pvx, pvy) = self.reader.pan_velocity;
            (vy, self.reader.scroll_velocity.abs().max(pvx.hypot(pvy)))
        });
        self.reader.update_resize_readout();
        self.ui.resize_path = self.reader.resize_path_label();
        // Drain transient messages the reader queued (boundary hit, zoom level)
        // into the shell's timed toast.
        let last_toast = self.reader.pending_toasts.pop();
        self.reader.pending_toasts.clear();
        if let Some(m) = last_toast {
            self.toast(m);
        }
        // Persist the read position: the reader owns `index`, the shell owns the
        // volume key + settings. Cheap per-frame; flushed to disk on exit.
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), self.reader.index);
        }
        // Advance the read-tracking furthest-page mark for the library's read state.
        self.note_progress();
        if let Some((_, t)) = &self.toast
            && t.elapsed() >= TOAST_DURATION
        {
            self.toast = None;
        }
        self.ui.toast = self.toast.as_ref().map(|(m, _)| m.clone());
        // Hide the top bar in fullscreen, revealing it when the cursor is at the top edge.
        let fullscreen = self.window.fullscreen().is_some();
        let reveal = 48.0 * self.window.scale_factor() as f32;
        self.ui.show_bar = !fullscreen || (self.cursor_y as f32) < reveal;
        // Edge hover arrows: only in page-flip reader mode, below the top bar,
        // while the cursor is inside the window.
        let win_w = self.reader.viewport.w.max(1) as f32;
        let edge = win_w * EDGE_FRAC;
        let in_reader = self.reader.source.is_some() && !self.library_view && !self.reader.scroll_mode;
        let below_bar = (self.cursor_y as f32) >= reveal;
        let cx = self.cursor_x as f32;
        self.ui.hover_left = in_reader && self.cursor_in_window && below_bar && cx < edge;
        self.ui.hover_right =
            in_reader && self.cursor_in_window && below_bar && cx > win_w - edge;
        // Bottom seekbar: hidden by default, revealed when the cursor nears the
        // bottom edge (a touch taller than the floating pill so a drag stays in
        // the reveal zone). Cleared here so it vanishes when no volume is open.
        self.ui.seek_show = false;
        if let Some(src) = &self.reader.source {
            let len = src.len();
            let (anchor, facing) = self.reader.visible_pages();
            let in_cache = self.reader.cache.contains(anchor);
            // A page whose decode errored is in `failed`; treat it as not-loading so
            // we show a failure notice (file name + reason) instead of spinning.
            let fail_err: Option<String> =
                if in_cache { None } else { self.reader.failed.get(&anchor).cloned() };
            let failed = fail_err.is_some();
            let loading = !in_cache && !failed;
            if in_cache {
                self.reader.last_drawn = Some(anchor);
                if let Some(t0) = self.open_t0.take()
                    && cfg!(debug_assertions)
                {
                    eprintln!(
                        "[open] first page drawn in {:?} (page {})",
                        t0.elapsed(),
                        anchor + 1
                    );
                }
            }
            // Count from the visible anchor (and span the pair), so the status line
            // agrees with the window title instead of drifting by one in spread mode.
            let shown = match facing {
                Some(b) => format!("{}-{}", anchor + 1, b + 1),
                None => format!("{}", anchor + 1),
            };
            self.ui.status = format!(
                "{}/{}{}",
                shown,
                len,
                if failed {
                    "  [failed]"
                } else if loading {
                    "  …"
                } else {
                    ""
                },
            );
            self.ui.failed = fail_err.map(|reason| (src.name(anchor).to_string(), reason));
            // Show the centered spinner only after this page's decode has been
            // pending a beat, so fast flips don't flash it. The timer restarts
            // whenever the anchor changes, so a quick page reached at the end of
            // a slow-seek streak still gets its full grace period.
            if loading {
                let since = match self.loading_pending {
                    Some((a, t)) if a == anchor => t,
                    _ => {
                        let t = Instant::now();
                        self.loading_pending = Some((anchor, t));
                        t
                    }
                };
                self.ui.loading = since.elapsed() >= LOADING_INDICATOR_DELAY;
            } else {
                self.loading_pending = None;
                self.ui.loading = false;
            }
            self.ui.seek_index = self.reader.index;
            self.ui.seek_total = len;
            self.ui.seek_rtl = self.reader.direction == Direction::Rtl;
            self.ui.seek_style = ui::SeekbarStyle::Bar;
            self.ui.seek_buffered.clear();
            self.ui.seek_buffered.extend(self.reader.cache.buffered_indices());
            self.ui.seek_lq_buffered.clear();
            self.ui.seek_lq_buffered.extend(self.reader.lq_cache.buffered_indices());
            // Window-relative, so measure against the *surface* height, not the
            // page viewport (which is inset by the top bar) — otherwise the reveal
            // zone would sit a bar's height too high.
            let win_h = self.gpu.config.height.max(1) as f32;
            let near_bottom =
                self.cursor_in_window && (self.cursor_y as f32) > win_h - reveal * 1.5;
            self.ui.seek_show =
                self.settings.seekbar_enabled && !self.library_view && len > 1 && near_bottom;
        }
        // A background open is in flight: show the spinner over the current page
        // (or the dark fill on a first open) until the new source lands.
        if self.opening {
            self.ui.loading = true;
        }
        let anim_t = self.anim_origin.elapsed();
        let anim_page = self.playback.page;
        let anim_frame = self.playback.frame;
        let anim_playing = self.playback.playing;
        // Did this frame draw an animation that is advancing on its own? If so the
        // end-of-frame redraw decision keeps the loop running to play it.
        let mut drew_live_anim = false;
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.reader.page_texture(q.page_index).map(|t| {
                    // The animation under user control shows its selected frame;
                    // any other animated page free-runs on the wall clock; stills
                    // return their sole view. The redraw-on-demand loop keeps
                    // frames flowing while one is live (`drew_live_anim`).
                    let view = if Some(q.page_index) == anim_page {
                        drew_live_anim |= anim_playing && t.frame_count() > 1;
                        t.frame_view(anim_frame)
                    } else {
                        drew_live_anim |= t.is_animation();
                        t.view_at(anim_t)
                    };
                    self.page_pipeline.prepare_quad(
                        &self.gpu.device,
                        &self.gpu.queue,
                        q.slot,
                        t,
                        view,
                        q.scale,
                        q.offset,
                        q.rot,
                        q.alpha,
                        q.blur,
                        q.spine,
                        q.spine_strength,
                    )
                })
            })
            .collect();

        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.reconfigure();
                // On-demand loop: this frame produced nothing — retry, don't stall.
                self.window.request_redraw();
                return;
            }
            _ => {
                self.window.request_redraw();
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Page-letterbox clear, theme-aware: #202020 surround on dark, white on
        // light/e-ink. The surface is non-sRGB, so the stored byte is value*255
        // (0x20 = 32). Transparent pages composite over this via the page
        // pipeline's premultiplied-alpha blend.
        let dark = self.settings.theme.is_dark(self.system_dark);
        let bg = if dark { 32.0 / 255.0 } else { 1.0 };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("page"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg,
                            g: bg,
                            b: bg,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !page_bgs.is_empty() {
                // Draw into the region below the top bar. The quads were built in NDC
                // against `reader.viewport`, which is already the inset height, so this
                // maps them onto exactly that sub-rect — no engine-side origin needed.
                // `LoadOp::Clear` above still covers the whole surface, so the strip
                // behind the bar keeps the background color.
                let inset = self.top_inset_px();
                if inset > 0.0 {
                    pass.set_viewport(
                        0.0,
                        inset,
                        self.gpu.config.width as f32,
                        (self.gpu.config.height as f32 - inset).max(1.0),
                        0.0,
                        1.0,
                    );
                }
                pass.set_pipeline(&self.page_pipeline.pipeline);
                for bg in &page_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
        }

        // egui chrome. Covers are decoded *after* the frame (below), driven by the
        // sections the library view reports visible, so nothing is decoded here.
        self.ui.scanning = self.scanning;
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ui_state = &mut self.ui;
        let lib = &self.library;
        let library_view = self.library_view;
        let libctx = crate::library::LibCtx {
            progress: &self.settings.progress,
            last_pages: &self.settings.last_pages,
            collapsed: &self.settings.collapsed,
            current_key: self.volume_key.as_deref(),
            recents: &self.settings.recents,
        };
        // Theme the egui chrome (top bar, library grid, help/info windows) to match
        // the page letterbox. Fonts are installed once at init and untouched by this.
        self.egui_ctx.set_visuals(if dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });
        // Where a live page-flip drag's leading edge is, so the chrome can cast a
        // shadow off it. Read before the run (the closure can't borrow the reader),
        // and `None` whenever no drag is in flight — an idle frame pays one read.
        let drag_seam = if library_view {
            None
        } else {
            self.reader.drag_seam()
        };
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui::chrome(ctx, ui_state, lib, &libctx, library_view);
            // Page-flip drag drop shadow — over the page, under the chrome.
            if let Some((frac, si)) = drag_seam {
                ui::drag_shadow(ctx, frac, si);
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        // Apply toggle-button requests (take effect next frame). This frame's quads
        // were built *before* these land, so anything that fired must buy the next
        // frame itself (`ui_acted`) — the on-demand loop won't supply one otherwise.
        let mut ui_acted = false;
        if std::mem::take(&mut self.ui.req_toggle_dir) {
            self.apply_action(Action::ToggleDir);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_cycle_fit) {
            self.apply_action(Action::CycleFit);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_layout) {
            self.apply_action(Action::ToggleLayout);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_transition) {
            self.apply_action(Action::TogglePageTransition);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_stretch) {
            self.apply_action(Action::ToggleStretch);
            ui_acted = true;
        }
        // Panel-only (no key): the reader takes the combined enabled × strength.
        if std::mem::take(&mut self.ui.req_spine_toggle) {
            self.settings.spine_shadow_enabled = !self.settings.spine_shadow_enabled;
            self.reader.spine_strength = effective_spine(&self.settings);
            config::save(&self.settings);
            self.toast(if self.settings.spine_shadow_enabled {
                "Spine shadow: on"
            } else {
                "Spine shadow: off"
            });
            ui_acted = true;
        }
        if let Some(v) = self.ui.req_spine_strength.take() {
            self.settings.spine_shadow_strength = v.clamp(0.0, 1.0);
            self.reader.spine_strength = effective_spine(&self.settings);
            ui_acted = true;
        }
        // Deferred config write: once per slider release, not once per drag frame.
        if std::mem::take(&mut self.ui.req_spine_save) {
            config::save(&self.settings);
            ui_acted = true;
        }
        // Wheel scroll speed: same live-apply / deferred-save split as the spine
        // strength above (the wheel reads the setting directly, so nothing else
        // has to be told about the change).
        if let Some(v) = self.ui.req_scroll_speed.take() {
            self.settings.scroll_speed = v.clamp(0.25, 3.0);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_scroll_speed_save) {
            config::save(&self.settings);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_resume) {
            self.settings.resume_on_startup = !self.settings.resume_on_startup;
            config::save(&self.settings);
            ui_acted = true;
        }
        // Settings-panel requests (mirror Android's options popup).
        if std::mem::take(&mut self.ui.req_toggle_scroll) {
            self.apply_action(Action::ToggleScroll);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_pairing) {
            self.apply_action(Action::ToggleSpreadOffset);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_rotate) {
            self.apply_action(Action::Rotate);
            ui_acted = true;
        }
        if let Some(v) = self.ui.req_set_fit.take() {
            // Direct fit set (panel radio), mirroring Action::CycleFit's reset.
            self.reader.fit = fit_from_u8(v);
            self.reader.zoom = 1.0;
            self.reader.pan_x = 0.0;
            self.reader.pan_y = 0.0;
            self.settings.fit = v;
            config::save(&self.settings);
            ui_acted = true;
        }
        if let Some(t) = self.ui.req_set_theme.take() {
            self.settings.theme = t;
            config::save(&self.settings);
            self.window.request_redraw(); // repaint chrome + letterbox under the new theme
            ui_acted = true;
        }
        if let Some(p) = self.ui.req_set_perf.take() {
            self.settings.perf = p;
            config::save(&self.settings);
            // Re-tiers the live reader in place (no reopen); a no-op if the pick
            // resolves to the tier already running.
            self.apply_perf();
            ui_acted = true;
        }
        // Seekbar jump: re-clamp against the live source, skip a redundant goto
        // (which would needlessly reset pan when landing on the current page).
        if let Some(page) = self.ui.seek_request.take()
            && let Some(src) = &self.reader.source
        {
            let page = page.min(src.len().saturating_sub(1));
            if page != self.reader.index {
                self.reader.goto(page);
                ui_acted = true;
            }
        }
        // Animation control-panel clicks (drained after the egui frame).
        if std::mem::take(&mut self.ui.anim_req_toggle_play) {
            self.playback_toggle();
            ui_acted = true;
        }
        let step = std::mem::take(&mut self.ui.anim_req_step);
        if step != 0 {
            self.playback_step(step);
            ui_acted = true;
        }
        if let Some(f) = self.ui.anim_req_seek.take() {
            self.playback_seek(f);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.anim_req_hide) {
            self.playback.hidden = true;
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_update)
            && !self.updating
            && let Some(u) = self.update.clone()
        {
            self.updating = true;
            self.update_error = None;
            let (tx, rx) = std::sync::mpsc::channel();
            let wake = frame_waker(&self.window);
            std::thread::spawn(move || {
                let _ = tx.send(update::apply(&u));
                wake(); // send-then-wake
            });
            self.update_apply_rx = Some(rx);
            ui_acted = true;
        }
        if let Some(root) = self.ui.pending_library.take() {
            // Clear the grid to the "Scanning…" state and scan the new root off-thread.
            self.set_library(Library::empty());
            self.library_view = true;
            self.settings.library_root = Some(root.to_string_lossy().into_owned());
            config::save(&self.settings);
            self.start_scan(root);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.rescan)
            && let Some(root) = self.library.root.clone()
        {
            // Keep the current grid visible until the fresh scan lands (no flicker).
            self.start_scan(root);
            ui_acted = true;
        }
        if std::mem::take(&mut self.ui.req_toggle_library) {
            self.library_view = !self.library_view;
            // The library is the home screen — never flip to a bookless reader.
            if self.reader.source.is_none() {
                self.library_view = true;
            }
            ui_acted = true;
        }
        // Collapse/expand a series section (header click). The collapsed *set* is
        // persisted, so the default (everything expanded) stores nothing.
        if let Some(dir) = self.ui.toggle_series.take() {
            let key = dir.to_string_lossy().into_owned();
            if !self.settings.collapsed.remove(&key) {
                self.settings.collapsed.insert(key);
            }
            config::save(&self.settings);
            ui_acted = true;
        }
        // A cover click sets `pending_open` (drained next frame like the top-bar
        // open buttons), so opening a volume from the grid flows through `open`.
        let ppp = self.egui_ctx.pixels_per_point();
        let primitives = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point: ppp,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        let user_cmds = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &primitives,
            &screen,
        );
        {
            let mut egui_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut egui_pass, &primitives, &screen);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.gpu
            .queue
            .submit(user_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        // Queue/evict library covers *after* presenting, so eviction can never free
        // a texture this frame just drew. The decode itself runs off-thread; finished
        // covers are uploaded next frame in `poll_background`.
        if self.library_view {
            self.pump_covers();
        }

        // On-demand rendering: request the next frame only while something is still
        // *moving*; otherwise the loop sleeps until an event wakes it. Mirrors the
        // Android shell's redraw guard.
        //
        // Everything else that used to be listed here has a better source of frames:
        // background work (decodes, library scans, cover decodes, folder rebuilds,
        // update checks) wakes the loop from the producing thread the moment it
        // sends — one frame per landing, which is exactly what drawing it needs —
        // and the *timed* transitions (toast expiry, spinner grace, watch debounce)
        // get their one frame from the deadline scheduler in `about_to_wait`.
        // Free-running at refresh rate for any of them would only burn a full egui
        // pass + present per frame to redraw an unchanged screen.
        //
        // So what remains is animation (each frame differs from the last) plus the
        // one-shot confirmations that this frame's own drawing was already stale.
        let egui_animating = full_output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay.is_zero());
        if ui_acted                            // a chrome request landed after quads were built
            || self.opening                    // background open: spinner over the old page
            || !self.reader.view_settled       // resize/zoom debounce needs a settle frame
            // Draw-time flag, not transition_active(): a clock re-check can see
            // the animation as just-expired even though this frame drew it
            // mid-fade, freezing a half-faded ghost (decision must match draw).
            || self.reader.animation_drawn()   // a page-turn frame was drawn
            || drew_live_anim                  // a GIF/WebP is playing
            || egui_animating                  // egui-driven animation (bar reveal, spinner, …)
        {
            self.window.request_redraw();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{effective_tier, DeviceTier};
    use crate::config::PerfPref;
    use yosh_engine::reader::Budget;

    /// The load-bearing property of the whole performance setting: the shipped
    /// default (`Auto`) on a machine that isn't on battery must build **exactly**
    /// the budget the desktop had before any of this existed. `for_tier(High) ==
    /// derive` is pinned engine-side; what's pinned here is that Auto-on-mains
    /// really does resolve to `High`.
    #[test]
    fn auto_on_ac_is_the_historical_desktop_budget() {
        for mem in [64u64, 512, 8192] {
            for cpus in [2usize, 8, 32] {
                let tier = effective_tier(PerfPref::Auto, false);
                assert_eq!(tier, DeviceTier::High);
                assert_eq!(
                    Budget::for_tier(tier, mem, cpus),
                    Budget::derive(mem, cpus),
                    "Auto on AC must be bit-identical to the old derive path"
                );
            }
        }
    }

    /// Auto throttles to `Mid` on battery; a pinned profile ignores the power
    /// source entirely (so a laptop can be held at full aggression unplugged).
    #[test]
    fn auto_throttles_on_battery_and_pins_override_it() {
        assert_eq!(effective_tier(PerfPref::Auto, true), DeviceTier::Mid);
        for on_battery in [false, true] {
            assert_eq!(effective_tier(PerfPref::Low, on_battery), DeviceTier::Low);
            assert_eq!(effective_tier(PerfPref::Mid, on_battery), DeviceTier::Mid);
            assert_eq!(effective_tier(PerfPref::High, on_battery), DeviceTier::High);
        }
    }

    /// A mouse detent is a whole notch and steps the zoom ladder once; a precision
    /// touchpad's Ctrl+two-finger scroll arrives as a stream of fractional notches and
    /// must bank up to whole steps instead of stepping per event.
    #[test]
    fn wheel_notches_bank_into_whole_ladder_steps() {
        use super::bank_notches;
        // Mouse: one detent, one step, nothing left over.
        let mut bank = 0.0;
        assert_eq!(bank_notches(&mut bank, 1.0), 1);
        assert_eq!(bank, 0.0);
        assert_eq!(bank_notches(&mut bank, -1.0), -1);

        // Touchpad: ten deltas worth 0.3 notches each = 3 steps total, never more
        // than one at a time, with the remainder carried between events.
        let mut bank = 0.0;
        let mut total = 0;
        for _ in 0..10 {
            let s = bank_notches(&mut bank, 0.3);
            assert!((0..=1).contains(&s), "a 0.3-notch delta stepped {s} stops");
            total += s;
        }
        assert_eq!(total, 3, "10 x 0.3 notches = 3 whole steps");
        assert!(bank > 0.0 && bank < 1.0, "leftover fraction is kept: {bank}");

        // Reversing drops the bank, so a part-notch one way can't be spent the other.
        let mut bank = 0.0;
        assert_eq!(bank_notches(&mut bank, 0.9), 0);
        assert_eq!(bank_notches(&mut bank, -0.5), 0, "0.9 up must not complete a down step");
        assert_eq!(bank, -0.5);
    }
}

//! Android shell for yosh.
//!
//! Drives the reusable [`yosh_engine::reader::Reader`] in the winit frame loop —
//! the same poll → decode-view debounce → prefetch → build-quads → draw sequence
//! the desktop shell runs, minus the egui chrome. Renders real manga pages via
//! the engine's decode pool + `PagePipeline`.
//!
//! Storage: a tap in the top strip opens the library browser; the "Open file…"
//! button there launches the system document picker (SAF) through the
//! `YoshActivity` Java bridge, and the chosen `content://` file's bytes are read
//! off its descriptor and handed to `ZipSource::from_bytes`. With nothing open, an
//! empty-state helper explains how to open a comic.
//! Tap left third = previous page, right two-thirds = next.
//!
//! The whole crate is Android-only; the desktop shell is `crates/yosh`.
#![cfg(target_os = "android")]

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use jni::objects::{JObject, JString, JValue};
use jni::JavaVM;
use winit::application::ApplicationHandler;
use winit::event::{Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::android::activity::AndroidApp;
use winit::platform::android::EventLoopBuilderExtAndroid;
use winit::window::{Window, WindowId};

use yosh_engine::gpu::GpuContext;
use yosh_engine::layout::{view_pages, view_start, Layout};
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::{DecodePool, Waker};
use yosh_engine::reader::{
    drag_commits, drag_dir, Budget, DeviceTier, Direction, Reader, Viewport,
};
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, SevenzSource, ZipSource};
// RAR/CBR is gated behind the off-by-default `rar` feature (Linux/CI builds only —
// see Cargo.toml). The engine only exposes `RarSource` when its `rar` feature is on.
#[cfg(feature = "rar")]
use yosh_engine::source::RarSource;
use yosh_engine::texpool::TexturePool;

/// Entry point android-activity calls on the native-activity thread.
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    log::info!("yosh-android starting");
    // winit allows exactly one EventLoop per process. android-activity re-invokes
    // android_main when the activity is recreated in a still-cached process (e.g.
    // Back destroys the activity, then the user reopens) — building a second
    // EventLoop there panics. Defensively bail on re-entry so Android relaunches us
    // fresh instead. (The process::exit at the end normally prevents re-entry from
    // happening at all; this guards the timing edge where a reopen races the exit.)
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        log::warn!("android_main re-entered in a live process; exiting for a clean restart");
        std::process::exit(0);
    }
    // KEEP_SCREEN_ON is *not* set here: it's applied per frame from the reading
    // state (see the `applied_keep_on` block in `RedrawRequested`), because holding
    // the screen awake while the user is browsing the library — or staring at the
    // empty state — is just battery drain with no page to read.
    // Per-comic reading positions persist in the app's private dir.
    let pos_path = app.internal_data_path().map(|p| p.join("positions.tsv"));
    let positions = pos_path.as_deref().map(load_positions).unwrap_or_default();
    // Reading progress (furthest page seen + total) → the library's read states.
    let progress_path = app.internal_data_path().map(|p| p.join("progress.tsv"));
    let progress = progress_path.as_deref().map(load_progress).unwrap_or_default();
    // Series the user collapsed in the library (default expanded).
    let collapsed_path = app.internal_data_path().map(|p| p.join("collapsed.txt"));
    let collapsed = collapsed_path.as_deref().map(load_collapsed).unwrap_or_default();
    // Most-recently-read volume keys (MRU, newest first): resume target + shelf.
    let recents_path = app.internal_data_path().map(|p| p.join("recents.txt"));
    let recents = recents_path.as_deref().map(load_recents).unwrap_or_default();
    let lib_dir_file = app.internal_data_path().map(|p| p.join("libroot.txt"));
    let init_lib_dir = lib_dir_file
        .as_deref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("/storage/emulated/0"));
    // Persisted viewing options (direction / layout / fit); default RTL manga order.
    let view_file = app.internal_data_path().map(|p| p.join("view.txt"));
    let init_view = view_file
        .as_deref()
        .map(load_view)
        .unwrap_or((
            Direction::Rtl,
            LayoutMode::Single,
            FitMode::Window,
            true,
            false,
            true,
            ThemePref::System,
            PerfPref::Auto,
            false,
            DEFAULT_SPINE_STRENGTH,
            false,
        ));
    let event_loop = EventLoop::builder()
        .with_android_app(app.clone())
        .build()
        .expect("build event loop");
    let (open_tx, open_rx) = std::sync::mpsc::channel();
    let (sib_tx, sib_rx) = std::sync::mpsc::channel();
    let (recents_ok_tx, recents_ok_rx) = std::sync::mpsc::channel();
    let mut shell = Shell {
        app: None,
        android_app: app,
        frame_waker: None,
        picker_pending: false,
        open_gen: 0,
        opening: None,
        opening_key: None,
        open_tx,
        open_rx,
        resume_fallback: false,
        sib_cache: None,
        sib_tx,
        sib_rx,
        pending_book_nav: None,
        recents_ok: HashSet::new(),
        recents_ok_tx,
        recents_ok_rx,
        dirty_positions: false,
        dirty_progress: false,
        dirty_recents: false,
        dirty_collapsed: false,
        dirty_view: false,
        dirty_libroot: false,
        dirty_since: None,
        positions,
        pos_path,
        progress,
        progress_path,
        collapsed,
        collapsed_path,
        recents,
        recents_path,
        current_key: None,
        has_files: false,
        init_lib_dir,
        lib_dir_file,
        init_view,
        view_file,
        touches: HashMap::new(),
        gesture_start: None,
        page_drag: false,
        scroll_drag: false,
        last_fling_tick: Instant::now(),
        drag_samples: VecDeque::new(),
        pinch: None,
        applied_immersive: None,
        applied_keep_on: None,
        parked_source: None,
        status_bar_px: None,
    };
    if let Err(e) = event_loop.run_app(&mut shell) {
        log::error!("event loop exited with error: {e}");
    }
    // run_app returns only when the activity is destroyed (not on background — that
    // delivers Suspended and keeps running). Exit the process so the next launch is a
    // fresh process with a fresh EventLoop, instead of leaving a cached process that
    // would re-enter android_main and hit the one-EventLoop-per-process limit.
    log::info!("yosh-android event loop ended; exiting process");
    std::process::exit(0);
}

/// Cap on the most-recently-read list (`recents.txt`); the head is the resume target.
const RECENTS_CAP: usize = 32;

/// How long a dirty persisted-state file waits before [`Shell::flush_saves`] writes
/// it, so a burst of page turns costs one write instead of one per flip.
const SAVE_DEBOUNCE: Duration = Duration::from_secs(2);

/// Spine-shadow peak darkening when the user hasn't picked one (matches desktop).
const DEFAULT_SPINE_STRENGTH: f32 = 0.55;

/// Result of a background open: the comic's identity key (path or content:// URI)
/// and its page source, or a message to log. The key travels *with* the result
/// because the launch resume also decides which recent to open off-thread.
type OpenResult = Result<(String, Arc<dyn PageSource>), String>;

/// One off-thread info-overlay result: the open generation and the page index it
/// was gathered for, plus the `(label, value)` rows. The two tags let the main
/// thread drop a result it has already moved past — a book switch bumps the
/// generation, a page turn changes the index.
type InfoRows = (u64, usize, Vec<(String, String)>);

/// Outcome of resolving the current book's neighbour in its folder. `Cold` means
/// the listing isn't cached yet — a background scan has been kicked and the caller
/// should park its request rather than scan on the UI thread.
enum SibLookup {
    Cold,
    Missing,
    Found(PathBuf),
}

struct Shell {
    app: Option<App>,
    /// Kept for JNI into the `YoshActivity` Java bridge (vm + activity pointers).
    android_app: AndroidApp,
    /// "Something new to draw" callback for off-thread producers: it just calls
    /// `Window::request_redraw`, which is thread-safe (it flags the window and
    /// wakes the ALooper). Handed to the decode pool via `Reader::set_waker` and to
    /// the library producers, so the frame loop can genuinely idle between results
    /// instead of polling for them. Re-made per window in `resumed`.
    frame_waker: Option<Waker>,
    /// A document-picker launch is awaiting its result.
    picker_pending: bool,

    // Async open: building a page source is pure I/O — a `read_dir`, a zip central
    // directory, a whole-archive slurp off a SAF descriptor — and on phone storage
    // that is exactly the multi-second stall that turns into an ANR. It runs on a
    // worker thread; only the newest result is applied, so mashing "next book"
    // supersedes in-flight opens instead of queuing stale swaps.
    open_gen: u64,
    /// Display name of the volume currently being opened (`None` = nothing in
    /// flight). Drives the spinner card, so it must be cleared on *every* landing.
    opening: Option<String>,
    /// Identity of that volume, when it's known up front (the launch resume picks
    /// its book off-thread, so it isn't). Next/prev-book resolves against this
    /// rather than the still-open book, so a second tap during an open advances
    /// from the pending target instead of repeating it.
    opening_key: Option<String>,
    open_tx: Sender<(u64, OpenResult)>,
    open_rx: Receiver<(u64, OpenResult)>,
    /// The in-flight open is the launch resume (which also picks *which* recent to
    /// open, off-thread): a failure falls back to the library / empty state rather
    /// than leaving the spinner's aftermath on a blank screen.
    resume_fallback: bool,

    /// Cached parent-dir comic listing for next/prev-book (`(parent, comics)`),
    /// warmed in the background on open so the first boundary hit doesn't pay a
    /// `read_dir` + per-entry stat on the UI thread.
    sib_cache: Option<(PathBuf, Vec<PathBuf>)>,
    sib_tx: Sender<(PathBuf, Vec<PathBuf>)>,
    sib_rx: Receiver<(PathBuf, Vec<PathBuf>)>,
    /// A next/prev-book request that arrived while `sib_cache` was cold, held until
    /// the scan lands (then replayed). Requests accumulate, so mashing "next book"
    /// three times before the scan finishes jumps three volumes rather than one.
    pending_book_nav: Option<i64>,

    /// Which `recents` entries still exist on disk. Computed off-thread (a stat per
    /// entry against SD-card storage is not per-frame work) and refreshed when the
    /// library opens; the library render only reads it.
    recents_ok: HashSet<String>,
    recents_ok_tx: Sender<HashSet<String>>,
    recents_ok_rx: Receiver<HashSet<String>>,

    // Debounced atomic saves. Every one of these is a whole-file rewrite, and page
    // turns used to fire two of them synchronously per flip. Now a change only
    // *marks* its file dirty; `flush_saves` writes what's dirty once the marks have
    // settled (or immediately when forced: suspend, book switch), via a temp file +
    // rename so a kill mid-write can't truncate the real one.
    dirty_positions: bool,
    dirty_progress: bool,
    dirty_recents: bool,
    dirty_collapsed: bool,
    dirty_view: bool,
    dirty_libroot: bool,
    /// When the oldest un-flushed mark was made (`None` = nothing dirty).
    dirty_since: Option<Instant>,

    /// Per-comic last-read page, keyed by comic identity (content:// URI or path).
    positions: HashMap<String, usize>,
    /// Where `positions` persists (app's private dir).
    pos_path: Option<PathBuf>,
    /// Per-comic reading progress: (furthest 1-based page reached, total pages).
    /// Distinct from `positions` (the *current* page — re-reading from the start
    /// must not unmark a finished volume). Drives the library's read/unread
    /// fades and the series status labels.
    progress: HashMap<String, (u32, u32)>,
    progress_path: Option<PathBuf>,
    /// Library series the user collapsed (storing the collapsed set makes
    /// expanded the default for anything new).
    collapsed: std::collections::HashSet<PathBuf>,
    collapsed_path: Option<PathBuf>,
    /// Most-recently-read volume keys (newest first): resume-on-launch picks the
    /// first reopenable one; the library's "Recently read" row renders them.
    recents: Vec<String>,
    recents_path: Option<PathBuf>,
    /// Identity of the currently-open comic, for saving its position.
    current_key: Option<String>,
    /// Cached all-files-access state (refreshed on resume + library toggle).
    has_files: bool,
    /// The library dir to open the browser at (persisted across launches).
    init_lib_dir: PathBuf,
    lib_dir_file: Option<PathBuf>,
    /// Persisted viewing options (direction, layout, fit, page-turn animation) +
    /// where they live.
    init_view: InitView,
    view_file: Option<PathBuf>,
    /// Active touch points by finger id, for swipe / pinch-zoom / pan.
    touches: HashMap<u64, (f64, f64)>,
    /// Single-finger gesture start (for swipe-vs-tap on release).
    gesture_start: Option<(f64, f64)>,
    /// A single-finger move locked in as an interactive page drag (the page
    /// follows the finger; the engine renders it via `Reader::drag_update`).
    /// Locks once the motion is clearly horizontal; cleared on release/pinch.
    page_drag: bool,
    /// A single-finger vertical drag locked in as continuous scroll (scroll mode).
    scroll_drag: bool,
    /// Last `fling_tick` time, for the per-frame dt of the inertial scroll glide.
    last_fling_tick: Instant,
    /// Recent `(time, x|y)` samples of the dragging finger (~last 100 ms): x for the
    /// page-flip flick-to-commit, y for the scroll-release fling velocity.
    drag_samples: VecDeque<(Instant, f64)>,
    /// Active pinch, captured at its start (see `Pinch`).
    pinch: Option<Pinch>,
    /// Last immersive state pushed to Android, to avoid redundant JNI each frame.
    applied_immersive: Option<bool>,
    /// Last KEEP_SCREEN_ON state pushed to the window, same idea. `None` ⇒ not
    /// applied yet, which is also the post-resume state: the flag lives on the
    /// native window, and a resume builds a new one.
    applied_keep_on: Option<bool>,

    /// The open book's `(key, source)`, kept alive across a suspend so the resume
    /// can re-attach it instead of re-reading the archive off storage. Only the
    /// current book is parked (each `open_comic` replaces it), and it costs nothing
    /// while that book is open — the reader holds the very same `Arc`.
    ///
    /// This is what makes a resume cheap: a full-teardown resume otherwise pays the
    /// whole open again (a `read_dir`, a zip central directory, or — for a 7z/RAR —
    /// a full re-decompress). It also gets SAF `content://` books through a suspend,
    /// which nothing else can: their bytes came off a one-shot file descriptor that
    /// can't be reopened, so before this a backgrounded picked book came back as a
    /// failed open. Dropped by [`Shell::memory_warning`], since it can be several
    /// hundred MB of archive.
    parked_source: Option<(String, Arc<dyn PageSource>)>,
    /// Status-bar height in px (queried once, lazily); used to inset chrome under
    /// the bars. `None` until first queried.
    status_bar_px: Option<i32>,
}

/// How long the next/prev-book boundary prompt stays armed and on screen.
const BOOK_PROMPT_MS: u64 = 3000;

/// Which screen edge the boundary-prompt card sits on: the edge whose tap zone
/// triggers `dir` (next = left in RTL / right in LTR; prev mirrors). Shared by
/// the egui draw and `on_tap`'s fallback hit-rect so they can't disagree.
fn book_prompt_on_left(dir: i64, rtl: bool) -> bool {
    (dir > 0) == rtl
}
/// Width of the boundary prompt card, egui points.
const BOOK_PROMPT_W_PT: f32 = 260.0;

/// See [`App::book_prompt`].
struct BookPrompt {
    /// +1 = next book (armed at the last page), -1 = previous (first page).
    dir: i64,
    /// When it was armed (the confirm window + the card's visibility run from here).
    at: Instant,
    /// The resolved sibling comic, if any (None at the ends / for SAF sources).
    sibling: Option<PathBuf>,
    /// The sibling's display name (empty when `sibling` is None).
    title: String,
}

/// Active two-finger pinch, captured at start so each move can both scale
/// about — and pan with — the finger midpoint (zoom-to-focal-point).
#[derive(Clone, Copy)]
struct Pinch {
    dist0: f64,       // finger separation when the pinch began
    zoom0: f32,       // reader.zoom when it began
    pan0: (f32, f32), // reader.pan_x / pan_y when it began
    mid0: (f64, f64), // finger midpoint (screen px) when it began
}

/// The live app: surface + the engine device context, the page pipeline, and the
/// reader that owns the decode pool / cache / nav state.
struct App {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    page_pipeline: PagePipeline,
    reader: Reader,
    anim_origin: Instant,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// Library browser overlay state.
    library_view: bool,
    /// The configured library root (persisted); the browser can't go above it.
    lib_root: PathBuf,
    lib_dir: PathBuf,
    lib_entries: Vec<Entry>,
    /// Decoded cover thumbnails by comic path (egui textures), filled off-thread.
    thumbs: HashMap<PathBuf, egui::TextureHandle>,
    /// One message per *requested* cover — `None` when the decode failed, so the
    /// in-flight count below can't leak on an unreadable archive.
    thumb_rx: std::sync::mpsc::Receiver<(PathBuf, Option<egui::ColorImage>)>,
    /// Queue into the persistent cover-decode worker (see [`spawn_cover_worker`]):
    /// batches of comic paths in, decoded covers back out through `thumb_rx`.
    cover_tx: std::sync::mpsc::Sender<Vec<PathBuf>>,
    /// Covers already queued for decode (lazy, per visible series section).
    queued_covers: std::collections::HashSet<PathBuf>,
    /// Covers sent to the worker whose result hasn't been drained yet. Unlike
    /// `queued_covers` (which remembers every request, so a decoded — or
    /// undecodable — cover is never re-queued) this drops back to zero, which is
    /// what the redraw guard needs: "results are still coming", not "results ever
    /// came".
    covers_inflight: usize,
    /// The shell's frame waker (see [`Shell::frame_waker`]), carried here so the
    /// App's own off-thread producers — the series scan — can wake the loop when
    /// their results land.
    frame_waker: Option<Waker>,
    /// Frame stamp a cover was last drawn (LRU for the `thumbs` cap — a big
    /// library would otherwise pin hundreds of MB of egui textures).
    thumb_used: HashMap<PathBuf, u64>,
    /// Monotonic render counter for `thumb_used`.
    frame_no: u64,
    /// The yosh mascot logo (embedded PNG), shown on the "no comic open" card.
    /// Decoded lazily on first paint of the empty state.
    logo: Option<egui::TextureHandle>,
    /// The library's series sections (scanned off-thread; see `spawn_library_scan`).
    series: Vec<Series>,
    series_rx: std::sync::mpsc::Receiver<Vec<Series>>,
    series_tx: std::sync::mpsc::Sender<Vec<Series>>,
    /// A series scan is in flight ("Scanning…" placeholder).
    series_pending: bool,
    /// Library sub-mode: the old folder grid, kept solely for picking a new
    /// library root ("Change library…"); false = the series view.
    lib_browse: bool,
    /// Reading chrome: seekbar shows when `controls`; the gear opens the
    /// viewing-options popup. The zone hints show only briefly after the controls
    /// are revealed (see `controls_shown_at`), so they don't clutter while reading.
    controls: bool,
    controls_shown_at: Instant,
    /// Display name of the open comic (basename of `Shell::current_key`), shown
    /// atop the seekbar. Empty when nothing is open.
    book_title: String,
    /// Armed boundary prompt: a next/prev action at the last/first page showed
    /// a "Next/Previous book" card (cover + title once the cover decode lands);
    /// repeating the action — or tapping the card — within [`BOOK_PROMPT_MS`]
    /// opens the sibling. `sibling: None` ⇒ no neighbor (or a SAF-picked book,
    /// which can't resolve folder siblings) — the card just names the boundary.
    book_prompt: Option<BookPrompt>,
    show_options: bool,
    show_info: bool,
    /// Page index the heavy info-overlay metadata (Format/Color/Size/Modified) was
    /// last built for, so the page bytes are re-read only on a page change while the
    /// info popup is open — not every frame. `None` ⇒ rebuild on next render.
    info_for: Option<usize>,
    /// Cached (label, value) metadata lines for `info_for`'s page. Spliced into
    /// `build_info`'s per-frame lines so the byte read stays gated. Placeholders
    /// until the background read lands (see `info_rx`).
    info_meta: Vec<(String, String)>,
    /// The info overlay's metadata needs `read_page` + `modified` — disk I/O, and on
    /// a 7z still decompressing, a wait for that entry to land. It's gathered on a
    /// thread and tagged with `(open_gen, page index)` so a result for a book that
    /// has since been switched, or a page already turned past, is dropped.
    info_tx: Sender<InfoRows>,
    info_rx: Receiver<InfoRows>,
    /// Off-thread listing for the "choose a library root" folder browser, tagged
    /// with the directory it describes (a newer navigation supersedes it).
    browse_tx: Sender<(PathBuf, Vec<Entry>)>,
    browse_rx: Receiver<(PathBuf, Vec<Entry>)>,
    /// A browse listing is in flight ("Scanning…" placeholder).
    browse_pending: bool,
    /// The egui frame changed the viewing options / library root; the Shell picks
    /// these up after `render` and debounces the actual file writes.
    view_dirty: bool,
    libroot_dirty: bool,
    /// The user's layout choice. `reader.layout` is the concrete `Single`/`Spread`
    /// this resolves to (orientation-dependent for `Auto`); this is the source of
    /// truth that persists. See `apply_resolved_layout`.
    layout_mode: LayoutMode,
    /// Reopen the last book on launch (persisted in view.txt; toggled in options).
    resume_on_startup: bool,
    /// Book-gutter shading on un-joined two-page spreads, and its peak darkening
    /// (persisted in view.txt; toggled in options). The reader takes the product.
    spine_shadow_on: bool,
    spine_shadow_strength: f32,
    /// Chrome theme preference (persisted in view.txt; toggled in options).
    theme: ThemePref,
    /// Performance profile (persisted in view.txt; toggled in options). Applied
    /// live by `apply_perf` — no book reopen.
    perf: PerfPref,
    /// The tier the hardware probe chose, used whenever `perf` is `Auto`. Probed
    /// once per GPU build (a resume re-probes, which is free and correct).
    auto_tier: DeviceTier,
    /// The `Budget::for_tier` inputs, kept so the performance setting can recompute
    /// a budget at runtime without re-reading `/proc`.
    mem_budget_mb: u64,
    cpus: usize,
    /// Cached OS night-mode flag (re-read on resume); resolves `ThemePref::System`.
    system_dark: bool,
    /// Light/dark actually pushed into the egui context (`ctx.set_visuals` is
    /// persistent state, so re-applying it every frame just rebuilds a `Visuals`
    /// for nothing). `None` ⇒ not applied yet. Mirrors `Shell::applied_immersive`.
    applied_light: Option<bool>,
}

/// A library browser row: a folder to descend into (or open if it holds images),
/// or a comic archive to open.
enum Entry {
    Dir(PathBuf),
    Comic(PathBuf),
}

/// One library series: a folder that directly holds volumes (comic archives
/// and/or image-folder comics). Built by [`spawn_library_scan`].
struct Series {
    dir: PathBuf,
    name: String,
    volumes: Vec<PathBuf>,
}

/// A volume's read state, derived from the shell's progress/positions maps.
#[derive(Clone, Copy, PartialEq)]
enum VolState {
    Unread,
    /// Started but not finished; the fraction read (0..1) drives the progress bar.
    InProgress(f32),
    Finished,
}

/// Derive a volume's read state. `Finished` once the furthest page seen reached
/// the total; a legacy `positions` entry without progress data counts as
/// started (pre-tracking books can't claim a furthest page).
fn vol_state(
    progress: &HashMap<String, (u32, u32)>,
    positions: &HashMap<String, usize>,
    key: &str,
) -> VolState {
    match progress.get(key) {
        Some(&(furthest, total)) if total > 0 && furthest >= total => VolState::Finished,
        Some(&(furthest, total)) => {
            VolState::InProgress(furthest as f32 / total.max(1) as f32)
        }
        None if positions.contains_key(key) => VolState::InProgress(0.0),
        None => VolState::Unread,
    }
}

/// The series header's right-side status label.
fn series_status(states: &[VolState]) -> String {
    let unread = states.iter().filter(|s| **s == VolState::Unread).count();
    let reading = states.iter().any(|s| matches!(s, VolState::InProgress(_)));
    if reading {
        if unread > 0 {
            format!("Reading · {unread} unread")
        } else {
            "Reading".to_string()
        }
    } else if unread > 0 {
        format!("{unread} unread")
    } else {
        "Finished".to_string()
    }
}

/// Max folder depth `spawn_library_scan` descends looking for series.
const SERIES_MAX_DEPTH: usize = 5;

/// Walk `root` off-thread and group its comics into [`Series`] — every folder
/// that directly holds at least one volume. Sent back whole (drained in render
/// like the cover decodes), so a deep tree or slow storage never hitches the UI.
fn spawn_library_scan(root: PathBuf, tx: std::sync::mpsc::Sender<Vec<Series>>, waker: Option<Waker>) {
    std::thread::spawn(move || {
        let mut out = Vec::new();
        if walk_series(&root, 0, &mut out) {
            // The root itself is an image-folder comic: a one-volume "series".
            out.push(Series {
                name: name_of(&root),
                volumes: vec![root.clone()],
                dir: root,
            });
        }
        // Path order keeps nested series next to their parents, naturally sorted.
        out.sort_by(|a, b| {
            natord::compare(
                &a.dir.to_string_lossy().to_lowercase(),
                &b.dir.to_string_lossy().to_lowercase(),
            )
        });
        // Send, then wake: the loop is idle while the scan runs, so the result only
        // reaches the screen if this schedules the frame that drains `series_rx`.
        let _ = tx.send(out);
        if let Some(w) = &waker {
            w();
        }
    });
}

/// One `read_dir` per folder: image files make the folder itself a volume
/// (returns true; the caller adds it and does not descend), archives become
/// volumes, and remaining sub-folders recurse as potential series.
fn walk_series(dir: &Path, depth: usize, out: &mut Vec<Series>) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    let mut volumes: Vec<PathBuf> = Vec::new();
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if is_comic_archive(&p) {
            volumes.push(p);
        } else if is_image_ext(&p) {
            return true; // an image-folder comic — the caller's volume
        }
    }
    for d in subdirs {
        if depth < SERIES_MAX_DEPTH && walk_series(&d, depth + 1, out) {
            volumes.push(d);
        }
    }
    if !volumes.is_empty() {
        volumes.sort_by(|a, b| natord::compare(&name_of(a).to_lowercase(), &name_of(b).to_lowercase()));
        out.push(Series {
            name: name_of(dir),
            dir: dir.to_path_buf(),
            volumes,
        });
    }
    false
}

/// Shell-owned state the library view reads each frame (the App is rebuilt on
/// every resume, so durable state lives on the Shell and is lent to `render`).
struct LibCtx<'a> {
    progress: &'a HashMap<String, (u32, u32)>,
    positions: &'a HashMap<String, usize>,
    collapsed: &'a std::collections::HashSet<PathBuf>,
    current_key: Option<&'a str>,
    /// Most-recently-read keys (newest first) for the "Recently read" row.
    recents: &'a [String],
    /// Which of those still exist on disk (see [`Shell::recents_ok`]) — the render
    /// filters against this set instead of stat-ing every recent, every frame.
    recents_ok: &'a HashSet<String>,
}

/// Per-frame owned snapshot of one series section for the egui closure.
struct SectionRow {
    dir: PathBuf,
    name: String,
    expanded: bool,
    status: String,
    /// Empty when collapsed (no need to clone what won't draw).
    volumes: Vec<VolCell>,
}

/// Per-frame owned snapshot of one volume cell.
struct VolCell {
    path: PathBuf,
    label: String,
    thumb: Option<egui::TextureHandle>,
    state: VolState,
    is_current: bool,
}

/// The user's page-layout *choice*. The engine's `Layout` is binary
/// (`Single`/`Spread`); this adds `Auto`, which the shell resolves to a concrete
/// `Layout` from the live viewport each frame: portrait → single, landscape →
/// two-page spread. So physically rotating the tablet switches modes — single while
/// upright, spread when turned sideways for a double-page.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    Single,
    Spread,
    Auto,
}

impl LayoutMode {
    /// Resolve to a concrete engine `Layout` for a viewport of `w × h` px.
    /// Landscape is strictly `w > h`; portrait and square fall to single.
    fn resolve(self, w: u32, h: u32) -> Layout {
        match self {
            LayoutMode::Single => Layout::Single,
            LayoutMode::Spread => Layout::Spread,
            LayoutMode::Auto => {
                if w > h {
                    Layout::Spread
                } else {
                    Layout::Single
                }
            }
        }
    }

    /// Persistence / info-overlay token.
    fn label(self) -> &'static str {
        match self {
            LayoutMode::Single => "single",
            LayoutMode::Spread => "spread",
            LayoutMode::Auto => "auto",
        }
    }
}

/// Chrome theme preference. `System` follows the OS day/night setting (read via
/// JNI on Android — winit doesn't surface it); `Light`/`Dark` force it. Light is
/// the e-ink-friendly mode (white background, dark text, opaque popups); the dark
/// default is unusable on a reflective e-ink panel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThemePref {
    System,
    Light,
    Dark,
}

impl ThemePref {
    /// Resolve to dark-vs-light, consulting the cached system night-mode flag for `System`.
    fn is_dark(self, system_dark: bool) -> bool {
        match self {
            ThemePref::System => system_dark,
            ThemePref::Light => false,
            ThemePref::Dark => true,
        }
    }
    /// Persistence token (view.txt slot 7).
    fn label(self) -> &'static str {
        match self {
            ThemePref::System => "system",
            ThemePref::Light => "light",
            ThemePref::Dark => "dark",
        }
    }
    fn parse(tok: Option<&&str>) -> Self {
        match tok {
            Some(&"light") => ThemePref::Light,
            Some(&"dark") => ThemePref::Dark,
            _ => ThemePref::System,
        }
    }
}

/// Performance profile: how hard the reader is allowed to work this device.
/// A single picker rather than individual knobs — the `Budget` fields are
/// interdependent (a wider prefetch window with a small cache just evicts itself),
/// so exposing them separately would only let a user build a worse configuration.
/// `Auto` (the default) uses the probed [`DeviceTier`]; the rest pin a tier, which
/// is how a flagship keeps today's aggressive behavior (`Performance` = the tier
/// that applies no ceilings) and a device the heuristic misjudged can be corrected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PerfPref {
    Auto,
    Low,
    Mid,
    High,
}

impl PerfPref {
    /// The tier this pins, or `None` for `Auto` (use the probed one).
    fn tier(self) -> Option<DeviceTier> {
        match self {
            PerfPref::Auto => None,
            PerfPref::Low => Some(DeviceTier::Low),
            PerfPref::Mid => Some(DeviceTier::Mid),
            PerfPref::High => Some(DeviceTier::High),
        }
    }
    /// Persistence token (view.txt slot 8).
    fn label(self) -> &'static str {
        match self {
            PerfPref::Auto => "auto",
            PerfPref::Low => "low",
            PerfPref::Mid => "mid",
            PerfPref::High => "high",
        }
    }
    fn parse(tok: Option<&&str>) -> Self {
        match tok {
            Some(&"low") => PerfPref::Low,
            Some(&"mid") => PerfPref::Mid,
            Some(&"high") => PerfPref::High,
            _ => PerfPref::Auto,
        }
    }
}

/// Cross-shell actions an egui frame requests (handled after the egui run, since
/// they need `Shell` state the reader/render path doesn't own).
#[derive(Default)]
struct FrameReqs {
    /// Open this comic (by path).
    open: Option<PathBuf>,
    /// Request all-files access.
    grant: bool,
    /// Open the SAF single-file picker (fallback for files outside the library).
    open_picker: bool,
    /// Open the library browser (from the empty-state helper).
    open_library: bool,
    /// Toggle a library series' collapsed state (persisted on the Shell).
    toggle_series: Option<PathBuf>,
    /// Seekbar page button: reading-order step (-1 prev / +1 next).
    page_nav: i64,
    /// Seekbar book button: open the prev/next sibling comic in the folder (-1/+1).
    book_nav: i64,
    /// Seekbar bookshelf button: return to the library, leaving the book warm.
    to_library: bool,
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
        );
        self.has_files = has_all_files(&self.android_app);
        // Anything still dirty belongs to the App that's about to be torn down (the
        // view options are read back off it), so write it out before `init_view` is
        // re-read below — otherwise a pending options change would be reverted by
        // its own file.
        self.flush_saves(true);
        // Every options toggle writes view.txt immediately, but `init_view` is the
        // snapshot this process *launched* with — and a resume rebuilds the whole
        // App from it. Re-read the file so options changed since launch (theme,
        // fit, performance profile, …) survive a background/foreground cycle
        // instead of silently reverting.
        if let Some(f) = &self.view_file {
            self.init_view = load_view(f);
        }
        // On-demand rendering: the loop sleeps until an event arrives, and every
        // off-thread producer wakes it through `frame_waker` when it has something
        // to show. `Wait` is winit's default, but the whole redraw guard below is
        // built on it, so pin it explicitly rather than inherit it.
        event_loop.set_control_flow(ControlFlow::Wait);
        // A fresh window means a fresh waker (the old one holds the old window and
        // dies with the App it belonged to).
        self.frame_waker = Some({
            let w = window.clone();
            Arc::new(move || w.request_redraw())
        });
        // KEEP_SCREEN_ON is a flag on the *native window*, which this resume
        // replaces — so forget what we last pushed and let the first frame re-derive
        // and re-apply it.
        self.applied_keep_on = None;
        log::info!(
            "all-files access: {} | scale {}",
            self.has_files,
            window.scale_factor()
        );
        // What to restore after a resume-rebuild (all None/false on a true first launch).
        let mut rebuild_book: Option<String> = None;
        let mut resume_active = false; // this resumed() is a forced rebuild (app was live)
        let mut resume_library = false; // the library overlay was up when we rebuilt
        // Resume from background: unconditionally rebuild the GPU. Reusing the cached
        // adapter against the post-background surface works on most devices, but some
        // e-ink GPUs reject it ("Surface does not support the adapter's queue family") via
        // an async validation error that no synchronous check (is_surface_supported, an
        // error scope / uncaptured-error handler, get_current_texture's status, or a poll)
        // catches before it lands — so tear the context down and rebuild from scratch (a
        // fresh adapter is always compatible with its fresh surface, like cold start).
        // Resume isn't the zero-hitch reading hot-path, so re-decoding the current page
        // from its saved position is an acceptable cost for never crashing.
        if self.app.is_some() {
            resume_active = true;
            resume_library = self.app.as_ref().unwrap().library_view;
            // Save the open book's page and clear current_key so the reopen below restores
            // that saved page (not the fresh reader's index 0).
            if let (Some(k), Some(app)) = (self.current_key.take(), self.app.as_ref()) {
                self.positions.insert(k.clone(), app.reader.index);
                self.mark_positions();
                self.flush_saves(true); // the App holding the view state is about to go
                rebuild_book = Some(k);
            }
            self.app = None; // drops the old GPU context, Reader, decode pool + textures
            // fall through to the full build below.
        }

        // First launch (or a forced rebuild): full build.
        let instance = GpuContext::create_instance();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        // A phone has exactly one GPU, so the preference picks no different
        // adapter — but `LowPower` is the honest hint to the driver's power
        // governor for a workload that draws a couple of quads per frame.
        let ctx = GpuContext::create(instance, Some(&surface), wgpu::PowerPreference::LowPower);
        // Make wgpu errors non-fatal: the default handler panics, which is how a stray
        // validation error (notably surface.configure rejecting a post-background surface
        // on some e-ink GPUs) takes the app down. Log instead; the resume path detects the
        // bad surface via get_current_texture's status and rebuilds the GPU.
        ctx.device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
            log::error!("wgpu error (non-fatal): {e}");
        }));
        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: GpuContext::choose_surface_format(&caps),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&ctx.device, &config);

        let page_pipeline = PagePipeline::new(&ctx.device, config.format);
        let egui_ctx = egui::Context::default();
        // One font set: Phosphor icon glyphs (seekbar buttons) plus a CJK fallback.
        // Phosphor is installed unconditionally; egui's bundled fonts have no CJK
        // glyphs, so add the system Noto Sans CJK as a fallback when present so
        // Japanese comic / file names render instead of tofu squares.
        let mut fonts = egui::FontDefinitions::default();
        egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Fill);
        if let Some(cjk) = cjk_font() {
            fonts.font_data.insert("cjk".to_owned(), cjk);
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(fam).or_default().push("cjk".to_owned());
            }
        }
        egui_ctx.set_fonts(fonts);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            Some(8192),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &ctx.device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let (mem_budget_mb, total_mb) = device_mem();
        let max_khz = max_cpu_khz();
        let auto_tier = device_tier(total_mb, max_khz);
        let perf = self.init_view.7;
        let tier = perf.tier().unwrap_or(auto_tier);
        let budget = Budget::for_tier(tier, mem_budget_mb, cpus);
        // One line with everything the tier decision rests on: a field report of a
        // device behaving badly should be diagnosable from logcat alone.
        log::info!(
            "device: {total_mb} MB RAM, {cpus} cpus, max {} MHz, gpu \"{}\" → auto tier {auto_tier:?}, perf {} → {tier:?}",
            max_khz.map_or(0, |k| k / 1000),
            ctx.adapter_info.name,
            perf.label(),
        );
        log::info!("budget: {budget:?} (mem slice {mem_budget_mb} MB)");
        let tex_pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));
        let mut reader = Reader::new(
            ctx.device.clone(),
            ctx.queue.clone(),
            tex_pool,
            budget,
            self.init_view.2, // fit
            // Concrete layout, resolved from the saved mode against the initial
            // window size (so an `Auto` launch already matches the orientation).
            self.init_view.1.resolve(config.width, config.height),
            self.init_view.4, // scroll mode (persisted)
            self.init_view.0, // direction (default RTL)
            0,
            true, // two_tier: LQ while seeking → HQ on settle
        );
        reader.transition_enabled = self.init_view.3; // page-turn animation (persisted; default on)
        // No-upscale fit (persisted; default off = stretch small pages, as before).
        reader.fit_no_upscale = self.init_view.10;
        // Spine shadow: the shell owns on/off × strength, the reader takes one number.
        reader.spine_strength = if self.init_view.8 { self.init_view.9 } else { 0.0 };
        // Landing decodes schedule their own frame from the worker thread; every
        // pool this reader builds inherits the callback.
        reader.set_waker(self.frame_waker.clone());
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (series_tx, series_rx) = std::sync::mpsc::channel();
        let (info_tx, info_rx) = std::sync::mpsc::channel();
        let (browse_tx, browse_rx) = std::sync::mpsc::channel();
        // On-disk cover-thumbnail cache dir (app-private `…/thumbs`): covers load
        // from here instead of re-decoding the full first page on every open. It's
        // owned by the cover worker, the only thing that touches it.
        let thumb_cache_dir = self.android_app.internal_data_path().map(|p| p.join("thumbs"));
        let cover_tx = spawn_cover_worker(thumb_cache_dir, thumb_tx, self.frame_waker.clone());
        self.app = Some(App {
            window: window.clone(),
            ctx,
            surface,
            config,
            page_pipeline,
            reader,
            anim_origin: Instant::now(),
            egui_ctx,
            egui_state,
            egui_renderer,
            library_view: false,
            lib_root: self.init_lib_dir.clone(),
            lib_dir: self.init_lib_dir.clone(),
            lib_entries: Vec::new(),
            thumbs: HashMap::new(),
            thumb_rx,
            cover_tx,
            queued_covers: std::collections::HashSet::new(),
            covers_inflight: 0,
            frame_waker: self.frame_waker.clone(),
            thumb_used: HashMap::new(),
            frame_no: 0,
            logo: None,
            series: Vec::new(),
            series_rx,
            series_tx,
            series_pending: false,
            lib_browse: false,
            controls: true,
            controls_shown_at: Instant::now(),
            book_title: String::new(),
            book_prompt: None,
            show_options: false,
            show_info: false,
            info_for: None,
            info_meta: Vec::new(),
            info_tx,
            info_rx,
            browse_tx,
            browse_rx,
            browse_pending: false,
            view_dirty: false,
            libroot_dirty: false,
            layout_mode: self.init_view.1,
            resume_on_startup: self.init_view.5,
            spine_shadow_on: self.init_view.8,
            spine_shadow_strength: self.init_view.9,
            theme: self.init_view.6,
            perf,
            auto_tier,
            mem_budget_mb,
            cpus,
            system_dark: system_dark(&self.android_app),
            applied_light: None,
        });
        // Pick up where we left off: with resume on, reopen the most-recent
        // reopenable book (at its saved page); else, if a library is configured, open
        // it as the home; else fall through to the empty-state card (onboarding). SAF
        // content:// one-offs can't be silently rebuilt, so resume skips them.
        let lib_configured = self.init_lib_dir != PathBuf::from("/storage/emulated/0");
        // Reopen after the (re)build. Priority: a SAF picker return opens the picked file;
        // a resume-rebuild restores what was showing (the open book at its page, else the
        // library if it was up / is the home, else the empty-state card); a true first
        // launch resumes the most-recent book, else the configured library, else the card.
        if self.picker_pending
            && let Some(uri) = take_picked_uri(&self.android_app)
        {
            self.picker_pending = false;
            self.open_picked(&uri);
        } else if let Some(key) = rebuild_book {
            // The book that was open is almost always still *parked* — the source
            // outlives the App teardown above — so re-attach it directly: zero I/O,
            // no spinner, the page is back on screen on the first frame. This is the
            // whole point of `parked_source`, and it's the only way a SAF
            // `content://` book can survive a suspend at all (its descriptor is
            // long gone; only these bytes remain).
            match self.parked_source.take() {
                Some((k, src)) if k == key => self.open_comic(key, src),
                parked => {
                    // Something else was parked (or nothing was) — put it back and
                    // rebuild from storage. Reopening is asynchronous, so a resume
                    // still never blocks on re-reading an archive; it just draws the
                    // spinner for the first frames. A book that vanished (or a
                    // content:// one-off, which can't be rebuilt) fails the open and
                    // lands on the library, exactly as the old `is_reopenable_fs`
                    // filter did — see `resume_fallback`.
                    self.parked_source = parked;
                    self.open_path(PathBuf::from(key));
                    self.resume_fallback = resume_library || lib_configured; // after: start_open clears it
                }
            }
        } else if resume_active {
            if resume_library || lib_configured {
                self.open_library();
            }
        } else if self.init_view.5 && !self.recents.is_empty() {
            self.resume_recent();
            self.resume_fallback = lib_configured; // after: start_open clears it
        } else if lib_configured {
            self.open_library();
        }
        window.request_redraw();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Persist the current position + library dir before the OS may kill us.
        // Everything is written **synchronously** here: the process can be killed
        // the moment this returns, so the debounce is force-flushed rather than
        // trusted to a later frame that may never come.
        if let (Some(k), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(k, app.reader.index);
        }
        self.dirty_positions = true;
        self.dirty_progress = true;
        self.dirty_recents = true;
        self.dirty_libroot = true;
        self.dirty_since.get_or_insert_with(Instant::now);
        self.flush_saves(true);
        // Stop decoding. Android may leave a backgrounded app alive for *hours*
        // before it needs the memory, and an unparked pool would spend all of it
        // grinding the whole-volume thumbnail tail at 8 threads — burning battery
        // into textures no one can see. `park` also makes the work already in
        // flight abandon at its next yield point, so this takes effect in
        // milliseconds rather than after the current decode.
        //
        // (Today's `resumed` drops the whole App anyway, so nothing ever calls
        // `unpark` — the park matters for the window *between* the two, which is
        // the part that can last hours.)
        if let Some(app) = self.app.as_mut() {
            if let Some(pool) = &app.reader.pool {
                pool.park();
            }
            // The LQ preview tier is the single biggest thing we hold (up to
            // `lq_cap` GPU textures for the whole volume) and the most rebuildable:
            // shed it so the OS has less reason to kill us while we're away.
            app.reader.lq_cache.clear();
            // Both of the above invalidate the prefetch memo — the next foreground
            // frame must rebuild the job list and the thumbnail tail from scratch.
            app.reader.invalidate_jobs();
        }
    }

    /// The OS is under memory pressure and we are a candidate for death (Android
    /// `onLowMemory`). Everything shed here is *derived* state that costs time, not
    /// correctness, to rebuild — no reading state, no unsaved position.
    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        // The parked source is the biggest single thing we hold (a whole in-memory
        // archive for a SAF/7z/RAR book) and the most optional — losing it costs the
        // *next* resume a re-open, nothing else.
        let dropped_park = self.parked_source.take().is_some();
        let mut shed_lq = 0;
        if let Some(app) = self.app.as_mut() {
            // The LQ preview tier: up to `lq_cap` whole-volume thumbnails, all of
            // which the tail refills on demand. The live `cache` deliberately stays
            // — evicting it would blank the page the user is looking at.
            shed_lq = app.reader.lq_cache.len();
            app.reader.lq_cache.clear();
            // After the cache clear, not before: `PageCache::clear` recycles its
            // textures *into* the pool, so clearing the pool first would just refill
            // it with what we are trying to give back.
            app.reader.tex_pool.clear();
            // The thumbnails just went away; the next frame's prefetch has to
            // reconsider them from scratch rather than trust its memo.
            app.reader.invalidate_jobs();
            app.window.request_redraw();
        }
        log::warn!(
            "memory warning: shed {shed_lq} LQ previews + the texture pool{}",
            if dropped_park { " + the parked source" } else { "" },
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui see the event first (seekbar drag); gate nav on what it consumes,
        // and wake the loop if egui wants to repaint (e.g. a slider being dragged).
        // RedrawRequested itself is exempt: egui-winit reports `repaint: true` for it
        // ("paint now", not "schedule another frame"), so honoring it would re-arm a
        // redraw from within every frame — a feedback loop that silently defeats the
        // on-demand guard at the end of `render` and burns battery rendering a static
        // page at display refresh rate. (Same bug + fix as the desktop shell.)
        let egui_consumed = if let Some(app) = self.app.as_mut() {
            let resp = app.egui_state.on_window_event(&app.window, &event);
            if resp.repaint && !matches!(event, WindowEvent::RedrawRequested) {
                app.window.request_redraw();
            }
            resp.consumed
        } else {
            false
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(app) = self.app.as_mut() {
                    app.config.width = size.width.max(1);
                    app.config.height = size.height.max(1);
                    app.surface.configure(&app.ctx.device, &app.config);
                    app.window.request_redraw();
                }
            }
            WindowEvent::Touch(Touch {
                phase,
                location,
                id,
                ..
            }) => self.handle_touch(phase, id, location.x, location.y, egui_consumed),
            WindowEvent::RedrawRequested => {
                // A picker result may have landed while we were backgrounded.
                if self.picker_pending {
                    if let Some(uri) = take_picked_uri(&self.android_app) {
                        self.picker_pending = false;
                        self.open_picked(&uri);
                    }
                }
                // The single drain point for everything the shell's background
                // threads produce (opens, sibling listings, recents existence).
                self.poll_background();
                // Advance an inertial scroll glide (if any) before drawing this frame.
                self.tick_scroll_fling();
                let has_files = self.has_files;
                // Status-bar height (queried once; constant). Chrome is inset by it
                // while the bars are shown so they don't cover the top buttons.
                let status_px = *self
                    .status_bar_px
                    .get_or_insert_with(|| status_bar_height(&self.android_app));
                let opening = self.opening.clone();
                let open_gen = self.open_gen;
                let reqs = match &mut self.app {
                    Some(app) => app.render(
                        has_files,
                        status_px,
                        opening.as_deref(),
                        open_gen,
                        LibCtx {
                            progress: &self.progress,
                            positions: &self.positions,
                            collapsed: &self.collapsed,
                            current_key: self.current_key.as_deref(),
                            recents: &self.recents,
                            recents_ok: &self.recents_ok,
                        },
                    ),
                    None => FrameReqs::default(),
                };
                // Keep the read-tracking map current (cheap; persisted with the
                // position saves) — covers seekbar jumps applied inside render.
                self.note_progress();
                // The egui frame may have changed the viewing options / library root;
                // both live on the App, so it flags them and the debounce is the
                // Shell's business.
                let (view_dirty, libroot_dirty) = match self.app.as_mut() {
                    Some(app) => (
                        std::mem::take(&mut app.view_dirty),
                        std::mem::take(&mut app.libroot_dirty),
                    ),
                    None => (false, false),
                };
                if view_dirty {
                    self.dirty_view = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                }
                if libroot_dirty {
                    self.dirty_libroot = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                }
                if let Some(dir) = reqs.toggle_series {
                    if !self.collapsed.remove(&dir) {
                        self.collapsed.insert(dir);
                    }
                    self.dirty_collapsed = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                    if let Some(app) = self.app.as_ref() {
                        app.window.request_redraw();
                    }
                }
                if reqs.grant {
                    request_all_files(&self.android_app);
                }
                if reqs.open_picker {
                    launch_picker(&self.android_app);
                    self.picker_pending = true;
                }
                if reqs.open_library {
                    self.open_library();
                }
                if let Some(path) = reqs.open {
                    self.open_path(path);
                }
                if reqs.page_nav != 0 {
                    self.flip(reqs.page_nav);
                }
                if reqs.book_nav != 0 {
                    self.open_sibling_book(reqs.book_nav);
                }
                if reqs.to_library {
                    self.open_library();
                }
                // Reading view with controls hidden → immersive (bars hidden);
                // otherwise show the bars (controls up, library, or empty screen).
                let immersive = self.current_key.is_some()
                    && self
                        .app
                        .as_ref()
                        .is_some_and(|a| !a.controls && !a.library_view);
                if self.applied_immersive != Some(immersive) {
                    set_immersive(&self.android_app, immersive);
                    self.applied_immersive = Some(immersive);
                }
                // Hold the screen awake only while there is actually a page to read
                // — a reader that never touches the screen for minutes at a time
                // does need this, but the library grid and the empty state are just
                // ordinary UI and should dim on the system timeout like anything
                // else. Same cached-flag pattern as `applied_immersive`: the window
                // flag is a JNI round-trip, so only push it when it changes.
                let keep_on = self.current_key.is_some()
                    && self.app.as_ref().is_some_and(|a| !a.library_view);
                if self.applied_keep_on != Some(keep_on) {
                    set_keep_screen_on(&self.android_app, keep_on);
                    self.applied_keep_on = Some(keep_on);
                }
                // Opportunistic: writes what has been dirty long enough. Frames stop
                // when the reader idles, so this is a best-effort early write — the
                // guaranteed ones are the forced flushes on suspend / book switch.
                self.flush_saves(false);
            }
            _ => {}
        }
    }
}

impl Shell {
    /// Route a touch event into swipe/tap (one finger) or pinch-zoom/pan (two).
    fn handle_touch(&mut self, phase: TouchPhase, id: u64, x: f64, y: f64, egui_consumed: bool) {
        let library = self.app.as_ref().map(|a| a.library_view).unwrap_or(false);
        match phase {
            TouchPhase::Started => {
                self.touches.insert(id, (x, y));
                if library {
                    return;
                }
                if self.touches.len() == 1 {
                    self.gesture_start = Some((x, y));
                    self.page_drag = false;
                    self.scroll_drag = false;
                    // A finger down catches any in-flight scroll glide.
                    if let Some(app) = self.app.as_mut() {
                        app.reader.stop_fling();
                    }
                    self.drag_samples.clear();
                } else if self.touches.len() == 2 {
                    // Begin a pinch; cancel the single-finger gesture — including
                    // a live page drag, which snaps back.
                    self.gesture_start = None;
                    if self.page_drag {
                        self.page_drag = false;
                        if let Some(app) = self.app.as_mut() {
                            app.reader.drag_cancel();
                            app.window.request_redraw();
                        }
                    }
                    if let (Some((d, mx, my)), Some(app)) =
                        (self.two_finger_metrics(), self.app.as_ref())
                    {
                        self.pinch = Some(Pinch {
                            dist0: d,
                            zoom0: app.reader.zoom,
                            pan0: (app.reader.pan_x, app.reader.pan_y),
                            mid0: (mx, my),
                        });
                    }
                }
            }
            TouchPhase::Moved => {
                let prev = self.touches.insert(id, (x, y));
                if library {
                    return;
                }
                if let Some(p) = self.pinch {
                    // Pinch → zoom (engine re-decodes HQ once it settles), anchored
                    // to the finger midpoint so the content under the fingers stays
                    // put (and follows a two-finger drag).
                    if p.dist0 > 1.0 {
                        if let (Some((d, mx, my)), Some(app)) =
                            (self.two_finger_metrics(), self.app.as_mut())
                        {
                            let (sw, sh) = (app.config.width as f32, app.config.height as f32);
                            // Raw target from finger spread, then a fit "detent": within
                            // ONE gesture the zoom can approach the fit scale (zoom == 1.0)
                            // but not cross it, so a single max zoom-out from above — or
                            // zoom-in from below — lands exactly on fit. Crossing requires
                            // releasing and re-pinching (then zoom0 ~= 1.0, so no barrier).
                            // Barrier side comes from the immutable zoom0; we hard-clamp (no
                            // dist0 re-baseline) so the barrier holds — at the cost of a
                            // small dead zone if the user over-pinches past fit and reverses
                            // mid-gesture.
                            const FIT: f32 = 1.0;
                            const EPS: f32 = 0.001; // matches the fit-reset button's "fitted" test
                            let raw = p.zoom0 * (d / p.dist0) as f32;
                            app.reader.zoom = if p.zoom0 > FIT + EPS {
                                raw.max(FIT) // started above fit: can't drop below it this gesture
                            } else if p.zoom0 < FIT - EPS {
                                raw.min(FIT) // started below fit: can't rise above it this gesture
                            } else {
                                raw // started at fit: free to cross either way (re-pinch path)
                            };
                            app.reader.clamp_zoom_native();
                            // Actual (post-clamp) scale ratio: keep the content point
                            // under the initial midpoint pinned to the current one.
                            let k = app.reader.zoom / p.zoom0;
                            app.reader.pan_x =
                                mx as f32 - sw / 2.0 - k * (p.mid0.0 as f32 - sw / 2.0 - p.pan0.0);
                            app.reader.pan_y =
                                my as f32 - sh / 2.0 - k * (p.mid0.1 as f32 - sh / 2.0 - p.pan0.1);
                            app.reader.clamp_pan();
                            app.window.request_redraw();
                        }
                    }
                } else if self.touches.len() == 1 {
                    // Single finger while zoomed in → pan.
                    let zoomed = self.app.as_ref().map(|a| a.reader.zoom > 1.001).unwrap_or(false);
                    if zoomed {
                        if let (Some((px, py)), Some(app)) = (prev, self.app.as_mut()) {
                            app.reader.pan_x += (x - px) as f32;
                            app.reader.pan_y += (y - py) as f32;
                            app.reader.clamp_pan();
                            app.window.request_redraw();
                        }
                    } else if !egui_consumed
                        && let Some((sx, sy)) = self.gesture_start
                    {
                        let (dx, dy) = (x - sx, y - sy);
                        let scroll_mode =
                            self.app.as_ref().map(|a| a.reader.scroll_mode).unwrap_or(false);
                        if scroll_mode {
                            // Scroll mode: a vertical drag scrolls the strip
                            // continuously (incremental, finger-tracking). Locks once
                            // the motion is clearly vertical so taps/seekbar survive.
                            if !self.scroll_drag {
                                let h =
                                    self.app.as_ref().map(|a| a.config.height as f64).unwrap_or(1.0);
                                self.scroll_drag = dy.abs() > h * 0.01 && dy.abs() > dx.abs();
                            }
                            if self.scroll_drag {
                                // Sample (time, y) for the release fling velocity
                                // (reuses drag_samples; page-flip uses it for x, but
                                // the two modes are mutually exclusive per gesture).
                                let now = Instant::now();
                                self.drag_samples.push_back((now, y));
                                while self
                                    .drag_samples
                                    .front()
                                    .is_some_and(|(t, _)| now.duration_since(*t).as_millis() > 100)
                                {
                                    self.drag_samples.pop_front();
                                }
                                if let (Some((_, py)), Some(app)) = (prev, self.app.as_mut()) {
                                    app.reader.top_offset -= (y - py) as f32;
                                    app.reader.normalize();
                                    app.window.request_redraw();
                                }
                            }
                        } else {
                            // Page-flip: the page follows the finger (Chunky-style),
                            // the neighbor revealed underneath. Locks once the motion
                            // is clearly horizontal, so taps and the seekbar stay intact.
                            if !self.page_drag {
                                let w =
                                    self.app.as_ref().map(|a| a.config.width as f64).unwrap_or(1.0);
                                self.page_drag = dx.abs() > w * 0.015 && dx.abs() > dy.abs();
                            }
                            if self.page_drag
                                && let Some(app) = self.app.as_mut()
                            {
                                let now = Instant::now();
                                self.drag_samples.push_back((now, x));
                                while self
                                    .drag_samples
                                    .front()
                                    .is_some_and(|(t, _)| now.duration_since(*t).as_millis() > 100)
                                {
                                    self.drag_samples.pop_front();
                                }
                                app.reader.drag_update(dx as f32);
                                app.window.request_redraw();
                            }
                        }
                    }
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&id);
                if self.touches.len() < 2 {
                    self.pinch = None;
                }
                if self.touches.is_empty() {
                    let was_drag = std::mem::take(&mut self.page_drag);
                    let was_scroll = std::mem::take(&mut self.scroll_drag);
                    let start = self.gesture_start.take();
                    if was_scroll {
                        // Scroll release → inertial fling from the recent y-velocity.
                        if !matches!(phase, TouchPhase::Cancelled) {
                            let now = Instant::now();
                            let vy = self.drag_samples.front().map_or(0.0, |(t0, y0)| {
                                let dt = now.duration_since(*t0).as_secs_f64();
                                if dt > 0.005 { (y - y0) / dt } else { 0.0 }
                            });
                            self.last_fling_tick = now;
                            if let Some(app) = self.app.as_mut() {
                                // strip velocity = −finger velocity (flick up → forward)
                                app.reader.start_fling(-vy as f32);
                                app.window.request_redraw();
                            }
                        }
                    } else if was_drag {
                        // The interactive drag owns this gesture end-to-end; the
                        // old end-of-gesture swipe must not also fire.
                        if matches!(phase, TouchPhase::Cancelled) {
                            if let Some(app) = self.app.as_mut() {
                                app.reader.drag_cancel();
                                app.window.request_redraw();
                            }
                        } else {
                            // Release velocity from the ~100 ms sample window —
                            // decides flick-to-commit on short drags.
                            let now = Instant::now();
                            let v = self.drag_samples.front().map_or(0.0, |(t0, x0)| {
                                let dt = now.duration_since(*t0).as_secs_f64();
                                if dt > 0.005 { (x - x0) / dt } else { 0.0 }
                            });
                            let committed = self
                                .app
                                .as_mut()
                                .is_some_and(|a| a.reader.drag_release(v as f32));
                            if committed {
                                self.after_flip();
                            } else {
                                // A commit-strength swipe into the volume boundary
                                // counts as intent to leave the book: arm/confirm
                                // the next/prev-book prompt. (A reversal or a
                                // sub-threshold release does not — same rules as
                                // a real commit, via the shared drag_commits.)
                                let edge_dir = self.app.as_ref().and_then(|a| {
                                    let (sx, _) = start?;
                                    let dxf = (x - sx) as f32;
                                    let w = a.config.width as f32;
                                    let dir = drag_dir(a.reader.direction, dxf);
                                    (drag_commits(dxf, v as f32, w) && a.reader.at_edge(dir))
                                        .then_some(dir)
                                });
                                if let Some(dir) = edge_dir {
                                    self.boundary_hit(dir);
                                } else if let Some(app) = self.app.as_mut() {
                                    app.window.request_redraw(); // play the snap-back
                                }
                            }
                        }
                        self.drag_samples.clear();
                    } else if let Some((sx, sy)) = start
                        && !library
                        && !egui_consumed
                    {
                        self.handle_gesture(sx, sy, x, y);
                    }
                }
            }
        }
    }

    /// Distance and midpoint (screen px) of the first two active touch points.
    /// Returning both from one call keeps them on the same finger pair.
    fn two_finger_metrics(&self) -> Option<(f64, f64, f64)> {
        let mut it = self.touches.values();
        let a = it.next()?;
        let b = it.next()?;
        let dist = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        Some((dist, (a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0))
    }

    /// A single-finger gesture ended without locking into a page drag: a
    /// near-stationary release is a tap; anything that travelled was a zoomed
    /// pan (or a vertical scrub) and is not. Unzoomed horizontal motion locks
    /// into the interactive drag long before reaching here, so flipping is
    /// handled at `TouchPhase::Ended` by `drag_release`.
    fn handle_gesture(&mut self, sx: f64, sy: f64, ex: f64, ey: f64) {
        let w = self.app.as_ref().map(|a| a.config.width as f64).unwrap_or(1.0);
        let (dx, dy) = (ex - sx, ey - sy);
        if dx.abs() < w * 0.015 && dy.abs() < w * 0.015 {
            self.on_tap(sx, sy);
        }
    }

    /// Flip `dir` pages and persist the new position. Running out of pages at a
    /// volume boundary arms (or confirms) the next/prev-book prompt instead.
    fn flip(&mut self, dir: i64) {
        let Some(app) = self.app.as_mut() else { return };
        if app.reader.step(dir) {
            self.after_flip();
        } else if self.app.as_ref().is_some_and(|a| a.reader.at_edge(dir)) {
            self.boundary_hit(dir);
        }
    }

    /// Post-flip bookkeeping shared by tap flips and committed drags (the step
    /// already happened): reset the view to fit and persist the position.
    fn after_flip(&mut self) {
        if let Some(app) = self.app.as_mut() {
            app.reader.zoom = 1.0;
            app.reader.pan_x = 0.0;
            app.reader.pan_y = 0.0;
            app.window.request_redraw();
            let idx = app.reader.index;
            if let Some(k) = self.current_key.clone() {
                self.positions.insert(k, idx);
                self.mark_positions();
            }
        }
    }

    /// Drain everything the shell's background threads deliver. Called once per
    /// frame from `RedrawRequested`, before the render — every producer wakes the
    /// loop after sending, so a result never waits for an unrelated frame.
    fn poll_background(&mut self) {
        // Finished opens, newest generation only: a later open bumped `open_gen`,
        // so a superseded result is dropped rather than swapping in a stale book.
        loop {
            match self.open_rx.try_recv() {
                Ok((generation, res)) => {
                    if generation != self.open_gen {
                        continue; // superseded by a newer open
                    }
                    self.opening = None;
                    self.opening_key = None;
                    let fallback = std::mem::take(&mut self.resume_fallback);
                    match res {
                        Ok((key, src)) if !src.is_empty() => {
                            log::info!("open {key}: {} pages", src.len());
                            self.open_comic(key, src);
                        }
                        Ok((key, _)) => {
                            log::warn!("no images found in {key}");
                            self.after_failed_open(fallback);
                        }
                        Err(e) => {
                            log::error!("open failed: {e}");
                            self.after_failed_open(fallback);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                // The Shell owns the Sender, so this can't happen — but fold it into
                // "nothing is opening" anyway so the spinner can never stick.
                Err(TryRecvError::Disconnected) => {
                    self.opening = None;
                    self.opening_key = None;
                    break;
                }
            }
        }
        // Sibling listings only warm the next/prev-book cache — nothing to show…
        let mut sibs_landed = false;
        while let Ok(entry) = self.sib_rx.try_recv() {
            self.sib_cache = Some(entry);
            sibs_landed = true;
        }
        if sibs_landed {
            // …unless a book jump was parked waiting for exactly this listing: replay
            // it now. If the book moved folders meanwhile this re-warms and re-parks
            // instead — still terminating, since each pass needs a fresh scan.
            if let Some(d) = self.pending_book_nav.take() {
                self.open_sibling_book(d);
            }
            // An armed boundary prompt resolved to "no neighbour" only because the
            // cache was cold: fill it in now that the listing is here.
            if self
                .app
                .as_ref()
                .and_then(|a| a.book_prompt.as_ref())
                .is_some_and(|p| p.sibling.is_none())
            {
                self.refresh_prompt_sibling();
            }
        }
        while let Ok(ok) = self.recents_ok_rx.try_recv() {
            self.recents_ok = ok;
        }
    }

    /// An open that produced nothing: with `fallback` (the launch/resume reopen)
    /// land on the library rather than leaving the user staring at the last frame
    /// of the spinner.
    fn after_failed_open(&mut self, fallback: bool) {
        if fallback {
            self.open_library();
        } else if let Some(app) = self.app.as_ref() {
            app.window.request_redraw(); // drop the spinner
        }
    }

    /// Open the library browser at the configured root, refreshing all-files
    /// access first (the browser shows the grant prompt if it's missing). Shared by
    /// Advance the inertial scroll glide one frame (a no-op unless flinging): apply
    /// the velocity to the strip, decay it, and schedule the next frame while it lasts.
    fn tick_scroll_fling(&mut self) {
        let flinging = self.app.as_ref().map(|a| a.reader.flinging()).unwrap_or(false);
        if !flinging {
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_fling_tick).as_secs_f32().clamp(0.0, 0.05);
        self.last_fling_tick = now;
        if let Some(app) = self.app.as_mut()
            && app.reader.fling_tick(dt)
        {
            app.window.request_redraw();
        }
    }

    /// the top-strip tap and the empty-state "Browse library" button.
    fn open_library(&mut self) {
        self.has_files = has_all_files(&self.android_app);
        // Which recents still exist is decided once here, off-thread, instead of
        // stat-ing all 32 of them on every library frame.
        self.refresh_recents_ok();
        if let Some(app) = self.app.as_mut() {
            app.library_view = true;
            app.lib_browse = false;
            if app.series.is_empty() && !app.series_pending {
                app.kick_series_scan();
            }
            app.window.request_redraw();
        }
    }

    /// Tap-zones: top strip opens the library; thin side edges flip (direction-aware);
    /// the large center toggles the reading chrome (seekbar + hints).
    fn on_tap(&mut self, x: f64, y: f64) {
        let Some((w, h)) = self
            .app
            .as_ref()
            .map(|a| (a.config.width as f64, a.config.height as f64))
        else {
            return;
        };
        // When the library is open, every tap belongs to egui (the ✕ button closes
        // it). egui's "consumed" flag lags a frame on a touch press, so without this
        // a tap near the top would fall through here and toggle the library shut
        // before egui could register the row's click.
        if self.app.as_ref().map(|a| a.library_view).unwrap_or(false) {
            return;
        }
        // While the next/prev-book prompt is up it acts like a lightweight modal.
        // Tapping the card — or the same next/prev-page edge zone that armed it
        // (the one the card anchors to) — opens the sibling: the "tap again to
        // open" gesture. Any *other* tap just dismisses the prompt and is
        // swallowed (e.g. tapping the opposite edge dismisses instead of paging
        // back; center/top neither toggle chrome nor open the library).
        if self.app.as_ref().is_some_and(|a| a.book_prompt.is_some()) {
            let confirm_dir = self.app.as_ref().and_then(|app| {
                let p = app.book_prompt.as_ref()?;
                let rtl = app.reader.direction == Direction::Rtl;
                let on_left = book_prompt_on_left(p.dir, rtl);
                // The confirm edge is the one whose page-flip matches p.dir (left
                // in RTL for next / LTR for prev, mirrored) — the side the card
                // anchors to. Its whole vertical strip confirms, card or not.
                let on_edge = if on_left {
                    x < w * EDGE_ZONE as f64
                } else {
                    x > w * (1.0 - EDGE_ZONE as f64)
                };
                // Card rect: anchored 16pt in from that edge, ~centered vertically
                // (covers the card's center-side half that's past the edge zone).
                let ppp = app.egui_ctx.pixels_per_point() as f64;
                let half_w = (BOOK_PROMPT_W_PT as f64 / 2.0 + 12.0) * ppp;
                let half_h = 200.0 * ppp;
                let cx = if on_left {
                    (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                } else {
                    w - (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                };
                let on_card = (x - cx).abs() < half_w && (y - h / 2.0).abs() < half_h;
                ((on_card || on_edge)
                    && p.sibling.is_some()
                    && p.at.elapsed().as_millis() < BOOK_PROMPT_MS as u128)
                    .then_some(p.dir)
            });
            if let Some(dir) = confirm_dir {
                if let Some(app) = self.app.as_mut() {
                    app.book_prompt = None;
                }
                self.open_sibling_book(dir);
            } else if let Some(app) = self.app.as_mut() {
                app.book_prompt = None; // any other tap dismisses, swallowed
                app.window.request_redraw();
            }
            return;
        }
        if y < h * TOP_ZONE as f64 {
            // Top strip opens the library browser.
            self.open_library();
            return;
        }
        let rtl = self
            .app
            .as_ref()
            .map(|a| a.reader.direction == Direction::Rtl)
            .unwrap_or(false);
        // In scroll mode there are no discrete pages to flip, so the side edges fall
        // through to the center action (toggle the reading chrome).
        let scroll_mode = self.app.as_ref().map(|a| a.reader.scroll_mode).unwrap_or(false);
        if !scroll_mode && x < w * EDGE_ZONE as f64 {
            // Left edge: next in RTL, previous in LTR.
            self.flip(if rtl { 1 } else { -1 });
        } else if !scroll_mode && x > w * (1.0 - EDGE_ZONE as f64) {
            // Right edge: previous in RTL, next in LTR.
            self.flip(if rtl { -1 } else { 1 });
        } else if let Some(app) = self.app.as_mut() {
            // Center: toggle the reading chrome (also closes the options popup).
            // (An armed book prompt is handled by the gate above, so it never
            // reaches here.)
            app.controls = !app.controls;
            if app.controls {
                app.controls_shown_at = Instant::now(); // re-show the zone hints
            } else {
                app.show_options = false;
                app.show_info = false;
            }
            app.window.request_redraw();
        }
    }

    /// The previous/next comic in the current comic's folder (`dir` = -1/+1),
    /// natural-sorted like the library. `Missing` at the ends or for non-path
    /// (SAF-picked) sources, whose keys aren't filesystem paths.
    ///
    /// Reads the background-warmed [`Shell::sib_cache`] only. On a miss it kicks the
    /// scan and answers `Cold` — a `read_dir` plus a stat per entry on the UI thread
    /// is precisely the hitch this phase exists to remove.
    fn sibling_book_path(&mut self, dir: i64) -> SibLookup {
        // Base "current" on the pending target while an open is in flight, so a
        // second next-book tap advances from the not-yet-loaded neighbour instead
        // of asking for it again.
        let Some(cur) = self.opening_key.clone().or_else(|| self.current_key.clone()) else {
            return SibLookup::Missing;
        };
        let cur_path = PathBuf::from(&cur);
        let Some(parent) = cur_path.parent().map(|p| p.to_path_buf()) else {
            return SibLookup::Missing;
        };
        if cur.starts_with("content://") {
            return SibLookup::Missing; // SAF one-off: no folder to walk
        }
        if !matches!(&self.sib_cache, Some((p, _)) if *p == parent) {
            self.warm_sib_cache(&cur_path);
            return SibLookup::Cold;
        }
        let comics = &self.sib_cache.as_ref().unwrap().1;
        let Some(i) = comics.iter().position(|p| *p == cur_path) else {
            return SibLookup::Missing;
        };
        let j = i as i64 + dir;
        match (0..comics.len() as i64).contains(&j) {
            true => SibLookup::Found(comics[j as usize].clone()),
            false => SibLookup::Missing,
        }
    }

    /// Scan the current book's folder for sibling comics on a background thread and
    /// hand the listing back for `sib_cache` (consumed in `poll_background`).
    fn warm_sib_cache(&self, vol: &Path) {
        let Some(parent) = vol.parent().map(|p| p.to_path_buf()) else {
            return;
        };
        let tx = self.sib_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let comics: Vec<PathBuf> = scan_dir(&parent)
                .into_iter()
                .filter_map(|e| match e {
                    Entry::Comic(p) => Some(p),
                    _ => None,
                })
                .collect();
            // Send, then wake: a parked jump replays on the frame this schedules.
            let _ = tx.send((parent, comics));
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Open the previous/next comic in the current comic's folder (`dir` = -1/+1).
    /// No-op at the ends or for non-path sources. With a cold cache the request is
    /// parked (accumulating, so repeated taps travel that many books) and replayed
    /// when the listing lands.
    fn open_sibling_book(&mut self, dir: i64) {
        match self.sibling_book_path(dir) {
            SibLookup::Found(p) => self.open_path(p),
            SibLookup::Cold => {
                self.pending_book_nav = Some(self.pending_book_nav.take().unwrap_or(0) + dir);
            }
            SibLookup::Missing => {}
        }
    }

    /// Fill in an armed boundary prompt's neighbour once the folder listing lands
    /// (the prompt is shown immediately on the boundary hit, before the scan is
    /// back, so its cover + "tap again to open" arrive a moment later).
    fn refresh_prompt_sibling(&mut self) {
        let Some(dir) = self
            .app
            .as_ref()
            .and_then(|a| a.book_prompt.as_ref())
            .map(|p| p.dir)
        else {
            return;
        };
        let SibLookup::Found(path) = self.sibling_book_path(dir) else {
            return;
        };
        self.queue_cover(&path);
        if let Some(app) = self.app.as_mut()
            && let Some(p) = app.book_prompt.as_mut()
        {
            p.title = name_of(&path);
            p.sibling = Some(path);
            app.window.request_redraw();
        }
    }

    /// Queue a comic's cover for background decode, unless it's already decoded.
    /// (Shared by the boundary prompt's two arming paths.)
    fn queue_cover(&mut self, path: &Path) {
        if let Some(app) = self.app.as_mut()
            && !app.thumbs.contains_key(path)
            && app.cover_tx.send(vec![path.to_path_buf()]).is_ok()
        {
            app.covers_inflight += 1;
        }
    }

    /// A next/prev action ran out of pages (`Reader::at_edge`). First hit arms
    /// the "Next/Previous book" prompt (resolving the sibling once and kicking
    /// its cover decode); a repeat in the same direction within the window opens
    /// it. (Tapping the card directly is handled by its egui button / `on_tap`.)
    fn boundary_hit(&mut self, dir: i64) {
        let confirm = self
            .app
            .as_ref()
            .and_then(|a| a.book_prompt.as_ref())
            .is_some_and(|p| {
                p.dir == dir
                    && p.sibling.is_some()
                    && p.at.elapsed().as_millis() < BOOK_PROMPT_MS as u128
            });
        if confirm {
            if let Some(app) = self.app.as_mut() {
                app.book_prompt = None;
            }
            self.open_sibling_book(dir);
            return;
        }
        // Cache-only lookup: a cold cache arms the card with no neighbour yet and
        // `poll_background` fills it in when the folder listing lands (the prompt is
        // on screen instantly either way).
        let sibling = match self.sibling_book_path(dir) {
            SibLookup::Found(p) => Some(p),
            SibLookup::Cold | SibLookup::Missing => None,
        };
        let title = sibling.as_deref().map(name_of).unwrap_or_default();
        if let Some(p) = &sibling {
            // Cover preview: same off-thread pipeline as the library; the result
            // lands via the per-frame `thumb_rx` drain (woken by the worker) and the
            // card picks it up while the prompt is showing.
            self.queue_cover(p);
        }
        if let Some(app) = self.app.as_mut() {
            app.book_prompt = Some(BookPrompt {
                dir,
                at: Instant::now(),
                sibling,
                title,
            });
            app.window.request_redraw();
        }
    }

    /// Open a comic by path: route by type (archive / image folder) to an engine
    /// source, then through `open_comic` (which restores its saved position).
    /// The source construction runs on a background thread — see [`Shell::start_open`].
    fn open_path(&mut self, path: PathBuf) {
        let key = path.to_string_lossy().into_owned();
        self.start_open(Some(key.clone()), move || {
            build_source(&path)
                .map(|src| (key, src))
                .ok_or_else(|| format!("could not open {path:?}"))
        });
    }

    /// Kick the launch resume: pick the most-recently-read book that still exists
    /// and open it — **both** steps off-thread, since deciding which recent is still
    /// on disk is a stat per entry and this runs before the first frame.
    fn resume_recent(&mut self) {
        let recents = self.recents.clone();
        self.start_open(None, move || {
            let key = recents
                .into_iter()
                .find(|k| is_reopenable_fs(k))
                .ok_or_else(|| "no reopenable recent".to_string())?;
            let src =
                build_source(Path::new(&key)).ok_or_else(|| format!("could not open {key}"))?;
            Ok((key, src))
        });
    }

    /// Begin a background open. `target` is the comic's identity when the caller
    /// knows it (it names the spinner and anchors next/prev-book); `job` does all
    /// the I/O and returns that identity plus the source — the key comes back from
    /// the job because the launch resume also *chooses* the book off-thread.
    ///
    /// Each call bumps `open_gen`, so only the newest result is applied: repeated
    /// next-book taps supersede in-flight opens instead of queuing stale swaps.
    fn start_open<F>(&mut self, target: Option<String>, job: F)
    where
        F: FnOnce() -> OpenResult + Send + 'static,
    {
        self.open_gen = self.open_gen.wrapping_add(1);
        let generation = self.open_gen;
        self.opening = Some(target.as_deref().map(book_display_name).unwrap_or_default());
        self.opening_key = target;
        self.resume_fallback = false; // a caller that wants it re-arms it after
        let tx = self.open_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            // Send, then wake: the loop idles while the open runs, so the result
            // only reaches the screen if this schedules the frame that drains it.
            let _ = tx.send((generation, job()));
            if let Some(w) = &waker {
                w();
            }
        });
        if let Some(app) = self.app.as_ref() {
            app.window.request_redraw(); // put the spinner up this frame
        }
    }

    /// Point the reader at `src` (identity `key`): save the outgoing comic's
    /// position, restore this one's, attach the source. Persists to disk.
    fn open_comic(&mut self, key: String, src: Arc<dyn PageSource>) {
        if let (Some(old), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(old, app.reader.index);
        }
        let start = self
            .positions
            .get(&key)
            .copied()
            .unwrap_or(0)
            .min(src.len().saturating_sub(1));
        if src.is_partial() {
            // No on-screen toast system on Android yet, so just log it; the archive
            // still opens with the pages recovered before the truncation.
            log::info!("partial archive recovered: {} pages", src.len());
        }
        // Park it for the next resume *before* handing it to the reader: a suspend
        // tears the whole App down, and re-attaching these same bytes is the
        // difference between a resume that's instant and one that re-reads the
        // archive off storage. Only one book is ever parked — this replaces whatever
        // the last one was, so switching books can't accumulate archives in memory.
        self.parked_source = Some((key.clone(), src.clone()));
        if let Some(app) = self.app.as_mut() {
            attach_source(
                &mut app.reader,
                &app.ctx.device.clone(),
                &app.ctx.queue.clone(),
                src,
                start,
            );
            app.book_title = book_display_name(&key);
            app.book_prompt = None; // a boundary prompt belongs to the old book
            app.window.request_redraw();
        }
        self.current_key = Some(key.clone());
        // Bump to the front of the recently-read list (MRU, deduped, capped).
        self.recents.retain(|k| k != &key);
        self.recents.insert(0, key.clone());
        self.recents.truncate(RECENTS_CAP);
        // The book we just opened demonstrably exists, so the library's recents row
        // can show it without waiting for another existence pass, and its folder is
        // worth warming for the first next/prev-book request. (SAF one-offs are
        // neither reopenable nor inside a browsable folder, so they get neither.)
        if !key.starts_with("content://") {
            self.recents_ok.insert(key.clone());
            self.warm_sib_cache(Path::new(&key));
        }
        self.positions.insert(key, start);
        self.dirty_recents = true;
        self.mark_positions();
        // A book switch is a natural durability point (and rare): write now rather
        // than leave the outgoing book's page riding on a debounce.
        self.flush_saves(true);
    }

    /// Note that the reading position — and the progress map that rides with it —
    /// changed. The write itself is debounced; see [`Shell::flush_saves`].
    fn mark_positions(&mut self) {
        self.dirty_positions = true;
        self.dirty_progress = true;
        self.dirty_since.get_or_insert_with(Instant::now);
    }

    /// Write out whatever has been marked dirty. `force` writes immediately (book
    /// switch, suspend, teardown); otherwise nothing happens until the oldest mark
    /// is [`SAVE_DEBOUNCE`] old, so a burst of page turns costs one write.
    ///
    /// Each file is rewritten whole through a `<path>.tmp` + rename, so a kill in
    /// the middle of a write leaves the previous file intact rather than a truncated
    /// one (the model is `thumbcache::store`). The writes stay on the calling
    /// thread: they are small, now rare, and `suspended()` must have them on disk
    /// before it returns.
    fn flush_saves(&mut self, force: bool) {
        let Some(since) = self.dirty_since else { return };
        if !force && since.elapsed() < SAVE_DEBOUNCE {
            return;
        }
        self.dirty_since = None;
        if std::mem::take(&mut self.dirty_positions)
            && let Some(path) = &self.pos_path
        {
            let mut out = String::new();
            for (k, v) in &self.positions {
                out.push_str(&format!("{v}\t{k}\n"));
            }
            write_atomic(path, &out);
        }
        if std::mem::take(&mut self.dirty_progress)
            && let Some(path) = &self.progress_path
        {
            let mut out = String::new();
            for (k, (furthest, total)) in &self.progress {
                out.push_str(&format!("{furthest}\t{total}\t{k}\n"));
            }
            write_atomic(path, &out);
        }
        if std::mem::take(&mut self.dirty_recents)
            && let Some(path) = &self.recents_path
        {
            write_atomic(path, &self.recents.join("\n"));
        }
        if std::mem::take(&mut self.dirty_collapsed)
            && let Some(path) = &self.collapsed_path
        {
            let mut out = String::new();
            for p in &self.collapsed {
                out.push_str(&format!("{}\n", p.display()));
            }
            write_atomic(path, &out);
        }
        // The viewing options and the library root live on the App (which the
        // egui frame mutates); read the current values back off it at write time.
        if std::mem::take(&mut self.dirty_view)
            && let (Some(path), Some(app)) = (&self.view_file, &self.app)
        {
            save_view(
                path,
                app.reader.direction,
                app.layout_mode,
                app.reader.fit,
                app.reader.transition_enabled,
                app.reader.scroll_mode,
                app.resume_on_startup,
                app.theme,
                app.perf,
                app.spine_shadow_on,
                app.spine_shadow_strength,
                app.reader.fit_no_upscale,
            );
        }
        if std::mem::take(&mut self.dirty_libroot)
            && let (Some(path), Some(app)) = (&self.lib_dir_file, &self.app)
        {
            write_atomic(path, &app.lib_root.to_string_lossy());
        }
    }

    /// Recompute, off-thread, which `recents` entries still exist on disk.
    fn refresh_recents_ok(&self) {
        let keys = self.recents.clone();
        let tx = self.recents_ok_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let ok: HashSet<String> = keys.into_iter().filter(|k| is_reopenable_fs(k)).collect();
            let _ = tx.send(ok);
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Update the in-memory read-tracking entry for the open comic: the furthest
    /// page the reader has had on screen (1-based; the far page of a spread
    /// counts) and the volume's total. Called every rendered frame while reading
    /// — cheap map math; the file write happens with `save_positions`.
    fn note_progress(&mut self) {
        // `as_deref`, not `clone`: this runs every rendered frame, and the owned
        // key is only ever needed the first time a book is seen.
        let (Some(key), Some(app)) = (self.current_key.as_deref(), self.app.as_ref()) else {
            return;
        };
        let Some(src) = &app.reader.source else { return };
        let len = src.len();
        if len == 0 {
            return;
        }
        let (a, b) =
            view_pages(app.reader.layout, app.reader.index, len, app.reader.spread_offset);
        let seen = (b.unwrap_or(a) + 1) as u32;
        // `key` borrows `current_key`, the update touches `progress` — disjoint
        // fields, so no clone is needed to satisfy the borrow checker either.
        match self.progress.get_mut(key) {
            Some(e) => {
                e.0 = e.0.max(seen);
                e.1 = len as u32;
            }
            None => {
                self.progress.insert(key.to_string(), (seen, len as u32));
            }
        }
    }

    /// Read the chosen content:// file's bytes off its descriptor and open it.
    /// The JNI call that hands us the descriptor stays here (it wants the activity),
    /// but the descriptor is just an integer — reading a whole archive off it, and
    /// parsing it, happen on the open worker. A multi-hundred-MB SAF archive used to
    /// slurp on the UI thread, which is an ANR with a progress bar drawn on it.
    fn open_picked(&mut self, uri: &str) {
        let fd = open_fd(&self.android_app, uri);
        if fd < 0 {
            log::warn!("openFd failed for {uri}");
            return;
        }
        let key = uri.to_string();
        self.start_open(Some(key.clone()), move || {
            // We own the fd (Java detachFd'd it); reading to end + drop closes it.
            let mut file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut bytes)
                .map_err(|e| format!("read picked fd: {e}"))?;
            drop(file);
            let src = ZipSource::from_bytes(bytes).map_err(|e| format!("from_bytes: {e}"))?;
            Ok((key, Arc::new(src) as Arc<dyn PageSource>))
        });
    }
}

/// Rewrite `path` atomically: write a sibling `<path>.tmp`, then rename it over the
/// target. A rename within a directory is atomic, so a kill (or a battery pull)
/// mid-write leaves the previous file whole instead of a half-written one — these
/// are whole-file rewrites of state the user would notice losing.
fn write_atomic(path: &Path, contents: &str) {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Point the reader at a new page source: rebuild the decode pool, reset state,
/// kick prefetch. Shared by the test-comic open and the picker.
fn attach_source(
    reader: &mut Reader,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    src: Arc<dyn PageSource>,
    start: usize,
) {
    let pool = DecodePool::new(
        src.clone(),
        device.clone(),
        queue.clone(),
        reader.tex_pool.clone(),
        reader.workers,
    );
    // A new pool starts wakerless: re-install the shell's callback or this book's
    // decodes would land without ever scheduling the frame that draws them.
    pool.set_waker(reader.waker.clone());
    reader.pool = Some(pool);
    reader.cache.clear();
    reader.lq_cache.clear();
    reader.failed.clear();
    reader.rotation = 0; // each comic opens upright (mirrors the desktop shell)
    reader.index = start;
    reader.source = Some(src);
    // Fresh pool, fresh caches: drop the cross-frame prefetch memo *and* the
    // thumbnail-tail pivot, or the new pool's empty tail would stay empty until the
    // reader happened to travel a full stride.
    reader.invalidate_jobs();
    reader.prefetch();
}

/// The persisted viewing options, in `view.txt`'s positional order: direction,
/// layout mode, fit, page-turn animation, scroll mode, resume-on-startup, chrome
/// theme, performance profile, spine shadow on/off, spine-shadow strength,
/// no-upscale fit.
type InitView = (
    Direction,
    LayoutMode,
    FitMode,
    bool,
    bool,
    bool,
    ThemePref,
    PerfPref,
    bool,
    f32,
    bool,
);

/// Load the persisted per-comic positions (one `index\tkey` line each).
/// Parse persisted viewing options ("dir,layout,fit,anim"); default RTL / single /
/// window / animation on.
fn load_view(path: &Path) -> InitView {
    let (mut dir, mut lay, mut fit, mut anim, mut scroll, mut resume, mut theme, mut perf) = (
        Direction::Rtl,
        LayoutMode::Single,
        FitMode::Window,
        true,
        false,
        true,
        ThemePref::System,
        PerfPref::Auto,
    );
    let (mut spine, mut spine_strength) = (false, DEFAULT_SPINE_STRENGTH);
    let mut no_upscale = false;
    if let Ok(s) = std::fs::read_to_string(path) {
        let t: Vec<&str> = s.trim().split(',').collect();
        if t.first() == Some(&"ltr") {
            dir = Direction::Ltr;
        }
        // Layout slot gained "auto"; "single"/"spread" (and old files) still parse.
        lay = match t.get(1) {
            Some(&"spread") => LayoutMode::Spread,
            Some(&"auto") => LayoutMode::Auto,
            _ => LayoutMode::Single,
        };
        fit = match t.get(2) {
            Some(&"width") => FitMode::Width,
            Some(&"height") => FitMode::Height,
            Some(&"actual") => FitMode::Actual,
            _ => FitMode::Window,
        };
        // Page-turn animation: 4th slot; absent (older files) ⇒ on.
        anim = t.get(3) != Some(&"off");
        // 5th slot: scroll mode; absent ⇒ page-flip.
        scroll = t.get(4) == Some(&"scroll");
        // 6th slot: resume-on-launch; absent ⇒ on.
        resume = t.get(5) != Some(&"noresume");
        // 7th slot: chrome theme; absent (older files) ⇒ follow the system.
        theme = ThemePref::parse(t.get(6));
        // 8th slot: performance profile; absent (older files) ⇒ auto (the probed tier).
        perf = PerfPref::parse(t.get(7));
        // 9th slot: spine shadow; absent (older files) ⇒ off.
        spine = t.get(8) == Some(&"spine");
        // 10th slot: spine-shadow strength as a whole percent; absent ⇒ the default.
        spine_strength = t
            .get(9)
            .and_then(|v| v.parse::<i32>().ok())
            .map(|p| p.clamp(0, 100) as f32 / 100.0)
            .unwrap_or(DEFAULT_SPINE_STRENGTH);
        // 11th slot: no-upscale fit; absent (older files) ⇒ off (stretch to fit).
        no_upscale = t.get(10) == Some(&"noupscale");
    }
    (dir, lay, fit, anim, scroll, resume, theme, perf, spine, spine_strength, no_upscale)
}

/// Persist viewing options as
/// "dir,layout,fit,anim,scroll,resume,theme,perf,spine,spine_strength,no_upscale".
#[allow(clippy::too_many_arguments)]
fn save_view(
    path: &Path,
    dir: Direction,
    lay: LayoutMode,
    fit: FitMode,
    anim: bool,
    scroll: bool,
    resume: bool,
    theme: ThemePref,
    perf: PerfPref,
    spine: bool,
    spine_strength: f32,
    no_upscale: bool,
) {
    let d = if dir == Direction::Rtl { "rtl" } else { "ltr" };
    let l = lay.label();
    let f = match fit {
        FitMode::Width => "width",
        FitMode::Height => "height",
        FitMode::Actual => "actual",
        FitMode::Window => "window",
    };
    let a = if anim { "on" } else { "off" };
    let s = if scroll { "scroll" } else { "flip" };
    let r = if resume { "resume" } else { "noresume" };
    let t = theme.label();
    let p = perf.label();
    let sp = if spine { "spine" } else { "nospine" };
    let ss = (spine_strength.clamp(0.0, 1.0) * 100.0).round() as i32;
    let nu = if no_upscale { "noupscale" } else { "stretch" };
    write_atomic(path, &format!("{d},{l},{f},{a},{s},{r},{t},{p},{sp},{ss},{nu}"));
}

fn load_progress(path: &std::path::Path) -> HashMap<String, (u32, u32)> {
    let mut map = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for line in s.lines() {
            // `furthest \t total \t key` — key last (it's the free-form field).
            if let Some((furthest, rest)) = line.split_once('\t')
                && let Some((total, key)) = rest.split_once('\t')
                && let (Ok(f), Ok(t)) = (furthest.parse(), total.parse())
            {
                map.insert(key.to_string(), (f, t));
            }
        }
    }
    map
}

fn load_collapsed(path: &std::path::Path) -> std::collections::HashSet<PathBuf> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

/// Load the most-recently-read list (one volume key per line, newest first).
fn load_recents(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.is_empty()).map(String::from).collect())
        .unwrap_or_default()
}

/// True if `key` is a filesystem path we can silently rebuild a source from on
/// launch (not a SAF `content://` URI, and still present on disk).
fn is_reopenable_fs(key: &str) -> bool {
    !key.starts_with("content://") && std::path::Path::new(key).exists()
}

fn load_positions(path: &std::path::Path) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for line in s.lines() {
            if let Some((idx, key)) = line.split_once('\t') {
                if let Ok(i) = idx.parse() {
                    map.insert(key.to_string(), i);
                }
            }
        }
    }
    map
}

/// The system CJK fallback font (`Noto Sans CJK`), loaded **once per process**.
///
/// egui's bundled fonts have no CJK glyphs, so Japanese comic and file names would
/// render as tofu squares without this. The catch is that the file is ~40 MB and a
/// resume rebuilds the whole egui context: re-reading and re-copying it on every
/// foreground was one of the biggest single costs on the resume path. `FontData` is
/// immutable and egui takes it behind an `Arc`, so every egui context this process
/// builds can be handed the same one. `None` ⇒ the device has no such font (older /
/// non-CJK builds), and the fallback is simply not installed.
fn cjk_font() -> Option<Arc<egui::FontData>> {
    static CJK_FONT: std::sync::OnceLock<Option<Arc<egui::FontData>>> = std::sync::OnceLock::new();
    CJK_FONT
        .get_or_init(|| {
            let bytes = std::fs::read("/system/fonts/NotoSansCJK-Regular.ttc").ok()?;
            log::info!("loaded CJK fallback font ({} KB)", bytes.len() / 1024);
            Some(Arc::new(egui::FontData::from_owned(bytes)))
        })
        .clone()
}

/// Read one `/proc/meminfo` field in MB (the values are in kB).
fn meminfo_mb(text: &str, field: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix(field))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .map(|kb| kb / 1024)
}

/// `(budget_mb, total_mb)` — the memory the reader may use for its page cache +
/// GPU textures, and the device's raw RAM (which the tier heuristic reads).
///
/// Decoded pages and GPU textures are *native* allocations, not bounded by the
/// (small) Java heap, so a healthy slice of device RAM is fine. But **MemTotal
/// alone is a lie about what's available**: the old `MemTotal/8` made every ≥3 GB
/// phone — a cheap one included — saturate every `Budget::derive` clamp at the
/// desktop maxima, which is the main reason yosh ran badly on slow devices. Take
/// the smaller of a total-RAM slice and a third of what's actually free right now,
/// then clamp into a range that keeps a small device conservative and a big one
/// from claiming more than the reader can usefully hold.
///
/// **Probed once per process.** `MemTotal` is a constant, and while `MemAvailable`
/// genuinely moves, re-reading it per resume bought nothing: the budget it feeds is
/// a coarse tier of caps, and a resume is exactly when the number is least
/// meaningful (we have just torn our own textures down, so free memory is
/// transiently inflated). First-launch `MemAvailable` is as good a sample as any,
/// and it keeps the resume path off `/proc`. The real answer to a squeeze is
/// [`Shell::memory_warning`], which sheds caches on demand rather than guessing.
fn device_mem() -> (u64, u64) {
    static DEVICE_MEM: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *DEVICE_MEM.get_or_init(|| {
        let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total = meminfo_mb(&info, "MemTotal:").unwrap_or(4096);
        // Older kernels lack MemAvailable; fall back to the total-only estimate.
        let avail = meminfo_mb(&info, "MemAvailable:").unwrap_or(total / 2);
        let budget = (total / 8).min(avail / 3).clamp(192, 1024);
        (budget, total)
    })
}

/// Highest CPU clock (kHz) the SoC advertises, across all cores. This is the
/// big.LITTLE discriminator the tier heuristic needs: all-little budget SoCs top
/// out around 1.8–2.0 GHz while a flagship prime core is 2.8 GHz+, and core *count*
/// can't tell them apart (both are 8). `None` when sysfs isn't readable (some
/// devices restrict it) — the caller then tiers on RAM alone.
fn max_cpu_khz() -> Option<u64> {
    let dir = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut best: Option<u64> = None;
    for e in dir.flatten() {
        let p = e.path().join("cpufreq/cpuinfo_max_freq");
        if let Ok(s) = std::fs::read_to_string(&p)
            && let Ok(khz) = s.trim().parse::<u64>()
        {
            best = Some(best.map_or(khz, |b: u64| b.max(khz)));
        }
    }
    best
}

/// Classify the device from RAM + peak CPU clock. Deliberately conservative in the
/// middle: only a clearly-flagship device (≥ 8 GB **and** ≥ 2.5 GHz) gets `High`,
/// which is the tier that applies no ceilings at all. Never tier on GPU name —
/// the same GPU string ships across wildly different thermal envelopes.
///
/// (Pixel 9 Pro XL → High; Snapdragon 695 / 6 GB → Mid; Helio G85 / 4 GB → Low.)
fn device_tier(total_mb: u64, max_khz: Option<u64>) -> DeviceTier {
    let gb = total_mb as f64 / 1024.0;
    match max_khz {
        Some(khz) if khz < 2_100_000 => DeviceTier::Low,
        Some(khz) => {
            if gb < 4.0 {
                DeviceTier::Low
            } else if gb >= 8.0 && khz >= 2_500_000 {
                DeviceTier::High
            } else {
                DeviceTier::Mid
            }
        }
        // Clock unreadable: RAM-only fallback.
        None if gb < 4.0 => DeviceTier::Low,
        None if gb >= 8.0 => DeviceTier::High,
        None => DeviceTier::Mid,
    }
}

/// Bottom seekbar: "page / total" + a draggable slider that requests a jump.
/// Hidden for a single-page source.
/// Image-info overlay as a closable centered popup (Android take on the "I"
/// overlay): check it, then ✕ to dismiss — not a persistent overlay.
fn info_popup(ctx: &egui::Context, lines: &[(String, String)], close: &mut bool) {
    egui::Area::new(egui::Id::new("info_popup"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -20.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                ui.label(egui::RichText::new("Info").strong().size(18.0));
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
                        egui::Button::new(egui::RichText::new("× Close").size(16.0)),
                    )
                    .clicked()
                {
                    *close = true;
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
fn seekbar(
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
    let sw = ctx.screen_rect().width();
    // The seekbar hand-paints the slider/buttons (egui can't style them per-widget
    // here), so pick high-contrast tones for light/e-ink off the active theme.
    let light = !ctx.style().visuals.dark_mode;
    // Translucent panel so the page shows through behind the controls. Keep the
    // popup's color/stroke/rounding, just drop the fill's alpha.
    let frame = translucent_popup(&ctx.style());
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
fn empty_state(
    ctx: &egui::Context,
    logo: Option<&egui::TextureHandle>,
    open_library: &mut bool,
    open_picker: &mut bool,
) {
    egui::Area::new(egui::Id::new("empty_state"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width((ctx.screen_rect().width() * 0.8).min(420.0));
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
                    ui.label(egui::RichText::new("No comic open").strong().size(22.0));
                    ui.label(
                        egui::RichText::new("Open a comic to start reading.")
                            .size(15.0)
                            .color(weak),
                    );
                    ui.add_space(4.0);
                    ui.spacing_mut().button_padding = egui::vec2(18.0, 12.0);
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(format!(
                                "{}  Browse library",
                                egui_phosphor::fill::BOOKS
                            ))
                            .size(18.0),
                        ))
                        .clicked()
                    {
                        *open_library = true;
                    }
                    if ui
                        .add(egui::Button::new(egui::RichText::new("Open file…").size(18.0)))
                        .clicked()
                    {
                        *open_picker = true;
                    }
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new("Tip: tap the top of the screen any time to open your library.")
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
fn opening_card(ctx: &egui::Context, name: &str) {
    egui::Area::new(egui::Id::new("opening_card"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.set_max_width((ctx.screen_rect().width() * 0.8).min(360.0));
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 10.0);
                    ui.add(egui::Spinner::new().size(40.0));
                    ui.label(egui::RichText::new("Opening…").strong().size(18.0));
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

/// Tap-zone fractions, shared with `on_tap`: top strip opens the library, the side
/// edges flip.
const TOP_ZONE: f32 = 0.10;
const EDGE_ZONE: f32 = 0.20;

/// One volume in a series row: cover (faded when finished, thin progress bar
/// when started, highlight stroke when currently open) + truncated name.
/// Clicking opens it.
fn volume_cell(ui: &mut egui::Ui, v: &VolCell, reqs: &mut FrameReqs, close_lib: &mut bool) {
    const CELL_W: f32 = 140.0;
    const COVER_H: f32 = 186.0;
    ui.allocate_ui(egui::vec2(CELL_W, COVER_H + 38.0), |ui| {
        ui.vertical(|ui| {
            let finished = v.state == VolState::Finished;
            let r = if let Some(t) = &v.thumb {
                let mut img = egui::Image::new(t).fit_to_exact_size(egui::vec2(CELL_W, COVER_H));
                if finished {
                    // Read volumes fade out (Chunky-style).
                    img = img.tint(egui::Color32::from_white_alpha(72));
                }
                ui.add(egui::ImageButton::new(img))
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
            let mut label = egui::RichText::new(v.label.as_str()).size(12.0);
            if finished {
                label = label.weak();
            }
            ui.add_sized([CELL_W, 30.0], egui::Label::new(label).truncate());
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
fn book_prompt_card(
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
                            egui::RichText::new(if dir > 0 { "Last page" } else { "First page" })
                                .size(18.0)
                                .strong(),
                        );
                        return;
                    }
                    ui.label(
                        egui::RichText::new(if dir > 0 { "Next book" } else { "Previous book" })
                            .size(18.0)
                            .strong(),
                    );
                    ui.add_space(8.0);
                    let cover_w = BOOK_PROMPT_W_PT - 24.0;
                    let clicked = if let Some(t) = thumb {
                        ui.add(egui::ImageButton::new(
                            egui::Image::new(t).fit_to_exact_size(egui::vec2(cover_w, 240.0)),
                        ))
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
                    ui.label(egui::RichText::new("tap again to open").size(12.0).weak());
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
fn drag_shadow(ctx: &egui::Context, frac: f32, si: f32) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("drag_shadow"),
    ));
    let rect = ctx.screen_rect();
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
fn zone_hints(ctx: &egui::Context, rtl: bool, scroll: bool) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("zone_hints"),
    ));
    let rect = ctx.screen_rect();
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
        &format!("{}  Library", egui_phosphor::fill::BOOKS),
    );
    // Side flip labels only in page-flip.
    if !scroll {
        let (left, right) = if rtl { ("Next ›", "‹ Prev") } else { ("‹ Prev", "Next ›") };
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
fn options_popup(
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
    set_scroll: &mut Option<bool>,
    set_theme: &mut Option<ThemePref>,
    set_perf: &mut Option<PerfPref>,
    toggle_offset: &mut bool,
    rotate: &mut bool,
) {
    egui::Area::new(egui::Id::new("view_options"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -40.0))
        .show(ctx, |ui| {
            translucent_popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().button_padding = egui::vec2(16.0, 10.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 10.0);

                ui.label(egui::RichText::new("Reading mode").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(!scroll_on, egui::RichText::new("Page-flip").size(16.0))
                        .clicked()
                    {
                        *set_scroll = Some(false);
                    }
                    if ui
                        .selectable_label(scroll_on, egui::RichText::new("Scroll").size(16.0))
                        .clicked()
                    {
                        *set_scroll = Some(true);
                    }
                });

                ui.label(egui::RichText::new("Reading direction").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(dir == Direction::Ltr, egui::RichText::new("→ LTR").size(16.0))
                        .clicked()
                    {
                        *set_dir = Some(Direction::Ltr);
                    }
                    if ui
                        .selectable_label(dir == Direction::Rtl, egui::RichText::new("← RTL").size(16.0))
                        .clicked()
                    {
                        *set_dir = Some(Direction::Rtl);
                    }
                });

                ui.label(egui::RichText::new("Page layout").strong());
                ui.horizontal(|ui| {
                    for (m, text) in [
                        (LayoutMode::Single, "Single"),
                        (LayoutMode::Spread, "Two-page"),
                        (LayoutMode::Auto, "Auto"),
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
                        egui::RichText::new("Single in portrait, two-page in landscape — rotate to switch.")
                            .size(12.0)
                            .color(egui::Color32::from_white_alpha(150)),
                    );
                }
                // Shift which pages are paired together (e.g. so a cover sits alone).
                // Shown whenever the *effective* layout is a spread (incl. Auto-landscape).
                if effective_spread
                    && ui
                        .button(egui::RichText::new("Shift page pairing").size(16.0))
                        .clicked()
                {
                    *toggle_offset = true;
                }

                ui.label(egui::RichText::new("Fit").strong());
                ui.horizontal(|ui| {
                    for (f, text) in [
                        (FitMode::Window, "Window"),
                        (FitMode::Width, "Width"),
                        (FitMode::Height, "Height"),
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
                ui.label(egui::RichText::new("Stretch small pages").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(stretch_on, egui::RichText::new("On").size(16.0))
                        .clicked()
                    {
                        *set_no_upscale = Some(false);
                    }
                    if ui
                        .selectable_label(!stretch_on, egui::RichText::new("Off").size(16.0))
                        .clicked()
                    {
                        *set_no_upscale = Some(true);
                    }
                });
                ui.label(
                    egui::RichText::new("Off: fit never scales past 100% native (zoom still can).")
                        .size(12.0)
                        .color(egui::Color32::from_white_alpha(150)),
                );

                // Manual page rotation (single-page draws only — the engine ignores
                // it for spreads). Per-comic and transient: not persisted, reset to
                // upright on the next open, exactly like the desktop `R` key.
                ui.label(egui::RichText::new("Rotation").strong());
                if ui
                    .button(egui::RichText::new(format!("Rotate 90° ⟳  ·  now {}°", rotation as u32 * 90)).size(16.0))
                    .clicked()
                {
                    *rotate = true;
                }

                ui.label(egui::RichText::new("Page-turn animation").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(transition_on, egui::RichText::new("On").size(16.0))
                        .clicked()
                    {
                        *set_transition = Some(true);
                    }
                    if ui
                        .selectable_label(!transition_on, egui::RichText::new("Off").size(16.0))
                        .clicked()
                    {
                        *set_transition = Some(false);
                    }
                });

                // Book-gutter shading only means anything on an un-joined pair, so
                // show it with the other spread-only controls' gate.
                if effective_spread {
                    ui.label(egui::RichText::new("Spine shadow").strong());
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(spine_on, egui::RichText::new("On").size(16.0))
                            .clicked()
                        {
                            *set_spine_on = Some(true);
                        }
                        if ui
                            .selectable_label(!spine_on, egui::RichText::new("Off").size(16.0))
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
                                .text("strength"),
                        )
                        .changed()
                    {
                        *set_spine_strength = Some(strength);
                    }
                }

                ui.label(egui::RichText::new("Resume on startup").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(resume_on, egui::RichText::new("On").size(16.0))
                        .clicked()
                    {
                        *set_resume = Some(true);
                    }
                    if ui
                        .selectable_label(!resume_on, egui::RichText::new("Off").size(16.0))
                        .clicked()
                    {
                        *set_resume = Some(false);
                    }
                });

                ui.label(egui::RichText::new("Theme").strong());
                ui.horizontal(|ui| {
                    for (t, text) in [
                        (ThemePref::System, "System"),
                        (ThemePref::Light, "Light"),
                        (ThemePref::Dark, "Dark"),
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
                ui.label(egui::RichText::new("Performance").strong());
                ui.horizontal(|ui| {
                    for (p, text) in [
                        (PerfPref::Auto, "Auto"),
                        (PerfPref::Low, "Battery saver"),
                        (PerfPref::Mid, "Balanced"),
                        (PerfPref::High, "Performance"),
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
                    egui::RichText::new(
                        "Auto matches your device. Lower settings decode fewer pages ahead and use less memory and battery.",
                    )
                    .size(12.0)
                    .color(egui::Color32::from_white_alpha(150)),
                );
            });
        });
}

// --- JNI bridge to YoshActivity ---------------------------------------------

/// Launch the SAF document picker via `YoshActivity.openDocument()`.
fn launch_picker(app: &AndroidApp) {
    match with_env(app, |env, activity| {
        env.call_method(activity, "openDocument", "()V", &[])?;
        Ok(())
    }) {
        Ok(()) => log::info!("launched document picker"),
        Err(e) => log::error!("launch picker failed: {e}"),
    }
}

/// True if the app holds all-files access (can browse the library by path).
fn has_all_files(app: &AndroidApp) -> bool {
    with_env(app, |env, activity| {
        Ok(env.call_method(activity, "hasAllFiles", "()Z", &[])?.z()?)
    })
    .unwrap_or(false)
}

/// True if the OS is in night (dark) mode — resolves the `System` theme. winit
/// doesn't surface the system theme on Android, so read the uiMode night flag via
/// JNI (available since API 29). Defaults to dark on failure (the historical look).
fn system_dark(app: &AndroidApp) -> bool {
    with_env(app, |env, activity| {
        Ok(env.call_method(activity, "isSystemDark", "()Z", &[])?.z()?)
    })
    .unwrap_or(true)
}

/// Open Settings so the user can grant all-files access.
fn request_all_files(app: &AndroidApp) {
    let _ = with_env(app, |env, activity| {
        env.call_method(activity, "requestAllFiles", "()V", &[])?;
        Ok(())
    });
}

/// Take the picked content:// URI (string) if one has arrived, else None.
fn take_picked_uri(app: &AndroidApp) -> Option<String> {
    with_env(app, |env, activity| {
        let res = env.call_method(activity, "takePickedUri", "()Ljava/lang/String;", &[])?;
        let obj = res.l()?;
        if obj.is_null() {
            return Ok(None);
        }
        let s: String = env.get_string(&JString::from(obj))?.into();
        Ok(Some(s))
    })
    .ok()
    .flatten()
}

/// Open a content:// URI to an owned file descriptor (-1 on failure).
fn open_fd(app: &AndroidApp, uri: &str) -> i32 {
    with_env(app, |env, activity| {
        let juri = env.new_string(uri)?;
        let res = env.call_method(
            activity,
            "openFd",
            "(Ljava/lang/String;)I",
            &[JValue::Object(&juri)],
        )?;
        Ok(res.i()?)
    })
    .unwrap_or(-1)
}

/// Hold (true) / release (false) the KEEP_SCREEN_ON window flag, via the
/// `YoshActivity` JNI bridge. Deliberately NOT `AndroidApp::set_window_flags`:
/// that wrapper holds android-activity's glue mutex across
/// `ANativeActivity_setWindowFlags`, and calling it from the event-loop thread
/// mid-dispatch deadlocks against the UI thread's lifecycle callbacks on some
/// ROMs (froze the whole app on ZUI / Android 16 — the Java main thread futex-
/// waited in `NativeActivity.onPause` while our loop never returned from the
/// flags call). The bridge just `runOnUiThread`s the flag change, fire-and-forget.
fn set_keep_screen_on(app: &AndroidApp, on: bool) {
    let _ = with_env(app, |env, activity| {
        env.call_method(activity, "setKeepScreenOn", "(Z)V", &[JValue::Bool(on as u8)])?;
        Ok(())
    });
}

/// Hide (true) / show (false) the Android status + navigation bars. The window
/// stays edge-to-edge; chrome is inset by `status_bar_height` while bars show.
fn set_immersive(app: &AndroidApp, immersive: bool) {
    let _ = with_env(app, |env, activity| {
        env.call_method(
            activity,
            "setImmersive",
            "(Z)V",
            &[JValue::Bool(immersive as u8)],
        )?;
        Ok(())
    });
}

/// Status-bar height in px (0 if unknown) — a platform constant used to inset the
/// library / empty-state chrome so the bars don't cover it.
fn status_bar_height(app: &AndroidApp) -> i32 {
    with_env(app, |env, activity| {
        Ok(env.call_method(activity, "statusBarHeight", "()I", &[])?.i()?)
    })
    .unwrap_or(0)
}

/// Run a closure with an attached `JNIEnv` and the `YoshActivity` instance,
/// clearing any pending Java exception afterwards.
fn with_env<T>(
    app: &AndroidApp,
    f: impl FnOnce(&mut jni::JNIEnv, &JObject) -> jni::errors::Result<T>,
) -> jni::errors::Result<T> {
    let vm = unsafe { JavaVM::from_raw(app.vm_as_ptr().cast()) }?;
    let mut env = vm.attach_current_thread()?;
    let activity = unsafe { JObject::from_raw(app.activity_as_ptr().cast()) };
    let r = f(&mut env, &activity);
    if env.exception_check().unwrap_or(false) {
        let _ = env.exception_clear();
    }
    r
}

/// Gather the info overlay's byte-derived metadata for page `index`: format,
/// colour profile, compressed size and mtime. A free function over the bare source
/// because it runs on a **background** thread — `read_page` and `modified` are disk
/// I/O, and on an archive still decompressing `read_page` blocks until that entry
/// lands, which would freeze the UI for as long as the extraction takes.
fn page_meta(src: &dyn PageSource, index: usize) -> Vec<(String, String)> {
    let modified = src.modified(index).unwrap_or_else(|| "—".to_string());
    let (size, fmt, color) = match src.read_page(index).ok() {
        Some(b) => {
            let detail = yosh_engine::meta::probe(&b).2;
            let color = yosh_engine::icc::extract_icc(&b)
                .as_deref()
                .and_then(yosh_engine::icc::describe)
                .unwrap_or_else(|| "—".to_string());
            (yosh_engine::meta::human_size(b.len() as u64), detail, color)
        }
        None => ("—".to_string(), "—".to_string(), "—".to_string()),
    };
    vec![
        ("Format".to_string(), fmt),
        ("Color".to_string(), color),
        ("Size".to_string(), size),
        ("Modified".to_string(), modified),
    ]
}

impl App {
    /// Start an off-thread series scan of the library root (results land via
    /// `series_rx` in render).
    fn kick_series_scan(&mut self) {
        self.series_pending = true;
        spawn_library_scan(
            self.lib_root.clone(),
            self.series_tx.clone(),
            self.frame_waker.clone(),
        );
        self.window.request_redraw();
    }

    /// List `lib_dir` for the "choose a library root" browser on a background
    /// thread (`read_dir` + a stat per entry — no different from any other storage
    /// walk on a phone). The result is tagged with its directory, so a listing the
    /// user has already navigated away from is discarded.
    fn kick_browse_scan(&mut self) {
        self.browse_pending = true;
        self.lib_entries.clear();
        let dir = self.lib_dir.clone();
        let tx = self.browse_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let entries = scan_dir(&dir);
            let _ = tx.send((dir, entries));
            if let Some(w) = &waker {
                w();
            }
        });
        self.window.request_redraw();
    }

    /// Refresh the heavy info-overlay metadata (Format / Color / Size / Modified)
    /// for the displayed page. A no-op unless the page changed since the last build.
    ///
    /// The values need the page bytes — disk I/O, and on an archive still
    /// decompressing a wait for that entry — so they are gathered **off-thread**
    /// (`page_meta`), tagged with `(open_gen, page)`; the overlay shows "…"
    /// placeholders until the result lands in `render`'s drain. Mirrors the desktop
    /// `page_info` pattern.
    fn refresh_info_meta(&mut self, open_gen: u64) {
        let Some(src) = &self.reader.source else {
            self.info_for = None;
            self.info_meta.clear();
            return;
        };
        let len = src.len();
        let idx = view_pages(self.reader.layout, self.reader.index, len, self.reader.spread_offset).0;
        if self.info_for == Some(idx) {
            return;
        }
        self.info_for = Some(idx);
        self.info_meta = ["Format", "Color", "Size", "Modified"]
            .into_iter()
            .map(|k| (k.to_string(), "…".to_string()))
            .collect();
        let src = src.clone();
        let tx = self.info_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let _ = tx.send((open_gen, idx, page_meta(src.as_ref(), idx)));
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Build the image-info overlay lines from the current page + reader state
    /// (the Android take on the desktop's "I" overlay). The heavy metadata
    /// (`info_meta`) is refreshed separately via `refresh_info_meta` (gated).
    fn build_info(&self) -> Vec<(String, String)> {
        let Some(src) = &self.reader.source else {
            return Vec::new();
        };
        let len = src.len();
        let idx = view_pages(self.reader.layout, self.reader.index, len, self.reader.spread_offset).0;
        let mut lines = vec![
            ("Book".to_string(), self.book_title.clone()),
            ("File".to_string(), src.name(idx).to_string()),
            ("Page".to_string(), format!("{} / {}", idx + 1, len)),
            (
                "Screen".to_string(),
                format!("{} × {}", self.config.width, self.config.height),
            ),
        ];
        // Format / Color / Size / Modified — read from the page bytes, refreshed
        // only on a page change (see `refresh_info_meta`) so this stays per-frame cheap.
        lines.extend(self.info_meta.iter().cloned());
        if let Some(p) = self.reader.cache.get(idx) {
            lines.push(("Source".to_string(), format!("{} × {}", p.src_w, p.src_h)));
            lines.push(("Decoded".to_string(), format!("{} × {}", p.w, p.h)));
            lines.push((
                "Resize".to_string(),
                if p.lq {
                    "LQ (fast bilinear)".to_string()
                } else {
                    p.path.label().to_string()
                },
            ));
            // Single-resize invariant readout: the GPU should sample 1:1.
            let gpu = match self.reader.page_target_h(idx).cmp(&p.target_h) {
                std::cmp::Ordering::Equal => "1:1",
                std::cmp::Ordering::Greater => "↑ upscale",
                std::cmp::Ordering::Less => "↓ downscale",
            };
            lines.push(("GPU".to_string(), gpu.to_string()));
        } else {
            lines.push(("Decoded".to_string(), "decoding…".to_string()));
        }
        lines.push((
            "Zoom".to_string(),
            format!("{:.0}%", self.reader.effective_zoom_pct()),
        ));
        lines.push(("Fit".to_string(), self.reader.fit.label().to_string()));
        // For Auto, show the resolved layout too (e.g. "auto · spread").
        let layout_str = match self.layout_mode {
            LayoutMode::Auto => format!("auto · {}", self.reader.layout.label()),
            m => m.label().to_string(),
        };
        lines.push(("Layout".to_string(), layout_str));
        lines.push((
            "Direction".to_string(),
            self.reader.direction.label().to_string(),
        ));
        // LQ preview tier: fill progress + what's on screen for this page.
        let showing = if self.reader.cache.contains(idx) {
            "HQ"
        } else if self.reader.lq_cache.contains(idx) {
            "LQ preview"
        } else {
            "—"
        };
        lines.push((
            "LQ tier".to_string(),
            format!("{}/{} · {}", self.reader.lq_cache.len(), len, showing),
        ));
        lines
    }

    /// Flag the viewing options as changed. The Shell picks this up after the frame
    /// and debounces the write (`flush_saves`), reading the values back off the App
    /// then — so a run of toggles costs one atomic rewrite of `view.txt`, not one
    /// per tap.
    fn persist_view(&mut self) {
        self.view_dirty = true;
    }

    /// Apply the performance profile to the live reader — **no book reopen**. The
    /// budget's fields are all runtime-settable: the window sizes are plain fields,
    /// both caches and the texture pool can be re-capped in place (evicting down
    /// keeps the pages nearest the read position, so nothing on screen flashes),
    /// and the worker count is the one thing that needs a new pool. Rebuilding the
    /// pool is cheap and hitch-free because teardown is signal-only and the caches
    /// are deliberately **kept** — the current page stays decoded and on screen
    /// while the new pool refills around it.
    fn apply_perf(&mut self) {
        let tier = self.perf.tier().unwrap_or(self.auto_tier);
        let b = Budget::for_tier(tier, self.mem_budget_mb, self.cpus);
        log::info!("perf {} → tier {tier:?}, budget: {b:?}", self.perf.label());
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
        // Nothing open (library view): the fields above are enough — the pool is
        // built with `reader.workers` when the next book opens.
        let Some(src) = self.reader.source.clone() else {
            return;
        };
        let pool = DecodePool::new(
            src,
            self.ctx.device.clone(),
            self.ctx.queue.clone(),
            self.reader.tex_pool.clone(),
            b.workers,
        );
        // A fresh pool starts wakerless and with empty queues: re-install the wake
        // callback, then force a full rebuild of both the job list and the
        // thumbnail tail (`JobsKey` doesn't capture budget fields, so without this
        // the memoized key would suppress the very prefetch that refills the pool).
        pool.set_waker(self.reader.waker.clone());
        self.reader.pool = Some(pool);
        self.reader.invalidate_jobs();
        self.reader.prefetch();
    }

    /// Resolve the layout mode against the current viewport and, if it differs from
    /// the engine's concrete layout, switch — re-anchoring the read position into the
    /// new view and re-prefetching. This is what makes `Auto` follow device rotation;
    /// for fixed Single/Spread it's a per-frame no-op. Does NOT persist (the *mode*
    /// is unchanged — only its orientation-dependent resolution).
    fn apply_resolved_layout(&mut self) {
        let desired = self.layout_mode.resolve(self.config.width, self.config.height);
        if desired != self.reader.layout {
            self.reader.layout = desired;
            self.reader.index =
                view_start(desired, self.reader.index, self.reader.spread_offset);
            self.reader.prefetch();
        }
    }

    fn render(
        &mut self,
        has_files: bool,
        status_bar_px: i32,
        opening: Option<&str>,
        open_gen: u64,
        lib: LibCtx<'_>,
    ) -> FrameReqs {
        self.frame_no += 1;
        // Resolved chrome theme this frame: drives the page-letterbox clear color
        // (below) and egui's visuals (set at the top of the egui run). The seekbar's
        // hardcoded colors read it back off `ui.visuals().dark_mode`.
        let light = !self.theme.is_dark(self.system_dark);
        self.reader.viewport = Viewport {
            w: self.config.width,
            h: self.config.height,
        };
        // Resolve Auto → concrete layout from the current orientation *before* the
        // decode view / prefetch / build-quads below read `reader.layout`, so a
        // rotation switches this same frame (no one-frame flash of the old layout).
        self.apply_resolved_layout();
        // Drain finished decodes into the right cache (full-res → cache, LQ-tier
        // thumbnails → lq_cache) and record failures.
        self.reader.drain_pool();
        self.reader.update_decode_view();
        // Scroll mode: keep the (anchor, top_offset) valid as page heights resolve,
        // before prefetch reads the visible window.
        if self.reader.scroll_mode {
            self.reader.normalize();
        }
        self.reader.prefetch();

        let quads = if self.reader.scroll_mode {
            self.reader.build_scroll_quads()
        } else {
            self.reader.build_quads()
        };
        let anim_t = self.anim_origin.elapsed();
        // Did this frame draw a free-running animation (GIF/WebP)? The end-of-frame
        // redraw guard keeps the loop alive while one is on screen — without this,
        // animations only advance on input events.
        let mut drew_live_anim = false;
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.reader.page_texture(q.page_index).map(|t| {
                    drew_live_anim |= t.is_animation();
                    let view = t.view_at(anim_t);
                    self.page_pipeline.prepare_quad(
                        &self.ctx.device,
                        &self.ctx.queue,
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

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.ctx.device, &self.config);
                self.window.request_redraw();
                return FrameReqs::default();
            }
            _ => {
                self.window.request_redraw();
                return FrameReqs::default();
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("page"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Letterbox behind pages: white in light/e-ink mode (a big dark
                        // fill is unusable on a reflective panel), else the dark #202020.
                        load: wgpu::LoadOp::Clear(if light {
                            wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
                        } else {
                            wgpu::Color {
                                r: 32.0 / 255.0,
                                g: 32.0 / 255.0,
                                b: 32.0 / 255.0,
                                a: 1.0,
                            }
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
                pass.set_pipeline(&self.page_pipeline.pipeline);
                for bg in &page_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
        }
        // Drain decoded covers into egui textures, and any finished series scan.
        while let Ok((path, img)) = self.thumb_rx.try_recv() {
            self.covers_inflight = self.covers_inflight.saturating_sub(1);
            // A cover that wouldn't decode (unreadable or empty archive) stays in
            // `queued_covers`, so it isn't retried every frame — it just never
            // gets a thumbnail.
            let Some(img) = img else { continue };
            let handle =
                self.egui_ctx
                    .load_texture(path.to_string_lossy(), img, egui::TextureOptions::default());
            self.thumb_used.insert(path.clone(), self.frame_no);
            self.thumbs.insert(path, handle);
        }
        while let Ok(series) = self.series_rx.try_recv() {
            self.series = series;
            self.series_pending = false;
        }
        // Off-thread info-overlay metadata: both tags must still match — a result for
        // a book that has since been switched (generation) or for a page already
        // turned past (index) describes something other than what's on screen.
        while let Ok((generation, idx, meta)) = self.info_rx.try_recv() {
            if generation == open_gen && self.info_for == Some(idx) {
                self.info_meta = meta;
            }
        }
        // Off-thread browse listing, keyed by the directory it describes (a newer
        // navigation already replaced `lib_dir`, so its own result is what lands).
        while let Ok((dir, entries)) = self.browse_rx.try_recv() {
            if dir == self.lib_dir {
                self.lib_entries = entries;
                self.browse_pending = false;
            }
        }
        // egui chrome over the page: the library browser when open, else the seekbar.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let len = self.reader.source.as_ref().map(|s| s.len()).unwrap_or(0);
        let cur = self.reader.index;
        let library_view = self.library_view;
        let controls = self.controls;
        // Buffered (decode-ahead ready) page indices for the seekbar's cache bar.
        // Built only when the seekbar will actually draw (same condition as the
        // `seekbar()` arm below) — `lq_buffered` alone is an lq_cap-sized Vec, and
        // reading with the chrome hidden is the common case.
        let seekbar_visible = controls && !library_view && len > 0;
        let (buffered, lq_buffered): (Vec<usize>, Vec<usize>) = if seekbar_visible {
            (
                self.reader.cache.buffered_indices().collect(),
                self.reader.lq_cache.buffered_indices().collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let book_title = self.book_title.clone();
        let hints_visible = controls && self.controls_shown_at.elapsed().as_millis() < 1500;
        let show_options = self.show_options;
        let show_info = self.show_info;
        if show_info {
            self.refresh_info_meta(open_gen); // gated, off-thread → Format/Color/Size/Modified
        } else {
            self.info_for = None; // rebuild the metadata when the popup is next opened
        }
        let info_lines = if show_info { self.build_info() } else { Vec::new() };
        let rtl = self.reader.direction == Direction::Rtl;
        let cur_dir = self.reader.direction;
        // Resolved concrete layout (apply_resolved_layout ran at the top of render),
        // and the user's chosen mode for the popup's selection state.
        let cur_layout = self.reader.layout;
        let cur_layout_mode = self.layout_mode;
        let cur_fit = self.reader.fit;
        // The popup row is phrased as "Stretch small pages", so invert the flag.
        let cur_stretch_on = !self.reader.fit_no_upscale;
        let cur_transition = self.reader.transition_enabled;
        let cur_spine_on = self.spine_shadow_on;
        let cur_spine_strength = self.spine_shadow_strength;
        let cur_resume = self.resume_on_startup;
        let cur_scroll = self.reader.scroll_mode;
        let cur_theme = self.theme;
        let cur_perf = self.perf;
        let cur_rotation = self.reader.rotation;
        let drag_seam = self.reader.drag_seam();
        let cur_zoom = self.reader.zoom;
        // Path labels for the library header — only the library view shows them,
        // so don't format two paths per reading frame.
        let (lib_dir_str, lib_root_str) = if library_view {
            (
                self.lib_dir.display().to_string(),
                self.lib_root.display().to_string(),
            )
        } else {
            (String::new(), String::new())
        };
        let browse = self.lib_browse;
        let series_pending = self.series_pending;
        let browse_pending = self.browse_pending;
        // Browse mode (choose a library root) lists folders only — owned snapshot.
        let entries: Vec<(String, PathBuf)> = if library_view && browse {
            self.lib_entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Dir(p) => Some((name_of(p), p.clone())),
                    Entry::Comic(_) => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        // Series-view snapshot: per-volume read state + covers, per-series status
        // label + collapse state. Owned, so the closure borrows no `self`.
        let sections: Vec<SectionRow> = if library_view && !browse {
            self.series
                .iter()
                .map(|s| {
                    let expanded = !lib.collapsed.contains(&s.dir);
                    let states: Vec<VolState> = s
                        .volumes
                        .iter()
                        .map(|v| vol_state(lib.progress, lib.positions, &v.to_string_lossy()))
                        .collect();
                    let status = series_status(&states);
                    let volumes = if expanded {
                        s.volumes
                            .iter()
                            .zip(&states)
                            .map(|(v, st)| VolCell {
                                label: name_of(v),
                                thumb: self.thumbs.get(v).cloned(),
                                state: *st,
                                is_current: lib.current_key
                                    == Some(v.to_string_lossy().as_ref()),
                                path: v.clone(),
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    SectionRow {
                        dir: s.dir.clone(),
                        name: s.name.clone(),
                        expanded,
                        status,
                        volumes,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // "Recently read" cells (newest first), built like `sections` so the egui
        // closure borrows no `self`. Filtered to reopenable filesystem entries — the
        // only ones with a cover + a working tap-to-open (SAF one-offs are skipped).
        // The filter reads the background-computed set (`Shell::recents_ok`); it used
        // to `Path::exists` all 32 recents on every library frame.
        let recents_cells: Vec<VolCell> = if library_view && !browse {
            lib.recents
                .iter()
                .filter(|k| lib.recents_ok.contains(k.as_str()))
                .take(12)
                .map(|k| {
                    let path = PathBuf::from(k);
                    VolCell {
                        label: name_of(&path),
                        thumb: self.thumbs.get(&path).cloned(),
                        state: vol_state(lib.progress, lib.positions, k.as_str()),
                        is_current: lib.current_key == Some(k.as_str()),
                        path,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // Boundary prompt: lazily clear once its window lapses; visibility is
        // sampled ONCE here and reused for both the draw and the redraw guard
        // (the draw-time-consistency rule — see `Reader::animation_drawn`).
        if self
            .book_prompt
            .as_ref()
            .is_some_and(|p| p.at.elapsed().as_millis() >= BOOK_PROMPT_MS as u128)
        {
            self.book_prompt = None;
        }
        let prompt_card: Option<(i64, String, Option<egui::TextureHandle>, bool)> =
            if !library_view && len > 0 {
                self.book_prompt.as_ref().map(|p| {
                    let thumb = p.sibling.as_ref().and_then(|s| self.thumbs.get(s)).cloned();
                    (p.dir, p.title.clone(), thumb, p.sibling.is_some())
                })
            } else {
                None
            };
        let prompt_visible = prompt_card.is_some();
        let mut seek_to: Option<usize> = None;
        let mut reqs = FrameReqs::default();
        let mut go_up = false;
        let mut nav_to: Option<PathBuf> = None;
        let mut close_lib = false;
        let mut set_root = false;
        let mut enter_browse = false;
        let mut leave_browse = false;
        let mut rescan = false;
        let mut toggle_series: Option<PathBuf> = None;
        // Volume paths in expanded sections that were actually on screen this
        // frame: bumps their cover LRU stamps + lazily queues missing decodes.
        let mut visible_covers: Vec<PathBuf> = Vec::new();
        let mut open_options = false;
        let mut set_dir: Option<Direction> = None;
        let mut set_layout: Option<LayoutMode> = None;
        let mut set_fit: Option<FitMode> = None;
        let mut set_no_upscale: Option<bool> = None;
        let mut set_transition: Option<bool> = None;
        let mut set_spine_on: Option<bool> = None;
        let mut set_spine_strength: Option<f32> = None;
        let mut set_resume: Option<bool> = None;
        let mut set_scroll: Option<bool> = None;
        let mut set_theme: Option<ThemePref> = None;
        let mut set_perf: Option<PerfPref> = None;
        let mut toggle_offset = false;
        let mut rotate = false;
        let mut cycle_fit = false;
        let mut to_library = false;
        let mut page_nav: i64 = 0;
        let mut book_nav: i64 = 0;
        let mut open_info = false;
        let mut close_info = false;
        // Browse mode may climb anywhere (picking a new root); only the
        // filesystem root has no Up.
        let at_root = self.lib_dir.parent().is_none();
        // Library header "Resume <book>" button — drawn only when a book is open (the
        // library is the home; there's nothing to return to otherwise). It names the
        // book so it hints the destination; clip the title (by char count, since titles
        // are often CJK) so a long name can't push the other header buttons off-screen.
        let resume_label = (len > 0).then(|| {
            let total = self.book_title.chars().count();
            let t: String = self.book_title.chars().take(14).collect();
            let t = if total > 14 { format!("{t}…") } else { t };
            format!("{}  Resume {t}", egui_phosphor::fill::BOOK_OPEN)
        });
        // While the bars are shown the window is still full-bleed (NativeActivity
        // ignores fitSystemWindows), so inset the library's top chrome by the
        // status-bar height — otherwise the clock/icons cover the top buttons.
        let top_inset_pt = status_bar_px as f32 / self.egui_ctx.pixels_per_point();
        // `set_visuals` is persistent context state, so it only has to run when the
        // resolved theme actually flips (first frame, options toggle, OS night-mode
        // change) — not once per frame.
        let apply_visuals = self.applied_light != Some(light);
        self.applied_light = Some(light);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Theme the whole frame up front: standard widgets (library grid, popups,
            // empty-state card, series headers) follow this; the hand-painted chrome
            // (seekbar, letterbox) reads `dark_mode` back off the visuals.
            if apply_visuals {
                ctx.set_visuals(if light {
                    egui::Visuals::light()
                } else {
                    egui::Visuals::dark()
                });
            }
            if library_view {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(top_inset_pt);
                    // egui paints a fade-out gradient at scroll edges (default
                    // strength 0.5); on the cover grid it dims the thumbnails at the
                    // screen bottom, reading as a stray drop shadow. Turn it off.
                    ui.spacing_mut().scroll.fade.strength = 0.0;
                    if !has_files {
                        ui.add_space(40.0);
                        ui.label("Grant access to your files to browse your comics:");
                        if ui.button("Grant all-files access").clicked() {
                            reqs.grant = true;
                        }
                    } else if browse {
                        // Choose-a-library-root folder browser (folders only).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if ui.button(egui::RichText::new("← Back").size(18.0)).clicked() {
                                leave_browse = true;
                            }
                            if !at_root
                                && ui.button(egui::RichText::new("⬆ Up").size(18.0)).clicked()
                            {
                                go_up = true;
                            }
                            if ui
                                .button(egui::RichText::new("📌 Set as library").size(18.0))
                                .clicked()
                            {
                                set_root = true;
                            }
                        });
                        ui.label(egui::RichText::new(&lib_dir_str).size(13.0));
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (label, path) in &entries {
                                if ui
                                    .add_sized(
                                        [ui.available_width(), 44.0],
                                        egui::Button::new(
                                            egui::RichText::new(format!("📁 {label}")).size(16.0),
                                        ),
                                    )
                                    .clicked()
                                {
                                    nav_to = Some(path.clone());
                                }
                            }
                            if entries.is_empty() {
                                ui.label(
                                    egui::RichText::new(if browse_pending {
                                        "Scanning…"
                                    } else {
                                        "(no sub-folders)"
                                    })
                                    .weak(),
                                );
                            }
                        });
                    } else {
                        // Series view: collapsible sections, one horizontal
                        // cover row per series (Chunky-style).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if let Some(label) = &resume_label
                                && ui.button(egui::RichText::new(label).size(18.0)).clicked()
                            {
                                close_lib = true;
                            }
                            if ui
                                .button(egui::RichText::new("📂 Change library…").size(18.0))
                                .clicked()
                            {
                                enter_browse = true;
                            }
                            if ui.button(egui::RichText::new("Open file…").size(18.0)).clicked() {
                                reqs.open_picker = true;
                            }
                            if ui.button(egui::RichText::new("⟳").size(18.0)).clicked() {
                                rescan = true;
                            }
                        });
                        ui.label(egui::RichText::new(&lib_root_str).size(13.0));
                        ui.separator();
                        if sections.is_empty() {
                            ui.add_space(40.0);
                            ui.vertical_centered(|ui| {
                                ui.label(if series_pending {
                                    "Scanning library…"
                                } else {
                                    "No comics found here — use “Change library…” to pick your comics folder."
                                });
                            });
                        }
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            // "Recently read" row above the series (most-recent first).
                            if !recents_cells.is_empty() {
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new("Recently read").size(18.0).strong(),
                                );
                                visible_covers
                                    .extend(recents_cells.iter().map(|v| v.path.clone()));
                                egui::ScrollArea::horizontal()
                                    .id_salt("recents_row")
                                    .scroll_bar_visibility(
                                        egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                                    )
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            for v in &recents_cells {
                                                volume_cell(ui, v, &mut reqs, &mut close_lib);
                                            }
                                        });
                                    });
                                ui.add_space(8.0);
                                ui.separator();
                            }
                            for sec in &sections {
                                // Full-width tappable header: chevron + name
                                // left, status label right.
                                let (rect, resp) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), 40.0),
                                    egui::Sense::click(),
                                );
                                let chev = if sec.expanded {
                                    egui_phosphor::fill::CARET_DOWN
                                } else {
                                    egui_phosphor::fill::CARET_RIGHT
                                };
                                let strong = ui.visuals().strong_text_color();
                                let weak = ui.visuals().weak_text_color();
                                ui.painter().text(
                                    rect.left_center() + egui::vec2(6.0, 0.0),
                                    egui::Align2::LEFT_CENTER,
                                    format!("{chev}  {}", sec.name),
                                    egui::FontId::proportional(18.0),
                                    strong,
                                );
                                ui.painter().text(
                                    rect.right_center() - egui::vec2(6.0, 0.0),
                                    egui::Align2::RIGHT_CENTER,
                                    &sec.status,
                                    egui::FontId::proportional(13.0),
                                    weak,
                                );
                                if resp.clicked() {
                                    toggle_series = Some(sec.dir.clone());
                                }
                                if sec.expanded {
                                    // Lazy covers: only sections actually on
                                    // screen queue decodes / refresh their LRU.
                                    if ui.is_rect_visible(rect) {
                                        visible_covers
                                            .extend(sec.volumes.iter().map(|v| v.path.clone()));
                                    }
                                    egui::ScrollArea::horizontal()
                                        .id_salt(&sec.dir)
                                        .scroll_bar_visibility(
                                            egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                                        )
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                for v in &sec.volumes {
                                                    volume_cell(ui, v, &mut reqs, &mut close_lib);
                                                }
                                            });
                                        });
                                }
                                ui.add_space(8.0);
                            }
                        });
                    }
                });
            } else if let Some(name) = opening {
                // A book is being built on a worker thread: say so. Over the outgoing
                // page when switching books, over the clear color on a cold start.
                // egui's `Spinner` requests its own repaints, which the redraw guard's
                // `egui_animating` leg honors — so it actually spins while the loop
                // is otherwise idle.
                opening_card(ctx, name);
            } else if len == 0 {
                // No comic open: show the how-to-open helper instead of a blank screen.
                if self.logo.is_none()
                    && let Some(img) = decode_logo()
                {
                    self.logo = Some(ctx.load_texture(
                        "yosh_logo",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
                empty_state(ctx, self.logo.as_ref(), &mut reqs.open_library, &mut reqs.open_picker);
            } else if controls {
                seekbar(
                    ctx,
                    &book_title,
                    cur,
                    len,
                    rtl,
                    cur_layout == Layout::Spread,
                    cur_fit,
                    (cur_zoom - 1.0).abs() > 0.001,
                    &buffered,
                    &lq_buffered,
                    &mut seek_to,
                    &mut open_options,
                    &mut open_info,
                    &mut page_nav,
                    &mut book_nav,
                    &mut toggle_offset,
                    &mut cycle_fit,
                    &mut to_library,
                );
                if hints_visible {
                    zone_hints(ctx, rtl, cur_scroll);
                }
                if show_options {
                    options_popup(
                        ctx,
                        cur_dir,
                        cur_layout_mode,
                        cur_layout == Layout::Spread,
                        cur_fit,
                        cur_stretch_on,
                        cur_transition,
                        cur_spine_on,
                        cur_spine_strength,
                        cur_resume,
                        cur_scroll,
                        cur_theme,
                        cur_perf,
                        cur_rotation,
                        &mut set_dir,
                        &mut set_layout,
                        &mut set_fit,
                        &mut set_no_upscale,
                        &mut set_transition,
                        &mut set_spine_on,
                        &mut set_spine_strength,
                        &mut set_resume,
                        &mut set_scroll,
                        &mut set_theme,
                        &mut set_perf,
                        &mut toggle_offset,
                        &mut rotate,
                    );
                }
                if show_info {
                    info_popup(ctx, &info_lines, &mut close_info);
                }
            }
            // Next/prev-book boundary prompt — over the page, chrome or not.
            // (`prompt_card` is None in the library / empty state.)
            if let Some((dir, title, thumb, has_sibling)) = &prompt_card {
                book_prompt_card(ctx, *dir, rtl, title, thumb.as_ref(), *has_sibling, &mut book_nav);
            }
            // Page-flip drag drop shadow — over the page, chrome or not.
            if !library_view
                && let Some((frac, si)) = drag_seam
            {
                drag_shadow(ctx, frac, si);
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        // egui-driven animations (panel fade-ins, the empty-state card, button
        // feedback) request immediate repaints via `repaint_delay == 0`. The redraw
        // guard below must honor that or those animations freeze on their first
        // frame now that the loop genuinely idles between events.
        let egui_animating = full_output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay.is_zero());
        if let Some(p) = seek_to {
            self.reader.goto(p);
        }
        if set_root {
            // Browse mode picked a new library root: back to the series view and
            // rescan it. The root itself is persisted by the Shell's debounced flush
            // (it was previously only written on suspend).
            self.lib_root = self.lib_dir.clone();
            self.lib_browse = false;
            self.libroot_dirty = true;
            self.kick_series_scan();
        }
        if enter_browse {
            self.lib_browse = true;
            self.lib_dir = self.lib_root.clone();
            self.kick_browse_scan();
        }
        if leave_browse {
            self.lib_browse = false;
        }
        if rescan {
            self.kick_series_scan();
        }
        if go_up {
            // Browse mode may climb past the current root — that's how a root
            // *elsewhere* gets picked.
            if let Some(parent) = self.lib_dir.parent() {
                self.lib_dir = parent.to_path_buf();
                self.kick_browse_scan();
            }
        }
        if let Some(d) = nav_to {
            self.lib_dir = d;
            self.kick_browse_scan();
        }
        if close_lib {
            self.library_view = false;
        }
        reqs.toggle_series = toggle_series;
        // Lazy covers for the sections that were on screen: bump LRU stamps,
        // queue the missing ones, and evict the least-recently-drawn past the
        // cap (a big library would otherwise pin hundreds of MB of textures).
        let mut to_decode: Vec<PathBuf> = Vec::new();
        for p in visible_covers {
            self.thumb_used.insert(p.clone(), self.frame_no);
            if !self.thumbs.contains_key(&p) && self.queued_covers.insert(p.clone()) {
                to_decode.push(p);
            }
        }
        if !to_decode.is_empty() {
            // Count them as in flight only once the worker has them: a dead worker
            // (App torn down) would otherwise leave the redraw guard armed forever.
            let queued = to_decode.len();
            if self.cover_tx.send(to_decode).is_ok() {
                self.covers_inflight += queued;
            }
        }
        const THUMB_CAP: usize = 256;
        if self.thumbs.len() > THUMB_CAP {
            let mut by_age: Vec<(u64, PathBuf)> = self
                .thumbs
                .keys()
                .map(|k| (self.thumb_used.get(k).copied().unwrap_or(0), k.clone()))
                .collect();
            by_age.sort();
            for (_, k) in by_age.into_iter().take(self.thumbs.len() - THUMB_CAP) {
                self.thumbs.remove(&k); // dropping the handle frees the texture
                self.thumb_used.remove(&k);
                self.queued_covers.remove(&k); // allow a later re-decode
            }
        }
        if open_options {
            self.show_options = !self.show_options;
        }
        if open_info {
            self.show_info = !self.show_info;
        }
        if close_info {
            self.show_info = false;
        }
        if let Some(d) = set_dir {
            self.reader.direction = d;
            self.persist_view();
        }
        if let Some(m) = set_layout {
            self.layout_mode = m;
            self.persist_view();
            // Resolve + re-anchor immediately (no one-frame lag); for Auto this also
            // sets the concrete layout to match the current orientation.
            self.apply_resolved_layout();
        }
        if let Some(f) = set_fit {
            self.reader.fit = f;
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(v) = set_no_upscale {
            // Whether fits may upscale changes every small page's displayed size, so
            // the derived view state is re-derived with it (zoom clamp against the
            // native scale, pan clamp against the drawn box, scroll anchor against
            // the strip heights) before re-prefetching at the new decode targets —
            // `JobsKey` carries the flag, so the job list rebuilds exactly once.
            self.reader.fit_no_upscale = v;
            self.reader.clamp_zoom_native();
            self.reader.clamp_pan();
            if self.reader.scroll_mode {
                self.reader.normalize();
            }
            self.reader.prefetch();
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_transition {
            self.reader.transition_enabled = v;
            self.persist_view();
        }
        if let Some(v) = set_spine_on {
            self.spine_shadow_on = v;
            self.reader.spine_strength = if v { self.spine_shadow_strength } else { 0.0 };
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_spine_strength {
            self.spine_shadow_strength = v.clamp(0.0, 1.0);
            if self.spine_shadow_on {
                self.reader.spine_strength = self.spine_shadow_strength;
            }
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_resume {
            self.resume_on_startup = v;
            self.persist_view();
        }
        if let Some(v) = set_scroll {
            self.reader.scroll_mode = v;
            self.reader.top_offset = 0.0;
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(t) = set_theme {
            self.theme = t;
            self.persist_view();
            self.window.request_redraw(); // re-paint the chrome under the new theme
        }
        if let Some(p) = set_perf {
            self.perf = p;
            self.apply_perf(); // live: no reopen, current page stays on screen
            self.persist_view();
            self.window.request_redraw();
        }
        if cycle_fit {
            if (self.reader.zoom - 1.0).abs() > 0.001 {
                // Zoomed (in OR out of the fit scale): restore the active fit by
                // dropping the zoom/pan (reader.fit already holds the fit that was
                // active before zooming).
                self.reader.zoom = 1.0;
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
            } else {
                // Already fitted: advance to the next fit mode.
                self.reader.fit = self.reader.fit.cycle();
                self.persist_view();
            }
            self.reader.prefetch();
            self.window.request_redraw();
        }
        if toggle_offset {
            self.reader.spread_offset ^= 1;
            self.reader.index =
                view_start(self.reader.layout, self.reader.index, self.reader.spread_offset);
            self.reader.prefetch();
        }
        if rotate {
            self.reader.rotation = (self.reader.rotation + 1) % 4;
            // The rotated box has new bounds, so any prior pan is now out of range.
            self.reader.pan_x = 0.0;
            self.reader.pan_y = 0.0;
            self.reader.prefetch(); // re-decode at the rotation-aware 1:1 target
            self.window.request_redraw();
        }
        reqs.page_nav = page_nav;
        reqs.book_nav = book_nav;
        reqs.to_library = to_library;
        let ppp = self.egui_ctx.pixels_per_point();
        let primitives = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ppp,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
        }
        let user_cmds = self.egui_renderer.update_buffers(
            &self.ctx.device,
            &self.ctx.queue,
            &mut enc,
            &primitives,
            &screen,
        );
        {
            let mut egui_pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
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
            self.egui_renderer.render(&mut egui_pass, &primitives, &screen);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.ctx
            .queue
            .submit(user_cmds.into_iter().chain(std::iter::once(enc.finish())));
        self.window.pre_present_notify();
        frame.present();
        // On-demand redraw: idle on a settled, decoded reading page. Pending work
        // no longer needs a leg here — everything that produces a result off-thread
        // (the decode pool, the cover worker, the series scan) sends and then calls
        // the frame waker, which is `request_redraw`, so the frame that drains it
        // is scheduled by the landing itself. What's left are the things the loop
        // can only learn about by drawing again: an unsettled decode view (one
        // confirmation frame, self-limiting), timed chrome, and animations.
        // The library still gets a leg while results are outstanding — a cheap
        // belt-and-braces for the *first* frame after a scan/cover is queued, since
        // that producer may finish before we even reach the guard.
        // Nav/zoom/pan and egui repaints wake the loop from the event handler.
        if (self.library_view && (self.series_pending || self.covers_inflight > 0))
            || !self.reader.view_settled
            || hints_visible
            // An open is in flight. Its worker wakes the loop when it lands, but a
            // resume in the meantime replaces the window that waker belongs to — so
            // keep drawing until the result has actually been drained. Self-limiting
            // (the open finishes), and the spinner is animating anyway whenever it's
            // the thing on screen.
            || opening.is_some()
            // Draw-time flag, NOT transition_active()/drag_active(): re-sampling
            // the clock here can see the animation as just-expired even though
            // the frame above drew it mid-fade — freezing a half-faded ghost of
            // the outgoing page on screen (the decision must match the draw).
            || self.reader.animation_drawn() // page-turn / drag frame was drawn
            || prompt_visible // boundary prompt on screen (timed; same sample as draw)
            || drew_live_anim // a GIF/WebP is playing on screen
            || egui_animating // egui fade/feedback animation mid-flight
        {
            self.window.request_redraw();
        }
        reqs
    }
}

/// Last path component as a display string.
fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

/// Human-readable book name from a `current_key` (a filesystem path or a
/// content:// URI). Paths → basename; SAF document URIs (e.g.
/// `…/document/primary:Manga/Title.cbz`) → last decoded segment after `/` or `:`.
fn book_display_name(key: &str) -> String {
    if key.starts_with("content://") {
        // Minimal percent-decode of just the separators SAF uses, then take the
        // final component. Good enough for a display name; fall back to the raw
        // key if there's nothing better to show.
        let decoded = key.replace("%2F", "/").replace("%2f", "/").replace("%3A", ":");
        let last = decoded
            .rsplit(['/', ':'])
            .find(|s| !s.is_empty())
            .unwrap_or(key);
        return last.to_string();
    }
    name_of(Path::new(key))
}

fn ext_lower(p: &Path) -> Option<String> {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Is this a comic archive yosh can open on Android (RAR excluded)?
fn is_comic_archive(p: &Path) -> bool {
    match ext_lower(p).as_deref() {
        Some("cbz" | "zip" | "cb7" | "7z") => true,
        #[cfg(feature = "rar")]
        Some("cbr" | "rar") => true,
        _ => false,
    }
}

/// Build an engine page source from a comic path (archive or image folder).
fn build_source(path: &Path) -> Option<Arc<dyn PageSource>> {
    if path.is_dir() {
        FolderSource::new(path)
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn PageSource>)
    } else {
        match ext_lower(path).as_deref() {
            Some("cbz" | "zip") => ZipSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            Some("cb7" | "7z") => SevenzSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            #[cfg(feature = "rar")]
            Some("cbr" | "rar") => RarSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            _ => None,
        }
    }
}

/// Decode a comic's cover (page 0) to a small egui image, via the on-disk thumbnail
/// cache (a cache hit reads a tiny PNG instead of decoding the full first page).
fn decode_cover(
    path: &Path,
    cache_dir: Option<&Path>,
    resizer: &mut fast_image_resize::Resizer,
) -> Option<egui::ColorImage> {
    let rgba = yosh_engine::thumbcache::load_or_decode(cache_dir, path, 320, resizer, || {
        let src = build_source(path)?;
        (src.len() > 0).then(|| src.read_page(0).ok()).flatten()
    })?;
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

/// Decode the embedded yosh mascot logo for the "no comic open" card. The PNG is
/// transparent; the engine's decode premultiplies its alpha, so build the egui
/// image from premultiplied bytes (`from_rgba_unmultiplied` would double-multiply
/// and darken the anti-aliased edges).
fn decode_logo() -> Option<egui::ColorImage> {
    const LOGO: &[u8] = include_bytes!("../../yosh/assets/yosh.png");
    let mut resizer = fast_image_resize::Resizer::new();
    let decoded = yosh_engine::decode::decode_and_downscale(LOGO, 256, &mut resizer).ok()?;
    let rgba = yosh_engine::decode::to_rgba_image(decoded);
    Some(egui::ColorImage::from_rgba_premultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

/// Spawn the single, persistent cover-decode worker: batches of comic paths in
/// (`Sender<Vec<PathBuf>>`), decoded covers streamed back out through `tx`,
/// reading/writing the on-disk thumbnail cache as it goes. Each cover is sent
/// then followed by a `waker` call, so a library sitting idle draws the covers as
/// they arrive without the frame loop polling for them; a path that fails to
/// decode still sends (`None`) so the shell's in-flight count always drains.
///
/// One thread for the App's whole lifetime, reusing one `Resizer`, instead of a
/// thread per batch — scrolling a library used to mint a fresh thread (and a
/// fresh resizer) every few frames, all of them competing for the same cores as
/// the decode pool. It is detached and never joined: when the App is dropped its
/// `Sender` goes with it, `recv` fails, and the loop returns (a batch already in
/// progress finishes first, then the `tx` send fails and it returns anyway).
fn spawn_cover_worker(
    cache_dir: Option<PathBuf>,
    tx: std::sync::mpsc::Sender<(PathBuf, Option<egui::ColorImage>)>,
    waker: Option<Waker>,
) -> std::sync::mpsc::Sender<Vec<PathBuf>> {
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Vec<PathBuf>>();
    std::thread::spawn(move || {
        let mut resizer = fast_image_resize::Resizer::new();
        while let Ok(paths) = job_rx.recv() {
            for path in paths {
                let img = decode_cover(&path, cache_dir.as_deref(), &mut resizer);
                if tx.send((path, img)).is_err() {
                    return; // receiver gone (App torn down)
                }
                if let Some(w) = &waker {
                    w();
                }
            }
        }
    });
    job_tx
}

/// List a directory's sub-folders + comic archives, folders first, natural order.
fn scan_dir(dir: &Path) -> Vec<Entry> {
    let mut entries: Vec<Entry> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.is_dir() {
                Some(Entry::Dir(p))
            } else if is_comic_archive(&p) {
                Some(Entry::Comic(p))
            } else {
                None
            }
        })
        .collect();
    entries.sort_by(|a, b| {
        let key = |e: &Entry| match e {
            Entry::Dir(p) => (0, name_of(p).to_lowercase()),
            Entry::Comic(p) => (1, name_of(p).to_lowercase()),
        };
        let (ad, an) = key(a);
        let (bd, bn) = key(b);
        ad.cmp(&bd).then_with(|| natord::compare(&an, &bn))
    });
    entries
}

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

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use jni::objects::{JObject, JString, JValue};
use jni::JavaVM;
use winit::application::ApplicationHandler;
use winit::event::{Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::android::activity::{AndroidApp, WindowManagerFlags};
use winit::platform::android::EventLoopBuilderExtAndroid;
use winit::window::{Window, WindowId};

use yosh_engine::gpu::GpuContext;
use yosh_engine::layout::{view_pages, view_start, Layout};
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::DecodePool;
use yosh_engine::reader::{drag_commits, drag_dir, Budget, Direction, Reader, Viewport};
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, SevenzSource, ZipSource};
use yosh_engine::texpool::TexturePool;

/// Entry point android-activity calls on the native-activity thread.
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    log::info!("yosh-android starting");
    // A reader should keep the screen awake while a page is up.
    app.set_window_flags(WindowManagerFlags::KEEP_SCREEN_ON, WindowManagerFlags::empty());
    // Per-comic reading positions persist in the app's private dir.
    let pos_path = app.internal_data_path().map(|p| p.join("positions.tsv"));
    let positions = pos_path.as_deref().map(load_positions).unwrap_or_default();
    // Reading progress (furthest page seen + total) → the library's read states.
    let progress_path = app.internal_data_path().map(|p| p.join("progress.tsv"));
    let progress = progress_path.as_deref().map(load_progress).unwrap_or_default();
    // Series the user collapsed in the library (default expanded).
    let collapsed_path = app.internal_data_path().map(|p| p.join("collapsed.txt"));
    let collapsed = collapsed_path.as_deref().map(load_collapsed).unwrap_or_default();
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
        .unwrap_or((Direction::Rtl, LayoutMode::Single, FitMode::Window, true));
    let event_loop = EventLoop::builder()
        .with_android_app(app.clone())
        .build()
        .expect("build event loop");
    let mut shell = Shell {
        app: None,
        android_app: app,
        picker_pending: false,
        positions,
        pos_path,
        progress,
        progress_path,
        collapsed,
        collapsed_path,
        current_key: None,
        has_files: false,
        init_lib_dir,
        lib_dir_file,
        init_view,
        view_file,
        touches: HashMap::new(),
        gesture_start: None,
        page_drag: false,
        drag_samples: VecDeque::new(),
        pinch: None,
        applied_immersive: None,
        status_bar_px: None,
    };
    event_loop.run_app(&mut shell).expect("run event loop");
}

struct Shell {
    app: Option<App>,
    /// Kept for JNI into the `YoshActivity` Java bridge (vm + activity pointers).
    android_app: AndroidApp,
    /// A document-picker launch is awaiting its result.
    picker_pending: bool,
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
    /// Identity of the currently-open comic, for saving its position.
    current_key: Option<String>,
    /// Cached all-files-access state (refreshed on resume + library toggle).
    has_files: bool,
    /// The library dir to open the browser at (persisted across launches).
    init_lib_dir: PathBuf,
    lib_dir_file: Option<PathBuf>,
    /// Persisted viewing options (direction, layout, fit, page-turn animation) +
    /// where they live.
    init_view: (Direction, LayoutMode, FitMode, bool),
    view_file: Option<PathBuf>,
    /// Active touch points by finger id, for swipe / pinch-zoom / pan.
    touches: HashMap<u64, (f64, f64)>,
    /// Single-finger gesture start (for swipe-vs-tap on release).
    gesture_start: Option<(f64, f64)>,
    /// A single-finger move locked in as an interactive page drag (the page
    /// follows the finger; the engine renders it via `Reader::drag_update`).
    /// Locks once the motion is clearly horizontal; cleared on release/pinch.
    page_drag: bool,
    /// Recent `(time, x)` samples of the dragging finger (~last 100 ms), for the
    /// release velocity that decides flick-to-commit.
    drag_samples: VecDeque<(Instant, f64)>,
    /// Active pinch, captured at its start (see `Pinch`).
    pinch: Option<Pinch>,
    /// Last immersive state pushed to Android, to avoid redundant JNI each frame.
    applied_immersive: Option<bool>,
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
    thumb_rx: std::sync::mpsc::Receiver<(PathBuf, egui::ColorImage)>,
    thumb_tx: std::sync::mpsc::Sender<(PathBuf, egui::ColorImage)>,
    /// Covers already queued for decode (lazy, per visible series section).
    queued_covers: std::collections::HashSet<PathBuf>,
    /// Frame stamp a cover was last drawn (LRU for the `thumbs` cap — a big
    /// library would otherwise pin hundreds of MB of egui textures).
    thumb_used: HashMap<PathBuf, u64>,
    /// Monotonic render counter for `thumb_used`.
    frame_no: u64,
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
    /// The user's layout choice. `reader.layout` is the concrete `Single`/`Spread`
    /// this resolves to (orientation-dependent for `Auto`); this is the source of
    /// truth that persists. See `apply_resolved_layout`.
    layout_mode: LayoutMode,
    /// Where viewing options persist (mirrors Shell.view_file).
    view_file: Option<PathBuf>,
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
fn spawn_library_scan(root: PathBuf, tx: std::sync::mpsc::Sender<Vec<Series>>) {
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
        let _ = tx.send(out);
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
    /// Seekbar close button: close the current comic, back to the empty state.
    close_book: bool,
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
        );
        self.has_files = has_all_files(&self.android_app);
        log::info!(
            "all-files access: {} | scale {}",
            self.has_files,
            window.scale_factor()
        );
        // Resume from background (incl. returning from the picker): rebuild only
        // the surface against the existing device + reader — no re-decode.
        if self.app.is_some() {
            {
                let app = self.app.as_mut().unwrap();
                app.surface = app
                    .ctx
                    .instance
                    .create_surface(window.clone())
                    .expect("recreate surface");
                app.surface.configure(&app.ctx.device, &app.config);
                app.window = window.clone();
                app.window.request_redraw();
            }
            // Returning from the picker: Android delivers onActivityResult before
            // onResume, so the URI is ready by now. (RedrawRequested polls too, as
            // a backup in case the redraw loop isn't continuous.)
            if self.picker_pending {
                log::info!("resumed with picker_pending; polling");
                match take_picked_uri(&self.android_app) {
                    Some(uri) => {
                        self.picker_pending = false;
                        self.open_picked(&uri);
                    }
                    None => log::info!("resumed: no picked uri yet"),
                }
            }
            return;
        }

        // First launch: full build.
        let instance = GpuContext::create_instance();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let ctx = GpuContext::create(instance, Some(&surface));
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
        if let Ok(bytes) = std::fs::read("/system/fonts/NotoSansCJK-Regular.ttc") {
            fonts
                .font_data
                .insert("cjk".to_owned(), egui::FontData::from_owned(bytes).into());
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
        let budget = Budget::derive(device_mem_budget_mb(), cpus);
        log::info!("budget: {budget:?} ({cpus} cpus)");
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
            false, // scroll_mode: page-flip
            self.init_view.0, // direction (default RTL)
            0,
            true, // two_tier: LQ while seeking → HQ on settle
        );
        reader.transition_enabled = self.init_view.3; // page-turn animation (persisted; default on)
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (series_tx, series_rx) = std::sync::mpsc::channel();
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
            thumb_tx,
            queued_covers: std::collections::HashSet::new(),
            thumb_used: HashMap::new(),
            frame_no: 0,
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
            layout_mode: self.init_view.1,
            view_file: self.view_file.clone(),
        });
        // Nothing is open on first launch — the empty-state helper (see `render`)
        // explains how to open a comic. Restore the last comic? No: positions are
        // per-comic, and we don't persist which one was last open, so we land on the
        // empty state and let the user pick from the library / file picker.
        window.request_redraw();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Persist the current position + library dir before the OS may kill us.
        if let (Some(k), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(k, app.reader.index);
        }
        self.save_positions();
        if let (Some(f), Some(app)) = (self.lib_dir_file.clone(), self.app.as_ref()) {
            let _ = std::fs::write(f, app.lib_root.to_string_lossy().as_bytes());
        }
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
                let has_files = self.has_files;
                // Status-bar height (queried once; constant). Chrome is inset by it
                // while the bars are shown so they don't cover the top buttons.
                let status_px = *self
                    .status_bar_px
                    .get_or_insert_with(|| status_bar_height(&self.android_app));
                let reqs = match &mut self.app {
                    Some(app) => app.render(
                        has_files,
                        status_px,
                        LibCtx {
                            progress: &self.progress,
                            positions: &self.positions,
                            collapsed: &self.collapsed,
                            current_key: self.current_key.as_deref(),
                        },
                    ),
                    None => FrameReqs::default(),
                };
                // Keep the read-tracking map current (cheap; persisted with the
                // position saves) — covers seekbar jumps applied inside render.
                self.note_progress();
                if let Some(dir) = reqs.toggle_series {
                    if !self.collapsed.remove(&dir) {
                        self.collapsed.insert(dir);
                    }
                    self.save_collapsed();
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
                if reqs.close_book {
                    self.close_comic();
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
                        // Single finger, not zoomed → interactive page drag: the
                        // page follows the finger (Chunky-style), with the neighbor
                        // view revealed underneath. Locks once the motion is
                        // clearly horizontal, so taps and the seekbar stay intact.
                        let (dx, dy) = (x - sx, y - sy);
                        if !self.page_drag {
                            let w = self.app.as_ref().map(|a| a.config.width as f64).unwrap_or(1.0);
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
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&id);
                if self.touches.len() < 2 {
                    self.pinch = None;
                }
                if self.touches.is_empty() {
                    let was_drag = std::mem::take(&mut self.page_drag);
                    let start = self.gesture_start.take();
                    if was_drag {
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
                self.save_positions();
            }
        }
    }

    /// Open the library browser at the configured root, refreshing all-files
    /// access first (the browser shows the grant prompt if it's missing). Shared by
    /// the top-strip tap and the empty-state "Browse library" button.
    fn open_library(&mut self) {
        self.has_files = has_all_files(&self.android_app);
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
        if x < w * EDGE_ZONE as f64 {
            // Left edge: next in RTL, previous in LTR.
            self.flip(if rtl { 1 } else { -1 });
        } else if x > w * (1.0 - EDGE_ZONE as f64) {
            // Right edge: previous in RTL, next in LTR.
            self.flip(if rtl { -1 } else { 1 });
        } else {
            // A tap on the armed next/prev-book card confirms it. The card is
            // also a real egui button, but egui's consumed flag lags a frame on
            // touch presses (see the library note above), so a quick tap can
            // fall through to here — hit-test the card's centered rect
            // ourselves. (When egui *does* consume it, this never runs and the
            // button sets `book_nav`; either way the prompt clears before the
            // other path could fire, so a tap can't double-advance.)
            let confirm_dir = self.app.as_ref().and_then(|app| {
                let p = app.book_prompt.as_ref()?;
                let ppp = app.egui_ctx.pixels_per_point() as f64;
                let half_w = (BOOK_PROMPT_W_PT as f64 / 2.0 + 12.0) * ppp;
                let half_h = 200.0 * ppp;
                // Same edge anchoring as the draw (16pt inset from the side).
                let rtl = app.reader.direction == Direction::Rtl;
                let cx = if book_prompt_on_left(p.dir, rtl) {
                    (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                } else {
                    w - (16.0 + BOOK_PROMPT_W_PT as f64 / 2.0) * ppp
                };
                let inside = (x - cx).abs() < half_w && (y - h / 2.0).abs() < half_h;
                (inside
                    && p.sibling.is_some()
                    && p.at.elapsed().as_millis() < BOOK_PROMPT_MS as u128)
                    .then_some(p.dir)
            });
            if let Some(dir) = confirm_dir {
                if let Some(app) = self.app.as_mut() {
                    app.book_prompt = None;
                }
                self.open_sibling_book(dir);
                return;
            }
            if let Some(app) = self.app.as_mut() {
                // Center: toggle the reading chrome (also closes the options popup).
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
    }

    /// The previous/next comic in the current comic's folder (`dir` = -1/+1),
    /// natural-sorted like the library. `None` at the ends or for non-path
    /// (SAF-picked) sources, whose keys aren't filesystem paths.
    fn sibling_book_path(&self, dir: i64) -> Option<PathBuf> {
        let cur = self.current_key.clone()?;
        let cur_path = PathBuf::from(&cur);
        let parent = cur_path.parent()?;
        let comics: Vec<PathBuf> = scan_dir(parent)
            .into_iter()
            .filter_map(|e| match e {
                Entry::Comic(p) => Some(p),
                _ => None,
            })
            .collect();
        let i = comics.iter().position(|p| *p == cur_path)?;
        let j = i as i64 + dir;
        (0..comics.len() as i64)
            .contains(&j)
            .then(|| comics[j as usize].clone())
    }

    /// Open the previous/next comic in the current comic's folder (`dir` = -1/+1).
    /// No-op at the ends or for non-path sources.
    fn open_sibling_book(&mut self, dir: i64) {
        if let Some(p) = self.sibling_book_path(dir) {
            self.open_path(p);
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
        let sibling = self.sibling_book_path(dir);
        let title = sibling.as_deref().map(name_of).unwrap_or_default();
        if let Some(app) = self.app.as_mut() {
            if let Some(p) = &sibling
                && !app.thumbs.contains_key(p)
            {
                // Cover preview: same off-thread pipeline as the library; the
                // result lands via the per-frame `thumb_rx` drain and the card
                // picks it up while the prompt is still showing.
                spawn_cover_decode(vec![p.clone()], app.thumb_tx.clone());
            }
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
    fn open_path(&mut self, path: PathBuf) {
        match build_source(&path) {
            Some(src) => {
                log::info!("open {:?}: {} pages", path, src.len());
                self.open_comic(path.to_string_lossy().into_owned(), src);
            }
            None => log::error!("could not open {path:?}"),
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
        self.positions.insert(key, start);
        self.save_positions();
    }

    /// Close the current comic: persist its position, tear down the page source
    /// (the inverse of `attach_source`), and fall back to the empty "no comic open"
    /// state (`len == 0` renders `empty_state`).
    fn close_comic(&mut self) {
        if let (Some(k), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(k, app.reader.index);
        }
        self.save_positions();
        self.current_key = None;
        if let Some(app) = self.app.as_mut() {
            app.reader.pool = None;
            app.reader.source = None;
            app.reader.cache.clear();
            app.reader.lq_cache.clear();
            app.reader.failed.clear();
            app.book_title.clear();
            app.book_prompt = None;
            app.controls = false;
            app.show_options = false;
            app.show_info = false;
            app.window.request_redraw();
        }
    }

    fn save_positions(&self) {
        let Some(path) = &self.pos_path else { return };
        let mut out = String::new();
        for (k, v) in &self.positions {
            out.push_str(&format!("{v}\t{k}\n"));
        }
        let _ = std::fs::write(path, out);
        // Progress rides along: every position-save moment is also the right
        // durability point for the read-tracking map (so they can't drift).
        self.save_progress();
    }

    fn save_progress(&self) {
        let Some(path) = &self.progress_path else { return };
        let mut out = String::new();
        for (k, (furthest, total)) in &self.progress {
            out.push_str(&format!("{furthest}\t{total}\t{k}\n"));
        }
        let _ = std::fs::write(path, out);
    }

    fn save_collapsed(&self) {
        let Some(path) = &self.collapsed_path else { return };
        let mut out = String::new();
        for p in &self.collapsed {
            out.push_str(&format!("{}\n", p.display()));
        }
        let _ = std::fs::write(path, out);
    }

    /// Update the in-memory read-tracking entry for the open comic: the furthest
    /// page the reader has had on screen (1-based; the far page of a spread
    /// counts) and the volume's total. Called every rendered frame while reading
    /// — cheap map math; the file write happens with `save_positions`.
    fn note_progress(&mut self) {
        let (Some(key), Some(app)) = (self.current_key.clone(), self.app.as_ref()) else {
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
        let e = self.progress.entry(key).or_insert((0, 0));
        e.0 = e.0.max(seen);
        e.1 = len as u32;
    }

    /// Read the chosen content:// file's bytes off its descriptor and open it.
    fn open_picked(&mut self, uri: &str) {
        let fd = open_fd(&self.android_app, uri);
        if fd < 0 {
            log::warn!("openFd failed for {uri}");
            return;
        }
        // We own the fd (Java detachFd'd it); reading to end + drop closes it.
        let mut file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };
        let mut bytes = Vec::new();
        if let Err(e) = std::io::Read::read_to_end(&mut file, &mut bytes) {
            log::error!("read picked fd: {e}");
            return;
        }
        drop(file);
        match ZipSource::from_bytes(bytes) {
            Ok(src) => {
                log::info!("picked comic: {} pages", src.len());
                self.open_comic(uri.to_string(), Arc::new(src));
            }
            Err(e) => log::error!("from_bytes: {e}"),
        }
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
    reader.pool = Some(DecodePool::new(
        src.clone(),
        device.clone(),
        queue.clone(),
        reader.tex_pool.clone(),
        reader.workers,
    ));
    reader.cache.clear();
    reader.lq_cache.clear();
    reader.failed.clear();
    reader.index = start;
    reader.source = Some(src);
    reader.prefetch();
}

/// Load the persisted per-comic positions (one `index\tkey` line each).
/// Parse persisted viewing options ("dir,layout,fit,anim"); default RTL / single /
/// window / animation on.
fn load_view(path: &Path) -> (Direction, LayoutMode, FitMode, bool) {
    let (mut dir, mut lay, mut fit, mut anim) =
        (Direction::Rtl, LayoutMode::Single, FitMode::Window, true);
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
    }
    (dir, lay, fit, anim)
}

/// Persist viewing options as "dir,layout,fit,anim".
fn save_view(path: &Path, dir: Direction, lay: LayoutMode, fit: FitMode, anim: bool) {
    let d = if dir == Direction::Rtl { "rtl" } else { "ltr" };
    let l = lay.label();
    let f = match fit {
        FitMode::Width => "width",
        FitMode::Height => "height",
        FitMode::Actual => "actual",
        FitMode::Window => "window",
    };
    let a = if anim { "on" } else { "off" };
    let _ = std::fs::write(path, format!("{d},{l},{f},{a}"));
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

/// Memory (MB) the reader may use for its page cache + GPU textures, from
/// `/proc/meminfo`. Decoded pages and GPU textures are *native* allocations, not
/// bounded by the (small) Java heap, so a healthy slice of device RAM is fine —
/// more cache + a wider prefetch window means fewer stalls seeking heavy pages.
fn device_mem_budget_mb() -> u64 {
    let total = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("MemTotal:"))?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })
        .map(|kb| kb / 1024)
        .unwrap_or(4096);
    (total / 8).max(256)
}

/// Bottom seekbar: "page / total" + a draggable slider that requests a jump.
/// Hidden for a single-page source.
/// Image-info overlay as a closable centered popup (Android take on the "I"
/// overlay): check it, then ✕ to dismiss — not a persistent overlay.
fn info_popup(ctx: &egui::Context, lines: &[(String, String)], close: &mut bool) {
    egui::Area::new(egui::Id::new("info_popup"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -20.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
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
    close_book: &mut bool,
) {
    use egui_phosphor::fill as ph;
    if len <= 1 {
        return;
    }
    // A floating pill lifted off the bottom edge (the very edge collides with the
    // system gesture bar + is awkward to grab), fattened for touch.
    let sw = ctx.screen_rect().width();
    // Translucent panel so the page shows through behind the controls. Keep the
    // popup's color/stroke/rounding, just drop the fill's alpha.
    let frame = egui::Frame::popup(&ctx.style());
    let f = frame.fill;
    let frame = frame.fill(egui::Color32::from_rgba_unmultiplied(f.r(), f.g(), f.b(), 200));
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
                    let seeked = egui::Color32::from_rgb(96, 185, 255); // bright blue: handle + seeked rail
                    let unseeked = egui::Color32::from_rgb(120, 124, 130); // neutral grey: unseeked track
                    let ring = egui::Stroke::new(2.0, egui::Color32::WHITE);
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
                        let lq_tick = egui::Color32::from_rgba_unmultiplied(120, 165, 140, 55);
                        let hq_tick = egui::Color32::from_rgba_unmultiplied(120, 165, 140, 150);
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
                    // PerfectViewer-style palette: green nav arrows, an orange book for
                    // book nav, a red book for close, white utility icons.
                    let green = egui::Color32::from_rgb(124, 200, 80);
                    let white = egui::Color32::from_gray(235);
                    let red = egui::Color32::from_rgb(206, 74, 66);
                    let orange = egui::Color32::from_rgb(212, 140, 56);
                    let big = |ui: &mut egui::Ui, txt: &str, color: egui::Color32| {
                        ui.add_sized(
                            [bw, 46.0],
                            egui::Button::new(egui::RichText::new(txt).size(22.0).color(color)),
                        )
                        .clicked()
                    };
                    // A solid orange book with a glyph poking past its edge — used for
                    // prev/next-book (a green arrow off the left/right edge) and close (a
                    // red × off the top-right corner). The book colour is consistent for
                    // all three; offsetting the glyph past the edge (rather than dead
                    // centre) keeps both the book and the glyph legible. The glyph's side
                    // is positional (fixed); the action flips with direction at the call.
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
                    if big(ui, ph::GEAR, white) {
                        *open_options = true;
                    }
                    // Fit-mode button: when zoomed (in or out of the fit scale), the
                    // first tap drops the zoom to restore the active fit (icon dimmed
                    // to show no fit is active); otherwise it cycles fit modes. The icon
                    // reflects the current fit (1:1 stays as text).
                    let fit_color = if zoomed {
                        egui::Color32::from_white_alpha(120)
                    } else {
                        white
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
                    if spread && big(ui, ph::ARROWS_LEFT_RIGHT, white) {
                        *toggle_offset = true;
                    }
                    if big(ui, ph::ARROW_FAT_RIGHT, green) {
                        *page_nav = if rtl { -1 } else { 1 };
                    }
                    if book_overlay(ui, ph::ARROW_FAT_RIGHT, egui::vec2(11.0, 1.0), 16.0, green) {
                        *book_nav = if rtl { -1 } else { 1 };
                    }
                    if big(ui, ph::INFO, white) {
                        *open_info = true;
                    }
                    if book_overlay(ui, ph::X, egui::vec2(10.0, -8.0), 15.0, red) {
                        *close_book = true;
                    }
                });
            });
        });
}

/// Nothing-open helper: with no comic loaded the screen is otherwise just the dark
/// clear color, so explain how to open one. Centered card with the two ways in —
/// browse the library or pick a single file — plus the tap-the-top tip.
fn empty_state(ctx: &egui::Context, open_library: &mut bool, open_picker: &mut bool) {
    egui::Area::new(egui::Id::new("empty_state"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width((ctx.screen_rect().width() * 0.8).min(420.0));
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 12.0);
                    ui.label(egui::RichText::new("📖").size(56.0));
                    ui.label(egui::RichText::new("No comic open").strong().size(22.0));
                    ui.label(
                        egui::RichText::new("Open a comic to start reading.")
                            .size(15.0)
                            .color(egui::Color32::from_white_alpha(180)),
                    );
                    ui.add_space(4.0);
                    ui.spacing_mut().button_padding = egui::vec2(18.0, 12.0);
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("📚 Browse library").size(18.0),
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
                            .color(egui::Color32::from_white_alpha(140)),
                    );
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

/// Faint labels + outlines over the tap zones (shown briefly with the controls) so
/// the layout is discoverable. Painted, NOT laid out as widgets: an egui Area
/// registers under the pointer and makes the tap report "consumed", blocking the
/// edge/top zones. A background-layer painter does no hit-testing, so taps fall
/// through to nav.
fn zone_hints(ctx: &egui::Context, rtl: bool) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("zone_hints"),
    ));
    let rect = ctx.screen_rect();
    // Outline the tap zones (top strip + side edges). Two-tone — a dark stroke under
    // a lighter one — so the lines read on both white and dark manga pages.
    let dark = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(120));
    let light = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(200));
    let ty = rect.top() + rect.height() * TOP_ZONE;
    let lx = rect.left() + rect.width() * EDGE_ZONE;
    let rx = rect.right() - rect.width() * EDGE_ZONE;
    let yr = egui::Rangef::new(ty, rect.bottom());
    for s in [dark, light] {
        painter.hline(rect.x_range(), ty, s);
        painter.vline(lx, yr, s);
        painter.vline(rx, yr, s);
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
        "📚 Library",
    );
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

/// Viewing-options popup (from the seekbar gear): reading direction, page layout,
/// fit. Records the chosen value into the `set_*` outs; the caller applies it.
#[allow(clippy::too_many_arguments)]
fn options_popup(
    ctx: &egui::Context,
    dir: Direction,
    layout: LayoutMode,
    effective_spread: bool,
    fit: FitMode,
    transition_on: bool,
    set_dir: &mut Option<Direction>,
    set_layout: &mut Option<LayoutMode>,
    set_fit: &mut Option<FitMode>,
    set_transition: &mut Option<bool>,
    toggle_offset: &mut bool,
) {
    egui::Area::new(egui::Id::new("view_options"))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -40.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().button_padding = egui::vec2(16.0, 10.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 10.0);

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

impl App {
    /// Start an off-thread series scan of the library root (results land via
    /// `series_rx` in render).
    fn kick_series_scan(&mut self) {
        self.series_pending = true;
        spawn_library_scan(self.lib_root.clone(), self.series_tx.clone());
        self.window.request_redraw();
    }

    /// Build the image-info overlay lines from the current page + reader state
    /// (the Android take on the desktop's "I" overlay).
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

    /// Write the current viewing options to disk (immediate, like positions).
    /// Persists the layout *mode* (so `Auto` saves as "auto", not whatever
    /// orientation it happened to resolve to).
    fn persist_view(&self) {
        if let Some(f) = &self.view_file {
            save_view(
                f,
                self.reader.direction,
                self.layout_mode,
                self.reader.fit,
                self.reader.transition_enabled,
            );
        }
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

    fn render(&mut self, has_files: bool, status_bar_px: i32, lib: LibCtx<'_>) -> FrameReqs {
        self.frame_no += 1;
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
        self.reader.prefetch();

        let quads = self.reader.build_quads();
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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 32.0 / 255.0,
                            g: 32.0 / 255.0,
                            b: 32.0 / 255.0,
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
                pass.set_pipeline(&self.page_pipeline.pipeline);
                for bg in &page_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
        }
        // Drain decoded covers into egui textures, and any finished series scan.
        while let Ok((path, img)) = self.thumb_rx.try_recv() {
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
        // egui chrome over the page: the library browser when open, else the seekbar.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let len = self.reader.source.as_ref().map(|s| s.len()).unwrap_or(0);
        let cur = self.reader.index;
        // Buffered (decode-ahead ready) page indices for the seekbar's cache bar.
        let buffered: Vec<usize> = self.reader.cache.buffered_indices().collect();
        let lq_buffered: Vec<usize> = self.reader.lq_cache.buffered_indices().collect();
        let library_view = self.library_view;
        let controls = self.controls;
        let book_title = self.book_title.clone();
        let hints_visible = controls && self.controls_shown_at.elapsed().as_millis() < 1500;
        let show_options = self.show_options;
        let show_info = self.show_info;
        let info_lines = if show_info { self.build_info() } else { Vec::new() };
        let rtl = self.reader.direction == Direction::Rtl;
        let cur_dir = self.reader.direction;
        // Resolved concrete layout (apply_resolved_layout ran at the top of render),
        // and the user's chosen mode for the popup's selection state.
        let cur_layout = self.reader.layout;
        let cur_layout_mode = self.layout_mode;
        let cur_fit = self.reader.fit;
        let cur_transition = self.reader.transition_enabled;
        let cur_zoom = self.reader.zoom;
        let lib_dir_str = self.lib_dir.display().to_string();
        let lib_root_str = self.lib_root.display().to_string();
        let browse = self.lib_browse;
        let series_pending = self.series_pending;
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
        let mut set_transition: Option<bool> = None;
        let mut toggle_offset = false;
        let mut cycle_fit = false;
        let mut close_book = false;
        let mut page_nav: i64 = 0;
        let mut book_nav: i64 = 0;
        let mut open_info = false;
        let mut close_info = false;
        // Browse mode may climb anywhere (picking a new root); only the
        // filesystem root has no Up.
        let at_root = self.lib_dir.parent().is_none();
        // While the bars are shown the window is still full-bleed (NativeActivity
        // ignores fitSystemWindows), so inset the library's top chrome by the
        // status-bar height — otherwise the clock/icons cover the top buttons.
        let top_inset_pt = status_bar_px as f32 / self.egui_ctx.pixels_per_point();
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if library_view {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(top_inset_pt);
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
                                ui.label(egui::RichText::new("(no sub-folders)").weak());
                            }
                        });
                    } else {
                        // Series view: collapsible sections, one horizontal
                        // cover row per series (Chunky-style).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if ui.button(egui::RichText::new("× Close").size(18.0)).clicked() {
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
                            for sec in &sections {
                                // Full-width tappable header: chevron + name
                                // left, status label right.
                                let (rect, resp) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), 40.0),
                                    egui::Sense::click(),
                                );
                                let chev = if sec.expanded { "▾" } else { "▸" };
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
            } else if len == 0 {
                // No comic open: show the how-to-open helper instead of a blank screen.
                empty_state(ctx, &mut reqs.open_library, &mut reqs.open_picker);
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
                    &mut close_book,
                );
                if hints_visible {
                    zone_hints(ctx, rtl);
                }
                if show_options {
                    options_popup(
                        ctx,
                        cur_dir,
                        cur_layout_mode,
                        cur_layout == Layout::Spread,
                        cur_fit,
                        cur_transition,
                        &mut set_dir,
                        &mut set_layout,
                        &mut set_fit,
                        &mut set_transition,
                        &mut toggle_offset,
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
            // Browse mode picked a new library root: back to the series view
            // and rescan it. (The root persists via `suspended`, as before.)
            self.lib_root = self.lib_dir.clone();
            self.lib_browse = false;
            self.kick_series_scan();
        }
        if enter_browse {
            self.lib_browse = true;
            self.lib_dir = self.lib_root.clone();
            self.lib_entries = scan_dir(&self.lib_dir);
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
                self.lib_entries = scan_dir(&self.lib_dir);
            }
        }
        if let Some(d) = nav_to {
            self.lib_dir = d;
            self.lib_entries = scan_dir(&self.lib_dir);
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
            spawn_cover_decode(to_decode, self.thumb_tx.clone());
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
        if let Some(v) = set_transition {
            self.reader.transition_enabled = v;
            self.persist_view();
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
        reqs.page_nav = page_nav;
        reqs.book_nav = book_nav;
        reqs.close_book = close_book;
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
        // On-demand redraw: idle on a settled, decoded reading page. Keep going
        // while the library is open (covers stream in / scrolling), the decode view
        // hasn't settled (resize/zoom re-decode), or the current page is still
        // decoding (not yet cached and not failed). Nav/zoom/pan and egui repaints
        // wake the loop via request_redraw in the event handler.
        // Keep drawing while the library is open, the view hasn't settled, or the
        // current page isn't HQ yet (missing, or only the LQ seek-tier — so it
        // sharpens to HQ once seeking stops).
        // Redraw-loop diagnostic: a rare, so-far-unreproducible post-resume state
        // kept the loop running on a static screen (~144 fps). While rendering is
        // continuous this logs which guard leg is responsible ~every 2 s; an idle
        // app logs nothing. Remove once the culprit is caught in logcat.
        if self.frame_no % 300 == 0 {
            log::info!(
                "redraw? lib={} unsettled={} not_hq={} lq={} hints={} anim={} prompt={} live={} egui={} surface={}x{}",
                self.library_view,
                !self.reader.view_settled,
                !self.reader.view_is_hq(),
                self.reader.lq_fill_pending(),
                hints_visible,
                self.reader.animation_drawn(),
                prompt_visible,
                drew_live_anim,
                egui_animating,
                self.config.width,
                self.config.height
            );
        }
        if self.library_view
            || !self.reader.view_settled
            || !self.reader.view_is_hq()
            || self.reader.lq_fill_pending()
            || hints_visible
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
    matches!(ext_lower(p).as_deref(), Some("cbz" | "zip" | "cb7" | "7z"))
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
            _ => None,
        }
    }
}

/// Decode a comic's cover (page 0) to a small egui image.
fn decode_cover(path: &Path, resizer: &mut fast_image_resize::Resizer) -> Option<egui::ColorImage> {
    let src = build_source(path)?;
    if src.len() == 0 {
        return None;
    }
    let bytes = src.read_page(0).ok()?;
    let decoded = yosh_engine::decode::decode_and_downscale(&bytes, 320, resizer).ok()?;
    let rgba = yosh_engine::decode::to_rgba_image(decoded);
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

/// Spawn a worker that decodes the given comics' covers and streams them back.
fn spawn_cover_decode(paths: Vec<PathBuf>, tx: std::sync::mpsc::Sender<(PathBuf, egui::ColorImage)>) {
    std::thread::spawn(move || {
        let mut resizer = fast_image_resize::Resizer::new();
        for path in paths {
            if let Some(img) = decode_cover(&path, &mut resizer) {
                if tx.send((path, img)).is_err() {
                    break;
                }
            }
        }
    });
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

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

use std::collections::HashMap;
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
use yosh_engine::reader::{Budget, Direction, Reader, Viewport};
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
        .unwrap_or((Direction::Rtl, LayoutMode::Single, FitMode::Window));
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
        current_key: None,
        has_files: false,
        init_lib_dir,
        lib_dir_file,
        init_view,
        view_file,
        touches: HashMap::new(),
        gesture_start: None,
        pinch: None,
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
    /// Identity of the currently-open comic, for saving its position.
    current_key: Option<String>,
    /// Cached all-files-access state (refreshed on resume + library toggle).
    has_files: bool,
    /// The library dir to open the browser at (persisted across launches).
    init_lib_dir: PathBuf,
    lib_dir_file: Option<PathBuf>,
    /// Persisted viewing options (direction, layout, fit) + where they live.
    init_view: (Direction, LayoutMode, FitMode),
    view_file: Option<PathBuf>,
    /// Active touch points by finger id, for swipe / pinch-zoom / pan.
    touches: HashMap<u64, (f64, f64)>,
    /// Single-finger gesture start (for swipe-vs-tap on release).
    gesture_start: Option<(f64, f64)>,
    /// Active pinch, captured at its start (see `Pinch`).
    pinch: Option<Pinch>,
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
    /// The dir whose covers were queued for decode (so we don't re-queue).
    thumb_dir: Option<PathBuf>,
    /// Reading chrome: seekbar shows when `controls`; the gear opens the
    /// viewing-options popup. The zone hints show only briefly after the controls
    /// are revealed (see `controls_shown_at`), so they don't clutter while reading.
    controls: bool,
    controls_shown_at: Instant,
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
    /// Seekbar page button: reading-order step (-1 prev / +1 next).
    page_nav: i64,
    /// Seekbar book button: open the prev/next sibling comic in the folder (-1/+1).
    book_nav: i64,
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
        // egui's bundled fonts have no CJK glyphs; add the system Noto Sans CJK as a
        // fallback so Japanese comic / file names render instead of tofu squares.
        if let Ok(bytes) = std::fs::read("/system/fonts/NotoSansCJK-Regular.ttc") {
            let mut fonts = egui::FontDefinitions::default();
            fonts
                .font_data
                .insert("cjk".to_owned(), egui::FontData::from_owned(bytes).into());
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(fam).or_default().push("cjk".to_owned());
            }
            egui_ctx.set_fonts(fonts);
        }
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
        let reader = Reader::new(
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
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
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
            thumb_dir: None,
            controls: true,
            controls_shown_at: Instant::now(),
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
        let egui_consumed = if let Some(app) = self.app.as_mut() {
            let resp = app.egui_state.on_window_event(&app.window, &event);
            if resp.repaint {
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
                let reqs = match self.app.as_mut() {
                    Some(app) => app.render(has_files),
                    None => FrameReqs::default(),
                };
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
                } else if self.touches.len() == 2 {
                    // Begin a pinch; cancel the single-finger gesture.
                    self.gesture_start = None;
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
                            app.reader.zoom = p.zoom0 * (d / p.dist0) as f32;
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
                    }
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&id);
                if self.touches.len() < 2 {
                    self.pinch = None;
                }
                if self.touches.is_empty() {
                    if let Some((sx, sy)) = self.gesture_start.take() {
                        if !library && !egui_consumed {
                            self.handle_gesture(sx, sy, x, y);
                        }
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

    /// A single-finger gesture ended: a horizontal swipe flips (only when not
    /// zoomed, where a drag is a pan instead), otherwise it's a tap.
    fn handle_gesture(&mut self, sx: f64, sy: f64, ex: f64, ey: f64) {
        let (w, zoom) = self
            .app
            .as_ref()
            .map(|a| (a.config.width as f64, a.reader.zoom))
            .unwrap_or((1.0, 1.0));
        let (dx, dy) = (ex - sx, ey - sy);
        let is_swipe = dx.abs() > w * 0.08 && dx.abs() > dy.abs() * 1.4;
        if is_swipe {
            if zoom <= 1.001 {
                let rtl = self
                    .app
                    .as_ref()
                    .map(|a| a.reader.direction == Direction::Rtl)
                    .unwrap_or(false);
                // Drag metaphor: LTR swipe-left = next; RTL swipe-right = next.
                let next = if rtl { dx > 0.0 } else { dx < 0.0 };
                self.flip(if next { 1 } else { -1 });
            }
            // else: a drag while zoomed was a pan, not a flip.
        } else {
            self.on_tap(sx, sy);
        }
    }

    /// Flip `dir` pages (resetting zoom/pan to fit) and persist the new position.
    fn flip(&mut self, dir: i64) {
        if let Some(app) = self.app.as_mut() {
            app.reader.step(dir);
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
            app.lib_dir = app.lib_root.clone();
            app.lib_entries = scan_dir(&app.lib_dir);
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
        } else if let Some(app) = self.app.as_mut() {
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

    /// Open the previous/next comic in the current comic's folder (`dir` = -1/+1),
    /// natural-sorted like the library. No-op at the ends or for non-path sources.
    fn open_sibling_book(&mut self, dir: i64) {
        let Some(cur) = self.current_key.clone() else {
            return;
        };
        let cur_path = PathBuf::from(&cur);
        let Some(parent) = cur_path.parent() else {
            return;
        };
        let comics: Vec<PathBuf> = scan_dir(parent)
            .into_iter()
            .filter_map(|e| match e {
                Entry::Comic(p) => Some(p),
                _ => None,
            })
            .collect();
        if let Some(i) = comics.iter().position(|p| *p == cur_path) {
            let j = i as i64 + dir;
            if (0..comics.len() as i64).contains(&j) {
                self.open_path(comics[j as usize].clone());
            }
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
            app.window.request_redraw();
        }
        self.current_key = Some(key.clone());
        self.positions.insert(key, start);
        self.save_positions();
    }

    fn save_positions(&self) {
        let Some(path) = &self.pos_path else { return };
        let mut out = String::new();
        for (k, v) in &self.positions {
            out.push_str(&format!("{v}\t{k}\n"));
        }
        let _ = std::fs::write(path, out);
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
/// Parse persisted viewing options ("dir,layout,fit"); default RTL / single / window.
fn load_view(path: &Path) -> (Direction, LayoutMode, FitMode) {
    let (mut dir, mut lay, mut fit) = (Direction::Rtl, LayoutMode::Single, FitMode::Window);
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
    }
    (dir, lay, fit)
}

/// Persist viewing options as "dir,layout,fit".
fn save_view(path: &Path, dir: Direction, lay: LayoutMode, fit: FitMode) {
    let d = if dir == Direction::Rtl { "rtl" } else { "ltr" };
    let l = lay.label();
    let f = match fit {
        FitMode::Width => "width",
        FitMode::Height => "height",
        FitMode::Actual => "actual",
        FitMode::Window => "window",
    };
    let _ = std::fs::write(path, format!("{d},{l},{f}"));
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
    cur: usize,
    len: usize,
    rtl: bool,
    spread: bool,
    buffered: &[usize],
    lq_buffered: &[usize],
    seek_to: &mut Option<usize>,
    open_options: &mut bool,
    open_info: &mut bool,
    page_nav: &mut i64,
    book_nav: &mut i64,
    toggle_offset: &mut bool,
) {
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
                ui.horizontal(|ui| {
                    // Right-pad the current page to the total's digit width in a
                    // monospace font, so "1 / 200" and "200 / 200" occupy the same
                    // width and the slider doesn't shift as the page number grows.
                    let digits = len.to_string().len();
                    ui.label(
                        egui::RichText::new(format!("{:>digits$} / {}", cur + 1, len))
                            .size(18.0)
                            .monospace(),
                    );
                    ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
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
                // One centered row of big touch buttons. The nav arrows are POSITIONAL
                // (left = leftward in reading); the action flips with direction. Outer
                // arrows = book, inner = page; ⚙ options, ℹ info, ↔ pairing (spread).
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let bw = 58.0;
                    let count = if spread { 7 } else { 6 };
                    let sp = ui.spacing().item_spacing.x;
                    let total = count as f32 * bw + (count - 1) as f32 * sp;
                    ui.add_space(((ui.available_width() - total) * 0.5).max(0.0));
                    let big = |ui: &mut egui::Ui, txt: &str| {
                        ui.add_sized([bw, 46.0], egui::Button::new(egui::RichText::new(txt).size(22.0)))
                            .clicked()
                    };
                    if big(ui, "⚙") {
                        *open_options = true;
                    }
                    if big(ui, "«") {
                        *book_nav = if rtl { 1 } else { -1 };
                    }
                    if big(ui, "‹") {
                        *page_nav = if rtl { 1 } else { -1 };
                    }
                    if spread && big(ui, "↔") {
                        *toggle_offset = true;
                    }
                    if big(ui, "›") {
                        *page_nav = if rtl { -1 } else { 1 };
                    }
                    if big(ui, "»") {
                        *book_nav = if rtl { -1 } else { 1 };
                    }
                    if big(ui, "ℹ") {
                        *open_info = true;
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
    set_dir: &mut Option<Direction>,
    set_layout: &mut Option<LayoutMode>,
    set_fit: &mut Option<FitMode>,
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
    /// Queue a background cover decode for the comics in the current library dir
    /// (once per dir; results stream back via `thumb_rx`).
    fn queue_covers(&mut self) {
        if self.thumb_dir.as_deref() == Some(self.lib_dir.as_path()) {
            return;
        }
        self.thumb_dir = Some(self.lib_dir.clone());
        let paths: Vec<PathBuf> = self
            .lib_entries
            .iter()
            .filter_map(|e| match e {
                Entry::Comic(p) if !self.thumbs.contains_key(p) => Some(p.clone()),
                _ => None,
            })
            .collect();
        if !paths.is_empty() {
            spawn_cover_decode(paths, self.thumb_tx.clone());
        }
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
            save_view(f, self.reader.direction, self.layout_mode, self.reader.fit);
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

    fn render(&mut self, has_files: bool) -> FrameReqs {
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
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.reader.page_texture(q.page_index).map(|t| {
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
        // Drain decoded covers into egui textures; queue any new ones for this dir.
        while let Ok((path, img)) = self.thumb_rx.try_recv() {
            let handle =
                self.egui_ctx
                    .load_texture(path.to_string_lossy(), img, egui::TextureOptions::default());
            self.thumbs.insert(path, handle);
        }
        if self.library_view {
            self.queue_covers();
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
        let lib_dir_str = self.lib_dir.display().to_string();
        // Snapshot the rows the closure needs (owned), so it borrows no `self`.
        let entries: Vec<(bool, String, PathBuf, Option<egui::TextureHandle>)> = self
            .lib_entries
            .iter()
            .map(|e| {
                let (is_dir, p) = match e {
                    Entry::Dir(p) => (true, p),
                    Entry::Comic(p) => (false, p),
                };
                let thumb = if is_dir { None } else { self.thumbs.get(p).cloned() };
                (is_dir, name_of(p), p.clone(), thumb)
            })
            .collect();
        let mut seek_to: Option<usize> = None;
        let mut reqs = FrameReqs::default();
        let mut go_up = false;
        let mut nav_to: Option<PathBuf> = None;
        let mut close_lib = false;
        let mut set_root = false;
        let mut open_options = false;
        let mut set_dir: Option<Direction> = None;
        let mut set_layout: Option<LayoutMode> = None;
        let mut set_fit: Option<FitMode> = None;
        let mut toggle_offset = false;
        let mut page_nav: i64 = 0;
        let mut book_nav: i64 = 0;
        let mut open_info = false;
        let mut close_info = false;
        let at_root = self.lib_dir == self.lib_root;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if library_view {
                egui::CentralPanel::default().show(ctx, |ui| {
                    if !has_files {
                        ui.add_space(40.0);
                        ui.label("Grant access to your files to browse your comics:");
                        if ui.button("Grant all-files access").clicked() {
                            reqs.grant = true;
                        }
                    } else {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if !at_root
                                && ui.button(egui::RichText::new("⬆ Up").size(18.0)).clicked()
                            {
                                go_up = true;
                            }
                            if ui.button(egui::RichText::new("× Close").size(18.0)).clicked() {
                                close_lib = true;
                            }
                            if ui
                                .button(egui::RichText::new("📌 Set as library").size(18.0))
                                .clicked()
                            {
                                set_root = true;
                            }
                            if ui.button(egui::RichText::new("Open file…").size(18.0)).clicked() {
                                reqs.open_picker = true;
                            }
                        });
                        ui.label(egui::RichText::new(&lib_dir_str).size(13.0));
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            let cell = egui::vec2(168.0, 256.0);
                            let cols =
                                ((ui.available_width() / (cell.x + 10.0)).floor() as usize).max(1);
                            for row in entries.chunks(cols) {
                                ui.horizontal(|ui| {
                                    for (is_dir, label, path, thumb) in row {
                                        let clicked = ui
                                            .allocate_ui(cell, |ui| {
                                                ui.vertical(|ui| {
                                                    let r = if *is_dir {
                                                        ui.add_sized(
                                                            [cell.x, 200.0],
                                                            egui::Button::new(
                                                                egui::RichText::new("📁").size(80.0),
                                                            ),
                                                        )
                                                    } else if let Some(t) = thumb {
                                                        ui.add(egui::ImageButton::new(
                                                            egui::Image::new(t).fit_to_exact_size(
                                                                egui::vec2(cell.x, 200.0),
                                                            ),
                                                        ))
                                                    } else {
                                                        ui.add_sized(
                                                            [cell.x, 200.0],
                                                            egui::Button::new(
                                                                egui::RichText::new("…").size(28.0),
                                                            ),
                                                        )
                                                    };
                                                    ui.add_sized(
                                                        [cell.x, 34.0],
                                                        egui::Label::new(
                                                            egui::RichText::new(label.as_str())
                                                                .size(13.0),
                                                        )
                                                        .truncate(),
                                                    );
                                                    r.clicked()
                                                })
                                                .inner
                                            })
                                            .inner;
                                        if clicked {
                                            if *is_dir {
                                                nav_to = Some(path.clone());
                                            } else {
                                                reqs.open = Some(path.clone());
                                                close_lib = true;
                                            }
                                        }
                                    }
                                });
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
                    cur,
                    len,
                    rtl,
                    cur_layout == Layout::Spread,
                    &buffered,
                    &lq_buffered,
                    &mut seek_to,
                    &mut open_options,
                    &mut open_info,
                    &mut page_nav,
                    &mut book_nav,
                    &mut toggle_offset,
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
                        &mut set_dir,
                        &mut set_layout,
                        &mut set_fit,
                        &mut toggle_offset,
                    );
                }
                if show_info {
                    info_popup(ctx, &info_lines, &mut close_info);
                }
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        if let Some(p) = seek_to {
            self.reader.goto(p);
        }
        if set_root {
            self.lib_root = self.lib_dir.clone();
        }
        if go_up && self.lib_dir != self.lib_root {
            if let Some(parent) = self.lib_dir.parent() {
                self.lib_dir = parent.to_path_buf();
                self.lib_entries = scan_dir(&self.lib_dir);
            }
        }
        if let Some(d) = nav_to {
            // A folder of images is itself a comic; otherwise descend into it.
            if is_image_folder(&d) {
                reqs.open = Some(d);
                close_lib = true;
            } else {
                self.lib_dir = d;
                self.lib_entries = scan_dir(&self.lib_dir);
            }
        }
        if close_lib {
            self.library_view = false;
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
        if toggle_offset {
            self.reader.spread_offset ^= 1;
            self.reader.index =
                view_start(self.reader.layout, self.reader.index, self.reader.spread_offset);
            self.reader.prefetch();
        }
        reqs.page_nav = page_nav;
        reqs.book_nav = book_nav;
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
        if self.library_view
            || !self.reader.view_settled
            || !self.reader.view_is_hq()
            || self.reader.lq_fill_pending()
            || hints_visible
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

fn ext_lower(p: &Path) -> Option<String> {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Is this a comic archive yosh can open on Android (RAR excluded)?
fn is_comic_archive(p: &Path) -> bool {
    matches!(ext_lower(p).as_deref(), Some("cbz" | "zip" | "cb7" | "7z"))
}

/// Does this directory directly contain image files (i.e. is itself a comic)?
fn is_image_folder(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.path().is_file() && is_image_ext(&e.path())))
        .unwrap_or(false)
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

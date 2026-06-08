//! Android shell for yosh.
//!
//! Drives the reusable [`yosh_engine::reader::Reader`] in the winit frame loop —
//! the same poll → decode-view debounce → prefetch → build-quads → draw sequence
//! the desktop shell runs, minus the egui chrome. Renders real manga pages via
//! the engine's decode pool + `PagePipeline`.
//!
//! Storage: a tap in the top strip opens the system document picker (SAF) through
//! the `YoshActivity` Java bridge; the chosen `content://` file's bytes are read
//! off its descriptor and handed to `ZipSource::from_bytes`. (A test `.cbz` from
//! the app's external dir is also opened on launch, as a fallback for dev.)
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
use yosh_engine::layout::{view_start, Layout};
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::{DecodePool, Msg};
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
    let data_path = app.external_data_path();
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
        .unwrap_or((Direction::Rtl, Layout::Single, FitMode::Window, false));
    let event_loop = EventLoop::builder()
        .with_android_app(app.clone())
        .build()
        .expect("build event loop");
    let mut shell = Shell {
        app: None,
        data_path,
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
    /// The app's external files dir (…/Android/data/<pkg>/files), where a test
    /// comic is `adb push`ed for dev until the picker is the only path.
    data_path: Option<PathBuf>,
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
    /// Persisted viewing options (direction, layout, fit, jump) + where they live.
    init_view: (Direction, Layout, FitMode, bool),
    view_file: Option<PathBuf>,
    /// Active touch points by finger id, for swipe / pinch-zoom / pan.
    touches: HashMap<u64, (f64, f64)>,
    /// Single-finger gesture start (for swipe-vs-tap on release).
    gesture_start: Option<(f64, f64)>,
    /// Active pinch: (initial finger distance, zoom when it began).
    pinch: Option<(f64, f32)>,
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
    /// Where viewing options persist (mirrors Shell.view_file).
    view_file: Option<PathBuf>,
}

/// A library browser row: a folder to descend into (or open if it holds images),
/// or a comic archive to open.
enum Entry {
    Dir(PathBuf),
    Comic(PathBuf),
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
            self.init_view.1, // layout
            false,            // scroll_mode: page-flip
            self.init_view.3, // jump (seek mode)
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
            view_file: self.view_file.clone(),
        });
        // Dev fallback: open a pushed test comic if present (restores its position).
        if let Some(cbz) = self.data_path.as_ref().map(|p| p.join("test.cbz")) {
            if cbz.exists() {
                match ZipSource::new(&cbz) {
                    Ok(src) => self.open_comic(cbz.to_string_lossy().into_owned(), Arc::new(src)),
                    Err(e) => log::error!("open {cbz:?} failed: {e}"),
                }
            }
        }
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
                    if let Some(d) = self.touch_distance() {
                        let z = self.app.as_ref().map(|a| a.reader.zoom).unwrap_or(1.0);
                        self.pinch = Some((d, z));
                    }
                }
            }
            TouchPhase::Moved => {
                let prev = self.touches.insert(id, (x, y));
                if library {
                    return;
                }
                if let Some((d0, z0)) = self.pinch {
                    // Pinch → zoom (engine re-decodes HQ once it settles).
                    if d0 > 1.0 {
                        if let Some(d) = self.touch_distance() {
                            if let Some(app) = self.app.as_mut() {
                                app.reader.zoom = z0 * (d / d0) as f32;
                                app.reader.clamp_zoom_native();
                                app.window.request_redraw();
                            }
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

    /// Distance between the first two active touch points.
    fn touch_distance(&self) -> Option<f64> {
        let mut it = self.touches.values();
        let a = it.next()?;
        let b = it.next()?;
        Some(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
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
            self.has_files = has_all_files(&self.android_app);
            if let Some(app) = self.app.as_mut() {
                app.library_view = true;
                app.lib_dir = app.lib_root.clone();
                app.lib_entries = scan_dir(&app.lib_dir);
                app.window.request_redraw();
            }
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
    reader.failed.clear();
    reader.index = start;
    reader.source = Some(src);
    reader.prefetch();
}

/// Load the persisted per-comic positions (one `index\tkey` line each).
/// Parse persisted viewing options ("dir,layout,fit,seek"); default RTL / single
/// / window / step.
fn load_view(path: &Path) -> (Direction, Layout, FitMode, bool) {
    let (mut dir, mut lay, mut fit, mut jump) =
        (Direction::Rtl, Layout::Single, FitMode::Window, false);
    if let Ok(s) = std::fs::read_to_string(path) {
        let t: Vec<&str> = s.trim().split(',').collect();
        if t.first() == Some(&"ltr") {
            dir = Direction::Ltr;
        }
        if t.get(1) == Some(&"spread") {
            lay = Layout::Spread;
        }
        fit = match t.get(2) {
            Some(&"width") => FitMode::Width,
            Some(&"height") => FitMode::Height,
            Some(&"actual") => FitMode::Actual,
            _ => FitMode::Window,
        };
        if t.get(3) == Some(&"jump") {
            jump = true;
        }
    }
    (dir, lay, fit, jump)
}

/// Persist viewing options as "dir,layout,fit,seek".
fn save_view(path: &Path, dir: Direction, lay: Layout, fit: FitMode, jump: bool) {
    let d = if dir == Direction::Rtl { "rtl" } else { "ltr" };
    let l = if lay == Layout::Spread { "spread" } else { "single" };
    let f = match fit {
        FitMode::Width => "width",
        FitMode::Height => "height",
        FitMode::Actual => "actual",
        FitMode::Window => "window",
    };
    let j = if jump { "jump" } else { "step" };
    let _ = std::fs::write(path, format!("{d},{l},{f},{j}"));
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
#[allow(clippy::too_many_arguments)]
fn seekbar(
    ctx: &egui::Context,
    cur: usize,
    len: usize,
    rtl: bool,
    spread: bool,
    seek_to: &mut Option<usize>,
    open_options: &mut bool,
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
    egui::Area::new(egui::Id::new("seekbar"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -40.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width((sw * 0.85).min(960.0));
                ui.spacing_mut().interact_size.y = 30.0;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{} / {}", cur + 1, len)).size(18.0));
                    if ui.button(egui::RichText::new("⚙").size(20.0)).clicked() {
                        *open_options = true;
                    }
                    ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
                    // RTL: map so page 1 sits on the right and progress runs leftward.
                    let mut sv = if rtl { len - 1 - cur } else { cur };
                    if ui
                        .add(egui::Slider::new(&mut sv, 0..=len - 1).show_value(false))
                        .changed()
                    {
                        *seek_to = Some(if rtl { len - 1 - sv } else { sv });
                    }
                });
                // Button row. Arrows are POSITIONAL (left = leftward in reading); the
                // action they trigger flips with direction. Outer = book, inner = page.
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    let btn = |ui: &mut egui::Ui, txt: &str| {
                        ui.add_sized([54.0, 38.0], egui::Button::new(egui::RichText::new(txt).size(22.0)))
                            .clicked()
                    };
                    if btn(ui, "«") {
                        *book_nav = if rtl { 1 } else { -1 };
                    }
                    if btn(ui, "‹") {
                        *page_nav = if rtl { 1 } else { -1 };
                    }
                    if spread && btn(ui, "↔") {
                        *toggle_offset = true;
                    }
                    if btn(ui, "›") {
                        *page_nav = if rtl { -1 } else { 1 };
                    }
                    if btn(ui, "»") {
                        *book_nav = if rtl { -1 } else { 1 };
                    }
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
    // Outline the tap zones (top strip + side edges) so the layout reads at a glance.
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(110));
    let ty = rect.top() + rect.height() * TOP_ZONE;
    let lx = rect.left() + rect.width() * EDGE_ZONE;
    let rx = rect.right() - rect.width() * EDGE_ZONE;
    painter.hline(rect.x_range(), ty, stroke);
    painter.vline(lx, egui::Rangef::new(ty, rect.bottom()), stroke);
    painter.vline(rx, egui::Rangef::new(ty, rect.bottom()), stroke);
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
    layout: Layout,
    fit: FitMode,
    jump: bool,
    set_dir: &mut Option<Direction>,
    set_layout: &mut Option<Layout>,
    set_fit: &mut Option<FitMode>,
    set_jump: &mut Option<bool>,
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
                    if ui
                        .selectable_label(layout == Layout::Single, egui::RichText::new("Single").size(16.0))
                        .clicked()
                    {
                        *set_layout = Some(Layout::Single);
                    }
                    if ui
                        .selectable_label(layout == Layout::Spread, egui::RichText::new("Two-page").size(16.0))
                        .clicked()
                    {
                        *set_layout = Some(Layout::Spread);
                    }
                });
                // Shift which pages are paired together (e.g. so a cover sits alone).
                if layout == Layout::Spread
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

                ui.label(egui::RichText::new("Seek mode").strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(!jump, egui::RichText::new("Step").size(16.0))
                        .clicked()
                    {
                        *set_jump = Some(false);
                    }
                    if ui
                        .selectable_label(jump, egui::RichText::new("Jump (fast seek)").size(16.0))
                        .clicked()
                    {
                        *set_jump = Some(true);
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

    /// Write the current viewing options to disk (immediate, like positions).
    fn persist_view(&self) {
        if let Some(f) = &self.view_file {
            save_view(
                f,
                self.reader.direction,
                self.reader.layout,
                self.reader.fit,
                self.reader.jump,
            );
        }
    }

    fn render(&mut self, has_files: bool) -> FrameReqs {
        self.reader.viewport = Viewport {
            w: self.config.width,
            h: self.config.height,
        };
        // Drain finished decodes into the cache.
        if let Some(pool) = &self.reader.pool {
            for msg in pool.poll() {
                match msg {
                    Msg::Done { index, page } => {
                        self.reader.est_aspect = page.h as f32 / page.w as f32;
                        self.reader.cache.insert(index, page, self.reader.index);
                    }
                    Msg::Failed { index, error } => {
                        self.reader.failed.insert(index, error);
                    }
                }
            }
        }
        self.reader.update_decode_view();
        self.reader.prefetch();

        let quads = self.reader.build_quads();
        let anim_t = self.anim_origin.elapsed();
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.reader.cache.get(q.page_index).map(|t| {
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
        let library_view = self.library_view;
        let controls = self.controls;
        let hints_visible = controls && self.controls_shown_at.elapsed().as_millis() < 1500;
        let show_options = self.show_options;
        let rtl = self.reader.direction == Direction::Rtl;
        let cur_dir = self.reader.direction;
        let cur_layout = self.reader.layout;
        let cur_fit = self.reader.fit;
        let cur_jump = self.reader.jump;
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
        let mut set_layout: Option<Layout> = None;
        let mut set_fit: Option<FitMode> = None;
        let mut set_jump: Option<bool> = None;
        let mut toggle_offset = false;
        let mut page_nav: i64 = 0;
        let mut book_nav: i64 = 0;
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
                            if ui.button(egui::RichText::new("✕ Close").size(18.0)).clicked() {
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
            } else if controls {
                seekbar(
                    ctx,
                    cur,
                    len,
                    rtl,
                    cur_layout == Layout::Spread,
                    &mut seek_to,
                    &mut open_options,
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
                        cur_layout,
                        cur_fit,
                        cur_jump,
                        &mut set_dir,
                        &mut set_layout,
                        &mut set_fit,
                        &mut set_jump,
                        &mut toggle_offset,
                    );
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
        if let Some(d) = set_dir {
            self.reader.direction = d;
            self.persist_view();
        }
        if let Some(l) = set_layout {
            self.reader.layout = l;
            self.reader.index = view_start(l, self.reader.index, self.reader.spread_offset);
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(f) = set_fit {
            self.reader.fit = f;
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(j) = set_jump {
            self.reader.jump = j;
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

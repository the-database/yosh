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
use yosh_engine::layout::Layout;
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
            FitMode::Window,
            Layout::Single,
            false, // scroll_mode: page-flip
            false, // jump: step mode
            Direction::Ltr,
            0,
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
        // Let egui see the event first (seekbar drag); gate nav on what it consumes.
        let egui_consumed = if let Some(app) = self.app.as_mut() {
            app.egui_state.on_window_event(&app.window, &event).consumed
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
                self.flip(if dx < 0.0 { 1 } else { -1 });
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

    /// Tap-zones: top strip opens the library; otherwise left third = prev, rest = next.
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
        if y < h * 0.12 {
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
        // Page nav: left third = previous, rest = next.
        if x < w / 3.0 {
            self.flip(-1);
        } else {
            self.flip(1);
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
fn seekbar(ctx: &egui::Context, cur: usize, len: usize, seek_to: &mut Option<usize>) {
    if len <= 1 {
        return;
    }
    // A floating pill lifted off the bottom edge (the very edge collides with the
    // system gesture bar + is awkward to grab), fattened for touch.
    let sw = ctx.screen_rect().width();
    egui::Area::new(egui::Id::new("seekbar"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -48.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width((sw * 0.8).min(900.0));
                ui.spacing_mut().interact_size.y = 32.0;
                ui.horizontal(|ui| {
                    let mut p = cur;
                    ui.label(egui::RichText::new(format!("{} / {}", cur + 1, len)).size(18.0));
                    ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
                    if ui
                        .add(egui::Slider::new(&mut p, 0..=len - 1).show_value(false))
                        .changed()
                    {
                        *seek_to = Some(p);
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
            } else {
                seekbar(ctx, cur, len, &mut seek_to);
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
        // TODO(power): redraw on-demand instead of continuously — needs a reliable
        // "pending work" signal from the reader; a naive cache/settled check idled
        // before the first decode landed.
        self.window.request_redraw();
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

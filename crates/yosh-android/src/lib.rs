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
use std::path::PathBuf;
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
use yosh_engine::source::{PageSource, ZipSource};
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
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
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
        // Persist the current position before the OS may kill us.
        if let (Some(k), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(k, app.reader.index);
        }
        self.save_positions();
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
                phase: TouchPhase::Started,
                location,
                ..
            }) if !egui_consumed => self.on_tap(location.x, location.y),
            WindowEvent::RedrawRequested => {
                // A picker result may have landed while we were backgrounded.
                if self.picker_pending {
                    if let Some(uri) = take_picked_uri(&self.android_app) {
                        self.picker_pending = false;
                        self.open_picked(&uri);
                    }
                }
                if let Some(app) = self.app.as_mut() {
                    app.render();
                }
            }
            _ => {}
        }
    }
}

impl Shell {
    /// Tap-zones: top strip opens the picker; otherwise left third = prev, rest = next.
    fn on_tap(&mut self, x: f64, y: f64) {
        let Some((w, h)) = self
            .app
            .as_ref()
            .map(|a| (a.config.width as f64, a.config.height as f64))
        else {
            return;
        };
        if y < h * 0.12 {
            launch_picker(&self.android_app);
            self.picker_pending = true;
        } else if let Some(app) = self.app.as_mut() {
            if x < w / 3.0 {
                app.reader.step(-1);
            } else {
                app.reader.step(1);
            }
            app.window.request_redraw();
            let idx = app.reader.index;
            // Persist on each turn (tiny file, sub-ms) so position survives a hard
            // kill, not just a clean background.
            if let Some(k) = self.current_key.clone() {
                self.positions.insert(k, idx);
                self.save_positions();
            }
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
    egui::TopBottomPanel::bottom("seekbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(format!("{} / {}", cur + 1, len));
            let mut p = cur;
            // Span the slider across the rest of the bar.
            ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
            if ui
                .add(egui::Slider::new(&mut p, 0..=len - 1).show_value(false))
                .changed()
            {
                *seek_to = Some(p);
            }
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
    fn render(&mut self) {
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
        // egui chrome (seekbar) over the page, in the same encoder.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let len = self.reader.source.as_ref().map(|s| s.len()).unwrap_or(0);
        let cur = self.reader.index;
        let mut seek_to: Option<usize> = None;
        let full_output = self
            .egui_ctx
            .run(raw_input, |ctx| seekbar(ctx, cur, len, &mut seek_to));
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        if let Some(p) = seek_to {
            self.reader.goto(p);
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
    }
}

//! Android shell for yosh.
//!
//! Drives the reusable [`yosh_engine::reader::Reader`] in the winit frame loop —
//! the same poll → decode-view debounce → prefetch → build-quads → draw sequence
//! the desktop shell runs, minus the egui chrome. Renders real manga pages via
//! the engine's decode pool + `PagePipeline`, with tap-zones to flip pages.
//!
//! Storage (step 1): opens `test.cbz` from the app's external files dir, pushed
//! with `adb push`. The user-facing SAF picker (content:// → `from_bytes`) lands
//! next. The whole crate is Android-only; the desktop shell is `crates/yosh`.
#![cfg(target_os = "android")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
    let event_loop = EventLoop::builder()
        .with_android_app(app)
        .build()
        .expect("build event loop");
    let mut shell = Shell { app: None, data_path };
    event_loop.run_app(&mut shell).expect("run event loop");
}

struct Shell {
    app: Option<App>,
    /// The app's external files dir (…/Android/data/<pkg>/files), where a test
    /// comic is `adb push`ed until the SAF picker lands.
    data_path: Option<PathBuf>,
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
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
        );
        // Resume from background: rebuild only the surface against the existing
        // device + reader — no re-decode (the recreate_surface pattern).
        if let Some(app) = self.app.as_mut() {
            app.surface = app
                .ctx
                .instance
                .create_surface(window.clone())
                .expect("recreate surface");
            app.surface.configure(&app.ctx.device, &app.config);
            app.window = window.clone();
            window.request_redraw();
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
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        // A conservative per-app memory budget for now; a later step queries the
        // device's real heap class. Desktop-equivalent budget plumbing.
        let budget = Budget::derive(256, cpus);
        let tex_pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));
        let mut reader = Reader::new(
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
        self.open_test_comic(&mut reader, &ctx);

        window.request_redraw();
        self.app = Some(App {
            window,
            ctx,
            surface,
            config,
            page_pipeline,
            reader,
            anim_origin: Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                app.config.width = size.width.max(1);
                app.config.height = size.height.max(1);
                app.surface.configure(&app.ctx.device, &app.config);
                app.window.request_redraw();
            }
            // Tap-zones: left third = previous page, right two-thirds = next.
            WindowEvent::Touch(Touch {
                phase: TouchPhase::Started,
                location,
                ..
            }) => {
                if location.x < app.config.width as f64 / 3.0 {
                    app.reader.step(-1);
                } else {
                    app.reader.step(1);
                }
                app.window.request_redraw();
            }
            WindowEvent::RedrawRequested => app.render(),
            _ => {}
        }
    }
}

impl Shell {
    fn open_test_comic(&self, reader: &mut Reader, ctx: &GpuContext) {
        let Some(dp) = &self.data_path else {
            log::warn!("no external data path; nothing to open");
            return;
        };
        let cbz = dp.join("test.cbz");
        if !cbz.exists() {
            log::warn!("no test comic at {cbz:?} — `adb push` a .cbz there");
            return;
        }
        match ZipSource::new(&cbz) {
            Ok(src) => {
                let src: Arc<dyn PageSource> = Arc::new(src);
                log::info!("opened {:?}: {} pages", cbz, src.len());
                reader.pool = Some(DecodePool::new(
                    src.clone(),
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    reader.tex_pool.clone(),
                    reader.workers,
                ));
                reader.cache.clear();
                reader.index = 0;
                reader.source = Some(src);
                reader.prefetch();
            }
            Err(e) => log::error!("open {cbz:?} failed: {e}"),
        }
    }
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
                        // #202020 (non-sRGB surface → stored byte is value*255).
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
        self.ctx.queue.submit([enc.finish()]);
        self.window.pre_present_notify();
        frame.present();
        // TODO(power): redraw on-demand (only while decoding / unsettled) instead
        // of continuously — needs a reliable "pending work" signal from the reader;
        // a naive cache/settled check idled before the first decode landed.
        self.window.request_redraw();
    }
}

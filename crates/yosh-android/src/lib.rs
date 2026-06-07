//! Minimal Android shell for yosh.
//!
//! Brings up a winit window and the **engine's** wgpu `GpuContext`, then clears
//! the surface each frame — proving the reusable `yosh-engine` (decode pipeline +
//! reading-state machine) drives Android's Vulkan surface through the same
//! winit/wgpu path the desktop shell uses. The real reader (the engine's
//! `Reader` + draw list, touch input, suspend/resume → `Gpu::recreate_surface`,
//! a content-URI FD → `ZipSource::from_bytes`, egui chrome) lands on top of this.
//!
//! The whole crate is Android-only; the desktop shell lives in `crates/yosh`.
#![cfg(target_os = "android")]

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::android::activity::AndroidApp;
use winit::platform::android::EventLoopBuilderExtAndroid;
use winit::window::{Window, WindowId};

use yosh_engine::gpu::GpuContext;

/// Entry point android-activity calls on the native-activity thread.
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    log::info!("yosh-android starting");
    let event_loop = EventLoop::builder()
        .with_android_app(app)
        .build()
        .expect("build event loop");
    event_loop
        .run_app(&mut Shell::default())
        .expect("run event loop");
}

#[derive(Default)]
struct Shell {
    gpu: Option<Gpu>,
}

/// Surface + the engine's device context, mirroring the desktop `Gpu` wrapper.
struct Gpu {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
        );
        // Surface first (needs the window), then the engine builds the device.
        let instance = GpuContext::create_instance();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
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
        window.request_redraw();
        self.gpu = Some(Gpu {
            window,
            ctx,
            surface,
            config,
        });
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Android destroys the surface on background; drop our handle. The engine
        // device/queue (and a real reader's pool/cache) would survive, and
        // `resumed` rebuilds the surface against them.
        self.gpu = None;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.config.width = size.width.max(1);
                gpu.config.height = size.height.max(1);
                gpu.surface.configure(&gpu.ctx.device, &gpu.config);
            }
            WindowEvent::RedrawRequested => {
                let frame = match gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        gpu.surface.configure(&gpu.ctx.device, &gpu.config);
                        gpu.window.request_redraw();
                        return;
                    }
                    // Timeout / Occluded / Validation: skip this frame, try again.
                    _ => {
                        gpu.window.request_redraw();
                        return;
                    }
                };
                let view = frame.texture.create_view(&Default::default());
                let mut enc = gpu.ctx.device.create_command_encoder(&Default::default());
                {
                    let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.05,
                                    g: 0.05,
                                    b: 0.07,
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
                }
                gpu.ctx.queue.submit([enc.finish()]);
                gpu.window.pre_present_notify();
                frame.present();
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

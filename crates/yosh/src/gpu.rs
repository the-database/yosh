//! Desktop GPU wrapper: the reusable device/queue context (`yosh_engine::gpu`)
//! plus the winit-bound surface and its resize/reconfigure lifecycle.
//!
//! Device/queue/adapter creation and the surface-format pick live in the engine
//! so they can be reused behind any shell; this file owns only the parts tied to
//! a `winit::Window` (surface creation) and the surface's per-frame config.

use std::sync::Arc;

use winit::window::Window;
use yosh_engine::gpu::GpuContext;

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        // Surface first (needs the winit window), then the engine builds the
        // device/queue against it. The instance must exist before the surface.
        let instance = GpuContext::create_instance();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let ctx = GpuContext::create(instance, Some(&surface));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = GpuContext::choose_surface_format(&caps);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&ctx.device, &config);

        // Keep the device/queue (Arc-shared with the pool); the context's
        // instance + adapter drop here, exactly as in the pre-split code.
        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            surface,
            config,
            adapter_info: ctx.adapter_info.clone(),
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn reconfigure(&self) {
        self.surface.configure(&self.device, &self.config);
    }
}

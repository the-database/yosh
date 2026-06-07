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
    /// The reusable device/queue context. Kept (rather than dropped after setup)
    /// so the surface can be rebuilt against the *same* device after the OS tears
    /// it down on background (Android) — the decode pool, cache, and GPU textures,
    /// all device-owned, survive across the gap.
    ctx: GpuContext,
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

        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            adapter_info: ctx.adapter_info.clone(),
            surface,
            config,
            ctx,
        }
    }

    /// Rebuild the surface against the existing device for a (new) window — after
    /// the OS destroyed the old one on background. Only the window-bound surface
    /// is replaced; the device, decode pool, cache and textures are untouched, so
    /// the reader resumes without re-decoding. Reuses the chosen format/size, so
    /// it stays consistent with what egui-wgpu was told.
    pub fn recreate_surface(&mut self, window: Arc<Window>) {
        let surface = self
            .ctx
            .instance
            .create_surface(window)
            .expect("recreate surface");
        surface.configure(&self.ctx.device, &self.config);
        self.surface = surface;
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

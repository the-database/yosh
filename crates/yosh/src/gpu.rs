//! wgpu context: instance / adapter / device / queue / surface.
//!
//! The wgpu-29-specific shapes here (5-field `InstanceDescriptor`, futures for
//! `request_adapter`/`request_device`, `CurrentSurfaceTexture` enum,
//! `multiview_mask`) were proven out in `crates/present_bench`.

use std::sync::Arc;

use winit::window::Window;

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("request adapter");
        let adapter_info = adapter.get_info();
        // Request the adapter's full max texture dimension (wgpu's default is only
        // 8192) so very wide/tall images — e.g. a 16k-px panorama — fit in one GPU
        // texture. Decoded pages are clamped to this limit in decode.rs.
        let required_limits = wgpu::Limits {
            max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits,
            ..Default::default()
        }))
        .expect("request device");
        crate::decode::MAX_TEX_DIM.store(
            device.limits().max_texture_dimension_2d,
            std::sync::atomic::Ordering::Relaxed,
        );

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Prefer a NON-sRGB surface format: page textures store sRGB-encoded
        // grayscale bytes that we want to pass through unchanged. (egui-wgpu is
        // told this exact format and gamma-adjusts its own output accordingly.)
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

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
        surface.configure(&device, &config);

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            surface,
            config,
            adapter_info,
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

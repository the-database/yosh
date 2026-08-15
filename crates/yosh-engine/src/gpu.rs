//! Reusable wgpu device/queue context.
//!
//! This is the platform-agnostic half of GPU setup: instance + adapter + device +
//! queue creation, plus the surface-format pick. It holds **no** surface and no
//! windowing types (`winit`), so a shell creates the surface for its own platform
//! (a desktop `winit::Window`, an Android `ANativeWindow`, …) and hands a borrow
//! in for adapter selection. The desktop shell's `Gpu` (in `crates/yosh`) wraps
//! this and owns the surface + its resize/reconfigure lifecycle.
//!
//! The wgpu-29-specific shapes here (5-field `InstanceDescriptor`, futures for
//! `request_adapter`/`request_device`) were proven out in `crates/present_bench`.

use std::sync::Arc;

/// Instance + adapter + device + queue. The device/queue (`Arc`-shared) are what
/// the `DecodePool` and renderer use; the instance/adapter are retained so a
/// shell can (re)create surfaces against them — Android destroys its surface on
/// background and must rebuild it without tearing down the device.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// The wgpu instance (5-field `InstanceDescriptor`, PRIMARY backends). The
    /// shell creates its surface from this before calling [`GpuContext::create`].
    pub fn create_instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        })
    }

    /// Request an adapter (compatible with `compatible`, the shell's surface) and
    /// a device/queue. Requests the adapter's *full* `max_texture_dimension_2d`
    /// (wgpu defaults to only 8192) so very wide/tall pages fit in one texture,
    /// and publishes that limit to `decode::MAX_TEX_DIM` — which must happen
    /// before the first `page_target_h`, i.e. before any rendering, so callers
    /// build the context up front.
    ///
    /// `power` is the shell's call: a desktop with a discrete GPU wants
    /// `HighPerformance`, a phone (one GPU, so the hint picks nothing different)
    /// asks for `LowPower` — the right signal to the driver's power governor at
    /// no cost in throughput.
    pub fn create(
        instance: wgpu::Instance,
        compatible: Option<&wgpu::Surface<'static>>,
        power: wgpu::PowerPreference,
    ) -> Self {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power,
            force_fallback_adapter: false,
            compatible_surface: compatible,
        }))
        .expect("request adapter");
        let adapter_info = adapter.get_info();
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
        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        }
    }

    /// Prefer a NON-sRGB surface format: page textures store sRGB-encoded
    /// grayscale bytes that we want to pass through unchanged. (egui-wgpu is told
    /// this exact format and gamma-adjusts its own output accordingly.)
    pub fn choose_surface_format(caps: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
        caps.formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0])
    }
}

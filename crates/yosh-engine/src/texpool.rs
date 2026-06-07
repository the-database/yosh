//! Thread-safe pool of reusable page textures, keyed by (gray, w, h). Decode
//! workers `get()` a texture to upload into; the cache `put()`s textures back
//! when pages are evicted — cutting GPU allocation churn during fast scroll.
//!
//! Exact per-page decode targets (the single-resize invariant) mint many distinct
//! sizes across a session of resizing/zooming, so the pool is bounded both
//! per-size (`max_per_bucket`) and globally (`max_total`) with eviction, to keep
//! VRAM from creeping. Page-flipping reuses one size, so the cap never bites there.

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Default)]
struct Inner {
    buckets: HashMap<(bool, u32, u32), Vec<wgpu::Texture>>,
    total: usize,
}

pub struct TexturePool {
    inner: Mutex<Inner>,
    max_per_bucket: usize,
    max_total: usize,
}

impl TexturePool {
    pub fn new() -> Self {
        Self::with_max_total(24)
    }

    /// Create with a specific global texture cap (supplied by the device `Budget`
    /// so constrained devices keep less VRAM live).
    pub fn with_max_total(max_total: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            max_per_bucket: 8,
            max_total,
        }
    }

    /// Reuse a matching texture if available, else create one.
    pub fn get(&self, device: &wgpu::Device, gray: bool, w: u32, h: u32) -> wgpu::Texture {
        {
            let mut inner = self.inner.lock().unwrap();
            let key = (gray, w, h);
            if let Some(v) = inner.buckets.get_mut(&key)
                && let Some(t) = v.pop()
            {
                let empty = v.is_empty();
                if empty {
                    inner.buckets.remove(&key);
                }
                inner.total -= 1;
                return t;
            }
        }
        create_page_texture(device, gray, w, h)
    }

    /// Return a texture for reuse. Dropped if its bucket is full; older pooled
    /// textures are evicted first to stay under the global cap.
    pub fn put(&self, tex: wgpu::Texture, gray: bool, w: u32, h: u32) {
        let mut inner = self.inner.lock().unwrap();
        while inner.total >= self.max_total {
            // Evict one texture from any non-empty bucket (bounds VRAM during
            // resize/zoom size churn; the current working size stays recyclable).
            let Some(key) = inner
                .buckets
                .iter()
                .find(|(_, v)| !v.is_empty())
                .map(|(k, _)| *k)
            else {
                break;
            };
            if let Some(v) = inner.buckets.get_mut(&key) {
                v.pop();
                if v.is_empty() {
                    inner.buckets.remove(&key);
                }
                inner.total -= 1;
            }
        }
        let v = inner.buckets.entry((gray, w, h)).or_default();
        if v.len() < self.max_per_bucket {
            v.push(tex);
            inner.total += 1;
        }
    }
}

pub fn create_page_texture(device: &wgpu::Device, gray: bool, w: u32, h: u32) -> wgpu::Texture {
    let format = if gray {
        wgpu::TextureFormat::R8Unorm
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("page_tex"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

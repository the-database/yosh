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

struct Inner {
    buckets: HashMap<(bool, u32, u32), Vec<wgpu::Texture>>,
    total: usize,
    /// Global cap, **inside** the mutex so it can be re-set on a live pool (the
    /// runtime performance setting) through the `Arc<TexturePool>` every worker
    /// holds — there is no `&mut TexturePool` anywhere once decoding starts.
    max_total: usize,
}

impl Inner {
    /// Drop one pooled texture from any non-empty bucket. `false` ⇒ the pool was
    /// already empty (so an eviction loop must stop rather than spin).
    fn evict_one(&mut self) -> bool {
        let Some(key) = self.buckets.iter().find(|(_, v)| !v.is_empty()).map(|(k, _)| *k) else {
            return false;
        };
        if let Some(v) = self.buckets.get_mut(&key) {
            v.pop();
            if v.is_empty() {
                self.buckets.remove(&key);
            }
            self.total -= 1;
        }
        true
    }
}

pub struct TexturePool {
    inner: Mutex<Inner>,
    max_per_bucket: usize,
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}

impl TexturePool {
    pub fn new() -> Self {
        Self::with_max_total(24)
    }

    /// Create with a specific global texture cap (supplied by the device `Budget`
    /// so constrained devices keep less VRAM live).
    pub fn with_max_total(max_total: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                buckets: HashMap::new(),
                total: 0,
                max_total,
            }),
            max_per_bucket: 8,
        }
    }

    /// Re-cap the pool at runtime (the performance setting changing the device
    /// `Budget`), shedding pooled textures immediately so a lowered cap actually
    /// returns VRAM instead of waiting for the next `put`. Only *idle* recycled
    /// textures live here, so nothing on screen is affected.
    pub fn set_max_total(&self, max_total: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.max_total = max_total;
        while inner.total > max_total {
            if !inner.evict_one() {
                break;
            }
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
        // Make room first (bounds VRAM during resize/zoom size churn; the current
        // working size stays recyclable).
        while inner.total >= inner.max_total {
            if !inner.evict_one() {
                break;
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

//! Thread-safe pool of reusable page textures, keyed by (gray, w, h). Decode
//! workers `get()` a texture to upload into; the cache `put()`s textures back
//! when pages are evicted — cutting GPU allocation churn during fast scroll.

use std::collections::HashMap;
use std::sync::Mutex;

pub struct TexturePool {
    buckets: Mutex<HashMap<(bool, u32, u32), Vec<wgpu::Texture>>>,
    max_per_bucket: usize,
}

impl TexturePool {
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            max_per_bucket: 8,
        }
    }

    /// Reuse a matching texture if available, else create one.
    pub fn get(&self, device: &wgpu::Device, gray: bool, w: u32, h: u32) -> wgpu::Texture {
        if let Some(v) = self.buckets.lock().unwrap().get_mut(&(gray, w, h)) {
            if let Some(t) = v.pop() {
                return t;
            }
        }
        create_page_texture(device, gray, w, h)
    }

    /// Return a texture for reuse (dropped if the bucket is full).
    pub fn put(&self, tex: wgpu::Texture, gray: bool, w: u32, h: u32) {
        let mut buckets = self.buckets.lock().unwrap();
        let v = buckets.entry((gray, w, h)).or_default();
        if v.len() < self.max_per_bucket {
            v.push(tex);
        }
    }
}

/// Upload single-channel (gray) or RGBA8 pixels into an existing texture.
pub fn write_pixels(queue: &wgpu::Queue, tex: &wgpu::Texture, pixels: &[u8], w: u32, h: u32, gray: bool) {
    let bpp = if gray { 1 } else { 4 };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * bpp),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
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

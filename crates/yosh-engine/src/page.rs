//! Page-quad rendering: upload a decoded page to a GPU texture (R8 for gray,
//! RGBA8 for color) and draw it as a textured quad at a caller-computed
//! placement. Supports up to 2 quads per frame (single page or two-page spread)
//! via independent per-slot uniform buffers.

use std::time::Duration;

use crate::decode::{DecodedImage, ResizePath};
use crate::texpool::TexturePool;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FitMode {
    Window,
    Width,
    Height,
    /// 1:1 — one image pixel per device pixel, no resize (DPI ignored). Pages
    /// are decoded at full source resolution (see `decode_and_downscale`).
    Actual,
}

impl FitMode {
    pub fn cycle(self) -> Self {
        match self {
            FitMode::Window => FitMode::Width,
            FitMode::Width => FitMode::Height,
            FitMode::Height => FitMode::Actual,
            FitMode::Actual => FitMode::Window,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            FitMode::Window => "fit",
            FitMode::Width => "width",
            FitMode::Height => "height",
            FitMode::Actual => "1:1",
        }
    }
}

/// Screen-pixels per page-pixel for a fit mode (given page/content dims).
pub fn fit_scale(fit: FitMode, sw: f32, sh: f32, pw: f32, ph: f32) -> f32 {
    match fit {
        FitMode::Window => (sw / pw).min(sh / ph),
        FitMode::Width => sw / pw,
        FitMode::Height => sh / ph,
        FitMode::Actual => 1.0, // 1 page pixel → 1 device pixel
    }
}

/// Max quads drawn in one frame. Page-flip uses ≤2 (single / spread); the
/// continuous-scroll strip can show several partial pages at once.
pub const MAX_QUADS: usize = 8;

const SHADER: &str = r#"
struct Uniforms {
    scale: vec2<f32>,
    offset: vec2<f32>,
    gray: u32,
    rotation: u32,
    alpha: f32,
    blur: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    // Unit quad [0,1]^2 (uv space, y down) as two triangles.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    // offset = top-left corner in NDC (+x right, +y up); uv.y=0 is the page top.
    // The on-screen rectangle (offset/scale) is the rotated bounding box computed
    // by the app; here we only turn the sampled UVs so the texture fills it rotated.
    let ndc = vec2<f32>(u.offset.x + c.x * u.scale.x, u.offset.y - c.y * u.scale.y);
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    // 90° steps clockwise about the quad center (u.rotation = 0/1/2/3).
    var p = c - vec2<f32>(0.5, 0.5);
    switch (u.rotation) {
        case 1u: { p = vec2<f32>( p.y, -p.x); } // 90° CW
        case 2u: { p = vec2<f32>(-p.x, -p.y); } // 180°
        case 3u: { p = vec2<f32>(-p.y,  p.x); } // 270° CW
        default: {}
    }
    out.uv = p + vec2<f32>(0.5, 0.5);
    return out;
}

// Motion-blur tap count for the page-turn transition (odd ⇒ one tap is centered).
// Only used when u.blur != 0; normal page draws take the single-sample branch below
// and are unaffected.
const BLUR_TAPS: i32 = 7;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var s: vec4<f32>;
    if (u.blur == 0.0) {
        // Normal page draw: one 1:1 sample at the texel center, exactly as before.
        s = textureSample(tex, samp, in.uv);
    } else {
        // Fading flip overlay: a horizontal motion blur along the slide axis, so the
        // outgoing page smears as it sweeps away (and stops clashing as a second
        // sharp image over the incoming page). u.blur is uniform, so this branch is
        // uniform control flow (textureSample is legal here).
        var sum = vec4<f32>(0.0);
        let half_span = f32(BLUR_TAPS - 1) * 0.5;
        for (var i: i32 = 0; i < BLUR_TAPS; i = i + 1) {
            let f = (f32(i) - half_span) / half_span; // -1..1
            sum = sum + textureSample(tex, samp, vec2<f32>(in.uv.x + f * u.blur, in.uv.y));
        }
        s = sum / f32(BLUR_TAPS);
    }
    if (u.gray != 0u) {
        // Premultiply the fade into the opaque gray page so it composites over the
        // page underneath (u.alpha == 1.0 ⇒ the original opaque output).
        let g = s.r;
        return vec4<f32>(g * u.alpha, g * u.alpha, g * u.alpha, u.alpha);
    }
    // Color pages are stored premultiplied (see decode.rs); scaling by alpha keeps
    // them premultiplied for the pipeline's premultiplied-alpha blend. At alpha==1
    // this is the original passthrough — transparent areas show the background.
    return s * u.alpha;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    scale: [f32; 2],
    offset: [f32; 2],
    gray: u32,
    rotation: u32, // 0/1/2/3 = 0/90/180/270° CW (UV turn in the vertex shader)
    alpha: f32,    // opacity multiplier (1.0 normally; < 1.0 for a fading flip overlay)
    blur: f32,     // horizontal motion-blur smear half-width in UV (0.0 normally)
}

/// One frame of an animated page (GIF/WebP), beyond frame 0. Frame 0 lives in the
/// `PageTexture`'s own `texture`/`view`; these are frames 1..N.
struct AnimFrame {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// How long this frame is shown, in milliseconds.
    delay_ms: u32,
}

/// A decoded page resident on the GPU.
pub struct PageTexture {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub w: u32,
    pub h: u32,
    /// Native source dimensions (pre-downscale), for reporting true zoom.
    pub src_w: u32,
    pub src_h: u32,
    pub gray: bool,
    /// Which CPU resize path produced this page (for the info overlay readout).
    pub path: ResizePath,
    /// Decode target height this page was produced for (the `target_h` the pool
    /// used). Lets the cache detect pages decoded at a stale resolution after a
    /// zoom/resize and re-decode them in place without blanking the display.
    pub target_h: u32,
    /// Decoded via the fast LQ (seeking) tier — re-decode to HQ once settled.
    pub lq: bool,
    /// Extra frames/layers 1..N (frame 0 is `texture`/`view`). `None` for stills.
    /// Animation (GIF/WebP) frames share this page's size; `.ico` layers may each
    /// be a different size, so recycling keys off each texture's own dimensions.
    anim: Option<Vec<AnimFrame>>,
    /// Frame 0's display time, ms (only meaningful when `anim` is `Some`).
    frame0_delay_ms: u32,
    /// Sum of all frame delays, ms — the animation loop period (0 for stills).
    anim_total_ms: u32,
    /// True for an auto-playing animation (GIF/WebP); false for a still or for
    /// `.ico` layers (stepped manually, no play/pause). Gates the panel controls.
    animated: bool,
}

impl PageTexture {
    /// Whether this page auto-plays (GIF/WebP). False for stills and `.ico` layers.
    pub fn is_animation(&self) -> bool {
        self.animated
    }

    /// Number of frames/layers (1 for a still page).
    pub fn frame_count(&self) -> usize {
        1 + self.anim.as_ref().map_or(0, |a| a.len())
    }

    /// The view for a specific frame index (clamped). Frame 0 is the base view.
    pub fn frame_view(&self, i: usize) -> &wgpu::TextureView {
        match &self.anim {
            None => &self.view,
            Some(frames) => match i.checked_sub(1) {
                None => &self.view, // frame 0
                Some(j) => frames.get(j).map_or(&self.view, |f| &f.view),
            },
        }
    }

    /// The display time (ms) of a specific frame index.
    pub fn frame_delay_ms(&self, i: usize) -> u32 {
        if i == 0 {
            return self.frame0_delay_ms;
        }
        self.anim
            .as_ref()
            .and_then(|frames| frames.get(i - 1))
            .map_or(self.frame0_delay_ms, |f| f.delay_ms)
    }

    /// The texture view to display at animation time `t` (since some fixed
    /// origin). For a still page this is always frame 0; for an animation it
    /// walks the cumulative per-frame delays modulo the loop period.
    pub fn view_at(&self, t: Duration) -> &wgpu::TextureView {
        let Some(frames) = &self.anim else {
            return &self.view;
        };
        let mut acc = (t.as_millis() % self.anim_total_ms.max(1) as u128) as u32;
        if acc < self.frame0_delay_ms {
            return &self.view; // frame 0
        }
        acc -= self.frame0_delay_ms;
        for f in frames {
            if acc < f.delay_ms {
                return &f.view;
            }
            acc -= f.delay_ms;
        }
        // Rounding slack at the loop boundary: fall back to the last frame.
        frames.last().map_or(&self.view, |f| &f.view)
    }

    /// Return every GPU texture (frame 0 and any animation frames) to the pool
    /// for reuse (drops the views first). All frames share the page's bucket.
    pub fn recycle(self, pool: &TexturePool) {
        let PageTexture {
            texture,
            view,
            w,
            h,
            src_w: _,
            src_h: _,
            gray,
            path: _,
            target_h: _,
            lq: _,
            anim,
            frame0_delay_ms: _,
            anim_total_ms: _,
            animated: _,
        } = self;
        drop(view);
        pool.put(texture, gray, w, h);
        if let Some(frames) = anim {
            for AnimFrame { texture, view, delay_ms: _ } in frames {
                drop(view);
                // `.ico` layers differ in size, so key by each texture's own dims.
                let (fw, fh) = (texture.width(), texture.height());
                pool.put(texture, gray, fw, fh);
            }
        }
    }
}

pub struct PagePipeline {
    pub pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubos: Vec<wgpu::Buffer>,
}

impl PagePipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("page_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("page_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("page_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("page_pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Premultiplied-alpha "over": src + dst*(1-src.a). Opaque and
                    // grayscale pages (alpha 1) are unaffected; transparent color
                    // pages composite over the cleared background.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let make_ubo = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("page_ubo"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            pipeline,
            bgl,
            sampler,
            ubos: (0..MAX_QUADS).map(|_| make_ubo()).collect(),
        }
    }

    /// Upload a decoded page to a GPU texture (reusing one from the pool).
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &DecodedImage,
        pool: &TexturePool,
        target_h: u32,
    ) -> PageTexture {
        let bpp = if img.gray { 1u32 } else { 4u32 };
        let texture = pool.get(device, img.gray, img.w, img.h);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(img.w * bpp),
                rows_per_image: Some(img.h),
            },
            wgpu::Extent3d {
                width: img.w,
                height: img.h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        PageTexture {
            texture,
            view,
            w: img.w,
            h: img.h,
            src_w: img.src_w,
            src_h: img.src_h,
            gray: img.gray,
            path: img.path,
            target_h,
            lq: false,
            anim: None,
            frame0_delay_ms: 0,
            anim_total_ms: 0,
            animated: false,
        }
    }

    /// Upload a multi-frame page — an animation (GIF/WebP) or `.ico` layers. Frame
    /// 0 becomes the base `PageTexture` (exactly as `upload`); each remaining frame
    /// gets its own pooled texture (sizes may differ for `.ico`). `frames` is the
    /// decode's `(frame, delay_ms)` list (≥2 entries); `animated` is true for a
    /// GIF/WebP (auto-play) and false for `.ico` layers (manual stepping only).
    pub fn upload_animated(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frames: Vec<(DecodedImage, u32)>,
        pool: &TexturePool,
        target_h: u32,
        animated: bool,
    ) -> PageTexture {
        let total: u32 = frames.iter().map(|(_, d)| *d).sum();
        let mut iter = frames.into_iter();
        let (img0, d0) = iter.next().expect("upload_animated: empty frames");
        let mut base = Self::upload(device, queue, &img0, pool, target_h);
        let anim: Vec<AnimFrame> = iter
            .map(|(img, delay_ms)| {
                // Reuse `upload` to allocate+fill a pooled texture, then peel off
                // its texture/view (PageTexture has no Drop, so the partial move
                // is fine; the unused still-fields just drop).
                let pt = Self::upload(device, queue, &img, pool, target_h);
                AnimFrame { texture: pt.texture, view: pt.view, delay_ms }
            })
            .collect();
        base.anim = Some(anim);
        base.frame0_delay_ms = d0;
        base.anim_total_ms = total.max(1);
        base.animated = animated;
        base
    }

    /// Write quad `slot`'s uniform (NDC scale + top-left offset) and return its
    /// bind group, ready to bind and `draw(0..6)`. `view` is the texture view to
    /// sample — usually `page.view`, or `page.view_at(t)` for an animated page.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_quad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        page: &PageTexture,
        view: &wgpu::TextureView,
        scale: [f32; 2],
        offset: [f32; 2],
        rotation: u32,
        alpha: f32,
        blur: f32,
    ) -> wgpu::BindGroup {
        let u = Uniforms {
            scale,
            offset,
            gray: page.gray as u32,
            rotation,
            alpha,
            blur,
        };
        queue.write_buffer(&self.ubos[slot], 0, bytemuck::bytes_of(&u));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ubos[slot].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

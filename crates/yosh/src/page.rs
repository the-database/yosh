//! Page-quad rendering: upload a decoded page to a GPU texture (R8 for gray,
//! RGBA8 for color) and draw it as a textured quad at a caller-computed
//! placement. Supports up to 2 quads per frame (single page or two-page spread)
//! via independent per-slot uniform buffers.

use crate::decode::DecodedImage;
use crate::texpool::TexturePool;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FitMode {
    Window,
    Width,
    Height,
}

impl FitMode {
    pub fn cycle(self) -> Self {
        match self {
            FitMode::Window => FitMode::Width,
            FitMode::Width => FitMode::Height,
            FitMode::Height => FitMode::Window,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            FitMode::Window => "fit",
            FitMode::Width => "width",
            FitMode::Height => "height",
        }
    }
}

/// Screen-pixels per page-pixel for a fit mode (given page/content dims).
pub fn fit_scale(fit: FitMode, sw: f32, sh: f32, pw: f32, ph: f32) -> f32 {
    match fit {
        FitMode::Window => (sw / pw).min(sh / ph),
        FitMode::Width => sw / pw,
        FitMode::Height => sh / ph,
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
    let ndc = vec2<f32>(u.offset.x + c.x * u.scale.x, u.offset.y - c.y * u.scale.y);
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = c;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(tex, samp, in.uv);
    if (u.gray != 0u) {
        return vec4<f32>(s.r, s.r, s.r, 1.0);
    }
    return vec4<f32>(s.rgb, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    scale: [f32; 2],
    offset: [f32; 2],
    gray: u32,
    _pad: [u32; 3],
}

/// A decoded page resident on the GPU.
pub struct PageTexture {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub w: u32,
    pub h: u32,
    pub gray: bool,
}

impl PageTexture {
    /// Return the GPU texture to the pool for reuse (drops the view first).
    pub fn recycle(self, pool: &TexturePool) {
        let PageTexture {
            texture,
            view,
            w,
            h,
            gray,
        } = self;
        drop(view);
        pool.put(texture, gray, w, h);
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
                    blend: None,
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
            gray: img.gray,
        }
    }

    /// Write quad `slot`'s uniform (NDC scale + top-left offset) and return its
    /// bind group, ready to bind and `draw(0..6)`.
    pub fn prepare_quad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        page: &PageTexture,
        scale: [f32; 2],
        offset: [f32; 2],
    ) -> wgpu::BindGroup {
        let u = Uniforms {
            scale,
            offset,
            gray: page.gray as u32,
            _pad: [0; 3],
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
                    resource: wgpu::BindingResource::TextureView(&page.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

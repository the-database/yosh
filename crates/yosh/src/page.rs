//! Page-quad rendering: upload a decoded page to a GPU texture (R8 for gray,
//! RGBA8 for color) and draw it as a fit-to-window textured quad.

use crate::decode::DecodedImage;

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
    // Map the unit quad into NDC via scale/offset. offset is the top-left corner
    // in NDC; +x right, +y up; uv.y=0 is the top of the page.
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

/// Screen-pixels per page-pixel for a fit mode.
pub fn fit_scale(fit: FitMode, sw: f32, sh: f32, pw: f32, ph: f32) -> f32 {
    match fit {
        FitMode::Window => (sw / pw).min(sh / ph),
        FitMode::Width => sw / pw,
        FitMode::Height => sh / ph,
    }
}

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
    _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub w: u32,
    pub h: u32,
    pub gray: bool,
}

pub struct PagePipeline {
    pub pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubo: wgpu::Buffer,
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
        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page_ubo"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            sampler,
            ubo,
        }
    }

    /// Upload a decoded page to a GPU texture.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &DecodedImage,
    ) -> PageTexture {
        let (format, bpp) = if img.gray {
            (wgpu::TextureFormat::R8Unorm, 1u32)
        } else {
            (wgpu::TextureFormat::Rgba8Unorm, 4u32)
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("page_tex"),
            size: wgpu::Extent3d {
                width: img.w,
                height: img.h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
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
            _texture: texture,
            view,
            w: img.w,
            h: img.h,
            gray: img.gray,
        }
    }

    /// Write the uniform for a draw of `page` (with fit mode + vertical pan) and
    /// return its bind group, ready to bind and `draw(0..6)`. (M1.5 generalizes
    /// for spreads.)
    ///
    /// Note: shares a single ubo, so this is only correct for one draw per frame.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page: &PageTexture,
        screen_w: u32,
        screen_h: u32,
        fit: FitMode,
        pan_y: f32,
    ) -> wgpu::BindGroup {
        let (sw, sh) = (screen_w.max(1) as f32, screen_h.max(1) as f32);
        let (pw, ph) = (page.w as f32, page.h as f32);
        let s = fit_scale(fit, sw, sh, pw, ph);
        let scale = [2.0 * pw * s / sw, 2.0 * ph * s / sh];
        // Horizontal: centered. Vertical: centered if it fits, else pan in [0,1]
        // (0 = page top at screen top, 1 = page bottom at screen bottom).
        let offset_y = if scale[1] <= 2.0 {
            scale[1] / 2.0
        } else {
            1.0 + pan_y.clamp(0.0, 1.0) * (scale[1] - 2.0)
        };
        let offset = [-scale[0] / 2.0, offset_y];
        let u = Uniforms {
            scale,
            offset,
            gray: page.gray as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&u));

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ubo.as_entire_binding(),
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

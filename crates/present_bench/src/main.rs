// Spike 2 — present ceiling.
//
// Pre-decodes ~N pages to single-channel R8 GPU textures, then cycles them
// through the wgpu surface as fast as possible (one full-screen textured quad,
// no decode in the loop), measuring sustained swaps/sec for each available
// present mode. Confirms present is NOT the bottleneck (should be >> 83/sec).
//
// IMPORTANT: run this with the window on the PHYSICAL monitor. Through the
// Parsec virtual display, Immediate/Mailbox are typically unavailable and Fifo
// caps at the virtual refresh — those numbers are meaningless.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = p[vi];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = textureSample(tex, samp, in.uv).r;
    return vec4<f32>(r, r, r, 1.0);
}
"#;

struct Page {
    w: u32,
    h: u32,
    pixels: Vec<u8>, // single-channel R8
}

struct Cfg {
    folder: PathBuf,
    target_h: u32,
    count: usize,
    modes_arg: String,
}

fn parse() -> Cfg {
    const USAGE: &str =
        "usage: present_bench <folder> [--mode immediate|mailbox|fifo|all] [--count 30] [--target-height 1440]";
    let mut a = std::env::args().skip(1);
    let mut folder = None;
    let mut target_h = 1440u32;
    let mut count = 30usize;
    let mut modes_arg = "all".to_string();
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--mode" => modes_arg = a.next().expect(USAGE),
            "--count" => count = a.next().expect(USAGE).parse().expect("bad count"),
            "--target-height" => target_h = a.next().expect(USAGE).parse().expect("bad height"),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                if folder.is_none() {
                    folder = Some(PathBuf::from(other));
                } else {
                    panic!("unexpected arg {other:?}; {USAGE}");
                }
            }
        }
    }
    Cfg {
        folder: folder.expect(USAGE),
        target_h,
        count,
        modes_arg,
    }
}

/// Decode the first `count` pages and downscale each to a single-channel R8
/// buffer at `target_h` tall.
fn load_pages(cfg: &Cfg) -> Vec<Page> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(&cfg.folder)
        .expect("read_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e.eq_ignore_ascii_case("png")))
        .collect();
    files.sort();
    files.truncate(cfg.count);
    assert!(!files.is_empty(), "no PNG files in {:?}", cfg.folder);

    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    let mut out = Vec::with_capacity(files.len());
    for p in &files {
        let bytes = std::fs::read(p).expect("read");
        let mut r = png::Decoder::new(std::io::Cursor::new(&bytes))
            .read_info()
            .expect("read_info");
        let size = r.output_buffer_size().expect("buffer size");
        let mut buf = vec![0u8; size];
        let info = r.next_frame(&mut buf).expect("next_frame");
        let (w, h) = (info.width, info.height);
        let ch = info.buffer_size() / (w as usize * h as usize);

        // Collapse to single channel (take channel 0 — fine for a present bench).
        let gray: Vec<u8> = if ch == 1 {
            buf
        } else {
            buf.iter().step_by(ch).copied().collect()
        };

        let tw = ((w as f64) * (cfg.target_h as f64) / (h as f64)).round() as u32;
        let src = ImageRef::new(w, h, &gray, PixelType::U8).expect("src");
        let mut dst = Image::new(tw, cfg.target_h, PixelType::U8);
        resizer.resize(&src, &mut dst, &opts).expect("resize");
        out.push(Page {
            w: tw,
            h: cfg.target_h,
            pixels: dst.into_vec(),
        });
    }
    println!(
        "loaded {} pages, downscaled to {}px tall (R8)",
        out.len(),
        cfg.target_h
    );
    out
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_groups: Vec<wgpu::BindGroup>,

    // benchmark state
    modes: Vec<wgpu::PresentMode>,
    phase: usize,
    frames: u64,
    phase_start: Instant,
    frame_idx: usize,
    results: Vec<(wgpu::PresentMode, f64)>,
}

impl Gpu {
    fn new(window: Arc<Window>, pages: &[Page], modes_arg: &str) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("adapter");
        println!("adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        println!("surface format: {format:?}   present modes available: {:?}", caps.present_modes);

        let want: Vec<wgpu::PresentMode> = match modes_arg {
            "immediate" => vec![wgpu::PresentMode::Immediate],
            "mailbox" => vec![wgpu::PresentMode::Mailbox],
            "fifo" => vec![wgpu::PresentMode::Fifo],
            _ => vec![
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Fifo,
            ],
        };
        let mut modes: Vec<wgpu::PresentMode> = want
            .into_iter()
            .filter(|m| caps.present_modes.contains(m))
            .collect();
        if modes.is_empty() {
            modes.push(wgpu::PresentMode::Fifo);
        }
        println!("benchmarking present modes (4s each): {modes:?}");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Shader + pipeline.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe"),
            layout: Some(&pipeline_layout),
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

        // Sampler + one bind group (texture) per page.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut bind_groups = Vec::with_capacity(pages.len());
        for pg in pages {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: pg.w,
                    height: pg.h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pg.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pg.w),
                    rows_per_image: Some(pg.h),
                },
                wgpu::Extent3d {
                    width: pg.w,
                    height: pg.h,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            bind_groups.push(bg);
        }

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_groups,
            modes,
            phase: 0,
            frames: 0,
            phase_start: Instant::now(),
            frame_idx: 0,
            results: Vec::new(),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Render one frame. Returns false when all phases are done.
    fn render(&mut self) -> bool {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return true;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return true;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("surface validation error");
                return true;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.pipeline);
            let idx = self.frame_idx % self.bind_groups.len();
            rp.set_bind_group(0, &self.bind_groups[idx], &[]);
            rp.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        frame.present();

        self.frames += 1;
        self.frame_idx += 1;

        if self.phase_start.elapsed() >= Duration::from_secs(4) {
            let secs = self.phase_start.elapsed().as_secs_f64();
            let fps = self.frames as f64 / secs;
            let mode = self.modes[self.phase];
            println!(
                "  {:?}: {:.0} swaps/s   ({} frames / {:.2}s)",
                mode, fps, self.frames, secs
            );
            self.results.push((mode, fps));
            self.phase += 1;
            if self.phase >= self.modes.len() {
                println!("\n== present ceiling summary (target = 83/sec) ==");
                for (m, f) in &self.results {
                    println!("  {:?}: {:.0} swaps/s   ({:.1}x target)", m, f, f / 83.0);
                }
                return false;
            }
            self.config.present_mode = self.modes[self.phase];
            self.surface.configure(&self.device, &self.config);
            self.frames = 0;
            self.phase_start = Instant::now();
        }
        true
    }
}

struct App {
    pages: Vec<Page>,
    modes_arg: String,
    gpu: Option<Gpu>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("present_bench — move me to the PHYSICAL monitor")
            .with_inner_size(winit::dpi::LogicalSize::new(1080.0, 1440.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        window.request_redraw();
        self.gpu = Some(Gpu::new(window, &self.pages, &self.modes_arg));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => gpu.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if !gpu.render() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }
}

fn main() {
    let cfg = parse();
    let pages = load_pages(&cfg);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        pages,
        modes_arg: cfg.modes_arg,
        gpu: None,
    };
    event_loop.run_app(&mut app).expect("run");
}

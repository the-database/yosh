//! Application: winit `ApplicationHandler`, owns GPU + egui + reader state.
//! M1.3: async decode pool + bounded cache + forward prefetch → hitch-free
//! navigation. The current page is drawn from the cache; if a target isn't ready
//! yet the last-drawn page is held (no flicker).

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::cache::PageCache;
use crate::gpu::Gpu;
use crate::page::{fit_scale, FitMode, PagePipeline};
use crate::pool::{DecodePool, Msg};
use crate::prefetch::desired_window;
use crate::source::{FolderSource, PageSource};
use crate::ui::{self, UiState};

const TARGET_H: u32 = 2160;
const WORKERS: usize = 8;
const CACHE_CAP: usize = 48;
const FWD: usize = 16;
const BACK: usize = 6;
const FWD_MAX: usize = 40;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Ltr,
    Rtl,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Direction::Ltr => "LTR",
            Direction::Rtl => "RTL",
        }
    }
}

pub struct App {
    initial_path: Option<PathBuf>,
    start_index: usize,
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui: UiState,

    page_pipeline: PagePipeline,
    source: Option<Arc<dyn PageSource>>,
    pool: Option<DecodePool>,
    cache: PageCache,
    failed: HashSet<usize>,
    index: usize,
    start_index: usize,
    last_drawn: Option<usize>,

    fit: FitMode,
    pan_y: f32,
    direction: Direction,
    cursor_x: f64,
    nav_times: VecDeque<Instant>,
}

impl App {
    pub fn new(initial_path: Option<PathBuf>, start_index: usize) -> Self {
        Self {
            initial_path,
            start_index,
            state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("yosh")
            .with_inner_size(winit::dpi::LogicalSize::new(1100.0, 1500.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gpu = Gpu::new(window.clone());

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            Some(8192),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let page_pipeline = PagePipeline::new(&gpu.device, gpu.config.format);

        let mut ui = UiState::default();
        ui.status = format!("{} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
        if let Some(p) = self.initial_path.take() {
            ui.pending_open = Some(p);
        }

        window.request_redraw();
        self.state = Some(State {
            window,
            gpu,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            page_pipeline,
            source: None,
            pool: None,
            cache: PageCache::new(CACHE_CAP),
            failed: HashSet::new(),
            index: 0,
            start_index: self.start_index,
            last_drawn: None,
            fit: FitMode::Window,
            pan_y: 0.0,
            direction: Direction::Rtl, // manga default; toggle with D
            cursor_x: 0.0,
            nav_times: VecDeque::new(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let response = state.egui_state.on_window_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !response.consumed =>
            {
                if let Some(action) = action_from(&event) {
                    state.apply_action(action);
                }
            }
            WindowEvent::CursorMoved { position, .. } => state.cursor_x = position.x,
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => state.on_wheel(delta),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !response.consumed => state.on_click(),
            _ => {}
        }

        if response.repaint {
            state.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

enum Action {
    Forward,
    Backward,
    Left,
    Right,
    First,
    Last,
    CycleFit,
    ToggleDir,
}

/// Map a key event to an action, preferring the physical key but falling back to
/// the logical key (covers injected events without scancodes). Vertical keys are
/// absolute (forward/backward); horizontal keys are resolved by reading direction.
fn action_from(ev: &KeyEvent) -> Option<Action> {
    if let PhysicalKey::Code(c) = ev.physical_key {
        match c {
            KeyCode::ArrowDown | KeyCode::Space | KeyCode::PageDown => return Some(Action::Forward),
            KeyCode::ArrowUp | KeyCode::PageUp => return Some(Action::Backward),
            KeyCode::ArrowRight => return Some(Action::Right),
            KeyCode::ArrowLeft => return Some(Action::Left),
            KeyCode::Home => return Some(Action::First),
            KeyCode::End => return Some(Action::Last),
            KeyCode::KeyF => return Some(Action::CycleFit),
            KeyCode::KeyD => return Some(Action::ToggleDir),
            _ => {}
        }
    }
    if let Key::Named(n) = &ev.logical_key {
        match n {
            NamedKey::ArrowDown | NamedKey::PageDown => return Some(Action::Forward),
            NamedKey::ArrowUp | NamedKey::PageUp => return Some(Action::Backward),
            NamedKey::ArrowRight => return Some(Action::Right),
            NamedKey::ArrowLeft => return Some(Action::Left),
            NamedKey::Home => return Some(Action::First),
            NamedKey::End => return Some(Action::Last),
            _ => {}
        }
    }
    None
}

impl State {
    fn apply_action(&mut self, action: Action) {
        match action {
            Action::Forward => self.step(1),
            Action::Backward => self.step(-1),
            // In RTL, "left" advances the story; in LTR, "right" does.
            Action::Right => self.step(if self.direction == Direction::Ltr { 1 } else { -1 }),
            Action::Left => self.step(if self.direction == Direction::Ltr { -1 } else { 1 }),
            Action::First => self.goto(0),
            Action::Last => {
                if let Some(s) = &self.source {
                    self.goto(s.len().saturating_sub(1));
                }
            }
            Action::CycleFit => {
                self.fit = self.fit.cycle();
                self.pan_y = 0.0;
            }
            Action::ToggleDir => {
                self.direction = match self.direction {
                    Direction::Ltr => Direction::Rtl,
                    Direction::Rtl => Direction::Ltr,
                };
            }
        }
    }

    fn step(&mut self, delta: i64) {
        let Some(src) = &self.source else { return };
        let len = src.len() as i64;
        if len == 0 {
            return;
        }
        let next = (self.index as i64 + delta).clamp(0, len - 1) as usize;
        if next != self.index {
            self.nav_times.push_back(Instant::now());
            self.goto(next);
        }
    }

    fn goto(&mut self, index: usize) {
        self.index = index;
        self.pan_y = 0.0; // start new page at the top
        self.prefetch();
    }

    /// Mouse wheel: pan within an overflowing page, or flip at the edges / when
    /// the page already fits.
    fn on_wheel(&mut self, delta: MouseScrollDelta) {
        let dy = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
        };
        if dy == 0.0 {
            return;
        }
        let overflow = self.current_overflows();
        if !overflow {
            // Page fits: wheel flips (down = forward).
            self.step(if dy < 0.0 { 1 } else { -1 });
            return;
        }
        let before = self.pan_y;
        self.pan_y -= dy * 0.15;
        if self.pan_y < 0.0 {
            if before <= 0.0005 {
                self.step(-1);
                self.pan_y = 1.0;
            } else {
                self.pan_y = 0.0;
            }
        } else if self.pan_y > 1.0 {
            if before >= 0.9995 {
                self.step(1);
                self.pan_y = 0.0;
            } else {
                self.pan_y = 1.0;
            }
        }
    }

    fn on_click(&mut self) {
        let half = self.gpu.config.width as f64 / 2.0;
        if self.cursor_x < half {
            self.apply_action(Action::Left);
        } else {
            self.apply_action(Action::Right);
        }
    }

    /// Does the current page overflow the window vertically under the active fit?
    fn current_overflows(&self) -> bool {
        let Some(pt) = self.cache.get(self.index) else {
            return false;
        };
        let (sw, sh) = (self.gpu.config.width.max(1) as f32, self.gpu.config.height.max(1) as f32);
        let s = fit_scale(self.fit, sw, sh, pt.w as f32, pt.h as f32);
        pt.h as f32 * s > sh + 0.5
    }

    /// Forward look-ahead distance, widened when flipping quickly.
    fn dynamic_fwd(&mut self) -> usize {
        let now = Instant::now();
        while let Some(&t) = self.nav_times.front() {
            if now.duration_since(t) > Duration::from_millis(800) {
                self.nav_times.pop_front();
            } else {
                break;
            }
        }
        (FWD + self.nav_times.len() * 4).min(FWD_MAX)
    }

    fn open(&mut self, path: &Path) {
        if path.is_dir() {
            match FolderSource::new(path) {
                Ok(src) if src.len() > 0 => {
                    let source: Arc<dyn PageSource> = Arc::new(src);
                    self.pool = Some(DecodePool::new(
                        source.clone(),
                        self.gpu.device.clone(),
                        self.gpu.queue.clone(),
                        TARGET_H,
                        WORKERS,
                    ));
                    self.cache.clear();
                    self.failed.clear();
                    self.last_drawn = None;
                    self.index = self.start_index.min(source.len() - 1);
                    self.start_index = 0;
                    self.ui.opened = Some(path.to_path_buf());
                    self.source = Some(source);
                    self.prefetch();
                }
                Ok(_) => self.ui.status = "no images in folder".into(),
                Err(e) => self.ui.status = format!("open failed: {e}"),
            }
        } else {
            self.ui.status = "archives supported in M1.6 — open a folder for now".into();
        }
    }

    /// Recompute the desired prefetch window and hand it to the pool.
    fn prefetch(&mut self) {
        let fwd = self.dynamic_fwd();
        let (Some(src), Some(pool)) = (&self.source, &self.pool) else {
            return;
        };
        let desired: Vec<usize> = desired_window(self.index, src.len(), fwd, BACK)
            .into_iter()
            .filter(|i| !self.cache.contains(*i) && !self.failed.contains(i))
            .collect();
        pool.set_jobs(desired);
    }

    fn render(&mut self) {
        if let Some(p) = self.ui.pending_open.take() {
            self.open(&p);
        }

        // Drain finished decodes into the cache.
        if let Some(pool) = &self.pool {
            for msg in pool.poll() {
                match msg {
                    Msg::Done { index, page } => self.cache.insert(index, page, self.index),
                    Msg::Failed { index } => {
                        self.failed.insert(index);
                    }
                }
            }
        }
        // Refresh the work list against the updated cache.
        self.prefetch();

        // Pick the page to draw: the current page if ready, else hold last-drawn.
        let draw_index = if self.cache.contains(self.index) {
            self.last_drawn = Some(self.index);
            Some(self.index)
        } else {
            self.last_drawn.filter(|i| self.cache.contains(*i))
        };

        if let Some(src) = &self.source {
            let loading = !self.cache.contains(self.index);
            self.ui.status = format!(
                "{}/{}  {} {}{}",
                self.index + 1,
                src.len(),
                self.fit.label(),
                self.direction.label(),
                if loading { "  …" } else { "" }
            );
        }

        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.reconfigure();
                return;
            }
            _ => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let page_bg = draw_index.and_then(|i| self.cache.get(i)).map(|pt| {
            self.page_pipeline.prepare(
                &self.gpu.device,
                &self.gpu.queue,
                pt,
                self.gpu.config.width,
                self.gpu.config.height,
                self.fit,
                self.pan_y,
            )
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("page"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(bg) = &page_bg {
                pass.set_pipeline(&self.page_pipeline.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // egui chrome.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ui_state = &mut self.ui;
        let full_output = self.egui_ctx.run(raw_input, |ctx| ui::chrome(ctx, ui_state));
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let ppp = self.egui_ctx.pixels_per_point();
        let primitives = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point: ppp,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        let user_cmds = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &primitives,
            &screen,
        );
        {
            let mut egui_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut egui_pass, &primitives, &screen);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.gpu
            .queue
            .submit(user_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();
    }
}

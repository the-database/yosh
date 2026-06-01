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
use crate::config;
use crate::gpu::Gpu;
use crate::layout::{self, Layout};
use crate::page::{fit_scale, FitMode, PagePipeline, PageTexture, MAX_QUADS};
use crate::pool::{DecodePool, Msg};
use crate::prefetch::desired_window;
use crate::source::{FolderSource, PageSource, RarSource, ZipSource};
use crate::ui::{self, UiState};

const TARGET_H: u32 = 2160;
const WORKERS: usize = 8;
const CACHE_CAP: usize = 48;
const FWD: usize = 16;
const BACK: usize = 6;
const FWD_MAX: usize = 40;
/// Pixels scrolled per mouse-wheel line in continuous-scroll mode.
const SCROLL_WHEEL_PX: f32 = 110.0;
/// Height/width estimate for not-yet-decoded pages in the scroll strip.
const DEFAULT_ASPECT: f32 = 1.5;

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
    layout: Layout,
    pan_y: f32,
    direction: Direction,
    cursor_x: f64,
    nav_times: VecDeque<Instant>,

    // Continuous-scroll mode (M2.1).
    scroll_mode: bool,
    top_offset: f32,  // pixels the anchor page (self.index) is scrolled above the viewport top
    est_aspect: f32,  // h/w estimate for undecoded pages in the strip

    settings: config::Settings,
    volume_key: Option<String>,
}

fn fit_from_u8(v: u8) -> FitMode {
    match v {
        1 => FitMode::Width,
        2 => FitMode::Height,
        _ => FitMode::Window,
    }
}

fn fit_to_u8(f: FitMode) -> u8 {
    match f {
        FitMode::Window => 0,
        FitMode::Width => 1,
        FitMode::Height => 2,
    }
}

/// A quad to draw this frame (NDC scale + top-left offset), referencing a cached page.
struct Quad {
    slot: usize,
    page_index: usize,
    scale: [f32; 2],
    offset: [f32; 2],
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
        let settings = config::load();

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
            fit: fit_from_u8(settings.fit),
            layout: if settings.layout_spread {
                Layout::Spread
            } else {
                Layout::Single
            },
            pan_y: 0.0,
            direction: if settings.direction_rtl {
                Direction::Rtl
            } else {
                Direction::Ltr
            },
            cursor_x: 0.0,
            nav_times: VecDeque::new(),
            scroll_mode: settings.scroll,
            top_offset: 0.0,
            est_aspect: DEFAULT_ASPECT,
            settings,
            volume_key: None,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let response = state.egui_state.on_window_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                state.persist();
                event_loop.exit();
            }
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
    ToggleLayout,
    ToggleScroll,
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
            KeyCode::KeyS => return Some(Action::ToggleLayout),
            KeyCode::KeyC => return Some(Action::ToggleScroll),
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
            Action::Forward => {
                if self.scroll_mode {
                    let vh = self.gpu.config.height as f32;
                    self.scroll_by(vh * 0.9);
                } else {
                    self.step(1);
                }
            }
            Action::Backward => {
                if self.scroll_mode {
                    let vh = self.gpu.config.height as f32;
                    self.scroll_by(-vh * 0.9);
                } else {
                    self.step(-1);
                }
            }
            // In RTL, "left" advances the story; in LTR, "right" does. (Page-flip only.)
            Action::Right if !self.scroll_mode => {
                self.step(if self.direction == Direction::Ltr { 1 } else { -1 })
            }
            Action::Left if !self.scroll_mode => {
                self.step(if self.direction == Direction::Ltr { -1 } else { 1 })
            }
            Action::Right | Action::Left => {}
            Action::First => self.goto(0),
            Action::Last => {
                if let Some(s) = &self.source {
                    self.goto(s.len().saturating_sub(1));
                }
            }
            Action::CycleFit => {
                self.fit = self.fit.cycle();
                self.pan_y = 0.0;
                self.settings.fit = fit_to_u8(self.fit);
                config::save(&self.settings);
            }
            Action::ToggleDir => {
                self.direction = match self.direction {
                    Direction::Ltr => Direction::Rtl,
                    Direction::Rtl => Direction::Ltr,
                };
                self.settings.direction_rtl = self.direction == Direction::Rtl;
                config::save(&self.settings);
            }
            Action::ToggleLayout => {
                self.layout = self.layout.toggled();
                // Snap to the current view's anchor so pairing is consistent.
                self.index = layout::view_start(self.layout, self.index);
                self.pan_y = 0.0;
                self.settings.layout_spread = self.layout == Layout::Spread;
                config::save(&self.settings);
                self.prefetch();
            }
            Action::ToggleScroll => {
                self.scroll_mode = !self.scroll_mode;
                self.top_offset = 0.0;
                self.settings.scroll = self.scroll_mode;
                config::save(&self.settings);
                self.prefetch();
            }
        }
    }

    fn step(&mut self, dir: i64) {
        let Some(src) = &self.source else { return };
        let len = src.len();
        if len == 0 {
            return;
        }
        let next = if dir > 0 {
            layout::next_view(self.layout, self.index, len)
        } else {
            layout::prev_view(self.layout, self.index)
        };
        if next != self.index {
            self.nav_times.push_back(Instant::now());
            self.goto(next);
        }
    }

    fn goto(&mut self, index: usize) {
        self.index = index;
        self.pan_y = 0.0; // start new page at the top
        self.top_offset = 0.0;
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), index);
        }
        self.prefetch();
    }

    /// Persist the current position + settings (called on close).
    fn persist(&mut self) {
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), self.index);
        }
        config::save(&self.settings);
    }

    /// Mouse wheel: pan within an overflowing page, or flip at the edges / when
    /// the page already fits.
    fn on_wheel(&mut self, delta: MouseScrollDelta) {
        if self.scroll_mode {
            let dy_px = match delta {
                MouseScrollDelta::LineDelta(_, y) => y * SCROLL_WHEEL_PX,
                MouseScrollDelta::PixelDelta(p) => p.y as f32,
            };
            self.scroll_by(-dy_px); // wheel down (y<0) scrolls the strip down
            return;
        }
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

    /// Top edge of a page (screen px, 0 = window top): centered if it fits,
    /// else panned by `pan_y` (0 = page top aligned, 1 = page bottom aligned).
    fn vertical_top(&self, dh: f32, sh: f32) -> f32 {
        if dh <= sh + 0.5 {
            (sh - dh) / 2.0
        } else {
            -self.pan_y.clamp(0.0, 1.0) * (dh - sh)
        }
    }

    fn quad_from_px(
        slot: usize,
        page_index: usize,
        x_px: f32,
        y_px: f32,
        dw: f32,
        dh: f32,
        sw: f32,
        sh: f32,
    ) -> Quad {
        Quad {
            slot,
            page_index,
            scale: [2.0 * dw / sw, 2.0 * dh / sh],
            offset: [-1.0 + 2.0 * x_px / sw, 1.0 - 2.0 * y_px / sh],
        }
    }

    fn single_quad(&self, idx: usize, t: &PageTexture, sw: f32, sh: f32) -> Quad {
        let s = fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32);
        let (dw, dh) = (t.w as f32 * s, t.h as f32 * s);
        let x = (sw - dw) / 2.0;
        Self::quad_from_px(0, idx, x, self.vertical_top(dh, sh), dw, dh, sw, sh)
    }

    /// Compute the quads to draw this frame (1 for single/last-held, 2 for a
    /// ready spread). Only includes pages present in the cache.
    fn build_quads(&self) -> Vec<Quad> {
        let Some(src) = &self.source else {
            return Vec::new();
        };
        let len = src.len();
        if len == 0 {
            return Vec::new();
        }
        let sw = self.gpu.config.width.max(1) as f32;
        let sh = self.gpu.config.height.max(1) as f32;

        let (a, b) = layout::view_pages(self.layout, self.index, len);
        let ta = self.cache.get(a);
        // Wide (landscape) page is a double-spread image → show it alone.
        let force_single = ta.map_or(false, |t| t.w > t.h);
        let b = if force_single { None } else { b };
        let tb = b.and_then(|bi| self.cache.get(bi).map(|t| (bi, t)));

        match (ta, tb) {
            (Some(ta), Some((bi, tb))) => {
                let combined_w = ta.w as f32 + tb.w as f32;
                let max_h = ta.h.max(tb.h) as f32;
                let s = fit_scale(self.fit, sw, sh, combined_w, max_h);
                let x0 = (sw - combined_w * s) / 2.0;
                // Screen order: LTR puts the lower index on the left; RTL reverses.
                let (l_idx, l_t, r_idx, r_t) = match self.direction {
                    Direction::Ltr => (a, ta, bi, tb),
                    Direction::Rtl => (bi, tb, a, ta),
                };
                let (dwl, dhl) = (l_t.w as f32 * s, l_t.h as f32 * s);
                let (dwr, dhr) = (r_t.w as f32 * s, r_t.h as f32 * s);
                vec![
                    Self::quad_from_px(0, l_idx, x0, self.vertical_top(dhl, sh), dwl, dhl, sw, sh),
                    Self::quad_from_px(
                        1,
                        r_idx,
                        x0 + dwl,
                        self.vertical_top(dhr, sh),
                        dwr,
                        dhr,
                        sw,
                        sh,
                    ),
                ]
            }
            (Some(ta), None) => vec![self.single_quad(a, ta, sw, sh)],
            _ => {
                // Anchor not decoded yet: hold the last-drawn page if still cached.
                if let Some(li) = self.last_drawn {
                    if let Some(t) = self.cache.get(li) {
                        return vec![self.single_quad(li, t, sw, sh)];
                    }
                }
                Vec::new()
            }
        }
    }

    fn page_display_h(&self, i: usize, sw: f32) -> f32 {
        match self.cache.get(i) {
            Some(t) => sw * (t.h as f32 / t.w as f32),
            None => sw * self.est_aspect,
        }
    }

    fn scroll_by(&mut self, dy: f32) {
        if self.source.is_none() {
            return;
        }
        self.top_offset += dy;
        let before = self.index;
        self.normalize();
        if self.index != before {
            self.nav_times.push_back(Instant::now());
        }
        self.prefetch();
    }

    /// Keep (index, top_offset) in range using best-known page heights, so the
    /// anchor stays valid as nearby pages decode (and their real heights land).
    fn normalize(&mut self) {
        let len = match &self.source {
            Some(s) => s.len(),
            None => return,
        };
        if len == 0 {
            return;
        }
        let sw = self.gpu.config.width.max(1) as f32;
        while self.index + 1 < len {
            let h = self.page_display_h(self.index, sw);
            if self.top_offset >= h {
                self.top_offset -= h;
                self.index += 1;
            } else {
                break;
            }
        }
        while self.top_offset < 0.0 && self.index > 0 {
            self.index -= 1;
            self.top_offset += self.page_display_h(self.index, sw);
        }
        if self.index == 0 && self.top_offset < 0.0 {
            self.top_offset = 0.0;
        }
        if self.index + 1 >= len {
            let vh = self.gpu.config.height as f32;
            let max_off = (self.page_display_h(len - 1, sw) - vh).max(0.0);
            if self.top_offset > max_off {
                self.top_offset = max_off;
            }
        }
    }

    /// Build the visible vertical-strip quads (width-fit, stacked top to bottom).
    fn build_scroll_quads(&self) -> Vec<Quad> {
        let Some(src) = &self.source else {
            return Vec::new();
        };
        let len = src.len();
        if len == 0 {
            return Vec::new();
        }
        let sw = self.gpu.config.width.max(1) as f32;
        let sh = self.gpu.config.height.max(1) as f32;
        let mut quads = Vec::new();
        let mut y = -self.top_offset;
        let mut i = self.index;
        let mut slot = 0;
        while i < len && y < sh && slot < MAX_QUADS {
            let dh_layout = self.page_display_h(i, sw);
            if y + dh_layout > 0.0 {
                if let Some(t) = self.cache.get(i) {
                    let dh = sw * (t.h as f32 / t.w as f32);
                    quads.push(Self::quad_from_px(slot, i, 0.0, y, sw, dh, sw, sh));
                    slot += 1;
                }
            }
            y += dh_layout;
            i += 1;
        }
        quads
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
        let built: Result<Arc<dyn PageSource>, String> = if path.is_dir() {
            FolderSource::new(path)
                .map(|s| Arc::new(s) as Arc<dyn PageSource>)
                .map_err(|e| e.to_string())
        } else {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase());
            match ext.as_deref() {
                Some("cbz") | Some("zip") => ZipSource::new(path)
                    .map(|s| Arc::new(s) as Arc<dyn PageSource>)
                    .map_err(|e| e.to_string()),
                Some("cbr") | Some("rar") => RarSource::new(path)
                    .map(|s| Arc::new(s) as Arc<dyn PageSource>)
                    .map_err(|e| e.to_string()),
                Some("7z") | Some("cb7") => Err("7z support is planned for M2".to_string()),
                _ => Err("unsupported file type (open a folder, CBZ, or CBR)".to_string()),
            }
        };
        match built {
            Ok(source) if source.len() > 0 => self.set_source(source, path),
            Ok(_) => self.ui.status = "no images found".into(),
            Err(e) => self.ui.status = format!("open failed: {e}"),
        }
    }

    fn set_source(&mut self, source: Arc<dyn PageSource>, path: &Path) {
        // Persist the previous volume's position before switching.
        if let Some(k) = self.volume_key.take() {
            self.settings.last_pages.insert(k, self.index);
        }
        let key = path.to_string_lossy().into_owned();
        let resume = self.settings.last_pages.get(&key).copied().unwrap_or(0);
        // CLI start index (if given) wins over the saved position.
        let idx = if self.start_index > 0 {
            self.start_index
        } else {
            resume
        };
        self.start_index = 0;

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
        self.nav_times.clear();
        self.index = idx.min(source.len() - 1);
        self.volume_key = Some(key);
        self.ui.opened = Some(path.to_path_buf());
        self.source = Some(source);
        self.prefetch();
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

    #[allow(deprecated)]
    fn render(&mut self) {
        if let Some(p) = self.ui.pending_open.take() {
            self.open(&p);
        }

        // Drain finished decodes into the cache.
        if let Some(pool) = &self.pool {
            for msg in pool.poll() {
                match msg {
                    Msg::Done { index, page } => {
                        self.est_aspect = page.h as f32 / page.w as f32;
                        self.cache.insert(index, page, self.index);
                    }
                    Msg::Failed { index } => {
                        self.failed.insert(index);
                    }
                }
            }
        }
        // Keep the scroll anchor valid as page heights resolve, then refresh work.
        if self.scroll_mode {
            self.normalize();
        }
        self.prefetch();

        // Decide what to draw this frame (scroll strip, or single/spread flip).
        let quads = if self.scroll_mode {
            self.build_scroll_quads()
        } else {
            self.build_quads()
        };
        self.ui.dir_label = self.direction.label();
        self.ui.fit_label = self.fit.label();
        self.ui.layout_label = if self.scroll_mode {
            "scroll"
        } else {
            self.layout.label()
        };
        if let Some(src) = &self.source {
            let len = src.len();
            let anchor = if self.scroll_mode {
                self.index
            } else {
                layout::view_pages(self.layout, self.index, len).0
            };
            let loading = !self.cache.contains(anchor);
            if !loading {
                self.last_drawn = Some(anchor);
            }
            self.ui.status = format!(
                "{}/{}{}",
                self.index + 1,
                len,
                if loading { "  …" } else { "" }
            );
        }
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.cache.get(q.page_index).map(|t| {
                    self.page_pipeline.prepare_quad(
                        &self.gpu.device,
                        &self.gpu.queue,
                        q.slot,
                        t,
                        q.scale,
                        q.offset,
                    )
                })
            })
            .collect();

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
            if !page_bgs.is_empty() {
                pass.set_pipeline(&self.page_pipeline.pipeline);
                for bg in &page_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
        }

        // egui chrome.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ui_state = &mut self.ui;
        let full_output = self.egui_ctx.run(raw_input, |ctx| ui::chrome(ctx, ui_state));
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        // Apply toggle-button requests (take effect next frame).
        if std::mem::take(&mut self.ui.req_toggle_dir) {
            self.apply_action(Action::ToggleDir);
        }
        if std::mem::take(&mut self.ui.req_cycle_fit) {
            self.apply_action(Action::CycleFit);
        }
        if std::mem::take(&mut self.ui.req_toggle_layout) {
            self.apply_action(Action::ToggleLayout);
        }
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

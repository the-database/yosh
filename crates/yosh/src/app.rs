//! Application: winit `ApplicationHandler`, owns GPU + egui + reader state.
//! M1.3: async decode pool + bounded cache + forward prefetch → hitch-free
//! navigation. The current page is drawn from the cache; if a target isn't ready
//! yet the last-drawn page is held (no flicker).

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use fast_image_resize::Resizer;

use crate::cache::PageCache;
use crate::config;
use crate::decode::decode_and_downscale;
use crate::downscale::Downscaler;
use crate::gpu::Gpu;
use crate::layout::{self, Layout};
use crate::library::{cover_bytes, Library};
use crate::page::{fit_scale, FitMode, PagePipeline, PageTexture, MAX_QUADS};
use crate::pool::{DecodePool, Msg};
use crate::prefetch::desired_window;
use crate::source::{is_image_ext, FolderSource, PageSource, RarSource, SevenzSource, ZipSource};
use crate::texpool::TexturePool;
use crate::ui::{self, UiState};

// Decode target height tracks the on-screen page size (see `desired_target_h`)
// so the high-quality linear-light downscale does the *full* reduction in one
// pass. Otherwise pages decode larger than shown and the GPU re-downscales them
// with a plain bilinear (no mipmaps) → halftone aliasing/moiré.
const TARGET_H_DEFAULT: u32 = 1440;
const MIN_TARGET: u32 = 480;
const MAX_TARGET: u32 = 3840;
const TARGET_QUANTUM: u32 = 128;
const WORKERS: usize = 8;
const CACHE_CAP: usize = 48;
const FWD: usize = 16;
const BACK: usize = 6;
const FWD_MAX: usize = 40;
/// Pixels scrolled per mouse-wheel line in continuous-scroll mode.
const SCROLL_WHEEL_PX: f32 = 110.0;
/// Height/width estimate for not-yet-decoded pages in the scroll strip.
const DEFAULT_ASPECT: f32 = 1.5;
/// Library cover thumbnail height, and how many to decode per frame.
const THUMB_H: u32 = 360;
const THUMB_BUDGET: usize = 2;
/// Grace period before the centered loading spinner appears, so quick page
/// loads don't flash an indicator on every flip — only genuinely slow decodes
/// (e.g. seeking fast through very high-res pages) cross it.
const LOADING_INDICATOR_DELAY: Duration = Duration::from_millis(150);
/// Fraction of the window width on each side that flips pages on click (and
/// shows a hover arrow). The middle is reserved for double-click → fullscreen.
const EDGE_FRAC: f32 = 0.15;
/// Max gap between two middle-zone clicks to count as a double-click.
const DOUBLE_CLICK: Duration = Duration::from_millis(350);

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
    spread_offset: usize, // spread pairing parity (0 or 1), per-volume
    zoom: f32,       // page-flip zoom factor (1.0 = fit)
    pan_x: f32,      // page-flip pan offset in screen px (from centered)
    pan_y: f32,
    direction: Direction,
    cursor_x: f64,
    cursor_y: f64,
    mouse_down: bool,
    drag_dist: f32, // accumulated drag distance, to distinguish click from pan
    cursor_in_window: bool, // gates the edge-hover navigation arrows
    last_mid_click: Option<Instant>, // middle-zone double-click → fullscreen
    jump: bool, // seek mode (key J): true = "jump" (skip ahead), false = "step" (see every page, default)
    nav_times: VecDeque<Instant>,

    // Continuous-scroll mode (M2.1).
    scroll_mode: bool,
    top_offset: f32,  // pixels the anchor page (self.index) is scrolled above the viewport top
    est_aspect: f32,  // h/w estimate for undecoded pages in the strip

    settings: config::Settings,
    volume_key: Option<String>,
    tex_pool: Arc<TexturePool>,
    downscaler: Arc<Downscaler>,
    gpu_flag: Arc<AtomicBool>,
    /// Decode resolution (page height in px), tracked to the display size so the
    /// CPU linear-light resize is the only downscale. Read by pool workers.
    target_h: Arc<AtomicU32>,
    /// Last-computed desired target, for debouncing re-decode across resize/zoom.
    pending_target: u32,
    /// Page index the Tab info overlay text was built for (None = rebuild needed).
    info_for: Option<usize>,
    /// The anchor page currently waiting to decode and when that wait began,
    /// used to delay the loading spinner *per page* (so a fast page reached at
    /// the end of a slow-seek streak still gets its own grace period). None as
    /// soon as the anchor is ready.
    loading_pending: Option<(usize, Instant)>,

    library: Library,
    library_view: bool,
    thumb_resizer: Resizer,
}

fn fit_from_u8(v: u8) -> FitMode {
    match v {
        1 => FitMode::Width,
        2 => FitMode::Height,
        3 => FitMode::Actual,
        _ => FitMode::Window,
    }
}

fn fit_to_u8(f: FitMode) -> u8 {
    match f {
        FitMode::Window => 0,
        FitMode::Width => 1,
        FitMode::Height => 2,
        FitMode::Actual => 3,
    }
}

/// Human-readable byte size for the info overlay.
fn human_size(n: u64) -> String {
    const KB: u64 = 1 << 10;
    const MB: u64 = 1 << 20;
    if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

/// Probe an encoded image's header for `(width, height, "FORMAT · detail")`
/// without a full decode. Returns `(0, 0, ...)` if dimensions can't be read.
fn probe(b: &[u8]) -> (u32, u32, String) {
    let be16 = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]) as u32;
    let le16 = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as u32;
    // PNG
    if b.len() >= 26 && b[..4] == [0x89, 0x50, 0x4E, 0x47] {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        let color = match b[25] {
            0 => "grayscale",
            2 => "RGB",
            3 => "indexed",
            4 => "grayscale+alpha",
            6 => "RGBA",
            _ => "?",
        };
        return (w, h, format!("PNG · {}-bit {}", b[24], color));
    }
    // JPEG: scan for a Start-Of-Frame marker.
    if b.len() >= 4 && b[0] == 0xFF && b[1] == 0xD8 {
        let mut i = 2;
        while i + 9 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let m = b[i + 1];
            let is_sof = (0xC0..=0xCF).contains(&m) && m != 0xC4 && m != 0xC8 && m != 0xCC;
            if is_sof {
                let kind = match b[i + 9] {
                    1 => "grayscale",
                    3 => "YCbCr",
                    4 => "CMYK",
                    _ => "?",
                };
                return (be16(i + 7), be16(i + 5), format!("JPEG · {}-bit {}", b[i + 4], kind));
            }
            if i + 3 >= b.len() {
                break;
            }
            i += 2 + u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        }
        return (0, 0, "JPEG".to_string());
    }
    // GIF
    if b.len() >= 10 && &b[0..3] == b"GIF" {
        return (le16(6), le16(8), "GIF".to_string());
    }
    // BMP
    if b.len() >= 26 && &b[0..2] == b"BM" {
        let w = i32::from_le_bytes([b[18], b[19], b[20], b[21]]).unsigned_abs();
        let h = i32::from_le_bytes([b[22], b[23], b[24], b[25]]).unsigned_abs();
        return (w, h, "BMP".to_string());
    }
    // WebP (RIFF container)
    if b.len() >= 30 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        match &b[12..16] {
            b"VP8X" => {
                let w = 1 + (b[24] as u32 | (b[25] as u32) << 8 | (b[26] as u32) << 16);
                let h = 1 + (b[27] as u32 | (b[28] as u32) << 8 | (b[29] as u32) << 16);
                return (w, h, "WebP".to_string());
            }
            b"VP8 " => {
                return (le16(26) & 0x3FFF, le16(28) & 0x3FFF, "WebP".to_string());
            }
            _ => return (0, 0, "WebP".to_string()),
        }
    }
    // AVIF / HEIF (ISO-BMFF): dimensions need a box walk; report the format only.
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        return (0, 0, "AVIF".to_string());
    }
    (0, 0, "image".to_string())
}

/// Best-effort: load a system CJK font so Japanese/Chinese/Korean text (paths,
/// filenames, library titles) renders in the egui chrome. Appended as a
/// fallback so Latin keeps the default look; silently skipped if none found.
fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/Library/Fonts/Hiragino Sans GB.ttc",
    ];
    let Some(bytes) = CANDIDATES.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".to_owned(),
        // The CJK fallback's glyphs render above the Latin baseline; nudge them
        // down so mixed JP/EN lines (top bar, info overlay, titles) line up.
        std::sync::Arc::new(egui::FontData::from_owned(bytes).tweak(egui::FontTweak {
            y_offset_factor: 0.18,
            ..Default::default()
        })),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// The window / taskbar / title-bar icon, decoded from the embedded logo PNG.
/// (The icon embedded in the .exe via the build script is used by Explorer and
/// shortcuts, but winit needs the window icon set explicitly at runtime.)
fn window_icon() -> Option<winit::window::Icon> {
    let img = image::load_from_memory(include_bytes!("../assets/yosh.png"))
        .ok()?
        .to_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
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
            .with_window_icon(window_icon())
            .with_inner_size(winit::dpi::LogicalSize::new(1100.0, 1500.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let mut gpu = Gpu::new(window.clone());

        let egui_ctx = egui::Context::default();
        install_cjk_font(&egui_ctx);
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
        let mut settings = config::load();
        gpu.set_turbo(settings.turbo);
        let tex_pool = Arc::new(TexturePool::new());
        let downscaler = Arc::new(Downscaler::new(&gpu.device));
        // GPU downscale is disabled for now (the single-bilinear-blit path can't
        // match the HQ CPU resize); the code is kept dormant behind
        // `pool::GPU_DOWNSCALE_ENABLED` for a future HQ-GPU rewrite. Forced off
        // here regardless of the persisted `settings.gpu`.
        let gpu_flag = Arc::new(AtomicBool::new(false));

        let mut ui = UiState::default();
        ui.status = format!("{} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
        if let Some(p) = self.initial_path.take() {
            ui.pending_open = Some(p);
        }

        let library = match &settings.library_root {
            Some(r) => Library::scan(std::path::Path::new(r)),
            None => Library::empty(),
        };
        // Open straight into the grid if nothing was passed to read.
        let library_view = ui.pending_open.is_none() && !library.volumes.is_empty();
        // Show the keys overlay once, on the first launch ever, then persist so it
        // never auto-opens again (F1 / "? Help" reopen it on demand).
        if !settings.help_seen {
            ui.help_open = true;
            settings.help_seen = true;
            config::save(&settings);
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
            cache: PageCache::new(CACHE_CAP, tex_pool.clone()),
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
            spread_offset: 0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            cursor_y: 0.0,
            mouse_down: false,
            drag_dist: 0.0,
            cursor_in_window: false,
            last_mid_click: None,
            jump: settings.jump,
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
            tex_pool,
            downscaler,
            gpu_flag,
            target_h: Arc::new(AtomicU32::new(TARGET_H_DEFAULT)),
            pending_target: 0,
            info_for: None,
            loading_pending: None,
            library,
            library_view,
            thumb_resizer: Resizer::new(),
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
            WindowEvent::DroppedFile(path) => state.ui.pending_open = Some(path),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(action) = action_from(&event) && !response.consumed {
                    if matches!(action, Action::Quit) {
                        state.persist();
                        event_loop.exit();
                    } else {
                        state.apply_action(action);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.on_cursor_moved(position.x, position.y)
            }
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => state.on_wheel(delta),
            WindowEvent::MouseInput {
                state: btn,
                button: MouseButton::Left,
                ..
            } if !response.consumed => state.on_left_button(btn == ElementState::Pressed),
            WindowEvent::CursorEntered { .. } => state.cursor_in_window = true,
            WindowEvent::CursorLeft { .. } => state.cursor_in_window = false,
            // CursorLeft isn't guaranteed on focus loss / occlusion (alt-tab, a
            // fast monitor switch), which would otherwise strand a hover arrow
            // on screen. A later CursorMoved / CursorEntered re-arms the flag.
            WindowEvent::Focused(false) | WindowEvent::Occluded(true) => {
                state.cursor_in_window = false
            }
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
    TogglePresent,
    ZoomIn,
    ZoomOut,
    // View presets (number keys): each sets a complete page-flip view at once.
    PresetWindow,
    PresetWidth,
    PresetActual,
    PresetSpreadLtr,
    PresetSpreadRtl,
    ToggleHelp,
    ToggleFullscreen,
    ToggleSpreadOffset,
    ToggleInfo,
    PrevVolume,
    NextVolume,
    ToggleJump,
    Quit,
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
            KeyCode::KeyD => return Some(Action::ToggleDir),
            KeyCode::KeyS => return Some(Action::ToggleLayout),
            KeyCode::KeyO => return Some(Action::ToggleSpreadOffset),
            KeyCode::KeyC => return Some(Action::ToggleScroll),
            KeyCode::KeyT => return Some(Action::TogglePresent),
            KeyCode::KeyJ => return Some(Action::ToggleJump),
            KeyCode::Equal | KeyCode::NumpadAdd => return Some(Action::ZoomIn),
            KeyCode::Minus | KeyCode::NumpadSubtract => return Some(Action::ZoomOut),
            KeyCode::Digit9 | KeyCode::Numpad9 => return Some(Action::PresetWindow),
            KeyCode::Digit8 | KeyCode::Numpad8 => return Some(Action::PresetWidth),
            KeyCode::Digit7 | KeyCode::Numpad7 => return Some(Action::PresetSpreadLtr),
            KeyCode::Digit6 | KeyCode::Numpad6 => return Some(Action::PresetSpreadRtl),
            KeyCode::Digit0 | KeyCode::Numpad0 => return Some(Action::PresetActual),
            KeyCode::F1 => return Some(Action::ToggleHelp),
            KeyCode::KeyI => return Some(Action::ToggleInfo),
            KeyCode::F11 => return Some(Action::ToggleFullscreen),
            KeyCode::Escape => return Some(Action::Quit),
            KeyCode::BracketLeft => return Some(Action::PrevVolume),
            KeyCode::BracketRight => return Some(Action::NextVolume),
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
            NamedKey::F1 => return Some(Action::ToggleHelp),
            NamedKey::Escape => return Some(Action::Quit),
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
            Action::CycleFit if self.scroll_mode => {
                // In scroll: toggle width-fit (zoom 1) vs height-fit (a typical
                // page ~fills the viewport height).
                self.pan_x = 0.0;
                if (self.zoom - 1.0).abs() < 0.01 {
                    let sw = self.gpu.config.width.max(1) as f32;
                    let sh = self.gpu.config.height.max(1) as f32;
                    let cw = sh / self.est_aspect.max(0.1);
                    self.zoom = (cw / sw).clamp(0.2, 8.0);
                } else {
                    self.zoom = 1.0;
                }
            }
            Action::CycleFit => {
                self.fit = self.fit.cycle();
                self.zoom = 1.0;
                self.pan_x = 0.0;
                self.pan_y = 0.0;
                self.settings.fit = fit_to_u8(self.fit);
                config::save(&self.settings);
            }
            Action::ZoomIn => {
                self.zoom = (self.zoom * 1.25).min(8.0);
                self.clamp_pan();
            }
            Action::ZoomOut => {
                self.zoom = (self.zoom / 1.25).max(1.0);
                self.clamp_pan();
            }
            Action::PresetWindow => self.apply_view(FitMode::Window, false, None),
            Action::PresetWidth => self.apply_view(FitMode::Width, false, None),
            Action::PresetActual => self.apply_view(FitMode::Actual, false, None),
            Action::PresetSpreadLtr => {
                self.apply_view(FitMode::Window, true, Some(Direction::Ltr))
            }
            Action::PresetSpreadRtl => {
                self.apply_view(FitMode::Window, true, Some(Direction::Rtl))
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
                self.index = layout::view_start(self.layout, self.index, self.spread_offset);
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
            Action::TogglePresent => {
                self.settings.turbo = !self.settings.turbo;
                self.gpu.set_turbo(self.settings.turbo);
                config::save(&self.settings);
            }
            Action::ToggleHelp => self.ui.help_open = !self.ui.help_open,
            Action::ToggleInfo => {
                self.ui.info_open = !self.ui.info_open;
                self.info_for = None; // rebuild the overlay text next render
            }
            Action::ToggleFullscreen => {
                let fs = match self.window.fullscreen() {
                    Some(_) => None,
                    None => Some(Fullscreen::Borderless(None)),
                };
                self.window.set_fullscreen(fs);
            }
            Action::ToggleSpreadOffset => {
                self.spread_offset ^= 1;
                if let Some(k) = &self.volume_key {
                    self.settings
                        .spread_offsets
                        .insert(k.clone(), self.spread_offset as u8);
                    config::save(&self.settings);
                }
                // Re-anchor so the current view re-pairs with the new parity.
                self.index = layout::view_start(self.layout, self.index, self.spread_offset);
                self.prefetch();
            }
            Action::PrevVolume => self.jump_volume(-1),
            Action::NextVolume => self.jump_volume(1),
            Action::ToggleJump => {
                self.jump = !self.jump;
                self.settings.jump = self.jump;
                config::save(&self.settings);
            }
            // Esc → quit is intercepted in `window_event` (needs the event loop),
            // so it never reaches here.
            Action::Quit => {}
        }
    }

    fn step(&mut self, dir: i64) {
        let Some(src) = &self.source else { return };
        let len = src.len();
        if len == 0 {
            return;
        }
        // "Step" seek (default; toggle "jump" with J): don't flip while the
        // current page is still decoding, so you see every page instead of
        // skipping past it. "Jump" skips ahead for fast long-distance seeks.
        if !self.jump {
            let cur = layout::view_pages(self.layout, self.index, len, self.spread_offset).0;
            if !self.cache.contains(cur) {
                return;
            }
        }
        let next = if dir > 0 {
            layout::next_view(self.layout, self.index, len, self.spread_offset)
        } else {
            layout::prev_view(self.layout, self.index, len, self.spread_offset)
        };
        if next != self.index {
            self.nav_times.push_back(Instant::now());
            self.goto(next);
        }
    }

    fn goto(&mut self, index: usize) {
        self.index = index;
        self.pan_x = 0.0;
        self.pan_y = 0.0; // start new page centered
        self.top_offset = 0.0;
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), index);
        }
        self.prefetch();
    }

    /// Open the previous (`delta < 0`) or next (`delta > 0`) sibling volume of
    /// the same kind — folder ↔ folder, archive ↔ archive — in natural-sort
    /// order within the current volume's parent directory (`[` / `]`). The
    /// reading mode/position of the current volume is persisted by `open`; the
    /// new one resumes its own saved page. No-op at the ends or with nothing open.
    fn jump_volume(&mut self, delta: i64) {
        let Some(key) = self.volume_key.clone() else {
            return;
        };
        let cur = PathBuf::from(&key);
        let sibs = crate::library::sibling_volumes(&cur);
        let cur_name = cur.file_name();
        let Some(idx) = sibs.iter().position(|p| p.file_name() == cur_name) else {
            return;
        };
        let Ok(target) = usize::try_from(idx as i64 + delta) else {
            return; // before the first
        };
        if let Some(path) = sibs.get(target).cloned() {
            self.open(&path);
        }
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
        // Vertical pan in px; flip pages at the top/bottom edges.
        let sh = self.gpu.config.height.max(1) as f32;
        let maxp = ((self.current_display_h() - sh) / 2.0).max(0.0);
        let next = self.pan_y.clamp(-maxp, maxp) + dy * 80.0;
        if next > maxp + 0.5 {
            self.step(-1); // panned above the top -> previous page, land at its bottom
            self.pan_y = -1.0e6;
        } else if next < -maxp - 0.5 {
            self.step(1); // panned below the bottom -> next page, land at its top
            self.pan_y = 1.0e6;
        } else {
            self.pan_y = next;
        }
    }

    /// A clean click: the left/right edge strips flip pages; the wide middle
    /// does nothing on a single click but toggles fullscreen on a double-click.
    fn on_click(&mut self) {
        let w = self.gpu.config.width.max(1) as f64;
        let edge = (w * EDGE_FRAC as f64).max(1.0);
        if self.cursor_x < edge {
            self.last_mid_click = None;
            self.apply_action(Action::Left);
        } else if self.cursor_x > w - edge {
            self.last_mid_click = None;
            self.apply_action(Action::Right);
        } else {
            let now = Instant::now();
            let is_double = self
                .last_mid_click
                .is_some_and(|t| now.duration_since(t) < DOUBLE_CLICK);
            if is_double {
                self.last_mid_click = None;
                self.apply_action(Action::ToggleFullscreen);
            } else {
                self.last_mid_click = Some(now);
            }
        }
    }

    fn on_cursor_moved(&mut self, x: f64, y: f64) {
        let dx = (x - self.cursor_x) as f32;
        let dy = (y - self.cursor_y) as f32;
        self.cursor_x = x;
        self.cursor_y = y;
        self.cursor_in_window = true;
        if !self.mouse_down {
            return;
        }
        self.drag_dist += dx.abs() + dy.abs();
        if self.scroll_mode {
            // Grab the strip: pan horizontally, scroll vertically.
            self.pan_x += dx;
            self.top_offset -= dy;
            self.clamp_pan();
            self.normalize();
        } else {
            self.pan_x += dx;
            self.pan_y += dy;
            self.clamp_pan();
        }
    }

    /// Left button: a clean press/release (little movement) flips; a press +
    /// drag pans instead.
    fn on_left_button(&mut self, pressed: bool) {
        if pressed {
            self.mouse_down = true;
            self.drag_dist = 0.0;
        } else {
            if self.mouse_down && self.drag_dist < 6.0 {
                self.on_click();
            }
            self.mouse_down = false;
        }
    }

    /// Does the current page overflow the window vertically under the active fit?
    fn current_overflows(&self) -> bool {
        let Some(pt) = self.cache.get(self.index) else {
            return false;
        };
        let (sw, sh) = (self.gpu.config.width.max(1) as f32, self.gpu.config.height.max(1) as f32);
        let s = fit_scale(self.fit, sw, sh, pt.w as f32, pt.h as f32) * self.zoom;
        pt.h as f32 * s > sh + 0.5
    }

    /// Top edge (screen px): centered, then panned by `pan_y`, clamped so the
    /// page can't pull away from the viewport edge when larger than it.
    fn vertical_top(&self, dh: f32, sh: f32) -> f32 {
        let maxp = ((dh - sh) / 2.0).max(0.0);
        (sh - dh) / 2.0 + self.pan_y.clamp(-maxp, maxp)
    }

    /// Left edge (screen px): centered, then panned by `pan_x`, clamped.
    fn horizontal_left(&self, dw: f32, sw: f32) -> f32 {
        let maxp = ((dw - sw) / 2.0).max(0.0);
        (sw - dw) / 2.0 + self.pan_x.clamp(-maxp, maxp)
    }

    /// Displayed height of the current page under the active fit + zoom.
    fn current_display_h(&self) -> f32 {
        let sw = self.gpu.config.width.max(1) as f32;
        let sh = self.gpu.config.height.max(1) as f32;
        match self.cache.get(self.index) {
            Some(t) => {
                t.h as f32 * fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32) * self.zoom
            }
            None => sh,
        }
    }

    /// Clamp stored pan to the current page's overflow so dragging/zooming can't
    /// strand the view in an empty region.
    fn clamp_pan(&mut self) {
        let sw = self.gpu.config.width.max(1) as f32;
        let sh = self.gpu.config.height.max(1) as f32;
        if self.scroll_mode {
            let cw = sw * self.zoom;
            let mx = ((cw - sw) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            return;
        }
        if let Some(t) = self.cache.get(self.index) {
            let s = fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32) * self.zoom;
            let mx = ((t.w as f32 * s - sw) / 2.0).max(0.0);
            let my = ((t.h as f32 * s - sh) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            self.pan_y = self.pan_y.clamp(-my, my);
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
        let s = fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32) * self.zoom;
        let (dw, dh) = (t.w as f32 * s, t.h as f32 * s);
        Self::quad_from_px(
            0,
            idx,
            self.horizontal_left(dw, sw),
            self.vertical_top(dh, sh),
            dw,
            dh,
            sw,
            sh,
        )
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

        let (a, b) = layout::view_pages(self.layout, self.index, len, self.spread_offset);
        let ta = self.cache.get(a);
        // Wide (landscape) page is a double-spread image → show it alone.
        let force_single = ta.map_or(false, |t| t.w > t.h);
        let b = if force_single { None } else { b };
        let tb = b.and_then(|bi| self.cache.get(bi).map(|t| (bi, t)));

        match (ta, tb) {
            (Some(ta), Some((bi, tb))) => {
                // Facing pages share a display height. Size each to a common
                // reference height (its width following its own aspect ratio)
                // before fitting the pair to the window. Aspect ratios are
                // stable across decode resolutions, so if the two pages are
                // momentarily decoded at different heights — e.g. mid re-decode
                // after a fullscreen toggle / resize, where one updates a frame
                // before the other — neither page jumps size. (Identical to
                // per-pixel sizing when both heights already match.)
                let h_ref = ta.h.max(tb.h) as f32;
                let wa = ta.w as f32 * h_ref / ta.h.max(1) as f32;
                let wb = tb.w as f32 * h_ref / tb.h.max(1) as f32;
                let combined_w = wa + wb;
                let s = fit_scale(self.fit, sw, sh, combined_w, h_ref) * self.zoom;
                let x0 = self.horizontal_left(combined_w * s, sw);
                let dh = h_ref * s;
                // Screen order: LTR puts the lower index on the left; RTL reverses.
                let (l_idx, wl, r_idx, wr) = match self.direction {
                    Direction::Ltr => (a, wa, bi, wb),
                    Direction::Rtl => (bi, wb, a, wa),
                };
                let (dwl, dwr) = (wl * s, wr * s);
                vec![
                    Self::quad_from_px(0, l_idx, x0, self.vertical_top(dh, sh), dwl, dh, sw, sh),
                    Self::quad_from_px(1, r_idx, x0 + dwl, self.vertical_top(dh, sh), dwr, dh, sw, sh),
                ]
            }
            (Some(ta), None) => vec![self.single_quad(a, ta, sw, sh)],
            _ => {
                // Anchor not decoded yet: hold the last-drawn page if still cached.
                if let Some(li) = self.last_drawn
                    && let Some(t) = self.cache.get(li)
                {
                    return vec![self.single_quad(li, t, sw, sh)];
                }
                Vec::new()
            }
        }
    }

    fn page_display_h(&self, i: usize, sw: f32) -> f32 {
        let cw = sw * self.zoom; // strip content width (zoomable)
        match self.cache.get(i) {
            Some(t) => cw * (t.h as f32 / t.w as f32),
            None => cw * self.est_aspect,
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
        let cw = sw * self.zoom; // strip width (zoom); centered with horizontal pan
        let x = self.horizontal_left(cw, sw);
        let mut y = -self.top_offset;
        let mut i = self.index;
        let mut slot = 0;
        while i < len && y < sh && slot < MAX_QUADS {
            let dh = self.page_display_h(i, sw);
            if y + dh > 0.0 {
                if self.cache.get(i).is_some() {
                    quads.push(Self::quad_from_px(slot, i, x, y, cw, dh, sw, sh));
                    slot += 1;
                }
            }
            y += dh;
            i += 1;
        }
        quads
    }

    /// Decode up to `budget` not-yet-tried library cover thumbnails this frame
    /// and register them with egui.
    fn decode_thumbnails(&mut self, budget: usize) {
        let mut done = 0;
        for i in 0..self.library.volumes.len() {
            if done >= budget {
                break;
            }
            if self.library.volumes[i].thumb_tried {
                continue;
            }
            self.library.volumes[i].thumb_tried = true;
            done += 1;
            let Some(bytes) = cover_bytes(&self.library.volumes[i]) else {
                continue;
            };
            let img = match decode_and_downscale(&bytes, THUMB_H, &mut self.thumb_resizer) {
                Ok(img) => crate::decode::to_rgba_image(img), // egui samples RGBA
                Err(_) => continue,
            };
            // Library thumbnail (registered with egui, not stored in the page
            // cache) — its decode-target stamp is unused, so pass 0.
            let pt =
                PagePipeline::upload(&self.gpu.device, &self.gpu.queue, &img, &self.tex_pool, 0);
            let id = self.egui_renderer.register_native_texture(
                &self.gpu.device,
                &pt.view,
                wgpu::FilterMode::Linear,
            );
            self.library.volumes[i].thumb = Some(id);
            self.library.volumes[i].thumb_tex = Some(pt);
        }
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
        // (source, volume-key path, explicit start index)
        type Built = Result<(Arc<dyn PageSource>, PathBuf, Option<usize>), String>;
        let built: Built = if path.is_dir() {
            FolderSource::new(path)
                .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
                .map_err(|e| e.to_string())
        } else {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase());
            match ext.as_deref() {
                Some("cbz") | Some("zip") => ZipSource::new(path)
                    .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
                    .map_err(|e| e.to_string()),
                Some("cbr") | Some("rar") => RarSource::new(path)
                    .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
                    .map_err(|e| e.to_string()),
                Some("7z") | Some("cb7") => SevenzSource::new(path)
                    .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
                    .map_err(|e| e.to_string()),
                // A single image opens its containing folder, positioned at that
                // image, so you can seek forward/back within the folder.
                _ if is_image_ext(path) => {
                    let parent = path.parent().unwrap_or_else(|| Path::new("."));
                    FolderSource::new(parent)
                        .map(|s| {
                            let start = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .and_then(|n| s.index_of_name(n));
                            (Arc::new(s) as Arc<dyn PageSource>, parent.to_path_buf(), start)
                        })
                        .map_err(|e| e.to_string())
                }
                _ => Err(
                    "unsupported file type (open a folder, image, CBZ, CBR, or 7z)".to_string(),
                ),
            }
        };
        match built {
            Ok((source, key, start)) if source.len() > 0 => self.set_source(source, &key, start),
            Ok(_) => self.ui.status = "no images found".into(),
            Err(e) => self.ui.status = format!("open failed: {e}"),
        }
    }

    fn set_source(&mut self, source: Arc<dyn PageSource>, path: &Path, start: Option<usize>) {
        // Persist the previous volume's position before switching.
        if let Some(k) = self.volume_key.take() {
            self.settings.last_pages.insert(k, self.index);
        }
        let key = path.to_string_lossy().into_owned();
        self.spread_offset = self.settings.spread_offsets.get(&key).copied().unwrap_or(0) as usize;
        // Explicit start (e.g. a specific dropped image) wins; else CLI start
        // index; else the saved position.
        let idx = match start {
            Some(i) => i,
            None => {
                let resume = self.settings.last_pages.get(&key).copied().unwrap_or(0);
                if self.start_index > 0 {
                    self.start_index
                } else {
                    resume
                }
            }
        };
        self.start_index = 0;

        self.pool = Some(DecodePool::new(
            source.clone(),
            self.gpu.device.clone(),
            self.gpu.queue.clone(),
            self.tex_pool.clone(),
            self.downscaler.clone(),
            self.gpu_flag.clone(),
            self.target_h.clone(),
            WORKERS,
        ));
        self.cache.clear();
        self.failed.clear();
        self.last_drawn = None;
        self.info_for = None;
        self.nav_times.clear();
        self.index = idx.min(source.len() - 1);
        self.volume_key = Some(key);
        self.ui.opened = Some(path.to_path_buf());
        self.source = Some(source);
        self.library_view = false; // opening anything switches to the reader
        self.prefetch();
    }

    /// Desired decode height: the page's on-screen height (window height × zoom),
    /// quantized to avoid churn and clamped. Per-page it's further capped to the
    /// source height in `decode_and_downscale` (never upscale a page).
    /// Apply a view preset (number keys): set fit + layout (+ optional reading
    /// direction) at once, leave scroll, reset zoom/pan, re-anchor the spread
    /// pairing, persist, and refresh the prefetch window.
    fn apply_view(&mut self, fit: FitMode, spread: bool, dir: Option<Direction>) {
        self.scroll_mode = false;
        self.fit = fit;
        self.layout = if spread { Layout::Spread } else { Layout::Single };
        if let Some(d) = dir {
            self.direction = d;
        }
        self.index = layout::view_start(self.layout, self.index, self.spread_offset);
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
        self.settings.fit = fit_to_u8(self.fit);
        self.settings.layout_spread = self.layout == Layout::Spread;
        self.settings.direction_rtl = self.direction == Direction::Rtl;
        self.settings.scroll = false;
        config::save(&self.settings);
        self.prefetch();
    }

    /// Gather display info for page `index` (Tab overlay): reads the page bytes
    /// once and probes the header for resolution + format. Only called on a page
    /// change while the overlay is open, so the extra read is cheap.
    fn build_page_info(&self, index: usize) -> Vec<(String, String)> {
        let Some(src) = &self.source else {
            return Vec::new();
        };
        let name = src.name(index).to_string();
        let bytes = src.read_page(index).ok();
        let (res, fmt, size, color) = match &bytes {
            Some(b) => {
                let (w, h, detail) = probe(b);
                let res = if w == 0 || h == 0 {
                    "—".to_string()
                } else {
                    format!("{w} × {h}")
                };
                let color = crate::icc::extract_icc(b)
                    .as_deref()
                    .and_then(|p| crate::icc::describe(p))
                    .unwrap_or_else(|| "—".to_string());
                (res, detail, human_size(b.len() as u64), color)
            }
            None => (
                "—".to_string(),
                "—".to_string(),
                "—".to_string(),
                "—".to_string(),
            ),
        };
        let modified = src.modified(index).unwrap_or_else(|| "—".to_string());
        vec![
            ("File".to_string(), name),
            ("Page".to_string(), format!("{} / {}", index + 1, src.len())),
            ("Size".to_string(), size),
            ("Modified".to_string(), modified),
            ("Resolution".to_string(), res),
            ("Format".to_string(), fmt),
            ("Color".to_string(), color),
        ]
    }

    fn desired_target_h(&self) -> u32 {
        // 1:1 mode wants full source resolution; a huge target makes the per-page
        // `target_h.min(source_h)` clamp decode each page at its native height.
        if self.fit == FitMode::Actual && !self.scroll_mode {
            return u32::MAX;
        }
        let h = (self.gpu.config.height as f32 * self.zoom).round() as u32;
        let q = ((h + TARGET_QUANTUM / 2) / TARGET_QUANTUM) * TARGET_QUANTUM;
        q.clamp(MIN_TARGET, MAX_TARGET)
    }

    /// Re-point the decode target at the current display size. Debounced: only
    /// acts once the desired value settles (so a resize/zoom drag re-points once,
    /// not every frame). Does NOT clear the cache: `prefetch` re-decodes pages
    /// whose stamp is now stale while the old-resolution textures keep displaying,
    /// so there's no black frame (the swap is in place). Normal page-flipping
    /// never changes the target, so it never triggers a re-decode.
    fn update_target_h(&mut self) {
        let desired = self.desired_target_h();
        if desired != self.pending_target {
            self.pending_target = desired; // still settling
            return;
        }
        self.target_h.store(desired, Ordering::Relaxed);
    }

    /// Recompute the desired prefetch window and hand it to the pool. Queues a
    /// page if it's missing OR cached at a stale decode target (after a
    /// zoom/resize), so stale pages re-decode at the new resolution and overwrite
    /// in place — the old texture keeps displaying until the new one lands.
    fn prefetch(&mut self) {
        let fwd = self.dynamic_fwd();
        let cur = self.target_h.load(Ordering::Relaxed);
        let (Some(src), Some(pool)) = (&self.source, &self.pool) else {
            return;
        };
        let desired: Vec<usize> = desired_window(self.index, src.len(), fwd, BACK)
            .into_iter()
            .filter(|i| {
                !self.failed.contains(i)
                    && self.cache.get(*i).map_or(true, |p| p.target_h != cur)
            })
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
        // Track the decode resolution to the display size (debounced re-decode).
        self.update_target_h();
        // Keep the scroll anchor valid as page heights resolve, then refresh work.
        if self.scroll_mode {
            self.normalize();
        }
        self.prefetch();

        // Decide what to draw this frame (library grid hides the page).
        let quads = if self.library_view {
            Vec::new()
        } else if self.scroll_mode {
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
        self.ui.turbo_label = if self.settings.turbo { "turbo" } else { "vsync" };
        // Build the Tab info overlay text, reading the source once per page change.
        if self.ui.info_open && !self.library_view && self.info_for != Some(self.index) {
            self.ui.info = self.build_page_info(self.index);
            self.info_for = Some(self.index);
        }
        // Hide the top bar in fullscreen, revealing it when the cursor is at the top edge.
        let fullscreen = self.window.fullscreen().is_some();
        let reveal = 48.0 * self.window.scale_factor() as f32;
        self.ui.show_bar = !fullscreen || (self.cursor_y as f32) < reveal;
        // Edge hover arrows: only in page-flip reader mode, below the top bar,
        // while the cursor is inside the window.
        let win_w = self.gpu.config.width.max(1) as f32;
        let edge = win_w * EDGE_FRAC;
        let in_reader = self.source.is_some() && !self.library_view && !self.scroll_mode;
        let below_bar = (self.cursor_y as f32) >= reveal;
        let cx = self.cursor_x as f32;
        self.ui.hover_left = in_reader && self.cursor_in_window && below_bar && cx < edge;
        self.ui.hover_right =
            in_reader && self.cursor_in_window && below_bar && cx > win_w - edge;
        if let Some(src) = &self.source {
            let len = src.len();
            let anchor = if self.scroll_mode {
                self.index
            } else {
                layout::view_pages(self.layout, self.index, len, self.spread_offset).0
            };
            let loading = !self.cache.contains(anchor);
            if !loading {
                self.last_drawn = Some(anchor);
            }
            self.ui.status = format!(
                "{}/{}{}{}",
                self.index + 1,
                len,
                if loading { "  …" } else { "" },
                if self.jump { "  [jump]" } else { "  [step]" }
            );
            // Show the centered spinner only after this page's decode has been
            // pending a beat, so fast flips don't flash it. The timer restarts
            // whenever the anchor changes, so a quick page reached at the end of
            // a slow-seek streak still gets its full grace period.
            if loading {
                let since = match self.loading_pending {
                    Some((a, t)) if a == anchor => t,
                    _ => {
                        let t = Instant::now();
                        self.loading_pending = Some((anchor, t));
                        t
                    }
                };
                self.ui.loading = since.elapsed() >= LOADING_INDICATOR_DELAY;
            } else {
                self.loading_pending = None;
                self.ui.loading = false;
            }
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
        if self.library_view {
            self.decode_thumbnails(THUMB_BUDGET);
        }
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ui_state = &mut self.ui;
        let lib = &self.library;
        let library_view = self.library_view;
        let full_output = self
            .egui_ctx
            .run(raw_input, |ctx| ui::chrome(ctx, ui_state, lib, library_view));
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
        if std::mem::take(&mut self.ui.req_toggle_present) {
            self.apply_action(Action::TogglePresent);
        }
        if let Some(root) = self.ui.pending_library.take() {
            for v in &self.library.volumes {
                if let Some(id) = v.thumb {
                    self.egui_renderer.free_texture(&id);
                }
            }
            self.library = Library::scan(&root);
            self.library_view = true;
            self.settings.library_root = Some(root.to_string_lossy().into_owned());
            config::save(&self.settings);
        }
        if std::mem::take(&mut self.ui.req_toggle_library) {
            if !self.library.volumes.is_empty() {
                self.library_view = !self.library_view;
            }
        }
        if let Some(i) = self.ui.clicked_volume.take()
            && let Some(v) = self.library.volumes.get(i)
        {
            let path = v.path.clone();
            self.library_view = false;
            self.open(&path);
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

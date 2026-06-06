//! Application: winit `ApplicationHandler`, owns GPU + egui + reader state.
//! M1.3: async decode pool + bounded cache + forward prefetch → hitch-free
//! navigation. The current page is drawn from the cache; if a target isn't ready
//! yet the last-drawn page is held (no flicker).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use fast_image_resize::Resizer;

use yosh_engine::cache::PageCache;
use crate::config;
use yosh_engine::decode::decode_and_downscale;
use crate::gpu::Gpu;
use crate::library::{cover_bytes, Library};
use yosh_engine::page::{fit_scale, FitMode, PagePipeline, PageTexture, MAX_QUADS};
use yosh_engine::pool::{DecodePool, Msg};
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, RarSource, SevenzSource, ZipSource};
use yosh_engine::layout::{self, Layout};
use yosh_engine::prefetch::desired_window;
use yosh_engine::reader::{
    anchor_native_scale, clamp_zoom_multiplier, next_zoom_preset, quad_from_px, zoom_presets,
    Direction, Quad, Viewport, MAX_ZOOM_PCT, MIN_TARGET, MIN_ZOOM_PCT,
};
use yosh_engine::texpool::TexturePool;
use crate::ui::{self, UiState};
use crate::update;

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
/// How long a transient toast (boundary hit, zoom level) stays on screen.
const TOAST_DURATION: Duration = Duration::from_millis(1500);
/// Minimum time a zoomed-page scroll must dwell parked at the top/bottom edge
/// before a further scroll flips the page — turns the edge into a perceptible
/// hard stop instead of an instant jump to the next page.
const EDGE_DWELL: Duration = Duration::from_millis(350);

pub struct App {
    initial_path: Option<PathBuf>,
    start_index: usize,
    state: Option<State>,
}

/// Playback state for the animation (GIF/WebP) currently in view (BandiView-style
/// mini controls). `page` binds the controls to one page; when the viewed
/// animation changes it rebinds and resets. Non-controlled animations keep
/// free-running on the wall clock.
struct Playback {
    /// User hid the control panel (toggle with the key); playback continues.
    hidden: bool,
    /// Play/pause. Paused (or stepping) freezes on `frame`.
    playing: bool,
    /// The page index these controls govern (None when no animation is in view).
    page: Option<usize>,
    /// Current frame index (0-based).
    frame: usize,
    /// When `frame` was last advanced — drives play timing against frame delays.
    last: Instant,
}

impl Default for Playback {
    fn default() -> Self {
        Playback { hidden: false, playing: true, page: None, frame: 0, last: Instant::now() }
    }
}

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    /// Surface size mirrored from `gpu.config` for the reading math — refreshed at
    /// the top of `render()` and on resize, so it is always value-equal to the
    /// live surface. The seam that lets the reading methods stop reading `gpu`
    /// directly, ahead of their move into `yosh_engine::reader`.
    viewport: Viewport,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui: UiState,

    page_pipeline: PagePipeline,
    source: Option<Arc<dyn PageSource>>,
    pool: Option<DecodePool>,
    cache: PageCache,
    /// Pages whose decode errored, mapped to the error message (shown to the user).
    failed: HashMap<usize, String>,
    index: usize,
    start_index: usize,
    last_drawn: Option<usize>,

    fit: FitMode,
    layout: Layout,
    spread_offset: usize, // spread pairing parity (0 or 1), per-volume
    /// Page rotation in 90° CW steps (0..=3). Session-global, applied to
    /// single-page draws only; reset to 0 when a new volume opens.
    rotation: u8,
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
    /// Last observed window geometry (outer x/y, inner w/h, physical px) while in
    /// the normal — not maximized, not fullscreen — state. Seeded from the saved
    /// settings so a session spent entirely maximized still persists the prior
    /// restored rect; updated on move/resize; written back on exit.
    win_geom: Option<(i32, i32, u32, u32)>,
    volume_key: Option<String>,
    tex_pool: Arc<TexturePool>,
    /// Decode-view debounce: the last-seen `(surface_w, surface_h, zoom)`. Once it
    /// stops changing across frames the view is "settled" and target-change
    /// re-decodes are allowed — so a resize/zoom drag re-decodes once it lands, not
    /// every frame. Page-flipping doesn't change it, so it never re-decodes.
    pending_view: (u32, u32, f32),
    view_settled: bool,
    /// True while we've already logged a "settled view is GPU-downscaling" warning
    /// for the current occurrence, so the tripwire fires once per episode, not per
    /// frame. Cleared as soon as the view returns to 1:1/upscale.
    gpu_downscale_warned: bool,
    /// Page index the Tab info overlay text was built for (None = rebuild needed).
    info_for: Option<usize>,
    /// The anchor page currently waiting to decode and when that wait began,
    /// used to delay the loading spinner *per page* (so a fast page reached at
    /// the end of a slow-seek streak still gets its own grace period). None as
    /// soon as the anchor is ready.
    loading_pending: Option<(usize, Instant)>,
    /// Transient on-screen message (boundary reached, zoom level) + when it was
    /// raised; cleared after `TOAST_DURATION`.
    toast: Option<(String, Instant)>,
    /// When the zoomed-page wheel-pan first parked at the top/bottom edge (None
    /// when not at an edge). Gates the hard-stop dwell before flipping pages.
    pan_edge_at: Option<Instant>,
    /// Fixed origin for animation timing. A single shared clock is correct —
    /// every animated page derives its current frame from the same wall time, so
    /// all animations loop in step and the render loop stays stateless per page.
    anim_origin: Instant,
    /// Mini play/pause/step controls for the animation (GIF/WebP) currently in view.
    playback: Playback,
    /// Last window title pushed to the OS, so `render()` only calls `set_title`
    /// when the computed title actually changes (it's recomputed every frame).
    last_title: String,

    library: Library,
    library_view: bool,
    thumb_resizer: Resizer,

    // Auto-update: a background thread checks the latest public release on
    // launch; the UI offers a one-click in-place update.
    update_rx: Option<std::sync::mpsc::Receiver<update::Update>>,
    update: Option<update::Update>,
    update_apply_rx: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
    updating: bool,
    update_error: Option<String>,

    // Async open: source construction (the I/O-bound part of opening a volume)
    // runs on a background thread so a slow network-share open never freezes the
    // UI. Each `open` bumps `open_gen`; `render` applies only the newest result,
    // so rapid `[`/`]` supersede in-flight opens. `opening_key` tracks the
    // most-recently-targeted volume so successive `[`/`]` advance from the pending
    // target rather than repeating the same neighbor.
    open_gen: u64,
    opening: bool,
    opening_key: Option<PathBuf>,
    open_tx: std::sync::mpsc::Sender<(u64, Built)>,
    open_rx: std::sync::mpsc::Receiver<(u64, Built)>,
    /// Cached parent-dir scan for `[`/`]` — (parent dir, want_folder, natural-sort
    /// paths). Warmed in the background on open so the first jump doesn't pay it.
    sib_cache: Option<(PathBuf, bool, Vec<PathBuf>)>,
    sib_tx: std::sync::mpsc::Sender<(PathBuf, bool, Vec<PathBuf>)>,
    sib_rx: std::sync::mpsc::Receiver<(PathBuf, bool, Vec<PathBuf>)>,
}

/// Result of constructing a page source: `(source, volume-key path, explicit
/// start index)`, or an error message. Built off-thread by `build_source`.
type Built = Result<(Arc<dyn PageSource>, PathBuf, Option<usize>), String>;

/// Construct a page source for `path` (folder / archive / single image). Pure
/// I/O, touches no app state, so it runs on a background thread — a slow
/// (e.g. network-share) open never blocks the UI. The result is handed back to
/// the main thread and applied via `set_source`.
fn build_source(path: &Path) -> Built {
    if path.is_dir() {
        return FolderSource::new(path)
            .map(|s| (Arc::new(s) as Arc<dyn PageSource>, path.to_path_buf(), None))
            .map_err(|e| e.to_string());
    }
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
        // A single image opens its containing folder, positioned at that image,
        // so you can seek forward/back within the folder.
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
        _ => Err("unsupported file type (open a folder, image, CBZ, CBR, or 7z)".to_string()),
    }
}

/// True if the saved window rect overlaps any currently-connected monitor, so
/// we don't restore a window onto a display that's been unplugged (which would
/// strand it off-screen and unreachable). On overlap failure we drop only the
/// saved position; the size is still honored and the OS places the window.
fn geometry_on_screen(event_loop: &ActiveEventLoop, w: &config::WindowState) -> bool {
    let (wx0, wy0) = (w.x, w.y);
    let (wx1, wy1) = (w.x + w.w as i32, w.y + w.h as i32);
    event_loop.available_monitors().any(|m| {
        let mp = m.position();
        let ms = m.size();
        let (mx1, my1) = (mp.x + ms.width as i32, mp.y + ms.height as i32);
        wx0 < mx1 && wx1 > mp.x && wy0 < my1 && wy1 > mp.y
    })
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
/// Walk an ISO-BMFF (AVIF/HEIF) box tree to the first `ispe` (image spatial
/// extents) box and return its (width, height) — pure parsing, no decode, so it
/// works regardless of the `avif` feature. `meta` is a FullBox (4-byte
/// version/flags before its children); `iprp`/`ipco` are plain containers; the
/// `ispe` payload is version/flags(4) + width(4) + height(4), all big-endian.
fn iso_box_dims(b: &[u8]) -> Option<(u32, u32)> {
    // Find a child box by 4-byte type, returning its payload (after the header).
    fn find<'a>(mut b: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
        while b.len() >= 8 {
            let size = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
            let (header, end) = match size {
                1 => {
                    // 64-bit largesize follows the type.
                    let s = u64::from_be_bytes(b.get(8..16)?.try_into().ok()?) as usize;
                    (16, s)
                }
                0 => (8, b.len()), // extends to end
                s => (8, s),
            };
            if end < header || end > b.len() {
                return None;
            }
            if &b[4..8] == want {
                return Some(&b[header..end]);
            }
            b = &b[end..];
        }
        None
    }
    let meta = find(b, b"meta")?.get(4..)?; // skip meta's FullBox version/flags
    let ispe = find(find(find(meta, b"iprp")?, b"ipco")?, b"ispe")?;
    let w = u32::from_be_bytes(ispe.get(4..8)?.try_into().ok()?);
    let h = u32::from_be_bytes(ispe.get(8..12)?.try_into().ok()?);
    Some((w, h))
}

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
    // PSD / PSB (Photoshop): header is big-endian — rows@14, cols@18 (u32),
    // depth@22 and color mode@24 (u16).
    if b.len() >= 26 && &b[0..4] == b"8BPS" {
        let be32 = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let h = be32(14);
        let w = be32(18);
        let mode = match be16(24) {
            0 => "bitmap",
            1 => "grayscale",
            2 => "indexed",
            3 => "RGB",
            4 => "CMYK",
            7 => "multichannel",
            8 => "duotone",
            9 => "Lab",
            _ => "?",
        };
        return (w, h, format!("PSD · {}-bit {}", be16(22), mode));
    }
    // ICO: report the largest entry's size + how many layers it holds.
    if b.len() >= 6 && b[0..4] == [0x00, 0x00, 0x01, 0x00] {
        let count = le16(4) as usize;
        let dim = |v: u8| if v == 0 { 256 } else { v as u32 };
        let (mut mw, mut mh) = (0u32, 0u32);
        for i in 0..count {
            let off = 6 + i * 16;
            if off + 1 < b.len() {
                mw = mw.max(dim(b[off]));
                mh = mh.max(dim(b[off + 1]));
            }
        }
        return (mw, mh, format!("ICO · {count} layer{}", if count == 1 { "" } else { "s" }));
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
            b"VP8L" => {
                // After the 0x2F signature byte: 14-bit (width-1) then 14-bit
                // (height-1), LSB-first, packed across b[21..25].
                if b.len() >= 25 {
                    let bits = b[21] as u32
                        | (b[22] as u32) << 8
                        | (b[23] as u32) << 16
                        | (b[24] as u32) << 24;
                    return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, "WebP".to_string());
                }
                return (0, 0, "WebP".to_string());
            }
            _ => return (0, 0, "WebP".to_string()),
        }
    }
    // JPEG XL: bare codestream (FF 0A) or ISOBMFF box (".../JXL ..."). Parse just
    // the header via jxl-oxide for exact dimensions + color type (no pixel decode).
    if (b.len() >= 2 && b[0] == 0xFF && b[1] == 0x0A) || (b.len() >= 12 && &b[4..8] == b"JXL ") {
        if let Ok(img) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(b)) {
            let color = match img.pixel_format() {
                jxl_oxide::PixelFormat::Gray => "grayscale",
                jxl_oxide::PixelFormat::Graya => "grayscale+alpha",
                jxl_oxide::PixelFormat::Rgb => "RGB",
                jxl_oxide::PixelFormat::Rgba => "RGBA",
                jxl_oxide::PixelFormat::Cmyk => "CMYK",
                jxl_oxide::PixelFormat::Cmyka => "CMYK+alpha",
            };
            return (img.width(), img.height(), format!("JPEG XL · {color}"));
        }
        return (0, 0, "JPEG XL".to_string());
    }
    // AVIF / HEIF (ISO-BMFF): walk the box tree to the `ispe` for dimensions.
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        let (w, h) = iso_box_dims(b).unwrap_or((0, 0));
        let label = if matches!(&b[8..12], b"avif" | b"avis") { "AVIF" } else { "HEIF" };
        return (w, h, label.to_string());
    }
    // Generic fallback: let the `image` crate identify the format and read just the
    // dimensions (no full decode). Covers TIFF/TGA/DDS/EXR/HDR/QOI/PNM and anything
    // else the crate guesses by content. TGA has no magic bytes, so fall back to it
    // when content-guessing finds nothing (mirrors decode_other).
    if let Ok(guessed) = image::ImageReader::new(std::io::Cursor::new(b)).with_guessed_format() {
        let reader = if guessed.format().is_some() {
            guessed
        } else {
            image::ImageReader::with_format(std::io::Cursor::new(b), image::ImageFormat::Tga)
        };
        let label = match reader.format() {
            Some(image::ImageFormat::Tiff) => "TIFF".to_string(),
            Some(image::ImageFormat::Tga) => "TGA".to_string(),
            Some(image::ImageFormat::Dds) => "DDS".to_string(),
            Some(image::ImageFormat::OpenExr) => "OpenEXR".to_string(),
            Some(image::ImageFormat::Hdr) => "Radiance HDR".to_string(),
            Some(image::ImageFormat::Qoi) => "QOI".to_string(),
            Some(image::ImageFormat::Pnm) => "PNM".to_string(),
            Some(f) => format!("{f:?}"),
            None => "image".to_string(),
        };
        if let Ok((w, h)) = reader.into_dimensions() {
            return (w, h, label);
        }
        return (0, 0, label);
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

/// Bind the exe's own embedded icon to the window on Windows — the real taskbar fix.
///
/// `with_window_icon` / `with_taskbar_icon` build the icon from `yosh.png`, which
/// is 256x255 (non-square) and shares one HICON for both the small (title-bar) and
/// big (taskbar) slots. The small slot tolerates it; the taskbar's large slot does
/// not and falls back to the generic app icon. The `.ico` the build script embeds
/// in the exe is square and multi-resolution — the exact icon Explorer, the title
/// bar, and the installer already render correctly — so we pull it straight out of
/// the running exe with `ExtractIconExW` (large + small at the system icon sizes)
/// and set it as ICON_BIG / ICON_SMALL, plus the window-class icons as a backstop.
#[cfg(windows)]
fn bind_exe_icon(window: &Window) {
    use std::os::windows::ffi::OsStrExt;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::Shell::ExtractIconExW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GCLP_HICON, GCLP_HICONSM, ICON_BIG, ICON_SMALL, SendMessageW, SetClassLongPtrW, WM_SETICON,
    };

    let hwnd = match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Win32(h)) => h.hwnd.get() as *mut core::ffi::c_void,
        _ => return,
    };
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let wide: Vec<u16> = exe.as_os_str().encode_wide().chain(std::iter::once(0)).collect();

    let mut large: *mut core::ffi::c_void = std::ptr::null_mut();
    let mut small: *mut core::ffi::c_void = std::ptr::null_mut();
    // Icon group 0 = the application icon embedded by the build script.
    if unsafe { ExtractIconExW(wide.as_ptr(), 0, &mut large, &mut small, 1) } == 0 {
        return;
    }
    // The extracted HICONs are handed to the window (WM_SETICON / class icon) and
    // must outlive bind_exe_icon — the window references them, it doesn't copy. So
    // they are intentionally not destroyed here; this single, process-lifetime
    // window owns them and the OS reclaims them at exit.
    unsafe {
        if !large.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_BIG as usize, large as isize);
            SetClassLongPtrW(hwnd, GCLP_HICON, large as isize);
        }
        if !small.is_null() {
            SendMessageW(hwnd, WM_SETICON, ICON_SMALL as usize, small as isize);
            SetClassLongPtrW(hwnd, GCLP_HICONSM, small as isize);
        }
    }
}

/// Open the OS file browser at `path`'s containing folder and select `path`.
/// On Windows this uses `SHOpenFolderAndSelectItems`, which reuses an existing
/// Explorer window showing that folder instead of spawning a new one each time
/// (unlike `explorer.exe /select,`). Runs on a short-lived thread because it
/// initializes COM and may launch/raise Explorer — neither belongs on the UI thread.
#[cfg(windows)]
fn reveal_in_explorer(path: &std::path::Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, CoInitializeEx, CoUninitialize,
    };
    use windows_sys::Win32::UI::Shell::{ILCreateFromPathW, ILFree, SHOpenFolderAndSelectItems};

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    std::thread::spawn(move || unsafe {
        let _ = CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32);
        let pidl = ILCreateFromPathW(wide.as_ptr());
        if !pidl.is_null() {
            // cidl = 0 + apidl = null → open the item's parent and select the item.
            let _ = SHOpenFolderAndSelectItems(pidl, 0, std::ptr::null(), 0);
            ILFree(pidl);
        }
        CoUninitialize();
    });
}

/// Non-Windows fallback: best-effort open the containing folder (no selection),
/// so the Linux build stays functional and green.
#[cfg(not(windows))]
fn reveal_in_explorer(path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
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
        let mut settings = config::load();
        let attrs = Window::default_attributes()
            .with_title("yosh")
            .with_window_icon(window_icon());
        // Restore the last window geometry (size/position/maximized) if we saved
        // one; otherwise fall back to the default launch size. The saved position
        // is only reapplied when it still lands on a connected monitor, so a
        // window saved on a now-disconnected display doesn't open off-screen.
        let attrs = match settings.window {
            Some(w) => {
                let mut a = attrs.with_inner_size(PhysicalSize::new(w.w.max(1), w.h.max(1)));
                if geometry_on_screen(event_loop, &w) {
                    a = a.with_position(PhysicalPosition::new(w.x, w.y));
                }
                a.with_maximized(w.maximized)
            }
            None => attrs.with_inner_size(winit::dpi::LogicalSize::new(1100.0, 1500.0)),
        };
        // `with_window_icon` sets only the small (title-bar) icon; the taskbar
        // reads ICON_BIG, which winit exposes separately on Windows.
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_taskbar_icon(window_icon())
        };
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        // Override winit's PNG-derived icon with the exe's square embedded .ico so
        // the taskbar (ICON_BIG) shows the logo instead of the generic app icon.
        #[cfg(windows)]
        bind_exe_icon(&window);
        let gpu = Gpu::new(window.clone());

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
        let tex_pool = Arc::new(TexturePool::new());

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
        // Kick off a background update check against the public GitHub releases.
        let (update_tx, update_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            if let Some(u) = update::check() {
                let _ = update_tx.send(u);
            }
        });
        // Channels for background archive opens and sibling-volume prescans.
        let (open_tx, open_rx) = std::sync::mpsc::channel();
        let (sib_tx, sib_rx) = std::sync::mpsc::channel();
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
            failed: HashMap::new(),
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
            rotation: 0,
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
            win_geom: settings.window.map(|w| (w.x, w.y, w.w, w.h)),
            settings,
            volume_key: None,
            tex_pool,
            viewport: Viewport::default(),
            pending_view: (0, 0, 1.0),
            view_settled: false,
            gpu_downscale_warned: false,
            info_for: None,
            loading_pending: None,
            toast: None,
            pan_edge_at: None,
            anim_origin: Instant::now(),
            playback: Playback::default(),
            last_title: String::new(),
            library,
            library_view,
            thumb_resizer: Resizer::new(),
            update_rx: Some(update_rx),
            update: None,
            update_apply_rx: None,
            updating: false,
            update_error: None,
            open_gen: 0,
            opening: false,
            opening_key: None,
            open_tx,
            open_rx,
            sib_cache: None,
            sib_tx,
            sib_rx,
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
            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                // Keep the reading viewport in lock-step with the surface so an
                // input event between renders sees the new size (as it did when
                // these reads came straight from `gpu.config`).
                state.viewport = Viewport {
                    w: state.gpu.config.width,
                    h: state.gpu.config.height,
                };
                state.record_window_geometry();
            }
            WindowEvent::Moved(_) => state.record_window_geometry(),
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
            // The seekbar isn't scrollable, so don't let hovering it swallow the
            // wheel — keep routing it to the reader (clicks/drags still seek).
            WindowEvent::MouseWheel { delta, .. } if !response.consumed || state.ui.seek_hovered => {
                state.on_wheel(delta)
            }
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
    ToggleSeekbar,
    ToggleAnimBar,
    PrevVolume,
    NextVolume,
    ToggleJump,
    Rotate,
    ShowInExplorer,
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
            KeyCode::KeyJ => return Some(Action::ToggleJump),
            KeyCode::KeyB => return Some(Action::ToggleSeekbar),
            KeyCode::KeyG => return Some(Action::ToggleAnimBar),
            KeyCode::KeyE => return Some(Action::ShowInExplorer),
            KeyCode::KeyR => return Some(Action::Rotate),
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
                    let vh = self.viewport.h as f32;
                    self.scroll_by(vh * 0.9);
                } else {
                    self.step(1);
                }
            }
            Action::Backward => {
                if self.scroll_mode {
                    let vh = self.viewport.h as f32;
                    self.scroll_by(-vh * 0.9);
                } else {
                    self.step(-1);
                }
            }
            // In RTL, "left" advances the story; in LTR, "right" does. (Page-flip only.)
            Action::Right if !self.scroll_mode => {
                self.step(if self.direction == Direction::Ltr { 1 } else { -1 });
            }
            Action::Left if !self.scroll_mode => {
                self.step(if self.direction == Direction::Ltr { -1 } else { 1 });
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
                    let sw = self.viewport.w.max(1) as f32;
                    let sh = self.viewport.h.max(1) as f32;
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
            Action::ZoomIn => self.zoom_to_preset(true),
            Action::ZoomOut => self.zoom_to_preset(false),
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
                self.toast(format!("Direction: {}", self.direction.label()));
            }
            Action::ToggleLayout => {
                self.layout = self.layout.toggled();
                // Snap to the current view's anchor so pairing is consistent.
                self.index = layout::view_start(self.layout, self.index, self.spread_offset);
                self.pan_y = 0.0;
                self.settings.layout_spread = self.layout == Layout::Spread;
                config::save(&self.settings);
                self.prefetch();
                self.toast(format!("Layout: {}", self.layout.label()));
            }
            Action::ToggleScroll => {
                self.scroll_mode = !self.scroll_mode;
                self.top_offset = 0.0;
                self.settings.scroll = self.scroll_mode;
                config::save(&self.settings);
                self.prefetch();
                self.toast(if self.scroll_mode {
                    "Scroll mode"
                } else {
                    "Page-flip mode"
                });
            }
            Action::ToggleHelp => self.ui.help_open = !self.ui.help_open,
            Action::ToggleInfo => {
                self.ui.info_open = !self.ui.info_open;
                self.info_for = None; // rebuild the overlay text next render
            }
            Action::ToggleSeekbar => {
                self.settings.seekbar_enabled = !self.settings.seekbar_enabled;
                config::save(&self.settings);
            }
            Action::ToggleAnimBar => self.playback.hidden = !self.playback.hidden,
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
                self.toast(format!("Spread offset: {}", self.spread_offset));
            }
            Action::PrevVolume => self.jump_volume(-1),
            Action::NextVolume => self.jump_volume(1),
            Action::ToggleJump => {
                self.jump = !self.jump;
                self.settings.jump = self.jump;
                config::save(&self.settings);
                self.toast(if self.jump { "Jump mode" } else { "Step mode" });
            }
            Action::Rotate => {
                self.rotation = (self.rotation + 1) % 4;
                // Recenter: the rotated box has different bounds, so any prior pan
                // would now be out of range.
                self.pan_x = 0.0;
                self.pan_y = 0.0;
                self.prefetch(); // re-decode at the rotation-aware target (1:1)
                self.toast(format!("Rotation: {}\u{00b0}", self.rotation as u32 * 90));
            }
            Action::ShowInExplorer => self.reveal_current(),
            // Esc → quit is intercepted in `window_event` (needs the event loop),
            // so it never reaches here.
            Action::Quit => {}
        }
    }

    /// Raise a transient on-screen toast (boundary reached, zoom level).
    fn toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    /// "Show in Explorer" (key `E`): open the containing folder of the current
    /// item and select it. For a folder/single-image volume that's the current
    /// page's image file; for an archive it's the archive file itself.
    fn reveal_current(&mut self) {
        let Some(key) = self.volume_key.clone() else {
            self.toast("Nothing open");
            return;
        };
        let base = PathBuf::from(&key);
        let target = if base.is_dir() {
            // Folder (incl. single-image opens, which open the parent folder):
            // select the current page's file. `name(index)` is a flat file name.
            match self.source.as_ref() {
                Some(s) if s.len() > 0 => base.join(s.name(self.index)),
                _ => base,
            }
        } else {
            base // archive (or single file): select it in its folder
        };
        reveal_in_explorer(&target);
        self.toast("Shown in Explorer");
    }

    /// Flip one view in `dir`. Returns `true` if the position actually changed.
    /// At the first/last page it raises a toast and returns `false`; while the
    /// current page is still decoding in step mode it just returns `false`.
    fn step(&mut self, dir: i64) -> bool {
        let Some(src) = &self.source else { return false };
        let len = src.len();
        if len == 0 {
            return false;
        }
        // "Step" seek (default; toggle "jump" with J): don't flip while the
        // current page is still decoding, so you see every page instead of
        // skipping past it. "Jump" skips ahead for fast long-distance seeks. A
        // *failed* page never lands in the cache, so allow stepping past it —
        // otherwise next/prev gets stuck on an unopenable page.
        if !self.jump {
            let cur = layout::view_pages(self.layout, self.index, len, self.spread_offset).0;
            if !self.cache.contains(cur) && !self.failed.contains_key(&cur) {
                return false;
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
            true
        } else {
            // Nowhere to go — let the reader know why seeking did nothing.
            self.toast(if dir > 0 { "Last page" } else { "First page" });
            false
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
        // Base "current" on the pending target if an open is in flight, so a
        // second `[`/`]` advances from the not-yet-loaded neighbor instead of
        // repeating it.
        let cur = match self.opening_key.clone() {
            Some(p) => p,
            None => match &self.volume_key {
                Some(k) => PathBuf::from(k),
                None => return,
            },
        };
        let Some(parent) = cur.parent().map(|p| p.to_path_buf()) else {
            return;
        };
        let want_folder = cur.is_dir();
        // Use the background-warmed cache when it matches this folder; otherwise
        // scan synchronously this once (cold cache, e.g. a very first `[`/`]`).
        let hit = matches!(&self.sib_cache, Some((p, wf, _)) if *p == parent && *wf == want_folder);
        if !hit {
            let vols = crate::library::sibling_volumes(&cur);
            self.sib_cache = Some((parent, want_folder, vols));
        }
        let sibs = &self.sib_cache.as_ref().unwrap().2;
        let cur_name = cur.file_name();
        let Some(idx) = sibs.iter().position(|p| p.file_name() == cur_name) else {
            return;
        };
        let target = idx as i64 + delta;
        if target < 0 {
            self.toast("First book");
            return;
        }
        let next = sibs.get(target as usize).cloned();
        match next {
            Some(path) => self.open(&path),
            None => self.toast("Last book"),
        }
    }

    /// Persist the current position + settings (called on close).
    /// Snapshot the window's restored (non-maximized, non-fullscreen) geometry.
    /// Maximizing/fullscreen reports the filled-screen rect, which we don't want
    /// as the restore target, so those states are skipped — `win_geom` keeps the
    /// last normal rect, which is exactly what we persist.
    fn record_window_geometry(&mut self) {
        if self.window.is_maximized() || self.window.fullscreen().is_some() {
            return;
        }
        let Ok(pos) = self.window.outer_position() else {
            return;
        };
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.win_geom = Some((pos.x, pos.y, size.width, size.height));
        }
    }

    fn persist(&mut self) {
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), self.index);
        }
        // Save geometry + the current maximized flag. `win_geom` already holds the
        // restored rect (it's only updated while normal), so an un-maximize after
        // restart returns to the right size/position.
        if let Some((x, y, w, h)) = self.win_geom {
            self.settings.window = Some(config::WindowState {
                x,
                y,
                w,
                h,
                maximized: self.window.is_maximized(),
            });
        }
        config::save(&self.settings);
    }

    /// After a successful self-update: persist, launch the freshly-replaced exe
    /// (reopening the current volume), and exit.
    fn relaunch(&mut self) -> ! {
        self.persist();
        if let Ok(exe) = std::env::current_exe() {
            let mut cmd = std::process::Command::new(exe);
            if let Some(k) = &self.volume_key {
                cmd.arg(k);
            }
            let _ = cmd.spawn();
        }
        std::process::exit(0);
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
        // Vertical pan in px. At the top/bottom edge, hard-stop first: a scroll
        // that would overshoot the edge just parks at it; only a *further* scroll
        // once already parked flips the page. So scrolling to the end of a zoomed
        // page stops there instead of immediately jumping to the next page — you
        // have to keep scrolling past the stop to advance. Only reset the pan when
        // a flip actually happened (else the first/last page snaps to its edge).
        let sh = self.viewport.h.max(1) as f32;
        let maxp = ((self.current_display_h() - sh) / 2.0).max(0.0);
        let cur = self.pan_y.clamp(-maxp, maxp);
        let next = cur + dy * 80.0;
        let now = Instant::now();
        // True once we've been parked at an edge long enough that a further
        // scroll should flip (the hard stop the user has to keep scrolling past).
        let dwelt = self.pan_edge_at.is_some_and(|t| now.duration_since(t) >= EDGE_DWELL);
        if next > maxp + 0.5 {
            if cur >= maxp - 0.5 {
                // Parked at the top: flip to the previous page only after dwelling.
                if dwelt {
                    self.pan_edge_at = None;
                    self.pan_y = if self.step(-1) { -1.0e6 } else { maxp };
                } else {
                    self.pan_y = maxp; // hold the stop
                    self.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.pan_y = maxp; // just reached the top edge -> park + start dwell
                self.pan_edge_at = Some(now);
            }
        } else if next < -maxp - 0.5 {
            if cur <= -maxp + 0.5 {
                // Parked at the bottom: flip to the next page only after dwelling.
                if dwelt {
                    self.pan_edge_at = None;
                    self.pan_y = if self.step(1) { 1.0e6 } else { -maxp };
                } else {
                    self.pan_y = -maxp; // hold the stop
                    self.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.pan_y = -maxp; // just reached the bottom edge -> park + start dwell
                self.pan_edge_at = Some(now);
            }
        } else {
            self.pan_y = next;
            self.pan_edge_at = None; // panning within the page
        }
    }

    /// A clean click: the left/right edge strips flip pages; the wide middle
    /// does nothing on a single click but toggles fullscreen on a double-click.
    fn on_click(&mut self) {
        let w = self.viewport.w.max(1) as f64;
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
        let (sw, sh) = (self.viewport.w.max(1) as f32, self.viewport.h.max(1) as f32);
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
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        match self.cache.get(self.index) {
            Some(t) => {
                t.h as f32 * fit_scale(self.fit, sw, sh, t.w as f32, t.h as f32) * self.zoom
            }
            None => sh,
        }
    }

    /// Flip-mode anchor metrics `(sw, sh, fit_w, fit_h, dec_h, src_h)` — the inputs
    /// `anchor_native_scale` needs, computed once and shared by `anchor_scale`
    /// (current fit/zoom) and `fit_native_pct` (an arbitrary fit at zoom 1). `None`
    /// in scroll mode (no facing-pair layout) or before the anchor is decoded.
    fn anchor_metrics(&self) -> Option<(f32, f32, f32, f32, f32, f32)> {
        if self.scroll_mode {
            return None;
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        let len = self.source.as_ref()?.len();
        if len == 0 {
            return None;
        }
        let (a, b) = layout::view_pages(self.layout, self.index, len, self.spread_offset);
        let ta = self.cache.get(a)?;
        // Wide (landscape) page is shown alone; otherwise pair with `b` if ready.
        let force_single = ta.w > ta.h;
        let tb = if force_single { None } else { b.and_then(|bi| self.cache.get(bi)) };
        let (fit_w, fit_h, dec_h) = match tb {
            Some(tb) => {
                let h_ref = ta.h.max(tb.h) as f32;
                let wa = ta.w as f32 * h_ref / ta.h.max(1) as f32;
                let wb = tb.w as f32 * h_ref / tb.h.max(1) as f32;
                (wa + wb, h_ref, h_ref)
            }
            None => (ta.w as f32, ta.h as f32, ta.h as f32),
        };
        Some((sw, sh, fit_w, fit_h, dec_h, ta.src_h.max(1) as f32))
    }

    /// device-px-per-native-px of the in-view anchor page, matching exactly what
    /// `build_quads` draws (single vs. facing-pair dims). `None` while the anchor
    /// isn't decoded yet.
    fn anchor_scale(&self) -> Option<f32> {
        if self.scroll_mode {
            // Strip pages are laid out at width = sw * zoom (height follows aspect).
            let sw = self.viewport.w.max(1) as f32;
            let t = self.cache.get(self.index)?;
            return Some(sw * self.zoom / t.src_w.max(1) as f32);
        }
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        Some(anchor_native_scale(
            self.fit,
            (sw, sh),
            (fit_w, fit_h),
            dec_h,
            src_h,
            self.zoom,
        ))
    }

    /// The native zoom % the current anchor would display at under `fit` at zoom 1
    /// — used to splice fit-to-window / fit-to-width stops into the zoom ladder.
    /// `None` in scroll mode (handled inline in `zoom_ladder`) or before decode.
    fn fit_native_pct(&self, fit: FitMode) -> Option<f32> {
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        Some(anchor_native_scale(fit, (sw, sh), (fit_w, fit_h), dec_h, src_h, 1.0) * 100.0)
    }

    /// Zoom relative to the *original* image resolution (1 image px : 1 screen px
    /// = 100%), for the toast + info overlay. Derived from the same scale the
    /// renderer draws, so it tracks fit-to-window upscaling and facing pairs
    /// exactly. Falls back to the raw factor while the anchor isn't decoded.
    fn effective_zoom_pct(&self) -> f32 {
        let scale = self.anchor_scale().unwrap_or(self.zoom);
        (scale * 100.0).max(0.0)
    }

    /// Device-px per *decoded texel* for the in-view anchor — i.e. exactly how the
    /// GPU sampler scales the texture at draw time. `1.0` = sampling 1:1 (the HQ CPU
    /// resize did all the work); `>1` = GPU upscale (zoom-past-native magnification,
    /// the one allowed GPU resample); `<1` = GPU downscale (the soft/moiré path —
    /// only ever valid as a transient while a re-decode is in flight). `None` before
    /// the anchor is decoded.
    fn gpu_sample_scale(&self) -> Option<f32> {
        if self.scroll_mode {
            let t = self.cache.get(self.index)?;
            let sw = self.viewport.w.max(1) as f32;
            return Some(sw * self.zoom / t.w.max(1) as f32); // strip drawn at width sw*zoom
        }
        // Equals single_quad's draw scale `s`: native scale × (src_h / decoded_h).
        let (sw, sh, fit_w, fit_h, dec_h, src_h) = self.anchor_metrics()?;
        let native = anchor_native_scale(self.fit, (sw, sh), (fit_w, fit_h), dec_h, src_h, self.zoom);
        Some(native * src_h / dec_h.max(1.0))
    }

    /// The in-view anchor's full resize pipeline for the info overlay:
    /// `"<CPU resize path>  →  <GPU sampling state>"`. Empty until decoded.
    /// `(CPU resize-path label, GPU sample scale, re-decode-pending)` for the
    /// in-view anchor — `None` until it's decoded. `pending` means the texture's
    /// decode target no longer matches the *current* desired target, so a re-decode
    /// is due: any GPU downscale right now is transient and will converge. So
    /// `!pending && scale < 1` is the only genuine single-resize-invariant violation
    /// (decoded at the intended target, yet the GPU still has to shrink it).
    fn anchor_resize_state(&self) -> Option<(&'static str, f32, bool)> {
        let src = self.source.as_ref()?;
        let len = src.len();
        if len == 0 {
            return None;
        }
        let anchor = if self.scroll_mode {
            self.index
        } else {
            layout::view_pages(self.layout, self.index, len, self.spread_offset).0
        };
        let t = self.cache.get(anchor)?;
        let s = self.gpu_sample_scale()?;
        let pending = t.target_h != self.page_target_h(anchor);
        Some((t.path.label(), s, pending))
    }

    /// The in-view anchor's full resize pipeline for the info overlay:
    /// `"<CPU resize path>  →  <GPU sampling state>"`. Empty until decoded.
    fn resize_path_label(&self) -> String {
        let Some((cpu, s, pending)) = self.anchor_resize_state() else {
            return String::new();
        };
        let gpu = if (s - 1.0).abs() <= 0.01 {
            "GPU 1:1".to_string()
        } else if s > 1.0 {
            format!("GPU \u{2191}{s:.2}\u{d7} (magnify)")
        } else if pending {
            format!("GPU \u{2193}{s:.2}\u{d7} (re-decoding\u{2026})")
        } else {
            format!("GPU \u{2193}{s:.2}\u{d7} (LQ \u{2014} STUCK)")
        };
        format!("{cpu}  \u{2192}  {gpu}")
    }

    /// Refresh the live resize readout (`ui.resize_path`) and fire a one-shot debug
    /// warning only on a *genuine* violation: the anchor is decoded at its intended
    /// target (no re-decode pending) yet the GPU is still downscaling it. Re-decode
    /// transients (a fresh page still at its prefetch-guessed size, a zoom/resize not
    /// yet settled) are expected and are not warned.
    fn update_resize_readout(&mut self) {
        let stuck = !self.scroll_mode
            && matches!(self.anchor_resize_state(), Some((_, s, pending)) if !pending && s < 0.99);
        if stuck && !self.gpu_downscale_warned {
            eprintln!(
                "yosh: WARNING — view at its decode target is still GPU-downscaling (single-resize invariant violated): {}",
                self.resize_path_label()
            );
            self.gpu_downscale_warned = true;
        } else if !stuck {
            self.gpu_downscale_warned = false;
        }
        self.ui.resize_path = self.resize_path_label();
    }

    /// The active zoom ladder: the fixed presets plus the current page's
    /// fit-to-window and fit-to-width stops (which depend on its resolution),
    /// in-range, sorted, and de-duplicated. In scroll mode the two fit stops are
    /// the page's width-fit (the zoom-1 strip) and height-fit native percents.
    fn zoom_ladder(&self) -> Vec<f32> {
        let mut ladder = zoom_presets();
        let (lo, hi) = (MIN_ZOOM_PCT * 100.0, MAX_ZOOM_PCT * 100.0);
        let mut stops: Vec<f32> = Vec::new();
        if self.scroll_mode {
            if let Some(t) = self.cache.get(self.index) {
                let sw = self.viewport.w.max(1) as f32;
                let sh = self.viewport.h.max(1) as f32;
                stops.push(sw / t.src_w.max(1) as f32 * 100.0); // fit width (strip @ zoom 1)
                stops.push(sh / t.src_h.max(1) as f32 * 100.0); // fit window (height fills)
            }
        } else {
            for f in [FitMode::Window, FitMode::Width] {
                if let Some(p) = self.fit_native_pct(f) {
                    stops.push(p);
                }
            }
        }
        for p in stops {
            // Splice a fit stop only if it isn't essentially on a value already in
            // the ladder. Otherwise a fit level a hair off a round preset (e.g. a
            // fit-window of 69.99% next to the 70% preset) shadows the preset, and
            // zoom snaps to 69.99% / 70.01% instead of a clean 70%. Keeping the
            // round preset still gets the "(Fit window/width)" toast label via the
            // `near()` check in `zoom_to_preset`.
            if (lo..=hi).contains(&p) && !ladder.iter().any(|&q| (q - p).abs() <= q * 1e-3) {
                ladder.push(p);
            }
        }
        ladder.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ladder
    }

    /// Snap zoom to the next ladder stop above/below the current native %. The
    /// ladder mixes the fixed BandiView presets with this page's fit stops, so a
    /// step can land exactly on fit-to-window / fit-to-width. Works in both
    /// page-flip and scroll modes (both derive from `effective_zoom_pct`).
    fn zoom_to_preset(&mut self, zoom_in: bool) {
        let cur = self.effective_zoom_pct();
        let mut label: Option<&'static str> = None;
        if self.anchor_scale().is_some() && cur > 0.0 {
            let ladder = self.zoom_ladder();
            let target = next_zoom_preset(&ladder, cur, zoom_in);
            // Tag the stop if it is this page's fit-to-window / fit-to-width level.
            if !self.scroll_mode {
                let near = |p: Option<f32>| p.is_some_and(|p| (p - target).abs() <= target * 1e-3);
                if near(self.fit_native_pct(FitMode::Window)) {
                    label = Some("Fit window");
                } else if near(self.fit_native_pct(FitMode::Width)) {
                    label = Some("Fit width");
                }
            }
            self.zoom *= target / cur; // rescale the fit-multiplier to hit target %
        } else {
            // Anchor not decoded yet: coarse step; the next press snaps once it lands.
            self.zoom *= if zoom_in { 1.25 } else { 1.0 / 1.25 };
        }
        self.clamp_zoom_native();
        self.clamp_pan();
        let pct = self.effective_zoom_pct();
        match label {
            // Fit label on its own line so the "Zoom %" line stays centered
            // (the toast is center-aligned), aligned across zoom levels.
            Some(l) => self.toast(format!("Zoom {pct:.2}%\n({l})")),
            None => self.toast(format!("Zoom {pct:.2}%")),
        }
    }

    fn clamp_zoom_native(&mut self) {
        if self.zoom > 0.0
            && let Some(s) = self.anchor_scale()
        {
            let base = s / self.zoom;
            if base > 0.0 {
                self.zoom = clamp_zoom_multiplier(self.zoom, base);
            }
        }
    }

    /// Clamp stored pan to the current page's overflow so dragging/zooming can't
    /// strand the view in an empty region.
    fn clamp_pan(&mut self) {
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        if self.scroll_mode {
            let cw = sw * self.zoom;
            let mx = ((cw - sw) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            return;
        }
        if let Some(t) = self.cache.get(self.index) {
            // Match single_quad's rotated bounding box so pan clamps to the
            // displayed (possibly turned) page, not the source orientation.
            let single = self.layout == Layout::Single || t.w > t.h;
            let (ew, eh) = if single && self.rotation % 2 == 1 {
                (t.h as f32, t.w as f32)
            } else {
                (t.w as f32, t.h as f32)
            };
            let s = fit_scale(self.fit, sw, sh, ew, eh) * self.zoom;
            let mx = ((ew * s - sw) / 2.0).max(0.0);
            let my = ((eh * s - sh) / 2.0).max(0.0);
            self.pan_x = self.pan_x.clamp(-mx, mx);
            self.pan_y = self.pan_y.clamp(-my, my);
        }
    }

    fn single_quad(&self, idx: usize, t: &PageTexture, sw: f32, sh: f32) -> Quad {
        // A 90°/270° turn swaps the page's effective width/height for fitting; the
        // shader then turns the texture inside this (rotated) bounding box. The box
        // dimensions stay whole texels at the fit scale, so 1:1 sampling holds (the
        // decode target in `page_target_h` is rotation-aware to match).
        let (dw, dh) = if self.fit == FitMode::Actual {
            // 1:1: size from the *source* dims × zoom, not the decoded dims, so the
            // displayed box is the same native size whether the texture is full res
            // (zoom ≥ 1) or re-decoded smaller for zoom-out — the latter then samples
            // 1:1 instead of the GPU bilinear-downscaling a full-res texture.
            let (nw, nh) = if self.rotation % 2 == 1 {
                (t.src_h as f32, t.src_w as f32)
            } else {
                (t.src_w as f32, t.src_h as f32)
            };
            (nw * self.zoom, nh * self.zoom)
        } else {
            let (ew, eh) = if self.rotation % 2 == 1 {
                (t.h as f32, t.w as f32)
            } else {
                (t.w as f32, t.h as f32)
            };
            let s = fit_scale(self.fit, sw, sh, ew, eh) * self.zoom;
            (ew * s, eh * s)
        };
        // Snap the page to the device-pixel grid. At 1:1 (fit-to-window) a
        // fractional offset would make the bilinear sampler blend every column
        // 50/50 with its neighbour — a horizontal smear that also beats against
        // halftone screentones. Whole-pixel placement samples texel centers 1:1.
        quad_from_px(
            0,
            idx,
            self.horizontal_left(dw, sw).round(),
            self.vertical_top(dh, sh).round(),
            dw.round(),
            dh.round(),
            sw,
            sh,
            self.rotation as u32,
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
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;

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
                // Snap to the pixel grid (see single_quad). The right page starts
                // at the left's snapped right edge, so there's no sub-pixel seam.
                let yt = self.vertical_top(dh, sh).round();
                let dhr = dh.round();
                let xl = x0.round();
                let dwl_r = dwl.round();
                vec![
                    quad_from_px(0, l_idx, xl, yt, dwl_r, dhr, sw, sh, 0),
                    quad_from_px(1, r_idx, xl + dwl_r, yt, dwr.round(), dhr, sw, sh, 0),
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
        let len = match &self.source {
            Some(s) => s.len(),
            None => return,
        };
        let before = self.index;
        let before_off = self.top_offset;
        self.top_offset += dy;
        self.normalize();
        if self.index != before {
            self.nav_times.push_back(Instant::now());
        } else if dy.abs() > 0.5 && (self.top_offset - before_off).abs() < 0.5 {
            // The strip didn't move despite a scroll — clamped at an end.
            if dy < 0.0 && self.index == 0 && self.top_offset <= 0.5 {
                self.toast("First page");
            } else if dy > 0.0 && self.index + 1 >= len {
                self.toast("Last page");
            }
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
        let sw = self.viewport.w.max(1) as f32;
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
            let vh = self.viewport.h as f32;
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
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
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
                    quads.push(quad_from_px(slot, i, x, y, cw, dh, sw, sh, 0));
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
                Ok(img) => yosh_engine::decode::to_rgba_image(img), // egui samples RGBA
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

    /// Begin opening `path`. The source is built on a background thread (see
    /// `build_source`) so a slow network-share open never freezes the UI — the
    /// current page stays on screen under the spinner until the new source lands
    /// in `render`. Each call bumps `open_gen`; only the newest result is applied,
    /// so rapid `[`/`]` supersede in-flight opens instead of queuing stale swaps.
    fn open(&mut self, path: &Path) {
        self.open_gen = self.open_gen.wrapping_add(1);
        let generation = self.open_gen;
        let tx = self.open_tx.clone();
        let path = path.to_path_buf();
        self.opening = true;
        self.opening_key = Some(path.clone());
        std::thread::spawn(move || {
            let _ = tx.send((generation, build_source(&path)));
        });
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
            WORKERS,
        ));
        self.cache.clear();
        self.failed.clear();
        self.last_drawn = None;
        self.info_for = None;
        self.nav_times.clear();
        self.rotation = 0; // each volume opens upright
        self.index = idx.min(source.len() - 1);
        self.volume_key = Some(key);
        self.ui.opened = Some(path.to_path_buf());
        self.source = Some(source);
        self.library_view = false; // opening anything switches to the reader
        self.prefetch();
        // Warm the sibling-volume list for this folder in the background, so the
        // first `[`/`]` press doesn't pay the parent-dir scan on the main thread.
        self.warm_sib_cache(path);
    }

    /// Scan the current volume's folder for sibling volumes on a background
    /// thread and hand the result back for `sib_cache` (consumed in `render`).
    /// Keeps the parent-dir `read_dir` + per-entry stat off the UI thread so
    /// `[`/`]` stays responsive even on a network share.
    fn warm_sib_cache(&self, vol: &Path) {
        let tx = self.sib_tx.clone();
        let of = vol.to_path_buf();
        std::thread::spawn(move || {
            let Some(parent) = of.parent().map(|p| p.to_path_buf()) else {
                return;
            };
            let want_folder = of.is_dir();
            let _ = tx.send((parent, want_folder, crate::library::sibling_volumes(&of)));
        });
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
        // Tell the user what the preset just switched to (presets change fit +
        // layout + maybe direction at once, so summarize the resulting view).
        let view_label = if spread {
            format!("Spread, {}", self.direction.label())
        } else {
            let f = match fit {
                FitMode::Window => "Fit window",
                FitMode::Width => "Fit width",
                FitMode::Height => "Fit height",
                FitMode::Actual => "1:1",
            };
            format!("{f} (single)")
        };
        self.toast(view_label);
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
                let color = yosh_engine::icc::extract_icc(b)
                    .as_deref()
                    .and_then(|p| yosh_engine::icc::describe(p))
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

    /// Source aspect (w / h) for page `index`: from its decoded texture if present,
    /// else the in-view anchor's, else the running estimate. Used to size the decode
    /// target before the page itself is decoded (exact for the usual uniform-size
    /// volume; corrected in place once the page's own dimensions are known).
    fn page_aspect(&self, index: usize) -> f32 {
        if let Some(t) = self.cache.get(index) {
            return t.src_w as f32 / t.src_h.max(1) as f32;
        }
        if let Some(t) = self.cache.get(self.index) {
            return t.src_w as f32 / t.src_h.max(1) as f32;
        }
        1.0 / self.est_aspect.max(0.01) // est_aspect is h / w
    }

    /// The *exact* decode target (on-screen displayed pixel height) for page
    /// `index` under the active fit/zoom/layout. Decoding each page to this height
    /// makes the HQ CPU resize the only resample and the GPU sample 1:1 — the
    /// single-resize invariant. `target_dims` later caps it at the source height
    /// (so a display larger than native means full-res + GPU upscale, the one
    /// allowed exception). 1:1 keeps full source res (it draws at `zoom` directly).
    fn page_target_h(&self, index: usize) -> u32 {
        let aspect = self.page_aspect(index).max(0.001);
        // Cap the decode target so neither the texture height nor its aspect-derived
        // width exceeds the GPU's real max texture size. This replaces a former fixed
        // 3840 cap, which forced the GPU to *upscale* (and thus moiré) any page taller
        // than 3840 px viewed near native — the texture couldn't be decoded to the
        // shown size, so the GPU resampled it. Now the HQ CPU resize hits the display
        // size and the GPU stays 1:1 below native, all the way up to what the GPU can
        // hold. (`target_dims` still caps at the source height, so it never upscales.)
        let max_dim = yosh_engine::decode::MAX_TEX_DIM.load(std::sync::atomic::Ordering::Relaxed);
        let max_h = ((max_dim as f32 / aspect.max(1.0)).floor() as u32).max(MIN_TARGET);
        if self.fit == FitMode::Actual && !self.scroll_mode {
            // 1:1 displays at native × zoom. Target that height so the page decodes
            // to its *shown* size: `target_dims` caps at the source height, so
            // zoom ≥ 1 keeps full res (magnification GPU-upscales, the one allowed
            // GPU resample) while zoom < 1 decodes smaller → the HQ CPU resize does
            // the reduction and the GPU samples 1:1 (no bilinear-downscale moiré).
            // (Rotation-independent: a 90° turn swaps which screen edge the texture
            // height maps to, but the target works out to src_h × zoom either way.)
            return match self.cache.get(index) {
                Some(t) => {
                    ((t.src_h as f32 * self.zoom).round() as u32).clamp(MIN_TARGET, max_h)
                }
                None => u32::MAX, // native size unknown yet: decode full, re-decode once cached
            };
        }
        let sw = self.viewport.w.max(1) as f32;
        let sh = self.viewport.h.max(1) as f32;
        let target = if self.scroll_mode {
            // Continuous strip: width-fit at width sw*zoom, height follows aspect.
            sw * self.zoom / aspect
        } else {
            // A page is drawn alone when layout is Single or it's a wide
            // (landscape) page that force-shows alone — only then does rotation
            // apply. `content_aspect` is the on-screen box's width/height.
            let single = self.layout == Layout::Single || aspect > 1.0;
            let rotated = single && self.rotation % 2 == 1;
            let content_aspect = if rotated {
                1.0 / aspect // rotated single page: box is the inverse of the source
            } else if self.layout == Layout::Spread && aspect <= 1.0 {
                // Pair two non-wide pages (assume a same-size facing page — exact
                // for uniform volumes; wide pages always show alone).
                aspect * 2.0
            } else {
                aspect
            };
            let box_h = fit_scale(self.fit, sw, sh, content_aspect, 1.0) * self.zoom;
            // Decode target = the texture height that draws 1:1. For a rotated
            // single page the texture's height lands along the screen *width*, so
            // the target is the box width (box_h * content_aspect); else box height.
            if rotated { box_h * content_aspect } else { box_h }
        };
        (target.round() as u32).clamp(MIN_TARGET, max_h)
    }

    /// Debounce the decode view. While the surface size or zoom is changing (a
    /// resize/zoom drag) the view is "unsettled" and `prefetch` won't re-decode
    /// cached pages for a target change — it just keeps showing the old textures.
    /// Once the value holds for a frame the view settles and stale pages re-decode
    /// in place (no black frame). Page-flipping leaves the view settled, so it
    /// never re-decodes.
    fn update_decode_view(&mut self) {
        let desired = (self.viewport.w, self.viewport.h, self.zoom);
        self.view_settled = desired == self.pending_view;
        self.pending_view = desired;
    }

    /// Recompute the prefetch window and hand it to the pool with each page's exact
    /// decode target. A page is queued if it's missing, or (once the view has
    /// settled) if its decoded target no longer matches its current exact target —
    /// then it re-decodes at the new resolution and overwrites in place.
    fn prefetch(&mut self) {
        let fwd = self.dynamic_fwd();
        let settled = self.view_settled;
        let Some(src) = &self.source else {
            return;
        };
        let len = src.len();
        let jobs: Vec<(usize, u32)> = desired_window(self.index, len, fwd, BACK)
            .into_iter()
            .filter(|i| !self.failed.contains_key(i))
            .filter_map(|i| {
                let want = self.page_target_h(i);
                match self.cache.get(i) {
                    None => Some((i, want)),
                    Some(p) => (settled && p.target_h != want).then_some((i, want)),
                }
            })
            .collect();
        if let Some(pool) = &self.pool {
            pool.set_jobs(jobs);
        }
    }

    /// The page index whose animation controls are active (the in-view anchor, if
    /// it's an animated page and its texture is decoded).
    fn anim_anchor(&self) -> Option<usize> {
        if self.library_view {
            return None;
        }
        let len = self.source.as_ref()?.len();
        let anchor = if self.scroll_mode {
            self.index
        } else {
            layout::view_pages(self.layout, self.index, len, self.spread_offset).0
        };
        (self.cache.get(anchor)?.frame_count() > 1).then_some(anchor)
    }

    /// Advance/refresh playback for the in-view animation and publish the panel's
    /// display state to `self.ui`. Driven every frame by the continuous render
    /// loop.
    fn update_playback(&mut self) {
        let Some(anchor) = self.anim_anchor() else {
            self.playback.page = None;
            self.ui.anim_show = false;
            return;
        };
        let frames = self.cache.get(anchor).map_or(1, |t| t.frame_count());
        // GIF/WebP auto-play; `.ico` layers are stepped manually (no play/pause).
        let is_anim = self.cache.get(anchor).is_some_and(|t| t.is_animation());
        // Rebind (and reset) when the viewed page changes.
        if self.playback.page != Some(anchor) {
            self.playback.page = Some(anchor);
            self.playback.frame = 0;
            self.playback.playing = is_anim;
            self.playback.last = Instant::now();
        }
        if self.playback.playing {
            let now = Instant::now();
            // Advance through as many frames as real time has passed (respecting
            // each frame's own delay), so playback tracks wall time even if we
            // briefly lagged.
            loop {
                let d = self
                    .cache
                    .get(anchor)
                    .map_or(100, |t| t.frame_delay_ms(self.playback.frame))
                    .max(1) as u64;
                if now.duration_since(self.playback.last) >= Duration::from_millis(d) {
                    self.playback.last += Duration::from_millis(d);
                    self.playback.frame = (self.playback.frame + 1) % frames;
                } else {
                    break;
                }
            }
        }
        self.playback.frame = self.playback.frame.min(frames - 1);
        self.ui.anim_show = !self.playback.hidden;
        self.ui.anim_is_animation = is_anim;
        self.ui.anim_playing = self.playback.playing;
        self.ui.anim_frame = self.playback.frame;
        self.ui.anim_total = frames;
    }

    fn playback_toggle(&mut self) {
        self.playback.playing = !self.playback.playing;
        if self.playback.playing {
            self.playback.last = Instant::now(); // resume from now, no time-warp jump
        }
    }

    fn playback_frame_count(&self) -> usize {
        self.playback.page.and_then(|p| self.cache.get(p)).map_or(1, |t| t.frame_count())
    }

    /// Step the animation by `d` frames (pauses playback; wraps around).
    fn playback_step(&mut self, d: i32) {
        let frames = self.playback_frame_count();
        if frames <= 1 {
            return;
        }
        self.playback.playing = false;
        self.playback.frame = (self.playback.frame as i32 + d).rem_euclid(frames as i32) as usize;
    }

    /// Jump the animation to a specific frame (pauses playback).
    fn playback_seek(&mut self, frame: usize) {
        let frames = self.playback_frame_count();
        if frames <= 1 {
            return;
        }
        self.playback.playing = false;
        self.playback.frame = frame.min(frames - 1);
    }

    /// The dynamic window title: `{book} > {file} (W × H) [ pg / total ] - yosh`
    /// in the reader (resolution Firefox-tab style, shown once the page decodes),
    /// `Library - yosh` in the grid, plain `yosh` when nothing is open.
    fn title(&self) -> String {
        if self.library_view {
            return "Library - yosh".to_string();
        }
        let Some(src) = &self.source else {
            return "yosh".to_string();
        };
        let len = src.len();
        if len == 0 {
            return "yosh".to_string();
        }
        // The page actually shown (anchor): `index` in single/scroll, the first
        // page of the pair in a two-page spread.
        let anchor = if self.scroll_mode {
            self.index
        } else {
            layout::view_pages(self.layout, self.index, len, self.spread_offset).0
        }
        .min(len - 1);
        let name = src.name(anchor);
        // Just the basename — archive entries can carry a subfolder path.
        let file = std::path::Path::new(name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(name);
        // Native resolution of the shown page, Firefox-tab style. Pulled from the
        // decoded texture (`src_w`/`src_h` are pre-downscale source dims), so it
        // appears once the page lands and is empty while it's still decoding.
        let res = match self.cache.get(anchor) {
            Some(t) if t.src_w > 0 && t.src_h > 0 => format!(" ({} × {})", t.src_w, t.src_h),
            _ => String::new(),
        };
        let pos = format!("[ {} / {} ] - yosh", anchor + 1, len);
        match self.ui.opened.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()) {
            Some(book) => format!("{book} > {file}{res} {pos}"),
            None => format!("{file}{res} {pos}"),
        }
    }

    #[allow(deprecated)]
    fn render(&mut self) {
        // Mirror the live surface size into the reading viewport (the value the
        // reading math reads instead of `gpu.config`). Equal to `gpu.config` by
        // construction, so this is a no-op for behavior.
        self.viewport = Viewport {
            w: self.gpu.config.width,
            h: self.gpu.config.height,
        };
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
                    Msg::Failed { index, error } => {
                        self.failed.insert(index, error);
                    }
                }
            }
        }
        // Apply finished background opens, newest generation only — a later
        // `[`/`]` bumps `open_gen`, so a stale in-flight result is discarded.
        while let Ok((generation, built)) = self.open_rx.try_recv() {
            if generation != self.open_gen {
                continue; // superseded by a newer open()
            }
            self.opening = false;
            self.opening_key = None;
            match built {
                Ok((source, key, start)) if source.len() > 0 => {
                    self.set_source(source, &key, start)
                }
                Ok(_) => self.ui.status = "no images found".into(),
                Err(e) => self.ui.status = format!("open failed: {e}"),
            }
        }
        // Pick up background sibling-volume scans into the `[`/`]` cache.
        while let Ok(entry) = self.sib_rx.try_recv() {
            self.sib_cache = Some(entry);
        }
        // Auto-update: pick up the background check result and any apply result.
        if let Some(rx) = self.update_rx.take() {
            match rx.try_recv() {
                Ok(u) => self.update = Some(u),
                Err(std::sync::mpsc::TryRecvError::Empty) => self.update_rx = Some(rx),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
        if let Some(rx) = self.update_apply_rx.take() {
            match rx.try_recv() {
                Ok(Ok(())) => self.relaunch(),
                Ok(Err(e)) => {
                    self.updating = false;
                    self.update_error = Some(e);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => self.update_apply_rx = Some(rx),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.updating = false;
                    self.update_error = Some("update interrupted".into());
                }
            }
        }
        self.ui.update_version = self.update.as_ref().map(|u| u.version.clone());
        self.ui.updating = self.updating;
        self.ui.update_failed = self.update_error.is_some();
        // Debounce the decode view, so resize/zoom drags re-decode once on settle.
        self.update_decode_view();
        // Keep the scroll anchor valid as page heights resolve, then refresh work.
        if self.scroll_mode {
            self.normalize();
        }
        self.prefetch();
        // Advance the in-view animation's frame and refresh its control panel.
        self.update_playback();
        // Keep the OS titlebar in sync with the open book + page (change-only).
        let title = self.title();
        if title != self.last_title {
            self.window.set_title(&title);
            self.last_title = title;
        }

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
        // Build the Tab info overlay text, reading the source once per page change.
        if self.ui.info_open && !self.library_view && self.info_for != Some(self.index) {
            self.ui.info = self.build_page_info(self.index);
            self.info_for = Some(self.index);
        }
        // Live view state for the overlays: current zoom % (shown in the info
        // overlay, refreshed every frame so it tracks zooming without a rebuild)
        // and the active toast (dropped once it expires).
        self.ui.zoom_pct = self.effective_zoom_pct();
        self.update_resize_readout();
        if let Some((_, t)) = &self.toast
            && t.elapsed() >= TOAST_DURATION
        {
            self.toast = None;
        }
        self.ui.toast = self.toast.as_ref().map(|(m, _)| m.clone());
        // Hide the top bar in fullscreen, revealing it when the cursor is at the top edge.
        let fullscreen = self.window.fullscreen().is_some();
        let reveal = 48.0 * self.window.scale_factor() as f32;
        self.ui.show_bar = !fullscreen || (self.cursor_y as f32) < reveal;
        // Edge hover arrows: only in page-flip reader mode, below the top bar,
        // while the cursor is inside the window.
        let win_w = self.viewport.w.max(1) as f32;
        let edge = win_w * EDGE_FRAC;
        let in_reader = self.source.is_some() && !self.library_view && !self.scroll_mode;
        let below_bar = (self.cursor_y as f32) >= reveal;
        let cx = self.cursor_x as f32;
        self.ui.hover_left = in_reader && self.cursor_in_window && below_bar && cx < edge;
        self.ui.hover_right =
            in_reader && self.cursor_in_window && below_bar && cx > win_w - edge;
        // Bottom seekbar: hidden by default, revealed when the cursor nears the
        // bottom edge (a touch taller than the floating pill so a drag stays in
        // the reveal zone). Cleared here so it vanishes when no volume is open.
        self.ui.seek_show = false;
        self.ui.seek_hovered = false;
        if let Some(src) = &self.source {
            let len = src.len();
            let anchor = if self.scroll_mode {
                self.index
            } else {
                layout::view_pages(self.layout, self.index, len, self.spread_offset).0
            };
            let in_cache = self.cache.contains(anchor);
            // A page whose decode errored is in `failed`; treat it as not-loading so
            // we show a failure notice (file name + reason) instead of spinning.
            let fail_err: Option<String> =
                if in_cache { None } else { self.failed.get(&anchor).cloned() };
            let failed = fail_err.is_some();
            let loading = !in_cache && !failed;
            if in_cache {
                self.last_drawn = Some(anchor);
            }
            self.ui.status = format!(
                "{}/{}{}{}",
                self.index + 1,
                len,
                if failed {
                    "  [failed]"
                } else if loading {
                    "  …"
                } else {
                    ""
                },
                if self.jump { "  [jump]" } else { "  [step]" }
            );
            self.ui.failed = fail_err.map(|reason| (src.name(anchor).to_string(), reason));
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
            self.ui.seek_index = self.index;
            self.ui.seek_total = len;
            self.ui.seek_rtl = self.direction == Direction::Rtl;
            self.ui.seek_style = ui::SeekbarStyle::Bar;
            let win_h = self.viewport.h.max(1) as f32;
            let near_bottom =
                self.cursor_in_window && (self.cursor_y as f32) > win_h - reveal * 1.5;
            self.ui.seek_show =
                self.settings.seekbar_enabled && !self.library_view && len > 1 && near_bottom;
        }
        // A background open is in flight: show the spinner over the current page
        // (or the dark fill on a first open) until the new source lands.
        if self.opening {
            self.ui.loading = true;
        }
        let anim_t = self.anim_origin.elapsed();
        let anim_page = self.playback.page;
        let anim_frame = self.playback.frame;
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.cache.get(q.page_index).map(|t| {
                    // The animation under user control shows its selected frame;
                    // any other animated page free-runs on the wall clock; stills
                    // return their sole view. Continuous redraw drives both.
                    let view = if Some(q.page_index) == anim_page {
                        t.frame_view(anim_frame)
                    } else {
                        t.view_at(anim_t)
                    };
                    self.page_pipeline.prepare_quad(
                        &self.gpu.device,
                        &self.gpu.queue,
                        q.slot,
                        t,
                        view,
                        q.scale,
                        q.offset,
                        q.rot,
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
                        // #202020. The surface is non-sRGB, so the stored byte is
                        // value*255 (0x20 = 32). Transparent pages composite over
                        // this via the page pipeline's premultiplied-alpha blend.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 32.0 / 255.0,
                            g: 32.0 / 255.0,
                            b: 32.0 / 255.0,
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
        // Seekbar jump: re-clamp against the live source, skip a redundant goto
        // (which would needlessly reset pan when landing on the current page).
        if let Some(page) = self.ui.seek_request.take()
            && let Some(src) = &self.source
        {
            let page = page.min(src.len().saturating_sub(1));
            if page != self.index {
                self.goto(page);
            }
        }
        // Animation control-panel clicks (drained after the egui frame).
        if std::mem::take(&mut self.ui.anim_req_toggle_play) {
            self.playback_toggle();
        }
        let step = std::mem::take(&mut self.ui.anim_req_step);
        if step != 0 {
            self.playback_step(step);
        }
        if let Some(f) = self.ui.anim_req_seek.take() {
            self.playback_seek(f);
        }
        if std::mem::take(&mut self.ui.anim_req_hide) {
            self.playback.hidden = true;
        }
        if std::mem::take(&mut self.ui.req_update) && !self.updating {
            if let Some(u) = self.update.clone() {
                self.updating = true;
                self.update_error = None;
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(update::apply(&u));
                });
                self.update_apply_rx = Some(rx);
            }
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

#[cfg(test)]
mod tests {
    #[test]
    fn probe_reads_image_crate_dimensions() {
        use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
        // TIFF/QOI have no magic-byte branch in probe(); the generic fallback must
        // still report their resolution (the "probe data for resolution" rule).
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(7, 4, Rgba([1, 2, 3, 255])));
        for fmt in [ImageFormat::Tiff, ImageFormat::Qoi, ImageFormat::Tga] {
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, fmt).unwrap();
            let (w, h, label) = super::probe(&buf.into_inner());
            assert_eq!((w, h), (7, 4), "probe dims for {fmt:?}");
            assert!(!label.is_empty() && label != "image", "probe label for {fmt:?}: {label}");
        }
    }

    // A 2048-tall portrait page on a 4K (2160-tall) screen, fit-to-window and
    // height-constrained, is displayed at 2160 → ~105% of native, not 100%.
    #[test]
    fn anchor_native_scale_fit_to_window_reports_upscale() {
        let s = super::anchor_native_scale(
            super::FitMode::Window,
            (3840.0, 2160.0),
            (1448.0, 2048.0),
            2048.0,
            2048.0,
            1.0,
        );
        assert!((s - 2160.0 / 2048.0).abs() < 1e-4, "got {s}");
    }

    // 1:1 (Actual) at zoom 1 is exactly native: 100%.
    #[test]
    fn anchor_native_scale_actual_is_unity() {
        let s = super::anchor_native_scale(
            super::FitMode::Actual,
            (3840.0, 2160.0),
            (1448.0, 2048.0),
            2048.0,
            2048.0,
            1.0,
        );
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    // Proof of the single-resize invariant (page-flip path): a page decoded to
    // its per-page target (the displayed height `page_target_h` computes) is drawn
    // by `build_quads` at that *same* height, so the GPU sampler maps 1 texel : 1
    // pixel and adds no second resize. Checks the decode target and the draw size
    // agree across fit modes, aspects, zooms, and surface sizes.
    #[test]
    fn decode_target_matches_drawn_size() {
        use yosh_engine::page::{fit_scale, FitMode};
        for (sw, sh) in [(3840.0_f32, 2160.0_f32), (1920.0, 1080.0), (1600.0, 2560.0)] {
            for fit in [FitMode::Window, FitMode::Width, FitMode::Height] {
                for aspect in [0.5_f32, 0.69, 1.0, 1.5] {
                    for zoom in [0.1_f32, 0.5, 1.0] {
                        // Decode target = the page's displayed height (page_target_h).
                        let th = (fit_scale(fit, sw, sh, aspect, 1.0) * zoom).round().max(1.0);
                        let tw = (th * aspect).round().max(1.0);
                        // build_quads draws that decoded (tw x th) texture at height:
                        let drawn = th * fit_scale(fit, sw, sh, tw, th) * zoom;
                        assert!(
                            (drawn - th).abs() <= 2.0,
                            "fit {} a {aspect} z {zoom} {sw}x{sh}: drawn {drawn} vs texture {th}",
                            fit.label(),
                        );
                    }
                }
            }
        }
    }

    // Single-resize invariant under a 90°/270° turn: `page_target_h` swaps the
    // aspect and returns the on-screen box *width* as the decode target (texture
    // height), then `single_quad` turns the texture inside the rotated box. The
    // turned texture's height must still map 1 texel : 1 pixel along the screen
    // width — i.e. the rotated-draw fit scale stays ~1, so no second GPU resize.
    #[test]
    fn decode_target_matches_drawn_size_rotated() {
        use yosh_engine::page::{fit_scale, FitMode};
        for (sw, sh) in [(3840.0_f32, 2160.0_f32), (1920.0, 1080.0), (1600.0, 2560.0)] {
            for fit in [FitMode::Window, FitMode::Width, FitMode::Height] {
                for aspect in [0.5_f32, 0.69, 1.0, 1.5] {
                    for zoom in [0.1_f32, 0.5, 1.0] {
                        // page_target_h (rotated): content_aspect = 1/aspect, the
                        // target is the box width = box_h * content_aspect.
                        let box_h = fit_scale(fit, sw, sh, 1.0 / aspect, 1.0) * zoom;
                        let th = (box_h / aspect).round().max(1.0); // texture height (target)
                        let tw = (th * aspect).round().max(1.0); // texture width (source aspect)
                        // single_quad swaps (w,h) for the odd rotation: ew = th, eh = tw.
                        let s = fit_scale(fit, sw, sh, th, tw) * zoom;
                        let drawn_w = th * s; // screen width the turned texture's height fills
                        assert!(
                            (drawn_w - th).abs() <= 2.0,
                            "rot fit {} a {aspect} z {zoom} {sw}x{sh}: drawn_w {drawn_w} vs texture-h {th}",
                            fit.label(),
                        );
                    }
                }
            }
        }
    }

    // Single-resize invariant in 1:1 (Actual) fit when zoomed *out* (the fixed
    // path): the page targets its displayed native×zoom height, decodes to that,
    // and is drawn at the same size — so the GPU samples 1:1 and never bilinear-
    // downscales a full-res texture. (Surface size is irrelevant in 1:1.)
    #[test]
    fn decode_target_matches_drawn_size_actual_zoomed_out() {
        for (src_w, src_h) in [(1500.0_f32, 5200.0), (5200.0, 1500.0), (2048.0, 2048.0)] {
            let _ = src_w; // 1:1 sizes off src_h × zoom; width follows the same scale
            for zoom in [0.1_f32, 0.27, 0.5, 0.99] {
                // page_target_h (Actual): displayed height = src_h × zoom; target_dims
                // caps at the source height (no cap here since zoom < 1).
                let target = (src_h * zoom).round().max(1.0);
                let th = target.min(src_h); // decoded texture height
                let drawn = src_h * zoom; // single_quad draws the box at src_h × zoom
                let gpu_scale = drawn / th; // displayed ÷ decoded — must be ~1 (no resize)
                assert!(
                    (gpu_scale - 1.0).abs() <= 0.01,
                    "actual {src_w}x{src_h} z {zoom}: gpu_scale {gpu_scale} (drawn {drawn}, texture {th})",
                );
            }
        }
    }

    // A source taller than the *former* fixed 3840 cap, viewed below native, must
    // decode to its displayed height — capped only by the GPU's real max texture
    // size, aspect-aware so the width fits too — so the GPU samples 1:1. The old
    // 3840 cap forced a GPU upscale (e.g. a 5207px page at 80–90% → ↑1.08–1.22×),
    // which beats against the screentone → moiré. This models `page_target_h`'s cap.
    #[test]
    fn large_page_decodes_to_display_not_a_fixed_cap() {
        let max_dim = 8192u32; // default MAX_TEX_DIM (the GPU's real limit)
        for (src_w, src_h) in [(3600.0_f32, 5207.0), (5207.0, 3600.0), (4000.0, 6000.0)] {
            let aspect = src_w / src_h;
            let max_h = ((max_dim as f32 / aspect.max(1.0)).floor() as u32).max(super::MIN_TARGET);
            for zoom in [0.5_f32, 0.74, 0.8, 0.9, 1.0] {
                // page_target_h (Actual): displayed height = src_h × zoom, clamped to max_h.
                let target = ((src_h * zoom).round() as u32).clamp(super::MIN_TARGET, max_h);
                let th = (target as f32).min(src_h); // target_dims caps at source
                let tw = (th * aspect).round() as u32; // width follows source aspect
                let display = src_h * zoom; // single_quad (Actual) draws at src_h × zoom
                let gpu_scale = display / th;
                assert!(
                    (gpu_scale - 1.0).abs() <= 0.01,
                    "src {src_w}x{src_h} z {zoom}: gpu_scale {gpu_scale} (display {display}, tex {th})"
                );
                assert!(
                    th as u32 <= max_dim && tw <= max_dim,
                    "src {src_w}x{src_h} z {zoom}: texture {tw}x{th} exceeds GPU max {max_dim}"
                );
            }
        }
    }

    // The fit-multiplier clamp maps to native bounds: far zoom-out hits the 5%
    // floor, far zoom-in hits the 20000% ceiling, mid values pass through.
    #[test]
    fn zoom_multiplier_clamps_to_native_bounds() {
        let base = 2160.0_f32 / 2048.0; // native scale at zoom = 1 (fit-to-window)
        let lo = super::clamp_zoom_multiplier(1e-6, base);
        assert!((lo * base - super::MIN_ZOOM_PCT).abs() < 1e-4, "lo eff {}", lo * base);
        let hi = super::clamp_zoom_multiplier(1e9, base);
        assert!((hi * base - super::MAX_ZOOM_PCT).abs() < 1e-2, "hi eff {}", hi * base);
        let mid = super::clamp_zoom_multiplier(1.0, base);
        assert!((mid - 1.0).abs() < 1e-6, "mid {mid}");
    }

    // The BandiView ladder: 5, 10..300 by 10, 320..500 by 20, 550..20000 by 50.
    #[test]
    fn zoom_ladder_shape() {
        let p = super::zoom_presets();
        assert_eq!(p.first().copied(), Some(5.0));
        assert_eq!(p.last().copied(), Some(20000.0));
        assert!(p.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
        for v in [10.0, 100.0, 300.0, 320.0, 500.0, 550.0, 20000.0] {
            assert!(p.contains(&v), "ladder missing {v}");
        }
        let idx = |v: f32| p.iter().position(|&x| x == v).unwrap();
        assert_eq!(p[idx(300.0) + 1], 320.0, "300 -> 320 (step 20)");
        assert_eq!(p[idx(500.0) + 1], 550.0, "500 -> 550 (step 50)");
    }

    // +/- step to the neighbouring fixed stop, clamping at the ends.
    #[test]
    fn zoom_stepping_fixed() {
        let p = super::zoom_presets();
        let up = |c: f32| super::next_zoom_preset(&p, c, true);
        let dn = |c: f32| super::next_zoom_preset(&p, c, false);
        assert_eq!(up(71.0), 80.0);
        assert_eq!(up(80.0), 90.0);
        assert_eq!(up(300.0), 320.0);
        assert_eq!(up(500.0), 550.0);
        assert_eq!(up(20000.0), 20000.0, "clamps at the top");
        assert_eq!(dn(5.0), 5.0, "clamps at the bottom");
        assert_eq!(dn(95.0), 90.0);
        assert_eq!(dn(320.0), 300.0);
        assert_eq!(dn(550.0), 500.0);
    }

    // A spliced fit-% (e.g. 71.34) becomes a reachable stop between fixed presets.
    #[test]
    fn zoom_stepping_dynamic_stop() {
        let ladder = vec![70.0, 71.34, 80.0, 90.0];
        assert_eq!(super::next_zoom_preset(&ladder, 70.0, true), 71.34);
        assert_eq!(super::next_zoom_preset(&ladder, 71.34, true), 80.0);
        assert_eq!(super::next_zoom_preset(&ladder, 71.34, false), 70.0);
    }
}

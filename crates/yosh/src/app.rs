//! Application: winit `ApplicationHandler`, owns GPU + egui + reader state.
//! M1.3: async decode pool + bounded cache + forward prefetch → hitch-free
//! navigation. The current page is drawn from the cache; if a target isn't ready
//! yet the last-drawn page is held (no flicker).

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

use crate::config;
use yosh_engine::decode::decode_and_downscale;
use crate::gpu::Gpu;
use crate::library::{cover_bytes, Library};
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::DecodePool;
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, RarSource, SevenzSource, ZipSource};
use yosh_engine::layout::{self, Layout};
use yosh_engine::reader::{Budget, Direction, Reader, Viewport};
use yosh_engine::texpool::TexturePool;
use crate::ui::{self, UiState};
use crate::update;

/// Memory (MB) the reader may spend on its page cache + GPU textures. A slice of
/// system RAM on the desktop; an Android shell would pass its per-app heap class.
/// Falls back to a generous value (→ the full desktop budget) when RAM is unknown.
fn detect_mem_budget_mb() -> u64 {
    (total_ram_mb().unwrap_or(8192) / 16).max(64)
}

#[cfg(windows)]
fn total_ram_mb() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `s` is a correctly-sized, zeroed MEMORYSTATUSEX with dwLength set.
    (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some(s.ullTotalPhys / (1024 * 1024))
}

#[cfg(target_os = "linux")]
fn total_ram_mb() -> Option<u64> {
    // /proc/meminfo line: "MemTotal:   16384000 kB"
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = info
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb / 1024)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn total_ram_mb() -> Option<u64> {
    None
}
/// Pixels scrolled per mouse-wheel line in continuous-scroll mode.
const SCROLL_WHEEL_PX: f32 = 110.0;
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
    /// The reading-state machine + the engine resources it drives (page source,
    /// decode pool, cache, texture pool, wgpu device/queue). The shell mirrors the
    /// surface into `reader.viewport` and feeds it input each frame.
    reader: Reader,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui: UiState,

    page_pipeline: PagePipeline,
    cursor_x: f64,
    cursor_y: f64,
    mouse_down: bool,
    drag_dist: f32, // accumulated drag distance, to distinguish click from pan
    cursor_in_window: bool, // gates the edge-hover navigation arrows
    last_mid_click: Option<Instant>, // middle-zone double-click → fullscreen

    settings: config::Settings,
    /// Last observed window geometry (outer x/y, inner w/h, physical px) while in
    /// the normal — not maximized, not fullscreen — state. Seeded from the saved
    /// settings so a session spent entirely maximized still persists the prior
    /// restored rect; updated on move/resize; written back on exit.
    win_geom: Option<(i32, i32, u32, u32)>,
    volume_key: Option<String>,
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
        if let Some(state) = self.state.as_mut() {
            // Resuming from background: the OS tore the surface down (Android does
            // this on every background), but the device, decode pool, cache and
            // reader state all survive. Make a fresh window + surface and carry on
            // with no re-decode. Desktop fires `resumed` only once, so this branch
            // is Android's; re-binding egui to the new window is the future Android
            // shell's concern.
            let attrs = Window::default_attributes()
                .with_title("yosh")
                .with_window_icon(window_icon());
            let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
            state.gpu.recreate_surface(window.clone());
            state.window = window;
            state.window.request_redraw();
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
        // Size the reader's resource budget to the device: CPU count + a slice of
        // system RAM. Desktop reproduces the historical fixed budget; a constrained
        // device (e.g. Android) scales cache / textures / prefetch down.
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let budget = Budget::derive(detect_mem_budget_mb(), cpus);
        let tex_pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));

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
        let reader = Reader::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            tex_pool,
            budget,
            fit_from_u8(settings.fit),
            if settings.layout_spread {
                Layout::Spread
            } else {
                Layout::Single
            },
            settings.scroll,
            if settings.direction_rtl {
                Direction::Rtl
            } else {
                Direction::Ltr
            },
            self.start_index,
            false, // two_tier: desktop keeps the always-HQ pipeline
        );
        self.state = Some(State {
            window,
            gpu,
            reader,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            page_pipeline,
            cursor_x: 0.0,
            cursor_y: 0.0,
            mouse_down: false,
            drag_dist: 0.0,
            cursor_in_window: false,
            last_mid_click: None,
            win_geom: settings.window.map(|w| (w.x, w.y, w.w, w.h)),
            settings,
            volume_key: None,
            info_for: None,
            loading_pending: None,
            toast: None,
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
                state.reader.viewport = Viewport {
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
                if self.reader.scroll_mode {
                    let vh = self.reader.viewport.h as f32;
                    self.reader.scroll_by(vh * 0.9);
                } else {
                    self.reader.step(1);
                }
            }
            Action::Backward => {
                if self.reader.scroll_mode {
                    let vh = self.reader.viewport.h as f32;
                    self.reader.scroll_by(-vh * 0.9);
                } else {
                    self.reader.step(-1);
                }
            }
            // In RTL, "left" advances the story; in LTR, "right" does. (Page-flip only.)
            Action::Right if !self.reader.scroll_mode => {
                self.reader.step(if self.reader.direction == Direction::Ltr { 1 } else { -1 });
            }
            Action::Left if !self.reader.scroll_mode => {
                self.reader.step(if self.reader.direction == Direction::Ltr { -1 } else { 1 });
            }
            Action::Right | Action::Left => {}
            Action::First => self.reader.goto(0),
            Action::Last => {
                if let Some(s) = &self.reader.source {
                    self.reader.goto(s.len().saturating_sub(1));
                }
            }
            Action::CycleFit if self.reader.scroll_mode => {
                // In scroll: toggle width-fit (zoom 1) vs height-fit (a typical
                // page ~fills the viewport height).
                self.reader.pan_x = 0.0;
                if (self.reader.zoom - 1.0).abs() < 0.01 {
                    let sw = self.reader.viewport.w.max(1) as f32;
                    let sh = self.reader.viewport.h.max(1) as f32;
                    let cw = sh / self.reader.est_aspect.max(0.1);
                    self.reader.zoom = (cw / sw).clamp(0.2, 8.0);
                } else {
                    self.reader.zoom = 1.0;
                }
            }
            Action::CycleFit => {
                self.reader.fit = self.reader.fit.cycle();
                self.reader.zoom = 1.0;
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
                self.settings.fit = fit_to_u8(self.reader.fit);
                config::save(&self.settings);
            }
            Action::ZoomIn => self.reader.zoom_to_preset(true),
            Action::ZoomOut => self.reader.zoom_to_preset(false),
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
                self.reader.direction = match self.reader.direction {
                    Direction::Ltr => Direction::Rtl,
                    Direction::Rtl => Direction::Ltr,
                };
                self.settings.direction_rtl = self.reader.direction == Direction::Rtl;
                config::save(&self.settings);
                self.toast(format!("Direction: {}", self.reader.direction.label()));
            }
            Action::ToggleLayout => {
                self.reader.layout = self.reader.layout.toggled();
                // Snap to the current view's anchor so pairing is consistent.
                self.reader.index = layout::view_start(self.reader.layout, self.reader.index, self.reader.spread_offset);
                self.reader.pan_y = 0.0;
                self.settings.layout_spread = self.reader.layout == Layout::Spread;
                config::save(&self.settings);
                self.reader.prefetch();
                self.toast(format!("Layout: {}", self.reader.layout.label()));
            }
            Action::ToggleScroll => {
                self.reader.scroll_mode = !self.reader.scroll_mode;
                self.reader.top_offset = 0.0;
                self.settings.scroll = self.reader.scroll_mode;
                config::save(&self.settings);
                self.reader.prefetch();
                self.toast(if self.reader.scroll_mode {
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
                // Announce the change like the other toggles — a silent flip left
                // the seekbar disabled with no clue it had been turned off.
                self.toast(if self.settings.seekbar_enabled {
                    "Seekbar: on"
                } else {
                    "Seekbar: off"
                });
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
                self.reader.spread_offset ^= 1;
                if let Some(k) = &self.volume_key {
                    self.settings
                        .spread_offsets
                        .insert(k.clone(), self.reader.spread_offset as u8);
                    config::save(&self.settings);
                }
                // Re-anchor so the current view re-pairs with the new parity.
                self.reader.index = layout::view_start(self.reader.layout, self.reader.index, self.reader.spread_offset);
                self.reader.prefetch();
                self.toast(format!("Spread offset: {}", self.reader.spread_offset));
            }
            Action::PrevVolume => self.jump_volume(-1),
            Action::NextVolume => self.jump_volume(1),
            Action::Rotate => {
                self.reader.rotation = (self.reader.rotation + 1) % 4;
                // Recenter: the rotated box has different bounds, so any prior pan
                // would now be out of range.
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
                self.reader.prefetch(); // re-decode at the rotation-aware target (1:1)
                self.toast(format!("Rotation: {}\u{00b0}", self.reader.rotation as u32 * 90));
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
            match self.reader.source.as_ref() {
                Some(s) if s.len() > 0 => base.join(s.name(self.reader.index)),
                _ => base,
            }
        } else {
            base // archive (or single file): select it in its folder
        };
        reveal_in_explorer(&target);
        self.toast("Shown in Explorer");
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
            self.settings.last_pages.insert(k.clone(), self.reader.index);
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
        if self.reader.scroll_mode {
            let dy_px = match delta {
                MouseScrollDelta::LineDelta(_, y) => y * SCROLL_WHEEL_PX,
                MouseScrollDelta::PixelDelta(p) => p.y as f32,
            };
            self.reader.scroll_by(-dy_px); // wheel down (y<0) scrolls the strip down
            return;
        }
        let dy = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
        };
        if dy == 0.0 {
            return;
        }
        let overflow = self.reader.current_overflows();
        if !overflow {
            // Page fits: wheel flips (down = forward).
            self.reader.step(if dy < 0.0 { 1 } else { -1 });
            return;
        }
        // Vertical pan in px. At the top/bottom edge, hard-stop first: a scroll
        // that would overshoot the edge just parks at it; only a *further* scroll
        // once already parked flips the page. So scrolling to the end of a zoomed
        // page stops there instead of immediately jumping to the next page — you
        // have to keep scrolling past the stop to advance. Only reset the pan when
        // a flip actually happened (else the first/last page snaps to its edge).
        let sh = self.reader.viewport.h.max(1) as f32;
        let maxp = ((self.reader.current_display_h() - sh) / 2.0).max(0.0);
        let cur = self.reader.pan_y.clamp(-maxp, maxp);
        let next = cur + dy * 80.0;
        let now = Instant::now();
        // True once we've been parked at an edge long enough that a further
        // scroll should flip (the hard stop the user has to keep scrolling past).
        let dwelt = self.reader.pan_edge_at.is_some_and(|t| now.duration_since(t) >= EDGE_DWELL);
        if next > maxp + 0.5 {
            if cur >= maxp - 0.5 {
                // Parked at the top: flip to the previous page only after dwelling.
                if dwelt {
                    self.reader.pan_edge_at = None;
                    self.reader.pan_y = if self.reader.step(-1) { -1.0e6 } else { maxp };
                } else {
                    self.reader.pan_y = maxp; // hold the stop
                    self.reader.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.reader.pan_y = maxp; // just reached the top edge -> park + start dwell
                self.reader.pan_edge_at = Some(now);
            }
        } else if next < -maxp - 0.5 {
            if cur <= -maxp + 0.5 {
                // Parked at the bottom: flip to the next page only after dwelling.
                if dwelt {
                    self.reader.pan_edge_at = None;
                    self.reader.pan_y = if self.reader.step(1) { 1.0e6 } else { -maxp };
                } else {
                    self.reader.pan_y = -maxp; // hold the stop
                    self.reader.pan_edge_at.get_or_insert(now);
                }
            } else {
                self.reader.pan_y = -maxp; // just reached the bottom edge -> park + start dwell
                self.reader.pan_edge_at = Some(now);
            }
        } else {
            self.reader.pan_y = next;
            self.reader.pan_edge_at = None; // panning within the page
        }
    }

    /// A clean click: the left/right edge strips flip pages; the wide middle
    /// does nothing on a single click but toggles fullscreen on a double-click.
    fn on_click(&mut self) {
        let w = self.reader.viewport.w.max(1) as f64;
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
        if self.reader.scroll_mode {
            // Grab the strip: pan horizontally, scroll vertically.
            self.reader.pan_x += dx;
            self.reader.top_offset -= dy;
            self.reader.clamp_pan();
            self.reader.normalize();
        } else {
            self.reader.pan_x += dx;
            self.reader.pan_y += dy;
            self.reader.clamp_pan();
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
                PagePipeline::upload(&self.gpu.device, &self.gpu.queue, &img, &self.reader.tex_pool, 0);
            let id = self.egui_renderer.register_native_texture(
                &self.gpu.device,
                &pt.view,
                wgpu::FilterMode::Linear,
            );
            self.library.volumes[i].thumb = Some(id);
            self.library.volumes[i].thumb_tex = Some(pt);
        }
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
            self.settings.last_pages.insert(k, self.reader.index);
        }
        let key = path.to_string_lossy().into_owned();
        self.reader.spread_offset = self.settings.spread_offsets.get(&key).copied().unwrap_or(0) as usize;
        // Explicit start (e.g. a specific dropped image) wins; else CLI start
        // index; else the saved position.
        let idx = match start {
            Some(i) => i,
            None => {
                let resume = self.settings.last_pages.get(&key).copied().unwrap_or(0);
                if self.reader.start_index > 0 {
                    self.reader.start_index
                } else {
                    resume
                }
            }
        };
        self.reader.start_index = 0;

        self.reader.pool = Some(DecodePool::new(
            source.clone(),
            self.gpu.device.clone(),
            self.gpu.queue.clone(),
            self.reader.tex_pool.clone(),
            self.reader.workers,
        ));
        self.reader.cache.clear();
        self.reader.lq_cache.clear();
        self.reader.failed.clear();
        self.reader.last_drawn = None;
        self.info_for = None;
        self.reader.nav_times.clear();
        self.reader.rotation = 0; // each volume opens upright
        self.reader.index = idx.min(source.len() - 1);
        self.volume_key = Some(key);
        self.ui.opened = Some(path.to_path_buf());
        self.reader.source = Some(source);
        self.library_view = false; // opening anything switches to the reader
        self.reader.prefetch();
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
        self.reader.scroll_mode = false;
        self.reader.fit = fit;
        self.reader.layout = if spread { Layout::Spread } else { Layout::Single };
        if let Some(d) = dir {
            self.reader.direction = d;
        }
        self.reader.index = layout::view_start(self.reader.layout, self.reader.index, self.reader.spread_offset);
        self.reader.zoom = 1.0;
        self.reader.pan_x = 0.0;
        self.reader.pan_y = 0.0;
        self.settings.fit = fit_to_u8(self.reader.fit);
        self.settings.layout_spread = self.reader.layout == Layout::Spread;
        self.settings.direction_rtl = self.reader.direction == Direction::Rtl;
        self.settings.scroll = false;
        config::save(&self.settings);
        self.reader.prefetch();
        // Tell the user what the preset just switched to (presets change fit +
        // layout + maybe direction at once, so summarize the resulting view).
        let view_label = if spread {
            format!("Spread, {}", self.reader.direction.label())
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
        let Some(src) = &self.reader.source else {
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
        let mut lines = vec![
            ("File".to_string(), name),
            ("Page".to_string(), format!("{} / {}", index + 1, src.len())),
            ("Size".to_string(), size),
            ("Modified".to_string(), modified),
            ("Resolution".to_string(), res),
            ("Format".to_string(), fmt),
            ("Color".to_string(), color),
        ];
        // LQ preview tier: fill progress + what's on screen for this page (HQ
        // full-res, the soft LQ thumbnail, or neither yet).
        let showing = if self.reader.cache.contains(index) {
            "HQ"
        } else if self.reader.lq_cache.contains(index) {
            "LQ preview"
        } else {
            "—"
        };
        lines.push((
            "LQ tier".to_string(),
            format!("{}/{} · {}", self.reader.lq_cache.len(), src.len(), showing),
        ));
        lines
    }

    /// The in-view anchor page if it is an animated (GIF/WebP) page with its texture
    /// decoded — the page the mini playback controls govern.
    fn anim_anchor(&self) -> Option<usize> {
        if self.library_view {
            return None;
        }
        let len = self.reader.source.as_ref()?.len();
        let anchor = if self.reader.scroll_mode {
            self.reader.index
        } else {
            layout::view_pages(self.reader.layout, self.reader.index, len, self.reader.spread_offset).0
        };
        (self.reader.cache.get(anchor)?.frame_count() > 1).then_some(anchor)
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
        let frames = self.reader.cache.get(anchor).map_or(1, |t| t.frame_count());
        // GIF/WebP auto-play; `.ico` layers are stepped manually (no play/pause).
        let is_anim = self.reader.cache.get(anchor).is_some_and(|t| t.is_animation());
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
                    .reader
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
        self.playback.page.and_then(|p| self.reader.cache.get(p)).map_or(1, |t| t.frame_count())
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
        let Some(src) = &self.reader.source else {
            return "yosh".to_string();
        };
        let len = src.len();
        if len == 0 {
            return "yosh".to_string();
        }
        // The page actually shown (anchor): `index` in single/scroll, the first
        // page of the pair in a two-page spread.
        let anchor = if self.reader.scroll_mode {
            self.reader.index
        } else {
            layout::view_pages(self.reader.layout, self.reader.index, len, self.reader.spread_offset).0
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
        let res = match self.reader.cache.get(anchor) {
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
        self.reader.viewport = Viewport {
            w: self.gpu.config.width,
            h: self.gpu.config.height,
        };
        if let Some(p) = self.ui.pending_open.take() {
            self.open(&p);
        }

        // Drain finished decodes into the right cache (full-res → cache, LQ-tier
        // thumbnails → lq_cache) and record failures.
        self.reader.drain_pool();
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
                    let partial = source.is_partial();
                    let n = source.len();
                    self.set_source(source, &key, start);
                    if partial {
                        self.toast(format!("Partial archive — {n} pages recovered"));
                    }
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
        self.reader.update_decode_view();
        // Keep the scroll anchor valid as page heights resolve, then refresh work.
        if self.reader.scroll_mode {
            self.reader.normalize();
        }
        self.reader.prefetch();
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
        } else if self.reader.scroll_mode {
            self.reader.build_scroll_quads()
        } else {
            self.reader.build_quads()
        };
        self.ui.dir_label = self.reader.direction.label();
        self.ui.fit_label = self.reader.fit.label();
        self.ui.layout_label = if self.reader.scroll_mode {
            "scroll"
        } else {
            self.reader.layout.label()
        };
        // Build the Tab info overlay text, reading the source once per page change.
        if self.ui.info_open && !self.library_view && self.info_for != Some(self.reader.index) {
            self.ui.info = self.build_page_info(self.reader.index);
            self.info_for = Some(self.reader.index);
        }
        // Live view state for the overlays: current zoom % (shown in the info
        // overlay, refreshed every frame so it tracks zooming without a rebuild)
        // and the active toast (dropped once it expires).
        self.ui.zoom_pct = self.reader.effective_zoom_pct();
        self.reader.update_resize_readout();
        self.ui.resize_path = self.reader.resize_path_label();
        // Drain transient messages the reader queued (boundary hit, zoom level)
        // into the shell's timed toast.
        if let Some(m) = self.reader.pending_toasts.drain(..).last() {
            self.toast(m);
        }
        // Persist the read position: the reader owns `index`, the shell owns the
        // volume key + settings. Cheap per-frame; flushed to disk on exit.
        if let Some(k) = &self.volume_key {
            self.settings.last_pages.insert(k.clone(), self.reader.index);
        }
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
        let win_w = self.reader.viewport.w.max(1) as f32;
        let edge = win_w * EDGE_FRAC;
        let in_reader = self.reader.source.is_some() && !self.library_view && !self.reader.scroll_mode;
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
        if let Some(src) = &self.reader.source {
            let len = src.len();
            let anchor = if self.reader.scroll_mode {
                self.reader.index
            } else {
                layout::view_pages(self.reader.layout, self.reader.index, len, self.reader.spread_offset).0
            };
            let in_cache = self.reader.cache.contains(anchor);
            // A page whose decode errored is in `failed`; treat it as not-loading so
            // we show a failure notice (file name + reason) instead of spinning.
            let fail_err: Option<String> =
                if in_cache { None } else { self.reader.failed.get(&anchor).cloned() };
            let failed = fail_err.is_some();
            let loading = !in_cache && !failed;
            if in_cache {
                self.reader.last_drawn = Some(anchor);
            }
            self.ui.status = format!(
                "{}/{}{}",
                self.reader.index + 1,
                len,
                if failed {
                    "  [failed]"
                } else if loading {
                    "  …"
                } else {
                    ""
                },
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
            self.ui.seek_index = self.reader.index;
            self.ui.seek_total = len;
            self.ui.seek_rtl = self.reader.direction == Direction::Rtl;
            self.ui.seek_style = ui::SeekbarStyle::Bar;
            self.ui.seek_buffered.clear();
            self.ui.seek_buffered.extend(self.reader.cache.buffered_indices());
            self.ui.seek_lq_buffered.clear();
            self.ui.seek_lq_buffered.extend(self.reader.lq_cache.buffered_indices());
            let win_h = self.reader.viewport.h.max(1) as f32;
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
                self.reader.page_texture(q.page_index).map(|t| {
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
            && let Some(src) = &self.reader.source
        {
            let page = page.min(src.len().saturating_sub(1));
            if page != self.reader.index {
                self.reader.goto(page);
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

}

//! Decode + downscale a page's encoded bytes to a display-resolution buffer.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, JPEG XL →
//! `jxl-oxide` (pure Rust), else → `image` crate fallback (WebP/GIF/BMP/AVIF/…).
//! Normalizes to single-channel R8 (gray) or
//! RGBA8 (color), then downscales with a high-quality, content-aware filter
//! (inspired by MangaJaNaiConverterGui's final-resize strategy):
//!   - **color** → Lanczos3 in gamma space (no color conversion),
//!   - **grayscale** → Catmull-Rom in **true 16-bit linear light**: linearize
//!     sRGB → linear luminance, resample, then re-encode through the Dot Gain 20%
//!     curve so screentones stay inky (see `tone.rs`). Linear-light resampling is
//!     what suppresses halftone moiré.
//!
//! Color decodes that are *visually* grayscale (within a threshold) are detected
//! and routed through the grayscale path, matching MangaJaNai's behavior.

use std::sync::atomic::{AtomicU32, Ordering};

use fast_image_resize::Resizer;
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;
use image::AnimationDecoder;

use crate::sr::{SrOutcome, SrSkip, Upscaler};
use yosh_image::decode::{
    color_manage, decode_jpeg_scaled, decode_raw, is_grayscale as rgba_is_grayscale, is_opaque,
    premultiply_alpha, rgb_to_rgba, rgba_to_luma, rgba_to_rgb, GRAYSCALE_THRESHOLD,
};
use yosh_image::resize;

/// GPU `max_texture_dimension_2d`, published by `gpu.rs` at startup. A decoded page
/// can't exceed this in either dimension (it's one texture), so pages that would
/// go over are rejected with a clear error rather than downscaled. wgpu's default
/// (8192) until set; modern GPUs report 16384.
pub static MAX_TEX_DIM: AtomicU32 = AtomicU32::new(8192);

/// Decoded size for a source of `(w, h)` at a desired `target_h`: scale to
/// `target_h` (never upscaling past the source), preserving aspect.
fn target_dims(w: u32, h: u32, target_h: u32) -> (u32, u32) {
    let th = target_h.min(h).max(1);
    let tw = (((w as f64) * (th as f64) / (h as f64)).round() as u32).max(1);
    (tw, th)
}

/// Reject a page whose decoded size won't fit in a single GPU texture (full
/// resolution is preserved up to the limit; we never silently downscale past it).
fn check_fits(tw: u32, th: u32) -> Result<(), String> {
    let max = MAX_TEX_DIM.load(Ordering::Relaxed);
    if tw > max || th > max {
        Err(format!("image too large for the GPU ({tw}x{th}; max {max} px per side)"))
    } else {
        Ok(())
    }
}

/// Which CPU resize path produced a page. Surfaced in the info overlay so the
/// active pipeline — and whether the GPU then has to resize at all — is visible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResizePath {
    /// No downscale: decoded at native (target ≥ source). The GPU draws it 1:1, or
    /// upscales for zoom-past-native magnification.
    None,
    /// Grayscale source, resampled in true linear light (Catmull-Rom) then
    /// re-encoded through the Dot Gain 20% curve (the screentone-safe path).
    GrayLinear,
    /// Color source detected as visually gray: collapsed to luma + the gray path.
    GrayFromColor,
    /// Color source: Lanczos3 in gamma space (ICC→sRGB first if the page was tagged).
    Color,
    /// AI-upscaled by the given factor (the SR add-on), then brought to the target
    /// by the gray linear-light path — the model replaces the GPU magnification.
    SrGray(u8),
    /// AI-upscaled by the given factor, then Lanczos3 to the target.
    SrColor(u8),
}

impl ResizePath {
    pub fn label(self) -> &'static str {
        match self {
            ResizePath::None => "none (native res)",
            ResizePath::GrayLinear => "gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::GrayFromColor => "gray-from-color (Catmull-Rom + Dot Gain)",
            ResizePath::Color => "color (Lanczos3)",
            ResizePath::SrGray(2) => "AI 2\u{d7} + gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::SrGray(4) => "AI 4\u{d7} + gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::SrGray(_) => "AI + gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::SrColor(2) => "AI 2\u{d7} + color (Lanczos3)",
            ResizePath::SrColor(4) => "AI 4\u{d7} + color (Lanczos3)",
            ResizePath::SrColor(_) => "AI + color (Lanczos3)",
        }
    }
}

/// A decoded, downscaled page ready for GPU upload.
pub struct DecodedImage {
    pub w: u32,
    pub h: u32,
    /// Native (pre-downscale) source dimensions, kept so the UI can report the
    /// zoom level relative to the original image (the texture is decoded to ~the
    /// display size, so `w`/`h` alone can't reveal it).
    pub src_w: u32,
    pub src_h: u32,
    /// true => single-channel R8; false => RGBA8.
    pub gray: bool,
    /// Which CPU resize path produced this image (for the info overlay).
    pub path: ResizePath,
    pub pixels: Vec<u8>,
}

/// Expand a grayscale (R8) image to RGBA8 (r=g=b, a=255). Used for egui
/// thumbnails, since egui samples textures as RGBA (an R8 texture would render
/// red). No-op for images that are already color.
pub fn to_rgba_image(img: DecodedImage) -> DecodedImage {
    if !img.gray {
        return img;
    }
    let mut pixels = Vec::with_capacity(img.pixels.len() * 4);
    for &g in &img.pixels {
        pixels.extend_from_slice(&[g, g, g, 255]);
    }
    DecodedImage {
        w: img.w,
        h: img.h,
        src_w: img.src_w,
        src_h: img.src_h,
        gray: false,
        path: img.path,
        pixels,
    }
}

/// Grayscale strategy: resample in **true linear light** to kill halftone moiré.
/// Linearize the sRGB-encoded source to 16-bit linear luminance, Catmull-Rom
/// resample in that space, then re-encode through the Dot Gain 20% curve (which
/// darkens, keeping screentones inky). 16-bit intermediate avoids shadow banding.
fn downscale_gray(
    gray: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let pixels = resize::gray_linear(gray, (w, h), (tw, target_h), resizer)?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: true, path: ResizePath::GrayLinear, pixels })
}

/// Color strategy (MangaJaNai `standard_resize`): Lanczos3 in gamma space, no
/// color conversion.
fn downscale_color(
    rgba: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let pixels = resize::lanczos(rgba, 4, (w, h), (tw, target_h), resizer)?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: false, path: ResizePath::Color, pixels })
}

/// LQ grayscale: a fast 8-bit Bilinear downscale in gamma space. Skips the HQ
/// cost — the 32M-px sRGB→linear pass, the U16 Catmull-Rom, and the Dot-Gain
/// re-encode — so it's several times faster (and shows some screentone moiré).
/// Used transiently while seeking; the page re-decodes via `downscale_gray` on
/// settle.
fn downscale_gray_fast(
    gray: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let pixels = resize::bilinear(gray, 1, (w, h), (tw, target_h), resizer)?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: true, path: ResizePath::GrayLinear, pixels })
}

/// LQ color: a fast 8-bit Bilinear downscale (vs the HQ Lanczos3).
fn downscale_color_fast(
    rgba: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let pixels = resize::bilinear(rgba, 4, (w, h), (tw, target_h), resizer)?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: false, path: ResizePath::Color, pixels })
}

/// A still page decoded at full resolution and color-managed to sRGB, before any
/// resize — the shared front half of the HQ ([`decode_and_downscale`]) and AI
/// upscale ([`decode_sr`]) paths.
struct Prepared {
    w: u32,
    h: u32,
    /// Single-channel source (`full` is 1 byte per pixel); RGBA otherwise.
    gray_by_channels: bool,
    full: Vec<u8>,
    opaque: bool,
}

fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    let (w, h, gray_by_channels, mut full, profile) = decode_raw(bytes)?;
    // Color-manage to sRGB before any resampling (see `color_manage`).
    color_manage(gray_by_channels, &mut full, profile.as_deref());
    // A color page may carry transparency; opaque pages (the manga norm) keep the
    // unchanged fast path. Computed once and reused for the routing decisions.
    let opaque = gray_by_channels || is_opaque(&full);
    Ok(Prepared { w, h, gray_by_channels, full, opaque })
}

/// Decode page bytes (any supported format) and downscale to `target_h` on CPU,
/// picking the grayscale or color resize strategy.
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    finish(prepare(bytes)?, target_h, resizer)
}

/// The back half of [`decode_and_downscale`]: size a prepared page to `target_h`.
fn finish(p: Prepared, target_h: u32, resizer: &mut Resizer) -> Result<DecodedImage, String> {
    let Prepared { w, h, gray_by_channels, mut full, opaque } = p;
    // Decoded size = scale to the display height (never upscaling past the source).
    // A page bigger than one GPU texture is rejected (full res is preserved up to
    // the limit; we don't silently downscale a 16k-px image to a blurry one).
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;

    // No downscale needed (source already fits the target and the GPU limit):
    // show the decoded pixels unaltered — no resampling, no tone remap.
    if tw == w && th == h {
        if !opaque {
            premultiply_alpha(&mut full);
        }
        return Ok(DecodedImage { w, h, src_w: w, src_h: h, gray: gray_by_channels, path: ResizePath::None, pixels: full });
    }

    if gray_by_channels {
        // Already single-channel (1ch / GA PNG, L8 JPEG) — no scan needed.
        downscale_gray(&full, w, h, tw, th, resizer)
    } else if opaque && rgba_is_grayscale(&full, GRAYSCALE_THRESHOLD) {
        // Color-stored but visually gray (and opaque) → collapse to luma, gray
        // strategy. Transparent images skip this so their alpha is preserved.
        let mut img = downscale_gray(&rgba_to_luma(&full), w, h, tw, th, resizer)?;
        img.path = ResizePath::GrayFromColor; // same gray path, but source was color
        Ok(img)
    } else {
        if !opaque {
            premultiply_alpha(&mut full);
        }
        downscale_color(&full, w, h, tw, th, resizer)
    }
}

/// A page's full-size model output: what the model made of the whole source,
/// before the final resize. It doesn't depend on the display size, so a zoom or
/// window resize only needs [`sr_from_master`] again — not the model.
pub struct SrMaster {
    pub model: std::sync::Arc<str>,
    pub scale: u32,
    /// The source page's size; `pixels` is `scale` times it on both axes.
    pub src_w: u32,
    pub src_h: u32,
    /// One luma channel (`true`) or packed RGB.
    pub gray: bool,
    pub pixels: Vec<u8>,
}

/// Result of [`decode_sr`].
pub enum SrDecode {
    /// The AI-upscaled page, sized to the target like any HQ decode, and the
    /// model output it was sized from (kept by the pool for re-sizing).
    Page(DecodedImage, std::sync::Arc<SrMaster>),
    /// No upscaled page, and why — the normal decode stays. (A model or runtime
    /// failure is an `Err` instead, carrying its message.)
    Skipped(SrSkip),
    /// The caller cancelled mid-inference.
    Cancelled,
}

/// The AI-upscale sibling of [`decode_and_downscale`], for the pool's SR lane:
/// decode at full resolution, upscale through `up` (gray pages — stored gray or
/// visually gray — as one luma channel, color as RGB), then bring the result to
/// `target_h` through the same HQ resize as every other page. So the GPU samples
/// it 1:1: the model takes over exactly the magnification the GPU would otherwise
/// do, and the single-resize invariant holds. `src_w`/`src_h` stay the file's.
pub fn decode_sr(
    bytes: &[u8],
    target_h: u32,
    up: &dyn Upscaler,
    cancel: &dyn Fn() -> bool,
    resizer: &mut Resizer,
) -> Result<SrDecode, String> {
    if is_multi_image(bytes) {
        return Ok(SrDecode::Skipped(SrSkip::Unsupported));
    }
    let p = prepare(bytes)?;
    if target_h <= p.h || !p.opaque {
        return Ok(SrDecode::Skipped(SrSkip::Unsupported));
    }
    let gray = p.gray_by_channels || rgba_is_grayscale(&p.full, GRAYSCALE_THRESHOLD);
    let input = if p.gray_by_channels {
        p.full
    } else if gray {
        rgba_to_luma(&p.full)
    } else {
        rgba_to_rgb(&p.full)
    };
    let ch = if gray { 1 } else { 3 };
    let (scale, out, model) = match up.upscale(p.w, p.h, ch, &input, target_h as f32 / p.h as f32, cancel) {
        SrOutcome::Done { scale, pixels, model } => (scale, pixels, model),
        SrOutcome::Skip => return Ok(SrDecode::Skipped(SrSkip::NoModel)),
        SrOutcome::NotReady => return Ok(SrDecode::Skipped(SrSkip::NotReady)),
        SrOutcome::Cancelled => return Ok(SrDecode::Cancelled),
        SrOutcome::Failed(e) => return Err(e),
    };
    let master = std::sync::Arc::new(SrMaster { model, scale, src_w: p.w, src_h: p.h, gray, pixels: out });
    let img = sr_from_master(&master, target_h, resizer)?;
    Ok(SrDecode::Page(img, master))
}

/// Size a kept model output to `target_h` through the same HQ resize as every
/// other page (so the GPU samples it 1:1). The back half of [`decode_sr`], and
/// all a zoom/resize of an upscaled page needs while its model is unchanged.
pub fn sr_from_master(m: &SrMaster, target_h: u32, resizer: &mut Resizer) -> Result<DecodedImage, String> {
    let (sw, sh) = (m.src_w * m.scale, m.src_h * m.scale);
    let (tw, th) = target_dims(sw, sh, target_h);
    check_fits(tw, th)?;
    let mut img = if m.gray {
        if tw == sw && th == sh {
            DecodedImage { w: sw, h: sh, src_w: sw, src_h: sh, gray: true, path: ResizePath::None, pixels: m.pixels.clone() }
        } else {
            downscale_gray(&m.pixels, sw, sh, tw, th, resizer)?
        }
    } else {
        let rgba = rgb_to_rgba(&m.pixels, sw, sh);
        if tw == sw && th == sh {
            DecodedImage { w: sw, h: sh, src_w: sw, src_h: sh, gray: false, path: ResizePath::None, pixels: rgba }
        } else {
            downscale_color(&rgba, sw, sh, tw, th, resizer)?
        }
    };
    img.src_w = m.src_w;
    img.src_h = m.src_h;
    let s = m.scale.min(u8::MAX as u32) as u8;
    img.path = if m.gray { ResizePath::SrGray(s) } else { ResizePath::SrColor(s) };
    Ok(img)
}

/// LQ sibling of `decode_and_downscale`: decode + a cheap gamma-space Bilinear
/// resize, skipping ICC color management, the visually-grayscale detection, and
/// the linear-light path. The fast tier shown while seeking; a native-sized page
/// (no downscale) returns the same pixels HQ would, so nothing is lost there.
///
/// JPEGs additionally decode through a reduced-size IDCT (see `idct_eighths`) —
/// the biggest single win for the whole-volume thumbnail fill, which used to
/// full-res-decode every page of a volume just to shrink it to 540 px. **The HQ
/// path never does this**: its output must be the one exact resample of the full
/// source data.
fn decode_and_downscale_lq(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let ((w, h, gray_by_channels, full, _profile), (src_w, src_h)) =
        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            decode_jpeg_scaled(bytes, target_h)?
        } else {
            let d = decode_raw(bytes)?;
            let src = (d.0, d.1);
            (d, src)
        };
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;
    let mut img = if tw == w && th == h {
        DecodedImage { w, h, src_w: w, src_h: h, gray: gray_by_channels, path: ResizePath::None, pixels: full }
    } else if gray_by_channels {
        downscale_gray_fast(&full, w, h, tw, th, resizer)?
    } else {
        downscale_color_fast(&full, w, h, tw, th, resizer)?
    };
    // The IDCT hint means the decoded buffer may be smaller than the file, so the
    // source dims have to come from the header, not from what we decoded.
    img.src_w = src_w;
    img.src_h = src_h;
    Ok(img)
}

/// A decoded page: a single still image (the common case), an animation as an
/// ordered list of `(frame, delay_ms)` (GIF/WebP — auto-plays), or a set of
/// static layers (an `.ico`'s multiple resolutions — stepped manually, no
/// playback). Single-frame/-layer inputs collapse to `Still`.
pub enum DecodedPage {
    Still(DecodedImage),
    Animated(Vec<(DecodedImage, u32)>),
    Layered(Vec<DecodedImage>),
}

/// Downscale one already-decoded, canvas-composited RGBA frame to `target_h` via
/// the color (Lanczos3) path — the strategy for animated GIF frames (palette
/// color, not manga screentone, so the gray linear-light path doesn't apply).
fn downscale_rgba_frame(
    mut rgba: Vec<u8>,
    w: u32,
    h: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    // GIF/WebP frames can be transparent — premultiply before resize/return.
    if !is_opaque(&rgba) {
        premultiply_alpha(&mut rgba);
    }
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;
    if tw == w && th == h {
        return Ok(DecodedImage { w, h, src_w: w, src_h: h, gray: false, path: ResizePath::None, pixels: rgba });
    }
    downscale_color(&rgba, w, h, tw, th, resizer)
}

/// Turn an animation's decoded frames into a `DecodedPage`: downscale each frame
/// (color path) and keep its delay. The decoder (`GifDecoder` / `WebPDecoder`)
/// hands back each frame **pre-composited to the full canvas** (disposal already
/// applied), so each is a complete same-size RGBA image. A single frame collapses
/// to `Still` so a non-animated file pays no animation overhead.
fn frames_to_page(
    frames: Vec<image::Frame>,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
    if frames.is_empty() {
        return Err("animation: no frames".into());
    }
    let mut out: Vec<(DecodedImage, u32)> = Vec::with_capacity(frames.len());
    for frame in frames {
        // Delay as a ms ratio (numer/denom). Clamp tiny/zero delays to 100ms,
        // matching browsers (which treat <20ms as 100ms) so a 0ms frame can't pin
        // the loop.
        let (num, den) = frame.delay().numer_denom_ms();
        let ms = num.checked_div(den).unwrap_or(0);
        let delay = if ms < 20 { 100 } else { ms };
        let buf = frame.into_buffer(); // RgbaImage, full canvas
        let (w, h) = buf.dimensions();
        out.push((downscale_rgba_frame(buf.into_raw(), w, h, target_h, resizer)?, delay));
    }
    if out.len() == 1 {
        Ok(DecodedPage::Still(out.pop().unwrap().0))
    } else {
        Ok(DecodedPage::Animated(out))
    }
}

/// Decode every image inside an `.ico` (its multiple resolutions / "layers"),
/// largest first. Each entry decodes to RGBA8 at its own native size; they are
/// kept native (icons are tiny) and shown scaled to the page box.
fn decode_ico(bytes: &[u8]) -> Result<Vec<DecodedImage>, String> {
    let dir = ico::IconDir::read(std::io::Cursor::new(bytes)).map_err(|e| format!("ico: {e}"))?;
    let mut out: Vec<DecodedImage> = Vec::with_capacity(dir.entries().len());
    for entry in dir.entries() {
        let img = entry.decode().map_err(|e| format!("ico entry: {e}"))?;
        let (w, h) = (img.width(), img.height());
        let mut pixels = img.rgba_data().to_vec();
        // Icons are typically transparent — premultiply so the GPU blend composites
        // them over the background instead of showing garbage in clear areas.
        if !is_opaque(&pixels) {
            premultiply_alpha(&mut pixels);
        }
        out.push(DecodedImage { w, h, src_w: w, src_h: h, gray: false, path: ResizePath::None, pixels });
    }
    if out.is_empty() {
        return Err("ico: no entries".into());
    }
    // Largest first, so the default-shown layer is the highest resolution.
    out.sort_by_key(|i| std::cmp::Reverse((i.w as u64) * (i.h as u64)));
    Ok(out)
}

/// Pages [`decode_page`] decodes as more than one image: `.ico` (layers), GIF
/// (always frame-decoded) and animated WebP. Same magic checks as `decode_page`.
fn is_multi_image(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x00, 0x00, 0x01, 0x00])
        || bytes.starts_with(b"GIF8")
        || (bytes.len() >= 12
            && &bytes[0..4] == b"RIFF"
            && &bytes[8..12] == b"WEBP"
            && WebPDecoder::new(std::io::Cursor::new(bytes)).is_ok_and(|d| d.has_animation()))
}

/// Decode a page to a still image, an animation (GIF/WebP), or layered (`.ico`).
/// This is the entry point the decode pool uses; stills go through the unchanged
/// `decode_and_downscale` hot path.
pub fn decode_page(
    bytes: &[u8],
    target_h: u32,
    lq: bool,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
    // ICO: expose every contained image as a steppable layer (1 entry → still).
    if bytes.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        let mut layers = decode_ico(bytes)?;
        return Ok(if layers.len() == 1 {
            DecodedPage::Still(layers.pop().unwrap())
        } else {
            DecodedPage::Layered(layers)
        });
    }
    // GIF is always frame-decoded (a 1-frame GIF collapses back to a still).
    if bytes.starts_with(b"GIF8") {
        let frames = GifDecoder::new(std::io::Cursor::new(bytes))
            .map_err(|e| format!("gif: {e}"))?
            .into_frames()
            .collect_frames()
            .map_err(|e| format!("gif frames: {e}"))?;
        return frames_to_page(frames, target_h, resizer);
    }
    // WebP: frame-decode only when it's actually animated; a static WebP takes the
    // normal still path (with ICC color management) like any other image.
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
        && let Ok(dec) = WebPDecoder::new(std::io::Cursor::new(bytes))
            && dec.has_animation()
        {
            let frames = dec
                .into_frames()
                .collect_frames()
                .map_err(|e| format!("webp frames: {e}"))?;
            return frames_to_page(frames, target_h, resizer);
        }
    // Stills: the seek hot path. LQ uses the fast gamma-space resize; HQ is the
    // unchanged linear-light pipeline. (Animations/ICO above always decode HQ —
    // rare, and not the seek bottleneck.)
    let img = if lq {
        decode_and_downscale_lq(bytes, target_h, resizer)?
    } else {
        decode_and_downscale(bytes, target_h, resizer)?
    };
    Ok(DecodedPage::Still(img))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icc;
    use yosh_image::decode::idct_eighths;
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};

    fn encode_gif(frames: Vec<Frame>) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            for f in frames {
                enc.encode_frame(f).unwrap();
            }
        } // drop encoder → flush trailer
        buf
    }

    fn frame(rgba: [u8; 4]) -> Frame {
        let img = RgbaImage::from_pixel(4, 4, Rgba(rgba));
        Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1))
    }

    /// Encode a solid-color 4-component JPEG. `ink` is **ink-coverage** CMYK
    /// (0 = no ink), the convention `jpeg-encoder` takes and `jpeg-decoder` hands
    /// back — the encoder applies the Adobe inversion itself. `color_type` selects
    /// which Adobe APP14 transform is written: `Cmyk` → 0, `CmykAsYcck` → 2.
    fn encode_cmyk_jpeg(ink: [u8; 4], color_type: jpeg_encoder::ColorType) -> Vec<u8> {
        let (w, h) = (32u16, 32u16);
        let data: Vec<u8> = ink.iter().copied().cycle().take(w as usize * h as usize * 4).collect();
        let mut buf = Vec::new();
        jpeg_encoder::Encoder::new(&mut buf, 98)
            .encode(&data, w, h, color_type)
            .unwrap();
        buf
    }

    fn assert_solid_rgb(img: &DecodedImage, want: [u8; 3], what: &str) {
        assert!(!img.gray, "{what}: a CMYK page must decode as RGBA8");
        assert_eq!(img.pixels.len(), (img.w * img.h * 4) as usize, "{what}: RGBA8 length");
        // Sample the middle so JPEG block edges don't skew it.
        let i = (((img.h / 2) * img.w + img.w / 2) * 4) as usize;
        let got = [img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]];
        let ok = got.iter().zip(&want).all(|(g, w)| (*g as i32 - *w as i32).abs() <= 24);
        assert!(ok, "{what}: expected ~{want:?}, got {got:?}");
        assert_eq!(img.pixels[i + 3], 255, "{what}: opaque");
    }

    /// Issue #14: a CMYK JPEG used to fail outright with
    /// "jpeg: unsupported pixel format CMYK32". Both Adobe transforms must decode —
    /// plain CMYK (APP14 transform 0) and YCCK (transform 2), which is what
    /// Photoshop and ImageMagick actually emit.
    #[test]
    fn cmyk_and_ycck_jpegs_decode() {
        // Pure cyan ink → red channel fully absorbed, green/blue pass through.
        let cases = [
            (jpeg_encoder::ColorType::Cmyk, "CMYK (APP14 transform 0)"),
            (jpeg_encoder::ColorType::CmykAsYcck, "YCCK (APP14 transform 2)"),
        ];
        for (ct, what) in cases {
            let bytes = encode_cmyk_jpeg([255, 0, 0, 0], ct);
            let mut resizer = Resizer::new();
            match decode_page(&bytes, 32, false, &mut resizer).unwrap() {
                DecodedPage::Still(img) => assert_solid_rgb(&img, [0, 255, 255], what),
                _ => panic!("{what}: a jpeg is a still"),
            }
        }
    }

    /// The K channel must darken rather than invert — a sign error here would show
    /// as a near-white page instead of a near-black one.
    #[test]
    fn cmyk_black_ink_decodes_dark() {
        let bytes = encode_cmyk_jpeg([0, 0, 0, 255], jpeg_encoder::ColorType::Cmyk);
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 32, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => assert_solid_rgb(&img, [0, 0, 0], "full K ink"),
            _ => panic!("a jpeg is a still"),
        }
    }

    /// A CMYK ICC profile must be recognized so `decode_and_downscale` skips color
    /// management. It cannot be applied to the already-RGB pixels, and qcms does
    /// *not* reject the mismatch — it silently maps white to blue.
    #[test]
    fn cmyk_icc_profile_is_detected() {
        let mut prof = vec![0u8; 132];
        prof[16..20].copy_from_slice(b"CMYK");
        assert!(icc::is_cmyk(&prof));
        assert!(!icc::is_gray(&prof));

        let mut gray = vec![0u8; 132];
        gray[16..20].copy_from_slice(b"GRAY");
        assert!(!icc::is_cmyk(&gray), "GRAY must not read as CMYK");

        let mut rgb = vec![0u8; 132];
        rgb[16..20].copy_from_slice(b"RGB ");
        assert!(!icc::is_cmyk(&rgb), "RGB must not read as CMYK");
    }

    /// The IDCT-scale chooser must never pick a reduction that lands *below* the
    /// requested height — the CPU resize is a downscale-only path, so undershooting
    /// would mean upscaling a thumbnail (blurry) instead of reducing it. It must
    /// also actually reduce when there is headroom, which is the entire win.
    #[test]
    fn idct_scale_covers_the_target_and_still_reduces() {
        for src_h in [540u32, 1024, 1600, 2048, 4096, 5207] {
            for target in [90u32, 180, 360, 540, 1080, 2160] {
                let s = idct_eighths(src_h, target);
                assert!((1..=8).contains(&s), "src {src_h} → {target}: scale {s}");
                let out = src_h.saturating_mul(s).div_ceil(8);
                assert!(
                    out >= target.min(src_h),
                    "src {src_h} → {target}: reduced to {out}, below the target"
                );
                // And it is the *smallest* such scale (no wasted IDCT work).
                if s > 1 {
                    let smaller = src_h.saturating_mul(s / 2).div_ceil(8);
                    assert!(smaller < target, "src {src_h} → {target}: {s}/8 is bigger than needed");
                }
            }
        }
        // A page already at or below the target decodes at full scale.
        assert_eq!(idct_eighths(400, 540), 8);
        // A 540px thumb of a 4096px page: 1/8 (512) is short, 2/8 (1024) covers it.
        assert_eq!(idct_eighths(4096, 540), 2);
        // A huge page for a tiny thumb takes the cheapest IDCT there is.
        assert_eq!(idct_eighths(5207, 90), 1);
        // `u32::MAX` (the uncached 1:1 target) can't overflow into a bogus scale.
        assert_eq!(idct_eighths(5207, u32::MAX), 8);
    }

    /// End-to-end LQ JPEG decode: the output still lands at the exact requested
    /// height (the IDCT reduction is invisible in the result), and `src_w`/`src_h`
    /// keep describing the *file* — they drive the zoom readout and the 1:1 decode
    /// target, so reporting the reduced buffer's size there would misreport zoom and
    /// re-decode loops. The HQ path is unaffected, which the same page proves.
    #[test]
    fn lq_jpeg_decodes_scaled_but_reports_true_source_dims() {
        let (w, h) = (600u16, 1200u16);
        let rgb: Vec<u8> = (0..(w as usize * h as usize))
            .flat_map(|i| [(i % 251) as u8, (i % 253) as u8, (i % 257) as u8])
            .collect();
        let mut bytes = Vec::new();
        jpeg_encoder::Encoder::new(&mut bytes, 90)
            .encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb)
            .unwrap();

        let mut resizer = Resizer::new();
        for lq in [true, false] {
            match decode_page(&bytes, 150, lq, &mut resizer).unwrap() {
                DecodedPage::Still(img) => {
                    assert_eq!(img.h, 150, "lq={lq}: decoded to the exact target height");
                    assert_eq!((img.src_w, img.src_h), (600, 1200), "lq={lq}: true source dims");
                }
                _ => panic!("a jpeg is a still"),
            }
        }
    }

    #[test]
    fn multiframe_gif_decodes_as_animation() {
        let bytes = encode_gif(vec![frame([255, 0, 0, 255]), frame([0, 0, 255, 255])]);
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, false, &mut resizer).unwrap() {
            DecodedPage::Animated(fs) => {
                assert_eq!(fs.len(), 2, "both frames preserved");
                assert!(fs.iter().all(|(img, _)| img.w == 4 && img.h == 4 && !img.gray));
                // 100ms round-trips through the centisecond GIF delay field.
                assert!(fs.iter().all(|(_, d)| *d == 100), "delays = {:?}", fs.iter().map(|(_, d)| *d).collect::<Vec<_>>());
            }
            _ => panic!("expected an animation"),
        }
    }

    #[test]
    fn single_frame_gif_is_a_still() {
        let bytes = encode_gif(vec![frame([0, 255, 0, 255])]);
        let mut resizer = Resizer::new();
        assert!(matches!(
            decode_page(&bytes, 4, false, &mut resizer).unwrap(),
            DecodedPage::Still(_)
        ));
    }

    /// A minimal 4×4, 8-bit RGB PSD with raw (uncompressed) merged image data:
    /// header → empty color-mode/resources/layer sections → planar R,G,B planes.
    fn minimal_rgb_psd() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"8BPS"); // signature
        b.extend_from_slice(&1u16.to_be_bytes()); // version 1 = PSD
        b.extend_from_slice(&[0u8; 6]); // reserved
        b.extend_from_slice(&3u16.to_be_bytes()); // channels = RGB
        b.extend_from_slice(&4u32.to_be_bytes()); // height
        b.extend_from_slice(&4u32.to_be_bytes()); // width
        b.extend_from_slice(&8u16.to_be_bytes()); // depth = 8
        b.extend_from_slice(&3u16.to_be_bytes()); // color mode = RGB
        b.extend_from_slice(&0u32.to_be_bytes()); // color mode data: none
        b.extend_from_slice(&0u32.to_be_bytes()); // image resources: none
        b.extend_from_slice(&0u32.to_be_bytes()); // layer & mask info: none
        b.extend_from_slice(&0u16.to_be_bytes()); // compression = raw
        b.extend(std::iter::repeat_n(255u8, 16)); // R plane
        b.extend(std::iter::repeat_n(0u8, 16)); // G plane
        b.extend(std::iter::repeat_n(0u8, 16)); // B plane
        b
    }

    #[test]
    fn psd_decodes_flattened_composite() {
        let bytes = minimal_rgb_psd();
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (4, 4));
                assert!(!img.gray);
                assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255], "opaque red");
            }
            _ => panic!("psd should be a still"),
        }
    }

    #[test]
    fn png_16bit_decodes() {
        // A 16-bit RGBA PNG (e.g. from ImageMagick/Photoshop) — before the
        // normalize-to-8-bit transform this failed to decode (reported 8 channels).
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Sixteen);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[0xFFu8; 2 * 2 * 4 * 2]).unwrap(); // 2×2 RGBA16
        }
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 2, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (2, 2));
                assert!(!img.gray);
                assert_eq!(img.pixels.len(), 2 * 2 * 4, "down-converted to RGBA8");
            }
            _ => panic!("png is a still"),
        }
    }

    #[test]
    fn ico_decodes_as_layers() {
        use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
        let mut dir = IconDir::new(ResourceType::Icon);
        for sz in [16u32, 32u32] {
            let img = IconImage::from_rgba_data(sz, sz, vec![0xFFu8; (sz * sz * 4) as usize]);
            dir.add_entry(IconDirEntry::encode(&img).unwrap());
        }
        let mut bytes = Vec::new();
        dir.write(&mut bytes).unwrap();
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 64, false, &mut resizer).unwrap() {
            DecodedPage::Layered(layers) => {
                assert_eq!(layers.len(), 2);
                assert_eq!((layers[0].w, layers[0].h), (32, 32), "largest layer first");
                assert_eq!((layers[1].w, layers[1].h), (16, 16));
            }
            _ => panic!("multi-entry ico should be Layered"),
        }
    }

    #[test]
    fn transparent_png_is_premultiplied() {
        // 2x2 RGBA: opaque red, a fully-transparent pixel with garbage white RGB,
        // opaque green, and a half-alpha pixel.
        let data: [u8; 16] = [
            255, 0, 0, 255, // opaque red
            255, 255, 255, 0, // transparent — garbage RGB must premultiply to 0
            0, 255, 0, 255, // opaque green
            200, 100, 50, 128, // half alpha
        ];
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&data).unwrap();
        }
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 2, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert!(!img.gray, "transparent image keeps the color/alpha path");
                assert_eq!(&img.pixels[4..8], &[0, 0, 0, 0], "transparent RGB zeroed");
                assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255], "opaque unchanged");
                assert_eq!(&img.pixels[8..12], &[0, 255, 0, 255], "opaque unchanged");
                let p = &img.pixels[12..16]; // ~ rgb * 128/255
                assert_eq!(p[3], 128);
                assert!((p[0] as i32 - 100).abs() <= 1 && (p[1] as i32 - 50).abs() <= 1);
            }
            _ => panic!("png is a still"),
        }
    }

    #[test]
    fn tiff_and_qoi_decode_via_image_crate() {
        use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
        // These formats have no dedicated decoder in yosh — they round-trip through
        // the `image`-crate fallback (decode_other), proving the easy-batch support.
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(5, 3, Rgba([10, 20, 30, 255])));
        for fmt in [ImageFormat::Tiff, ImageFormat::Qoi, ImageFormat::Tga] {
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, fmt).unwrap();
            let bytes = buf.into_inner();
            let mut resizer = Resizer::new();
            match decode_page(&bytes, 3, false, &mut resizer).unwrap() {
                DecodedPage::Still(d) => assert_eq!((d.w, d.h), (5, 3), "{fmt:?} dims"),
                _ => panic!("{fmt:?} should be a still"),
            }
        }
    }

    /// A stand-in model: nearest-neighbour by a fixed factor, recording what it
    /// was handed (channel count, target scale).
    struct FakeUpscaler {
        scale: u32,
        outcome: fn() -> Option<SrOutcome>,
        seen: std::sync::Mutex<Vec<(u32, f32)>>,
    }

    impl FakeUpscaler {
        fn new(scale: u32) -> Self {
            Self { scale, outcome: || None, seen: Default::default() }
        }
    }

    impl Upscaler for FakeUpscaler {
        fn upscale(&self, w: u32, h: u32, ch: u32, px: &[u8], ts: f32, _: &dyn Fn() -> bool) -> SrOutcome {
            self.seen.lock().unwrap().push((ch, ts));
            if let Some(o) = (self.outcome)() {
                return o;
            }
            let (s, c) = (self.scale, ch as usize);
            let mut out = Vec::with_capacity(px.len() * (s * s) as usize);
            for y in 0..h * s {
                for x in 0..w * s {
                    let i = ((y / s) * w + x / s) as usize * c;
                    out.extend_from_slice(&px[i..i + c]);
                }
            }
            SrOutcome::Done { scale: s, pixels: out, model: "fake".into() }
        }
        fn epoch(&self) -> u64 {
            0
        }
        fn model_for(&self, _: u32, _: bool, _: f32) -> Option<std::sync::Arc<str>> {
            Some("fake".into())
        }
    }

    fn png_bytes(w: u32, h: u32, color: png::ColorType, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut enc = png::Encoder::new(&mut bytes, w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(data).unwrap();
        bytes
    }

    fn sr(bytes: &[u8], target_h: u32, up: &FakeUpscaler) -> SrDecode {
        decode_sr(bytes, target_h, up, &|| false, &mut Resizer::new()).unwrap()
    }

    /// The single-resize invariant on the SR path: the model's output is brought to
    /// exactly the on-screen size by the HQ CPU resize (so the GPU samples it 1:1),
    /// and the page still reports the *file's* dimensions for zoom/aspect math.
    #[test]
    fn sr_lands_exactly_at_the_target_with_source_dims() {
        let gray: Vec<u8> = (0..100 * 150).map(|i| (i % 251) as u8).collect();
        let bytes = png_bytes(100, 150, png::ColorType::Grayscale, &gray);
        let up = FakeUpscaler::new(2);
        let SrDecode::Page(img, _) = sr(&bytes, 240, &up) else { panic!("gray page upscales") };
        assert_eq!((img.w, img.h), (160, 240), "decoded to the drawn size");
        assert_eq!((img.src_w, img.src_h), (100, 150), "source dims are the file's");
        assert!(img.gray && img.path == ResizePath::SrGray(2));
        assert_eq!(img.pixels.len(), 160 * 240);
        assert_eq!(up.seen.lock().unwrap()[0], (1, 1.6), "gray goes in as one channel");

        // Exactly the model's size: no resample at all.
        let SrDecode::Page(img, _) = sr(&bytes, 300, &up) else { panic!() };
        assert_eq!((img.w, img.h), (200, 300));

        // Beyond the model's reach the texture caps at its output; the GPU magnifies
        // the rest (the one permitted GPU resample).
        let SrDecode::Page(img, _) = sr(&bytes, 900, &up) else { panic!() };
        assert_eq!((img.w, img.h), (200, 300));
    }

    #[test]
    fn sr_routes_color_and_visually_gray_pages() {
        // Saturated color → RGB in, color path out.
        let rgb: Vec<u8> = (0..40 * 60).flat_map(|i| [(i % 200) as u8, 30, 220]).collect();
        let up = FakeUpscaler::new(4);
        let SrDecode::Page(img, _) = sr(&png_bytes(40, 60, png::ColorType::Rgb, &rgb), 200, &up) else {
            panic!("color page upscales")
        };
        assert!(!img.gray && img.path == ResizePath::SrColor(4));
        assert_eq!((img.w, img.h, img.pixels.len()), (133, 200, 133 * 200 * 4));
        assert_eq!(up.seen.lock().unwrap()[0].0, 3, "color goes in as RGB");

        // An RGB file that is visually gray takes the 1-channel gray route.
        let g: Vec<u8> = (0..40 * 60).flat_map(|i| [(i % 250) as u8; 3]).collect();
        let up = FakeUpscaler::new(2);
        let SrDecode::Page(img, _) = sr(&png_bytes(40, 60, png::ColorType::Rgb, &g), 100, &up) else {
            panic!()
        };
        assert!(img.gray && img.path == ResizePath::SrGray(2));
        assert_eq!(up.seen.lock().unwrap()[0].0, 1);
    }

    #[test]
    fn sr_skips_what_it_should_not_touch() {
        let gray = vec![128u8; 50 * 80];
        let bytes = png_bytes(50, 80, png::ColorType::Grayscale, &gray);
        let up = FakeUpscaler::new(2);
        let unsupported = |d: SrDecode| matches!(d, SrDecode::Skipped(SrSkip::Unsupported));
        // Shown at or below source size: nothing for a model to do.
        assert!(unsupported(sr(&bytes, 80, &up)));
        assert!(unsupported(sr(&bytes, 40, &up)));
        // Transparent pages keep their alpha on the normal path.
        let rgba: Vec<u8> = (0..50 * 80).flat_map(|_| [10, 200, 30, 100]).collect();
        assert!(unsupported(sr(&png_bytes(50, 80, png::ColorType::Rgba, &rgba), 200, &up)));
        // Animations aren't stills.
        let mut gif = Vec::new();
        {
            use image::codecs::gif::GifEncoder;
            let mut enc = GifEncoder::new(&mut gif);
            for _ in 0..2 {
                enc.encode_frame(image::Frame::new(image::RgbaImage::new(4, 4))).unwrap();
            }
        }
        assert!(unsupported(sr(&gif, 64, &up)));
        assert!(up.seen.lock().unwrap().is_empty(), "the model was never called");

        // The model's own verdicts pass through.
        let mut up = FakeUpscaler::new(2);
        up.outcome = || Some(SrOutcome::NotReady);
        assert!(matches!(sr(&bytes, 200, &up), SrDecode::Skipped(SrSkip::NotReady)));
        up.outcome = || Some(SrOutcome::Skip);
        assert!(matches!(sr(&bytes, 200, &up), SrDecode::Skipped(SrSkip::NoModel)));
        up.outcome = || Some(SrOutcome::Failed("boom".into()));
        assert!(decode_sr(&bytes, 200, &up, &|| false, &mut Resizer::new()).is_err());
    }

    /// Re-sizing a kept model output gives exactly what a full decode → model →
    /// resize gives at that size — so a zoom of an upscaled page can skip the
    /// model (and the file) entirely.
    #[test]
    fn resizing_a_kept_model_output_matches_a_full_upscale() {
        let gray: Vec<u8> = (0..90 * 130).map(|i| (i * 37 % 256) as u8).collect();
        let bytes = png_bytes(90, 130, png::ColorType::Grayscale, &gray);
        let up = FakeUpscaler::new(4);
        let mut r = Resizer::new();
        let SrDecode::Page(_, master) = decode_sr(&bytes, 260, &up, &|| false, &mut r).unwrap() else {
            panic!()
        };
        assert_eq!((master.scale, master.src_w, master.src_h, master.pixels.len()), (4, 90, 130, 360 * 520));
        for target in [273, 400, 520, 700] {
            let SrDecode::Page(full, _) = decode_sr(&bytes, target, &up, &|| false, &mut r).unwrap() else {
                panic!()
            };
            let resized = sr_from_master(&master, target, &mut r).unwrap();
            assert_eq!((resized.w, resized.h), (full.w, full.h), "target {target}");
            assert!(resized.pixels == full.pixels, "target {target}: same pixels");
            assert_eq!((resized.src_w, resized.src_h, resized.path), (90, 130, full.path));
        }
    }

    /// The HQ path is `prepare` + `finish` now; it must still produce exactly what
    /// it did (a native-size page passes through untouched, a gray page takes the
    /// linear-light path).
    #[test]
    fn hq_decode_is_unchanged_by_the_sr_split() {
        let gray: Vec<u8> = (0..64 * 96).map(|i| (i * 7 % 256) as u8).collect();
        let bytes = png_bytes(64, 96, png::ColorType::Grayscale, &gray);
        let mut r = Resizer::new();
        let native = decode_and_downscale(&bytes, 200, &mut r).unwrap();
        assert_eq!((native.w, native.h, native.path), (64, 96, ResizePath::None));
        assert_eq!(native.pixels, gray, "native pixels unaltered");
        let small = decode_and_downscale(&bytes, 48, &mut r).unwrap();
        assert_eq!((small.w, small.h, small.path), (32, 48, ResizePath::GrayLinear));
    }
}

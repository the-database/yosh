//! Decode + downscale a page's encoded bytes to a display-resolution buffer.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, else → `image`
//! crate fallback (WebP/GIF/BMP/…). Normalizes to single-channel R8 (gray) or
//! RGBA8 (color), then downscales with a high-quality, content-aware filter
//! (inspired by MangaJaNaiConverterGui's final-resize strategy):
//!   - **color** → Lanczos3 in gamma space (no color conversion),
//!   - **grayscale** → Catmull-Rom in **true 16-bit linear light**: linearize
//!     sRGB → linear luminance, resample, then re-encode through the Dot Gain 20%
//!     curve so screentones stay inky (see `tone.rs`). Linear-light resampling is
//!     what suppresses halftone moiré.
//! Color decodes that are *visually* grayscale (within a threshold) are detected
//! and routed through the grayscale path, matching MangaJaNai's behavior.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::tone;

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

/// MangaJaNai's default `GrayscaleDetectionThreshold` (its slider spans 0..24).
/// Higher = more tolerant of slight color casts when deciding "is this gray?".
const GRAYSCALE_THRESHOLD: i32 = 12;

/// A decoded, downscaled page ready for GPU upload.
pub struct DecodedImage {
    pub w: u32,
    pub h: u32,
    /// true => single-channel R8; false => RGBA8.
    pub gray: bool,
    pub pixels: Vec<u8>,
}

fn rgb_to_rgba(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4] = rgb[i * 3];
        out[i * 4 + 1] = rgb[i * 3 + 1];
        out[i * 4 + 2] = rgb[i * 3 + 2];
        out[i * 4 + 3] = 255;
    }
    out
}

fn ga_to_gray(ga: &[u8]) -> Vec<u8> {
    ga.iter().step_by(2).copied().collect()
}

/// Returns (w, h, gray, normalized pixels [1ch gray or 4ch rgba]).
fn decode_png(bytes: &[u8]) -> Result<(u32, u32, bool, Vec<u8>), String> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .map_err(|e| format!("png read_info: {e}"))?;
    let size = reader.output_buffer_size().ok_or("png: no buffer size")?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png next_frame: {e}"))?;
    let (w, h) = (info.width, info.height);
    let ch = info.buffer_size() / ((w as usize) * (h as usize));
    match ch {
        1 => Ok((w, h, true, buf)),
        2 => Ok((w, h, true, ga_to_gray(&buf))),
        3 => Ok((w, h, false, rgb_to_rgba(&buf, w, h))),
        4 => Ok((w, h, false, buf)),
        other => Err(format!("png: unsupported channel count {other}")),
    }
}

fn decode_jpeg(bytes: &[u8]) -> Result<(u32, u32, bool, Vec<u8>), String> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    match info.pixel_format {
        PixelFormat::L8 => Ok((w, h, true, pixels)),
        PixelFormat::RGB24 => Ok((w, h, false, rgb_to_rgba(&pixels, w, h))),
        other => Err(format!("jpeg: unsupported pixel format {other:?}")),
    }
}

fn decode_other(bytes: &[u8]) -> Result<(u32, u32, bool, Vec<u8>), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, false, rgba.into_raw()))
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
        gray: false,
        pixels,
    }
}

/// Decode to full resolution (no resize), normalized to gray (1ch) or RGBA8.
fn decode_raw(bytes: &[u8]) -> Result<(u32, u32, bool, Vec<u8>), String> {
    if bytes.starts_with(&PNG_SIG) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else {
        decode_other(bytes)
    }
}

/// Decode at full resolution (for the GPU-downscale path).
pub fn decode_full(bytes: &[u8]) -> Result<DecodedImage, String> {
    let (w, h, gray, pixels) = decode_raw(bytes)?;
    Ok(DecodedImage { w, h, gray, pixels })
}

/// Decide whether an RGBA buffer is *effectively* grayscale within `threshold`.
/// Port of MangaJaNai's `cv_image_is_grayscale` (run_upscale.py): for every pixel
/// that isn't pure black or pure white, sum the (saturating) channel-pair
/// differences beyond `threshold`; the image is gray if the mean per-channel
/// difference is `<= threshold / 12`. Channel order is irrelevant (all pairs).
fn rgba_is_grayscale(rgba: &[u8], threshold: i32) -> bool {
    let mut diff_sum: u64 = 0;
    let mut non_bw: u64 = 0;
    for px in rgba.chunks_exact(4) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        if (r == 0 && g == 0 && b == 0) || (r == 255 && g == 255 && b == 255) {
            continue; // exclude pure black / pure white
        }
        non_bw += 1;
        // cv2.subtract saturates at 0: max(|a-b| - threshold, 0).
        let rg = ((r - g).abs() - threshold).max(0);
        let rb = ((r - b).abs() - threshold).max(0);
        let gb = ((g - b).abs() - threshold).max(0);
        diff_sum += (rg + rb + gb) as u64;
    }
    if non_bw == 0 {
        return false; // entirely pure black/white → treat as color (MJN does)
    }
    let ratio = diff_sum as f64 / (non_bw as f64 * 3.0);
    ratio <= threshold as f64 / 12.0
}

/// Collapse RGBA to a single luminance channel (ITU-R 601, matching cv2's
/// `COLOR_BGR2GRAY`): `Y = 0.299R + 0.587G + 0.114B`.
fn rgba_to_luma(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|px| {
            let y = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            y.round().clamp(0.0, 255.0) as u8
        })
        .collect()
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
    // sRGB device → 16-bit linear luminance (native-endian U16 byte buffer).
    let mut lin = Vec::with_capacity(gray.len() * 2);
    for &v in gray {
        lin.extend_from_slice(&tone::SRGB_TO_LINEAR[v as usize].to_ne_bytes());
    }
    let src = ImageRef::new(w, h, &lin, PixelType::U16).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U16);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    // Linear luminance → Dot Gain 20% device (8-bit).
    let enc = tone::linear_to_dotgain();
    let bytes = dst.into_vec();
    let pixels: Vec<u8> = bytes
        .chunks_exact(2)
        .map(|c| enc[u16::from_ne_bytes([c[0], c[1]]) as usize])
        .collect();
    Ok(DecodedImage { w: tw, h: target_h, gray: true, pixels })
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
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8x4);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage { w: tw, h: target_h, gray: false, pixels: dst.into_vec() })
}

/// Decode page bytes (any supported format) and downscale to `target_h` on CPU,
/// picking the grayscale or color resize strategy.
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let (w, h, gray_by_channels, full) = decode_raw(bytes)?;
    // Never upscale a page beyond its own resolution (target tracks display size,
    // which can exceed a low-res source).
    let target_h = target_h.min(h).max(1);
    let tw = (((w as f64) * (target_h as f64) / (h as f64)).round() as u32).max(1);

    if gray_by_channels {
        // Already single-channel (1ch / GA PNG, L8 JPEG) — no scan needed.
        downscale_gray(&full, w, h, tw, target_h, resizer)
    } else if rgba_is_grayscale(&full, GRAYSCALE_THRESHOLD) {
        // Color-stored but visually gray → collapse to luma, gray strategy.
        downscale_gray(&rgba_to_luma(&full), w, h, tw, target_h, resizer)
    } else {
        downscale_color(&full, w, h, tw, target_h, resizer)
    }
}

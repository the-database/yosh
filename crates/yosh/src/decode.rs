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
//! Color decodes that are *visually* grayscale (within a threshold) are detected
//! and routed through the grayscale path, matching MangaJaNai's behavior.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::ImageDecoder;

use crate::icc;
use crate::tone;

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

/// MangaJaNai's default `GrayscaleDetectionThreshold` (its slider spans 0..24).
/// Higher = more tolerant of slight color casts when deciding "is this gray?".
const GRAYSCALE_THRESHOLD: i32 = 12;

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

/// Returns (w, h, gray, normalized pixels [1ch gray or 4ch rgba], icc profile).
/// The ICC profile (if any) is read from the same decode — no second parse.
type Decoded = (u32, u32, bool, Vec<u8>, Option<Vec<u8>>);

fn decode_png(bytes: &[u8]) -> Result<Decoded, String> {
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
    let icc = reader.info().icc_profile.as_deref().map(<[u8]>::to_vec);
    match ch {
        1 => Ok((w, h, true, buf, icc)),
        2 => Ok((w, h, true, ga_to_gray(&buf), icc)),
        3 => Ok((w, h, false, rgb_to_rgba(&buf, w, h), icc)),
        4 => Ok((w, h, false, buf, icc)),
        other => Err(format!("png: unsupported channel count {other}")),
    }
}

fn decode_jpeg(bytes: &[u8]) -> Result<Decoded, String> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    let icc = d.icc_profile();
    match info.pixel_format {
        PixelFormat::L8 => Ok((w, h, true, pixels, icc)),
        PixelFormat::RGB24 => Ok((w, h, false, rgb_to_rgba(&pixels, w, h), icc)),
        other => Err(format!("jpeg: unsupported pixel format {other:?}")),
    }
}

/// Decode JPEG XL via the pure-Rust `jxl-oxide`. Renders the first frame (ignores
/// animation) and normalizes to the same gray/RGBA8 + ICC contract as the others;
/// jxl-oxide hands back samples in the image's own color space plus its embedded
/// ICC, so the downstream qcms→sRGB step (in `decode_and_downscale`) color-manages
/// it exactly like JPEG/PNG.
fn decode_jxl(bytes: &[u8]) -> Result<Decoded, String> {
    use jxl_oxide::{JxlImage, PixelFormat};
    let image = JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .map_err(|e| format!("jxl read: {e}"))?;
    let (w, h) = (image.width(), image.height());
    let icc = image.original_icc().map(<[u8]>::to_vec);
    let fmt = image.pixel_format();
    let render = image.render_frame(0).map_err(|e| format!("jxl render: {e}"))?;
    let fb = render.image_all_channels(); // interleaved f32, len = w*h*channels
    let buf = fb.buf();
    let ch = fb.channels();
    // jxl-oxide samples are f32 (≈[0,1] for SDR); clamp handles any HDR overshoot.
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    match fmt {
        // 1ch gray — map straight through.
        PixelFormat::Gray => Ok((w, h, true, buf.iter().map(|&v| to_u8(v)).collect(), icc)),
        // Gray+alpha — drop alpha to match the 1ch gray contract (like `ga_to_gray`).
        PixelFormat::Graya => {
            Ok((w, h, true, buf.chunks_exact(ch).map(|px| to_u8(px[0])).collect(), icc))
        }
        // RGB → RGBA8 (opaque).
        PixelFormat::Rgb => {
            let mut pixels = vec![0u8; (w as usize) * (h as usize) * 4];
            for (px, out) in buf.chunks_exact(ch).zip(pixels.chunks_exact_mut(4)) {
                out[0] = to_u8(px[0]);
                out[1] = to_u8(px[1]);
                out[2] = to_u8(px[2]);
                out[3] = 255;
            }
            Ok((w, h, false, pixels, icc))
        }
        // Already interleaved RGBA — map straight through.
        PixelFormat::Rgba => Ok((w, h, false, buf.iter().map(|&v| to_u8(v)).collect(), icc)),
        // CMYK(A) would need a CMS we don't enable; effectively never occurs for manga.
        other => Err(format!("jxl: unsupported pixel format {other:?}")),
    }
}

fn decode_other(bytes: &[u8]) -> Result<Decoded, String> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: {e}"))?;
    let mut decoder = reader.into_decoder().map_err(|e| format!("image: {e}"))?;
    let icc = decoder.icc_profile().ok().flatten();
    let img = image::DynamicImage::from_decoder(decoder).map_err(|e| format!("image: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, false, rgba.into_raw(), icc))
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
        pixels,
    }
}

/// JPEG XL signature: either a bare codestream (`FF 0A`) or the ISOBMFF
/// container's 12-byte JXL box (`\0\0\0\x0C JXL \r \n \x87 \n`).
fn is_jxl(bytes: &[u8]) -> bool {
    const JXL_BOX: [u8; 12] =
        [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A];
    bytes.starts_with(&[0xFF, 0x0A]) || bytes.starts_with(&JXL_BOX)
}

/// Decode to full resolution (no resize), normalized to gray (1ch) or RGBA8.
fn decode_raw(bytes: &[u8]) -> Result<Decoded, String> {
    if bytes.starts_with(&PNG_SIG) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if is_jxl(bytes) {
        decode_jxl(bytes)
    } else {
        decode_other(bytes)
    }
}

/// Decode at full resolution (for the dormant GPU-downscale path; not color-managed).
pub fn decode_full(bytes: &[u8]) -> Result<DecodedImage, String> {
    let (w, h, gray, pixels, _icc) = decode_raw(bytes)?;
    Ok(DecodedImage { w, h, src_w: w, src_h: h, gray, pixels })
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
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: true, pixels })
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
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: false, pixels: dst.into_vec() })
}

/// Decode page bytes (any supported format) and downscale to `target_h` on CPU,
/// picking the grayscale or color resize strategy.
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let (w, h, gray_by_channels, mut full, profile) = decode_raw(bytes)?;
    // Color-manage to sRGB before any resampling: a color page tagged with a
    // wider profile (e.g. Display P3) would otherwise render desaturated. The
    // profile comes from the same decode (no second parse); only color images
    // carrying a non-sRGB profile pay the transform — grayscale/untagged pages
    // are untouched, so seek throughput is unaffected.
    if !gray_by_channels {
        if let Some(p) = &profile {
            if !icc::is_srgb(p) {
                icc::to_srgb_rgba(p, &mut full);
            }
        }
    }
    // Never upscale a page beyond its own resolution (target tracks display size,
    // which can exceed a low-res source).
    let target_h = target_h.min(h).max(1);

    // No downscale needed (1:1 / "Actual" mode, or a source already <= target):
    // show the decoded pixels unaltered — no resampling, no tone remap.
    if target_h == h {
        return Ok(DecodedImage { w, h, src_w: w, src_h: h, gray: gray_by_channels, pixels: full });
    }

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

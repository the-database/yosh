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

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, ImageDecoder};

use crate::icc;
use crate::tone;

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

/// MangaJaNai's default `GrayscaleDetectionThreshold` (its slider spans 0..24).
/// Higher = more tolerant of slight color casts when deciding "is this gray?".
const GRAYSCALE_THRESHOLD: i32 = 12;

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
}

impl ResizePath {
    pub fn label(self) -> &'static str {
        match self {
            ResizePath::None => "none (native res)",
            ResizePath::GrayLinear => "gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::GrayFromColor => "gray-from-color (Catmull-Rom + Dot Gain)",
            ResizePath::Color => "color (Lanczos3)",
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

/// True if every RGBA pixel is fully opaque (alpha 255).
fn is_opaque(rgba: &[u8]) -> bool {
    rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255)
}

/// Premultiply R,G,B by A in place (gamma space — consistent with the rest of the
/// decode/resize path). Needed before downscaling and before the GPU's
/// premultiplied-alpha blend: it zeroes the garbage RGB encoders leave in
/// fully-transparent pixels and lets the bilinear sampler interpolate edges
/// without colour fringing.
fn premultiply_alpha(rgba: &mut [u8]) {
    for px in rgba.as_chunks_mut::<4>().0 {
        let a = px[3] as u32;
        px[0] = ((px[0] as u32 * a + 127) / 255) as u8;
        px[1] = ((px[1] as u32 * a + 127) / 255) as u8;
        px[2] = ((px[2] as u32 * a + 127) / 255) as u8;
    }
}

/// Returns (w, h, gray, normalized pixels [1ch gray or 4ch rgba], icc profile).
/// The ICC profile (if any) is read from the same decode — no second parse.
type Decoded = (u32, u32, bool, Vec<u8>, Option<Vec<u8>>);

fn decode_png(bytes: &[u8]) -> Result<Decoded, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Normalize to 8-bit colour: down-convert 16-bit → 8-bit and expand
    // palette / sub-8-bit grayscale / tRNS. Without this a 16-bit PNG reports 6
    // (RGB16) or 8 (RGBA16) bytes-per-pixel below and fails to decode (and a
    // paletted PNG would be misread as 1-channel gray). No-op for plain 8-bit.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
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
    // Peek at the headers before decoding: a 4-component JPEG (CMYK, or Adobe's
    // YCCK) goes to the `image` crate instead. Its JPEG backend is zune-jpeg, which
    // converts *both* Adobe transforms to RGB during decode; `jpeg-decoder` only
    // hands back raw CMYK32 that we'd have to convert (and ink-profile) ourselves.
    // Reading the info first and then decoding on the same decoder is supported —
    // the decode resumes from the already-parsed frame rather than re-parsing.
    d.read_info().map_err(|e| format!("jpeg read_info: {e}"))?;
    if d.info().map(|i| i.pixel_format) == Some(PixelFormat::CMYK32) {
        return decode_other(bytes);
    }
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
            for (px, out) in buf.chunks_exact(ch).zip(pixels.as_chunks_mut::<4>().0) {
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

/// Decode a Photoshop document to its flattened composite (the "merged image
/// data" Photoshop stores), via the pure-Rust `psd` crate. Handles 8-bit RGB(A)
/// documents only — CMYK / 16-bit / grayscale-mode / PSB files error out and the
/// page shows as failed. yosh reads PSD when browsing a folder/archive but is not
/// a registered `.psd` handler (that stays Photoshop's).
fn decode_psd(bytes: &[u8]) -> Result<Decoded, String> {
    let psd = psd::Psd::from_bytes(bytes).map_err(|e| format!("psd: {e:?}"))?;
    let (w, h) = (psd.width(), psd.height());
    // rgba() is the pre-composited final image: [R,G,B,A, …], len = w*h*4.
    Ok((w, h, false, psd.rgba(), None))
}

fn decode_other(bytes: &[u8]) -> Result<Decoded, String> {
    let guessed = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: {e}"))?;
    // TGA has no magic bytes, so content-guessing yields no format. Since the file
    // already passed the image-extension allowlist, fall back to decoding it as TGA.
    let reader = if guessed.format().is_some() {
        guessed
    } else {
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Tga)
    };
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
        path: img.path,
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

/// The smallest `jpeg-decoder` IDCT scale (in eighths: 1, 2, 4 or 8) whose output
/// height still **covers** `target_h`. The decoder can run a reduced-size inverse
/// DCT — 1/8, 1/4, 1/2 or full — which costs a fraction of the full IDCT and skips
/// the corresponding share of the upsample/color-convert work. Choosing the
/// smallest covering scale means the CPU resize still does a real (never an
/// upscaling) reduction to the exact target, so the LQ tier's output size is
/// unchanged — only the work to get there shrinks (≈4–16× less IDCT on a thumbnail).
fn idct_eighths(src_h: u32, target_h: u32) -> u32 {
    [1u32, 2, 4]
        .into_iter()
        .find(|&s| src_h.saturating_mul(s).div_ceil(8) >= target_h)
        .unwrap_or(8)
}

/// JPEG decode for the **LQ tier only**, asking the decoder for an IDCT-reduced
/// image when the target is far below native. Returns the usual [`Decoded`] (whose
/// `w`/`h` are the *reduced* buffer's) plus the file's true source dimensions,
/// which the caller must restore onto the [`DecodedImage`] — `src_w`/`src_h` drive
/// the zoom readout and the 1:1 decode target, and must always describe the file,
/// not the buffer.
fn decode_jpeg_scaled(bytes: &[u8], target_h: u32) -> Result<(Decoded, (u32, u32)), String> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    d.read_info().map_err(|e| format!("jpeg read_info: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    // 4-component (CMYK / YCCK) JPEGs go to the `image` crate, as in `decode_jpeg`.
    if info.pixel_format == PixelFormat::CMYK32 {
        let d = decode_other(bytes)?;
        let src = (d.0, d.1);
        return Ok((d, src));
    }
    let (src_w, src_h) = (info.width as u32, info.height as u32);
    let s = idct_eighths(src_h, target_h);
    if s < 8 {
        // Request the reduced size explicitly rather than passing the caller's
        // target: `choose_idct_size` matches on *either* axis, so an aspect-correct
        // request is what makes it land on the scale computed above.
        let req = |v: u32| v.saturating_mul(s).div_ceil(8).clamp(1, u16::MAX as u32) as u16;
        d.scale(req(src_w), req(src_h)).map_err(|e| format!("jpeg scale: {e}"))?;
    }
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?; // output size (post-scale)
    let (w, h) = (info.width as u32, info.height as u32);
    let icc = d.icc_profile();
    let decoded = match info.pixel_format {
        PixelFormat::L8 => (w, h, true, pixels, icc),
        PixelFormat::RGB24 => (w, h, false, rgb_to_rgba(&pixels, w, h), icc),
        other => return Err(format!("jpeg: unsupported pixel format {other:?}")),
    };
    Ok((decoded, (src_w, src_h)))
}

/// Decode to full resolution (no resize), normalized to gray (1ch) or RGBA8.
fn decode_raw(bytes: &[u8]) -> Result<Decoded, String> {
    if bytes.starts_with(&PNG_SIG) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if is_jxl(bytes) {
        decode_jxl(bytes)
    } else if bytes.starts_with(b"8BPS") {
        decode_psd(bytes)
    } else {
        decode_other(bytes)
    }
}

/// Decide whether an RGBA buffer is *effectively* grayscale within `threshold`.
/// Port of MangaJaNai's `cv_image_is_grayscale` (run_upscale.py): for every pixel
/// that isn't pure black or pure white, sum the (saturating) channel-pair
/// differences beyond `threshold`; the image is gray if the mean per-channel
/// difference is `<= threshold / 12`. Channel order is irrelevant (all pairs).
fn rgba_is_grayscale(rgba: &[u8], threshold: i32) -> bool {
    let mut diff_sum: u64 = 0;
    let mut non_bw: u64 = 0;
    for px in rgba.as_chunks::<4>().0 {
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
    rgba.as_chunks::<4>()
        .0
        .iter()
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| enc[u16::from_ne_bytes(*c) as usize])
        .collect();
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
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8x4);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: false, path: ResizePath::Color, pixels: dst.into_vec() })
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
    let src = ImageRef::new(w, h, gray, PixelType::U8).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: true, path: ResizePath::GrayLinear, pixels: dst.into_vec() })
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
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8x4);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage { w: tw, h: target_h, src_w: w, src_h: h, gray: false, path: ResizePath::Color, pixels: dst.into_vec() })
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
    // are untouched, so seek throughput is unaffected. A *grayscale* ICC (e.g. a
    // Dot Gain profile on a monochrome AVIF) is skipped: it can't be applied to
    // the RGBA buffer (channel mismatch → white), and the gray resize path below
    // handles its tone instead. A *CMYK* ICC (a print-sourced JPEG/TIFF) is
    // skipped for the same reason — see `icc::is_cmyk`, which fails silently
    // rather than loudly — and because the decoder already converted those pixels
    // to RGB, so the profile no longer describes the buffer.
    if !gray_by_channels
        && let Some(p) = &profile
            && !icc::is_srgb(p) && !icc::is_gray(p) && !icc::is_cmyk(p) {
                icc::to_srgb_rgba(p, &mut full);
            }
    // Decoded size = scale to the display height (never upscaling past the source).
    // A page bigger than one GPU texture is rejected (full res is preserved up to
    // the limit; we don't silently downscale a 16k-px image to a blurry one).
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;

    // A color page may carry transparency; opaque pages (the manga norm) keep the
    // unchanged fast path. Computed once and reused for the routing decisions.
    let opaque = gray_by_channels || is_opaque(&full);

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
}

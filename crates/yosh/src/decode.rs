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
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, ImageDecoder};

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
    } else if bytes.starts_with(b"8BPS") {
        decode_psd(bytes)
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
    // are untouched, so seek throughput is unaffected. A *grayscale* ICC (e.g. a
    // Dot Gain profile on a monochrome AVIF) is skipped: it can't be applied to
    // the RGBA buffer (channel mismatch → white), and the gray resize path below
    // handles its tone instead.
    if !gray_by_channels {
        if let Some(p) = &profile {
            if !icc::is_srgb(p) && !icc::is_gray(p) {
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

/// A decoded page: either a single still image (the common case — every format
/// except multi-frame GIF), or an animation as an ordered list of
/// `(frame, delay_ms)`. A single-frame GIF comes back as `Still`, so the rest of
/// the pipeline only pays the animation cost for GIFs that actually move.
pub enum DecodedPage {
    Still(DecodedImage),
    Animated(Vec<(DecodedImage, u32)>),
}

/// Downscale one already-decoded, canvas-composited RGBA frame to `target_h` via
/// the color (Lanczos3) path — the strategy for animated GIF frames (palette
/// color, not manga screentone, so the gray linear-light path doesn't apply).
fn downscale_rgba_frame(
    rgba: Vec<u8>,
    w: u32,
    h: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let target_h = target_h.min(h).max(1);
    if target_h == h {
        return Ok(DecodedImage { w, h, src_w: w, src_h: h, gray: false, pixels: rgba });
    }
    let tw = (((w as f64) * (target_h as f64) / (h as f64)).round() as u32).max(1);
    downscale_color(&rgba, w, h, tw, target_h, resizer)
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
        let ms = if den == 0 { 0 } else { num / den };
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

/// Decode a page to a still image or (for animated GIF / WebP) an animation. This
/// is the entry point the decode pool uses; stills go through the unchanged
/// `decode_and_downscale` hot path.
pub fn decode_page(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
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
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        if let Ok(dec) = WebPDecoder::new(std::io::Cursor::new(bytes))
            && dec.has_animation()
        {
            let frames = dec
                .into_frames()
                .collect_frames()
                .map_err(|e| format!("webp frames: {e}"))?;
            return frames_to_page(frames, target_h, resizer);
        }
    }
    Ok(DecodedPage::Still(decode_and_downscale(bytes, target_h, resizer)?))
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

    #[test]
    fn multiframe_gif_decodes_as_animation() {
        let bytes = encode_gif(vec![frame([255, 0, 0, 255]), frame([0, 0, 255, 255])]);
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, &mut resizer).unwrap() {
            DecodedPage::Animated(fs) => {
                assert_eq!(fs.len(), 2, "both frames preserved");
                assert!(fs.iter().all(|(img, _)| img.w == 4 && img.h == 4 && !img.gray));
                // 100ms round-trips through the centisecond GIF delay field.
                assert!(fs.iter().all(|(_, d)| *d == 100), "delays = {:?}", fs.iter().map(|(_, d)| *d).collect::<Vec<_>>());
            }
            DecodedPage::Still(_) => panic!("expected an animation"),
        }
    }

    #[test]
    fn single_frame_gif_is_a_still() {
        let bytes = encode_gif(vec![frame([0, 255, 0, 255])]);
        let mut resizer = Resizer::new();
        assert!(matches!(
            decode_page(&bytes, 4, &mut resizer).unwrap(),
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
        b.extend(std::iter::repeat(255u8).take(16)); // R plane
        b.extend(std::iter::repeat(0u8).take(16)); // G plane
        b.extend(std::iter::repeat(0u8).take(16)); // B plane
        b
    }

    #[test]
    fn psd_decodes_flattened_composite() {
        let bytes = minimal_rgb_psd();
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (4, 4));
                assert!(!img.gray);
                assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255], "opaque red");
            }
            DecodedPage::Animated(_) => panic!("psd should be a still"),
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
        match decode_page(&bytes, 2, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (2, 2));
                assert!(!img.gray);
                assert_eq!(img.pixels.len(), 2 * 2 * 4, "down-converted to RGBA8");
            }
            DecodedPage::Animated(_) => panic!("png is a still"),
        }
    }
}

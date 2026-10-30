//! Decode page/image bytes to full-resolution 8-bit pixels.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, JPEG XL →
//! `jxl-oxide`, PSD → `psd`, everything else → the `image` crate (WebP, GIF,
//! BMP, TIFF, AVIF with the `avif` feature, …). All pure Rust. Results are
//! normalized to one gray channel or RGBA8 (16-bit sources come back 8-bit).
//!
//! [`decode`] is the convenience entry point (color-managed to sRGB, straight
//! alpha kept). The lower-level pieces ([`decode_raw`], [`color_manage`],
//! [`decode_jpeg_scaled`]) are what yosh's display pipeline builds on.

use image::ImageDecoder;

use crate::icc;

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

/// MangaJaNai's default `GrayscaleDetectionThreshold` (its slider spans 0..24).
/// Higher = more tolerant of slight color casts when deciding "is this gray?".
pub const GRAYSCALE_THRESHOLD: i32 = 12;

/// A decoded image: `channels` is 1 (gray) or 4 (RGBA8, **straight** alpha).
/// Opaque color sources come back RGBA with alpha 255 — [`is_opaque`] tells, and
/// [`rgba_to_rgb`] packs them to RGB.
#[derive(Clone, Debug)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub channels: u8,
    pub pixels: Vec<u8>,
}

/// Decode to full resolution, color-managed to sRGB (an embedded non-sRGB ICC
/// profile is applied to color images), keeping alpha — a gray+alpha source
/// becomes RGBA. 16-bit sources come back 8-bit. No resize.
pub fn decode(bytes: &[u8]) -> Result<Image, String> {
    let (w, h, gray, mut pixels, profile) = decode_raw_with(bytes, true)?;
    color_manage(gray, &mut pixels, profile.as_deref());
    Ok(Image { w, h, channels: if gray { 1 } else { 4 }, pixels })
}

/// Apply an image's own ICC profile to bring RGBA pixels to sRGB, in place.
///
/// Color-manage to sRGB before any resampling: a color page tagged with a wider
/// profile (e.g. Display P3) would otherwise render desaturated. Only color
/// images carrying a non-sRGB profile pay the transform — gray/untagged images
/// are untouched. A *grayscale* ICC (e.g. a Dot Gain profile on a monochrome
/// AVIF) is skipped: it can't be applied to the RGBA buffer (channel mismatch →
/// white). A *CMYK* ICC (a print-sourced JPEG/TIFF) is skipped for the same
/// reason — see `icc::is_cmyk` — and because the decoder already converted those
/// pixels to RGB, so the profile no longer describes the buffer.
pub fn color_manage(gray: bool, pixels: &mut [u8], profile: Option<&[u8]>) {
    if !gray
        && let Some(p) = profile
        && !icc::is_srgb(p)
        && !icc::is_gray(p)
        && !icc::is_cmyk(p)
    {
        icc::to_srgb_rgba(p, pixels);
    }
}

/// Packed RGB → gray (ITU-R 601, as [`rgba_to_luma`]).
pub fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|px| {
            let y = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            y.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

pub fn rgb_to_rgba(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
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

/// Gray+alpha → straight RGBA (gray replicated into R, G, B).
fn ga_to_rgba(ga: &[u8]) -> Vec<u8> {
    ga.as_chunks::<2>().0.iter().flat_map(|&[g, a]| [g, g, g, a]).collect()
}

/// True if every RGBA pixel is fully opaque (alpha 255).
pub fn is_opaque(rgba: &[u8]) -> bool {
    rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255)
}

/// Premultiply R,G,B by A in place (gamma space — consistent with the rest of the
/// decode/resize path). Needed before downscaling and before the GPU's
/// premultiplied-alpha blend: it zeroes the garbage RGB encoders leave in
/// fully-transparent pixels and lets the bilinear sampler interpolate edges
/// without colour fringing.
pub fn premultiply_alpha(rgba: &mut [u8]) {
    for px in rgba.as_chunks_mut::<4>().0 {
        let a = px[3] as u32;
        px[0] = ((px[0] as u32 * a + 127) / 255) as u8;
        px[1] = ((px[1] as u32 * a + 127) / 255) as u8;
        px[2] = ((px[2] as u32 * a + 127) / 255) as u8;
    }
}

/// Returns (w, h, gray, normalized pixels [1ch gray or 4ch rgba], icc profile).
/// The ICC profile (if any) is read from the same decode — no second parse.
pub type Decoded = (u32, u32, bool, Vec<u8>, Option<Vec<u8>>);

fn decode_png(bytes: &[u8], keep_alpha: bool) -> Result<Decoded, String> {
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
        2 if keep_alpha => Ok((w, h, false, ga_to_rgba(&buf), icc)),
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
fn decode_jxl(bytes: &[u8], keep_alpha: bool) -> Result<Decoded, String> {
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
        // Gray+alpha, kept: expand to straight RGBA.
        PixelFormat::Graya if keep_alpha => Ok((
            w,
            h,
            false,
            buf.chunks_exact(ch)
                .flat_map(|px| {
                    let (g, a) = (to_u8(px[0]), to_u8(px[1]));
                    [g, g, g, a]
                })
                .collect(),
            icc,
        )),
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

/// JPEG XL signature: either a bare codestream (`FF 0A`) or the ISOBMFF
/// container's 12-byte JXL box (`\0\0\0\x0C JXL \r \n \x87 \n`).
pub fn is_jxl(bytes: &[u8]) -> bool {
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
pub fn idct_eighths(src_h: u32, target_h: u32) -> u32 {
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
pub fn decode_jpeg_scaled(bytes: &[u8], target_h: u32) -> Result<(Decoded, (u32, u32)), String> {
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

/// Decode to full resolution (no resize), normalized to gray (1ch) or RGBA8, with
/// the embedded ICC profile (not yet applied — see [`color_manage`]). A gray+alpha
/// source comes back as plain gray (alpha dropped): yosh's display contract.
/// [`decode`] is the color-managed, alpha-keeping entry point.
pub fn decode_raw(bytes: &[u8]) -> Result<Decoded, String> {
    decode_raw_with(bytes, false)
}

fn decode_raw_with(bytes: &[u8], keep_alpha: bool) -> Result<Decoded, String> {
    if bytes.starts_with(&PNG_SIG) {
        decode_png(bytes, keep_alpha)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if is_jxl(bytes) {
        decode_jxl(bytes, keep_alpha)
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
pub fn is_grayscale(rgba: &[u8], threshold: i32) -> bool {
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

/// Drop the alpha channel (an opaque page's RGBA → packed RGB, for the upscaler).
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&px[..3]);
    }
    out
}

/// Collapse RGBA to a single luminance channel (ITU-R 601, matching cv2's
/// `COLOR_BGR2GRAY`): `Y = 0.299R + 0.587G + 0.114B`.
pub fn rgba_to_luma(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|px| {
            let y = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            y.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

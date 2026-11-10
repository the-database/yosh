//! Decode page/image bytes to full-resolution 8-bit pixels.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, JPEG XL →
//! `jxl-oxide`, PSD → `psd`, everything else → the `image` crate (WebP, GIF,
//! BMP, TIFF, AVIF with the `avif` feature, …). All pure Rust. Results are
//! normalized to one gray channel or RGBA8 (16-bit sources come back 8-bit), and
//! turned upright by the file's EXIF Orientation (JPEG, PNG `eXIf`, WebP, TIFF;
//! JPEG XL applies its own orientation in `jxl-oxide`).
//!
//! An animation decodes to its first frame; [`animation`] tells one apart.
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
    let orientation = exif_orientation(reader.info().exif_metadata.as_deref());
    let decoded = match ch {
        1 => (w, h, true, buf, icc),
        2 if keep_alpha => (w, h, false, ga_to_rgba(&buf), icc),
        2 => (w, h, true, ga_to_gray(&buf), icc),
        3 => (w, h, false, rgb_to_rgba(&buf, w, h), icc),
        4 => (w, h, false, buf, icc),
        other => return Err(format!("png: unsupported channel count {other}")),
    };
    Ok(oriented(decoded, orientation))
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
    let orientation = exif_orientation(d.exif_data());
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    let icc = d.icc_profile();
    let decoded = match info.pixel_format {
        PixelFormat::L8 => (w, h, true, pixels, icc),
        PixelFormat::RGB24 => (w, h, false, rgb_to_rgba(&pixels, w, h), icc),
        other => return Err(format!("jpeg: unsupported pixel format {other:?}")),
    };
    Ok(oriented(decoded, orientation))
}

/// Decode JPEG XL via the pure-Rust `jxl-oxide`. Renders the first frame (ignores
/// animation), already turned upright by the codestream's orientation (`width()`/
/// `height()` are the oriented size; an Exif box's Orientation is ignored, as the
/// JPEG XL spec says), and normalizes to the same gray/RGBA8 + ICC contract as the others;
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
    // EXIF Orientation where the format carries it (WebP's EXIF chunk, TIFF's tag, a
    // CMYK JPEG's APP1); formats without it report none.
    let orientation = decoder.orientation().map_or(1, |o| o.to_exif());
    let img = image::DynamicImage::from_decoder(decoder).map_err(|e| format!("image: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok(oriented((w, h, false, rgba.into_raw(), icc), orientation))
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
/// `w`/`h` are the *reduced* buffer's) plus the file's true source dimensions
/// (upright, like the pixels: EXIF Orientation is applied here too),
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
    // `target_h` is the upright height: a page stored on its side is that tall in width.
    let orientation = exif_orientation(d.exif_data());
    let s = idct_eighths(if orientation >= 5 { src_w } else { src_h }, target_h);
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
    let src = if orientation >= 5 { (src_h, src_w) } else { (src_w, src_h) };
    Ok((oriented(decoded, orientation), src))
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

/// The EXIF Orientation (1..=8) in a raw Exif blob, which starts at its TIFF
/// header (as `jpeg-decoder`, `png` and the `image` crate hand it back). 1 (as
/// stored) when there is none or it can't be read.
fn exif_orientation(exif: Option<&[u8]>) -> u8 {
    exif.and_then(image::metadata::Orientation::from_exif_chunk).map_or(1, |o| o.to_exif())
}

/// The EXIF Orientation (1..=8; 1 = as stored) that [`decode`] and [`decode_raw`]
/// apply to `bytes`, from its headers alone (for showing a page's upright size
/// without decoding it). JPEG XL reports 1: `jxl-oxide` applies its codestream
/// orientation itself, and its sizes are already upright.
pub fn exif_orientation_of(bytes: &[u8]) -> u8 {
    use std::io::Cursor;
    if bytes.starts_with(&PNG_SIG) {
        png::Decoder::new(Cursor::new(bytes))
            .read_info()
            .map_or(1, |r| exif_orientation(r.info().exif_metadata.as_deref()))
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        let mut d = jpeg_decoder::Decoder::new(Cursor::new(bytes));
        if d.read_info().is_err() {
            return 1;
        }
        exif_orientation(d.exif_data())
    } else if is_jxl(bytes) || bytes.starts_with(b"8BPS") {
        1
    } else {
        image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .ok()
            .filter(|r| r.format().is_some())
            .and_then(|r| r.into_decoder().ok())
            .and_then(|mut d| d.orientation().ok())
            .map_or(1, |o| o.to_exif())
    }
}

/// `d` turned upright per EXIF `orientation`; 1 (and anything out of range)
/// leaves it untouched, so an untagged file's pixels are exactly as decoded.
fn oriented(d: Decoded, orientation: u8) -> Decoded {
    if !(2..=8).contains(&orientation) {
        return d;
    }
    let (w, h, gray, pixels, icc) = d;
    let (w, h, pixels) = apply_orientation(orientation, w, h, if gray { 1 } else { 4 }, &pixels);
    (w, h, gray, pixels, icc)
}

/// Turn a `w`×`h` buffer of `ch` interleaved bytes per pixel upright per an EXIF
/// Orientation value: 2 mirrors, 3 rotates 180°, 4 flips, 5 transposes, 6 rotates
/// 90° clockwise, 7 transverses, 8 rotates 90° counter-clockwise. 5..=8 swap the
/// width and height. Anything else returns a copy as is.
pub fn apply_orientation(orientation: u8, w: u32, h: u32, ch: usize, pixels: &[u8]) -> (u32, u32, Vec<u8>) {
    let (w, h) = (w as usize, h as usize);
    let (ow, oh) = if (5..=8).contains(&orientation) { (h, w) } else { (w, h) };
    // Which source pixel lands at output (x, y).
    let src: fn(usize, usize, usize, usize) -> (usize, usize) = match orientation {
        2 => |x, y, w, _| (w - 1 - x, y),
        3 => |x, y, w, h| (w - 1 - x, h - 1 - y),
        4 => |x, y, _, h| (x, h - 1 - y),
        5 => |x, y, _, _| (y, x),
        6 => |x, y, _, h| (y, h - 1 - x),
        7 => |x, y, w, h| (w - 1 - y, h - 1 - x),
        8 => |x, y, w, _| (w - 1 - y, x),
        _ => return (w as u32, h as u32, pixels.to_vec()),
    };
    let mut out = vec![0u8; pixels.len()];
    for (y, row) in out.chunks_exact_mut(ow * ch).enumerate().take(oh) {
        for (x, px) in row.chunks_exact_mut(ch).enumerate() {
            let (sx, sy) = src(x, y, w, h);
            let i = (sy * w + sx) * ch;
            px.copy_from_slice(&pixels[i..i + ch]);
        }
    }
    (ow as u32, oh as u32, out)
}

/// An animated image format, as [`animation`] finds it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Animation {
    Gif,
    Webp,
    Apng,
    Jxl,
    Avif,
}

impl Animation {
    pub fn name(self) -> &'static str {
        match self {
            Animation::Gif => "GIF",
            Animation::Webp => "WebP",
            Animation::Apng => "APNG",
            Animation::Jxl => "JPEG XL",
            Animation::Avif => "AVIF",
        }
    }
}

/// Whether `bytes` is an animation, and which kind: a GIF with more than one frame,
/// an animated WebP, an APNG of more than one frame, a JPEG XL with an animation
/// header, or an AVIF image sequence (`avis`). The decoders above see only the
/// first frame of one, so a batch tool that rewrites stills should copy these
/// as they are. Reads headers only (a GIF's block structure, no LZW).
pub fn animation(bytes: &[u8]) -> Option<Animation> {
    if bytes.starts_with(b"GIF8") {
        gif_frames_over_one(bytes).then_some(Animation::Gif)
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        image::codecs::webp::WebPDecoder::new(std::io::Cursor::new(bytes))
            .is_ok_and(|d| d.has_animation())
            .then_some(Animation::Webp)
    } else if bytes.starts_with(&PNG_SIG) {
        (apng_frames(bytes) > 1).then_some(Animation::Apng)
    } else if is_jxl(bytes) {
        jxl_is_animated(bytes).then_some(Animation::Jxl)
    } else if bytes.len() >= 16 && &bytes[4..8] == b"ftyp" {
        avif_is_sequence(bytes).then_some(Animation::Avif)
    } else {
        None
    }
}

/// A GIF's sub-block chain from `i` (each block length-prefixed, ended by a 0):
/// the offset just past it, or `None` if the file ends first.
fn skip_gif_sub_blocks(b: &[u8], mut i: usize) -> Option<usize> {
    loop {
        let n = *b.get(i)? as usize;
        i += 1 + n;
        if n == 0 {
            return Some(i);
        }
    }
}

/// Whether a GIF holds a second image (frame), walking its blocks.
fn gif_frames_over_one(b: &[u8]) -> bool {
    let table = |flags: u8| if flags & 0x80 != 0 { 3usize << ((flags & 7) + 1) } else { 0 };
    let Some(&flags) = b.get(10) else { return false };
    let mut i = 13 + table(flags); // header + logical screen descriptor + global color table
    let mut frames = 0;
    loop {
        match b.get(i) {
            Some(0x2C) => {
                frames += 1;
                if frames > 1 {
                    return true;
                }
                let Some(&flags) = b.get(i + 9) else { return false };
                // image descriptor + local color table + LZW minimum code size, then the data
                i += 10 + table(flags) + 1;
            }
            Some(0x21) => i += 2, // extension introducer + label, then its sub-blocks
            _ => return false,    // the trailer, or the file ends
        }
        match skip_gif_sub_blocks(b, i) {
            Some(next) => i = next,
            None => return false,
        }
    }
}

/// An APNG's frame count (its `acTL` chunk, which comes before the image data); 0
/// for a plain PNG.
fn apng_frames(b: &[u8]) -> u32 {
    let mut i = 8;
    while let Some(head) = b.get(i..i + 8) {
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        match &head[4..8] {
            b"acTL" => {
                return b.get(i + 8..i + 12).map_or(0, |n| u32::from_be_bytes([n[0], n[1], n[2], n[3]]));
            }
            b"IDAT" | b"IEND" => return 0,
            _ => i += 12 + len,
        }
    }
    0
}

/// Whether a JPEG XL's image header declares an animation. Feeds the file in
/// pieces until the header parses, so a large still isn't copied for it.
fn jxl_is_animated(b: &[u8]) -> bool {
    use jxl_oxide::{InitializeResult, JxlImage};
    let mut uninit = JxlImage::builder().build_uninit();
    for piece in b.chunks(64 * 1024) {
        if uninit.feed_bytes(piece).is_err() {
            return false;
        }
        match uninit.try_init() {
            Ok(InitializeResult::Initialized(image)) => return image.image_header().metadata.animation.is_some(),
            Ok(InitializeResult::NeedMoreData(u)) => uninit = u,
            Err(_) => return false,
        }
    }
    false
}

/// Whether an ISO-BMFF file's `ftyp` names the AVIF image-sequence brand (`avis`)
/// as its major or a compatible brand.
fn avif_is_sequence(b: &[u8]) -> bool {
    let size = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let ftyp = &b[..size.clamp(16, b.len())];
    // major brand at 8, minor version at 12, compatible brands from 16
    &ftyp[8..12] == b"avis" || ftyp[16..].as_chunks::<4>().0.contains(b"avis")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32, color: png::ColorType, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut enc = png::Encoder::new(&mut bytes, w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(data).unwrap();
        bytes
    }

    /// `decode` keeps alpha (gray+alpha → straight RGBA), unlike the display
    /// contract of `decode_raw` (which drops it); plain gray stays 1 channel.
    #[test]
    fn decode_keeps_gray_alpha_as_straight_rgba() {
        let ga = [10u8, 255, 200, 0, 90, 128, 255, 64];
        let bytes = png(2, 2, png::ColorType::GrayscaleAlpha, &ga);
        let img = decode(&bytes).unwrap();
        assert_eq!((img.w, img.h, img.channels), (2, 2, 4));
        assert_eq!(img.pixels, [10, 10, 10, 255, 200, 200, 200, 0, 90, 90, 90, 128, 255, 255, 255, 64]);
        let raw = decode_raw(&bytes).unwrap();
        assert!(raw.2 && raw.3 == [10, 200, 90, 255], "decode_raw drops alpha (yosh)");

        let gray = decode(&png(2, 1, png::ColorType::Grayscale, &[7, 8])).unwrap();
        assert_eq!((gray.channels, gray.pixels.as_slice()), (1, [7u8, 8].as_slice()));
        let rgb = decode(&png(1, 1, png::ColorType::Rgb, &[1, 2, 3])).unwrap();
        assert_eq!((rgb.channels, rgb.pixels.as_slice()), (4, [1u8, 2, 3, 255].as_slice()));
        assert!(is_opaque(&rgb.pixels));
        assert_eq!(rgba_to_rgb(&rgb.pixels), [1, 2, 3]);
        assert_eq!(rgb_to_luma(&[255, 255, 255, 0, 0, 0]), [255, 0]);
    }
    /// A minimal little-endian Exif blob (TIFF header + one IFD entry) holding
    /// `orientation`.
    fn exif(orientation: u16) -> Vec<u8> {
        let mut b = b"II*\0".to_vec();
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&0x0112u16.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&orientation.to_le_bytes());
        b.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        b
    }

    /// A `w`×`h` RGB gradient, different at every pixel, so every orientation differs.
    fn gradient(w: u32, h: u32) -> Vec<u8> {
        (0..w * h).flat_map(|i| [(i % w * 255 / w) as u8, (i / w * 255 / h) as u8, 128]).collect()
    }

    fn jpeg(w: u32, h: u32, rgb: &[u8], orientation: Option<u16>) -> Vec<u8> {
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
            .encode(rgb, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        if let Some(o) = orientation {
            let mut app1 = vec![0xFF, 0xE1, 0, 0];
            app1.extend_from_slice(b"Exif\0\0");
            app1.extend_from_slice(&exif(o));
            let len = (app1.len() - 2) as u16;
            app1[2..4].copy_from_slice(&len.to_be_bytes());
            out.splice(2..2, app1);
        }
        out
    }

    /// A PNG chunk (length, type, data, CRC).
    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut c = (data.len() as u32).to_be_bytes().to_vec();
        c.extend_from_slice(kind);
        c.extend_from_slice(data);
        let mut crc = crc32fast::Hasher::new();
        crc.update(kind);
        crc.update(data);
        c.extend_from_slice(&crc.finalize().to_be_bytes());
        c
    }

    /// `apply_orientation` agrees with the `image` crate's own EXIF transforms, for
    /// one channel and four.
    #[test]
    fn apply_orientation_matches_the_image_crate() {
        use image::metadata::Orientation;
        let (w, h) = (3u32, 2u32);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| i as u8).collect();
        let gray: Vec<u8> = (0..w * h).map(|i| i as u8 * 10).collect();
        for o in 1..=8u8 {
            let mut want = image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, rgba.clone()).unwrap());
            want.apply_orientation(Orientation::from_exif(o).unwrap());
            let (ow, oh, got) = apply_orientation(o, w, h, 4, &rgba);
            assert_eq!((ow, oh, got.as_slice()), (want.width(), want.height(), want.as_bytes()), "rgba, orientation {o}");

            let mut want = image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, gray.clone()).unwrap());
            want.apply_orientation(Orientation::from_exif(o).unwrap());
            let (ow, oh, got) = apply_orientation(o, w, h, 1, &gray);
            assert_eq!((ow, oh, got.as_slice()), (want.width(), want.height(), want.as_bytes()), "gray, orientation {o}");
        }
    }

    /// A JPEG with an EXIF Orientation decodes upright (full and IDCT-scaled), and
    /// the same JPEG without one decodes exactly as before.
    #[test]
    fn jpeg_exif_orientation_is_applied() {
        let (w, h) = (24u32, 16u32);
        let rgb = gradient(w, h);
        let plain = decode_raw(&jpeg(w, h, &rgb, None)).unwrap();
        assert_eq!((plain.0, plain.1), (w, h));
        assert_eq!(decode_raw(&jpeg(w, h, &rgb, Some(1))).unwrap().3, plain.3, "1 is as stored");
        assert_eq!(exif_orientation_of(&jpeg(w, h, &rgb, None)), 1);
        for o in 2..=8u8 {
            let bytes = jpeg(w, h, &rgb, Some(o as u16));
            let (ow, oh, want) = apply_orientation(o, w, h, 4, &plain.3);
            let got = decode_raw(&bytes).unwrap();
            assert_eq!((got.0, got.1), (ow, oh), "orientation {o}: upright size");
            assert!(got.3 == want, "orientation {o}: upright pixels");
            let img = decode(&bytes).unwrap();
            assert_eq!((img.w, img.h), (ow, oh), "decode() too");
            assert_eq!(exif_orientation_of(&bytes), o);

            let ((sw, sh, _, px, _), src) = decode_jpeg_scaled(&bytes, h.max(w)).unwrap();
            assert_eq!((sw, sh, src), (ow, oh, (ow, oh)), "orientation {o}: scaled path at full size");
            assert!(px == want);
        }
        // The IDCT scale goes by the upright height: 400x200 stored, 200x400 upright,
        // a 100 px target takes the 2/8 scale (a 50x100 buffer), not 4/8.
        let bytes = jpeg(400, 200, &gradient(400, 200), Some(6));
        let ((sw, sh, ..), src) = decode_jpeg_scaled(&bytes, 100).unwrap();
        assert_eq!(((sw, sh), src), ((50, 100), (200, 400)));
    }

    /// PNG (`eXIf`) and WebP (`EXIF` chunk) carry EXIF too.
    #[test]
    fn png_and_webp_exif_orientation_is_applied() {
        let (w, h) = (6u32, 4u32);
        let rgb = gradient(w, h);
        let rgba = rgb_to_rgba(&rgb, w, h);
        let (ow, oh, want) = apply_orientation(6, w, h, 4, &rgba);

        let mut p = png(w, h, png::ColorType::Rgb, &rgb);
        p.splice(33..33, chunk(b"eXIf", &exif(6))); // after the signature and IHDR
        let got = decode_raw(&p).unwrap();
        assert_eq!((got.0, got.1, &got.3), (ow, oh, &want), "png");
        assert_eq!(exif_orientation_of(&p), 6);
        assert_eq!(exif_orientation_of(&png(w, h, png::ColorType::Rgb, &rgb)), 1);

        let mut lossless = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut lossless)
            .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        // RIFF/WEBP + VP8X (EXIF flag) + the encoder's VP8L chunk + an EXIF chunk.
        let mut vp8x = vec![0x08, 0, 0, 0];
        vp8x.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
        vp8x.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
        let exif6 = exif(6);
        let mut body = b"WEBP".to_vec();
        for (kind, data) in [(b"VP8X", vp8x.as_slice()), (b"VP8L", &lossless[20..]), (b"EXIF", exif6.as_slice())] {
            body.extend_from_slice(kind);
            body.extend_from_slice(&(data.len() as u32).to_le_bytes());
            body.extend_from_slice(data);
            if data.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&(body.len() as u32).to_le_bytes());
        webp.extend_from_slice(&body);
        let got = decode_raw(&webp).unwrap();
        assert_eq!((got.0, got.1, &got.3), (ow, oh, &want), "webp");
        assert_eq!(exif_orientation_of(&webp), 6);
        assert_eq!(exif_orientation_of(&lossless), 1);
    }

    /// 8x8, two frames (lossless red, blue), from ImageMagick.
    const ANIMATED_WEBP: [u8; 140] = [
        0x52, 0x49, 0x46, 0x46, 0x84, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58, 0x0a, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x07, 0x00, 0x00, 0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00,
        0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x07, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c,
        0x0f, 0x00, 0x00, 0x00, 0x2f, 0x07, 0xc0, 0x01, 0x00, 0x07, 0x10, 0xfd, 0x8f, 0xfe, 0x07, 0x22, 0xa2, 0xff,
        0x01, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00,
        0x00, 0x07, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x0f, 0x00, 0x00, 0x00, 0x2f, 0x07,
        0xc0, 0x01, 0x00, 0x07, 0x10, 0xd1, 0xff, 0xfe, 0x07, 0x22, 0xa2, 0xff, 0x01, 0x00,
    ];

    /// 8x8 JPEG XL codestreams from ImageMagick (libjxl): two frames, and a still.
    const ANIMATED_JXL: [u8; 115] = [
        0xff, 0x0a, 0x41, 0x04, 0x01, 0xc8, 0x09, 0x00, 0x13, 0x08, 0x28, 0x00, 0xc0, 0x00, 0xb5, 0x9f, 0x20, 0x00,
        0x00, 0x15, 0x2a, 0xa3, 0x8c, 0x1b, 0xbc, 0x9c, 0xeb, 0xf9, 0xf2, 0x43, 0x87, 0xc5, 0xb4, 0x8d, 0xeb, 0x0c,
        0x6d, 0xb5, 0x6d, 0x61, 0x09, 0x63, 0xb3, 0xbd, 0x30, 0x48, 0x48, 0x38, 0x83, 0x8d, 0x8b, 0x04, 0xdf, 0x9c,
        0xa5, 0x91, 0x69, 0x06, 0x24, 0xa2, 0x92, 0x04, 0x00, 0x13, 0x28, 0x0a, 0x00, 0xb8, 0x00, 0xb5, 0x9f, 0x20,
        0x00, 0x00, 0x15, 0x2a, 0xa3, 0x8c, 0x1b, 0xbc, 0x9c, 0xeb, 0xf9, 0xf2, 0x43, 0x87, 0xc5, 0xb4, 0x8d, 0xeb,
        0x0c, 0x6d, 0xb5, 0x6d, 0x61, 0x09, 0x63, 0xb3, 0xbd, 0x30, 0x48, 0x48, 0x38, 0x6a, 0xd0, 0x9a, 0x4d, 0xf4,
        0x04, 0x61, 0x18, 0x27, 0x95, 0x24, 0x00,
    ];
    const STILL_JXL: [u8; 59] = [
        0xff, 0x0a, 0x41, 0x06, 0x00, 0x13, 0x88, 0x02, 0x00, 0xc0, 0x00, 0xb5, 0x9f, 0x20, 0x00, 0x00, 0x15, 0x2a,
        0xa3, 0x8c, 0x1b, 0xbc, 0x9c, 0xeb, 0xf9, 0xf2, 0x43, 0x87, 0xc5, 0xb4, 0x8d, 0xeb, 0x0c, 0x6d, 0xb5, 0x6d,
        0x61, 0x09, 0x63, 0xb3, 0xbd, 0x30, 0x48, 0x48, 0x38, 0x83, 0x8d, 0x8b, 0x04, 0xdf, 0x9c, 0xa5, 0x91, 0x69,
        0x06, 0x24, 0xa2, 0x92, 0x04,
    ];

    #[test]
    fn animations_are_told_from_stills() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};
        let gif = |colors: &[[u8; 4]]| {
            let mut buf = Vec::new();
            {
                let mut enc = GifEncoder::new(&mut buf);
                enc.set_repeat(Repeat::Infinite).unwrap();
                for &c in colors {
                    let f = Frame::from_parts(RgbaImage::from_pixel(4, 4, Rgba(c)), 0, 0, Delay::from_numer_denom_ms(100, 1));
                    enc.encode_frame(f).unwrap();
                }
            }
            buf
        };
        let two = gif(&[[255, 0, 0, 255], [0, 0, 255, 255]]);
        assert_eq!(animation(&two), Some(Animation::Gif));
        assert_eq!(animation(&gif(&[[255, 0, 0, 255]])), None, "one frame is a still");
        assert_eq!(animation(&two[..20]), None, "cut short before a frame");

        let apng = |frames: u32| {
            let mut bytes = Vec::new();
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_animated(frames, 0).unwrap();
            let mut w = enc.write_header().unwrap();
            for i in 0..frames {
                w.write_image_data(&[i as u8; 4]).unwrap();
            }
            w.finish().unwrap();
            bytes
        };
        assert_eq!(animation(&apng(3)), Some(Animation::Apng));
        assert_eq!(animation(&apng(1)), None, "a one-frame APNG is a still");
        assert_eq!(animation(&png(2, 1, png::ColorType::Grayscale, &[7, 8])), None);

        assert_eq!(animation(&ANIMATED_WEBP), Some(Animation::Webp));
        let mut still_webp = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut still_webp)
            .encode(&[1, 2, 3], 1, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        assert_eq!(animation(&still_webp), None);

        assert_eq!(animation(&ANIMATED_JXL), Some(Animation::Jxl));
        assert_eq!(animation(&STILL_JXL), None);

        let ftyp = |major: &[u8; 4], compatible: &[&[u8; 4]]| {
            let mut b = ((16 + 4 * compatible.len()) as u32).to_be_bytes().to_vec();
            b.extend_from_slice(b"ftyp");
            b.extend_from_slice(major);
            b.extend_from_slice(&[0; 4]);
            compatible.iter().for_each(|c| b.extend_from_slice(*c));
            b.extend_from_slice(&[0, 0, 0, 8, b'm', b'e', b't', b'a']);
            b
        };
        assert_eq!(animation(&ftyp(b"avis", &[b"avif", b"mif1"])), Some(Animation::Avif));
        assert_eq!(animation(&ftyp(b"avif", &[b"avif", b"avis"])), Some(Animation::Avif));
        assert_eq!(animation(&ftyp(b"avif", &[b"mif1", b"miaf"])), None);

        assert_eq!(animation(&jpeg(2, 2, &gradient(2, 2), None)), None);
        assert_eq!(animation(b""), None);
    }
}

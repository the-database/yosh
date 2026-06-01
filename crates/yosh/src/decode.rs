//! Decode + downscale a page's encoded bytes to a display-resolution buffer.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, else → `image`
//! crate fallback (WebP/GIF/BMP/…). Normalizes to single-channel R8 (gray) or
//! RGBA8 (color), then downscales (single-channel-aware) with a reused `Resizer`.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

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

/// Decode page bytes (any supported format) and downscale to `target_h` on CPU.
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let (w, h, gray, full) = decode_raw(bytes)?;

    let pt = if gray { PixelType::U8 } else { PixelType::U8x4 };
    let tw = (((w as f64) * (target_h as f64) / (h as f64)).round() as u32).max(1);

    let src = ImageRef::new(w, h, &full, pt).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, pt);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| format!("resize: {e}"))?;

    Ok(DecodedImage {
        w: tw,
        h: target_h,
        gray,
        pixels: dst.into_vec(),
    })
}

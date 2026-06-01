//! Decode + downscale a page's encoded bytes to a display-resolution buffer.
//!
//! M1.2: PNG only (the folder asset). Normalizes to single-channel R8 (gray) or
//! RGBA8 (color). M1.6 adds JPEG (`jpeg-decoder`) + an `image`-crate fallback via
//! `infer` routing. The decode pool (M1.3) reuses a per-thread `Resizer`/scratch.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

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

/// Decode PNG bytes and downscale to `target_h` (preserving aspect).
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .map_err(|e| format!("png read_info: {e}"))?;
    let size = reader.output_buffer_size().ok_or("png: no buffer size")?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png next_frame: {e}"))?;
    let (w, h) = (info.width, info.height);
    let px = (w as usize) * (h as usize);
    let ch = info.buffer_size() / px;

    let (gray, full): (bool, Vec<u8>) = match ch {
        1 => (true, buf),
        2 => (true, ga_to_gray(&buf)),
        3 => (false, rgb_to_rgba(&buf, w, h)),
        4 => (false, buf),
        other => return Err(format!("unsupported channel count {other}")),
    };

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

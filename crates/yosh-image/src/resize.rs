//! High-quality resampling to any size (down or up), on 8-bit buffers.
//!
//! - [`gray_linear`]: grayscale in **true linear light** — sRGB → 16-bit linear,
//!   Catmull-Rom, then re-encoded through the Dot Gain 20% curve (`tone`). This
//!   is what suppresses halftone-screentone moiré; resampling in gamma space does
//!   not.
//! - [`lanczos`]: color (or anything else) in gamma space, Lanczos3.
//! - [`bilinear`]: a fast gamma-space bilinear (yosh's transient seek tier).
//!
//! RGBA with alpha should be premultiplied before resizing (and un-premultiplied
//! after, if straight alpha is wanted) to avoid fringing.

pub use fast_image_resize::Resizer;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions};

use crate::tone;

fn pixel_type(channels: u8) -> Result<PixelType, String> {
    match channels {
        1 => Ok(PixelType::U8),
        2 => Ok(PixelType::U8x2),
        3 => Ok(PixelType::U8x3),
        4 => Ok(PixelType::U8x4),
        c => Err(format!("resize: unsupported channel count {c}")),
    }
}

fn convolve(
    src: &[u8],
    channels: u8,
    (w, h): (u32, u32),
    (tw, th): (u32, u32),
    filter: FilterType,
    resizer: &mut Resizer,
) -> Result<Vec<u8>, String> {
    let pt = pixel_type(channels)?;
    let src = ImageRef::new(w, h, src, pt).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, th, pt);
    resizer
        .resize(&src, &mut dst, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter)))
        .map_err(|e| format!("resize: {e}"))?;
    Ok(dst.into_vec())
}

/// Gray `w`x`h` → `tw`x`th` in linear light (Catmull-Rom), re-encoded through the
/// Dot Gain 20% curve. 16-bit intermediate avoids shadow banding.
pub fn gray_linear(
    gray: &[u8],
    (w, h): (u32, u32),
    (tw, th): (u32, u32),
    resizer: &mut Resizer,
) -> Result<Vec<u8>, String> {
    // sRGB device → 16-bit linear luminance (native-endian U16 byte buffer).
    let mut lin = Vec::with_capacity(gray.len() * 2);
    for &v in gray {
        lin.extend_from_slice(&tone::SRGB_TO_LINEAR[v as usize].to_ne_bytes());
    }
    let src = ImageRef::new(w, h, &lin, PixelType::U16).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, th, PixelType::U16);
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
    Ok(bytes.as_chunks::<2>().0.iter().map(|c| enc[u16::from_ne_bytes(*c) as usize]).collect())
}

/// `channels`-interleaved 8-bit `w`x`h` → `tw`x`th`, Lanczos3 in gamma space.
pub fn lanczos(
    src: &[u8],
    channels: u8,
    wh: (u32, u32),
    twh: (u32, u32),
    resizer: &mut Resizer,
) -> Result<Vec<u8>, String> {
    convolve(src, channels, wh, twh, FilterType::Lanczos3, resizer)
}

/// `channels`-interleaved 8-bit `w`x`h` → `tw`x`th`, bilinear in gamma space.
pub fn bilinear(
    src: &[u8],
    channels: u8,
    wh: (u32, u32),
    twh: (u32, u32),
    resizer: &mut Resizer,
) -> Result<Vec<u8>, String> {
    convolve(src, channels, wh, twh, FilterType::Bilinear, resizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Any size, up or down, every channel layout; a flat image stays flat.
    #[test]
    fn resizes_to_any_size_up_or_down() {
        let mut r = Resizer::new();
        let gray = vec![128u8; 40 * 30];
        for (tw, th) in [(17, 13), (40, 30), (97, 71)] {
            let out = gray_linear(&gray, (40, 30), (tw, th), &mut r).unwrap();
            assert_eq!(out.len(), (tw * th) as usize);
            assert!(out.windows(2).all(|w| w[0] == w[1]), "flat stays flat");
            for ch in [1u8, 3, 4] {
                let src = vec![200u8; 40 * 30 * ch as usize];
                let l = lanczos(&src, ch, (40, 30), (tw, th), &mut r).unwrap();
                assert_eq!(l.len(), (tw * th) as usize * ch as usize);
                assert!(l.iter().all(|&v| v == 200));
                assert_eq!(bilinear(&src, ch, (40, 30), (tw, th), &mut r).unwrap().len(), l.len());
            }
        }
        assert!(lanczos(&[0; 10], 5, (1, 2), (2, 2), &mut r).is_err());
    }
}

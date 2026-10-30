//! Output encoding: WebP (lossy or lossless), PNG, JPEG and AVIF, from 8-bit interleaved pixels
//! with 1 (gray), 2 (gray + alpha), 3 (RGB) or 4 (RGBA) channels.

use crate::settings::Format;

/// A finished image, ready to encode.
pub struct Out {
    pub w: u32,
    pub h: u32,
    pub channels: u8,
    pub pixels: Vec<u8>,
}

pub fn encode(img: &Out, format: Format, quality: u8, lossless: bool) -> Result<Vec<u8>, String> {
    match format {
        Format::Png => png(img),
        Format::Jpeg => jpeg(img, quality),
        Format::Webp => webp(img, quality, lossless),
        Format::Avif => avif(img, quality),
    }
}

/// Expand to 3 (RGB) or 4 (RGBA) channels, for encoders without a gray mode.
fn to_color(img: &Out) -> (u8, std::borrow::Cow<'_, [u8]>) {
    match img.channels {
        1 => (3, img.pixels.iter().flat_map(|&v| [v, v, v]).collect::<Vec<_>>().into()),
        2 => (4, img.pixels.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect::<Vec<_>>().into()),
        c => (c, img.pixels.as_slice().into()),
    }
}

fn png(img: &Out) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut enc = png::Encoder::new(&mut bytes, img.w, img.h);
    enc.set_color(match img.channels {
        1 => png::ColorType::Grayscale,
        2 => png::ColorType::GrayscaleAlpha,
        3 => png::ColorType::Rgb,
        _ => png::ColorType::Rgba,
    });
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| format!("png: {e}"))?;
    writer.write_image_data(&img.pixels).map_err(|e| format!("png: {e}"))?;
    writer.finish().map_err(|e| format!("png: {e}"))?;
    Ok(bytes)
}

/// JPEG has no alpha: a transparent image is flattened onto black, as 1.x did.
fn jpeg(img: &Out, quality: u8) -> Result<Vec<u8>, String> {
    if img.w > u16::MAX as u32 || img.h > u16::MAX as u32 {
        return Err(format!("jpeg: {}x{} is larger than JPEG allows (65535)", img.w, img.h));
    }
    let flatten = |px: &[u8], ch: usize| -> Vec<u8> {
        px.chunks_exact(ch)
            .flat_map(|p| {
                let a = p[ch - 1] as u32;
                p[..ch - 1].iter().map(move |&v| ((v as u32 * a + 127) / 255) as u8).collect::<Vec<_>>()
            })
            .collect()
    };
    let (data, color): (std::borrow::Cow<'_, [u8]>, _) = match img.channels {
        1 => (img.pixels.as_slice().into(), jpeg_encoder::ColorType::Luma),
        2 => (flatten(&img.pixels, 2).into(), jpeg_encoder::ColorType::Luma),
        3 => (img.pixels.as_slice().into(), jpeg_encoder::ColorType::Rgb),
        _ => (flatten(&img.pixels, 4).into(), jpeg_encoder::ColorType::Rgb),
    };
    let mut bytes = Vec::new();
    jpeg_encoder::Encoder::new(&mut bytes, quality.max(1))
        .encode(&data, img.w as u16, img.h as u16, color)
        .map_err(|e| format!("jpeg: {e}"))?;
    Ok(bytes)
}

fn webp(img: &Out, quality: u8, lossless: bool) -> Result<Vec<u8>, String> {
    let (ch, data) = to_color(img);
    let enc = if ch == 4 {
        webp::Encoder::from_rgba(&data, img.w, img.h)
    } else {
        webp::Encoder::from_rgb(&data, img.w, img.h)
    };
    let mem = if lossless {
        enc.encode_lossless()
    } else {
        enc.encode(quality as f32)
    };
    Ok(mem.to_vec())
}

fn avif(img: &Out, quality: u8) -> Result<Vec<u8>, String> {
    use ravif::{Img, RGB8, RGBA8};
    let (ch, data) = to_color(img);
    let enc = ravif::Encoder::new().with_quality(quality as f32).with_speed(6);
    let (w, h) = (img.w as usize, img.h as usize);
    let res = if ch == 4 {
        let px: Vec<RGBA8> = data.as_chunks::<4>().0.iter().map(|p| RGBA8::new(p[0], p[1], p[2], p[3])).collect();
        enc.encode_rgba(Img::new(&px[..], w, h))
    } else {
        let px: Vec<RGB8> = data.as_chunks::<3>().0.iter().map(|p| RGB8::new(p[0], p[1], p[2])).collect();
        enc.encode_rgb(Img::new(&px[..], w, h))
    };
    res.map(|r| r.avif_file).map_err(|e| format!("avif: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(channels: u8) -> Out {
        let (w, h) = (16u32, 8u32);
        let pixels = (0..w * h * channels as u32).map(|i| (i * 7 % 256) as u8).collect();
        Out { w, h, channels, pixels }
    }

    #[test]
    fn every_format_and_layout_encodes() {
        for ch in 1..=4 {
            let i = img(ch);
            let png_bytes = encode(&i, Format::Png, 80, false).unwrap();
            assert!(png_bytes.starts_with(b"\x89PNG"));
            let decoded = yosh_image::decode::decode(&png_bytes).unwrap();
            assert_eq!((decoded.w, decoded.h), (16, 8));
            assert!(encode(&i, Format::Jpeg, 80, false).unwrap().starts_with(&[0xFF, 0xD8]));
            assert!(&encode(&i, Format::Webp, 80, false).unwrap()[8..12] == b"WEBP");
            assert!(&encode(&i, Format::Webp, 80, true).unwrap()[8..12] == b"WEBP");
            assert!(&encode(&i, Format::Avif, 80, false).unwrap()[4..8] == b"ftyp");
        }
    }

    #[test]
    fn png_round_trips_gray_exactly() {
        let i = img(1);
        let d = yosh_image::decode::decode(&encode(&i, Format::Png, 80, false).unwrap()).unwrap();
        assert_eq!(d.channels, 1);
        assert_eq!(d.pixels, i.pixels);
    }
}

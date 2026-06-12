//! Image metadata probing for the info overlay: a header-only `probe` (dimensions
//! and a format string, no full decode) plus a human-readable byte-size formatter.
//! Shared by the desktop and Android shells so both info overlays read identically.

/// Human-readable byte size for the info overlay.
pub fn human_size(n: u64) -> String {
    const KB: u64 = 1 << 10;
    const MB: u64 = 1 << 20;
    if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

/// Walk an ISO-BMFF (AVIF/HEIF) box tree to the first `ispe` (image spatial
/// extents) box and return its (width, height) — pure parsing, no decode, so it
/// works regardless of the `avif` feature. `meta` is a FullBox (4-byte
/// version/flags before its children); `iprp`/`ipco` are plain containers; the
/// `ispe` payload is version/flags(4) + width(4) + height(4), all big-endian.
fn iso_box_dims(b: &[u8]) -> Option<(u32, u32)> {
    // Find a child box by 4-byte type, returning its payload (after the header).
    fn find<'a>(mut b: &'a [u8], want: &[u8; 4]) -> Option<&'a [u8]> {
        while b.len() >= 8 {
            let size = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
            let (header, end) = match size {
                1 => {
                    // 64-bit largesize follows the type.
                    let s = u64::from_be_bytes(b.get(8..16)?.try_into().ok()?) as usize;
                    (16, s)
                }
                0 => (8, b.len()), // extends to end
                s => (8, s),
            };
            if end < header || end > b.len() {
                return None;
            }
            if &b[4..8] == want {
                return Some(&b[header..end]);
            }
            b = &b[end..];
        }
        None
    }
    let meta = find(b, b"meta")?.get(4..)?; // skip meta's FullBox version/flags
    let ispe = find(find(find(meta, b"iprp")?, b"ipco")?, b"ispe")?;
    let w = u32::from_be_bytes(ispe.get(4..8)?.try_into().ok()?);
    let h = u32::from_be_bytes(ispe.get(8..12)?.try_into().ok()?);
    Some((w, h))
}

/// Probe an encoded image's header for `(width, height, "FORMAT · detail")`
/// without a full decode. Returns `(0, 0, ...)` if dimensions can't be read.
pub fn probe(b: &[u8]) -> (u32, u32, String) {
    let be16 = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]) as u32;
    let le16 = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as u32;
    // PNG
    if b.len() >= 26 && b[..4] == [0x89, 0x50, 0x4E, 0x47] {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        let color = match b[25] {
            0 => "grayscale",
            2 => "RGB",
            3 => "indexed",
            4 => "grayscale+alpha",
            6 => "RGBA",
            _ => "?",
        };
        return (w, h, format!("PNG · {}-bit {}", b[24], color));
    }
    // JPEG: scan for a Start-Of-Frame marker.
    if b.len() >= 4 && b[0] == 0xFF && b[1] == 0xD8 {
        let mut i = 2;
        while i + 9 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let m = b[i + 1];
            let is_sof = (0xC0..=0xCF).contains(&m) && m != 0xC4 && m != 0xC8 && m != 0xCC;
            if is_sof {
                let kind = match b[i + 9] {
                    1 => "grayscale",
                    3 => "YCbCr",
                    4 => "CMYK",
                    _ => "?",
                };
                return (be16(i + 7), be16(i + 5), format!("JPEG · {}-bit {}", b[i + 4], kind));
            }
            if i + 3 >= b.len() {
                break;
            }
            i += 2 + u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        }
        return (0, 0, "JPEG".to_string());
    }
    // GIF
    if b.len() >= 10 && &b[0..3] == b"GIF" {
        return (le16(6), le16(8), "GIF".to_string());
    }
    // PSD / PSB (Photoshop): header is big-endian — rows@14, cols@18 (u32),
    // depth@22 and color mode@24 (u16).
    if b.len() >= 26 && &b[0..4] == b"8BPS" {
        let be32 = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let h = be32(14);
        let w = be32(18);
        let mode = match be16(24) {
            0 => "bitmap",
            1 => "grayscale",
            2 => "indexed",
            3 => "RGB",
            4 => "CMYK",
            7 => "multichannel",
            8 => "duotone",
            9 => "Lab",
            _ => "?",
        };
        return (w, h, format!("PSD · {}-bit {}", be16(22), mode));
    }
    // ICO: report the largest entry's size + how many layers it holds.
    if b.len() >= 6 && b[0..4] == [0x00, 0x00, 0x01, 0x00] {
        let count = le16(4) as usize;
        let dim = |v: u8| if v == 0 { 256 } else { v as u32 };
        let (mut mw, mut mh) = (0u32, 0u32);
        for i in 0..count {
            let off = 6 + i * 16;
            if off + 1 < b.len() {
                mw = mw.max(dim(b[off]));
                mh = mh.max(dim(b[off + 1]));
            }
        }
        return (mw, mh, format!("ICO · {count} layer{}", if count == 1 { "" } else { "s" }));
    }
    // BMP
    if b.len() >= 26 && &b[0..2] == b"BM" {
        let w = i32::from_le_bytes([b[18], b[19], b[20], b[21]]).unsigned_abs();
        let h = i32::from_le_bytes([b[22], b[23], b[24], b[25]]).unsigned_abs();
        return (w, h, "BMP".to_string());
    }
    // WebP (RIFF container)
    if b.len() >= 30 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        match &b[12..16] {
            b"VP8X" => {
                let w = 1 + (b[24] as u32 | (b[25] as u32) << 8 | (b[26] as u32) << 16);
                let h = 1 + (b[27] as u32 | (b[28] as u32) << 8 | (b[29] as u32) << 16);
                return (w, h, "WebP".to_string());
            }
            b"VP8 " => {
                return (le16(26) & 0x3FFF, le16(28) & 0x3FFF, "WebP".to_string());
            }
            b"VP8L" => {
                // After the 0x2F signature byte: 14-bit (width-1) then 14-bit
                // (height-1), LSB-first, packed across b[21..25].
                if b.len() >= 25 {
                    let bits = b[21] as u32
                        | (b[22] as u32) << 8
                        | (b[23] as u32) << 16
                        | (b[24] as u32) << 24;
                    return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, "WebP".to_string());
                }
                return (0, 0, "WebP".to_string());
            }
            _ => return (0, 0, "WebP".to_string()),
        }
    }
    // JPEG XL: bare codestream (FF 0A) or ISOBMFF box (".../JXL ..."). Parse just
    // the header via jxl-oxide for exact dimensions + color type (no pixel decode).
    if (b.len() >= 2 && b[0] == 0xFF && b[1] == 0x0A) || (b.len() >= 12 && &b[4..8] == b"JXL ") {
        if let Ok(img) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(b)) {
            let color = match img.pixel_format() {
                jxl_oxide::PixelFormat::Gray => "grayscale",
                jxl_oxide::PixelFormat::Graya => "grayscale+alpha",
                jxl_oxide::PixelFormat::Rgb => "RGB",
                jxl_oxide::PixelFormat::Rgba => "RGBA",
                jxl_oxide::PixelFormat::Cmyk => "CMYK",
                jxl_oxide::PixelFormat::Cmyka => "CMYK+alpha",
            };
            return (img.width(), img.height(), format!("JPEG XL · {color}"));
        }
        return (0, 0, "JPEG XL".to_string());
    }
    // AVIF / HEIF (ISO-BMFF): walk the box tree to the `ispe` for dimensions.
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        let (w, h) = iso_box_dims(b).unwrap_or((0, 0));
        let label = if matches!(&b[8..12], b"avif" | b"avis") { "AVIF" } else { "HEIF" };
        return (w, h, label.to_string());
    }
    // Generic fallback: let the `image` crate identify the format and read just the
    // dimensions (no full decode). Covers TIFF/TGA/DDS/EXR/HDR/QOI/PNM and anything
    // else the crate guesses by content. TGA has no magic bytes, so fall back to it
    // when content-guessing finds nothing (mirrors decode_other).
    if let Ok(guessed) = image::ImageReader::new(std::io::Cursor::new(b)).with_guessed_format() {
        let reader = if guessed.format().is_some() {
            guessed
        } else {
            image::ImageReader::with_format(std::io::Cursor::new(b), image::ImageFormat::Tga)
        };
        let label = match reader.format() {
            Some(image::ImageFormat::Tiff) => "TIFF".to_string(),
            Some(image::ImageFormat::Tga) => "TGA".to_string(),
            Some(image::ImageFormat::Dds) => "DDS".to_string(),
            Some(image::ImageFormat::OpenExr) => "OpenEXR".to_string(),
            Some(image::ImageFormat::Hdr) => "Radiance HDR".to_string(),
            Some(image::ImageFormat::Qoi) => "QOI".to_string(),
            Some(image::ImageFormat::Pnm) => "PNM".to_string(),
            Some(f) => format!("{f:?}"),
            None => "image".to_string(),
        };
        if let Ok((w, h)) = reader.into_dimensions() {
            return (w, h, label);
        }
        return (0, 0, label);
    }
    (0, 0, "image".to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_reads_image_crate_dimensions() {
        use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
        // TIFF/QOI have no magic-byte branch in probe(); the generic fallback must
        // still report their resolution (the "probe data for resolution" rule).
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(7, 4, Rgba([1, 2, 3, 255])));
        for fmt in [ImageFormat::Tiff, ImageFormat::Qoi, ImageFormat::Tga] {
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, fmt).unwrap();
            let (w, h, label) = super::probe(&buf.into_inner());
            assert_eq!((w, h), (7, 4), "probe dims for {fmt:?}");
            assert!(!label.is_empty() && label != "image", "probe label for {fmt:?}: {label}");
        }
    }
}

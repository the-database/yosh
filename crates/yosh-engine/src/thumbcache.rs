//! On-disk cover-thumbnail cache, shared by the desktop and Android shells.
//!
//! A library's covers are otherwise re-decoded from their (often multi-MB) source
//! pages on every open. This caches each decoded thumbnail as a small PNG keyed by
//! the volume's path + filesystem mtime/size + target height, so a repeat open reads
//! a tiny image instead of decoding a huge one. All cache I/O is best-effort: any
//! failure falls back to a live decode, so the cache can never break loading.

use std::hash::{Hash as _, Hasher as _};
use std::path::Path;

use fast_image_resize::Resizer;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

use crate::decode::{decode_and_downscale, to_rgba_image, DecodedImage, ResizePath};

/// Cache filename for a volume's thumbnail, derived from its path + mtime + size +
/// target height. `None` if the volume's metadata can't be read (then we skip the
/// cache and decode live).
fn cache_name(vol_path: &Path, target_h: u32) -> Option<String> {
    let m = std::fs::metadata(vol_path).ok()?;
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    vol_path.to_string_lossy().hash(&mut h);
    mtime.hash(&mut h);
    m.len().hash(&mut h);
    target_h.hash(&mut h);
    Some(format!("{:016x}.png", h.finish()))
}

/// Build an RGBA `DecodedImage` from raw RGBA pixels (a cache hit is always RGBA).
fn rgba_image(w: u32, h: u32, pixels: Vec<u8>) -> DecodedImage {
    DecodedImage {
        w,
        h,
        src_w: w,
        src_h: h,
        gray: false,
        path: ResizePath::None,
        pixels,
    }
}

/// Read a cached thumbnail as RGBA, or `None` on any miss/error.
fn load(cache_file: &Path) -> Option<DecodedImage> {
    let bytes = std::fs::read(cache_file).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(rgba_image(w, h, img.into_raw()))
}

/// Write a decoded thumbnail to the cache as PNG (gray stays single-channel; color
/// is RGBA). Best-effort; a temp file + rename keeps a half-written file from being
/// read as valid.
fn store(cache_file: &Path, img: &DecodedImage) {
    let Some(dir) = cache_file.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let (color, expected) = if img.gray {
        (ExtendedColorType::L8, (img.w * img.h) as usize)
    } else {
        (ExtendedColorType::Rgba8, (img.w * img.h * 4) as usize)
    };
    if img.pixels.len() != expected {
        return; // guard against a malformed buffer
    }
    let mut buf = Vec::new();
    if PngEncoder::new(&mut buf)
        .write_image(&img.pixels, img.w, img.h, color)
        .is_err()
    {
        return;
    }
    // Unique-ish temp name so concurrent decode workers don't clobber each other.
    let tmp = cache_file.with_extension(format!("tmp{:x}", buf.len()));
    if std::fs::write(&tmp, &buf).is_ok() {
        let _ = std::fs::rename(&tmp, cache_file);
    }
}

/// Return a volume's cover thumbnail as an RGBA `DecodedImage`, reading from the
/// disk cache when possible. On a miss, `read_cover` supplies the (slow) source
/// bytes, which are decoded + downscaled to `target_h`, returned, and written to the
/// cache for next time. `cache_dir == None` (or unreadable metadata) just decodes
/// live without touching disk.
pub fn load_or_decode<F: FnOnce() -> Option<Vec<u8>>>(
    cache_dir: Option<&Path>,
    vol_path: &Path,
    target_h: u32,
    resizer: &mut Resizer,
    read_cover: F,
) -> Option<DecodedImage> {
    let cache_file = cache_dir
        .zip(cache_name(vol_path, target_h))
        .map(|(dir, name)| dir.join(name));

    if let Some(cf) = &cache_file
        && let Some(img) = load(cf)
    {
        return Some(img);
    }

    let bytes = read_cover()?;
    let decoded = decode_and_downscale(&bytes, target_h, resizer).ok()?;
    if let Some(cf) = &cache_file {
        store(cf, &decoded);
    }
    Some(to_rgba_image(decoded))
}

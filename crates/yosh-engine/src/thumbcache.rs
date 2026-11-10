//! On-disk cover-thumbnail cache, shared by the desktop and Android shells.
//!
//! A library's covers are otherwise re-decoded from their (often multi-MB) source
//! pages on every open. This caches each decoded thumbnail as a small PNG keyed by
//! the volume's path + filesystem mtime/size + target height, so a repeat open reads
//! a tiny image instead of decoding a huge one. All cache I/O is best-effort: any
//! failure falls back to a live decode, so the cache can never break loading.
//!
//! The names carry a format version ([`VERSION`]): when what a thumbnail looks like
//! changes for the same file (EXIF orientation turned covers upright), the version
//! goes up, and the first use of a cache folder in a process deletes the thumbnails
//! of every other version ([`prune_once`]), so nothing stale is shown or left behind.

use std::collections::HashSet;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fast_image_resize::Resizer;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

use crate::decode::{decode_and_downscale, to_rgba_image, DecodedImage, ResizePath};

/// The thumbnail format version, in every cache file name (`v2-<hash>.png`). Bump it
/// when a thumbnail of an unchanged file would now come out different; the old
/// version's files are then deleted ([`prune_once`]). Version 1 is the unprefixed
/// `<hash>.png` of yosh before EXIF orientation, whose covers of rotated JPEGs lie
/// sideways.
const VERSION: u32 = 2;

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
    Some(format!("v{VERSION}-{:016x}.png", h.finish()))
}

/// Whether `name` is a file this cache wrote under another [`VERSION`]: a thumbnail
/// or a temp file a store left behind (`<hash>.png`, `<hash>.tmp<n>` for version 1,
/// `v<n>-<hash>.…` after it). Anything else in the folder is never touched.
fn stale_name(name: &str) -> bool {
    let (version, rest) = match name.strip_prefix('v').and_then(|r| r.split_once('-')) {
        Some((v, rest)) => match v.parse::<u32>() {
            Ok(v) => (v, rest),
            Err(_) => return false,
        },
        None => (1, name),
    };
    let Some((hash, ext)) = rest.split_once('.') else {
        return false;
    };
    let hex = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    let ours = hash.len() == 16 && hex(hash) && (ext == "png" || ext.strip_prefix("tmp").is_some_and(hex));
    ours && version != VERSION
}

/// Delete the thumbnails of other versions from `dir`, once per folder per process
/// (the first cover load there; it runs on the cover thread, never the UI's).
/// Best-effort: a file that can't be deleted now goes on the next start.
fn prune_once(dir: &Path) {
    static DONE: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);
    if !DONE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(dir.to_path_buf()) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        if e.file_type().is_ok_and(|t| t.is_file()) && e.file_name().to_str().is_some_and(stale_name) {
            let _ = std::fs::remove_file(e.path());
        }
    }
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
pub fn load_or_decode<F: FnOnce() -> Option<std::sync::Arc<Vec<u8>>>>(
    cache_dir: Option<&Path>,
    vol_path: &Path,
    target_h: u32,
    resizer: &mut Resizer,
    read_cover: F,
) -> Option<DecodedImage> {
    if let Some(dir) = cache_dir {
        prune_once(dir);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_other_versions_of_our_own_files_are_stale() {
        // Version 1 (before EXIF orientation): bare hashes, and their temp files.
        assert!(stale_name("0123456789abcdef.png"));
        assert!(stale_name("0123456789abcdef.tmp1f3a"));
        assert!(stale_name("v3-0123456789abcdef.png"));
        assert!(!stale_name(&format!("v{VERSION}-0123456789abcdef.png")));
        assert!(!stale_name(&format!("v{VERSION}-0123456789abcdef.tmp99")));
        // Not ours: never deleted.
        for name in [
            "cover.png",
            "0123456789ABCDEF.png",
            "0123456789abcde.png",
            "0123456789abcdef0.png",
            "0123456789abcdef.jpg",
            "0123456789abcdef.tmp",
            "0123456789abcdef.tmpzz",
            "vx-0123456789abcdef.png",
            "v-0123456789abcdef.png",
            "desktop.ini",
            ".DS_Store",
        ] {
            assert!(!stale_name(name), "{name}");
        }
    }

    #[test]
    fn a_cache_folder_loses_the_old_versions_thumbnails_and_keeps_the_rest() {
        let dir = std::env::temp_dir().join(format!("yosh-thumbcache-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("0123456789abcdef.png")).unwrap(); // a folder: kept
        let vol = dir.join("vol.png");
        let mut img = image::GrayImage::new(40, 60);
        img.put_pixel(3, 4, image::Luma([200]));
        img.save(&vol).unwrap();
        for f in ["fedcba9876543210.png", "fedcba9876543210.tmp42", "keep.txt"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let bytes = std::sync::Arc::new(std::fs::read(&vol).unwrap());
        let mut resizer = Resizer::new();
        let img = load_or_decode(Some(&dir), &vol, 30, &mut resizer, || Some(bytes.clone())).unwrap();
        assert_eq!((img.w, img.h), (20, 30));
        let mut left: Vec<String> =
            std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        let new = cache_name(&vol, 30).unwrap();
        assert!(new.starts_with(&format!("v{VERSION}-")));
        let mut want = vec!["0123456789abcdef.png".to_string(), "keep.txt".into(), new, "vol.png".into()];
        want.sort();
        assert_eq!(left, want);
        // A hit now reads the new file.
        assert!(load_or_decode(Some(&dir), &vol, 30, &mut resizer, || None).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

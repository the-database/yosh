//! Page source abstraction: list pages and read one page's encoded bytes by
//! index, for the decode pool. Implementations: folder (M1.2), zip + rar (M1.6).

mod folder;
// RAR (CBR) is gated behind the on-by-default `rar` feature: the bundled UnRAR
// C++ (`unrar`) uses `lutimes`, which Android's Bionic libc lacks, so it can't
// cross-compile to Android. A shell that targets Android builds without it; the
// other formats (folder / CBZ / 7z) are unaffected.
#[cfg(feature = "rar")]
mod rar;
mod sevenz;
mod ziparc;
pub use folder::FolderSource;
#[cfg(feature = "rar")]
pub use rar::RarSource;
pub use sevenz::SevenzSource;
pub use ziparc::ZipSource;

use std::io;
use std::path::Path;
use std::sync::Arc;

use chardetng::{EncodingDetector, Iso2022JpDetection, Utf8Detection};
use encoding_rs::Encoding;

/// A source of comic pages. Must be `Send + Sync` so decode workers can pull
/// from it concurrently (folder/zip read in parallel; rar serializes internally).
pub trait PageSource: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Entry name for page `index` (file name / archive entry path).
    fn name(&self, index: usize) -> &str;
    /// Read the encoded image bytes for page `index`. May block (rar).
    ///
    /// Returns `Arc<Vec<u8>>`, not `Vec<u8>`: the sequential sources (rar / 7z)
    /// already hold every extracted page in an in-memory map behind an `Arc`, so
    /// handing that `Arc` out is a refcount bump instead of a full copy of a
    /// multi-MB page. Every consumer only ever reads the bytes (`&[u8]`), so the
    /// random-access sources (folder / zip) just wrap their fresh `Vec` — one
    /// allocation, no copy.
    fn read_page(&self, index: usize) -> io::Result<Arc<Vec<u8>>>;
    /// Modified timestamp for page `index`, formatted for display, if known.
    /// Default: unknown (archives without stored times, etc.).
    fn modified(&self, index: usize) -> Option<String> {
        let _ = index;
        None
    }
    /// True if this source is a *partial* recovery of a truncated/damaged archive —
    /// only the pages before the cutoff are present. Default: a complete source.
    fn is_partial(&self) -> bool {
        false
    }
    /// Ask the source to fail any `read_page` calls currently blocked waiting for
    /// data (the sequential sources: rar / 7z, where a page isn't readable until
    /// the extractor thread has walked to it). Reads started *later* are
    /// unaffected — this cancels the waiters that exist right now, it does not put
    /// the source into a permanently-failing state.
    ///
    /// Called by `DecodePool::Drop`, so a torn-down pool's workers stop parking on
    /// an archive nobody is reading any more and can exit; see that impl for the
    /// full chain. Default: a no-op — the random-access sources (folder / zip)
    /// never block, so they have no waiters to cancel.
    fn cancel_waits(&self) {}
}

/// Format a UNIX-epoch second count as `YYYY-MM-DD HH:MM UTC` (no time-zone
/// dependency). Uses Howard Hinnant's days→civil-date algorithm.
pub fn fmt_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02} UTC",
        year,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60
    )
}

/// Is this path an image we might be able to decode?
pub fn is_image_ext(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some(
            // Decoded by dedicated crates or the `image`-crate fallback. The
            // tif/tga/dds/exr/hdr/qoi/pnm group is decoded by the `image` crate's
            // default (pure-Rust) format set — they only need to be listed here.
            "png" | "jpg" | "jpeg" | "jpe" | "webp" | "gif" | "bmp" | "avif" | "jxl" | "psd"
                | "ico" | "tif" | "tiff" | "tga" | "dds" | "exr" | "hdr" | "qoi" | "pnm" | "ppm"
                | "pgm" | "pbm",
        )
    )
}

/// Is this archive entry name an image? (Handles `/`-separated archive paths.)
pub fn is_image_name(name: &str) -> bool {
    is_image_ext(Path::new(name))
}

/// Pick the codepage for one archive's legacy entry names.
///
/// Zip stores names as raw bytes and only flags UTF-8 via general-purpose bit 11; plenty
/// of Japanese/Chinese archives are written by tools that leave that flag clear and store
/// legacy codepage bytes (Shift-JIS, GBK, Big5, EUC-KR, …) instead.
///
/// Detection is done **once for the whole archive**, not per name. Filenames are far too
/// short to sniff individually — `第01話` is 6 bytes of CP932, which a detector will read
/// as Cyrillic about as readily as Japanese — but every entry in an archive was written by
/// one tool in one encoding, so pooling all of them turns a hopeless 6-byte sample into a
/// decisive one. Feeding names in archive order also keeps the verdict deterministic.
pub fn detect_legacy_encoding(raws: &[Vec<u8>]) -> &'static Encoding {
    // ISO-2022-JP is denied: it is stateful and effectively never used for filenames,
    // and allowing it lets stray escape bytes swing the guess.
    let mut det = EncodingDetector::new(Iso2022JpDetection::Deny);
    for raw in raws {
        det.feed(raw, false);
    }
    det.feed(&[], true);
    det.guess(None, Utf8Detection::Allow)
}

/// Decode one raw archive-entry name using the archive's detected codepage.
///
/// Valid UTF-8 always wins over `enc`: it covers correctly-flagged names and the very
/// common "UTF-8 bytes, flag never set" case, and UTF-8 is self-validating enough that a
/// name which parses cleanly is not a legacy-codepage name that happens to look like one.
pub fn decode_entry_name(raw: &[u8], enc: &'static Encoding) -> String {
    match std::str::from_utf8(raw) {
        Ok(s) => s.to_string(),
        Err(_) => enc.decode(raw).0.into_owned(),
    }
}

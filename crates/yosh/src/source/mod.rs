//! Page source abstraction: list pages and read one page's encoded bytes by
//! index, for the decode pool. Implementations: folder (M1.2), zip + rar (M1.6).

mod folder;
mod rar;
mod sevenz;
mod ziparc;
pub use folder::FolderSource;
pub use rar::RarSource;
pub use sevenz::SevenzSource;
pub use ziparc::ZipSource;

use std::io;
use std::path::Path;

/// A source of comic pages. Must be `Send + Sync` so decode workers can pull
/// from it concurrently (folder/zip read in parallel; rar serializes internally).
pub trait PageSource: Send + Sync {
    fn len(&self) -> usize;
    /// Entry name for page `index` (file name / archive entry path).
    fn name(&self, index: usize) -> &str;
    /// Read the encoded image bytes for page `index`. May block (rar).
    fn read_page(&self, index: usize) -> io::Result<Vec<u8>>;
    /// Modified timestamp for page `index`, formatted for display, if known.
    /// Default: unknown (archives without stored times, etc.).
    fn modified(&self, index: usize) -> Option<String> {
        let _ = index;
        None
    }
}

/// Format a UNIX-epoch second count as `YYYY-MM-DD HH:MM UTC` (no time-zone
/// dependency). Uses Howard Hinnant's days→civil-date algorithm.
pub fn fmt_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
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
            "png" | "jpg" | "jpeg" | "jpe" | "webp" | "gif" | "bmp" | "avif" | "jxl" | "psd"
                | "ico",
        )
    )
}

/// Is this archive entry name an image? (Handles `/`-separated archive paths.)
pub fn is_image_name(name: &str) -> bool {
    is_image_ext(Path::new(name))
}

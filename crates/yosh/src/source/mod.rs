//! Page source abstraction: list pages and read one page's encoded bytes by
//! index, for the decode pool. Implementations: folder (M1.2), zip + rar (M1.6).

mod folder;
pub use folder::FolderSource;

use std::io;
use std::path::Path;

/// A source of comic pages. Must be `Send + Sync` so decode workers can pull
/// from it concurrently (folder/zip read in parallel; rar serializes internally).
pub trait PageSource: Send + Sync {
    fn len(&self) -> usize;
    fn name(&self, index: usize) -> &str;
    /// Read the encoded image bytes for page `index`. May block (rar).
    fn read_page(&self, index: usize) -> io::Result<Vec<u8>>;
}

/// Is this path an image we might be able to decode?
pub fn is_image_ext(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "jpe" | "webp" | "gif" | "bmp" | "avif" | "jxl")
    )
}

//! Loose-image-folder source. Fully parallel reads (each page is an independent
//! file read).

use std::io;
use std::path::{Path, PathBuf};

use super::{is_image_ext, PageSource};

pub struct FolderSource {
    paths: Vec<PathBuf>,
    names: Vec<String>,
}

impl FolderSource {
    pub fn new(dir: &Path) -> io::Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                // `DirEntry::file_type()` is free on Windows (the type comes back
                // inline with the directory scan), so we avoid a `metadata()` stat —
                // one network round-trip per entry over a share — that `is_file()`
                // would cost. Only a symlink needs a real stat to resolve its target,
                // matching the old `is_file()` (which followed links); directories and
                // other types are dropped.
                let ft = e.file_type().ok()?;
                let p = e.path();
                let is_file = if ft.is_symlink() { p.is_file() } else { ft.is_file() };
                (is_file && is_image_ext(&p)).then_some(p)
            })
            .collect();
        // Natural sort by file name (so "p2" < "p10").
        paths.sort_by(|a, b| {
            let fa = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let fb = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
            natord::compare(fa, fb)
        });
        let names = paths
            .iter()
            .map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        Ok(Self { paths, names })
    }

    /// Index of the entry with the given file name, if present.
    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

impl PageSource for FolderSource {
    fn len(&self) -> usize {
        self.paths.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn read_page(&self, index: usize) -> io::Result<std::sync::Arc<Vec<u8>>> {
        std::fs::read(&self.paths[index]).map(std::sync::Arc::new)
    }

    fn modified(&self, index: usize) -> Option<String> {
        let secs = std::fs::metadata(self.paths.get(index)?)
            .ok()?
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        Some(super::fmt_unix(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::PageSource;

    #[test]
    fn lists_image_files_sorted_excluding_dirs_and_nonimages() {
        // Unique temp dir per process so concurrent test binaries don't collide.
        let dir = std::env::temp_dir().join(format!("yosh_folder_{}_list", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("02.jpg"), b"JPGDATA-TWO").unwrap();
        std::fs::write(dir.join("01.png"), b"PNGDATA-ONE").unwrap();
        std::fs::write(dir.join("notes.txt"), b"txt").unwrap();
        std::fs::write(dir.join("sub").join("inside.png"), b"nested").unwrap();

        let src = FolderSource::new(&dir).unwrap();
        // The subdirectory and the .txt are excluded; only the two images count.
        assert_eq!(src.len(), 2, "dir + non-image excluded");
        assert_eq!(src.name(0), "01.png"); // natural-sorted
        assert_eq!(src.name(1), "02.jpg");
        assert_eq!(*src.read_page(0).unwrap(), b"PNGDATA-ONE");
        assert_eq!(*src.read_page(1).unwrap(), b"JPGDATA-TWO");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn natural_sort_order() {
        let dir = std::env::temp_dir().join(format!("yosh_folder_{}_natsort", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["10.png", "2.png", "1.png"] {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
        let src = FolderSource::new(&dir).unwrap();
        let names: Vec<&str> = (0..src.len()).map(|i| src.name(i)).collect();
        assert_eq!(names, ["1.png", "2.png", "10.png"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

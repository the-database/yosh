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
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_image_ext(p))
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
}

impl PageSource for FolderSource {
    fn len(&self) -> usize {
        self.paths.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn read_page(&self, index: usize) -> io::Result<Vec<u8>> {
        std::fs::read(&self.paths[index])
    }
}

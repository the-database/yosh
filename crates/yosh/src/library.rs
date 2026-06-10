//! Library: scan a root folder for volumes (image subfolders + comic archives)
//! and hold lazily-decoded cover thumbnails for the browse grid.

use std::path::{Path, PathBuf};

use yosh_engine::source::{is_image_ext, FolderSource, PageSource, ZipSource};
use yosh_engine::page::PageTexture;

#[derive(Clone, Copy, PartialEq)]
pub enum VolKind {
    Folder,
    Zip,
    Rar,
    Sevenz,
}

pub struct Volume {
    pub path: PathBuf,
    pub name: String,
    pub kind: VolKind,
    pub thumb: Option<egui::TextureId>,
    /// Keeps the thumbnail texture/view alive while egui references it.
    pub thumb_tex: Option<PageTexture>,
    pub thumb_tried: bool,
}

impl Volume {
    fn new(path: PathBuf, name: String, kind: VolKind) -> Self {
        Self {
            path,
            name,
            kind,
            thumb: None,
            thumb_tex: None,
            thumb_tried: false,
        }
    }
}

pub struct Library {
    #[allow(dead_code)] // kept for future rescan / display
    pub root: Option<PathBuf>,
    pub volumes: Vec<Volume>,
}

impl Library {
    pub fn empty() -> Self {
        Self {
            root: None,
            volumes: Vec::new(),
        }
    }

    /// Scan `root`'s immediate children: subfolders containing images, and
    /// CBZ/CBR/ZIP/RAR/7z archives.
    pub fn scan(root: &Path) -> Self {
        let mut volumes = Vec::new();
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let path = e.path();
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                if path.is_dir() {
                    if dir_has_image(&path) {
                        volumes.push(Volume::new(path, name, VolKind::Folder));
                    }
                } else if let Some(kind) = archive_kind(&path) {
                    volumes.push(Volume::new(path, name, kind));
                }
            }
        }
        volumes.sort_by(|a, b| natord::compare(&a.name, &b.name));
        Self {
            root: Some(root.to_path_buf()),
            volumes,
        }
    }
}

/// Sibling volumes of `of` (same parent directory) that are the same *kind* —
/// folders if `of` is a folder, archives if `of` is an archive — in natural-sort
/// order. Used for prev/next-volume navigation (`[` / `]`); folders and archives
/// never mix. Includes `of` itself when it is a valid volume. Returns paths.
pub fn sibling_volumes(of: &Path) -> Vec<PathBuf> {
    let Some(parent) = of.parent() else {
        return Vec::new();
    };
    let want_folder = of.is_dir();
    Library::scan(parent)
        .volumes
        .into_iter()
        .filter(|v| (v.kind == VolKind::Folder) == want_folder)
        .map(|v| v.path)
        .collect()
}

fn dir_has_image(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.flatten().any(|e| {
            // Use the entry's cached type to avoid a stat per file (a network
            // round-trip on a share); only follow a symlink with a real check.
            let is_file = match e.file_type() {
                Ok(ft) if ft.is_symlink() => e.path().is_file(),
                Ok(ft) => ft.is_file(),
                Err(_) => false,
            };
            is_file && is_image_ext(&e.path())
        })
    })
}

fn archive_kind(p: &Path) -> Option<VolKind> {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("cbz") | Some("zip") => Some(VolKind::Zip),
        Some("cbr") | Some("rar") => Some(VolKind::Rar),
        Some("7z") | Some("cb7") => Some(VolKind::Sevenz),
        _ => None,
    }
}

/// Encoded bytes of a volume's first page, for a cover thumbnail. Only cheap
/// (random-access) sources — folders and ZIP; RAR/7z covers are skipped to
/// avoid spinning up a full sequential extractor for a thumbnail.
pub fn cover_bytes(vol: &Volume) -> Option<Vec<u8>> {
    match vol.kind {
        VolKind::Folder => {
            let s = FolderSource::new(&vol.path).ok()?;
            (s.len() > 0).then(|| s.read_page(0).ok()).flatten()
        }
        VolKind::Zip => {
            let s = ZipSource::new(&vol.path).ok()?;
            (s.len() > 0).then(|| s.read_page(0).ok()).flatten()
        }
        VolKind::Rar | VolKind::Sevenz => None,
    }
}

//! Library: recursively scan a root folder, grouping comics into *series*
//! (folders that directly hold volumes — image subfolders and/or archives), and
//! hold lazily-decoded cover thumbnails for the sectioned browse view. Read state
//! (unread / in-progress / finished) is derived from the persisted progress map.

use std::collections::{HashMap, HashSet};
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
    /// Monotonic stamp of the last frame this cover was on screen — drives LRU
    /// eviction so a deep library can't pin every cover texture at once.
    pub last_seen: u64,
}

impl Volume {
    fn new(path: PathBuf, name: String, kind: VolKind) -> Self {
        Self {
            path,
            name,
            kind,
            thumb: None,
            thumb_tex: None,
            last_seen: 0,
        }
    }
}

/// One library series: a folder that directly holds volumes (comic archives
/// and/or image-folder comics), shown as a collapsible section with a horizontal
/// cover row.
pub struct Series {
    pub dir: PathBuf,
    pub name: String,
    pub volumes: Vec<Volume>,
}

pub struct Library {
    pub root: Option<PathBuf>,
    pub series: Vec<Series>,
}

impl Library {
    pub fn empty() -> Self {
        Self {
            root: None,
            series: Vec::new(),
        }
    }

    /// Recursively scan `root` and group its comics into [`Series`] — every folder
    /// that directly holds at least one volume. Series are natural-sorted by path
    /// (so nested series sit next to their parents); volumes within a series are
    /// natural-sorted by name.
    pub fn scan(root: &Path) -> Self {
        let mut series = Vec::new();
        if walk_series(root, 0, &mut series) {
            // The root itself is an image-folder comic: a one-volume "series".
            series.push(Series {
                name: name_of(root),
                volumes: vec![Volume::new(root.to_path_buf(), name_of(root), VolKind::Folder)],
                dir: root.to_path_buf(),
            });
        }
        series.sort_by(|a, b| {
            natord::compare(
                &a.dir.to_string_lossy().to_lowercase(),
                &b.dir.to_string_lossy().to_lowercase(),
            )
        });
        Self {
            root: Some(root.to_path_buf()),
            series,
        }
    }

    /// No volumes anywhere (a series is only created when it has volumes, so this
    /// is just "no series").
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    pub fn all_volumes(&self) -> impl Iterator<Item = &Volume> {
        self.series.iter().flat_map(|s| s.volumes.iter())
    }
}

/// Max folder depth [`Library::scan`] descends looking for series.
const SERIES_MAX_DEPTH: usize = 5;

/// One `read_dir` per folder: image files make the folder itself a volume
/// (returns true; the caller adds it and does not descend), archives become
/// volumes, and remaining sub-folders recurse as potential series.
fn walk_series(dir: &Path, depth: usize, out: &mut Vec<Series>) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut volumes: Vec<Volume> = Vec::new();
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        // Prefer the entry's cached type to avoid a stat per file (a network
        // round-trip on a share); only follow a symlink with a real check.
        let is_dir = match e.file_type() {
            Ok(ft) if ft.is_symlink() => p.is_dir(),
            Ok(ft) => ft.is_dir(),
            Err(_) => p.is_dir(),
        };
        if is_dir {
            subdirs.push(p);
        } else if let Some(kind) = archive_kind(&p) {
            volumes.push(Volume::new(p.clone(), name_of(&p), kind));
        } else if is_image_ext(&p) {
            return true; // an image-folder comic — the caller's volume
        }
    }
    for d in subdirs {
        if depth < SERIES_MAX_DEPTH && walk_series(&d, depth + 1, out) {
            // `d` is itself an image-folder comic → a volume in *this* series.
            volumes.push(Volume::new(d.clone(), name_of(&d), VolKind::Folder));
        }
    }
    if !volumes.is_empty() {
        volumes.sort_by(|a, b| natord::compare(&a.name.to_lowercase(), &b.name.to_lowercase()));
        out.push(Series {
            name: name_of(dir),
            dir: dir.to_path_buf(),
            volumes,
        });
    }
    false
}

/// Shell read-state the library view reads each frame, lent into [`crate::ui::chrome`].
pub struct LibCtx<'a> {
    pub progress: &'a HashMap<String, (usize, usize)>,
    pub last_pages: &'a HashMap<String, usize>,
    pub collapsed: &'a HashSet<String>,
    pub current_key: Option<&'a str>,
    /// Most-recently-read volume paths, newest first — backs the "Recently read"
    /// shelf at the top of the library view.
    pub recents: &'a [String],
}

/// A volume's read state, derived from the shell's progress/last-page maps.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VolState {
    Unread,
    /// Started but not finished; the fraction read (0..1) drives the progress bar.
    InProgress(f32),
    Finished,
}

/// Derive a volume's read state. `Finished` once the furthest page seen reached
/// the total; a legacy `last_pages` entry without progress data counts as started
/// (pre-tracking books can't claim a furthest page).
pub fn vol_state(
    progress: &HashMap<String, (usize, usize)>,
    last_pages: &HashMap<String, usize>,
    key: &str,
) -> VolState {
    match progress.get(key) {
        Some(&(furthest, total)) if total > 0 && furthest >= total => VolState::Finished,
        Some(&(furthest, total)) => VolState::InProgress(furthest as f32 / total.max(1) as f32),
        None if last_pages.contains_key(key) => VolState::InProgress(0.0),
        None => VolState::Unread,
    }
}

/// The series header's right-side status label.
pub fn series_status(states: &[VolState]) -> String {
    let unread = states.iter().filter(|s| **s == VolState::Unread).count();
    let reading = states.iter().any(|s| matches!(s, VolState::InProgress(_)));
    if reading {
        if unread > 0 {
            format!("Reading · {unread} unread")
        } else {
            "Reading".to_string()
        }
    } else if unread > 0 {
        format!("{unread} unread")
    } else {
        "Finished".to_string()
    }
}

/// Sibling volumes of `of` (same parent directory) that are the same *kind* —
/// folders if `of` is a folder, archives if `of` is an archive — in natural-sort
/// order. Used for prev/next-volume navigation (`[` / `]`); folders and archives
/// never mix. Includes `of` itself when it is a valid volume. Returns paths.
///
/// This stays *single-level* (just `of`'s parent) — it is unrelated to the
/// sectioned grid's recursive scan.
pub fn sibling_volumes(of: &Path) -> Vec<PathBuf> {
    let Some(parent) = of.parent() else {
        return Vec::new();
    };
    let want_folder = of.is_dir();
    scan_dir_volumes(parent)
        .into_iter()
        .filter(|v| (v.kind == VolKind::Folder) == want_folder)
        .map(|v| v.path)
        .collect()
}

/// Immediate children of `root` that are volumes: subfolders containing images,
/// and CBZ/CBR/ZIP/RAR/7z archives, natural-sorted by name.
fn scan_dir_volumes(root: &Path) -> Vec<Volume> {
    let mut volumes = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let path = e.path();
            let name = name_of(&path);
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
    volumes
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
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
/// avoid spinning up a full sequential extractor for a thumbnail. Takes
/// `(path, kind)` rather than `&Volume` so an off-thread cover-decode worker can
/// call it without holding a borrow of the library.
pub fn cover_bytes(path: &Path, kind: VolKind) -> Option<Vec<u8>> {
    match kind {
        VolKind::Folder => {
            let s = FolderSource::new(path).ok()?;
            (s.len() > 0).then(|| s.read_page(0).ok()).flatten()
        }
        VolKind::Zip => {
            let s = ZipSource::new(path).ok()?;
            (s.len() > 0).then(|| s.read_page(0).ok()).flatten()
        }
        VolKind::Rar | VolKind::Sevenz => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn vol_state_transitions() {
        let mut progress: HashMap<String, (usize, usize)> = HashMap::new();
        let mut last_pages: HashMap<String, usize> = HashMap::new();

        // No data at all → Unread.
        assert_eq!(vol_state(&progress, &last_pages, "a"), VolState::Unread);

        // A last-page entry but no progress → started (pre-tracking volume).
        last_pages.insert("a".into(), 3);
        assert_eq!(vol_state(&progress, &last_pages, "a"), VolState::InProgress(0.0));

        // Partway through → InProgress with the read fraction.
        progress.insert("b".into(), (5, 10));
        assert_eq!(vol_state(&progress, &last_pages, "b"), VolState::InProgress(0.5));

        // Furthest reached the total → Finished.
        progress.insert("c".into(), (10, 10));
        assert_eq!(vol_state(&progress, &last_pages, "c"), VolState::Finished);

        // Re-reading past the end stays Finished (furthest >= total).
        progress.insert("d".into(), (12, 10));
        assert_eq!(vol_state(&progress, &last_pages, "d"), VolState::Finished);
    }

    #[test]
    fn series_status_labels() {
        use VolState::*;
        assert_eq!(series_status(&[Unread, Unread]), "2 unread");
        assert_eq!(series_status(&[Finished, Finished]), "Finished");
        assert_eq!(series_status(&[InProgress(0.5), Finished]), "Reading");
        assert_eq!(
            series_status(&[InProgress(0.5), Unread, Unread]),
            "Reading · 2 unread"
        );
        // An all-empty series reads as finished (no unread, none in progress).
        assert_eq!(series_status(&[]), "Finished");
    }

    #[test]
    fn scan_groups_nested_series() {
        // root/
        //   SeriesA/  v1.cbz  v2.cbz                  -> series "SeriesA" (2 vols)
        //   SeriesB/  ch1/<img>  ch2/<img>            -> series "SeriesB" (2 image-folder vols)
        //   loose.cbz                                 -> root is itself a series (1 vol)
        let tmp = std::env::temp_dir().join(format!("yosh_lib_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let a = tmp.join("SeriesA");
        let b = tmp.join("SeriesB");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(b.join("ch1")).unwrap();
        std::fs::create_dir_all(b.join("ch2")).unwrap();
        std::fs::write(a.join("v1.cbz"), b"PK").unwrap();
        std::fs::write(a.join("v2.cbz"), b"PK").unwrap();
        std::fs::write(b.join("ch1").join("001.jpg"), b"\xff\xd8").unwrap();
        std::fs::write(b.join("ch2").join("001.jpg"), b"\xff\xd8").unwrap();
        std::fs::write(tmp.join("loose.cbz"), b"PK").unwrap();

        let lib = Library::scan(&tmp);
        let by_name: HashMap<&str, usize> = lib
            .series
            .iter()
            .map(|s| (s.name.as_str(), s.volumes.len()))
            .collect();
        assert_eq!(by_name.get("SeriesA"), Some(&2));
        assert_eq!(by_name.get("SeriesB"), Some(&2));
        // The root folder holds `loose.cbz` directly → it's a series too.
        let root_name = tmp.file_name().unwrap().to_str().unwrap();
        assert_eq!(by_name.get(root_name), Some(&1));

        // Smoke-check the read-state helpers don't choke on the scanned set.
        let progress = HashMap::new();
        let last_pages = HashMap::new();
        let _collapsed: HashSet<String> = HashSet::new();
        for v in lib.all_volumes() {
            let key = v.path.to_string_lossy().into_owned();
            assert_eq!(vol_state(&progress, &last_pages, &key), VolState::Unread);
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

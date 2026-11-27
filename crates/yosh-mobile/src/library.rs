//! The library browser's model: series scan, read states, cover thumbnails.

use super::*;

/// A library browser row: a folder to descend into (or open if it holds images),
/// or a comic archive to open.
pub(crate) enum Entry {
    Dir(PathBuf),
    Comic(PathBuf),
}

/// One library series: a folder that directly holds volumes (comic archives
/// and/or image-folder comics). Built by [`spawn_library_scan`].
pub(crate) struct Series {
    pub(crate) dir: PathBuf,
    pub(crate) name: String,
    pub(crate) volumes: Vec<PathBuf>,
    /// Tile caption per volume, parallel to `volumes`: the filename minus its
    /// archive extension and the prefix the whole series repeats.
    pub(crate) captions: Vec<String>,
}

/// A volume's read state, derived from the shell's progress/positions maps.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum VolState {
    Unread,
    /// Started but not finished; the fraction read (0..1) drives the progress bar.
    InProgress(f32),
    Finished,
}

/// Derive a volume's read state. `Finished` once the furthest page seen reached
/// the total; a legacy `positions` entry without progress data counts as
/// started (pre-tracking books can't claim a furthest page).
pub(crate) fn vol_state(
    progress: &HashMap<String, (u32, u32)>,
    positions: &HashMap<String, usize>,
    key: &str,
) -> VolState {
    match progress.get(key) {
        Some(&(furthest, total)) if total > 0 && furthest >= total => VolState::Finished,
        Some(&(furthest, total)) => {
            VolState::InProgress(furthest as f32 / total.max(1) as f32)
        }
        None if positions.contains_key(key) => VolState::InProgress(0.0),
        None => VolState::Unread,
    }
}

/// The series header's right-side status label.
pub(crate) fn series_status(states: &[VolState]) -> String {
    let unread = states.iter().filter(|s| **s == VolState::Unread).count();
    let reading = states.iter().any(|s| matches!(s, VolState::InProgress(_)));
    if reading {
        if unread > 0 {
            (tr().reading_unread)(unread)
        } else {
            tr().reading.to_string()
        }
    } else if unread > 0 {
        (tr().unread)(unread)
    } else {
        tr().finished.to_string()
    }
}

/// Max folder depth `spawn_library_scan` descends looking for series.
const SERIES_MAX_DEPTH: usize = 5;

/// Walk `root` off-thread and group its comics into [`Series`] — every folder
/// that directly holds at least one volume. Sent back whole (drained in render
/// like the cover decodes), so a deep tree or slow storage never hitches the UI.
pub(crate) fn spawn_library_scan(root: PathBuf, tx: std::sync::mpsc::Sender<Vec<Series>>, waker: Option<Waker>) {
    std::thread::spawn(move || {
        let mut out = Vec::new();
        if walk_series(&root, 0, &mut out) {
            // The root itself is an image-folder comic: a one-volume "series".
            out.push(Series {
                name: name_of(&root),
                captions: vec![yosh_engine::caption::display_stem(&name_of(&root)).to_string()],
                volumes: vec![root.clone()],
                dir: root,
            });
        }
        // Path order keeps nested series next to their parents, naturally sorted.
        out.sort_by(|a, b| {
            natord::compare(
                &a.dir.to_string_lossy().to_lowercase(),
                &b.dir.to_string_lossy().to_lowercase(),
            )
        });
        // Send, then wake: the loop is idle while the scan runs, so the result only
        // reaches the screen if this schedules the frame that drains `series_rx`.
        let _ = tx.send(out);
        if let Some(w) = &waker {
            w();
        }
    });
}

/// One `read_dir` per folder: image files make the folder itself a volume
/// (returns true; the caller adds it and does not descend), archives become
/// volumes, and remaining sub-folders recurse as potential series.
fn walk_series(dir: &Path, depth: usize, out: &mut Vec<Series>) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    let mut volumes: Vec<PathBuf> = Vec::new();
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if is_comic_archive(&p) {
            volumes.push(p);
        } else if is_image_ext(&p) {
            return true; // an image-folder comic — the caller's volume
        }
    }
    for d in subdirs {
        if depth < SERIES_MAX_DEPTH && walk_series(&d, depth + 1, out) {
            volumes.push(d);
        }
    }
    if !volumes.is_empty() {
        volumes.sort_by(|a, b| natord::compare(&name_of(a).to_lowercase(), &name_of(b).to_lowercase()));
        // Captions drop the head the whole series repeats, so they're a per-series
        // computation — done here, on the scan thread, never per frame.
        let names: Vec<String> = volumes.iter().map(|p| name_of(p)).collect();
        let refs: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
        let captions = yosh_engine::caption::series_captions(&refs);
        out.push(Series {
            name: name_of(dir),
            dir: dir.to_path_buf(),
            volumes,
            captions,
        });
    }
    false
}

/// Shell-owned state the library view reads each frame (the App is rebuilt on
/// every resume, so durable state lives on the Shell and is lent to `render`).
pub(crate) struct LibCtx<'a> {
    pub(crate) progress: &'a HashMap<String, (u32, u32)>,
    pub(crate) positions: &'a HashMap<String, usize>,
    pub(crate) collapsed: &'a std::collections::HashSet<PathBuf>,
    pub(crate) current_key: Option<&'a str>,
    /// Most-recently-read keys (newest first) for the "Recently read" row.
    pub(crate) recents: &'a [String],
    /// Which of those still exist on disk (see [`Shell::recents_ok`]) — the render
    /// filters against this set instead of stat-ing every recent, every frame.
    pub(crate) recents_ok: &'a HashSet<String>,
}

/// Per-frame owned snapshot of one series section for the egui closure.
pub(crate) struct SectionRow {
    pub(crate) dir: PathBuf,
    pub(crate) name: String,
    pub(crate) expanded: bool,
    pub(crate) status: String,
    /// Empty when collapsed (no need to clone what won't draw).
    pub(crate) volumes: Vec<VolCell>,
}

/// Per-frame owned snapshot of one volume cell.
pub(crate) struct VolCell {
    pub(crate) path: PathBuf,
    pub(crate) label: String,
    pub(crate) thumb: Option<egui::TextureHandle>,
    pub(crate) state: VolState,
    pub(crate) is_current: bool,
}

/// Last path component as a display string.
pub(crate) fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

/// Human-readable book name from a `current_key` (a filesystem path or a
/// content:// URI). Paths → basename; SAF document URIs (e.g.
/// `…/document/primary:Manga/Title.cbz`) → last decoded segment after `/` or `:`.
pub(crate) fn book_display_name(key: &str) -> String {
    if is_document_key(key) {
        // Minimal percent-decode of just the separators SAF uses, then take the
        // final component. Good enough for a display name; fall back to the raw
        // key if there's nothing better to show.
        let decoded = key.replace("%2F", "/").replace("%2f", "/").replace("%3A", ":");
        let last = decoded
            .rsplit(['/', ':'])
            .find(|s| !s.is_empty())
            .unwrap_or(key);
        return last.to_string();
    }
    name_of(Path::new(key))
}

/// Decode a comic's cover (page 0) to a small egui image, via the on-disk thumbnail
/// cache (a cache hit reads a tiny PNG instead of decoding the full first page).
fn decode_cover(
    path: &Path,
    cache_dir: Option<&Path>,
    resizer: &mut fast_image_resize::Resizer,
) -> Option<egui::ColorImage> {
    let rgba = yosh_engine::thumbcache::load_or_decode(cache_dir, path, 320, resizer, || {
        let src = build_source(path)?;
        (!src.is_empty()).then(|| src.read_page(0).ok()).flatten()
    })?;
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [rgba.w as usize, rgba.h as usize],
        &rgba.pixels,
    ))
}

/// Spawn the single, persistent cover-decode worker: batches of comic paths in
/// (`Sender<Vec<PathBuf>>`), decoded covers streamed back out through `tx`,
/// reading/writing the on-disk thumbnail cache as it goes. Each cover is sent
/// then followed by a `waker` call, so a library sitting idle draws the covers as
/// they arrive without the frame loop polling for them; a path that fails to
/// decode still sends (`None`) so the shell's in-flight count always drains.
///
/// One thread for the App's whole lifetime, reusing one `Resizer`, instead of a
/// thread per batch — scrolling a library used to mint a fresh thread (and a
/// fresh resizer) every few frames, all of them competing for the same cores as
/// the decode pool. It is detached and never joined: when the App is dropped its
/// `Sender` goes with it, `recv` fails, and the loop returns (a batch already in
/// progress finishes first, then the `tx` send fails and it returns anyway).
pub(crate) fn spawn_cover_worker(
    cache_dir: Option<PathBuf>,
    tx: std::sync::mpsc::Sender<(PathBuf, Option<egui::ColorImage>)>,
    waker: Option<Waker>,
) -> std::sync::mpsc::Sender<Vec<PathBuf>> {
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Vec<PathBuf>>();
    std::thread::spawn(move || {
        let mut resizer = fast_image_resize::Resizer::new();
        while let Ok(paths) = job_rx.recv() {
            for path in paths {
                let img = decode_cover(&path, cache_dir.as_deref(), &mut resizer);
                if tx.send((path, img)).is_err() {
                    return; // receiver gone (App torn down)
                }
                if let Some(w) = &waker {
                    w();
                }
            }
        }
    });
    job_tx
}

/// List a directory's sub-folders + comic archives, folders first, natural order.
pub(crate) fn scan_dir(dir: &Path) -> Vec<Entry> {
    let mut entries: Vec<Entry> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.is_dir() {
                Some(Entry::Dir(p))
            } else if is_comic_archive(&p) {
                Some(Entry::Comic(p))
            } else {
                None
            }
        })
        .collect();
    entries.sort_by(|a, b| {
        let key = |e: &Entry| match e {
            Entry::Dir(p) => (0, name_of(p).to_lowercase()),
            Entry::Comic(p) => (1, name_of(p).to_lowercase()),
        };
        let (ad, an) = key(a);
        let (bd, bn) = key(b);
        ad.cmp(&bd).then_with(|| natord::compare(&an, &bn))
    });
    entries
}

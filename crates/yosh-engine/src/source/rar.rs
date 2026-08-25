//! CBR/RAR source. RAR is strictly sequential (no random access), so a single
//! reader thread decompresses entries into an in-memory map; the decode workers'
//! `read_page(i)` blocks until entry `i` has been produced.
//!
//! **Resume order, not archive order.** Opening at a saved position used to mean
//! waiting out every entry before it, because the walk was strictly front-to-back.
//! On a *non-solid* archive `skip()` is a plain fseek past the entry (UnRAR's
//! `RAR_SKIP` → `SeekToNext()`), so `with_start` runs two passes instead: pass 1
//! seeks to just before the resume page and decompresses from there to the end,
//! pass 2 reopens and backfills the pages it skipped. The resume page lands in
//! milliseconds and the rest arrives while you read. A *solid* archive can't do
//! this — there `skip()` still decompresses, only discarding the output — so it
//! stays single-pass, exactly as before.
//!
//! M1 keeps every extracted page in memory (bounded by archive size — fine for
//! typical CBRs on this hardware). Windowed eviction is an M2 optimization.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use unrar::Archive;

use super::{is_image_name, PageSource};

/// How many pages *before* the resume hint pass 1 also extracts, so paging
/// backwards off the resume point doesn't immediately stall on pass 2. Covers
/// the prefetch window's backward reach (`Budget.back`, at most 6 — reader.rs)
/// with room to spare; the cost of overshooting is a few extra entries.
const RESUME_MARGIN: usize = 8;

struct Ready {
    map: HashMap<usize, Arc<Vec<u8>>>,
    done: bool,
    error: Option<String>,
}

struct Shared {
    ready: Mutex<Ready>,
    cv: Condvar,
    abort: AtomicBool,
    /// Bumped by [`PageSource::cancel_waits`] to release everyone currently parked
    /// in `read_page`. An *epoch*, not a flag: a cancel aimed at a dying pool's
    /// waiters must not poison waits that start afterwards against the same source
    /// (a source outlives the pool that triggered the cancel whenever another `Arc`
    /// clone is still around), and each waiter only compares against the value it
    /// snapshotted when it started waiting.
    cancel_epoch: AtomicU64,
}

pub struct RarSource {
    names: Vec<String>,
    shared: Arc<Shared>,
    _reader: JoinHandle<()>,
}

impl RarSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        Self::with_start(path, None)
    }

    /// Open `path`, hinting that reading will begin around page `start` (a saved
    /// position or a CLI start index) so extraction can be ordered to serve that
    /// page first. The hint is **advisory**: the shell re-resolves the real start
    /// index authoritatively when it applies the source, and both passes together
    /// still extract every page, so a stale or wrong hint costs extraction order
    /// and nothing else.
    pub fn with_start(path: &Path, start: Option<usize>) -> io::Result<Self> {
        // List entries up front (archive order ~ reading order); natural-sort by name.
        let mut names: Vec<String> = Vec::new();
        let listing = Archive::new(path)
            .open_for_listing()
            .map_err(|e| io::Error::other(e.to_string()))?;
        // Archive-level flag, and it has to be read *before* the loop below, which
        // consumes the listing handle.
        let solid = listing.is_solid();
        for entry in listing {
            let entry = entry.map_err(|e| io::Error::other(e.to_string()))?;
            let name = entry.filename.to_string_lossy().into_owned();
            if is_image_name(&name) {
                names.push(name);
            }
        }
        names.sort_by(|a, b| natord::compare(a, b));

        let name_to_idx: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Where pass 1 starts extracting. Solid archives can't seek past an entry
        // without decompressing it, so skipping ahead buys nothing — they take the
        // single-pass path (`0`), which is byte-for-byte the old behaviour.
        let first_wanted = if solid {
            0
        } else {
            start
                .unwrap_or(0)
                .saturating_sub(RESUME_MARGIN)
                .min(names.len().saturating_sub(1))
        };

        let shared = Arc::new(Shared {
            ready: Mutex::new(Ready {
                map: HashMap::new(),
                done: false,
                error: None,
            }),
            cv: Condvar::new(),
            abort: AtomicBool::new(false),
            cancel_epoch: AtomicU64::new(0),
        });

        let reader = {
            let shared = shared.clone();
            let path = path.to_path_buf();
            std::thread::spawn(move || extract_all(path, name_to_idx, shared, first_wanted))
        };

        Ok(Self {
            names,
            shared,
            _reader: reader,
        })
    }
}

fn finish(shared: &Shared, error: Option<String>) {
    let mut g = shared.ready.lock().unwrap();
    g.done = true;
    g.error = error;
    drop(g);
    shared.cv.notify_all();
}

/// One front-to-back walk of the archive: extract every tracked entry whose sorted
/// page index satisfies `want`, and seek past everything else — on a non-solid
/// archive `skip()` is a pure fseek, which is what makes a second pass cheap.
///
/// Opens its own cursor because `OpenArchive` is `!Send` (it must live entirely
/// inside the thread that uses it) and because the cursor is forward-only: a
/// multi-pass walk *is* a reopen, and reopening a RAR is header parsing, not
/// decompression.
///
/// `Ok(true)` = walked to the end, `Ok(false)` = aborted mid-walk, `Err` = archive
/// error (already stringified).
fn run_pass(
    path: &Path,
    name_to_idx: &HashMap<String, usize>,
    shared: &Shared,
    want: impl Fn(usize) -> bool,
) -> Result<bool, String> {
    let mut cursor = Archive::new(path)
        .open_for_processing()
        .map_err(|e| e.to_string())?;
    loop {
        if shared.abort.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let header = match cursor.read_header() {
            Ok(Some(h)) => h,
            Ok(None) => return Ok(true),
            Err(e) => return Err(e.to_string()),
        };
        let name = header.entry().filename.to_string_lossy().into_owned();
        match name_to_idx.get(&name).copied().filter(|&idx| want(idx)) {
            Some(idx) => {
                let (bytes, next) = header.read().map_err(|e| e.to_string())?;
                {
                    let mut g = shared.ready.lock().unwrap();
                    g.map.insert(idx, Arc::new(bytes));
                }
                shared.cv.notify_all(); // send-then-wake
                cursor = next;
            }
            None => cursor = header.skip().map_err(|e| e.to_string())?,
        }
    }
}

/// Reader thread: extract every tracked entry into the map, resume-page first.
///
/// Cost model for `first_wanted > 0` (non-solid only): pass 1 fseeks through the
/// entries before it in microseconds and decompresses `first_wanted..end`, so the
/// page you resumed on is ready almost immediately; pass 2 then re-walks the
/// headers and backfills `0..first_wanted` in the background. `first_wanted == 0`
/// — no hint, a hint inside the margin, or a solid archive — collapses to the
/// original single front-to-back pass.
fn extract_all(
    path: PathBuf,
    name_to_idx: HashMap<String, usize>,
    shared: Arc<Shared>,
    first_wanted: usize,
) {
    match run_pass(&path, &name_to_idx, &shared, |idx| idx >= first_wanted) {
        Err(e) => return finish(&shared, Some(e)),
        // An abort still has to set `done`: a `read_page` that parks after the
        // extractor has walked away would otherwise wait forever. Pages already in
        // the map keep being served; a page that never arrived reports the existing
        // accurate "not found" message. (Abort only fires from `Drop`, so no user
        // ever sees it.)
        Ok(false) => return finish(&shared, None),
        Ok(true) => {}
    }
    if first_wanted > 0 {
        match run_pass(&path, &name_to_idx, &shared, |idx| idx < first_wanted) {
            Err(e) => return finish(&shared, Some(e)),
            Ok(false) => return finish(&shared, None),
            Ok(true) => {}
        }
    }
    finish(&shared, None);
}

impl PageSource for RarSource {
    fn len(&self) -> usize {
        self.names.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn read_page(&self, index: usize) -> io::Result<Arc<Vec<u8>>> {
        let mut guard = self.shared.ready.lock().unwrap();
        let epoch = self.shared.cancel_epoch.load(Ordering::Relaxed);
        loop {
            // The extracted page already lives in the map behind an `Arc`: hand
            // out a clone of that handle (refcount bump), never a byte copy.
            if let Some(bytes) = guard.map.get(&index) {
                return Ok(bytes.clone());
            }
            if guard.done {
                let msg = guard
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("rar: page {index} not found"));
                return Err(io::Error::other(msg));
            }
            // Checked *after* the map hit, so a cancel never denies a page that is
            // already extracted — cancelling releases waiters, it doesn't close the
            // source.
            if self.shared.cancel_epoch.load(Ordering::Relaxed) != epoch {
                return Err(io::Error::other("rar: read cancelled"));
            }
            guard = self.shared.cv.wait(guard).unwrap();
        }
    }

    fn cancel_waits(&self) {
        self.shared.cancel_epoch.fetch_add(1, Ordering::Relaxed);
        // **Load-bearing, not a stray statement.** `read_page` holds the ready mutex
        // continuously from its epoch check to `cv.wait`; taking that same mutex here
        // serializes the bump-and-notify against that window, so a waiter can't read
        // the old epoch, be signalled, and only then park — the classic lost wakeup,
        // which here would strand a worker for the life of the extraction.
        drop(self.shared.ready.lock().unwrap());
        self.shared.cv.notify_all();
    }
}

impl Drop for RarSource {
    fn drop(&mut self) {
        // Ask the reader to stop; let it finish its current entry (no join — avoid
        // blocking the UI on a large in-progress decompress).
        self.shared.abort.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
    }
}

//! 7z/CB7 source. 7z is block-compressed and typically *solid*, so a page can't
//! be read without decoding the block it sits in; a single reader thread decodes
//! blocks into an in-memory map and `read_page(i)` blocks until entry `i` has been
//! produced. (Same shape as `rar.rs`; bounded by archive size — M2.)
//!
//! **Blocks are decoded in reading order, not archive order.** The header is parsed
//! once up front (`Archive::open`), which is enough to know which block every page
//! lives in — and `BlockDecoder` re-seeks to a block's pack offset, so blocks may be
//! decoded in *any* order. `with_start` therefore ranks blocks by how near their
//! pages are to the resume position (the prefetch window's own metric) and decodes
//! the nearest one first, so reopening at page 200 doesn't wait out pages 0..199.
//! Blocks holding no tracked page are never decoded at all.
//!
//! Two things the older `ArchiveReader::for_each_entries` walk got wrong, fixed here
//! because this loop owns the iteration:
//! - Entries this source doesn't track (a `notes.txt` between two pages) are now
//!   **drained**, not ignored. Inside a solid block every entry is a window over one
//!   shared sequential stream, so leaving an untracked entry unread desynced the
//!   stream and corrupted every page after it in that block.
//! - Aborting now stops the *whole* extraction. `for_each_entries` discarded the
//!   per-block `Ok(false)`, so an abort ended the current block and then carried
//!   right on through every remaining one.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use sevenz_rust2::{Archive, BlockDecoder, Password};

use super::{is_image_name, PageSource};

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

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

pub struct SevenzSource {
    names: Vec<String>,
    shared: Arc<Shared>,
    _reader: JoinHandle<()>,
}

impl SevenzSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        Self::with_start(path, None)
    }

    /// Open `path`, hinting that reading will begin around page `start` (a saved
    /// position or a CLI start index) so the block containing that page is decoded
    /// first. The hint is **advisory**: the shell re-resolves the real start index
    /// authoritatively when it applies the source, and every block with a tracked
    /// page is decoded regardless, so a stale or wrong hint costs decode order and
    /// nothing else.
    pub fn with_start(path: &Path, start: Option<usize>) -> io::Result<Self> {
        // Header parse only — no block is decoded here. The parsed `Archive` is then
        // handed to the extractor thread (it is plain owned data), which saves a
        // second parse and is what lets that thread choose its own block order.
        let archive = Archive::open(path).map_err(to_io)?;
        let mut names: Vec<String> = archive
            .files
            .iter()
            .filter(|e| !e.is_directory() && is_image_name(e.name()))
            .map(|e| e.name().to_string())
            .collect();
        names.sort_by(|a, b| natord::compare(a, b));

        // Track pages by **file index**, parallel to `archive.files` — which is
        // exactly what the block walk below counts in. A name→page map would have to
        // be re-consulted per entry, and the walk has no name to consult it with that
        // is guaranteed unique anyway.
        let file_to_page: Vec<Option<usize>> = {
            let name_to_idx: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), i))
                .collect();
            archive
                .files
                .iter()
                .map(|f| {
                    if f.is_directory() {
                        None
                    } else {
                        name_to_idx.get(f.name()).copied()
                    }
                })
                .collect()
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

        let start = start.unwrap_or(0);
        let reader_thread = {
            let shared = shared.clone();
            let path = path.to_path_buf();
            std::thread::spawn(move || extract_all(path, archive, file_to_page, start, shared))
        };

        Ok(Self {
            names,
            shared,
            _reader: reader_thread,
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

/// The prefetch window's distance metric (`prefetch::desired_window`), duplicated
/// here on purpose so blocks are decoded in the same order the decode pool will ask
/// for their pages: forward outranks backward at equal distance, because that is the
/// direction reading travels. `s == 0` degenerates to plain archive order.
fn rank(i: usize, s: usize) -> u64 {
    if i >= s {
        (i - s) as u64 * 2
    } else {
        (s - i) as u64 * 3 + 1
    }
}

/// One block that holds at least one tracked page, and what the walk needs to know
/// about it. Blocks with no tracked page never become a plan, so they are never
/// decoded at all — a straight win over walking the whole archive.
struct BlockPlan {
    block: usize,
    /// The block's first *file* index (`block_first_file_index`), which is where the
    /// running counter in the decode loop starts.
    first: usize,
    /// The largest tracked file index in the block: once it has been read, nothing
    /// else in this block is wanted and the walk can stop early.
    last_tracked: usize,
    /// Best (lowest) `rank` over the block's pages — the sort key.
    key: u64,
}

/// Reader thread: decode every block that holds a tracked page, nearest-to-`start`
/// first, into the map.
fn extract_all(
    path: PathBuf,
    archive: Archive,
    file_to_page: Vec<Option<usize>>,
    start: usize,
    shared: Arc<Shared>,
) {
    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => return finish(&shared, Some(e.to_string())),
    };
    // No password, exactly as the previous `ArchiveReader::open(path, Password::empty())`
    // did — an encrypted-header archive fails identically to before, at `Archive::open`.
    let password = Password::empty();

    // --- Plan: group tracked pages by block, then rank the blocks. ---
    let mut plans: Vec<BlockPlan> = Vec::new();
    let mut plan_of_block: HashMap<usize, usize> = HashMap::new();
    // Pages whose entry belongs to no block at all: 7z keeps zero-length files
    // outside the packed streams, so they never surface in a block walk.
    let mut empty_pages: Vec<usize> = Vec::new();
    for (fi, blk) in archive.stream_map.file_block_index.iter().enumerate() {
        let Some(page) = file_to_page.get(fi).copied().flatten() else {
            continue;
        };
        let Some(b) = *blk else {
            empty_pages.push(page);
            continue;
        };
        let key = rank(page, start);
        // `file_block_index` is walked in ascending file order, so the last tracked
        // index seen for a block is its largest.
        match plan_of_block.get(&b).copied() {
            Some(pi) => {
                plans[pi].key = plans[pi].key.min(key);
                plans[pi].last_tracked = fi;
            }
            None => {
                plan_of_block.insert(b, plans.len());
                plans.push(BlockPlan {
                    block: b,
                    first: archive.stream_map.block_first_file_index[b],
                    last_tracked: fi,
                    key,
                });
            }
        }
    }
    // Block index breaks ties, so the order is deterministic and archive-biased.
    plans.sort_by_key(|p| (p.key, p.block));

    for plan in &plans {
        // Checked per block, because `BlockDecoder::for_each_entries` reports its own
        // `Ok(false)` and nothing above it used to act on that: an abort must end the
        // whole extraction, not just the block it happened to land in.
        if shared.abort.load(Ordering::Relaxed) {
            return finish(&shared, None);
        }
        // Running file index within the block. `for_each_entries` invokes the closure
        // exactly once per file in `first .. first + count`, in order, including
        // zero-size entries — so this counter stays in lockstep with it. Counting is
        // also how the count is obtained at all: `Block::num_unpack_sub_streams` is
        // `pub(crate)`.
        let mut fi = plan.first;
        let dec = BlockDecoder::new(1, plan.block, &archive, &password, &mut file);
        let walked = dec.for_each_entries(&mut |entry, rd| {
            if shared.abort.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let this = fi;
            fi += 1;
            match file_to_page.get(this).copied().flatten() {
                Some(page) => {
                    // Exact prealloc from the header (capped, so a bogus size can't ask
                    // for a wild allocation) — a multi-MB page otherwise regrows a
                    // dozen times. CRC verification still applies: `BlockDecoder` wraps
                    // tracked reads in a verifying reader, so a mismatch surfaces
                    // through this `?` and stops everything, as it always did.
                    let mut buf = Vec::with_capacity(entry.size.min(1 << 31) as usize);
                    std::io::Read::read_to_end(rd, &mut buf)?;
                    {
                        let mut g = shared.ready.lock().unwrap();
                        g.map.insert(page, Arc::new(buf));
                    }
                    shared.cv.notify_all(); // send-then-wake
                    if this == plan.last_tracked {
                        // Everything wanted from this block is in the map. Safe to stop
                        // here *because the entry was fully read first*: the per-entry
                        // windows are created lazily, so the ones we never reach are
                        // never built and the block stream simply goes away with the
                        // decoder.
                        return Ok(false);
                    }
                }
                // An untracked entry inside a solid block still has to be consumed:
                // every entry is a window over one shared sequential stream, so leaving
                // its bytes unread would leave the stream mid-entry and corrupt every
                // page after it in the block.
                None => {
                    std::io::copy(rd, &mut std::io::sink())?;
                }
            }
            Ok(true)
        });
        if let Err(e) = walked {
            return finish(&shared, Some(e.to_string()));
        }
    }

    // Zero-length tracked entries: no block ever produces them, so fill them here or
    // `read_page` would park on them until `done`.
    if !empty_pages.is_empty() {
        {
            let mut g = shared.ready.lock().unwrap();
            for page in empty_pages {
                g.map.insert(page, Arc::new(Vec::new()));
            }
        }
        shared.cv.notify_all();
    }
    finish(&shared, None);
}

impl PageSource for SevenzSource {
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
                    .unwrap_or_else(|| format!("7z: page {index} not found"));
                return Err(io::Error::other(msg));
            }
            // Checked *after* the map hit, so a cancel never denies a page that is
            // already extracted — cancelling releases waiters, it doesn't close the
            // source.
            if self.shared.cancel_epoch.load(Ordering::Relaxed) != epoch {
                return Err(io::Error::other("7z: read cancelled"));
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

impl Drop for SevenzSource {
    fn drop(&mut self) {
        self.shared.abort.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sevenz_rust2::{ArchiveEntry, ArchiveWriter, SourceReader};
    use std::io::Cursor;

    /// Temp-file convention borrowed from `ziparc.rs` (no `tempfile` dep): the pid
    /// plus `tag` keep concurrent tests from colliding on one path.
    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("yosh_7z_{}_{tag}.7z", std::process::id()))
    }

    /// A distinct payload per entry, long enough to actually compress. Byte-exact
    /// assertions against these are what turn a desynced block stream into a test
    /// failure instead of a plausible-looking page.
    fn payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| seed.wrapping_add((i % 251) as u8))
            .collect()
    }

    /// Non-solid archive (`7z a -ms=off`): `push_archive_entry` closes a pack per
    /// entry, so every file gets its own block.
    fn write_7z_nonsolid(tag: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let path = tmp(tag);
        let mut w = ArchiveWriter::create(&path).unwrap();
        for (name, bytes) in files {
            w.push_archive_entry(
                ArchiveEntry::new_file(name),
                Some(Cursor::new(bytes.to_vec())),
            )
            .unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// Solid archive (`7z a -ms=on`): one block per group. `push_archive_entries`
    /// packs the whole group into a single compressed stream, which is what makes
    /// intra-block entry order load-bearing.
    fn write_7z_solid(tag: &str, blocks: &[&[(&str, &[u8])]]) -> PathBuf {
        let path = tmp(tag);
        let mut w = ArchiveWriter::create(&path).unwrap();
        for group in blocks {
            let entries: Vec<ArchiveEntry> = group
                .iter()
                .map(|(name, _)| ArchiveEntry::new_file(name))
                .collect();
            let readers: Vec<SourceReader<Cursor<Vec<u8>>>> = group
                .iter()
                .map(|(_, bytes)| SourceReader::from(Cursor::new(bytes.to_vec())))
                .collect();
            w.push_archive_entries(entries, readers).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// The baseline: one block per page, every page byte-exact.
    #[test]
    fn nonsolid_round_trips() {
        let pages: Vec<Vec<u8>> = (0..4).map(|i| payload(0x20 + i as u8, 4096 + i)).collect();
        let files: Vec<(&str, &[u8])> = vec![
            ("01.png", &pages[0]),
            ("02.png", &pages[1]),
            ("03.png", &pages[2]),
            ("04.png", &pages[3]),
        ];
        let path = write_7z_nonsolid("nonsolid", &files);
        let src = SevenzSource::new(&path).unwrap();
        assert_eq!(src.len(), 4);
        for (i, want) in pages.iter().enumerate() {
            assert_eq!(&**src.read_page(i).unwrap(), &want[..], "page {i}");
        }
        drop(src);
        let _ = std::fs::remove_file(&path);
    }

    /// **Regression test for the untracked-entry desync.** Inside a solid block every
    /// entry is a window over *one* shared sequential stream, so an entry this source
    /// doesn't track — a `notes.txt` sitting between two pages — still has to be read
    /// to its end. The old walk simply returned without touching its reader, which
    /// left the block stream parked mid-`notes.txt`, and every page after it in the
    /// block decoded from the wrong offset (verified red before the rewrite: page 1
    /// came back as `ChecksumVerificationFailed`).
    #[test]
    fn solid_block_with_interleaved_non_image_stays_in_sync() {
        let p1 = payload(0x11, 5000);
        let notes = payload(0xa5, 3000);
        let p2 = payload(0x77, 5000);
        let path = write_7z_solid(
            "solid_interleaved",
            &[&[("01.png", &p1), ("notes.txt", &notes), ("02.png", &p2)]],
        );
        // The whole point is that the three entries share one stream.
        assert_eq!(Archive::open(&path).unwrap().blocks.len(), 1);
        let src = SevenzSource::new(&path).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(src.name(0), "01.png");
        assert_eq!(src.name(1), "02.png");
        assert_eq!(&**src.read_page(0).unwrap(), &p1[..]);
        assert_eq!(&**src.read_page(1).unwrap(), &p2[..]);
        drop(src);
        let _ = std::fs::remove_file(&path);
    }

    /// A resume hint decodes the *second* block first (page 4 lives there), so this
    /// pins both halves of the reordering: the pages still all arrive, and they all
    /// arrive intact — a block walked out of archive order must not depend on the one
    /// before it.
    #[test]
    fn with_start_on_multiblock_solid_round_trips_all_pages() {
        let pages: Vec<Vec<u8>> = (0..6).map(|i| payload(0x40 + i as u8, 3000 + i)).collect();
        let block_a: Vec<(&str, &[u8])> = vec![
            ("01.png", &pages[0]),
            ("02.png", &pages[1]),
            ("03.png", &pages[2]),
        ];
        let block_b: Vec<(&str, &[u8])> = vec![
            ("04.png", &pages[3]),
            ("05.png", &pages[4]),
            ("06.png", &pages[5]),
        ];
        let path = write_7z_solid("multiblock", &[&block_a, &block_b]);
        // Two real blocks, or the reordering under test never happens: with
        // `start = 4` the second block ranks 0 and the first ranks 7, so the walk
        // runs them back-to-front.
        assert_eq!(Archive::open(&path).unwrap().blocks.len(), 2);
        let src = SevenzSource::with_start(&path, Some(4)).unwrap();
        assert_eq!(src.len(), 6);
        for (i, want) in pages.iter().enumerate() {
            assert_eq!(&**src.read_page(i).unwrap(), &want[..], "page {i}");
        }
        drop(src);
        let _ = std::fs::remove_file(&path);
    }

    /// Page indices are the *natural-sorted* order, which is not archive order:
    /// `10.png` is stored first but reads second. The block walk indexes files by
    /// archive position, so this is the test that catches confusing the two.
    #[test]
    fn sorted_index_differs_from_archive_order() {
        let ten = payload(0x0a, 2500);
        let two = payload(0xb0, 2500);
        let path = write_7z_nonsolid("sortorder", &[("10.png", &ten), ("2.png", &two)]);
        let src = SevenzSource::new(&path).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(src.name(0), "2.png");
        assert_eq!(src.name(1), "10.png");
        assert_eq!(&**src.read_page(0).unwrap(), &two[..]);
        assert_eq!(&**src.read_page(1).unwrap(), &ten[..]);
        drop(src);
        let _ = std::fs::remove_file(&path);
    }
}

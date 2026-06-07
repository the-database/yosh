//! CBZ/ZIP source. Reads happen in parallel: each `read_page` uses its own
//! `ZipArchive` handle (the central directory is tiny for a comic), so worker
//! threads never share a cursor. Parsed handles are recycled through `pool`
//! instead of re-opening + re-parsing the archive on every read — that repeated
//! open + central-directory scan is a real cost over a network share.
//!
//! The archive can be backed by an on-disk path *or* an in-memory byte buffer
//! (e.g. bytes the shell read from an Android `content://` file descriptor, where
//! there is no path to reopen). The `Reader` enum unifies the two so the read +
//! pooling path is identical regardless of where the bytes live.

use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use zip::ZipArchive;

use super::{is_image_name, PageSource};

/// Idle parsed-archive handles kept for reuse. Bounded to roughly the decode
/// worker count — only idle handles are capped, so a drift from that just costs a
/// few extra short-lived opens, never correctness.
const POOL_CAP: usize = 8;

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// Where the archive bytes live.
enum Backend {
    /// On-disk file: each handle reopens it (a fresh `File` cursor per worker).
    Path(PathBuf),
    /// In-memory archive: each handle is a cheap `Cursor` over the shared buffer.
    Bytes(Arc<[u8]>),
}

/// A seekable reader over either backend, so the archive handle is one concrete
/// type (`ZipArchive<Reader>`) and the pool/read path needn't be generic.
enum Reader {
    File(File),
    Mem(Cursor<Arc<[u8]>>),
}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Reader::File(f) => f.read(buf),
            Reader::Mem(c) => c.read(buf),
        }
    }
}

impl Seek for Reader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Reader::File(f) => f.seek(pos),
            Reader::Mem(c) => c.seek(pos),
        }
    }
}

pub struct ZipSource {
    backend: Backend,
    names: Vec<String>,
    /// Recycled parsed handles (each owns its own cursor over the archive).
    pool: Mutex<Vec<ZipArchive<Reader>>>,
}

impl ZipSource {
    /// Open an on-disk `.cbz` / `.zip`.
    pub fn new(path: &Path) -> io::Result<Self> {
        Self::build(Backend::Path(path.to_path_buf()))
    }

    /// Open an archive already held in memory — e.g. bytes read from an Android
    /// `content://` file descriptor, where there is no filesystem path to reopen.
    pub fn from_bytes(bytes: Vec<u8>) -> io::Result<Self> {
        Self::build(Backend::Bytes(bytes.into()))
    }

    fn build(backend: Backend) -> io::Result<Self> {
        // `file_names()` reads straight from the in-memory central directory that
        // `ZipArchive::new` already parsed — no per-entry seek/local-header read. A
        // directory entry's name ends in `/`, so `is_image_name` rejects it; the
        // explicit `/` guard is belt-and-suspenders for an oddly-named dir.
        let names = {
            let zip = ZipArchive::new(Self::fresh(&backend)?).map_err(to_io)?;
            let mut names: Vec<String> = zip
                .file_names()
                .filter(|n| !n.ends_with('/') && is_image_name(n))
                .map(|n| n.to_string())
                .collect();
            names.sort_by(|a, b| natord::compare(a, b));
            names
        };
        Ok(Self {
            backend,
            names,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// A fresh, independent reader over the archive: a new file handle, or a new
    /// cursor sharing the in-memory buffer (a cheap `Arc` clone).
    fn fresh(backend: &Backend) -> io::Result<Reader> {
        match backend {
            Backend::Path(p) => Ok(Reader::File(File::open(p)?)),
            Backend::Bytes(b) => Ok(Reader::Mem(Cursor::new(b.clone()))),
        }
    }

    /// Take a parsed handle from the pool, or open + parse a fresh one.
    fn checkout(&self) -> io::Result<ZipArchive<Reader>> {
        let pooled = match self.pool.lock() {
            Ok(mut p) => p.pop(),
            Err(e) => e.into_inner().pop(), // tolerate a poisoned lock
        };
        match pooled {
            Some(zip) => Ok(zip),
            None => ZipArchive::new(Self::fresh(&self.backend)?).map_err(to_io),
        }
    }

    /// Return a handle to the pool for reuse, unless it's already full.
    fn checkin(&self, zip: ZipArchive<Reader>) {
        if let Ok(mut p) = self.pool.lock()
            && p.len() < POOL_CAP
        {
            p.push(zip);
        }
        // else: drop it (closes the File / releases the cursor).
    }
}

impl PageSource for ZipSource {
    fn len(&self) -> usize {
        self.names.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn read_page(&self, index: usize) -> io::Result<Vec<u8>> {
        let name = &self.names[index];
        let mut zip = self.checkout()?;
        // Confine the `ZipFile` borrow to this block: `read_to_end` fully drains
        // the bytes into an owned Vec, then `entry` drops — only then is `zip`
        // free to move back into the pool.
        let result = (|| -> io::Result<Vec<u8>> {
            let mut entry = zip.by_name(name).map_err(to_io)?;
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            Ok(buf)
        })();
        // Only recycle a handle whose read succeeded; an errored read may have
        // left the reader at an unexpected offset, so drop it and reopen next time.
        if result.is_ok() {
            self.checkin(zip);
        }
        result
    }

    fn modified(&self, index: usize) -> Option<String> {
        // Lazy: only the Tab info overlay asks, and it already reads the full page
        // there, so a one-off open for the timestamp is cheap. The central
        // directory carries last-modified, but the zip crate only surfaces it via
        // a `ZipFile`, so we open by name on demand rather than eagerly at `new`.
        let mut zip = ZipArchive::new(Self::fresh(&self.backend).ok()?).ok()?;
        let f = zip.by_name(self.names.get(index)?).ok()?;
        f.last_modified().map(|dt| {
            format!(
                "{:04}-{:02}-{:02} {:02}:{:02}",
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::PageSource;
    use std::io::Write as _;
    use std::sync::Arc;
    use zip::write::SimpleFileOptions;
    use zip::DateTime;

    /// Build a temp zip from `(name, bytes)` entries plus any directory entries,
    /// returning its path. `tag` keeps concurrent tests from colliding.
    fn write_zip(tag: &str, files: &[(&str, &[u8])], dirs: &[&str]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("yosh_zip_{}_{}.zip", std::process::id(), tag));
        let f = File::create(&path).unwrap();
        let mut w = zip::ZipWriter::new(f);
        let opts = SimpleFileOptions::default();
        for d in dirs {
            w.add_directory(*d, opts).unwrap();
        }
        for (name, bytes) in files {
            w.start_file(*name, opts).unwrap();
            w.write_all(bytes).unwrap();
        }
        w.finish().unwrap();
        path
    }

    #[test]
    fn lists_image_names_sorted_excluding_nonimages() {
        let path = write_zip(
            "list",
            &[
                ("02.jpg", b"jpg"),
                ("01.png", b"png"),
                ("notes.txt", b"txt"),
            ],
            &["sub/"],
        );
        let src = ZipSource::new(&path).unwrap();
        assert_eq!(src.len(), 2, "txt + directory excluded");
        assert_eq!(src.name(0), "01.png");
        assert_eq!(src.name(1), "02.jpg");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn natural_sort_order() {
        let path = write_zip(
            "natsort",
            &[("10.png", b"a"), ("2.png", b"b"), ("1.png", b"c")],
            &[],
        );
        let src = ZipSource::new(&path).unwrap();
        let names: Vec<&str> = (0..src.len()).map(|i| src.name(i)).collect();
        assert_eq!(names, ["1.png", "2.png", "10.png"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_bytes_reads_in_memory_archive() {
        // Build a zip entirely in memory (no temp file) and open it via
        // `from_bytes` — the path an Android `content://` FD takes (the shell
        // reads the descriptor into bytes, then hands them here).
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = SimpleFileOptions::default();
            w.start_file("01.png", opts).unwrap();
            w.write_all(b"PNGDATA-ONE").unwrap();
            w.start_file("02.jpg", opts).unwrap();
            w.write_all(b"JPGDATA-TWO").unwrap();
            w.add_directory("sub/", opts).unwrap();
            w.finish().unwrap();
        }
        let src = ZipSource::from_bytes(buf).unwrap();
        assert_eq!(src.len(), 2, "txt/dir excluded, 2 images");
        assert_eq!(src.name(0), "01.png");
        // Repeated reads exercise the in-memory handle pool.
        for _ in 0..10 {
            assert_eq!(src.read_page(0).unwrap(), b"PNGDATA-ONE");
            assert_eq!(src.read_page(1).unwrap(), b"JPGDATA-TWO");
        }
    }

    #[test]
    fn read_page_roundtrips_and_reuses_handles() {
        let path = write_zip(
            "read",
            &[("01.png", b"PNGDATA-ONE"), ("02.jpg", b"JPGDATA-TWO")],
            &[],
        );
        let src = ZipSource::new(&path).unwrap();
        // Repeated reads exercise the checkout/checkin handle pool.
        for _ in 0..20 {
            assert_eq!(src.read_page(0).unwrap(), b"PNGDATA-ONE");
            assert_eq!(src.read_page(1).unwrap(), b"JPGDATA-TWO");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn concurrent_reads_share_pool_safely() {
        let path = write_zip(
            "concurrent",
            &[("01.png", b"PNGDATA-ONE"), ("02.jpg", b"JPGDATA-TWO")],
            &[],
        );
        let src = Arc::new(ZipSource::new(&path).unwrap());
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let src = src.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let want: &[u8] = if t % 2 == 0 { b"PNGDATA-ONE" } else { b"JPGDATA-TWO" };
                        assert_eq!(src.read_page(t % 2).unwrap(), want);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn modified_is_formatted() {
        let path = std::env::temp_dir()
            .join(format!("yosh_zip_{}_modified.zip", std::process::id()));
        {
            let f = File::create(&path).unwrap();
            let mut w = zip::ZipWriter::new(f);
            let opts = SimpleFileOptions::default()
                .last_modified_time(DateTime::from_date_and_time(2021, 3, 14, 9, 26, 0).unwrap());
            w.start_file("01.png", opts).unwrap();
            w.write_all(b"png").unwrap();
            w.finish().unwrap();
        }
        let src = ZipSource::new(&path).unwrap();
        assert_eq!(src.modified(0).as_deref(), Some("2021-03-14 09:26"));
        let _ = std::fs::remove_file(&path);
    }
}

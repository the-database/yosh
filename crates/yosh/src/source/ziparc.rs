//! CBZ/ZIP source. Reads happen in parallel: each `read_page` uses its own
//! `ZipArchive` handle (the central directory is tiny for a comic), so worker
//! threads never share a cursor. Parsed handles are recycled through `pool`
//! instead of re-opening + re-parsing the archive on every read — that repeated
//! `File::open` + central-directory scan is a real cost over a network share.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use zip::ZipArchive;

use super::{is_image_name, PageSource};

/// Idle parsed-archive handles kept for reuse. Bounded to roughly the decode
/// worker count (`WORKERS` in app.rs) — only idle handles are capped, so a drift
/// from that constant just costs a few extra short-lived opens, never correctness.
const POOL_CAP: usize = 8;

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

pub struct ZipSource {
    path: PathBuf,
    names: Vec<String>,
    /// Recycled `ZipArchive<File>` handles (each owns its own `File` cursor).
    pool: Mutex<Vec<ZipArchive<File>>>,
}

impl ZipSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        // `file_names()` reads straight from the in-memory central directory that
        // `ZipArchive::new` already parsed — no per-entry seek/local-header read
        // (which, over a network share, would be one round-trip per page). A
        // directory entry's name ends in `/`, so `is_image_name` rejects it; the
        // explicit `/` guard is belt-and-suspenders for an oddly-named dir.
        let zip = ZipArchive::new(File::open(path)?).map_err(to_io)?;
        let mut names: Vec<String> = zip
            .file_names()
            .filter(|n| !n.ends_with('/') && is_image_name(n))
            .map(|n| n.to_string())
            .collect();
        names.sort_by(|a, b| natord::compare(a, b));
        Ok(Self {
            path: path.to_path_buf(),
            names,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// Take a parsed handle from the pool, or open + parse a fresh one.
    fn checkout(&self) -> io::Result<ZipArchive<File>> {
        let pooled = match self.pool.lock() {
            Ok(mut p) => p.pop(),
            Err(e) => e.into_inner().pop(), // tolerate a poisoned lock
        };
        match pooled {
            Some(zip) => Ok(zip),
            None => ZipArchive::new(File::open(&self.path)?).map_err(to_io),
        }
    }

    /// Return a handle to the pool for reuse, unless it's already full.
    fn checkin(&self, zip: ZipArchive<File>) {
        if let Ok(mut p) = self.pool.lock()
            && p.len() < POOL_CAP
        {
            p.push(zip);
        }
        // else: drop it (closes the File).
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
        let mut zip = ZipArchive::new(File::open(&self.path).ok()?).ok()?;
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

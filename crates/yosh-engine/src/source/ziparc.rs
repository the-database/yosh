//! CBZ/ZIP source. Reads happen in parallel: each `read_page` uses its own
//! `ZipArchive` handle (the central directory is tiny for a comic), so worker
//! threads never share a cursor. Parsed handles are recycled through `pool`
//! instead of re-opening + re-parsing the archive on every read — that repeated
//! open + central-directory scan is a real cost over a network share; the handle
//! `build` already parsed seeds the pool, so the very first worker read reuses it
//! rather than paying that cost once more before the pool has anything in it.
//!
//! Pages are addressed by *archive index*, not by name — `zip`'s name lookup cannot
//! round-trip an entry whose name isn't ASCII/UTF-8 (see `ZipSource::list_central`),
//! and legacy (Shift-JIS/GBK) entry names are decoded by us for display and sorting.
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

use super::{decode_entry_name, detect_legacy_encoding, is_image_name, PageSource};

/// Idle parsed-archive handles kept for reuse. Bounded to roughly the decode
/// worker count — only idle handles are capped, so a drift from that just costs a
/// few extra short-lived opens, never correctness.
const POOL_CAP: usize = 8;

/// Ceiling on the read buffer we pre-allocate from an entry's *declared* uncompressed
/// size. A corrupt or hostile header can claim a huge size, and `with_capacity` would
/// commit that much before a single byte is read; past this we just let the vec grow.
const MAX_PREALLOC: usize = 256 << 20; // 256 MiB

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

/// How a page's bytes are located. A normal archive resolves names through the
/// parsed central directory; a truncated/damaged archive that has no central
/// directory is recovered by scanning local file headers from the front, recording
/// each surviving entry's header offset for direct seek+read.
enum Index {
    /// Archive entry index per page, parallel to `names`. Pages are addressed
    /// positionally, never by name — see `list_central`.
    Central(Vec<usize>),
    /// Local-file-header byte offset per entry, parallel to `names`.
    Local(Vec<u64>),
}

pub struct ZipSource {
    backend: Backend,
    names: Vec<String>,
    index: Index,
    /// True when opened by local-header recovery (central directory missing): the
    /// archive is partial, so `names` holds only the pages that survived the cutoff.
    partial: bool,
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
        match ZipArchive::new(Self::fresh(&backend)?) {
            Ok(mut zip) => {
                let (names, idx) = Self::list_central(&mut zip);
                Ok(Self {
                    backend,
                    names,
                    index: Index::Central(idx),
                    partial: false,
                    // Seed the pool with the handle we just parsed instead of dropping
                    // it: the first decode worker checks this one out rather than
                    // paying a second `File::open` + central-directory parse (a real
                    // cost on a network share, right when the first page is wanted).
                    // Where `list_central` left the cursor is irrelevant — `read_page`
                    // seeks via `by_index`. Helps `Backend::Bytes` (Android
                    // `from_bytes`) too, where the re-parse is the avoidable part.
                    pool: Mutex::new(vec![zip]),
                })
            }
            // No usable central directory (truncated / damaged download, sync cut
            // off mid-transfer): recover whatever complete pages exist by walking the
            // local file headers from the front, BandiView-style.
            Err(central_err) => {
                let mut entries = Self::scan_local(&backend)?; // (name, local-header offset)
                if entries.is_empty() {
                    // Not a recoverable zip at all — report the original failure.
                    return Err(to_io(central_err));
                }
                entries.sort_by(|a, b| natord::compare(&a.0, &b.0));
                let (names, offsets): (Vec<String>, Vec<u64>) = entries.into_iter().unzip();
                Ok(Self {
                    backend,
                    names,
                    index: Index::Local(offsets),
                    partial: true,
                    // Recovered archives read straight from the local-header offsets,
                    // never through a parsed handle — nothing to seed the pool with.
                    pool: Mutex::new(Vec::new()),
                })
            }
        }
    }

    /// List a parsed archive's image entries as `(display name, archive index)`,
    /// naturally sorted by name.
    ///
    /// **Pages are addressed by index, never by name.** `zip` keys its entry map by the
    /// *raw* header bytes but `file_names()` hands back a decoded string, and `by_name`
    /// re-encodes that string as UTF-8 to look it up. When general-purpose bit 11 (the
    /// UTF-8 flag) is clear and the bytes aren't ASCII, the decoded name is a CP437
    /// transliteration whose UTF-8 form never matches the raw key — e.g. Shift-JIS
    /// `81 45` renders as `üE`, which re-encodes to `c3 bc 45`. So the round-trip
    /// `by_name(file_names()[i])` fails for *every* entry of a legacy-encoded archive
    /// ("specified file not found in archive"), which is exactly how a Japanese CBZ used
    /// to open with a full page list and then fail on every single page. `by_index`
    /// sidesteps the name map entirely.
    ///
    /// A dir entry's name ends in `/`, so `is_image_name` rejects it; the explicit `/`
    /// guard is belt-and-suspenders for an oddly-named dir.
    ///
    /// For the *display* name we want the real characters, not the transliteration. That
    /// same keyed-by-raw-bytes map gives us an exact, allocation-free test for which
    /// names need help: `index_for_name(n)` hashes `n`'s UTF-8 bytes against the raw
    /// keys, so a name that maps back to its own entry is already byte-exact (ASCII, or
    /// a properly UTF-8-flagged name) and is kept as-is. Only a name that *fails* that
    /// round-trip is a CP437 transliteration, and only for those do we re-read the raw
    /// header bytes and decode them ourselves — so Shift-JIS/GBK pages display and sort
    /// correctly. Both `file_names()` and the round-trip test read from the already-parsed
    /// in-memory central directory, so ASCII and UTF-8-flagged archives (the overwhelming
    /// majority, including Japanese ones written by modern zippers) pay nothing at all;
    /// only a genuinely legacy-encoded archive pays a local-header seek per entry
    /// (~1.5 ms for 500 entries).
    fn list_central(zip: &mut ZipArchive<Reader>) -> (Vec<String>, Vec<usize>) {
        let decoded: Vec<String> = zip.file_names().map(str::to_string).collect();
        // Compare against `Some(i)`, not just `is_some()`: a pathological archive holding
        // both the raw and the transliterated spelling of one name would otherwise let an
        // entry validate against its neighbour.
        let exact: Vec<bool> = decoded
            .iter()
            .enumerate()
            .map(|(i, n)| zip.index_for_name(n) == Some(i))
            .collect();
        // Image filtering can run on the transliterated name: CP437 maps ASCII to itself,
        // and no legacy trail byte is `.` (Shift-JIS trail bytes start at 0x40), so the
        // extension survives transliteration intact. Filtering first also keeps the raw
        // re-read below down to the pages we actually intend to show.
        let keep: Vec<usize> = (0..decoded.len())
            .filter(|&i| !decoded[i].ends_with('/') && is_image_name(&decoded[i]))
            .collect();

        // Only names that failed the round-trip need their raw bytes re-read. A raw read
        // that fails just leaves that entry on the crate's own decoding — one bad entry
        // shouldn't cost us the rest of the archive.
        let (idxs, raws): (Vec<usize>, Vec<Vec<u8>>) = keep
            .iter()
            .copied()
            .filter(|&i| !exact[i])
            .filter_map(|i| zip.by_index_raw(i).ok().map(|e| (i, e.name_raw().to_vec())))
            .unzip();
        let enc = detect_legacy_encoding(&raws);
        let mut legacy: Vec<Option<String>> = vec![None; decoded.len()];
        for (i, raw) in idxs.into_iter().zip(&raws) {
            legacy[i] = Some(decode_entry_name(raw, enc));
        }

        let mut entries: Vec<(String, usize)> = keep
            .into_iter()
            .map(|i| (legacy[i].take().unwrap_or_else(|| decoded[i].clone()), i))
            .collect();
        entries.sort_by(|a, b| natord::compare(&a.0, &b.0));
        entries.into_iter().unzip()
    }

    /// Recover an archive with no central directory by reading local file headers
    /// sequentially from the front. Returns `(image name, header byte offset)` for
    /// every *complete* entry, stopping at the first truncated/garbage header or
    /// truncated entry data — so a download cut off mid-page yields the pages before
    /// the cut. Each entry is drained (not just dropped) to advance to the next
    /// header and to detect a half-written final entry.
    fn scan_local(backend: &Backend) -> io::Result<Vec<(String, u64)>> {
        let mut reader = Self::fresh(backend)?;
        let mut entries = Vec::new();
        while let Ok(off) = reader.stream_position() {
            match zip::read::read_zipfile_from_stream(&mut reader) {
                Ok(Some(mut file)) => {
                    let name = file.name().to_string();
                    let keep = !name.ends_with('/') && is_image_name(&name);
                    // Drain past the data to reach the next header; a truncated final
                    // entry errors here → stop without recording the incomplete page.
                    if io::copy(&mut file, &mut io::sink()).is_err() {
                        break;
                    }
                    if keep {
                        entries.push((name, off));
                    }
                }
                Ok(None) => break, // clean end of the local-header stream
                Err(_) => break,   // truncated/garbage header → keep what we have
            }
        }
        Ok(entries)
    }

    /// Read one entry from a recovered (`Index::Local`) archive: seek to its local
    /// header and let the streaming reader parse + decompress just that entry. A
    /// fresh reader per call keeps parallel reads independent (no shared cursor).
    fn read_local(&self, offset: u64) -> io::Result<Vec<u8>> {
        let mut reader = Self::fresh(&self.backend)?;
        reader.seek(SeekFrom::Start(offset))?;
        match zip::read::read_zipfile_from_stream(&mut reader).map_err(to_io)? {
            Some(mut entry) => {
                let mut buf = Vec::with_capacity((entry.size() as usize).min(MAX_PREALLOC));
                entry.read_to_end(&mut buf)?;
                Ok(buf)
            }
            None => Err(io::Error::other("no entry at recovered offset")),
        }
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

    fn read_page(&self, index: usize) -> io::Result<Arc<Vec<u8>>> {
        // Recovered archive: read directly from the entry's local-header offset.
        let entry_idx = match &self.index {
            Index::Local(offsets) => return self.read_local(offsets[index]).map(Arc::new),
            Index::Central(idx) => idx[index],
        };
        let mut zip = self.checkout()?;
        // Confine the `ZipFile` borrow to this block: `read_to_end` fully drains
        // the bytes into an owned Vec, then `entry` drops — only then is `zip`
        // free to move back into the pool.
        let result = (|| -> io::Result<Vec<u8>> {
            let mut entry = zip.by_index(entry_idx).map_err(to_io)?;
            let mut buf = Vec::with_capacity((entry.size() as usize).min(MAX_PREALLOC));
            entry.read_to_end(&mut buf)?;
            Ok(buf)
        })();
        // Only recycle a handle whose read succeeded; an errored read may have
        // left the reader at an unexpected offset, so drop it and reopen next time.
        if result.is_ok() {
            self.checkin(zip);
        }
        result.map(Arc::new)
    }

    fn modified(&self, index: usize) -> Option<String> {
        // Lazy: only the Tab info overlay asks, and it already reads the full page
        // there, so a one-off open for the timestamp is cheap. Both paths surface
        // last-modified via a `ZipFile`: by archive index through the central directory,
        // or — for a recovered archive — by seeking to the entry's local header.
        let dt = match &self.index {
            Index::Central(idx) => {
                let mut zip = ZipArchive::new(Self::fresh(&self.backend).ok()?).ok()?;
                zip.by_index(*idx.get(index)?).ok()?.last_modified()
            }
            Index::Local(offsets) => {
                let mut reader = Self::fresh(&self.backend).ok()?;
                reader.seek(SeekFrom::Start(*offsets.get(index)?)).ok()?;
                zip::read::read_zipfile_from_stream(&mut reader).ok()??.last_modified()
            }
        };
        dt.map(|dt| {
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

    fn is_partial(&self) -> bool {
        self.partial
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

    /// Hand-build a stored-entry zip with **raw** name bytes and general-purpose flag
    /// 0 (no UTF-8 bit). `ZipWriter` only accepts `&str` names, so a legacy-codepage
    /// archive — the exact shape that used to break every page read — can't be produced
    /// with it; we emit the local headers, central directory and EOCD directly.
    fn raw_name_zip(entries: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut central: Vec<u8> = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            let crc = crc32fast::hash(data);
            let (n, sz) = (name.len() as u16, data.len() as u32);
            // Local file header: stored, no extra field, sizes known up front.
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0]); // version, flags=0, method=0 (stored)
            out.extend_from_slice(&[0, 0, 0x21, 0]); // mod time / date
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&sz.to_le_bytes()); // compressed
            out.extend_from_slice(&sz.to_le_bytes()); // uncompressed
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name);
            out.extend_from_slice(data);

            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0]); // made-by, needed, flags=0, method=0
            central.extend_from_slice(&[0, 0, 0x21, 0]);
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&sz.to_le_bytes());
            central.extend_from_slice(&sz.to_le_bytes());
            central.extend_from_slice(&n.to_le_bytes());
            central.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // extra, comment, disk
            central.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // internal + external attrs
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name);
        }
        let (cd_off, cd_len) = (out.len() as u32, central.len() as u32);
        let count = entries.len() as u16;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }

    /// Regression: a Shift-JIS-named archive (UTF-8 flag clear) must open *and* read.
    ///
    /// `zip` keys entries by raw name bytes but returns a CP437 transliteration, so the
    /// old `by_name(file_names()[i])` round-trip failed on every page with "specified
    /// file not found in archive" — the archive listed all its pages and then rendered
    /// none of them. Reading by index fixes it, and the raw bytes are decoded for display.
    #[test]
    fn reads_shift_jis_named_entries() {
        // "週刊/001.jpg" in CP932: 8f 54 8a a7 = 週刊
        let name: &[u8] = &[
            0x8f, 0x54, 0x8a, 0xa7, b'/', b'0', b'0', b'1', b'.', b'j', b'p', b'g',
        ];
        let bytes = raw_name_zip(&[(name, b"SJIS-PAGE-ONE")]);
        let src = ZipSource::from_bytes(bytes).unwrap();

        assert_eq!(src.len(), 1, "the .jpg entry must survive filtering");
        assert_eq!(
            *src.read_page(0).unwrap(),
            b"SJIS-PAGE-ONE",
            "reading a legacy-encoded entry must not fail"
        );
        assert_eq!(
            src.name(0),
            "週刊/001.jpg",
            "legacy name decoded for display, not CP437 mojibake"
        );
    }

    /// Reading is encoding-*agnostic*: pages are addressed by index, so an archive whose
    /// names don't round-trip must still read no matter what codepage produced them.
    /// Display names are best-effort detection on top of that — this pins the encodings
    /// that resolve correctly, and `reads` is the part that must never regress.
    #[test]
    fn reads_and_names_legacy_encodings() {
        let cases: &[(&'static encoding_rs::Encoding, &str)] = &[
            (encoding_rs::SHIFT_JIS, "週刊少年ジャンプ 2026年37・38号/001.jpg"),
            (encoding_rs::EUC_JP, "週刊少年ジャンプ/001.jpg"),
            (encoding_rs::GBK, "海贼王 第100话/001.jpg"),
            (encoding_rs::BIG5, "海賊王 第100話/001.jpg"),
            (encoding_rs::EUC_KR, "원피스 100화/001.jpg"),
            (encoding_rs::WINDOWS_1251, "Наруто том 5/001.jpg"),
            (encoding_rs::WINDOWS_1252, "café résumé/001.jpg"),
        ];
        for (enc, want) in cases {
            let raw = enc.encode(want).0.into_owned();
            let src = ZipSource::from_bytes(raw_name_zip(&[(&raw, b"PAGE")])).unwrap();
            assert_eq!(src.len(), 1, "{}: entry must survive filtering", enc.name());
            assert_eq!(
                *src.read_page(0).unwrap(),
                b"PAGE",
                "{}: reads must not depend on the name encoding",
                enc.name()
            );
            assert_eq!(src.name(0), *want, "{}: name decoded", enc.name());
        }
    }

    /// A properly UTF-8-flagged non-ASCII archive is already byte-exact, so it must be
    /// passed through untouched — never re-sniffed (which could mangle e.g. `café.jpg`).
    #[test]
    fn flagged_utf8_names_pass_through_unchanged() {
        let path = write_zip("utf8flag", &[("第01話/café.png", b"FLAGGED")], &[]);
        let src = ZipSource::new(&path).unwrap();
        assert_eq!(src.len(), 1);
        assert_eq!(src.name(0), "第01話/café.png");
        assert_eq!(*src.read_page(0).unwrap(), b"FLAGGED");
    }

    /// A UTF-8 name stored *without* the UTF-8 flag (very common) round-trips as UTF-8
    /// rather than being mis-sniffed as some legacy codepage.
    #[test]
    fn reads_unflagged_utf8_names() {
        let bytes = raw_name_zip(&[("表紙.png".as_bytes(), b"COVER")]);
        let src = ZipSource::from_bytes(bytes).unwrap();
        assert_eq!(src.len(), 1);
        assert_eq!(src.name(0), "表紙.png");
        assert_eq!(*src.read_page(0).unwrap(), b"COVER");
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
            assert_eq!(*src.read_page(0).unwrap(), b"PNGDATA-ONE");
            assert_eq!(*src.read_page(1).unwrap(), b"JPGDATA-TWO");
        }
    }

    /// `build` must seed the handle pool with the archive it just parsed, so the
    /// first worker read reuses it instead of re-opening + re-parsing.
    #[test]
    fn build_seeds_handle_pool() {
        let path = write_zip("seedpool", &[("01.png", b"PNGDATA-ONE")], &[]);
        let src = ZipSource::new(&path).unwrap();
        assert_eq!(
            src.pool.lock().unwrap().len(),
            1,
            "build-time handle recycled into the pool"
        );
        assert_eq!(*src.read_page(0).unwrap(), b"PNGDATA-ONE");
        assert_eq!(
            src.pool.lock().unwrap().len(),
            1,
            "checkout/checkin round-trips the seeded handle"
        );
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
            assert_eq!(*src.read_page(0).unwrap(), b"PNGDATA-ONE");
            assert_eq!(*src.read_page(1).unwrap(), b"JPGDATA-TWO");
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
                        assert_eq!(*src.read_page(t % 2).unwrap(), want);
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

    #[test]
    fn recovers_pages_from_truncated_archive() {
        // A complete 3-image zip, stored (uncompressed) so the layout is simple.
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            for (n, d) in [
                ("01.png", b"PAGE-ONE".as_slice()),
                ("02.png", b"PAGE-TWO"),
                ("03.png", b"PAGE-THREE"),
            ] {
                w.start_file(n, opts).unwrap();
                w.write_all(d).unwrap();
            }
            w.finish().unwrap();
        }
        // A complete archive opens via the central directory — not partial.
        let full = ZipSource::from_bytes(buf.clone()).unwrap();
        assert_eq!(full.len(), 3);
        assert!(!full.is_partial());

        // Cut into the 3rd local file header (PK\x03\x04): pages 1 & 2 are complete,
        // page 3 and the central directory are gone — a download cut off mid-page.
        let sig = [0x50u8, 0x4B, 0x03, 0x04];
        let third = buf
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == sig)
            .map(|(i, _)| i)
            .nth(2)
            .unwrap();
        buf.truncate(third + 8);

        // Recovery: the two complete pages survive, read back correctly, partial=true.
        let part = ZipSource::from_bytes(buf).unwrap();
        assert!(part.is_partial(), "opened via local-header recovery");
        assert_eq!(part.len(), 2, "two complete pages before the cutoff");
        assert_eq!(part.name(0), "01.png");
        assert_eq!(*part.read_page(0).unwrap(), b"PAGE-ONE");
        assert_eq!(*part.read_page(1).unwrap(), b"PAGE-TWO");
    }
}

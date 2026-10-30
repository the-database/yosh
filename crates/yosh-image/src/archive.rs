//! Whole-archive access in **archive order** — every entry (pages, `ComicInfo.xml`,
//! directories, …) with its raw bytes, for tools that rewrite an archive
//! faithfully. [`crate::source`] is the reader's view instead: image pages only,
//! natural-sorted, random access.
//!
//! - [`list`]: every entry's name, kind and size, without extracting anything.
//! - [`for_each`]: every entry with its bytes, streamed in order — one pass, so
//!   the sequential formats (RAR, 7z) cost a single decompression.
//! - [`walk_folder`]: a folder's files, natural-sorted.
//!
//! Zip entry names are repaired from legacy codepages (Shift-JIS, GBK, …) exactly
//! as the reader does ([`crate::source::decode_entry_name`]); RAR and 7z store
//! Unicode names. Separators are normalized to `/`.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::source::{decode_entry_name, detect_legacy_encoding};

/// One archive entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Path inside the archive, `/`-separated (directories without a trailing `/`).
    pub name: String,
    pub is_dir: bool,
    /// Uncompressed size in bytes (0 for directories).
    pub size: u64,
}

/// Archive formats this module reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Zip,
    /// RAR 4/5 (`rar` feature).
    Rar,
    SevenZ,
}

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// The archive kind of `path`: by magic bytes, else by extension
/// (`zip`/`cbz`, `rar`/`cbr`, `7z`/`cb7`). `None` if it's none of them.
pub fn kind(path: &Path) -> Option<Kind> {
    let mut head = [0u8; 6];
    if let Ok(mut f) = std::fs::File::open(path)
        && f.read_exact(&mut head).is_ok()
    {
        if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
            return Some(Kind::Zip);
        }
        if head.starts_with(b"Rar!\x1a\x07") {
            return Some(Kind::Rar);
        }
        if head.starts_with(&[b'7', b'z', 0xBC, 0xAF, 0x27, 0x1C]) {
            return Some(Kind::SevenZ);
        }
    }
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "zip" | "cbz" => Some(Kind::Zip),
        "rar" | "cbr" => Some(Kind::Rar),
        "7z" | "cb7" => Some(Kind::SevenZ),
        _ => None,
    }
}

fn kind_or_err(path: &Path) -> io::Result<Kind> {
    kind(path).ok_or_else(|| {
        io::Error::new(io::ErrorKind::Unsupported, format!("not a zip/rar/7z archive: {}", path.display()))
    })
}

fn normalize(name: &str) -> String {
    let n = name.replace('\\', "/");
    n.trim_end_matches('/').to_string()
}

/// Every entry of the archive at `path`, in archive order.
pub fn list(path: &Path) -> io::Result<Vec<Entry>> {
    match kind_or_err(path)? {
        Kind::Zip => {
            let mut zip = zip::ZipArchive::new(std::fs::File::open(path)?).map_err(to_io)?;
            zip_entries(&mut zip)
        }
        Kind::Rar => rar_list(path),
        Kind::SevenZ => {
            let archive = sevenz_rust2::Archive::open(path).map_err(to_io)?;
            Ok(sevenz_entries(&archive))
        }
    }
}

/// Stream every entry of the archive at `path`, in archive order, with its bytes
/// (empty for directories). `f` returns `Ok(false)` to stop early; an error from
/// `f` stops the walk and is returned.
pub fn for_each(path: &Path, mut f: impl FnMut(&Entry, &[u8]) -> io::Result<bool>) -> io::Result<()> {
    match kind_or_err(path)? {
        Kind::Zip => zip_for_each(path, &mut f),
        Kind::Rar => rar_for_each(path, &mut f),
        Kind::SevenZ => sevenz_for_each(path, &mut f),
    }
}

// ---------------- zip ----------------

fn zip_entries<R: io::Read + io::Seek>(zip: &mut zip::ZipArchive<R>) -> io::Result<Vec<Entry>> {
    let raws: Vec<Vec<u8>> = zip.file_names_raw().map(<[u8]>::to_vec).collect();
    // One codepage verdict for the whole archive, from its non-UTF-8 names (see
    // `detect_legacy_encoding`) — the same repair the reader applies to pages.
    let legacy: Vec<Vec<u8>> =
        raws.iter().filter(|r| std::str::from_utf8(r).is_err()).map(|r| r.to_vec()).collect();
    let enc = detect_legacy_encoding(&legacy);
    raws.iter()
        .enumerate()
        .map(|(i, raw)| {
            let is_dir = raw.ends_with(b"/");
            // Sizes from the central directory; `by_index_raw` decompresses nothing.
            let size = if is_dir { 0 } else { zip.by_index_raw(i).map_err(to_io)?.size() };
            Ok(Entry { name: normalize(&decode_entry_name(raw, enc)), is_dir, size })
        })
        .collect()
}

fn zip_for_each(path: &Path, f: &mut dyn FnMut(&Entry, &[u8]) -> io::Result<bool>) -> io::Result<()> {
    let mut zip = zip::ZipArchive::new(std::fs::File::open(path)?).map_err(to_io)?;
    let entries = zip_entries(&mut zip)?;
    for (i, entry) in entries.iter().enumerate() {
        let mut file = zip.by_index(i).map_err(to_io)?;
        let mut buf = Vec::with_capacity(entry.size.min(1 << 31) as usize);
        if !entry.is_dir {
            file.read_to_end(&mut buf)?;
        }
        drop(file);
        if !f(entry, &buf)? {
            break;
        }
    }
    Ok(())
}

// ---------------- rar ----------------

#[cfg(feature = "rar")]
fn rar_list(path: &Path) -> io::Result<Vec<Entry>> {
    let listing = unrar::Archive::new(path).open_for_listing().map_err(to_io)?;
    let mut out = Vec::new();
    for header in listing {
        let h = header.map_err(to_io)?;
        out.push(Entry {
            name: normalize(&h.filename.to_string_lossy()),
            is_dir: h.is_directory(),
            size: if h.is_directory() { 0 } else { h.unpacked_size },
        });
    }
    Ok(out)
}

#[cfg(feature = "rar")]
fn rar_for_each(path: &Path, f: &mut dyn FnMut(&Entry, &[u8]) -> io::Result<bool>) -> io::Result<()> {
    let mut cursor = unrar::Archive::new(path).open_for_processing().map_err(to_io)?;
    while let Some(header) = cursor.read_header().map_err(to_io)? {
        let e = header.entry();
        let entry = Entry {
            name: normalize(&e.filename.to_string_lossy()),
            is_dir: e.is_directory(),
            size: if e.is_directory() { 0 } else { e.unpacked_size },
        };
        let (bytes, next) = if entry.is_dir {
            (Vec::new(), header.skip().map_err(to_io)?)
        } else {
            header.read().map_err(to_io)?
        };
        if !f(&entry, &bytes)? {
            return Ok(());
        }
        cursor = next;
    }
    Ok(())
}

#[cfg(not(feature = "rar"))]
fn rar_list(_: &Path) -> io::Result<Vec<Entry>> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "built without RAR support (the `rar` feature)"))
}

#[cfg(not(feature = "rar"))]
fn rar_for_each(_: &Path, _: &mut dyn FnMut(&Entry, &[u8]) -> io::Result<bool>) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "built without RAR support (the `rar` feature)"))
}

// ---------------- 7z ----------------

fn sevenz_entries(archive: &sevenz_rust2::Archive) -> Vec<Entry> {
    archive
        .files
        .iter()
        .map(|e| Entry {
            name: normalize(e.name()),
            is_dir: e.is_directory(),
            size: if e.is_directory() { 0 } else { e.size() },
        })
        .collect()
}

/// 7z packs the data of files `first .. first + n` into block `b`, blocks in
/// ascending file order; directories and empty files have no data and belong to
/// no block. So: walk the blocks in order, emitting the data-less entries that
/// fall before each block's first file, then the block's own entries — which is
/// archive order, in one decompression pass.
fn sevenz_for_each(path: &Path, f: &mut dyn FnMut(&Entry, &[u8]) -> io::Result<bool>) -> io::Result<()> {
    use sevenz_rust2::{Archive, BlockDecoder, Password};
    let archive = Archive::open(path).map_err(to_io)?;
    let entries = sevenz_entries(&archive);
    let password = Password::empty();
    let mut file = std::fs::File::open(path)?;
    let blocks = &archive.stream_map.block_first_file_index;
    let in_block = &archive.stream_map.file_block_index;
    let mut next = 0usize; // next file index to emit
    // Emit data-less entries up to (excluding) `upto`; Ok(false) = stopped.
    let emit_empty = |upto: usize, next: &mut usize, f: &mut dyn FnMut(&Entry, &[u8]) -> io::Result<bool>| -> io::Result<bool> {
        while *next < upto {
            if in_block.get(*next).copied().flatten().is_none() && !f(&entries[*next], &[])? {
                return Ok(false);
            }
            *next += 1;
        }
        Ok(true)
    };
    for (b, &first) in blocks.iter().enumerate().take(archive.blocks.len()) {
        if !emit_empty(first, &mut next, f)? {
            return Ok(());
        }
        let mut stopped = false;
        let mut fi = first;
        let mut err: Option<io::Error> = None;
        let dec = BlockDecoder::new(1, b, &archive, &password, &mut file);
        dec.for_each_entries(&mut |_entry, rd| {
            let this = fi;
            fi += 1;
            let mut buf = Vec::with_capacity(entries[this].size.min(1 << 31) as usize);
            rd.read_to_end(&mut buf)?;
            match f(&entries[this], &buf) {
                Ok(true) => Ok(true),
                Ok(false) => {
                    stopped = true;
                    Ok(false)
                }
                Err(e) => {
                    err = Some(e);
                    Ok(false)
                }
            }
        })
        .map_err(to_io)?;
        if let Some(e) = err {
            return Err(e);
        }
        if stopped {
            return Ok(());
        }
        next = fi;
    }
    emit_empty(entries.len(), &mut next, f)?;
    Ok(())
}

// ---------------- folders ----------------

/// The files under `dir` (recursively if `recursive`), as paths, natural-sorted
/// by their path relative to `dir` (`page2` before `page10`; a subfolder's files
/// sort by their folder name first). Directories themselves aren't listed.
pub fn walk_folder(dir: &Path, recursive: bool) -> io::Result<Vec<PathBuf>> {
    fn walk(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let p = e.path();
            let ft = e.file_type()?;
            if ft.is_dir() || (ft.is_symlink() && p.is_dir()) {
                if recursive {
                    walk(&p, recursive, out)?;
                }
            } else {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, recursive, &mut out)?;
    let key = |p: &PathBuf| p.strip_prefix(dir).unwrap_or(p).to_string_lossy().replace('\\', "/");
    out.sort_by(|a, b| natord::compare(&key(a), &key(b)));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    fn tmp(tag: &str, ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!("yosh_archive_{}_{tag}.{ext}", std::process::id()))
    }

    fn payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add((i % 251) as u8)).collect()
    }

    fn collect(path: &Path) -> Vec<(Entry, Vec<u8>)> {
        let mut out = Vec::new();
        for_each(path, |e, b| {
            out.push((e.clone(), b.to_vec()));
            Ok(true)
        })
        .unwrap();
        out
    }

    /// Zip: archive order (not natural order), non-images and directories
    /// included, bytes and sizes exact, and `list` agrees with `for_each`.
    #[test]
    fn zip_lists_and_streams_every_entry_in_archive_order() {
        let path = tmp("zip", "cbz");
        {
            let mut w = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
            let opt = zip::write::SimpleFileOptions::default();
            w.start_file("ComicInfo.xml", opt).unwrap();
            w.write_all(b"<ComicInfo/>").unwrap();
            w.add_directory("vol/", opt).unwrap();
            w.start_file("vol/p10.png", opt).unwrap();
            w.write_all(&payload(1, 3000)).unwrap();
            w.start_file("vol/p2.png", opt).unwrap();
            w.write_all(&payload(2, 500)).unwrap();
            w.finish().unwrap();
        }
        assert_eq!(kind(&path), Some(Kind::Zip));
        let listed = list(&path).unwrap();
        let names: Vec<&str> = listed.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["ComicInfo.xml", "vol", "vol/p10.png", "vol/p2.png"]);
        assert!(listed[1].is_dir && !listed[2].is_dir);
        assert_eq!((listed[0].size, listed[2].size, listed[3].size), (12, 3000, 500));
        let streamed = collect(&path);
        assert_eq!(streamed.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>(), listed);
        assert_eq!(streamed[0].1, b"<ComicInfo/>");
        assert!(streamed[1].1.is_empty());
        assert_eq!(streamed[2].1, payload(1, 3000));
        assert_eq!(streamed[3].1, payload(2, 500));

        // Stopping early.
        let mut seen = 0;
        for_each(&path, |_, _| {
            seen += 1;
            Ok(seen < 2)
        })
        .unwrap();
        assert_eq!(seen, 2);
        let _ = std::fs::remove_file(&path);
    }

    /// 7z: data-less entries (a directory, an empty file) sit between and around
    /// blocks; the stream still comes out in archive order, byte-exact, across a
    /// solid multi-file block.
    #[test]
    fn sevenz_streams_in_archive_order_across_blocks_and_empty_entries() {
        use sevenz_rust2::{ArchiveEntry, ArchiveWriter, SourceReader};
        let path = tmp("7z", "cb7");
        {
            let mut w = ArchiveWriter::create(&path).unwrap();
            w.push_archive_entry::<Cursor<Vec<u8>>>(ArchiveEntry::new_directory("extras"), None).unwrap();
            w.push_archive_entry(ArchiveEntry::new_file("a.png"), Some(Cursor::new(payload(3, 4000)))).unwrap();
            w.push_archive_entry(ArchiveEntry::new_file("empty.txt"), Some(Cursor::new(Vec::new()))).unwrap();
            // One solid block holding two files.
            let entries = vec![ArchiveEntry::new_file("b.png"), ArchiveEntry::new_file("c.png")];
            let readers = vec![
                SourceReader::from(Cursor::new(payload(4, 2000))),
                SourceReader::from(Cursor::new(payload(5, 2500))),
            ];
            w.push_archive_entries(entries, readers).unwrap();
            w.push_archive_entry(ArchiveEntry::new_file("ComicInfo.xml"), Some(Cursor::new(b"<x/>".to_vec())))
                .unwrap();
            w.finish().unwrap();
        }
        assert_eq!(kind(&path), Some(Kind::SevenZ));
        let listed = list(&path).unwrap();
        let streamed = collect(&path);
        assert_eq!(streamed.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>(), listed);
        let got: Vec<(&str, usize)> = streamed.iter().map(|(e, b)| (e.name.as_str(), b.len())).collect();
        assert_eq!(
            got,
            [("extras", 0), ("a.png", 4000), ("empty.txt", 0), ("b.png", 2000), ("c.png", 2500), ("ComicInfo.xml", 4)]
        );
        assert!(streamed[0].0.is_dir);
        assert_eq!(streamed[3].1, payload(4, 2000));
        assert_eq!(streamed[4].1, payload(5, 2500));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn walk_folder_is_natural_sorted_and_optionally_recursive() {
        let dir = std::env::temp_dir().join(format!("yosh_walk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("ch10")).unwrap();
        std::fs::create_dir_all(dir.join("ch2")).unwrap();
        for f in ["p10.png", "p2.png", "ch2/x1.png", "ch10/x1.png", "ch2/x11.png", "ch2/x3.png"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        let rel = |v: Vec<PathBuf>| -> Vec<String> {
            v.iter().map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().replace('\\', "/")).collect()
        };
        assert_eq!(rel(walk_folder(&dir, false).unwrap()), ["p2.png", "p10.png"]);
        assert_eq!(
            rel(walk_folder(&dir, true).unwrap()),
            ["ch2/x1.png", "ch2/x3.png", "ch2/x11.png", "ch10/x1.png", "p2.png", "p10.png"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

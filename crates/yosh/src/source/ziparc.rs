//! CBZ/ZIP source. Reads happen in parallel: each `read_page` opens its own
//! `File` + `ZipArchive` (the central directory is tiny for a comic), so worker
//! threads never share a cursor.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use zip::ZipArchive;

use super::{is_image_name, PageSource};

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

pub struct ZipSource {
    path: PathBuf,
    names: Vec<String>,
    modified: Vec<Option<String>>,
}

impl ZipSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        let mut zip = ZipArchive::new(File::open(path)?).map_err(to_io)?;
        let mut entries: Vec<(String, Option<String>)> = (0..zip.len())
            .filter_map(|i| {
                let f = zip.by_index(i).ok()?;
                let name = f.name().to_string();
                if !(f.is_file() && is_image_name(&name)) {
                    return None;
                }
                let modified = f.last_modified().map(|dt| {
                    format!(
                        "{:04}-{:02}-{:02} {:02}:{:02}",
                        dt.year(),
                        dt.month(),
                        dt.day(),
                        dt.hour(),
                        dt.minute()
                    )
                });
                Some((name, modified))
            })
            .collect();
        entries.sort_by(|a, b| natord::compare(&a.0, &b.0));
        let (names, modified) = entries.into_iter().unzip();
        Ok(Self {
            path: path.to_path_buf(),
            names,
            modified,
        })
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
        let mut zip = ZipArchive::new(File::open(&self.path)?).map_err(to_io)?;
        let mut entry = zip.by_name(&self.names[index]).map_err(to_io)?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn modified(&self, index: usize) -> Option<String> {
        self.modified.get(index).cloned().flatten()
    }
}

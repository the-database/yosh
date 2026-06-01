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
}

impl ZipSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        let mut zip = ZipArchive::new(File::open(path)?).map_err(to_io)?;
        let mut names: Vec<String> = (0..zip.len())
            .filter_map(|i| {
                let f = zip.by_index(i).ok()?;
                let name = f.name().to_string();
                (f.is_file() && is_image_name(&name)).then_some(name)
            })
            .collect();
        names.sort_by(|a, b| natord::compare(a, b));
        Ok(Self {
            path: path.to_path_buf(),
            names,
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
}

//! 7z/CB7 source. Like RAR, 7z is typically solid (block-compressed), so random
//! access is expensive; a single reader thread extracts entries sequentially via
//! `for_each_entries` into an in-memory map, and `read_page(i)` blocks until
//! ready. (Same shape as `rar.rs`; bounded by archive size — M2.)

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use sevenz_rust2::{ArchiveReader, Password};

use super::{is_image_name, PageSource};

fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
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
}

pub struct SevenzSource {
    names: Vec<String>,
    shared: Arc<Shared>,
    _reader: JoinHandle<()>,
}

impl SevenzSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        // List image entries up front, natural-sorted.
        let reader = ArchiveReader::open(path, Password::empty()).map_err(to_io)?;
        let mut names: Vec<String> = reader
            .archive()
            .files
            .iter()
            .filter(|e| !e.is_directory() && is_image_name(e.name()))
            .map(|e| e.name().to_string())
            .collect();
        drop(reader);
        names.sort_by(|a, b| natord::compare(a, b));

        let name_to_idx: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let shared = Arc::new(Shared {
            ready: Mutex::new(Ready {
                map: HashMap::new(),
                done: false,
                error: None,
            }),
            cv: Condvar::new(),
            abort: AtomicBool::new(false),
        });

        let reader_thread = {
            let shared = shared.clone();
            let path = path.to_path_buf();
            std::thread::spawn(move || extract_all(path, name_to_idx, shared))
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

fn extract_all(path: PathBuf, name_to_idx: HashMap<String, usize>, shared: Arc<Shared>) {
    let mut reader = match ArchiveReader::open(&path, Password::empty()) {
        Ok(r) => r,
        Err(e) => return finish(&shared, Some(e.to_string())),
    };
    let result = reader.for_each_entries(|entry, rd| {
        if shared.abort.load(Ordering::Relaxed) {
            return Ok(false); // stop iterating
        }
        if let Some(&idx) = name_to_idx.get(entry.name()) {
            let mut buf = Vec::new();
            std::io::Read::read_to_end(rd, &mut buf)?;
            {
                let mut g = shared.ready.lock().unwrap();
                g.map.insert(idx, Arc::new(buf));
            }
            shared.cv.notify_all();
        }
        Ok(true)
    });
    match result {
        Ok(_) => finish(&shared, None),
        Err(e) => finish(&shared, Some(e.to_string())),
    }
}

impl PageSource for SevenzSource {
    fn len(&self) -> usize {
        self.names.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn read_page(&self, index: usize) -> io::Result<Vec<u8>> {
        let mut guard = self.shared.ready.lock().unwrap();
        loop {
            if let Some(bytes) = guard.map.get(&index) {
                return Ok(bytes.as_ref().clone());
            }
            if guard.done {
                let msg = guard
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("7z: page {index} not found"));
                return Err(io::Error::new(io::ErrorKind::Other, msg));
            }
            guard = self.shared.cv.wait(guard).unwrap();
        }
    }
}

impl Drop for SevenzSource {
    fn drop(&mut self) {
        self.shared.abort.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
    }
}

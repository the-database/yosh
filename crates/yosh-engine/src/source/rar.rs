//! CBR/RAR source. RAR is strictly sequential (no random access), so a single
//! reader thread decompresses entries front-to-back into an in-memory map; the
//! decode workers' `read_page(i)` blocks until entry `i` has been produced.
//!
//! M1 keeps every extracted page in memory (bounded by archive size — fine for
//! typical CBRs on this hardware). Windowed eviction is an M2 optimization.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use unrar::Archive;

use super::{is_image_name, PageSource};

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

pub struct RarSource {
    names: Vec<String>,
    shared: Arc<Shared>,
    _reader: JoinHandle<()>,
}

impl RarSource {
    pub fn new(path: &Path) -> io::Result<Self> {
        // List entries up front (archive order ~ reading order); natural-sort by name.
        let mut names: Vec<String> = Vec::new();
        let listing = Archive::new(path)
            .open_for_listing()
            .map_err(|e| io::Error::other(e.to_string()))?;
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

        let shared = Arc::new(Shared {
            ready: Mutex::new(Ready {
                map: HashMap::new(),
                done: false,
                error: None,
            }),
            cv: Condvar::new(),
            abort: AtomicBool::new(false),
        });

        let reader = {
            let shared = shared.clone();
            let path = path.to_path_buf();
            std::thread::spawn(move || extract_all(path, name_to_idx, shared))
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

/// Reader thread: decompress every tracked entry front-to-back into the map.
fn extract_all(path: PathBuf, name_to_idx: HashMap<String, usize>, shared: Arc<Shared>) {
    let mut cursor = match Archive::new(&path).open_for_processing() {
        Ok(c) => c,
        Err(e) => return finish(&shared, Some(e.to_string())),
    };
    loop {
        if shared.abort.load(Ordering::Relaxed) {
            return;
        }
        let header = match cursor.read_header() {
            Ok(Some(h)) => h,
            Ok(None) => break,
            Err(e) => return finish(&shared, Some(e.to_string())),
        };
        let name = header.entry().filename.to_string_lossy().into_owned();
        match name_to_idx.get(&name).copied() {
            Some(idx) => match header.read() {
                Ok((bytes, next)) => {
                    {
                        let mut g = shared.ready.lock().unwrap();
                        g.map.insert(idx, Arc::new(bytes));
                    }
                    shared.cv.notify_all();
                    cursor = next;
                }
                Err(e) => return finish(&shared, Some(e.to_string())),
            },
            None => match header.skip() {
                Ok(next) => cursor = next,
                Err(e) => return finish(&shared, Some(e.to_string())),
            },
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
                    .unwrap_or_else(|| format!("rar: page {index} not found"));
                return Err(io::Error::other(msg));
            }
            guard = self.shared.cv.wait(guard).unwrap();
        }
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

//! The batch: walk the input into an ordered stream of items, push them through a pipeline —
//! parallel decode/prepare, one inference thread (the models live there), parallel resize/encode —
//! and write the results back in input order, so archive entries keep their order.
//!
//! Nothing a single file does can stop the run: a file that fails is reported (`ERROR …`) and
//! counted as done, an archive entry that can't be decoded is copied as-is, and every output is
//! written under a temporary name and renamed into place only when complete.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

use yosh_image::archive;
use yosh_image::resize::Resizer;

use crate::process::{self, Job, Work};
use crate::progress;
use crate::upscaler::Upscaler;

/// Inputs the GUI counts as images and archives (matched case-insensitively).
pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "avif"];
pub const ARCHIVE_EXTENSIONS: &[&str] = &["zip", "cbz", "rar", "cbr"];
/// Inside an archive, also try these as images (anything that fails to decode is copied as-is).
const ENTRY_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "avif", "gif", "jxl", "tif", "tiff"];

fn ext_in(path: &Path, list: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| list.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Where a run's input and output are.
pub enum Input {
    File(PathBuf),
    Folder { dir: PathBuf, images: bool, archives: bool },
}

pub struct Output {
    pub dir: PathBuf,
    /// The file name template; `%filename%` is the input's name without its extension.
    pub filename: String,
    pub overwrite: bool,
}

impl Output {
    fn name_for(&self, input: &Path, ext: &str) -> String {
        let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        format!("{}.{ext}", self.filename.replace("%filename%", &stem))
    }
}

// ---- items through the pipeline ------------------------------------------------------------

#[derive(Clone)]
enum Dest {
    /// A file; `folder` picks the progress line (folder image vs. single file).
    File { path: PathBuf, folder: bool },
    /// An entry of the archive being written; `source` is its original name (for a raw copy).
    Entry { name: String, source: String },
}

enum Item {
    Image { seq: u64, label: String, src: Src, dest: Dest },
    Raw { seq: u64, name: String, bytes: Vec<u8>, is_dir: bool },
    BeginArchive { seq: u64, out: PathBuf, entries: usize },
    EndArchive { seq: u64 },
}

enum Src {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

enum Stage {
    Work { seq: u64, label: String, dest: Dest, work: Work, original: Option<Vec<u8>> },
    Done { seq: u64, done: Done },
}

enum Done {
    Write { dest: Dest, bytes: Vec<u8> },
    Raw { name: String, bytes: Vec<u8>, is_dir: bool },
    BeginArchive { out: PathBuf, entries: usize },
    EndArchive,
    Failed { label: String, dest: Dest, err: String, original: Option<Vec<u8>> },
}

fn catch<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
        Err(format!("internal error: {}", msg.unwrap_or_else(|| "panic".into())))
    })
}

// ---- walking -------------------------------------------------------------------------------

struct Walker<'a> {
    tx: SyncSender<Item>,
    seq: u64,
    out: &'a Output,
}

impl Walker<'_> {
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn send(&self, item: Item) -> bool {
        self.tx.send(item).is_ok()
    }

    fn skip(&self, path: &Path) -> bool {
        if !self.out.overwrite && path.exists() {
            progress::log(&format!("file exists, skip: {}", path.display()));
            return true;
        }
        false
    }

    fn image(&mut self, src: &Path, out_dir: &Path, format_ext: &str, folder: bool) -> bool {
        let dest = out_dir.join(self.out.name_for(src, format_ext));
        if self.skip(&dest) {
            return true;
        }
        let seq = self.next_seq();
        self.send(Item::Image { seq, label: src.display().to_string(), src: Src::Path(src.to_path_buf()), dest: Dest::File { path: dest, folder } })
    }

    fn archive(&mut self, src: &Path, out_dir: &Path, format_ext: &str) -> bool {
        let dest = out_dir.join(self.out.name_for(src, "cbz"));
        if self.skip(&dest) {
            return true;
        }
        let entries = match archive::list(src) {
            Ok(e) => e.len(),
            Err(e) => {
                progress::error(&src.display().to_string(), &e.to_string());
                return true;
            }
        };
        let seq = self.next_seq();
        if !self.send(Item::BeginArchive { seq, out: dest, entries }) {
            return false;
        }
        let mut alive = true;
        let res = archive::for_each(src, |entry, bytes| {
            let seq = self.next_seq();
            let label = format!("{} :: {}", src.display(), entry.name);
            let item = if !entry.is_dir && ext_in(Path::new(&entry.name), ENTRY_IMAGE_EXTENSIONS) {
                let name = Path::new(&entry.name).with_extension(format_ext).to_string_lossy().replace('\\', "/");
                Item::Image { seq, label, src: Src::Bytes(bytes.to_vec()), dest: Dest::Entry { name, source: entry.name.clone() } }
            } else {
                Item::Raw { seq, name: entry.name.clone(), bytes: bytes.to_vec(), is_dir: entry.is_dir }
            };
            alive = self.tx.send(item).is_ok();
            Ok(alive)
        });
        if let Err(e) = res {
            progress::error(&src.display().to_string(), &format!("reading the archive stopped early: {e}"));
        }
        let seq = self.next_seq();
        alive && self.send(Item::EndArchive { seq })
    }
}

fn walk(input: &Input, out: &Output, format_ext: &str, tx: SyncSender<Item>) {
    let mut w = Walker { tx, seq: 0, out };
    match input {
        Input::File(path) => {
            if ext_in(path, ARCHIVE_EXTENSIONS) {
                w.archive(path, &out.dir, format_ext);
            } else if ext_in(path, IMAGE_EXTENSIONS) {
                w.image(path, &out.dir, format_ext, false);
            } else {
                progress::error(&path.display().to_string(), "not an image or archive");
            }
        }
        Input::Folder { dir, images, archives } => {
            let files = match archive::walk_folder(dir, true) {
                Ok(f) => f,
                Err(e) => {
                    progress::error(&dir.display().to_string(), &e.to_string());
                    return;
                }
            };
            for f in files {
                let rel_dir = f.parent().and_then(|p| p.strip_prefix(dir).ok()).unwrap_or(Path::new(""));
                let out_dir = out.dir.join(rel_dir);
                let alive = if *images && ext_in(&f, IMAGE_EXTENSIONS) {
                    w.image(&f, &out_dir, format_ext, true)
                } else if *archives && ext_in(&f, ARCHIVE_EXTENSIONS) {
                    w.archive(&f, &out_dir, format_ext)
                } else {
                    true
                };
                if !alive {
                    return;
                }
            }
        }
    }
}

// ---- stages --------------------------------------------------------------------------------

fn prepare_stage(item: Item, job: &Job, resizer: &mut Resizer) -> Stage {
    match item {
        Item::Image { seq, label, src, dest } => {
            let (bytes, keep) = match src {
                Src::Path(p) => match std::fs::read(&p) {
                    Ok(b) => (b, false),
                    Err(e) => return Stage::Done { seq, done: Done::Failed { label, dest, err: e.to_string(), original: None } },
                },
                Src::Bytes(b) => (b, true),
            };
            match catch(|| process::prepare(&bytes, job, &label, resizer)) {
                Ok(work) => Stage::Work { seq, label, dest, work, original: keep.then_some(bytes) },
                Err(err) => Stage::Done { seq, done: Done::Failed { label, dest, err, original: keep.then_some(bytes) } },
            }
        }
        Item::Raw { seq, name, bytes, is_dir } => Stage::Done { seq, done: Done::Raw { name, bytes, is_dir } },
        Item::BeginArchive { seq, out, entries } => Stage::Done { seq, done: Done::BeginArchive { out, entries } },
        Item::EndArchive { seq } => Stage::Done { seq, done: Done::EndArchive },
    }
}

fn infer_stage(stage: Stage, up: &mut Option<Upscaler>, up_err: &Option<String>) -> Stage {
    match stage {
        Stage::Work { seq, label, dest, work, original } => {
            let res = match (needs_model(&work), up.as_mut()) {
                (false, _) => Ok(work),
                (true, Some(up)) => catch(|| process::infer(work, up)),
                (true, None) => Err(up_err.clone().unwrap_or_else(|| "no upscaling backend".into())),
            };
            match res {
                Ok(work) => Stage::Work { seq, label, dest, work, original },
                Err(err) => Stage::Done { seq, done: Done::Failed { label, dest, err, original } },
            }
        }
        done => done,
    }
}

fn finish_stage(stage: Stage, job: &Job, resizer: &mut Resizer) -> (u64, Done) {
    match stage {
        Stage::Work { seq, label, dest, work, original } => match catch(|| process::finish(work, job, resizer)) {
            Ok(bytes) => (seq, Done::Write { dest, bytes }),
            Err(err) => (seq, Done::Failed { label, dest, err, original }),
        },
        Stage::Done { seq, done } => (seq, done),
    }
}

/// `n` threads taking items from `rx`, each mapping them with `f` (given its own `Resizer`).
fn workers<I: Send + 'static, O: Send + 'static>(
    n: usize,
    rx: Receiver<I>,
    tx: SyncSender<O>,
    f: impl Fn(I, &mut Resizer) -> O + Send + Sync + 'static,
) -> Vec<thread::JoinHandle<()>> {
    let rx = Arc::new(Mutex::new(rx));
    let f = Arc::new(f);
    (0..n)
        .map(|_| {
            let (rx, tx, f) = (rx.clone(), tx.clone(), f.clone());
            thread::spawn(move || {
                let mut resizer = Resizer::new();
                loop {
                    let next = rx.lock().unwrap().recv();
                    let Ok(item) = next else { break };
                    if tx.send(f(item, &mut resizer)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect()
}

// ---- writing -------------------------------------------------------------------------------

fn part_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let part = part_path(path);
    std::fs::write(&part, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&part, path).map_err(|e| e.to_string())
}

struct ArchiveOut {
    out: PathBuf,
    zip: zip::ZipWriter<std::fs::File>,
    names: HashSet<String>,
}

impl ArchiveOut {
    fn start(out: &Path) -> Result<Self, String> {
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let file = std::fs::File::create(part_path(out)).map_err(|e| e.to_string())?;
        Ok(Self { out: out.to_path_buf(), zip: zip::ZipWriter::new(file), names: HashSet::new() })
    }

    /// Add a file; images are stored (already compressed), anything else deflated. A name already
    /// written (two pages that differed only by extension) gets the old extension kept in it.
    fn add(&mut self, name: &str, bytes: &[u8], compress: bool) -> Result<(), String> {
        use zip::write::SimpleFileOptions;
        let mut name = name.to_string();
        if !self.names.insert(name.clone()) {
            let p = Path::new(&name);
            let ext = p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            name = format!("{}-{}.{ext}", p.with_extension("").to_string_lossy(), self.names.len());
            self.names.insert(name.clone());
        }
        let method = if compress { zip::CompressionMethod::Deflated } else { zip::CompressionMethod::Stored };
        self.zip
            .start_file(name, SimpleFileOptions::default().compression_method(method).large_file(bytes.len() >= u32::MAX as usize))
            .map_err(|e| e.to_string())?;
        self.zip.write_all(bytes).map_err(|e| e.to_string())
    }

    fn add_dir(&mut self, name: &str) -> Result<(), String> {
        if self.names.insert(format!("{name}/")) {
            self.zip.add_directory(name, zip::write::SimpleFileOptions::default()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn finish(self) -> Result<(), String> {
        self.zip.finish().map_err(|e| e.to_string())?;
        std::fs::rename(part_path(&self.out), &self.out).map_err(|e| e.to_string())
    }
}

struct Sink {
    archive: Option<ArchiveOut>,
}

impl Sink {
    fn file_done(folder: bool) {
        if folder {
            progress::folder_image_done();
        } else {
            progress::image_done();
        }
    }

    fn handle(&mut self, done: Done) {
        match done {
            Done::Write { dest: Dest::File { path, folder }, bytes } => {
                match write_file(&path, &bytes) {
                    Ok(()) => progress::log(&format!("save image: {}", path.display())),
                    Err(e) => progress::error(&path.display().to_string(), &e),
                }
                Self::file_done(folder);
            }
            Done::Write { dest: Dest::Entry { name, .. }, bytes } => {
                if let Some(a) = &mut self.archive
                    && let Err(e) = a.add(&name, &bytes, false)
                {
                    progress::error(&name, &e);
                }
                progress::archive_entry_done();
            }
            Done::Raw { name, bytes, is_dir } => {
                if let Some(a) = &mut self.archive {
                    let r = if is_dir { a.add_dir(&name) } else { a.add(&name, &bytes, true) };
                    if let Err(e) = r {
                        progress::error(&name, &e);
                    }
                }
                progress::archive_entry_done();
            }
            Done::BeginArchive { out, entries } => {
                match ArchiveOut::start(&out) {
                    Ok(a) => self.archive = Some(a),
                    Err(e) => progress::error(&out.display().to_string(), &e),
                }
                progress::archive_start(entries);
            }
            Done::EndArchive => {
                if let Some(a) = self.archive.take() {
                    let out = a.out.clone();
                    match a.finish() {
                        Ok(()) => progress::log(&format!("save archive: {}", out.display())),
                        Err(e) => progress::error(&out.display().to_string(), &e),
                    }
                }
                progress::archive_done();
            }
            Done::Failed { label, dest, err, original } => {
                progress::error(&label, &err);
                match dest {
                    Dest::File { folder, .. } => Self::file_done(folder),
                    Dest::Entry { source, .. } => {
                        // keep the page: copy the original bytes under their original name
                        if let (Some(a), Some(bytes)) = (&mut self.archive, original)
                            && let Err(e) = a.add(&source, &bytes, false)
                        {
                            progress::error(&source, &e);
                        }
                        progress::archive_entry_done();
                    }
                }
            }
        }
    }
}

// ---- the run -------------------------------------------------------------------------------

/// Run the batch. `upscaler` is opened lazily by the inference thread (so a run that needs no
/// model never loads the runtime); `open_upscaler` returns it or why it couldn't open.
pub fn run(
    input: Input,
    output: Output,
    job: Job,
    open_upscaler: impl FnOnce() -> Result<Upscaler, String> + Send + 'static,
) {
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n = (threads / 2).clamp(2, 8);
    let job = Arc::new(job);
    let format_ext = job.format.ext().to_string();

    let (item_tx, item_rx) = sync_channel::<Item>(n * 2);
    let (prep_tx, prep_rx) = sync_channel::<Stage>(n);
    let (infer_tx, infer_rx) = sync_channel::<Stage>(n);
    let (done_tx, done_rx) = sync_channel::<(u64, Done)>(n * 2);

    let j = job.clone();
    let mut handles = workers(n, item_rx, prep_tx, move |item, r| prepare_stage(item, &j, r));

    handles.push(thread::spawn(move || {
        let mut up: Option<Upscaler> = None;
        let mut up_err: Option<String> = None;
        let mut open = Some(open_upscaler);
        for stage in prep_rx {
            if let Stage::Work { work, .. } = &stage
                && needs_model(work)
                && let Some(open) = open.take()
            {
                match open() {
                    Ok(u) => {
                        progress::log(&format!("Upscaling with {}", u.description));
                        up = Some(u);
                    }
                    Err(e) => {
                        progress::error("upscaling backend", &e);
                        up_err = Some(e);
                    }
                }
            }
            if infer_tx.send(infer_stage(stage, &mut up, &up_err)).is_err() {
                break;
            }
        }
    }));

    let j = job.clone();
    handles.extend(workers(n, infer_rx, done_tx, move |stage, r| finish_stage(stage, &j, r)));

    let walker = thread::spawn(move || walk(&input, &output, &format_ext, item_tx));

    // Write results in input order.
    let mut sink = Sink { archive: None };
    let mut pending: BTreeMap<u64, Done> = BTreeMap::new();
    let mut next = 1u64;
    for (seq, done) in done_rx {
        pending.insert(seq, done);
        while let Some(done) = pending.remove(&next) {
            sink.handle(done);
            next += 1;
        }
    }
    let _ = walker.join();
    for h in handles {
        let _ = h.join();
    }
}

fn needs_model(work: &Work) -> bool {
    work.chain.as_ref().is_some_and(|c| {
        let m = c.model_file_path.trim();
        !m.is_empty() && m != "No Model"
    })
}

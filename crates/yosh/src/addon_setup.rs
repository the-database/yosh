//! Installing and updating the AI add-on (`addon.rs`) with MangaJaNaiCLI's `mangajanai setup`.
//!
//! The add-on folder becomes a mangajanai install. yosh downloads the release archive it pins
//! (`mangajanai` + libaji and its backends, plus the VC++ runtime on Windows), checks its SHA-256,
//! unpacks it into the folder and runs `mangajanai setup --progress lines` there. Setup picks what
//! this PC needs (TensorRT and its GPU files, TensorRT-RTX, DirectML on Windows) and the
//! MangaJaNai / IllustrationJaNai models (into `models/`, where yosh looks), checks each download
//! against its own pins, and records what it installed in `setup.json`, so a re-run fetches only
//! what changed. `yosh-addon.json` records the release yosh installed: a newer pin here offers an
//! update. An add-on put together by hand has no record and is left alone.
//!
//! An installed (not portable) yosh runs setup in **store mode** (`--store --app-id yosh-addon
//! --app-path <the add-on's mangajanai>`), when the add-on's mangajanai knows it: TensorRT, its GPU
//! files, TensorRT-RTX and the models then go into the per-user pack store that MangaJaNaiCLI and
//! the other apps share (the-database/aji-runtime), once per user instead of once per app, and an
//! older add-on's matching files move there instead of downloading again. The add-on's
//! `aji-runtime.json` points libaji at them and lists the model folders (`addon::models_dirs`).
//! The ref is tied to the add-on's own program, so it lives exactly as long as the add-on folder:
//! *Remove the add-on* and the Windows uninstaller run `mangajanai setup --remove` first, and an
//! add-on folder deleted any other way leaves a dead ref that the store frees later. A portable
//! yosh keeps everything in its own add-on folder, as before.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};
use yosh_engine::pool::Waker;

/// The MangaJaNaiCLI release the add-on is installed from. A bump here offers installed add-ons
/// an update.
pub const CLI_VERSION: &str = "v0.8.0";
const CLI_REPO: &str = "the-database/MangaJaNaiCLI";
#[cfg(windows)]
const CLI_ASSET: (&str, &str, u64) = (
    "mangajanai-windows-x64.zip",
    "b3ed5c6b8fbfba993ab3c1bff3be58655fa5714a8af19437d9a0bf1387b1008c",
    11_168_775,
);
#[cfg(not(windows))]
const CLI_ASSET: (&str, &str, u64) = (
    "mangajanai-linux-x64.tar.zst",
    "a803268a5e851e17b4197519a2221f5a7fffb92d50d52e683969a2870990d0d0",
    9_274_753,
);
const PROGRAM: &str = if cfg!(windows) { "mangajanai.exe" } else { "mangajanai" };
const RECORD: &str = "yosh-addon.json";
const DOWNLOAD_DIR: &str = ".yosh-download";
const UA: &str = concat!("yosh/", env!("CARGO_PKG_VERSION"));
/// The add-on's id in the shared pack store (its `refs/<id>.json`).
const STORE_APP_ID: &str = "yosh-addon";

/// Which part of the install a download is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Part {
    /// The mangajanai archive itself.
    Installer,
    /// The GPU runtime (TensorRT, TensorRT-RTX, DirectML).
    Runtime,
    Models,
}

/// Where an install is. `frac` is the whole install's progress (0..1), by bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum Progress {
    Starting,
    Download { part: Part, frac: f32 },
    Unpack { frac: f32 },
}

pub enum Event {
    Progress(Progress),
    Done,
    Cancelled,
    Failed(String),
}

/// A running install. Events arrive on `rx`, each followed by a call to the waker.
pub struct Job {
    pub rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

impl Job {
    /// Stop the install: a download stops at its next read, setup is killed. What finished stays
    /// (setup.json records it), so the next install picks up from there.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(c) = self.child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            let _ = c.kill();
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Record {
    cli: String,
    /// The NVIDIA GPUs' compute capabilities ("8.6") when setup last ran: setup picks the
    /// TensorRT files (and TensorRT-RTX) by them. `None` in a record without them (not checked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gpus: Option<Vec<String>>,
}

fn read_record(dir: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(dir.join(RECORD)).ok()?).ok()
}

/// This machine's NVIDIA GPUs' compute capabilities, as the record keeps them.
#[cfg(feature = "sr")]
fn gpu_keys() -> Vec<String> {
    let mut v: Vec<String> =
        yosh_engine::nvml::gpus().iter().map(|g| format!("{}.{}", g.major, g.minor)).collect();
    v.sort();
    v.dedup();
    v
}

#[cfg(not(feature = "sr"))]
fn gpu_keys() -> Vec<String> {
    Vec::new()
}

/// Whether some GPU here has a compute capability setup didn't install for (`recorded`).
fn lacks_gpu(recorded: &[String], now: &[String]) -> bool {
    now.iter().any(|g| !recorded.contains(g))
}

/// Whether the add-on yosh installed in `dir` lacks files for one of this machine's NVIDIA GPUs:
/// it was set up on another PC (a portable yosh moved) or before a GPU swap. Re-running setup
/// fetches what's missing. Blocking (loads NVML): call it off the UI thread.
pub fn needs_setup_for_gpus(dir: &Path) -> bool {
    match read_record(dir).and_then(|r| r.gpus) {
        Some(recorded) => lacks_gpu(&recorded, &gpu_keys()),
        None => false,
    }
}

/// The MangaJaNaiCLI release yosh installed `dir` from, if yosh installed it.
pub fn installed_release(dir: &Path) -> Option<String> {
    read_record(dir).map(|r| r.cli)
}

/// Whether an install into `dir` began but didn't finish (the program is there, the record isn't):
/// Install is offered again, and setup picks up what's missing.
pub fn incomplete(dir: &Path) -> bool {
    dir.join(PROGRAM).is_file() && !dir.join(RECORD).is_file()
}

/// Whether yosh installed `dir` from an older (or other) release than the one it pins now.
pub fn update_available(dir: &Path) -> bool {
    installed_release(dir).is_some_and(|v| v != CLI_VERSION)
}

/// What the add-on folder holds on disk, in bytes (runtime, models, engine cache).
pub fn folder_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => folder_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// Whether setup put this add-on's packs in the shared pack store (its `setup.json` says so).
pub(crate) fn store_install(dir: &Path) -> bool {
    std::fs::read(dir.join("setup.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|j| j.get("store").is_some_and(|s| !s.is_null()))
}

/// A `mangajanai` command in `dir`, without a console window.
pub(crate) fn program(dir: &Path) -> Command {
    let mut cmd = Command::new(dir.join(PROGRAM));
    cmd.current_dir(dir).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Whether the add-on's mangajanai has store mode (`setup --store`, MangaJaNaiCLI after v0.3.1).
fn supports_store(dir: &Path) -> bool {
    program(dir)
        .arg("--help")
        .stderr(Stdio::null())
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("--app-id"))
}

/// The arguments for `mangajanai setup` in `dir`: store mode for an installed yosh whose
/// add-on supports it, the add-on's own folder otherwise. `YOSH_ADDON_MIRROR` adds `--mirror`
/// (local archives, for testing before a release is published).
fn setup_args(dir: &Path, portable: bool, store_ok: bool) -> Vec<std::ffi::OsString> {
    let mut a: Vec<std::ffi::OsString> = ["setup", "--progress", "lines"].iter().map(Into::into).collect();
    if !portable && store_ok {
        a.extend(["--store".into(), "--app-id".into(), STORE_APP_ID.into(), "--app-path".into(), dir.join(PROGRAM).into()]);
    }
    if let Some(m) = std::env::var_os("YOSH_ADDON_MIRROR") {
        a.extend(["--mirror".into(), m]);
    }
    a
}

/// Delete the add-on: everything in `dir` but a `rules.json` the user wrote (kept so a later
/// install picks it up again). A store install first drops its ref (`mangajanai setup
/// --remove`), freeing the packs no other app uses. Fails on the first file that can't be
/// deleted; on Windows that is libaji's files while they're loaded, so don't call this after an
/// upscaler started.
pub fn remove(dir: &Path) -> Result<(), String> {
    if store_install(dir) {
        match program(dir).args(["setup", "--remove"]).stdout(Stdio::null()).stderr(Stdio::null()).status() {
            Ok(s) if s.success() => {}
            // The folder goes anyway: its ref is then dead, and the store frees its packs later.
            r => eprintln!("yosh: [addon] dropping the pack store ref failed ({r:?}); removing the folder anyway"),
        }
    }
    let rules = dir.join("rules.json");
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    for e in rd.flatten() {
        let path = e.path();
        if path == rules {
            continue;
        }
        let r = match e.file_type() {
            Ok(t) if t.is_dir() => std::fs::remove_dir_all(&path),
            _ => std::fs::remove_file(&path),
        };
        r.map_err(|err| format!("{}: {err}", path.display()))?;
    }
    if !rules.is_file() {
        let _ = std::fs::remove_dir(dir);
    }
    Ok(())
}

/// What an install does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Unpack the pinned mangajanai release, then run setup (install or update).
    Full,
    /// Run the installed mangajanai's setup only: fetch what this machine's GPUs are missing.
    /// It only adds files (the ones already installed are skipped), so it can run while libaji
    /// is loaded.
    SetupOnly,
}

/// Install (or update) the add-on into `dir` on a background thread.
pub fn start(dir: PathBuf, mode: Mode, wake: Waker) -> Job {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let child = Arc::new(Mutex::new(None));
    let job = Job { rx, cancel: cancel.clone(), child: child.clone() };
    std::thread::spawn(move || {
        let send = |e: Event| {
            let _ = tx.send(e);
            wake();
        };
        send(Event::Progress(Progress::Starting));
        let ev = match install(&dir, mode, &cancel, &child, &tx, &wake) {
            Ok(()) => Event::Done,
            Err(_) if cancel.load(Ordering::Relaxed) => Event::Cancelled,
            Err(e) => Event::Failed(e),
        };
        if !matches!(ev, Event::Done) {
            eprintln!("yosh: [addon] install stopped: {}", match &ev {
                Event::Failed(e) => e.as_str(),
                _ => "cancelled",
            });
        }
        send(ev);
    });
    job
}

fn install(
    dir: &Path,
    mode: Mode,
    cancel: &AtomicBool,
    child: &Mutex<Option<Child>>,
    tx: &Sender<Event>,
    wake: &Waker,
) -> Result<(), String> {
    let progress = |p: Progress| {
        let _ = tx.send(Event::Progress(p));
        wake();
    };
    if mode == Mode::Full {
        unpack_release(dir, cancel, &progress)?;
    }
    run_setup(dir, mode, cancel, child, &progress)?;
    // Setup-only ran the installed program, so the record keeps its release.
    let cli = match mode {
        Mode::SetupOnly => read_record(dir).map(|r| r.cli),
        Mode::Full => None,
    }
    .unwrap_or_else(|| CLI_VERSION.into());
    let rec = serde_json::to_vec_pretty(&Record { cli, gpus: Some(gpu_keys()) }).unwrap_or_default();
    std::fs::write(dir.join(RECORD), rec).map_err(|e| format!("{RECORD}: {e}"))
}

/// Download the pinned mangajanai release (or YOSH_ADDON_ARCHIVE), check it and unpack it.
fn unpack_release(dir: &Path, cancel: &AtomicBool, progress: &dyn Fn(Progress)) -> Result<(), String> {
    let dl = dir.join(DOWNLOAD_DIR);
    std::fs::create_dir_all(&dl).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (asset, sha, bytes) = CLI_ASSET;
    let archive = dl.join(asset);
    // A local archive instead of the pinned release, for testing a MangaJaNaiCLI build before it
    // is published. Not pin-checked.
    match std::env::var_os("YOSH_ADDON_ARCHIVE") {
        Some(local) => {
            std::fs::copy(&local, &archive).map_err(|e| format!("{}: {e}", Path::new(&local).display()))?;
        }
        None => {
            let url = format!("https://github.com/{CLI_REPO}/releases/download/{CLI_VERSION}/{asset}");
            download(&url, &archive, bytes, cancel, |f| {
                progress(Progress::Download { part: Part::Installer, frac: f })
            })?;
            let got = sha256_file(&archive)?;
            if got != sha {
                let _ = std::fs::remove_file(&archive);
                return Err(format!("{asset} doesn't match the SHA-256 yosh pins"));
            }
        }
    }
    if cfg!(windows) {
        unpack_zip(&archive, dir)?;
    } else {
        unpack_tar_zst(&archive, dir)?;
    }
    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_dir(&dl);
    Ok(())
}

/// Run `mangajanai setup --progress lines` in `dir` and turn its lines into progress.
fn run_setup(
    dir: &Path,
    mode: Mode,
    cancel: &AtomicBool,
    child: &Mutex<Option<Child>>,
    progress: &dyn Fn(Progress),
) -> Result<(), String> {
    let mut cmd = program(dir);
    // Moving a private add-on into the store removes its own runtime files, so that happens only
    // in a full install or update (before libaji loads), never in a setup-only run beside a
    // loaded libaji; a store install stays one either way.
    let store_ok = supports_store(dir) && (mode == Mode::Full || store_install(dir));
    cmd.args(setup_args(dir, crate::config::portable(), store_ok)).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut c = cmd.spawn().map_err(|e| format!("{PROGRAM}: {e}"))?;
    let stdout = c.stdout.take();
    let stderr = c.stderr.take();
    *child.lock().unwrap_or_else(|p| p.into_inner()) = Some(c);
    if cancel.load(Ordering::Relaxed) {
        // Cancelled while it was starting: `Job::cancel` found no child to kill.
        if let Some(c) = child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            let _ = c.kill();
        }
    }
    // stderr on its own thread so neither pipe can fill up and stall setup. Its last error line
    // is what a failure reports.
    let err_thread = std::thread::spawn(move || {
        let mut last = String::new();
        if let Some(e) = stderr {
            // Only the error lines matter here; the rest is setup's human output for a terminal.
            for line in BufReader::new(e).lines().map_while(Result::ok) {
                if let Some(msg) = line.trim().strip_prefix("error: ") {
                    last = msg.to_string();
                }
            }
        }
        last
    });
    let mut parser = SetupLines::default();
    let mut done = false;
    if let Some(out) = stdout {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            match parser.feed(&line) {
                Some(Feed::Progress(p)) => progress(p),
                Some(Feed::Done) => done = true,
                None => {}
            }
        }
    }
    let status = child.lock().unwrap_or_else(|p| p.into_inner()).take().map(|mut c| c.wait());
    let err = err_thread.join().unwrap_or_default();
    match status {
        Some(Ok(s)) if s.success() && done => Ok(()),
        Some(Ok(s)) if !err.is_empty() => Err(format!("setup: {err} (exit {})", s.code().unwrap_or(-1))),
        Some(Ok(s)) => Err(format!("setup exited with code {}", s.code().unwrap_or(-1))),
        Some(Err(e)) => Err(format!("setup: {e}")),
        None => Err("setup didn't start".into()),
    }
}

enum Feed {
    Progress(Progress),
    Done,
}

/// Reads setup's `SETUP_*` lines (MangaJaNaiCLI `setup.rs`, `Lines`) into whole-install progress.
#[derive(Default)]
struct SetupLines {
    /// Each component's archive size, from `SETUP_PLAN`.
    sizes: Vec<(String, u64)>,
    /// What will be downloaded in all, from `SETUP_TOTAL`.
    total: u64,
    /// The bytes of the components whose download has finished.
    finished: u64,
}

impl SetupLines {
    fn frac(&self, extra: u64) -> f32 {
        if self.total == 0 { 1.0 } else { ((self.finished + extra) as f64 / self.total as f64).min(1.0) as f32 }
    }

    fn size(&self, name: &str) -> u64 {
        self.sizes.iter().find(|(n, _)| n == name).map_or(0, |(_, b)| *b)
    }

    fn feed(&mut self, line: &str) -> Option<Feed> {
        let mut f = line.split(' ');
        match f.next()? {
            "SETUP_PLAN" => {
                let name = f.next()?.to_string();
                let bytes = f.next()?.parse().ok()?;
                if f.next()? == "download" {
                    self.sizes.push((name, bytes));
                }
                None
            }
            "SETUP_TOTAL" => {
                self.total = f.next()?.parse().ok()?;
                Some(Feed::Progress(Progress::Download { part: Part::Runtime, frac: 0.0 }))
            }
            "SETUP_DOWNLOAD" => {
                let name = f.next()?;
                let got: u64 = f.next()?.parse().ok()?;
                let part = if name.starts_with("models-") { Part::Models } else { Part::Runtime };
                Some(Feed::Progress(Progress::Download { part, frac: self.frac(got.min(self.size(name))) }))
            }
            "SETUP_EXTRACT" => {
                self.finished += self.size(f.next()?);
                Some(Feed::Progress(Progress::Unpack { frac: self.frac(0) }))
            }
            "SETUP_DONE" => Some(Feed::Done),
            _ => None,
        }
    }
}

fn download(url: &str, dest: &Path, expected: u64, cancel: &AtomicBool, progress: impl Fn(f32)) -> Result<(), String> {
    let resp = ureq::get(url)
        .header("User-Agent", UA)
        .config()
        .timeout_connect(Some(Duration::from_secs(30)))
        .build()
        .call()
        .map_err(|e| format!("downloading {url}: {e}"))?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(expected);
    let mut reader = resp.into_body().into_reader();
    let mut out = std::fs::File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let (mut got, mut buf) = (0u64, vec![0u8; 1 << 16]);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("downloading {url}: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", dest.display()))?;
        got += n as u64;
        if total > 0 {
            progress((got as f64 / total as f64).min(1.0) as f32);
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The file name of an archive entry, if it is a plain name (the mangajanai archives are flat).
fn flat_name(entry: &str) -> Option<String> {
    let name = Path::new(entry).file_name()?.to_string_lossy().into_owned();
    (!name.is_empty() && name != "." && name != "..").then_some(name)
}

/// Unpack the Windows archive's files into `dir`.
#[cfg_attr(not(windows), allow(dead_code))]
fn unpack_zip(archive: &Path, dir: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| format!("{}: {e}", archive.display()))?;
    for i in 0..zip.len() {
        let mut e = zip.by_index(i).map_err(|e| format!("{}: {e}", archive.display()))?;
        if e.is_dir() {
            continue;
        }
        let entry = e.name().map_err(|err| format!("{}: {err}", archive.display()))?.into_owned();
        let Some(name) = flat_name(&entry) else { continue };
        let dest = dir.join(&name);
        let mut out = std::fs::File::create(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
        std::io::copy(&mut e, &mut out).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    Ok(())
}

/// Unpack the Linux archive's files into `dir`, keeping their permissions (the program and libaji
/// are executable).
#[cfg_attr(windows, allow(dead_code))]
fn unpack_tar_zst(archive: &Path, dir: &Path) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let zst = ruzstd::decoding::StreamingDecoder::new(BufReader::new(f))
        .map_err(|e| format!("{}: {e}", archive.display()))?;
    let mut tar = tar::Archive::new(zst);
    for e in tar.entries().map_err(|e| format!("{}: {e}", archive.display()))? {
        let mut e = e.map_err(|e| format!("{}: {e}", archive.display()))?;
        if !e.header().entry_type().is_file() {
            continue;
        }
        let path = e.path().map_err(|e| format!("{}: {e}", archive.display()))?.into_owned();
        let Some(name) = flat_name(&path.to_string_lossy()) else { continue };
        let dest = dir.join(&name);
        // Replace, don't write through: a running copy of the old file keeps its inode.
        let _ = std::fs::remove_file(&dest);
        e.unpack(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(lines: &[&str]) -> (SetupLines, Vec<Progress>, bool) {
        let mut p = SetupLines::default();
        let (mut out, mut done) = (Vec::new(), false);
        for l in lines {
            match p.feed(l) {
                Some(Feed::Progress(x)) => out.push(x),
                Some(Feed::Done) => done = true,
                None => {}
            }
        }
        (p, out, done)
    }

    #[test]
    fn setup_lines_give_whole_install_progress() {
        let (_, out, done) = feed_all(&[
            "SETUP_PLAN dml-runtime 100 installed",
            "SETUP_PLAN trt-runtime 300 download",
            "SETUP_PLAN models-mangajanai 100 download",
            "SETUP_TOTAL 400",
            "SETUP_DOWNLOAD trt-runtime 0 300",
            "SETUP_DOWNLOAD trt-runtime 150 300",
            "SETUP_EXTRACT trt-runtime",
            "SETUP_DOWNLOAD models-mangajanai 100 100",
            "SETUP_EXTRACT models-mangajanai",
            "SETUP_DONE",
        ]);
        assert!(done);
        assert_eq!(
            out,
            [
                Progress::Download { part: Part::Runtime, frac: 0.0 },
                Progress::Download { part: Part::Runtime, frac: 0.0 },
                Progress::Download { part: Part::Runtime, frac: 0.375 },
                Progress::Unpack { frac: 0.75 },
                Progress::Download { part: Part::Models, frac: 1.0 },
                Progress::Unpack { frac: 1.0 },
            ]
        );
    }

    #[test]
    fn nothing_to_download_is_done_at_once() {
        let (_, out, done) = feed_all(&["SETUP_PLAN trt-runtime 300 installed", "SETUP_TOTAL 0", "SETUP_DONE"]);
        assert!(done);
        assert_eq!(out, [Progress::Download { part: Part::Runtime, frac: 0.0 }]);
    }

    #[test]
    fn unknown_and_short_lines_are_ignored() {
        let (_, out, done) = feed_all(&["GPU: something", "SETUP_PLAN", "SETUP_DOWNLOAD x", "SETUP_FUTURE a b"]);
        assert!(out.is_empty() && !done);
    }

    #[test]
    fn remove_keeps_only_rules_json() {
        let dir = std::env::temp_dir().join(format!("yosh-addon-remove-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::create_dir_all(dir.join("cache")).unwrap();
        for (f, n) in [("aji.dll", 3), ("models/a.onnx", 5), ("cache/x.engine", 7), (RECORD, 2)] {
            std::fs::write(dir.join(f), vec![0u8; n]).unwrap();
        }
        assert_eq!(folder_size(&dir), 17);
        remove(&dir).unwrap();
        assert!(!dir.exists(), "no rules.json: the folder goes too");

        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::write(dir.join("models/a.onnx"), b"x").unwrap();
        std::fs::write(dir.join("rules.json"), b"[]").unwrap();
        remove(&dir).unwrap();
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, ["rules.json"]);
        let _ = std::fs::remove_dir_all(&dir);
        // A folder that's already gone is removed.
        remove(&dir).unwrap();
    }

    #[test]
    fn setup_args_choose_store_mode_for_installed_yosh_only() {
        let dir = Path::new("addon");
        let s = |v: Vec<std::ffi::OsString>| v.into_iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let plain = ["setup", "--progress", "lines"];
        if std::env::var_os("YOSH_ADDON_MIRROR").is_none() {
            assert_eq!(s(setup_args(dir, true, true)), plain, "portable: private");
            assert_eq!(s(setup_args(dir, false, false)), plain, "a mangajanai without --store: private");
            let store = s(setup_args(dir, false, true));
            assert_eq!(&store[..3], plain);
            assert_eq!(&store[3..6], ["--store", "--app-id", "yosh-addon"]);
            assert_eq!(store[6], "--app-path");
            assert_eq!(Path::new(&store[7]), dir.join(PROGRAM), "the ref lives as long as the add-on's program");
        }
    }

    #[test]
    fn store_installs_are_recognised_from_setup_json() {
        let dir = std::env::temp_dir().join(format!("yosh-addon-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!store_install(&dir));
        std::fs::write(dir.join("setup.json"), br#"{"components":{}}"#).unwrap();
        assert!(!store_install(&dir));
        std::fs::write(dir.join("setup.json"), br#"{"components":{},"store":{"app_id":"yosh-addon","app_path":"x"}}"#).unwrap();
        assert!(store_install(&dir));
        // remove() of a store install without its program still removes the folder
        remove(&dir).unwrap();
        assert!(!dir.exists());
    }

    #[test]
    fn a_gpu_setup_never_saw_needs_setup() {
        let rec = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(!lacks_gpu(&rec(&["12.0"]), &rec(&["12.0"])));
        // Moved from an RTX 50 PC to an RTX 30 one.
        assert!(lacks_gpu(&rec(&["12.0"]), &rec(&["8.6"])));
        // A second card of another generation.
        assert!(lacks_gpu(&rec(&["12.0"]), &rec(&["12.0", "8.9"])));
        // A card taken out, or no NVIDIA GPU (DirectML only): nothing to fetch.
        assert!(!lacks_gpu(&rec(&["12.0", "8.9"]), &rec(&["12.0"])));
        assert!(!lacks_gpu(&rec(&["12.0"]), &[]));
        // Set up on a PC without NVIDIA, now on one with.
        assert!(lacks_gpu(&[], &rec(&["8.6"])));
    }

    #[test]
    fn record_without_gpus_still_reads() {
        let r: Record = serde_json::from_str(r#"{"cli": "v0.3.1"}"#).unwrap();
        assert_eq!(r.cli, "v0.3.1");
        assert!(r.gpus.is_none());
    }

    #[test]
    fn archive_entries_unpack_flat() {
        assert_eq!(flat_name("./mangajanai").as_deref(), Some("mangajanai"));
        assert_eq!(flat_name("aji.dll").as_deref(), Some("aji.dll"));
        assert_eq!(flat_name("../../evil.dll").as_deref(), Some("evil.dll"));
        assert_eq!(flat_name("."), None);
    }

    /// The real release archives unpack (set YOSH_TEST_CLI_ARCHIVES to a folder holding them).
    #[test]
    fn release_archives_unpack() {
        let Some(src) = std::env::var_os("YOSH_TEST_CLI_ARCHIVES") else { return };
        let src = PathBuf::from(src);
        let out = std::env::temp_dir().join(format!("yosh-addon-unpack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        unpack_tar_zst(&src.join("mangajanai-linux-x64.tar.zst"), &out).unwrap();
        for f in ["mangajanai", "libaji.so", "libaji_trt.so"] {
            assert!(out.join(f).is_file(), "{f}");
        }
        unpack_zip(&src.join("mangajanai-windows-x64.zip"), &out).unwrap();
        for f in ["mangajanai.exe", "aji.dll", "aji_trt.dll", "aji_dml.dll", "vcruntime140.dll"] {
            assert!(out.join(f).is_file(), "{f}");
        }
        let _ = std::fs::remove_dir_all(&out);
    }
}

/// A whole install, through `start`, into the folder in YOSH_TEST_ADDON_DIR (with
/// YOSH_ADDON_ARCHIVE pointing at a local mangajanai archive, or the pinned release).
/// Downloads about 1 GB: run by hand with `--ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn install_end_to_end() {
    let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON_DIR").expect("YOSH_TEST_ADDON_DIR"));
    let job = start(dir.clone(), Mode::Full, Arc::new(|| {}));
    let mut last = String::new();
    loop {
        match job.rx.recv().expect("install thread") {
            Event::Progress(p) => {
                let s = format!("{p:?}");
                let s = s.split('.').next().unwrap_or_default().to_string() + &format!("{:?}", match p {
                    Progress::Download { frac, .. } | Progress::Unpack { frac } => (frac * 20.0) as u32,
                    Progress::Starting => 0,
                });
                if s != last {
                    eprintln!("{p:?}");
                    last = s;
                }
            }
            Event::Done => break,
            Event::Cancelled => panic!("cancelled"),
            Event::Failed(e) => panic!("{e}"),
        }
    }
    assert_eq!(installed_release(&dir).as_deref(), Some(CLI_VERSION));
    assert!(!incomplete(&dir) && crate::addon::installed(&dir));
}

/// After `install_end_to_end` in store mode (an installed yosh, AJI_RUNTIME_STORE pointing at a
/// test store): the add-on finds its models in the store, the upscaler starts on the store's
/// runtime, and *Remove the add-on* drops its ref, freeing the packs. Run by hand with
/// `--ignored --nocapture`.
#[cfg(all(test, feature = "sr"))]
#[test]
#[ignore]
fn store_addon_end_to_end() {
    let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON_DIR").expect("YOSH_TEST_ADDON_DIR"));
    let store = PathBuf::from(std::env::var_os("AJI_RUNTIME_STORE").expect("AJI_RUNTIME_STORE (never the real store)"));
    assert!(store_install(&dir), "not a store install");
    let dirs = crate::addon::models_dirs(&dir);
    eprintln!("model folders: {dirs:?}");
    assert!(dirs.len() > 1 && dirs[1..].iter().all(|d| d.starts_with(&store)));
    let rules = crate::profiles::Profiles::default().rules(crate::profiles::ProfileId::Balanced, &dirs).unwrap();
    for r in &rules {
        if let Some(m) = &r.model {
            assert!(crate::profiles::model_installed(&dirs, m), "{m} not found");
        }
    }
    let rx = crate::addon::start(dir.clone(), crate::config::UpscaleBackend::Auto, rules, Arc::new(|| {}));
    match rx.recv().expect("upscaler thread") {
        Ok(r) => eprintln!("upscaler running on {}", r.device),
        Err(e) => panic!("upscaler: {e}"),
    }
    assert!(store.join("refs").join(format!("{STORE_APP_ID}.json")).is_file());
}

/// *Remove the add-on* on a store install (after `store_addon_end_to_end`, in a process of its
/// own: yosh removes before libaji loads): the ref goes, and with no other app the store empties.
#[cfg(test)]
#[test]
#[ignore]
fn store_remove_end_to_end() {
    let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON_DIR").expect("YOSH_TEST_ADDON_DIR"));
    let store = PathBuf::from(std::env::var_os("AJI_RUNTIME_STORE").expect("AJI_RUNTIME_STORE (never the real store)"));
    assert!(store_install(&dir));
    remove(&dir).unwrap();
    assert!(!dir.join(PROGRAM).exists());
    assert!(!store.join("refs").join(format!("{STORE_APP_ID}.json")).exists(), "ref dropped");
    let left: Vec<_> = std::fs::read_dir(store.join("packs")).map(|r| r.flatten().map(|e| e.file_name()).collect()).unwrap_or_default();
    assert!(left.is_empty(), "packs left: {left:?}");
}

/// "Install for this GPU" on an add-on in YOSH_TEST_ADDON_DIR whose record and setup.json were
/// edited to look set up for another GPU: the check sees it, setup-only fetches what's missing,
/// and the record then matches this machine. Run by hand with `--ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn setup_for_gpus_end_to_end() {
    let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON_DIR").expect("YOSH_TEST_ADDON_DIR"));
    assert!(needs_setup_for_gpus(&dir), "the record should look like another GPU's");
    let before = installed_release(&dir);
    let job = start(dir.clone(), Mode::SetupOnly, Arc::new(|| {}));
    loop {
        match job.rx.recv().expect("install thread") {
            Event::Progress(p) => eprintln!("{p:?}"),
            Event::Done => break,
            Event::Cancelled => panic!("cancelled"),
            Event::Failed(e) => panic!("{e}"),
        }
    }
    assert!(!needs_setup_for_gpus(&dir));
    assert_eq!(installed_release(&dir), before, "setup-only keeps the release");
}

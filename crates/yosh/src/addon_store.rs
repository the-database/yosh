//! Where the shared pack store is, and moving it (Settings → AI upscaling → Storage), for an add-on
//! whose setup put its packs there (`addon_setup.rs`, store mode).
//!
//! The store (TensorRT, its GPU files, TensorRT-RTX and the models every JaNai app shares) is at
//! `%LOCALAPPDATA%\the-database\aji` (`$XDG_DATA_HOME/the-database/aji`). Moving it to another drive
//! leaves a link at that path (a junction on Windows, a symlink on Linux), so every app keeps
//! reaching it there. yosh doesn't link aji-runtime, which does the move (its `zip` dependency
//! needs a C toolchain): it runs the add-on's `mangajanai store ...` (MangaJaNaiCLI, `store.rs`)
//! without a console window, as MangaJaNai GUI runs `aji-runtime`:
//!
//! - `store location --json`: `{"root", "moved", "target", "available"}`. An add-on whose
//!   mangajanai predates `store` fails it, and Settings shows no Storage row.
//! - `store move FOLDER --progress lines`, `store move --default --progress lines`: progress as
//!   `STORE_MOVE <copied> <total>` lines, `STORE_DONE` at the end.
//! - `store reset`: drop a moved store whose folder is gone; setup then installs the add-on's packs
//!   again into an empty store at the default path.
//!
//! On Windows a move fails while any app has a file loaded from the store, yosh included (libaji
//! loads TensorRT from it), so a move asked for after an upscaler started runs at the next start
//! (`Settings::addon_store_move_pending`). While a moved store's folder is missing (an unplugged
//! drive) the upscaler doesn't start ([`missing_folder`]).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::Receiver;

use yosh_engine::pool::Waker;

use crate::addon_setup;

/// Where the store is, from `mangajanai store location --json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// The default path (a link when moved).
    pub root: PathBuf,
    pub moved: bool,
    /// The folder a moved store is in.
    pub target: Option<PathBuf>,
    /// False when a moved store's folder is missing.
    pub available: bool,
    /// The store's size on disk (0 when unknown or missing).
    pub bytes: u64,
}

impl Info {
    /// The folder the store's files are in.
    pub fn folder(&self) -> &Path {
        match &self.target {
            Some(t) if self.moved => t,
            _ => &self.root,
        }
    }

    pub fn missing(&self) -> bool {
        self.moved && !self.available
    }
}

/// Reads `store location --json`'s output.
pub fn parse_location(json: &str) -> Option<Info> {
    let j: serde_json::Value = serde_json::from_str(json).ok()?;
    Some(Info {
        root: PathBuf::from(j.get("root")?.as_str()?),
        moved: j.get("moved").and_then(|v| v.as_bool()).unwrap_or(false),
        target: j.get("target").and_then(|v| v.as_str()).map(PathBuf::from),
        available: j.get("available").and_then(|v| v.as_bool()).unwrap_or(true),
        bytes: 0,
    })
}

/// The store's default path, as aji-runtime resolves it (`AJI_RUNTIME_STORE` for tests).
pub fn default_root() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AJI_RUNTIME_STORE").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let base = if cfg!(windows) {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else {
        match std::env::var_os("XDG_DATA_HOME").filter(|p| !p.is_empty()) {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(std::env::var_os("HOME")?).join(".local").join("share"),
        }
    };
    Some(base.join("the-database").join("aji"))
}

/// When the store at `root` is a link to a folder that's gone (a moved store on an unplugged
/// drive): that folder. A couple of stats, no subprocess, so the upscaler start can check it.
pub fn missing_folder(root: &Path) -> Option<PathBuf> {
    let meta = std::fs::symlink_metadata(root).ok()?;
    if !meta.file_type().is_symlink() || root.is_dir() {
        return None;
    }
    let t = std::fs::read_link(root).ok()?;
    // Junctions read back with the NT prefix.
    let s = t.to_string_lossy();
    Some(s.strip_prefix(r"\??\").or_else(|| s.strip_prefix(r"\\?\")).map(PathBuf::from).unwrap_or(t))
}

/// Moving into `folder`: the store needs a new or empty folder, so one with files in it gets a
/// `JaNai shared runtime` folder of its own inside (as MangaJaNai GUI does).
pub fn move_target(folder: &Path) -> PathBuf {
    let empty = std::fs::read_dir(folder).map_or(true, |mut d| d.next().is_none());
    if empty { folder.to_path_buf() } else { folder.join("JaNai shared runtime") }
}

/// Asks the add-on in `dir` where the store is, and its size, on a background thread. `None`
/// when its mangajanai has no `store` command (an add-on from before it) or the query failed.
pub fn query(dir: PathBuf, wake: Waker) -> Receiver<Option<Info>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let info = addon_setup::program(&dir)
            .args(["store", "location", "--json"])
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| parse_location(&String::from_utf8_lossy(&o.stdout)))
            .map(|mut i| {
                if !i.missing() {
                    i.bytes = addon_setup::folder_size(i.folder());
                }
                i
            });
        let _ = tx.send(info);
        wake();
    });
    rx
}

/// What a Storage button does.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Move the store into this folder (new or empty).
    Move(PathBuf),
    /// Move it back to the default path.
    MoveBack,
    /// Drop a moved store whose folder is missing.
    Reset,
}

impl Action {
    fn args(&self) -> Vec<std::ffi::OsString> {
        let mut a: Vec<std::ffi::OsString> = vec!["store".into()];
        match self {
            Action::Move(to) => a.extend(["move".into(), to.into(), "--progress".into(), "lines".into()]),
            Action::MoveBack => a.extend(["move", "--default", "--progress", "lines"].map(Into::into)),
            Action::Reset => a.push("reset".into()),
        }
        a
    }

    /// As `Settings::addon_store_move_pending` keeps a move for the next start ("" = back to the
    /// default path); `None` for Reset, which never waits.
    pub fn pending_token(&self) -> Option<String> {
        match self {
            Action::Move(to) => Some(to.to_string_lossy().into_owned()),
            Action::MoveBack => Some(String::new()),
            Action::Reset => None,
        }
    }

    pub fn from_pending(token: &str) -> Action {
        if token.is_empty() { Action::MoveBack } else { Action::Move(PathBuf::from(token)) }
    }
}

pub enum Event {
    /// Bytes copied so far, of the total.
    Progress(u64, u64),
    Done,
    Failed(String),
}

/// A running `mangajanai store` command. Events arrive on `rx`, each followed by a call to the
/// waker. A move can't be cancelled: aji-runtime cleans up only on its own errors.
pub struct Job {
    pub rx: Receiver<Event>,
    pub action: Action,
}

/// One `STORE_MOVE <copied> <total>` line.
pub fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let mut f = line.split(' ');
    if f.next()? != "STORE_MOVE" {
        return None;
    }
    let done = f.next()?.parse().ok()?;
    let total = f.next()?.parse().ok()?;
    f.next().is_none().then_some((done, total))
}

/// Runs `action` with the add-on in `dir` on a background thread.
pub fn start(dir: PathBuf, action: Action, wake: Waker) -> Job {
    let (tx, rx) = std::sync::mpsc::channel();
    let args = action.args();
    std::thread::spawn(move || {
        let send = |e: Event| {
            let _ = tx.send(e);
            wake();
        };
        let ev = match run(&dir, &args, &|done, total| send(Event::Progress(done, total))) {
            Ok(()) => Event::Done,
            Err(e) => {
                eprintln!("yosh: [addon] store {args:?} failed: {e}");
                Event::Failed(e)
            }
        };
        send(ev);
    });
    Job { rx, action }
}

fn run(dir: &Path, args: &[std::ffi::OsString], progress: &dyn Fn(u64, u64)) -> Result<(), String> {
    let mut c = addon_setup::program(dir)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("mangajanai: {e}"))?;
    let stdout = c.stdout.take();
    let stderr = c.stderr.take();
    // stderr on its own thread so neither pipe can fill up; its last `error:` line is the reason.
    let err_thread = std::thread::spawn(move || {
        let mut last = String::new();
        if let Some(e) = stderr {
            for line in BufReader::new(e).lines().map_while(Result::ok) {
                if let Some(msg) = line.trim().strip_prefix("error: ") {
                    last = msg.to_string();
                }
            }
        }
        last
    });
    if let Some(out) = stdout {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if let Some((done, total)) = parse_progress(&line) {
                progress(done, total);
            }
        }
    }
    let status = c.wait().map_err(|e| format!("mangajanai: {e}"))?;
    let err = err_thread.join().unwrap_or_default();
    match status.code() {
        Some(0) => Ok(()),
        _ if !err.is_empty() => Err(err),
        code => Err(format!("mangajanai store exited with code {}", code.unwrap_or(-1))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_json_reads() {
        let i = parse_location(r#"{"available": true, "moved": false, "root": "C:\\u\\aji", "target": null}"#).unwrap();
        assert_eq!(i.folder(), Path::new(r"C:\u\aji"));
        assert!(!i.moved && !i.missing());
        let i = parse_location(r#"{"available": true, "moved": true, "root": "/h/aji", "target": "/mnt/d/aji"}"#).unwrap();
        assert_eq!(i.folder(), Path::new("/mnt/d/aji"));
        assert!(i.moved && !i.missing());
        let i = parse_location(r#"{"available": false, "moved": true, "root": "/h/aji", "target": "/mnt/d/aji"}"#).unwrap();
        assert!(i.missing());
        // An older mangajanai's output, or none.
        assert!(parse_location("default /h/aji").is_none());
        assert!(parse_location("").is_none());
        assert!(parse_location(r#"{"moved": true}"#).is_none());
    }

    #[test]
    fn progress_lines_read() {
        assert_eq!(parse_progress("STORE_MOVE 0 4000"), Some((0, 4000)));
        assert_eq!(parse_progress("STORE_MOVE 4000 4000"), Some((4000, 4000)));
        for l in ["STORE_DONE", "STORE_WAIT", "STORE_MOVE", "STORE_MOVE 1", "STORE_MOVE a b", "STORE_MOVE 1 2 3", "SETUP_DONE"] {
            assert_eq!(parse_progress(l), None, "{l}");
        }
    }

    #[test]
    fn commands_are_the_clis() {
        let s = |a: Action| a.args().into_iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
        assert_eq!(s(Action::Move(PathBuf::from("D:/x"))), ["store", "move", "D:/x", "--progress", "lines"]);
        assert_eq!(s(Action::MoveBack), ["store", "move", "--default", "--progress", "lines"]);
        assert_eq!(s(Action::Reset), ["store", "reset"]);
    }

    #[test]
    fn a_pending_move_round_trips() {
        for a in [Action::Move(PathBuf::from("D:/JaNai")), Action::MoveBack] {
            assert_eq!(Action::from_pending(&a.pending_token().unwrap()), a);
        }
        assert_eq!(Action::Reset.pending_token(), None);
    }

    #[test]
    fn a_folder_with_files_gets_its_own_subfolder() {
        let dir = std::env::temp_dir().join(format!("yosh-store-target-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(move_target(&dir), dir, "a new folder");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(move_target(&dir), dir, "an empty folder");
        std::fs::write(dir.join("x"), b"x").unwrap();
        assert_eq!(move_target(&dir), dir.join("JaNai shared runtime"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_link_to_a_missing_folder_is_missing() {
        let dir = std::env::temp_dir().join(format!("yosh-store-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (root, target) = (dir.join("aji"), dir.join("drive"));
        std::fs::create_dir_all(&target).unwrap();
        assert_eq!(missing_folder(&root), None, "nothing there yet");
        #[cfg(windows)]
        let linked = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&root)
            .arg(&target)
            .stdout(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        #[cfg(not(windows))]
        let linked = std::os::unix::fs::symlink(&target, &root).is_ok();
        assert!(linked);
        assert_eq!(missing_folder(&root), None, "the folder is there");
        std::fs::rename(&target, dir.join("unplugged")).unwrap();
        assert_eq!(missing_folder(&root).as_deref(), Some(target.as_path()));
        let _ = std::fs::remove_dir_all(&dir);
        // A real folder is never missing.
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(missing_folder(&root), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The Storage row's commands against a real mangajanai with `store`: the add-on folder in
/// YOSH_TEST_ADDON_DIR (its mangajanai is all it needs) and a test store in AJI_RUNTIME_STORE
/// (never the real one) holding a few files. Moves it into YOSH_TEST_STORE_DEST (new or empty)
/// and back. Run by hand with `--ignored --nocapture`.
#[cfg(test)]
#[test]
#[ignore]
fn store_move_end_to_end() {
    let dir = PathBuf::from(std::env::var_os("YOSH_TEST_ADDON_DIR").expect("YOSH_TEST_ADDON_DIR"));
    let root = PathBuf::from(std::env::var_os("AJI_RUNTIME_STORE").expect("AJI_RUNTIME_STORE (never the real store)"));
    let dest = PathBuf::from(std::env::var_os("YOSH_TEST_STORE_DEST").expect("YOSH_TEST_STORE_DEST"));
    let wake: Waker = std::sync::Arc::new(|| {});
    let info = || query(dir.clone(), wake.clone()).recv().unwrap().expect("store location");
    let run = |a: Action| {
        let job = start(dir.clone(), a, wake.clone());
        let mut last = None;
        loop {
            match job.rx.recv().unwrap() {
                Event::Progress(d, t) => last = Some((d, t)),
                Event::Done => return Ok(last),
                Event::Failed(e) => return Err(e),
            }
        }
    };
    let before = info();
    eprintln!("{before:?}");
    assert!(!before.moved && before.bytes > 0);
    assert_eq!(before.root, root);
    let (done, total) = run(Action::Move(dest.clone())).unwrap().expect("progress");
    assert_eq!(done, total);
    let moved = info();
    eprintln!("{moved:?}");
    assert!(moved.moved && !moved.missing() && moved.folder() == dest && moved.bytes == before.bytes);
    assert_eq!(missing_folder(&root), None);
    let e = run(Action::Reset).unwrap_err();
    eprintln!("reset while there: {e}");
    assert!(e.contains("move it back instead"));
    run(Action::MoveBack).unwrap();
    let back = info();
    assert!(!back.moved && back.bytes == before.bytes && !dest.exists());
}

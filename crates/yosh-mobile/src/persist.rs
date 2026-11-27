//! Persisted state: positions, progress, recents, view options (debounced atomic saves).

use super::*;

/// Cap on the most-recently-read list (`recents.txt`); the head is the resume target.
pub(crate) const RECENTS_CAP: usize = 32;

/// How long a dirty persisted-state file waits before [`Shell::flush_saves`] writes
/// it, so a burst of page turns costs one write instead of one per flip.
const SAVE_DEBOUNCE: Duration = Duration::from_secs(2);

impl Shell {
    /// Note that the reading position — and the progress map that rides with it —
    /// changed. The write itself is debounced; see [`Shell::flush_saves`].
    pub(crate) fn mark_positions(&mut self) {
        self.dirty_positions = true;
        self.dirty_progress = true;
        self.dirty_since.get_or_insert_with(Instant::now);
    }

    /// Write out whatever has been marked dirty. `force` writes immediately (book
    /// switch, suspend, teardown); otherwise nothing happens until the oldest mark
    /// is [`SAVE_DEBOUNCE`] old, so a burst of page turns costs one write.
    ///
    /// Each file is rewritten whole through a `<path>.tmp` + rename, so a kill in
    /// the middle of a write leaves the previous file intact rather than a truncated
    /// one (the model is `thumbcache::store`). The writes stay on the calling
    /// thread: they are small, now rare, and `suspended()` must have them on disk
    /// before it returns.
    pub(crate) fn flush_saves(&mut self, force: bool) {
        let Some(since) = self.dirty_since else { return };
        if !force && since.elapsed() < SAVE_DEBOUNCE {
            return;
        }
        self.dirty_since = None;
        if std::mem::take(&mut self.dirty_positions)
            && let Some(path) = &self.pos_path
        {
            let mut out = String::new();
            for (k, v) in &self.positions {
                out.push_str(&format!("{v}\t{k}\n"));
            }
            write_atomic(path, &out);
        }
        if std::mem::take(&mut self.dirty_progress)
            && let Some(path) = &self.progress_path
        {
            let mut out = String::new();
            for (k, (furthest, total)) in &self.progress {
                out.push_str(&format!("{furthest}\t{total}\t{k}\n"));
            }
            write_atomic(path, &out);
        }
        if std::mem::take(&mut self.dirty_recents)
            && let Some(path) = &self.recents_path
        {
            write_atomic(path, &self.recents.join("\n"));
        }
        if std::mem::take(&mut self.dirty_collapsed)
            && let Some(path) = &self.collapsed_path
        {
            let mut out = String::new();
            for p in &self.collapsed {
                out.push_str(&format!("{}\n", p.display()));
            }
            write_atomic(path, &out);
        }
        // The viewing options and the library root live on the App (which the
        // egui frame mutates); read the current values back off it at write time.
        if std::mem::take(&mut self.dirty_view)
            && let (Some(path), Some(app)) = (&self.view_file, &self.app)
        {
            save_view(
                path,
                app.reader.direction,
                app.layout_mode,
                app.reader.fit,
                app.reader.transition_enabled,
                app.reader.scroll_mode,
                app.resume_on_startup,
                app.theme,
                app.perf,
                app.spine_shadow_on,
                app.spine_shadow_strength,
                app.reader.fit_no_upscale,
                app.vol_keys,
                app.lang_pref,
            );
        }
        if std::mem::take(&mut self.dirty_libroot)
            && let (Some(path), Some(app)) = (&self.lib_dir_file, &self.app)
        {
            write_atomic(path, &app.lib_root.to_string_lossy());
        }
    }

    /// Recompute, off-thread, which `recents` entries still exist on disk.
    pub(crate) fn refresh_recents_ok(&self) {
        let keys = self.recents.clone();
        let tx = self.recents_ok_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let ok: HashSet<String> = keys.into_iter().filter(|k| is_reopenable_fs(k)).collect();
            let _ = tx.send(ok);
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Update the in-memory read-tracking entry for the open comic: the furthest
    /// page the reader has had on screen (1-based; the far page of a spread
    /// counts) and the volume's total. Called every rendered frame while reading
    /// — cheap map math; the file write happens with `save_positions`.
    pub(crate) fn note_progress(&mut self) {
        // `as_deref`, not `clone`: this runs every rendered frame, and the owned
        // key is only ever needed the first time a book is seen.
        let (Some(key), Some(app)) = (self.current_key.as_deref(), self.app.as_ref()) else {
            return;
        };
        let Some(src) = &app.reader.source else { return };
        let len = src.len();
        if len == 0 {
            return;
        }
        let (a, b) = app.reader.visible_pages();
        let seen = (b.unwrap_or(a) + 1) as u32;
        // `key` borrows `current_key`, the update touches `progress` — disjoint
        // fields, so no clone is needed to satisfy the borrow checker either.
        match self.progress.get_mut(key) {
            Some(e) => {
                e.0 = e.0.max(seen);
                e.1 = len as u32;
            }
            None => {
                self.progress.insert(key.to_string(), (seen, len as u32));
            }
        }
    }
}

/// Rewrite `path` atomically: write a sibling `<path>.tmp`, then rename it over the
/// target. A rename within a directory is atomic, so a kill (or a battery pull)
/// mid-write leaves the previous file whole instead of a half-written one — these
/// are whole-file rewrites of state the user would notice losing.
fn write_atomic(path: &Path, contents: &str) {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// The persisted viewing options, in `view.txt`'s positional order: direction,
/// layout mode, fit, page-turn animation, scroll mode, resume-on-startup, chrome
/// theme, performance profile, spine shadow on/off, spine-shadow strength,
/// no-upscale fit, volume keys turn pages.
pub(crate) type InitView = (
    Direction,
    LayoutMode,
    FitMode,
    bool,
    bool,
    bool,
    ThemePref,
    PerfPref,
    bool,
    f32,
    bool,
    VolKeys,
    Option<Lang>,
);

/// Load the persisted per-comic positions (one `index\tkey` line each).
/// Parse persisted viewing options ("dir,layout,fit,anim"); default RTL / single /
/// window / animation on.
pub(crate) fn load_view(path: &Path) -> InitView {
    let (mut dir, mut lay, mut fit, mut anim, mut scroll, mut resume, mut theme, mut perf) = (
        Direction::Rtl,
        LayoutMode::Single,
        FitMode::Window,
        true,
        false,
        true,
        ThemePref::System,
        PerfPref::Auto,
    );
    let (mut spine, mut spine_strength) = (false, DEFAULT_SPINE_STRENGTH);
    let mut no_upscale = false;
    let mut vol_keys = VolKeys::Off;
    let mut lang = None;
    if let Ok(s) = std::fs::read_to_string(path) {
        let t: Vec<&str> = s.trim().split(',').collect();
        if t.first() == Some(&"ltr") {
            dir = Direction::Ltr;
        }
        // Layout slot gained "auto"; "single"/"spread" (and old files) still parse.
        lay = match t.get(1) {
            Some(&"spread") => LayoutMode::Spread,
            Some(&"auto") => LayoutMode::Auto,
            _ => LayoutMode::Single,
        };
        fit = match t.get(2) {
            Some(&"width") => FitMode::Width,
            Some(&"height") => FitMode::Height,
            Some(&"actual") => FitMode::Actual,
            _ => FitMode::Window,
        };
        // Page-turn animation: 4th slot; absent (older files) ⇒ on.
        anim = t.get(3) != Some(&"off");
        // 5th slot: scroll mode; absent ⇒ page-flip.
        scroll = t.get(4) == Some(&"scroll");
        // 6th slot: resume-on-launch; absent ⇒ on.
        resume = t.get(5) != Some(&"noresume");
        // 7th slot: chrome theme; absent (older files) ⇒ follow the system.
        theme = ThemePref::parse(t.get(6));
        // 8th slot: performance profile; absent (older files) ⇒ auto (the probed tier).
        perf = PerfPref::parse(t.get(7));
        // 9th slot: spine shadow; absent (older files) ⇒ off.
        spine = t.get(8) == Some(&"spine");
        // 10th slot: spine-shadow strength as a whole percent; absent ⇒ the default.
        spine_strength = t
            .get(9)
            .and_then(|v| v.parse::<i32>().ok())
            .map(|p| p.clamp(0, 100) as f32 / 100.0)
            .unwrap_or(DEFAULT_SPINE_STRENGTH);
        // 11th slot: no-upscale fit; absent (older files) ⇒ off (stretch to fit).
        no_upscale = t.get(10) == Some(&"noupscale");
        // 12th slot: volume keys turn pages (and which way); absent (older files) ⇒ off.
        vol_keys = VolKeys::parse(t.get(11));
        // 13th slot: UI language token; absent or "system" ⇒ follow the OS.
        lang = t.get(12).and_then(|tok| Lang::from_token(tok));
    }
    (dir, lay, fit, anim, scroll, resume, theme, perf, spine, spine_strength, no_upscale, vol_keys, lang)
}

/// Persist viewing options as
/// "dir,layout,fit,anim,scroll,resume,theme,perf,spine,spine_strength,no_upscale,vol_keys,lang".
#[allow(clippy::too_many_arguments)]
fn save_view(
    path: &Path,
    dir: Direction,
    lay: LayoutMode,
    fit: FitMode,
    anim: bool,
    scroll: bool,
    resume: bool,
    theme: ThemePref,
    perf: PerfPref,
    spine: bool,
    spine_strength: f32,
    no_upscale: bool,
    vol_keys: VolKeys,
    lang: Option<Lang>,
) {
    let d = if dir == Direction::Rtl { "rtl" } else { "ltr" };
    let l = lay.label();
    let f = match fit {
        FitMode::Width => "width",
        FitMode::Height => "height",
        FitMode::Actual => "actual",
        FitMode::Window => "window",
    };
    let a = if anim { "on" } else { "off" };
    let s = if scroll { "scroll" } else { "flip" };
    let r = if resume { "resume" } else { "noresume" };
    let t = theme.label();
    let p = perf.label();
    let sp = if spine { "spine" } else { "nospine" };
    let ss = (spine_strength.clamp(0.0, 1.0) * 100.0).round() as i32;
    let nu = if no_upscale { "noupscale" } else { "stretch" };
    let vk = vol_keys.label();
    let lg = lang.map_or("system", Lang::token);
    write_atomic(path, &format!("{d},{l},{f},{a},{s},{r},{t},{p},{sp},{ss},{nu},{vk},{lg}"));
}

pub(crate) fn load_progress(path: &std::path::Path) -> HashMap<String, (u32, u32)> {
    let mut map = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for line in s.lines() {
            // `furthest \t total \t key` — key last (it's the free-form field).
            if let Some((furthest, rest)) = line.split_once('\t')
                && let Some((total, key)) = rest.split_once('\t')
                && let (Ok(f), Ok(t)) = (furthest.parse(), total.parse())
            {
                map.insert(key.to_string(), (f, t));
            }
        }
    }
    map
}

pub(crate) fn load_collapsed(path: &std::path::Path) -> std::collections::HashSet<PathBuf> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

/// Load the most-recently-read list (one volume key per line, newest first).
pub(crate) fn load_recents(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.is_empty()).map(String::from).collect())
        .unwrap_or_default()
}

/// True if `key` is a filesystem path we can silently rebuild a source from on
/// launch (not a picked document, and still present on disk).
pub(crate) fn is_reopenable_fs(key: &str) -> bool {
    !is_document_key(key) && std::path::Path::new(key).exists()
}

/// True if `key` names a document the platform picker handed us (a SAF
/// `content://` URI) rather than a filesystem path: it has no folder to walk for
/// next/prev book and can't be silently reopened on launch.
pub(crate) fn is_document_key(key: &str) -> bool {
    key.starts_with("content://")
}

pub(crate) fn load_positions(path: &std::path::Path) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for line in s.lines() {
            if let Some((idx, key)) = line.split_once('\t')
                && let Ok(i) = idx.parse()
            {
                map.insert(key.to_string(), i);
            }
        }
    }
    map
}

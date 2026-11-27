//! Opening books: page sources, the async open worker, next/prev book.

use super::*;

/// Result of a background open: the comic's identity key (path or content:// URI)
/// and its page source, or a message to log. The key travels *with* the result
/// because the launch resume also decides which recent to open off-thread.
pub(crate) type OpenResult = Result<(String, Arc<dyn PageSource>), String>;

/// Outcome of resolving the current book's neighbour in its folder. `Cold` means
/// the listing isn't cached yet — a background scan has been kicked and the caller
/// should park its request rather than scan on the UI thread.
pub(crate) enum SibLookup {
    Cold,
    Missing,
    Found(PathBuf),
}

impl Shell {
    /// The previous/next comic in the current comic's folder (`dir` = -1/+1),
    /// natural-sorted like the library. `Missing` at the ends or for non-path
    /// (SAF-picked) sources, whose keys aren't filesystem paths.
    ///
    /// Reads the background-warmed [`Shell::sib_cache`] only. On a miss it kicks the
    /// scan and answers `Cold` — a `read_dir` plus a stat per entry on the UI thread
    /// is precisely the hitch this phase exists to remove.
    pub(crate) fn sibling_book_path(&mut self, dir: i64) -> SibLookup {
        // Base "current" on the pending target while an open is in flight, so a
        // second next-book tap advances from the not-yet-loaded neighbour instead
        // of asking for it again.
        let Some(cur) = self.opening_key.clone().or_else(|| self.current_key.clone()) else {
            return SibLookup::Missing;
        };
        let cur_path = PathBuf::from(&cur);
        let Some(parent) = cur_path.parent().map(|p| p.to_path_buf()) else {
            return SibLookup::Missing;
        };
        if is_document_key(&cur) {
            return SibLookup::Missing; // SAF one-off: no folder to walk
        }
        if !matches!(&self.sib_cache, Some((p, _)) if *p == parent) {
            self.warm_sib_cache(&cur_path);
            return SibLookup::Cold;
        }
        let comics = &self.sib_cache.as_ref().unwrap().1;
        let Some(i) = comics.iter().position(|p| *p == cur_path) else {
            return SibLookup::Missing;
        };
        let j = i as i64 + dir;
        match (0..comics.len() as i64).contains(&j) {
            true => SibLookup::Found(comics[j as usize].clone()),
            false => SibLookup::Missing,
        }
    }

    /// Scan the current book's folder for sibling comics on a background thread and
    /// hand the listing back for `sib_cache` (consumed in `poll_background`).
    fn warm_sib_cache(&self, vol: &Path) {
        let Some(parent) = vol.parent().map(|p| p.to_path_buf()) else {
            return;
        };
        let tx = self.sib_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let comics: Vec<PathBuf> = scan_dir(&parent)
                .into_iter()
                .filter_map(|e| match e {
                    Entry::Comic(p) => Some(p),
                    _ => None,
                })
                .collect();
            // Send, then wake: a parked jump replays on the frame this schedules.
            let _ = tx.send((parent, comics));
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Open the previous/next comic in the current comic's folder (`dir` = -1/+1).
    /// No-op at the ends or for non-path sources. With a cold cache the request is
    /// parked (accumulating, so repeated taps travel that many books) and replayed
    /// when the listing lands.
    pub(crate) fn open_sibling_book(&mut self, dir: i64) {
        match self.sibling_book_path(dir) {
            SibLookup::Found(p) => self.open_path(p),
            SibLookup::Cold => {
                self.pending_book_nav = Some(self.pending_book_nav.take().unwrap_or(0) + dir);
            }
            SibLookup::Missing => {}
        }
    }

    /// Fill in an armed boundary prompt's neighbour once the folder listing lands
    /// (the prompt is shown immediately on the boundary hit, before the scan is
    /// back, so its cover + "tap again to open" arrive a moment later).
    pub(crate) fn refresh_prompt_sibling(&mut self) {
        let Some(dir) = self
            .app
            .as_ref()
            .and_then(|a| a.book_prompt.as_ref())
            .map(|p| p.dir)
        else {
            return;
        };
        let SibLookup::Found(path) = self.sibling_book_path(dir) else {
            return;
        };
        self.queue_cover(&path);
        if let Some(app) = self.app.as_mut()
            && let Some(p) = app.book_prompt.as_mut()
        {
            p.title = name_of(&path);
            p.sibling = Some(path);
            app.window.request_redraw();
        }
    }

    /// Queue a comic's cover for background decode, unless it's already decoded.
    /// (Shared by the boundary prompt's two arming paths.)
    pub(crate) fn queue_cover(&mut self, path: &Path) {
        if let Some(app) = self.app.as_mut()
            && !app.thumbs.contains_key(path)
            && app.cover_tx.send(vec![path.to_path_buf()]).is_ok()
        {
            app.covers_inflight += 1;
        }
    }

    /// Open a comic by path: route by type (archive / image folder) to an engine
    /// source, then through `open_comic` (which restores its saved position).
    /// The source construction runs on a background thread — see [`Shell::start_open`].
    pub(crate) fn open_path(&mut self, path: PathBuf) {
        let key = path.to_string_lossy().into_owned();
        self.start_open(Some(key.clone()), move || {
            build_source(&path)
                .map(|src| (key, src))
                .ok_or_else(|| format!("could not open {path:?}"))
        });
    }

    /// Kick the launch resume: pick the most-recently-read book that still exists
    /// and open it — **both** steps off-thread, since deciding which recent is still
    /// on disk is a stat per entry and this runs before the first frame.
    pub(crate) fn resume_recent(&mut self) {
        let recents = self.recents.clone();
        self.start_open(None, move || {
            let key = recents
                .into_iter()
                .find(|k| is_reopenable_fs(k))
                .ok_or_else(|| "no reopenable recent".to_string())?;
            let src =
                build_source(Path::new(&key)).ok_or_else(|| format!("could not open {key}"))?;
            Ok((key, src))
        });
    }

    /// Begin a background open. `target` is the comic's identity when the caller
    /// knows it (it names the spinner and anchors next/prev-book); `job` does all
    /// the I/O and returns that identity plus the source — the key comes back from
    /// the job because the launch resume also *chooses* the book off-thread.
    ///
    /// Each call bumps `open_gen`, so only the newest result is applied: repeated
    /// next-book taps supersede in-flight opens instead of queuing stale swaps.
    pub(crate) fn start_open<F>(&mut self, target: Option<String>, job: F)
    where
        F: FnOnce() -> OpenResult + Send + 'static,
    {
        self.open_gen = self.open_gen.wrapping_add(1);
        let generation = self.open_gen;
        self.opening = Some(target.as_deref().map(book_display_name).unwrap_or_default());
        self.opening_key = target;
        self.resume_fallback = false; // a caller that wants it re-arms it after
        let tx = self.open_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            // Send, then wake: the loop idles while the open runs, so the result
            // only reaches the screen if this schedules the frame that drains it.
            let _ = tx.send((generation, job()));
            if let Some(w) = &waker {
                w();
            }
        });
        if let Some(app) = self.app.as_ref() {
            app.window.request_redraw(); // put the spinner up this frame
        }
    }

    /// Point the reader at `src` (identity `key`): save the outgoing comic's
    /// position, restore this one's, attach the source. Persists to disk.
    pub(crate) fn open_comic(&mut self, key: String, src: Arc<dyn PageSource>) {
        if let (Some(old), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(old, app.reader.index);
        }
        let start = self
            .positions
            .get(&key)
            .copied()
            .unwrap_or(0)
            .min(src.len().saturating_sub(1));
        if src.is_partial() {
            // No on-screen toast system on Android yet, so just log it; the archive
            // still opens with the pages recovered before the truncation.
            log::info!("partial archive recovered: {} pages", src.len());
        }
        // Park it for the next resume *before* handing it to the reader: a suspend
        // tears the whole App down, and re-attaching these same bytes is the
        // difference between a resume that's instant and one that re-reads the
        // archive off storage. Only one book is ever parked — this replaces whatever
        // the last one was, so switching books can't accumulate archives in memory.
        self.parked_source = Some((key.clone(), src.clone()));
        if let Some(app) = self.app.as_mut() {
            attach_source(
                &mut app.reader,
                &app.ctx.device.clone(),
                &app.ctx.queue.clone(),
                src,
                start,
            );
            app.book_title = book_display_name(&key);
            app.book_prompt = None; // a boundary prompt belongs to the old book
            app.window.request_redraw();
        }
        self.current_key = Some(key.clone());
        // Bump to the front of the recently-read list (MRU, deduped, capped).
        self.recents.retain(|k| k != &key);
        self.recents.insert(0, key.clone());
        self.recents.truncate(RECENTS_CAP);
        // The book we just opened demonstrably exists, so the library's recents row
        // can show it without waiting for another existence pass, and its folder is
        // worth warming for the first next/prev-book request. (SAF one-offs are
        // neither reopenable nor inside a browsable folder, so they get neither.)
        if !is_document_key(&key) {
            self.recents_ok.insert(key.clone());
            self.warm_sib_cache(Path::new(&key));
        }
        self.positions.insert(key, start);
        self.dirty_recents = true;
        self.mark_positions();
        // A book switch is a natural durability point (and rare): write now rather
        // than leave the outgoing book's page riding on a debounce.
        self.flush_saves(true);
    }

    /// Open a picked document by its identity key. The platform resolves the key
    /// on this thread (Android: a JNI call for the descriptor) and hands back the
    /// slow part — reading and parsing the whole archive — which runs on the open
    /// worker. A multi-hundred-MB SAF archive used to slurp on the UI thread, which
    /// is an ANR with a progress bar drawn on it.
    pub(crate) fn open_picked(&mut self, uri: &str) {
        let Some(job) = self.platform.open_picked(uri) else { return };
        let key = uri.to_string();
        self.start_open(Some(key.clone()), move || Ok((key, job()?)));
    }
}

/// Point the reader at a new page source: rebuild the decode pool, reset state,
/// kick prefetch. Shared by the test-comic open and the picker.
fn attach_source(
    reader: &mut Reader,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    src: Arc<dyn PageSource>,
    start: usize,
) {
    let pool = DecodePool::new(
        src.clone(),
        device.clone(),
        queue.clone(),
        reader.tex_pool.clone(),
        reader.workers,
    );
    // A new pool starts wakerless: re-install the shell's callback or this book's
    // decodes would land without ever scheduling the frame that draws them.
    pool.set_waker(reader.waker.clone());
    reader.pool = Some(pool);
    reader.reset_volume_state();
    reader.rotation = 0; // each comic opens upright (mirrors the desktop shell)
    reader.source = Some(src);
    reader.open_at(start);
    // Fresh pool, fresh caches: drop the cross-frame prefetch memo *and* the
    // thumbnail-tail pivot, or the new pool's empty tail would stay empty until the
    // reader happened to travel a full stride.
    reader.invalidate_jobs();
    reader.prefetch();
}

fn ext_lower(p: &Path) -> Option<String> {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Is this a comic archive yosh can open (RAR/CBR only with the `rar` feature)?
pub(crate) fn is_comic_archive(p: &Path) -> bool {
    match ext_lower(p).as_deref() {
        Some("cbz" | "zip" | "cb7" | "7z") => true,
        #[cfg(feature = "rar")]
        Some("cbr" | "rar") => true,
        _ => false,
    }
}

/// Build an engine page source from a comic path (archive or image folder).
pub(crate) fn build_source(path: &Path) -> Option<Arc<dyn PageSource>> {
    if path.is_dir() {
        FolderSource::new(path)
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn PageSource>)
    } else {
        match ext_lower(path).as_deref() {
            Some("cbz" | "zip") => ZipSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            Some("cb7" | "7z") => SevenzSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            #[cfg(feature = "rar")]
            Some("cbr" | "rar") => RarSource::new(path)
                .ok()
                .map(|s| Arc::new(s) as Arc<dyn PageSource>),
            _ => None,
        }
    }
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

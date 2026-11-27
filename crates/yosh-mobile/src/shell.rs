//! The winit `ApplicationHandler`: window + GPU lifecycle, the frame loop, background-result draining.

use super::*;

/// The mobile shell: winit's `ApplicationHandler` for a platform crate's event
/// loop. Build it with [`Shell::new`] over a [`Platform`] and hand it to
/// `EventLoop::run_app`; it owns the window, GPU, reader and every piece of
/// persisted state, and rebuilds the window + GPU on each `resumed`.
pub struct Shell {
    pub(crate) app: Option<App>,
    /// Everything the shell asks of the OS (storage, picker, window flags, device
    /// probes). On Android: JNI into the `YoshActivity` Java bridge.
    pub(crate) platform: Rc<dyn Platform>,
    /// "Something new to draw" callback for off-thread producers: it just calls
    /// `Window::request_redraw`, which is thread-safe (it flags the window and
    /// wakes the ALooper). Handed to the decode pool via `Reader::set_waker` and to
    /// the library producers, so the frame loop can genuinely idle between results
    /// instead of polling for them. Re-made per window in `resumed`.
    pub(crate) frame_waker: Option<Waker>,
    /// A document-picker launch is awaiting its result.
    picker_pending: bool,

    // Async open: building a page source is pure I/O — a `read_dir`, a zip central
    // directory, a whole-archive slurp off a SAF descriptor — and on phone storage
    // that is exactly the multi-second stall that turns into an ANR. It runs on a
    // worker thread; only the newest result is applied, so mashing "next book"
    // supersedes in-flight opens instead of queuing stale swaps.
    pub(crate) open_gen: u64,
    /// Display name of the volume currently being opened (`None` = nothing in
    /// flight). Drives the spinner card, so it must be cleared on *every* landing.
    pub(crate) opening: Option<String>,
    /// Identity of that volume, when it's known up front (the launch resume picks
    /// its book off-thread, so it isn't). Next/prev-book resolves against this
    /// rather than the still-open book, so a second tap during an open advances
    /// from the pending target instead of repeating it.
    pub(crate) opening_key: Option<String>,
    pub(crate) open_tx: Sender<(u64, OpenResult)>,
    open_rx: Receiver<(u64, OpenResult)>,
    /// The in-flight open is the launch resume (which also picks *which* recent to
    /// open, off-thread): a failure falls back to the library / empty state rather
    /// than leaving the spinner's aftermath on a blank screen.
    pub(crate) resume_fallback: bool,

    /// Cached parent-dir comic listing for next/prev-book (`(parent, comics)`),
    /// warmed in the background on open so the first boundary hit doesn't pay a
    /// `read_dir` + per-entry stat on the UI thread.
    pub(crate) sib_cache: Option<(PathBuf, Vec<PathBuf>)>,
    pub(crate) sib_tx: Sender<(PathBuf, Vec<PathBuf>)>,
    sib_rx: Receiver<(PathBuf, Vec<PathBuf>)>,
    /// A next/prev-book request that arrived while `sib_cache` was cold, held until
    /// the scan lands (then replayed). Requests accumulate, so mashing "next book"
    /// three times before the scan finishes jumps three volumes rather than one.
    pub(crate) pending_book_nav: Option<i64>,

    /// Which `recents` entries still exist on disk. Computed off-thread (a stat per
    /// entry against SD-card storage is not per-frame work) and refreshed when the
    /// library opens; the library render only reads it.
    pub(crate) recents_ok: HashSet<String>,
    pub(crate) recents_ok_tx: Sender<HashSet<String>>,
    recents_ok_rx: Receiver<HashSet<String>>,

    // Debounced atomic saves. Every one of these is a whole-file rewrite, and page
    // turns used to fire two of them synchronously per flip. Now a change only
    // *marks* its file dirty; `flush_saves` writes what's dirty once the marks have
    // settled (or immediately when forced: suspend, book switch), via a temp file +
    // rename so a kill mid-write can't truncate the real one.
    pub(crate) dirty_positions: bool,
    pub(crate) dirty_progress: bool,
    pub(crate) dirty_recents: bool,
    pub(crate) dirty_collapsed: bool,
    pub(crate) dirty_view: bool,
    pub(crate) dirty_libroot: bool,
    /// When the oldest un-flushed mark was made (`None` = nothing dirty).
    pub(crate) dirty_since: Option<Instant>,

    /// Per-comic last-read page, keyed by comic identity (content:// URI or path).
    pub(crate) positions: HashMap<String, usize>,
    /// Where `positions` persists (app's private dir).
    pub(crate) pos_path: Option<PathBuf>,
    /// Per-comic reading progress: (furthest 1-based page reached, total pages).
    /// Distinct from `positions` (the *current* page — re-reading from the start
    /// must not unmark a finished volume). Drives the library's read/unread
    /// fades and the series status labels.
    pub(crate) progress: HashMap<String, (u32, u32)>,
    pub(crate) progress_path: Option<PathBuf>,
    /// Library series the user collapsed (storing the collapsed set makes
    /// expanded the default for anything new).
    pub(crate) collapsed: std::collections::HashSet<PathBuf>,
    pub(crate) collapsed_path: Option<PathBuf>,
    /// Most-recently-read volume keys (newest first): resume-on-launch picks the
    /// first reopenable one; the library's "Recently read" row renders them.
    pub(crate) recents: Vec<String>,
    pub(crate) recents_path: Option<PathBuf>,
    /// Identity of the currently-open comic, for saving its position.
    pub(crate) current_key: Option<String>,
    /// Cached all-files-access state (refreshed on resume + library toggle).
    pub(crate) has_files: bool,
    /// The library dir to open the browser at (persisted across launches).
    init_lib_dir: PathBuf,
    pub(crate) lib_dir_file: Option<PathBuf>,
    /// Persisted viewing options (direction, layout, fit, page-turn animation) +
    /// where they live.
    init_view: InitView,
    pub(crate) view_file: Option<PathBuf>,
    /// The shared touch state machine (finger bookkeeping, lock thresholds, pinch,
    /// release velocities, glide starts) — the same one the desktop shell drives,
    /// so a physics fix lands on both platforms at once. It also owns the glide
    /// tick clock (`tick`).
    pub(crate) gestures: TouchGestures,
    /// Last immersive state pushed to Android, to avoid redundant JNI each frame.
    pub(crate) applied_immersive: Option<bool>,
    /// Last KEEP_SCREEN_ON state pushed to the window, same idea. `None` ⇒ not
    /// applied yet, which is also the post-resume state: the flag lives on the
    /// native window, and a resume builds a new one.
    applied_keep_on: Option<bool>,

    /// The open book's `(key, source)`, kept alive across a suspend so the resume
    /// can re-attach it instead of re-reading the archive off storage. Only the
    /// current book is parked (each `open_comic` replaces it), and it costs nothing
    /// while that book is open — the reader holds the very same `Arc`.
    ///
    /// This is what makes a resume cheap: a full-teardown resume otherwise pays the
    /// whole open again (a `read_dir`, a zip central directory, or — for a 7z/RAR —
    /// a full re-decompress). It also gets SAF `content://` books through a suspend,
    /// which nothing else can: their bytes came off a one-shot file descriptor that
    /// can't be reopened, so before this a backgrounded picked book came back as a
    /// failed open. Dropped by [`Shell::memory_warning`], since it can be several
    /// hundred MB of archive.
    pub(crate) parked_source: Option<(String, Arc<dyn PageSource>)>,
    /// Status-bar height in px (queried once, lazily); used to inset chrome under
    /// the bars. `None` until first queried.
    pub(crate) status_bar_px: Option<i32>,
}

impl Shell {
    /// Load the persisted state from the platform's data dir and build the shell.
    /// No window or GPU yet: those come with the first `resumed`.
    pub fn new(platform: Rc<dyn Platform>) -> Shell {
        // KEEP_SCREEN_ON is *not* set here: it's applied per frame from the reading
        // state (see the `applied_keep_on` block in `RedrawRequested`), because holding
        // the screen awake while the user is browsing the library — or staring at the
        // empty state — is just battery drain with no page to read.
        // Per-comic reading positions persist in the app's private dir.
        let pos_path = platform.data_dir().map(|p| p.join("positions.tsv"));
        let positions = pos_path.as_deref().map(load_positions).unwrap_or_default();
        // Reading progress (furthest page seen + total) → the library's read states.
        let progress_path = platform.data_dir().map(|p| p.join("progress.tsv"));
        let progress = progress_path.as_deref().map(load_progress).unwrap_or_default();
        // Series the user collapsed in the library (default expanded).
        let collapsed_path = platform.data_dir().map(|p| p.join("collapsed.txt"));
        let collapsed = collapsed_path.as_deref().map(load_collapsed).unwrap_or_default();
        // Most-recently-read volume keys (MRU, newest first): resume target + shelf.
        let recents_path = platform.data_dir().map(|p| p.join("recents.txt"));
        let recents = recents_path.as_deref().map(load_recents).unwrap_or_default();
        let lib_dir_file = platform.data_dir().map(|p| p.join("libroot.txt"));
        let init_lib_dir = lib_dir_file
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| PathBuf::from(s.trim()))
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| platform.default_library_root());
        // Persisted viewing options (direction / layout / fit); default RTL manga order.
        let view_file = platform.data_dir().map(|p| p.join("view.txt"));
        let init_view = view_file
            .as_deref()
            .map(load_view)
            .unwrap_or((
                Direction::Rtl,
                LayoutMode::Single,
                FitMode::Window,
                true,
                false,
                true,
                ThemePref::System,
                PerfPref::Auto,
                false,
                DEFAULT_SPINE_STRENGTH,
                false,
                VolKeys::Off,
                None,
            ));
        let (open_tx, open_rx) = std::sync::mpsc::channel();
        let (sib_tx, sib_rx) = std::sync::mpsc::channel();
        let (recents_ok_tx, recents_ok_rx) = std::sync::mpsc::channel();
        Shell {
            app: None,
            platform,
            frame_waker: None,
            picker_pending: false,
            open_gen: 0,
            opening: None,
            opening_key: None,
            open_tx,
            open_rx,
            resume_fallback: false,
            sib_cache: None,
            sib_tx,
            sib_rx,
            pending_book_nav: None,
            recents_ok: HashSet::new(),
            recents_ok_tx,
            recents_ok_rx,
            dirty_positions: false,
            dirty_progress: false,
            dirty_recents: false,
            dirty_collapsed: false,
            dirty_view: false,
            dirty_libroot: false,
            dirty_since: None,
            positions,
            pos_path,
            progress,
            progress_path,
            collapsed,
            collapsed_path,
            recents,
            recents_path,
            current_key: None,
            has_files: false,
            init_lib_dir,
            lib_dir_file,
            init_view,
            view_file,
            gestures: TouchGestures::new(),
            applied_immersive: None,
            applied_keep_on: None,
            parked_source: None,
            status_bar_px: None,
        }
    }
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("yosh"))
                .expect("create window"),
        );
        self.has_files = self.platform.has_library_access();
        // Anything still dirty belongs to the App that's about to be torn down (the
        // view options are read back off it), so write it out before `init_view` is
        // re-read below — otherwise a pending options change would be reverted by
        // its own file.
        self.flush_saves(true);
        // Every options toggle writes view.txt immediately, but `init_view` is the
        // snapshot this process *launched* with — and a resume rebuilds the whole
        // App from it. Re-read the file so options changed since launch (theme,
        // fit, performance profile, …) survive a background/foreground cycle
        // instead of silently reverting.
        if let Some(f) = &self.view_file {
            self.init_view = load_view(f);
        }
        // On-demand rendering: the loop sleeps until an event arrives, and every
        // off-thread producer wakes it through `frame_waker` when it has something
        // to show. `Wait` is winit's default, but the whole redraw guard below is
        // built on it, so pin it explicitly rather than inherit it.
        event_loop.set_control_flow(ControlFlow::Wait);
        // A fresh window means a fresh waker (the old one holds the old window and
        // dies with the App it belonged to).
        self.frame_waker = Some({
            let w = window.clone();
            Arc::new(move || w.request_redraw())
        });
        // KEEP_SCREEN_ON is a flag on the *native window*, which this resume
        // replaces — so forget what we last pushed and let the first frame re-derive
        // and re-apply it.
        self.applied_keep_on = None;
        log::info!(
            "all-files access: {} | scale {}",
            self.has_files,
            window.scale_factor()
        );
        // What to restore after a resume-rebuild (all None/false on a true first launch).
        let mut rebuild_book: Option<String> = None;
        let mut resume_active = false; // this resumed() is a forced rebuild (app was live)
        let mut resume_library = false; // the library overlay was up when we rebuilt
        // Resume from background: unconditionally rebuild the GPU. Reusing the cached
        // adapter against the post-background surface works on most devices, but some
        // e-ink GPUs reject it ("Surface does not support the adapter's queue family") via
        // an async validation error that no synchronous check (is_surface_supported, an
        // error scope / uncaptured-error handler, get_current_texture's status, or a poll)
        // catches before it lands — so tear the context down and rebuild from scratch (a
        // fresh adapter is always compatible with its fresh surface, like cold start).
        // Resume isn't the zero-hitch reading hot-path, so re-decoding the current page
        // from its saved position is an acceptable cost for never crashing.
        if self.app.is_some() {
            resume_active = true;
            resume_library = self.app.as_ref().unwrap().library_view;
            // Save the open book's page and clear current_key so the reopen below restores
            // that saved page (not the fresh reader's index 0).
            if let (Some(k), Some(app)) = (self.current_key.take(), self.app.as_ref()) {
                self.positions.insert(k.clone(), app.reader.index);
                self.mark_positions();
                self.flush_saves(true); // the App holding the view state is about to go
                rebuild_book = Some(k);
            }
            self.app = None; // drops the old GPU context, Reader, decode pool + textures
            // fall through to the full build below.
        }

        // First launch (or a forced rebuild): full build.
        let instance = GpuContext::create_instance();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        // A phone has exactly one GPU, so the preference picks no different
        // adapter — but `LowPower` is the honest hint to the driver's power
        // governor for a workload that draws a couple of quads per frame.
        let ctx = GpuContext::create(instance, Some(&surface), wgpu::PowerPreference::LowPower);
        // Make wgpu errors non-fatal: the default handler panics, which is how a stray
        // validation error (notably surface.configure rejecting a post-background surface
        // on some e-ink GPUs) takes the app down. Log instead; the resume path detects the
        // bad surface via get_current_texture's status and rebuilds the GPU.
        ctx.device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
            log::error!("wgpu error (non-fatal): {e}");
        }));
        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: GpuContext::choose_surface_format(&caps),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto, // wgpu's historical (sRGB) choice
        };
        surface.configure(&ctx.device, &config);

        let page_pipeline = PagePipeline::new(&ctx.device, config.format);
        let egui_ctx = egui::Context::default();
        // One font set: Phosphor icon glyphs (seekbar buttons) plus a CJK fallback.
        // Phosphor is installed unconditionally; egui's bundled fonts have no CJK
        // glyphs, so add the system Noto Sans CJK as a fallback when present so
        // Japanese comic / file names render instead of tofu squares.
        // The UI language is resolved first: it picks the CJK face (Japanese vs.
        // Korean vs. Simplified/Traditional Chinese glyph forms).
        let system_lang = Lang::from_tag(&self.platform.locale_tag());
        i18n::set_lang(self.init_view.12.unwrap_or(system_lang));
        install_fonts(&egui_ctx, self.platform.as_ref(), i18n::lang().cjk_face());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            Some(8192),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &ctx.device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let (mem_budget_mb, total_mb) = device_mem(self.platform.as_ref());
        let max_khz = self.platform.max_cpu_khz();
        let auto_tier = device_tier(total_mb, max_khz);
        let perf = self.init_view.7;
        let tier = perf.tier().unwrap_or(auto_tier);
        let budget = Budget::for_tier(tier, mem_budget_mb, cpus);
        // One line with everything the tier decision rests on: a field report of a
        // device behaving badly should be diagnosable from logcat alone.
        log::info!(
            "device: {total_mb} MB RAM, {cpus} cpus, max {} MHz, gpu \"{}\" → auto tier {auto_tier:?}, perf {} → {tier:?}",
            max_khz.map_or(0, |k| k / 1000),
            ctx.adapter_info.name,
            perf.label(),
        );
        log::info!("budget: {budget:?} (mem slice {mem_budget_mb} MB)");
        let tex_pool = Arc::new(TexturePool::with_max_total(budget.texpool_max));
        let mut reader = Reader::new(
            ctx.device.clone(),
            ctx.queue.clone(),
            tex_pool,
            budget,
            self.init_view.2, // fit
            // Concrete layout, resolved from the saved mode against the initial
            // window size (so an `Auto` launch already matches the orientation).
            self.init_view.1.resolve(config.width, config.height),
            self.init_view.4, // scroll mode (persisted)
            self.init_view.0, // direction (default RTL)
            0,
            true, // two_tier: LQ while seeking → HQ on settle
        );
        reader.transition_enabled = self.init_view.3; // page-turn animation (persisted; default on)
        // No-upscale fit (persisted; default off = stretch small pages, as before).
        reader.fit_no_upscale = self.init_view.10;
        // Spine shadow: the shell owns on/off × strength, the reader takes one number.
        reader.spine_strength = if self.init_view.8 { self.init_view.9 } else { 0.0 };
        // Landing decodes schedule their own frame from the worker thread; every
        // pool this reader builds inherits the callback.
        reader.set_waker(self.frame_waker.clone());
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (series_tx, series_rx) = std::sync::mpsc::channel();
        let (info_tx, info_rx) = std::sync::mpsc::channel();
        let (browse_tx, browse_rx) = std::sync::mpsc::channel();
        // On-disk cover-thumbnail cache dir (app-private `…/thumbs`): covers load
        // from here instead of re-decoding the full first page on every open. It's
        // owned by the cover worker, the only thing that touches it.
        let thumb_cache_dir = self.platform.data_dir().map(|p| p.join("thumbs"));
        let cover_tx = spawn_cover_worker(thumb_cache_dir, thumb_tx, self.frame_waker.clone());
        self.app = Some(App {
            window: window.clone(),
            ctx,
            surface,
            config,
            page_pipeline,
            reader,
            anim_origin: Instant::now(),
            egui_ctx,
            egui_state,
            egui_renderer,
            library_view: false,
            lib_root: self.init_lib_dir.clone(),
            lib_dir: self.init_lib_dir.clone(),
            lib_entries: Vec::new(),
            thumbs: HashMap::new(),
            thumb_rx,
            cover_tx,
            queued_covers: std::collections::HashSet::new(),
            covers_inflight: 0,
            frame_waker: self.frame_waker.clone(),
            thumb_used: HashMap::new(),
            frame_no: 0,
            logo: None,
            series: Vec::new(),
            series_rx,
            series_tx,
            series_pending: false,
            lib_browse: false,
            controls: true,
            controls_shown_at: Instant::now(),
            book_title: String::new(),
            book_prompt: None,
            show_options: false,
            show_info: false,
            info_for: None,
            info_meta: Vec::new(),
            info_tx,
            info_rx,
            browse_tx,
            browse_rx,
            browse_pending: false,
            view_dirty: false,
            libroot_dirty: false,
            layout_mode: self.init_view.1,
            resume_on_startup: self.init_view.5,
            vol_keys: self.init_view.11,
            lang_pref: self.init_view.12,
            system_lang,
            spine_shadow_on: self.init_view.8,
            spine_shadow_strength: self.init_view.9,
            theme: self.init_view.6,
            perf,
            auto_tier,
            mem_budget_mb,
            cpus,
            system_dark: self.platform.system_dark(),
            platform: self.platform.clone(),
            applied_light: None,
        });
        // Pick up where we left off: with resume on, reopen the most-recent
        // reopenable book (at its saved page); else, if a library is configured, open
        // it as the home; else fall through to the empty-state card (onboarding). SAF
        // content:// one-offs can't be silently rebuilt, so resume skips them.
        let lib_configured = self.init_lib_dir != self.platform.default_library_root();
        // Reopen after the (re)build. Priority: a SAF picker return opens the picked file;
        // a resume-rebuild restores what was showing (the open book at its page, else the
        // library if it was up / is the home, else the empty-state card); a true first
        // launch resumes the most-recent book, else the configured library, else the card.
        if self.picker_pending
            && let Some(uri) = self.platform.take_picked()
        {
            self.picker_pending = false;
            self.open_picked(&uri);
        } else if let Some(key) = rebuild_book {
            // The book that was open is almost always still *parked* — the source
            // outlives the App teardown above — so re-attach it directly: zero I/O,
            // no spinner, the page is back on screen on the first frame. This is the
            // whole point of `parked_source`, and it's the only way a SAF
            // `content://` book can survive a suspend at all (its descriptor is
            // long gone; only these bytes remain).
            match self.parked_source.take() {
                Some((k, src)) if k == key => self.open_comic(key, src),
                parked => {
                    // Something else was parked (or nothing was) — put it back and
                    // rebuild from storage. Reopening is asynchronous, so a resume
                    // still never blocks on re-reading an archive; it just draws the
                    // spinner for the first frames. A book that vanished (or a
                    // content:// one-off, which can't be rebuilt) fails the open and
                    // lands on the library, exactly as the old `is_reopenable_fs`
                    // filter did — see `resume_fallback`.
                    self.parked_source = parked;
                    self.open_path(PathBuf::from(key));
                    self.resume_fallback = resume_library || lib_configured; // after: start_open clears it
                }
            }
        } else if resume_active {
            if resume_library || lib_configured {
                self.open_library();
            }
        } else if self.init_view.5 && !self.recents.is_empty() {
            self.resume_recent();
            self.resume_fallback = lib_configured; // after: start_open clears it
        } else if lib_configured {
            self.open_library();
        }
        window.request_redraw();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Persist the current position + library dir before the OS may kill us.
        // Everything is written **synchronously** here: the process can be killed
        // the moment this returns, so the debounce is force-flushed rather than
        // trusted to a later frame that may never come.
        if let (Some(k), Some(app)) = (self.current_key.clone(), self.app.as_ref()) {
            self.positions.insert(k, app.reader.index);
        }
        self.dirty_positions = true;
        self.dirty_progress = true;
        self.dirty_recents = true;
        self.dirty_libroot = true;
        self.dirty_since.get_or_insert_with(Instant::now);
        self.flush_saves(true);
        // Drop any half-finished gesture: the fingers that made it are long gone by
        // the time we come back, and `resumed` rebuilds the whole `App` anyway. (Its
        // 150 ms bounce windows would expire on their own; this just keeps a stale
        // lock from surviving a background.)
        self.gestures = TouchGestures::new();
        // Stop decoding. Android may leave a backgrounded app alive for *hours*
        // before it needs the memory, and an unparked pool would spend all of it
        // grinding the whole-volume thumbnail tail at 8 threads — burning battery
        // into textures no one can see. `park` also makes the work already in
        // flight abandon at its next yield point, so this takes effect in
        // milliseconds rather than after the current decode.
        //
        // (Today's `resumed` drops the whole App anyway, so nothing ever calls
        // `unpark` — the park matters for the window *between* the two, which is
        // the part that can last hours.)
        if let Some(app) = self.app.as_mut() {
            if let Some(pool) = &app.reader.pool {
                pool.park();
            }
            // The LQ preview tier is the single biggest thing we hold (up to
            // `lq_cap` GPU textures for the whole volume) and the most rebuildable:
            // shed it so the OS has less reason to kill us while we're away.
            app.reader.lq_cache.clear();
            // Both of the above invalidate the prefetch memo — the next foreground
            // frame must rebuild the job list and the thumbnail tail from scratch.
            app.reader.invalidate_jobs();
        }
    }

    /// The OS is under memory pressure and we are a candidate for death (Android
    /// `onLowMemory`). Everything shed here is *derived* state that costs time, not
    /// correctness, to rebuild — no reading state, no unsaved position.
    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        // The parked source is the biggest single thing we hold (a whole in-memory
        // archive for a SAF/7z/RAR book) and the most optional — losing it costs the
        // *next* resume a re-open, nothing else.
        let dropped_park = self.parked_source.take().is_some();
        let mut shed_lq = 0;
        if let Some(app) = self.app.as_mut() {
            // The LQ preview tier: up to `lq_cap` whole-volume thumbnails, all of
            // which the tail refills on demand. The live `cache` deliberately stays
            // — evicting it would blank the page the user is looking at.
            shed_lq = app.reader.lq_cache.len();
            app.reader.lq_cache.clear();
            // After the cache clear, not before: `PageCache::clear` recycles its
            // textures *into* the pool, so clearing the pool first would just refill
            // it with what we are trying to give back.
            app.reader.tex_pool.clear();
            // The thumbnails just went away; the next frame's prefetch has to
            // reconsider them from scratch rather than trust its memo.
            app.reader.invalidate_jobs();
            app.window.request_redraw();
        }
        log::warn!(
            "memory warning: shed {shed_lq} LQ previews + the texture pool{}",
            if dropped_park { " + the parked source" } else { "" },
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui see the event first (seekbar drag); gate nav on what it consumes,
        // and wake the loop if egui wants to repaint (e.g. a slider being dragged).
        // RedrawRequested itself is exempt: egui-winit reports `repaint: true` for it
        // ("paint now", not "schedule another frame"), so honoring it would re-arm a
        // redraw from within every frame — a feedback loop that silently defeats the
        // on-demand guard at the end of `render` and burns battery rendering a static
        // page at display refresh rate. (Same bug + fix as the desktop shell.)
        let egui_consumed = if let Some(app) = self.app.as_mut() {
            let resp = app.egui_state.on_window_event(&app.window, &event);
            if resp.repaint && !matches!(event, WindowEvent::RedrawRequested) {
                app.window.request_redraw();
            }
            resp.consumed
        } else {
            false
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(app) = self.app.as_mut() {
                    app.config.width = size.width.max(1);
                    app.config.height = size.height.max(1);
                    app.surface.configure(&app.ctx.device, &app.config);
                    app.window.request_redraw();
                }
            }
            WindowEvent::Touch(Touch {
                phase,
                location,
                id,
                ..
            }) => self.handle_touch(phase, id, location.x, location.y, egui_consumed),
            WindowEvent::KeyboardInput { event, .. } => self.handle_key(&event, egui_consumed),
            WindowEvent::RedrawRequested => {
                // A picker result may have landed while we were backgrounded.
                if self.picker_pending
                    && let Some(uri) = self.platform.take_picked()
                {
                    self.picker_pending = false;
                    self.open_picked(&uri);
                }
                // The single drain point for everything the shell's background
                // threads produce (opens, sibling listings, recents existence).
                self.poll_background();
                // Advance an inertial scroll glide (if any) before drawing this frame.
                self.tick_scroll_fling();
                let has_files = self.has_files;
                // Status-bar height (queried once; constant). Chrome is inset by it
                // while the bars are shown so they don't cover the top buttons.
                let status_px = *self
                    .status_bar_px
                    .get_or_insert_with(|| self.platform.top_inset_px());
                let opening = self.opening.clone();
                let open_gen = self.open_gen;
                let reqs = match &mut self.app {
                    Some(app) => app.render(
                        has_files,
                        status_px,
                        opening.as_deref(),
                        open_gen,
                        LibCtx {
                            progress: &self.progress,
                            positions: &self.positions,
                            collapsed: &self.collapsed,
                            current_key: self.current_key.as_deref(),
                            recents: &self.recents,
                            recents_ok: &self.recents_ok,
                        },
                    ),
                    None => FrameReqs::default(),
                };
                // Keep the read-tracking map current (cheap; persisted with the
                // position saves) — covers seekbar jumps applied inside render.
                self.note_progress();
                // The egui frame may have changed the viewing options / library root;
                // both live on the App, so it flags them and the debounce is the
                // Shell's business.
                let (view_dirty, libroot_dirty) = match self.app.as_mut() {
                    Some(app) => (
                        std::mem::take(&mut app.view_dirty),
                        std::mem::take(&mut app.libroot_dirty),
                    ),
                    None => (false, false),
                };
                if view_dirty {
                    self.dirty_view = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                }
                if libroot_dirty {
                    self.dirty_libroot = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                }
                if let Some(dir) = reqs.toggle_series {
                    if !self.collapsed.remove(&dir) {
                        self.collapsed.insert(dir);
                    }
                    self.dirty_collapsed = true;
                    self.dirty_since.get_or_insert_with(Instant::now);
                    if let Some(app) = self.app.as_ref() {
                        app.window.request_redraw();
                    }
                }
                if reqs.grant {
                    self.platform.request_library_access();
                }
                if reqs.open_picker {
                    self.platform.launch_picker();
                    self.picker_pending = true;
                }
                if reqs.open_library {
                    self.open_library();
                }
                if let Some(path) = reqs.open {
                    self.open_path(path);
                }
                if reqs.page_nav != 0 {
                    self.flip(reqs.page_nav);
                }
                if reqs.book_nav != 0 {
                    self.open_sibling_book(reqs.book_nav);
                }
                if reqs.to_library {
                    self.open_library();
                }
                // Reading view with controls hidden → immersive (bars hidden);
                // otherwise show the bars (controls up, library, or empty screen).
                let immersive = self.current_key.is_some()
                    && self
                        .app
                        .as_ref()
                        .is_some_and(|a| !a.controls && !a.library_view);
                if self.applied_immersive != Some(immersive) {
                    self.platform.set_immersive(immersive);
                    self.applied_immersive = Some(immersive);
                }
                // Hold the screen awake only while there is actually a page to read
                // — a reader that never touches the screen for minutes at a time
                // does need this, but the library grid and the empty state are just
                // ordinary UI and should dim on the system timeout like anything
                // else. Same cached-flag pattern as `applied_immersive`: the window
                // flag is a JNI round-trip, so only push it when it changes.
                let keep_on = self.current_key.is_some()
                    && self.app.as_ref().is_some_and(|a| !a.library_view);
                if self.applied_keep_on != Some(keep_on) {
                    self.platform.set_keep_screen_on(keep_on);
                    self.applied_keep_on = Some(keep_on);
                }
                // Opportunistic: writes what has been dirty long enough. Frames stop
                // when the reader idles, so this is a best-effort early write — the
                // guaranteed ones are the forced flushes on suspend / book switch.
                self.flush_saves(false);
            }
            _ => {}
        }
    }
}

impl Shell {
    /// Drain everything the shell's background threads deliver. Called once per
    /// frame from `RedrawRequested`, before the render — every producer wakes the
    /// loop after sending, so a result never waits for an unrelated frame.
    pub(crate) fn poll_background(&mut self) {
        // Finished opens, newest generation only: a later open bumped `open_gen`,
        // so a superseded result is dropped rather than swapping in a stale book.
        loop {
            match self.open_rx.try_recv() {
                Ok((generation, res)) => {
                    if generation != self.open_gen {
                        continue; // superseded by a newer open
                    }
                    self.opening = None;
                    self.opening_key = None;
                    let fallback = std::mem::take(&mut self.resume_fallback);
                    match res {
                        Ok((key, src)) if !src.is_empty() => {
                            log::info!("open {key}: {} pages", src.len());
                            self.open_comic(key, src);
                        }
                        Ok((key, _)) => {
                            log::warn!("no images found in {key}");
                            self.after_failed_open(fallback);
                        }
                        Err(e) => {
                            log::error!("open failed: {e}");
                            self.after_failed_open(fallback);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                // The Shell owns the Sender, so this can't happen — but fold it into
                // "nothing is opening" anyway so the spinner can never stick.
                Err(TryRecvError::Disconnected) => {
                    self.opening = None;
                    self.opening_key = None;
                    break;
                }
            }
        }
        // Sibling listings only warm the next/prev-book cache — nothing to show…
        let mut sibs_landed = false;
        while let Ok(entry) = self.sib_rx.try_recv() {
            self.sib_cache = Some(entry);
            sibs_landed = true;
        }
        if sibs_landed {
            // …unless a book jump was parked waiting for exactly this listing: replay
            // it now. If the book moved folders meanwhile this re-warms and re-parks
            // instead — still terminating, since each pass needs a fresh scan.
            if let Some(d) = self.pending_book_nav.take() {
                self.open_sibling_book(d);
            }
            // An armed boundary prompt resolved to "no neighbour" only because the
            // cache was cold: fill it in now that the listing is here.
            if self
                .app
                .as_ref()
                .and_then(|a| a.book_prompt.as_ref())
                .is_some_and(|p| p.sibling.is_none())
            {
                self.refresh_prompt_sibling();
            }
        }
        while let Ok(ok) = self.recents_ok_rx.try_recv() {
            self.recents_ok = ok;
        }
    }

    /// An open that produced nothing: with `fallback` (the launch/resume reopen)
    /// land on the library rather than leaving the user staring at the last frame
    /// of the spinner.
    fn after_failed_open(&mut self, fallback: bool) {
        if fallback {
            self.open_library();
        } else if let Some(app) = self.app.as_ref() {
            app.window.request_redraw(); // drop the spinner
        }
    }

    /// Open the library browser at the configured root, refreshing all-files
    /// access first (the browser shows the grant prompt if it's missing). Shared by
    /// Advance the inertial glides one frame (a no-op unless one is running): apply
    /// the velocity to the strip and/or the panned page, decay it, and schedule the
    /// next frame while it lasts. This is what animates the pan throw.
    fn tick_scroll_fling(&mut self) {
        if let Some(app) = self.app.as_mut()
            && self.gestures.tick(&mut app.reader, Instant::now())
        {
            app.window.request_redraw();
        }
    }

    /// the top-strip tap and the empty-state "Browse library" button.
    pub(crate) fn open_library(&mut self) {
        self.has_files = self.platform.has_library_access();
        // Which recents still exist is decided once here, off-thread, instead of
        // stat-ing all 32 of them on every library frame.
        self.refresh_recents_ok();
        if let Some(app) = self.app.as_mut() {
            app.library_view = true;
            app.lib_browse = false;
            if app.series.is_empty() && !app.series_pending {
                app.kick_series_scan();
            }
            app.window.request_redraw();
        }
    }
}

//! The per-window `App`: GPU, egui, the reader, and the frame's render.

use super::*;

/// One off-thread info-overlay result: the open generation and the page index it
/// was gathered for, plus the `(label, value)` rows. The two tags let the main
/// thread drop a result it has already moved past — a book switch bumps the
/// generation, a page turn changes the index.
type InfoRows = (u64, usize, Vec<(String, String)>);

/// See [`App::book_prompt`].
pub(crate) struct BookPrompt {
    /// +1 = next book (armed at the last page), -1 = previous (first page).
    pub(crate) dir: i64,
    /// When it was armed (the confirm window + the card's visibility run from here).
    pub(crate) at: Instant,
    /// The resolved sibling comic, if any (None at the ends / for SAF sources).
    pub(crate) sibling: Option<PathBuf>,
    /// The sibling's display name (empty when `sibling` is None).
    pub(crate) title: String,
}

/// The live app: surface + the engine device context, the page pipeline, and the
/// reader that owns the decode pool / cache / nav state.
pub(crate) struct App {
    pub(crate) window: Arc<Window>,
    pub(crate) ctx: GpuContext,
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) page_pipeline: PagePipeline,
    pub(crate) reader: Reader,
    pub(crate) anim_origin: Instant,
    pub(crate) egui_ctx: egui::Context,
    pub(crate) egui_state: egui_winit::State,
    pub(crate) egui_renderer: egui_wgpu::Renderer,
    /// Library browser overlay state.
    pub(crate) library_view: bool,
    /// The configured library root (persisted); the browser can't go above it.
    pub(crate) lib_root: PathBuf,
    pub(crate) lib_dir: PathBuf,
    pub(crate) lib_entries: Vec<Entry>,
    /// Decoded cover thumbnails by comic path (egui textures), filled off-thread.
    pub(crate) thumbs: HashMap<PathBuf, egui::TextureHandle>,
    /// One message per *requested* cover — `None` when the decode failed, so the
    /// in-flight count below can't leak on an unreadable archive.
    pub(crate) thumb_rx: std::sync::mpsc::Receiver<(PathBuf, Option<egui::ColorImage>)>,
    /// Queue into the persistent cover-decode worker (see [`spawn_cover_worker`]):
    /// batches of comic paths in, decoded covers back out through `thumb_rx`.
    pub(crate) cover_tx: std::sync::mpsc::Sender<Vec<PathBuf>>,
    /// Covers already queued for decode (lazy, per visible series section).
    pub(crate) queued_covers: std::collections::HashSet<PathBuf>,
    /// Covers sent to the worker whose result hasn't been drained yet. Unlike
    /// `queued_covers` (which remembers every request, so a decoded — or
    /// undecodable — cover is never re-queued) this drops back to zero, which is
    /// what the redraw guard needs: "results are still coming", not "results ever
    /// came".
    pub(crate) covers_inflight: usize,
    /// The shell's frame waker (see [`Shell::frame_waker`]), carried here so the
    /// App's own off-thread producers — the series scan — can wake the loop when
    /// their results land.
    pub(crate) frame_waker: Option<Waker>,
    /// Frame stamp a cover was last drawn (LRU for the `thumbs` cap — a big
    /// library would otherwise pin hundreds of MB of egui textures).
    pub(crate) thumb_used: HashMap<PathBuf, u64>,
    /// Monotonic render counter for `thumb_used`.
    pub(crate) frame_no: u64,
    /// The yosh mascot logo (embedded PNG), shown on the "no comic open" card.
    /// Decoded lazily on first paint of the empty state.
    pub(crate) logo: Option<egui::TextureHandle>,
    /// The library's series sections (scanned off-thread; see `spawn_library_scan`).
    pub(crate) series: Vec<Series>,
    pub(crate) series_rx: std::sync::mpsc::Receiver<Vec<Series>>,
    pub(crate) series_tx: std::sync::mpsc::Sender<Vec<Series>>,
    /// A series scan is in flight ("Scanning…" placeholder).
    pub(crate) series_pending: bool,
    /// Library sub-mode: the old folder grid, kept solely for picking a new
    /// library root ("Change library…"); false = the series view.
    pub(crate) lib_browse: bool,
    /// Reading chrome: seekbar shows when `controls`; the gear opens the
    /// viewing-options popup. The zone hints show only briefly after the controls
    /// are revealed (see `controls_shown_at`), so they don't clutter while reading.
    pub(crate) controls: bool,
    pub(crate) controls_shown_at: Instant,
    /// Display name of the open comic (basename of `Shell::current_key`), shown
    /// atop the seekbar. Empty when nothing is open.
    pub(crate) book_title: String,
    /// Armed boundary prompt: a next/prev action at the last/first page showed
    /// a "Next/Previous book" card (cover + title once the cover decode lands);
    /// repeating the action — or tapping the card — within [`BOOK_PROMPT_MS`]
    /// opens the sibling. `sibling: None` ⇒ no neighbor (or a SAF-picked book,
    /// which can't resolve folder siblings) — the card just names the boundary.
    pub(crate) book_prompt: Option<BookPrompt>,
    pub(crate) show_options: bool,
    pub(crate) show_info: bool,
    /// Page index the heavy info-overlay metadata (Format/Color/Size/Modified) was
    /// last built for, so the page bytes are re-read only on a page change while the
    /// info popup is open — not every frame. `None` ⇒ rebuild on next render.
    pub(crate) info_for: Option<usize>,
    /// Cached (label, value) metadata lines for `info_for`'s page. Spliced into
    /// `build_info`'s per-frame lines so the byte read stays gated. Placeholders
    /// until the background read lands (see `info_rx`).
    pub(crate) info_meta: Vec<(String, String)>,
    /// The info overlay's metadata needs `read_page` + `modified` — disk I/O, and on
    /// a 7z still decompressing, a wait for that entry to land. It's gathered on a
    /// thread and tagged with `(open_gen, page index)` so a result for a book that
    /// has since been switched, or a page already turned past, is dropped.
    pub(crate) info_tx: Sender<InfoRows>,
    pub(crate) info_rx: Receiver<InfoRows>,
    /// Off-thread listing for the "choose a library root" folder browser, tagged
    /// with the directory it describes (a newer navigation supersedes it).
    pub(crate) browse_tx: Sender<(PathBuf, Vec<Entry>)>,
    pub(crate) browse_rx: Receiver<(PathBuf, Vec<Entry>)>,
    /// A browse listing is in flight ("Scanning…" placeholder).
    pub(crate) browse_pending: bool,
    /// The egui frame changed the viewing options / library root; the Shell picks
    /// these up after `render` and debounces the actual file writes.
    pub(crate) view_dirty: bool,
    pub(crate) libroot_dirty: bool,
    /// The user's layout choice. `reader.layout` is the concrete `Single`/`Spread`
    /// this resolves to (orientation-dependent for `Auto`); this is the source of
    /// truth that persists. See `apply_resolved_layout`.
    pub(crate) layout_mode: LayoutMode,
    /// Reopen the last book on launch (persisted in view.txt; toggled in options).
    pub(crate) resume_on_startup: bool,
    /// Whether/which way the volume rocker flips pages while reading (persisted
    /// in view.txt; picked in options). Off ⇒ it adjusts the volume as usual.
    pub(crate) vol_keys: VolKeys,
    /// UI language picked in options (persisted in view.txt); `None` follows the
    /// OS, whose language (`system_lang`) is read once per App build.
    pub(crate) lang_pref: Option<Lang>,
    pub(crate) system_lang: Lang,
    /// Book-gutter shading on un-joined two-page spreads, and its peak darkening
    /// (persisted in view.txt; toggled in options). The reader takes the product.
    pub(crate) spine_shadow_on: bool,
    pub(crate) spine_shadow_strength: f32,
    /// Chrome theme preference (persisted in view.txt; toggled in options).
    pub(crate) theme: ThemePref,
    /// Performance profile (persisted in view.txt; toggled in options). Applied
    /// live by `apply_perf` — no book reopen.
    pub(crate) perf: PerfPref,
    /// The tier the hardware probe chose, used whenever `perf` is `Auto`. Probed
    /// once per GPU build (a resume re-probes, which is free and correct).
    pub(crate) auto_tier: DeviceTier,
    /// The `Budget::for_tier` inputs, kept so the performance setting can recompute
    /// a budget at runtime without re-reading `/proc`.
    pub(crate) mem_budget_mb: u64,
    pub(crate) cpus: usize,
    /// Cached OS night-mode flag (re-read on resume); resolves `ThemePref::System`.
    pub(crate) system_dark: bool,
    /// The shell's platform hooks (the font reload on a language switch needs them).
    pub(crate) platform: Rc<dyn Platform>,
    /// Light/dark actually pushed into the egui context (`ctx.set_visuals` is
    /// persistent state, so re-applying it every frame just rebuilds a `Visuals`
    /// for nothing). `None` ⇒ not applied yet. Mirrors `Shell::applied_immersive`.
    pub(crate) applied_light: Option<bool>,
}

/// Cross-shell actions an egui frame requests (handled after the egui run, since
/// they need `Shell` state the reader/render path doesn't own).
#[derive(Default)]
pub(crate) struct FrameReqs {
    /// Open this comic (by path).
    pub(crate) open: Option<PathBuf>,
    /// Request all-files access.
    pub(crate) grant: bool,
    /// Open the SAF single-file picker (fallback for files outside the library).
    pub(crate) open_picker: bool,
    /// Open the library browser (from the empty-state helper).
    pub(crate) open_library: bool,
    /// Toggle a library series' collapsed state (persisted on the Shell).
    pub(crate) toggle_series: Option<PathBuf>,
    /// Seekbar page button: reading-order step (-1 prev / +1 next).
    pub(crate) page_nav: i64,
    /// Seekbar book button: open the prev/next sibling comic in the folder (-1/+1).
    pub(crate) book_nav: i64,
    /// Seekbar bookshelf button: return to the library, leaving the book warm.
    pub(crate) to_library: bool,
}

/// Gather the info overlay's byte-derived metadata for page `index`: format,
/// colour profile, compressed size and mtime. A free function over the bare source
/// because it runs on a **background** thread — `read_page` and `modified` are disk
/// I/O, and on an archive still decompressing `read_page` blocks until that entry
/// lands, which would freeze the UI for as long as the extraction takes.
fn page_meta(src: &dyn PageSource, index: usize) -> Vec<(String, String)> {
    let modified = src.modified(index).unwrap_or_else(|| "—".to_string());
    let (size, fmt, color) = match src.read_page(index).ok() {
        Some(b) => {
            let detail = yosh_engine::meta::probe(&b).2;
            let color = yosh_engine::icc::extract_icc(&b)
                .as_deref()
                .and_then(yosh_engine::icc::describe)
                .unwrap_or_else(|| "—".to_string());
            (yosh_engine::meta::human_size(b.len() as u64), detail, color)
        }
        None => ("—".to_string(), "—".to_string(), "—".to_string()),
    };
    vec![
        (tr().info_format.to_string(), fmt),
        (tr().info_color.to_string(), color),
        (tr().info_size.to_string(), size),
        (tr().info_modified.to_string(), modified),
    ]
}

impl App {
    /// Start an off-thread series scan of the library root (results land via
    /// `series_rx` in render).
    pub(crate) fn kick_series_scan(&mut self) {
        self.series_pending = true;
        spawn_library_scan(
            self.lib_root.clone(),
            self.series_tx.clone(),
            self.frame_waker.clone(),
        );
        self.window.request_redraw();
    }

    /// List `lib_dir` for the "choose a library root" browser on a background
    /// thread (`read_dir` + a stat per entry — no different from any other storage
    /// walk on a phone). The result is tagged with its directory, so a listing the
    /// user has already navigated away from is discarded.
    fn kick_browse_scan(&mut self) {
        self.browse_pending = true;
        self.lib_entries.clear();
        let dir = self.lib_dir.clone();
        let tx = self.browse_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let entries = scan_dir(&dir);
            let _ = tx.send((dir, entries));
            if let Some(w) = &waker {
                w();
            }
        });
        self.window.request_redraw();
    }

    /// Refresh the heavy info-overlay metadata (Format / Color / Size / Modified)
    /// for the displayed page. A no-op unless the page changed since the last build.
    ///
    /// The values need the page bytes — disk I/O, and on an archive still
    /// decompressing a wait for that entry — so they are gathered **off-thread**
    /// (`page_meta`), tagged with `(open_gen, page)`; the overlay shows "…"
    /// placeholders until the result lands in `render`'s drain. Mirrors the desktop
    /// `page_info` pattern.
    fn refresh_info_meta(&mut self, open_gen: u64) {
        let Some(src) = &self.reader.source else {
            self.info_for = None;
            self.info_meta.clear();
            return;
        };
        let idx = self.reader.visible_pages().0;
        if self.info_for == Some(idx) {
            return;
        }
        self.info_for = Some(idx);
        let t = tr();
        self.info_meta = [t.info_format, t.info_color, t.info_size, t.info_modified]
            .into_iter()
            .map(|k| (k.to_string(), "…".to_string()))
            .collect();
        let src = src.clone();
        let tx = self.info_tx.clone();
        let waker = self.frame_waker.clone();
        std::thread::spawn(move || {
            let _ = tx.send((open_gen, idx, page_meta(src.as_ref(), idx)));
            if let Some(w) = &waker {
                w();
            }
        });
    }

    /// Build the image-info overlay lines from the current page + reader state
    /// (the Android take on the desktop's "I" overlay). The heavy metadata
    /// (`info_meta`) is refreshed separately via `refresh_info_meta` (gated).
    fn build_info(&self) -> Vec<(String, String)> {
        let Some(src) = &self.reader.source else {
            return Vec::new();
        };
        let len = src.len();
        let idx = self.reader.visible_pages().0;
        let t = tr();
        let mut lines = vec![
            (t.info_book.to_string(), self.book_title.clone()),
            (t.info_file.to_string(), src.name(idx).to_string()),
            (t.info_page.to_string(), format!("{} / {}", idx + 1, len)),
            (
                t.info_screen.to_string(),
                format!("{} × {}", self.config.width, self.config.height),
            ),
        ];
        // Format / Color / Size / Modified — read from the page bytes, refreshed
        // only on a page change (see `refresh_info_meta`) so this stays per-frame cheap.
        lines.extend(self.info_meta.iter().cloned());
        if let Some(p) = self.reader.cache.get(idx) {
            lines.push((t.info_source.to_string(), format!("{} × {}", p.src_w, p.src_h)));
            lines.push((t.info_decoded.to_string(), format!("{} × {}", p.w, p.h)));
            lines.push((
                t.info_resize.to_string(),
                if p.lq {
                    "LQ (fast bilinear)".to_string()
                } else {
                    p.path.label().to_string()
                },
            ));
            // Single-resize invariant readout: the GPU should sample 1:1.
            let gpu = match self.reader.page_target_h(idx).cmp(&p.target_h) {
                std::cmp::Ordering::Equal => "1:1",
                std::cmp::Ordering::Greater => "↑ upscale",
                std::cmp::Ordering::Less => "↓ downscale",
            };
            lines.push((t.info_gpu.to_string(), gpu.to_string()));
        } else {
            lines.push((t.info_decoded.to_string(), "decoding…".to_string()));
        }
        lines.push((
            t.info_zoom.to_string(),
            format!("{:.0}%", self.reader.effective_zoom_pct()),
        ));
        lines.push((t.fit.to_string(), self.reader.fit.label().to_string()));
        // For Auto, show the resolved layout too (e.g. "auto · spread").
        let layout_str = match self.layout_mode {
            LayoutMode::Auto => format!("auto · {}", self.reader.layout.label()),
            m => m.label().to_string(),
        };
        lines.push((t.info_layout.to_string(), layout_str));
        lines.push((
            t.info_direction.to_string(),
            self.reader.direction.label().to_string(),
        ));
        // LQ preview tier: fill progress + what's on screen for this page.
        let showing = if self.reader.cache.contains(idx) {
            "HQ"
        } else if self.reader.lq_cache.contains(idx) {
            "LQ preview"
        } else {
            "—"
        };
        lines.push((
            t.info_lq_tier.to_string(),
            format!("{}/{} · {}", self.reader.lq_cache.len(), len, showing),
        ));
        lines
    }

    /// Flag the viewing options as changed. The Shell picks this up after the frame
    /// and debounces the write (`flush_saves`), reading the values back off the App
    /// then — so a run of toggles costs one atomic rewrite of `view.txt`, not one
    /// per tap.
    fn persist_view(&mut self) {
        self.view_dirty = true;
    }

    /// Apply the performance profile to the live reader — **no book reopen**. The
    /// budget's fields are all runtime-settable: the window sizes are plain fields,
    /// both caches and the texture pool can be re-capped in place (evicting down
    /// keeps the pages nearest the read position, so nothing on screen flashes),
    /// and the worker count is the one thing that needs a new pool. Rebuilding the
    /// pool is cheap and hitch-free because teardown is signal-only and the caches
    /// are deliberately **kept** — the current page stays decoded and on screen
    /// while the new pool refills around it.
    fn apply_perf(&mut self) {
        let tier = self.perf.tier().unwrap_or(self.auto_tier);
        let b = Budget::for_tier(tier, self.mem_budget_mb, self.cpus);
        log::info!("perf {} → tier {tier:?}, budget: {b:?}", self.perf.label());
        let current = self.reader.index;
        self.reader.workers = b.workers;
        self.reader.fwd = b.fwd;
        self.reader.fwd_max = b.fwd_max;
        self.reader.back = b.back;
        self.reader.actual_cap_vh = b.actual_cap_vh;
        self.reader.lq_thumb_h = b.lq_thumb_h;
        self.reader.lq_tier = b.lq_tier;
        self.reader.cache.set_cap(b.cache_cap, current);
        self.reader.lq_cache.set_cap(b.lq_cap, current);
        self.reader.tex_pool.set_max_total(b.texpool_max);
        // Nothing open (library view): the fields above are enough — the pool is
        // built with `reader.workers` when the next book opens.
        let Some(src) = self.reader.source.clone() else {
            return;
        };
        let pool = DecodePool::new(
            src,
            self.ctx.device.clone(),
            self.ctx.queue.clone(),
            self.reader.tex_pool.clone(),
            b.workers,
        );
        // A fresh pool starts wakerless and with empty queues: re-install the wake
        // callback, then force a full rebuild of both the job list and the
        // thumbnail tail (`JobsKey` doesn't capture budget fields, so without this
        // the memoized key would suppress the very prefetch that refills the pool).
        pool.set_waker(self.reader.waker.clone());
        self.reader.pool = Some(pool);
        self.reader.invalidate_jobs();
        self.reader.prefetch();
    }

    /// Resolve the layout mode against the current viewport and, if it differs from
    /// the engine's concrete layout, switch — re-anchoring the read position into the
    /// new view and re-prefetching. This is what makes `Auto` follow device rotation;
    /// for fixed Single/Spread it's a per-frame no-op. Does NOT persist (the *mode*
    /// is unchanged — only its orientation-dependent resolution).
    fn apply_resolved_layout(&mut self) {
        let desired = self.layout_mode.resolve(self.config.width, self.config.height);
        if desired != self.reader.layout {
            self.reader.layout = desired;
            self.reader.snap_to_view_start();
            self.reader.prefetch();
        }
    }

    pub(crate) fn render(
        &mut self,
        has_files: bool,
        status_bar_px: i32,
        opening: Option<&str>,
        open_gen: u64,
        lib: LibCtx<'_>,
    ) -> FrameReqs {
        self.frame_no += 1;
        // Resolved chrome theme this frame: drives the page-letterbox clear color
        // (below) and egui's visuals (set at the top of the egui run). The seekbar's
        // hardcoded colors read it back off `ui.visuals().dark_mode`.
        let light = !self.theme.is_dark(self.system_dark);
        self.reader.viewport = Viewport {
            w: self.config.width,
            h: self.config.height,
        };
        // Resolve Auto → concrete layout from the current orientation *before* the
        // decode view / prefetch / build-quads below read `reader.layout`, so a
        // rotation switches this same frame (no one-frame flash of the old layout).
        self.apply_resolved_layout();
        // Drain finished decodes into the right cache (full-res → cache, LQ-tier
        // thumbnails → lq_cache) and record failures.
        self.reader.drain_pool();
        self.reader.update_decode_view();
        // Scroll mode: keep the (anchor, top_offset) valid as page heights resolve,
        // before prefetch reads the visible window.
        if self.reader.scroll_mode {
            self.reader.normalize();
        }
        self.reader.prefetch();

        let quads = if self.reader.scroll_mode {
            self.reader.build_scroll_quads()
        } else {
            self.reader.build_quads()
        };
        let anim_t = self.anim_origin.elapsed();
        // Did this frame draw a free-running animation (GIF/WebP)? The end-of-frame
        // redraw guard keeps the loop alive while one is on screen — without this,
        // animations only advance on input events.
        let mut drew_live_anim = false;
        let page_bgs: Vec<wgpu::BindGroup> = quads
            .iter()
            .filter_map(|q| {
                self.reader.page_texture(q.page_index).map(|t| {
                    drew_live_anim |= t.is_animation();
                    let view = t.view_at(anim_t);
                    self.page_pipeline.prepare_quad(
                        &self.ctx.device,
                        &self.ctx.queue,
                        q.slot,
                        t,
                        view,
                        q.scale,
                        q.offset,
                        q.rot,
                        q.alpha,
                        q.blur,
                        q.spine,
                        q.spine_strength,
                    )
                })
            })
            .collect();

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.ctx.device, &self.config);
                self.window.request_redraw();
                return FrameReqs::default();
            }
            _ => {
                self.window.request_redraw();
                return FrameReqs::default();
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("page"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Letterbox behind pages: white in light/e-ink mode (a big dark
                        // fill is unusable on a reflective panel), else the dark #202020.
                        load: wgpu::LoadOp::Clear(if light {
                            wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
                        } else {
                            wgpu::Color {
                                r: 32.0 / 255.0,
                                g: 32.0 / 255.0,
                                b: 32.0 / 255.0,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !page_bgs.is_empty() {
                pass.set_pipeline(&self.page_pipeline.pipeline);
                for bg in &page_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
        }
        // Drain decoded covers into egui textures, and any finished series scan.
        while let Ok((path, img)) = self.thumb_rx.try_recv() {
            self.covers_inflight = self.covers_inflight.saturating_sub(1);
            // A cover that wouldn't decode (unreadable or empty archive) stays in
            // `queued_covers`, so it isn't retried every frame — it just never
            // gets a thumbnail.
            let Some(img) = img else { continue };
            let handle =
                self.egui_ctx
                    .load_texture(path.to_string_lossy(), img, egui::TextureOptions::default());
            self.thumb_used.insert(path.clone(), self.frame_no);
            self.thumbs.insert(path, handle);
        }
        while let Ok(series) = self.series_rx.try_recv() {
            self.series = series;
            self.series_pending = false;
        }
        // Off-thread info-overlay metadata: both tags must still match — a result for
        // a book that has since been switched (generation) or for a page already
        // turned past (index) describes something other than what's on screen.
        while let Ok((generation, idx, meta)) = self.info_rx.try_recv() {
            if generation == open_gen && self.info_for == Some(idx) {
                self.info_meta = meta;
            }
        }
        // Off-thread browse listing, keyed by the directory it describes (a newer
        // navigation already replaced `lib_dir`, so its own result is what lands).
        while let Ok((dir, entries)) = self.browse_rx.try_recv() {
            if dir == self.lib_dir {
                self.lib_entries = entries;
                self.browse_pending = false;
            }
        }
        // egui chrome over the page: the library browser when open, else the seekbar.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let len = self.reader.source.as_ref().map(|s| s.len()).unwrap_or(0);
        let cur = self.reader.index;
        let library_view = self.library_view;
        let controls = self.controls;
        // Buffered (decode-ahead ready) page indices for the seekbar's cache bar.
        // Built only when the seekbar will actually draw (same condition as the
        // `seekbar()` arm below) — `lq_buffered` alone is an lq_cap-sized Vec, and
        // reading with the chrome hidden is the common case.
        let seekbar_visible = controls && !library_view && len > 0;
        let (buffered, lq_buffered): (Vec<usize>, Vec<usize>) = if seekbar_visible {
            (
                self.reader.cache.buffered_indices().collect(),
                self.reader.lq_cache.buffered_indices().collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let book_title = self.book_title.clone();
        let hints_visible = controls && self.controls_shown_at.elapsed().as_millis() < 1500;
        let show_options = self.show_options;
        let show_info = self.show_info;
        if show_info {
            self.refresh_info_meta(open_gen); // gated, off-thread → Format/Color/Size/Modified
        } else {
            self.info_for = None; // rebuild the metadata when the popup is next opened
        }
        let info_lines = if show_info { self.build_info() } else { Vec::new() };
        let rtl = self.reader.direction == Direction::Rtl;
        let cur_dir = self.reader.direction;
        // Resolved concrete layout (apply_resolved_layout ran at the top of render),
        // and the user's chosen mode for the popup's selection state.
        let cur_layout = self.reader.layout;
        let cur_layout_mode = self.layout_mode;
        let cur_fit = self.reader.fit;
        // The popup row is phrased as "Stretch small pages", so invert the flag.
        let cur_stretch_on = !self.reader.fit_no_upscale;
        let cur_transition = self.reader.transition_enabled;
        let cur_spine_on = self.spine_shadow_on;
        let cur_spine_strength = self.spine_shadow_strength;
        let cur_resume = self.resume_on_startup;
        let cur_vol_keys = self.vol_keys;
        let cur_lang = self.lang_pref;
        let cur_scroll = self.reader.scroll_mode;
        let cur_theme = self.theme;
        let cur_perf = self.perf;
        let cur_rotation = self.reader.rotation;
        let drag_seam = self.reader.drag_seam();
        let cur_zoom = self.reader.zoom;
        // Path labels for the library header — only the library view shows them,
        // so don't format two paths per reading frame.
        let (lib_dir_str, lib_root_str) = if library_view {
            (
                self.lib_dir.display().to_string(),
                self.lib_root.display().to_string(),
            )
        } else {
            (String::new(), String::new())
        };
        let browse = self.lib_browse;
        let series_pending = self.series_pending;
        let browse_pending = self.browse_pending;
        // Browse mode (choose a library root) lists folders only — owned snapshot.
        let entries: Vec<(String, PathBuf)> = if library_view && browse {
            self.lib_entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Dir(p) => Some((name_of(p), p.clone())),
                    Entry::Comic(_) => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        // Series-view snapshot: per-volume read state + covers, per-series status
        // label + collapse state. Owned, so the closure borrows no `self`.
        let sections: Vec<SectionRow> = if library_view && !browse {
            self.series
                .iter()
                .map(|s| {
                    let expanded = !lib.collapsed.contains(&s.dir);
                    let states: Vec<VolState> = s
                        .volumes
                        .iter()
                        .map(|v| vol_state(lib.progress, lib.positions, &v.to_string_lossy()))
                        .collect();
                    let status = series_status(&states);
                    let volumes = if expanded {
                        s.volumes
                            .iter()
                            .zip(&s.captions)
                            .zip(&states)
                            .map(|((v, caption), st)| VolCell {
                                label: caption.clone(),
                                thumb: self.thumbs.get(v).cloned(),
                                state: *st,
                                is_current: lib.current_key
                                    == Some(v.to_string_lossy().as_ref()),
                                path: v.clone(),
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    SectionRow {
                        dir: s.dir.clone(),
                        name: s.name.clone(),
                        expanded,
                        status,
                        volumes,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // "Recently read" cells (newest first), built like `sections` so the egui
        // closure borrows no `self`. Filtered to reopenable filesystem entries — the
        // only ones with a cover + a working tap-to-open (SAF one-offs are skipped).
        // The filter reads the background-computed set (`Shell::recents_ok`); it used
        // to `Path::exists` all 32 recents on every library frame.
        let recents_cells: Vec<VolCell> = if library_view && !browse {
            lib.recents
                .iter()
                .filter(|k| lib.recents_ok.contains(k.as_str()))
                .take(12)
                .map(|k| {
                    let path = PathBuf::from(k);
                    VolCell {
                        // No series header over this shelf to carry the series name,
                        // so the label keeps it — only the extension goes.
                        label: yosh_engine::caption::display_stem(&name_of(&path)).to_string(),
                        thumb: self.thumbs.get(&path).cloned(),
                        state: vol_state(lib.progress, lib.positions, k.as_str()),
                        is_current: lib.current_key == Some(k.as_str()),
                        path,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        // Boundary prompt: lazily clear once its window lapses; visibility is
        // sampled ONCE here and reused for both the draw and the redraw guard
        // (the draw-time-consistency rule — see `Reader::animation_drawn`).
        if self
            .book_prompt
            .as_ref()
            .is_some_and(|p| p.at.elapsed().as_millis() >= BOOK_PROMPT_MS as u128)
        {
            self.book_prompt = None;
        }
        let prompt_card: Option<(i64, String, Option<egui::TextureHandle>, bool)> =
            if !library_view && len > 0 {
                self.book_prompt.as_ref().map(|p| {
                    let thumb = p.sibling.as_ref().and_then(|s| self.thumbs.get(s)).cloned();
                    (p.dir, p.title.clone(), thumb, p.sibling.is_some())
                })
            } else {
                None
            };
        let prompt_visible = prompt_card.is_some();
        let mut seek_to: Option<usize> = None;
        let mut reqs = FrameReqs::default();
        let mut go_up = false;
        let mut nav_to: Option<PathBuf> = None;
        let mut close_lib = false;
        let mut set_root = false;
        let mut enter_browse = false;
        let mut leave_browse = false;
        let mut rescan = false;
        let mut toggle_series: Option<PathBuf> = None;
        // Volume paths in expanded sections that were actually on screen this
        // frame: bumps their cover LRU stamps + lazily queues missing decodes.
        let mut visible_covers: Vec<PathBuf> = Vec::new();
        let mut open_options = false;
        let mut set_dir: Option<Direction> = None;
        let mut set_layout: Option<LayoutMode> = None;
        let mut set_fit: Option<FitMode> = None;
        let mut set_no_upscale: Option<bool> = None;
        let mut set_transition: Option<bool> = None;
        let mut set_spine_on: Option<bool> = None;
        let mut set_spine_strength: Option<f32> = None;
        let mut set_resume: Option<bool> = None;
        let mut set_vol_keys: Option<VolKeys> = None;
        let mut set_lang: Option<Option<Lang>> = None;
        let mut set_scroll: Option<bool> = None;
        let mut set_theme: Option<ThemePref> = None;
        let mut set_perf: Option<PerfPref> = None;
        let mut toggle_offset = false;
        let mut rotate = false;
        let mut cycle_fit = false;
        let mut to_library = false;
        let mut page_nav: i64 = 0;
        let mut book_nav: i64 = 0;
        let mut open_info = false;
        let mut close_info = false;
        // Browse mode may climb anywhere (picking a new root); only the
        // filesystem root has no Up.
        let at_root = self.lib_dir.parent().is_none();
        // Library header "Resume <book>" button — drawn only when a book is open (the
        // library is the home; there's nothing to return to otherwise). It names the
        // book so it hints the destination; clip the title (by char count, since titles
        // are often CJK) so a long name can't push the other header buttons off-screen.
        let resume_label = (len > 0).then(|| {
            let t = i18n::truncate_cols(&self.book_title, 14);
            format!("{}  {}", egui_phosphor::fill::BOOK_OPEN, (tr().resume_book)(&t))
        });
        // While the bars are shown the window is still full-bleed (NativeActivity
        // ignores fitSystemWindows), so inset the library's top chrome by the
        // status-bar height — otherwise the clock/icons cover the top buttons.
        let top_inset_pt = status_bar_px as f32 / self.egui_ctx.pixels_per_point();
        // `set_visuals` is persistent context state, so it only has to run when the
        // resolved theme actually flips (first frame, options toggle, OS night-mode
        // change) — not once per frame.
        let apply_visuals = self.applied_light != Some(light);
        self.applied_light = Some(light);
        let full_output = self.egui_ctx.run_ui(raw_input, |root| {
            // Panels go in the root `Ui` (egui 0.36); Areas still take the `Context`,
            // a cheap shared handle.
            let ctx = &root.ctx().clone();
            // Theme the whole frame up front: standard widgets (library grid, popups,
            // empty-state card, series headers) follow this; the hand-painted chrome
            // (seekbar, letterbox) reads `dark_mode` back off the visuals.
            if apply_visuals {
                ctx.set_visuals(if light {
                    egui::Visuals::light()
                } else {
                    egui::Visuals::dark()
                });
            }
            if library_view {
                egui::CentralPanel::default().show(root, |ui| {
                    ui.add_space(top_inset_pt);
                    // egui paints a fade-out gradient at scroll edges (default
                    // strength 0.5); on the cover grid it dims the thumbnails at the
                    // screen bottom, reading as a stray drop shadow. Turn it off.
                    ui.spacing_mut().scroll.fade.strength = 0.0;
                    if !has_files {
                        ui.add_space(40.0);
                        ui.label(tr().grant_prompt);
                        if ui.button(tr().grant_button).clicked() {
                            reqs.grant = true;
                        }
                    } else if browse {
                        // Choose-a-library-root folder browser (folders only).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if ui.button(egui::RichText::new(tr().back).size(18.0)).clicked() {
                                leave_browse = true;
                            }
                            if !at_root
                                && ui.button(egui::RichText::new(tr().up).size(18.0)).clicked()
                            {
                                go_up = true;
                            }
                            if ui
                                .button(egui::RichText::new(tr().set_as_library).size(18.0))
                                .clicked()
                            {
                                set_root = true;
                            }
                        });
                        ui.label(egui::RichText::new(&lib_dir_str).size(13.0));
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (label, path) in &entries {
                                if ui
                                    .add_sized(
                                        [ui.available_width(), 44.0],
                                        egui::Button::new(
                                            egui::RichText::new(format!("📁 {label}")).size(16.0),
                                        ),
                                    )
                                    .clicked()
                                {
                                    nav_to = Some(path.clone());
                                }
                            }
                            if entries.is_empty() {
                                ui.label(
                                    egui::RichText::new(if browse_pending {
                                        tr().scanning
                                    } else {
                                        tr().no_subfolders
                                    })
                                    .weak(),
                                );
                            }
                        });
                    } else {
                        // Series view: collapsible sections, one horizontal
                        // cover row per series (Chunky-style).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
                            if let Some(label) = &resume_label
                                && ui.button(egui::RichText::new(label).size(18.0)).clicked()
                            {
                                close_lib = true;
                            }
                            if ui
                                .button(egui::RichText::new(tr().change_library).size(18.0))
                                .clicked()
                            {
                                enter_browse = true;
                            }
                            if ui.button(egui::RichText::new(tr().open_file).size(18.0)).clicked() {
                                reqs.open_picker = true;
                            }
                            if ui.button(egui::RichText::new("⟳").size(18.0)).clicked() {
                                rescan = true;
                            }
                        });
                        ui.label(egui::RichText::new(&lib_root_str).size(13.0));
                        ui.separator();
                        if sections.is_empty() {
                            ui.add_space(40.0);
                            ui.vertical_centered(|ui| {
                                ui.label(if series_pending {
                                    tr().scanning_library
                                } else {
                                    tr().no_comics_found
                                });
                            });
                        }
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            // "Recently read" row above the series (most-recent first).
                            if !recents_cells.is_empty() {
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new(tr().recently_read).size(18.0).strong(),
                                );
                                visible_covers
                                    .extend(recents_cells.iter().map(|v| v.path.clone()));
                                egui::ScrollArea::horizontal()
                                    .id_salt("recents_row")
                                    .scroll_bar_visibility(
                                        egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                                    )
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            for v in &recents_cells {
                                                volume_cell(ui, v, &mut reqs, &mut close_lib);
                                            }
                                        });
                                    });
                                ui.add_space(8.0);
                                ui.separator();
                            }
                            for sec in &sections {
                                // Full-width tappable header: chevron + name
                                // left, status label right.
                                let (rect, resp) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), 40.0),
                                    egui::Sense::click(),
                                );
                                let chev = if sec.expanded {
                                    egui_phosphor::fill::CARET_DOWN
                                } else {
                                    egui_phosphor::fill::CARET_RIGHT
                                };
                                let strong = ui.visuals().strong_text_color();
                                let weak = ui.visuals().weak_text_color();
                                ui.painter().text(
                                    rect.left_center() + egui::vec2(6.0, 0.0),
                                    egui::Align2::LEFT_CENTER,
                                    format!("{chev}  {}", sec.name),
                                    egui::FontId::proportional(18.0),
                                    strong,
                                );
                                ui.painter().text(
                                    rect.right_center() - egui::vec2(6.0, 0.0),
                                    egui::Align2::RIGHT_CENTER,
                                    &sec.status,
                                    egui::FontId::proportional(13.0),
                                    weak,
                                );
                                if resp.clicked() {
                                    toggle_series = Some(sec.dir.clone());
                                }
                                if sec.expanded {
                                    // Lazy covers: only sections actually on
                                    // screen queue decodes / refresh their LRU.
                                    if ui.is_rect_visible(rect) {
                                        visible_covers
                                            .extend(sec.volumes.iter().map(|v| v.path.clone()));
                                    }
                                    egui::ScrollArea::horizontal()
                                        .id_salt(&sec.dir)
                                        .scroll_bar_visibility(
                                            egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                                        )
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                for v in &sec.volumes {
                                                    volume_cell(ui, v, &mut reqs, &mut close_lib);
                                                }
                                            });
                                        });
                                }
                                ui.add_space(8.0);
                            }
                        });
                    }
                });
            } else if let Some(name) = opening {
                // A book is being built on a worker thread: say so. Over the outgoing
                // page when switching books, over the clear color on a cold start.
                // egui's `Spinner` requests its own repaints, which the redraw guard's
                // `egui_animating` leg honors — so it actually spins while the loop
                // is otherwise idle.
                opening_card(ctx, name);
            } else if len == 0 {
                // No comic open: show the how-to-open helper instead of a blank screen.
                if self.logo.is_none()
                    && let Some(img) = decode_logo()
                {
                    self.logo = Some(ctx.load_texture(
                        "yosh_logo",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
                empty_state(ctx, self.logo.as_ref(), &mut reqs.open_library, &mut reqs.open_picker);
            } else if controls {
                seekbar(
                    ctx,
                    &book_title,
                    cur,
                    len,
                    rtl,
                    cur_layout == Layout::Spread,
                    cur_fit,
                    (cur_zoom - 1.0).abs() > 0.001,
                    &buffered,
                    &lq_buffered,
                    &mut seek_to,
                    &mut open_options,
                    &mut open_info,
                    &mut page_nav,
                    &mut book_nav,
                    &mut toggle_offset,
                    &mut cycle_fit,
                    &mut to_library,
                );
                if hints_visible {
                    zone_hints(ctx, rtl, cur_scroll);
                }
                if show_options {
                    options_popup(
                        ctx,
                        cur_dir,
                        cur_layout_mode,
                        cur_layout == Layout::Spread,
                        cur_fit,
                        cur_stretch_on,
                        cur_transition,
                        cur_spine_on,
                        cur_spine_strength,
                        cur_resume,
                        cur_vol_keys,
                        cur_lang,
                        cur_scroll,
                        cur_theme,
                        cur_perf,
                        cur_rotation,
                        &mut set_dir,
                        &mut set_layout,
                        &mut set_fit,
                        &mut set_no_upscale,
                        &mut set_transition,
                        &mut set_spine_on,
                        &mut set_spine_strength,
                        &mut set_resume,
                        &mut set_vol_keys,
                        &mut set_lang,
                        &mut set_scroll,
                        &mut set_theme,
                        &mut set_perf,
                        &mut toggle_offset,
                        &mut rotate,
                        top_inset_pt,
                    );
                }
                if show_info {
                    info_popup(ctx, &info_lines, &mut close_info);
                }
            }
            // Next/prev-book boundary prompt — over the page, chrome or not.
            // (`prompt_card` is None in the library / empty state.)
            if let Some((dir, title, thumb, has_sibling)) = &prompt_card {
                book_prompt_card(ctx, *dir, rtl, title, thumb.as_ref(), *has_sibling, &mut book_nav);
            }
            // Page-flip drag drop shadow — over the page, chrome or not.
            if !library_view
                && let Some((frac, si)) = drag_seam
            {
                drag_shadow(ctx, frac, si);
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        // egui-driven animations (panel fade-ins, the empty-state card, button
        // feedback) request immediate repaints via `repaint_delay == 0`. The redraw
        // guard below must honor that or those animations freeze on their first
        // frame now that the loop genuinely idles between events.
        let egui_animating = full_output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay.is_zero());
        if let Some(p) = seek_to {
            self.reader.goto(p);
        }
        if set_root {
            // Browse mode picked a new library root: back to the series view and
            // rescan it. The root itself is persisted by the Shell's debounced flush
            // (it was previously only written on suspend).
            self.lib_root = self.lib_dir.clone();
            self.lib_browse = false;
            self.libroot_dirty = true;
            self.kick_series_scan();
        }
        if enter_browse {
            self.lib_browse = true;
            self.lib_dir = self.lib_root.clone();
            self.kick_browse_scan();
        }
        if leave_browse {
            self.lib_browse = false;
        }
        if rescan {
            self.kick_series_scan();
        }
        if go_up {
            // Browse mode may climb past the current root — that's how a root
            // *elsewhere* gets picked.
            if let Some(parent) = self.lib_dir.parent() {
                self.lib_dir = parent.to_path_buf();
                self.kick_browse_scan();
            }
        }
        if let Some(d) = nav_to {
            self.lib_dir = d;
            self.kick_browse_scan();
        }
        if close_lib {
            self.library_view = false;
        }
        reqs.toggle_series = toggle_series;
        // Lazy covers for the sections that were on screen: bump LRU stamps,
        // queue the missing ones, and evict the least-recently-drawn past the
        // cap (a big library would otherwise pin hundreds of MB of textures).
        let mut to_decode: Vec<PathBuf> = Vec::new();
        for p in visible_covers {
            self.thumb_used.insert(p.clone(), self.frame_no);
            if !self.thumbs.contains_key(&p) && self.queued_covers.insert(p.clone()) {
                to_decode.push(p);
            }
        }
        if !to_decode.is_empty() {
            // Count them as in flight only once the worker has them: a dead worker
            // (App torn down) would otherwise leave the redraw guard armed forever.
            let queued = to_decode.len();
            if self.cover_tx.send(to_decode).is_ok() {
                self.covers_inflight += queued;
            }
        }
        const THUMB_CAP: usize = 256;
        if self.thumbs.len() > THUMB_CAP {
            let mut by_age: Vec<(u64, PathBuf)> = self
                .thumbs
                .keys()
                .map(|k| (self.thumb_used.get(k).copied().unwrap_or(0), k.clone()))
                .collect();
            by_age.sort();
            for (_, k) in by_age.into_iter().take(self.thumbs.len() - THUMB_CAP) {
                self.thumbs.remove(&k); // dropping the handle frees the texture
                self.thumb_used.remove(&k);
                self.queued_covers.remove(&k); // allow a later re-decode
            }
        }
        if open_options {
            self.show_options = !self.show_options;
        }
        if open_info {
            self.show_info = !self.show_info;
        }
        if close_info {
            self.show_info = false;
        }
        if let Some(d) = set_dir {
            self.reader.direction = d;
            self.persist_view();
        }
        if let Some(m) = set_layout {
            self.layout_mode = m;
            self.persist_view();
            // Resolve + re-anchor immediately (no one-frame lag); for Auto this also
            // sets the concrete layout to match the current orientation.
            self.apply_resolved_layout();
        }
        if let Some(f) = set_fit {
            self.reader.fit = f;
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(v) = set_no_upscale {
            // Whether fits may upscale changes every small page's displayed size, so
            // the derived view state is re-derived with it (zoom clamp against the
            // native scale, pan clamp against the drawn box, scroll anchor against
            // the strip heights) before re-prefetching at the new decode targets —
            // `JobsKey` carries the flag, so the job list rebuilds exactly once.
            self.reader.fit_no_upscale = v;
            self.reader.clamp_zoom_native();
            self.reader.clamp_pan();
            if self.reader.scroll_mode {
                self.reader.normalize();
            }
            self.reader.prefetch();
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_transition {
            self.reader.transition_enabled = v;
            self.persist_view();
        }
        if let Some(v) = set_spine_on {
            self.spine_shadow_on = v;
            self.reader.spine_strength = if v { self.spine_shadow_strength } else { 0.0 };
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_spine_strength {
            self.spine_shadow_strength = v.clamp(0.0, 1.0);
            if self.spine_shadow_on {
                self.reader.spine_strength = self.spine_shadow_strength;
            }
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_resume {
            self.resume_on_startup = v;
            self.persist_view();
        }
        if let Some(v) = set_vol_keys {
            self.vol_keys = v;
            self.persist_view();
        }
        if let Some(v) = set_lang {
            // Live: per-frame text follows `set_lang` at once; the fonts change only
            // when the new language wants another face of the CJK collection, and
            // the cached info overlay is rebuilt.
            self.lang_pref = v;
            let lang = v.unwrap_or(self.system_lang);
            if lang.cjk_face() != i18n::lang().cjk_face() {
                install_fonts(&self.egui_ctx, self.platform.as_ref(), lang.cjk_face());
            }
            i18n::set_lang(lang);
            self.info_for = None;
            self.persist_view();
            self.window.request_redraw();
        }
        if let Some(v) = set_scroll {
            self.reader.scroll_mode = v;
            self.reader.top_offset = 0.0;
            self.reader.prefetch();
            self.persist_view();
        }
        if let Some(t) = set_theme {
            self.theme = t;
            self.persist_view();
            self.window.request_redraw(); // re-paint the chrome under the new theme
        }
        if let Some(p) = set_perf {
            self.perf = p;
            self.apply_perf(); // live: no reopen, current page stays on screen
            self.persist_view();
            self.window.request_redraw();
        }
        if cycle_fit {
            if (self.reader.zoom - 1.0).abs() > 0.001 {
                // Zoomed (in OR out of the fit scale): restore the active fit by
                // dropping the zoom/pan (reader.fit already holds the fit that was
                // active before zooming).
                self.reader.zoom = 1.0;
                self.reader.pan_x = 0.0;
                self.reader.pan_y = 0.0;
            } else {
                // Already fitted: advance to the next fit mode.
                self.reader.fit = self.reader.fit.cycle();
                self.persist_view();
            }
            self.reader.prefetch();
            self.window.request_redraw();
        }
        if toggle_offset {
            self.reader.spread_offset ^= 1;
            self.reader.snap_to_view_start();
            self.reader.prefetch();
        }
        if rotate {
            self.reader.rotation = (self.reader.rotation + 1) % 4;
            // The rotated box has new bounds, so any prior pan is now out of range.
            self.reader.pan_x = 0.0;
            self.reader.pan_y = 0.0;
            self.reader.prefetch(); // re-decode at the rotation-aware 1:1 target
            self.window.request_redraw();
        }
        reqs.page_nav = page_nav;
        reqs.book_nav = book_nav;
        reqs.to_library = to_library;
        let ppp = self.egui_ctx.pixels_per_point();
        let primitives = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ppp,
        };
        // egui 0.36 can queue several deltas per texture in one frame; apply in order.
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
            }
        }
        let user_cmds = self.egui_renderer.update_buffers(
            &self.ctx.device,
            &self.ctx.queue,
            &mut enc,
            &primitives,
            &screen,
        );
        {
            let mut egui_pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut egui_pass, &primitives, &screen);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.ctx
            .queue
            .submit(user_cmds.into_iter().chain(std::iter::once(enc.finish())));
        self.window.pre_present_notify();
        self.ctx.queue.present(frame);
        // On-demand redraw: idle on a settled, decoded reading page. Pending work
        // no longer needs a leg here — everything that produces a result off-thread
        // (the decode pool, the cover worker, the series scan) sends and then calls
        // the frame waker, which is `request_redraw`, so the frame that drains it
        // is scheduled by the landing itself. What's left are the things the loop
        // can only learn about by drawing again: an unsettled decode view (one
        // confirmation frame, self-limiting), timed chrome, and animations.
        // The library still gets a leg while results are outstanding — a cheap
        // belt-and-braces for the *first* frame after a scan/cover is queued, since
        // that producer may finish before we even reach the guard.
        // Nav/zoom/pan and egui repaints wake the loop from the event handler.
        if (self.library_view && (self.series_pending || self.covers_inflight > 0))
            || !self.reader.view_settled
            || hints_visible
            // An open is in flight. Its worker wakes the loop when it lands, but a
            // resume in the meantime replaces the window that waker belongs to — so
            // keep drawing until the result has actually been drained. Self-limiting
            // (the open finishes), and the spinner is animating anyway whenever it's
            // the thing on screen.
            || opening.is_some()
            // Draw-time flag, NOT transition_active()/drag_active(): re-sampling
            // the clock here can see the animation as just-expired even though
            // the frame above drew it mid-fade — freezing a half-faded ghost of
            // the outgoing page on screen (the decision must match the draw).
            || self.reader.animation_drawn() // page-turn / drag frame was drawn
            || prompt_visible // boundary prompt on screen (timed; same sample as draw)
            || drew_live_anim // a GIF/WebP is playing on screen
            || egui_animating // egui fade/feedback animation mid-flight
        {
            self.window.request_redraw();
        }
        reqs
    }
}

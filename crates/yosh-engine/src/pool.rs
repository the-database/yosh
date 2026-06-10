//! Decode pool: N worker threads that read → decode → downscale → upload a page
//! to a GPU texture, off the main thread. wgpu `Device`/`Queue` are `Send+Sync`,
//! so workers create textures and `write_texture` themselves; the main thread
//! only swaps in finished textures.
//!
//! The job list is rebuilt by the scheduler each navigation (nearest-first), so
//! workers always pick the highest-priority page relative to the latest position.

use std::collections::HashSet;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use fast_image_resize::Resizer;

use crate::decode::{decode_page, DecodedPage};
use crate::source::PageSource;
use crate::page::{PagePipeline, PageTexture};
use crate::texpool::TexturePool;

pub enum Msg {
    /// A finished page. `thumb` distinguishes a whole-volume LQ *thumbnail* (routes
    /// to the reader's `lq_cache`) from a normal window decode (routes to `cache`).
    Done { index: usize, page: PageTexture, thumb: bool },
    Failed { index: usize, error: String },
}

struct JobState {
    /// Pending decodes as `(page index, exact target height, lq, thumb)` — the target
    /// is the page's on-screen displayed height; `lq` requests the fast (seeking) tier;
    /// `thumb` marks a whole-volume LQ-tier thumbnail (small target, routed separately).
    jobs: Vec<(usize, u32, bool, bool)>,
    /// The source workers read from. Lives here (rather than captured per worker) so a
    /// live folder refresh can swap in a grown listing via `set_source` without tearing
    /// down and rebuilding the thread pool. A worker clones it while claiming a job.
    source: Arc<dyn PageSource>,
    inflight: HashSet<usize>,
    /// Indices the latest prefetch window still wants decoded (the raw `set_jobs`
    /// list, *before* the inflight filter). A worker checks this at its yield
    /// points and abandons a page that has fallen out of the window — so a far
    /// jump or fast scrub doesn't make workers finish now-offscreen decodes first.
    wanted: HashSet<usize>,
    running: bool,
}

pub struct DecodePool {
    shared: Arc<(Mutex<JobState>, Condvar)>,
    results: Receiver<Msg>,
    handles: Vec<JoinHandle<()>>,
}

impl DecodePool {
    pub fn new(
        source: Arc<dyn PageSource>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        tex_pool: Arc<TexturePool>,
        workers: usize,
    ) -> Self {
        let shared = Arc::new((
            Mutex::new(JobState {
                jobs: Vec::new(),
                source,
                inflight: HashSet::new(),
                wanted: HashSet::new(),
                running: true,
            }),
            Condvar::new(),
        ));
        let (tx, rx): (Sender<Msg>, Receiver<Msg>) = channel();
        let mut handles = Vec::new();

        for _ in 0..workers {
            let shared = shared.clone();
            let device = device.clone();
            let queue = queue.clone();
            let tex_pool = tex_pool.clone();
            let tx = tx.clone();
            handles.push(std::thread::spawn(move || {
                let mut resizer = Resizer::new();
                // Has `index` fallen out of the wanted window? If so, drop it from
                // `inflight` (so it can be re-queued later) and report stale. Called
                // at the read→decode and decode→upload boundaries to abandon work the
                // latest navigation made useless.
                let stale = |index: usize| -> bool {
                    let (m, _) = &*shared;
                    let mut st = m.lock().unwrap();
                    if st.wanted.contains(&index) {
                        false
                    } else {
                        st.inflight.remove(&index);
                        true
                    }
                };
                let drop_inflight = |index: usize| {
                    let (m, _) = &*shared;
                    m.lock().unwrap().inflight.remove(&index);
                };
                loop {
                    // Wait for and claim the highest-priority job (index + its
                    // exact, per-page decode target height), grabbing the current
                    // source under the same lock so a live `set_source` swap is
                    // picked up on the next claimed job.
                    let (index, th, lq, thumb, source): (usize, u32, bool, bool, Arc<dyn PageSource>) = {
                        let (m, cv) = &*shared;
                        let mut st = m.lock().unwrap();
                        loop {
                            if !st.running {
                                return;
                            }
                            if !st.jobs.is_empty() {
                                let (i, h, lq, thumb) = st.jobs.remove(0);
                                st.inflight.insert(i);
                                let source = st.source.clone();
                                break (i, h, lq, thumb, source);
                            }
                            st = cv.wait(st).unwrap();
                        }
                    };

                    let bytes = match source.read_page(index) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            drop_inflight(index);
                            if tx
                                .send(Msg::Failed { index, error: format!("read failed: {e}") })
                                .is_err()
                            {
                                return; // receiver gone
                            }
                            continue;
                        }
                    };

                    // Bail before the expensive decode if the jump landed during the read.
                    if stale(index) {
                        continue;
                    }

                    let decoded = decode_page(&bytes, th, lq, &mut resizer);

                    // Bail before upload if the page left the window during the decode —
                    // skips the GPU upload, the texpool/cache churn, and a pointless `Done`.
                    if stale(index) {
                        continue;
                    }

                    let page: Result<PageTexture, String> = match decoded {
                        Ok(DecodedPage::Still(img)) => {
                            Ok(PagePipeline::upload(&device, &queue, &img, &tex_pool, th))
                        }
                        Ok(DecodedPage::Animated(frames)) => Ok(PagePipeline::upload_animated(
                            &device, &queue, frames, &tex_pool, th, true,
                        )),
                        // `.ico` layers: same multi-frame texture, but not an
                        // auto-playing animation (no delays, manual stepping).
                        Ok(DecodedPage::Layered(layers)) => Ok(PagePipeline::upload_animated(
                            &device,
                            &queue,
                            layers.into_iter().map(|i| (i, 0u32)).collect(),
                            &tex_pool,
                            th,
                            false,
                        )),
                        Err(e) => Err(e),
                    };

                    drop_inflight(index);

                    let msg = match page {
                        Ok(mut page) => {
                            page.lq = lq;
                            Msg::Done { index, page, thumb }
                        }
                        Err(error) => Msg::Failed { index, error },
                    };
                    if tx.send(msg).is_err() {
                        return; // receiver gone
                    }
                }
            }));
        }

        Self {
            shared,
            results: rx,
            handles,
        }
    }

    /// Replace the work list with `desired` (`(index, target_h)`, nearest-first),
    /// skipping pages already in flight. Wakes idle workers.
    pub fn set_jobs(&self, desired: Vec<(usize, u32, bool, bool)>) {
        let (m, cv) = &*self.shared;
        let mut st = m.lock().unwrap();
        // `wanted` captures the full window (including in-flight indices that remain
        // in it) so legitimately-running decodes aren't falsely cancelled; the job
        // queue then drops in-flight pages to avoid re-decoding them.
        st.wanted = desired.iter().map(|(i, _, _, _)| *i).collect();
        let filtered: Vec<(usize, u32, bool, bool)> = desired
            .into_iter()
            .filter(|(i, _, _, _)| !st.inflight.contains(i))
            .collect();
        st.jobs = filtered;
        drop(st);
        cv.notify_all();
    }

    /// Swap the source the workers read from, without tearing down the thread pool.
    /// Used when a live folder refresh grows the page list by *appending* (existing
    /// indices unchanged): in-flight decodes stay valid, and there is no worker-join
    /// hitch while new pages are landing. A reorder that shifts indices instead rebuilds
    /// the pool (in `Reader::apply_refreshed_source`) so stale in-flight results can't land.
    pub fn set_source(&self, source: Arc<dyn PageSource>) {
        let (m, _) = &*self.shared;
        m.lock().unwrap().source = source;
    }

    /// Drain finished pages (non-blocking).
    pub fn poll(&self) -> Vec<Msg> {
        let mut out = Vec::new();
        while let Ok(m) = self.results.try_recv() {
            out.push(m);
        }
        out
    }
}

impl Drop for DecodePool {
    fn drop(&mut self) {
        {
            let (m, cv) = &*self.shared;
            m.lock().unwrap().running = false;
            cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

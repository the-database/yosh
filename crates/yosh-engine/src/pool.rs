//! Decode pool: N worker threads that read → decode → downscale → upload a page
//! to a GPU texture, off the main thread. wgpu `Device`/`Queue` are `Send+Sync`,
//! so workers create textures and `write_texture` themselves; the main thread
//! only swaps in finished textures.
//!
//! The job list is rebuilt by the scheduler each navigation (nearest-first), so
//! workers always pick the highest-priority page relative to the latest position.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use fast_image_resize::Resizer;

use crate::decode::{decode_page, DecodedPage};
use crate::source::PageSource;
use crate::page::{PagePipeline, PageTexture};
use crate::texpool::TexturePool;

// `Done` (a full `PageTexture`) dwarfs `Failed` (a `String`); boxing it would buy
// nothing — messages are transient and drained every frame.
#[allow(clippy::large_enum_variant)]
pub enum Msg {
    /// A finished page. `thumb` distinguishes a whole-volume LQ *thumbnail* (routes
    /// to the reader's `lq_cache`) from a normal window decode (routes to `cache`).
    Done { index: usize, page: PageTexture, thumb: bool },
    Failed { index: usize, error: String },
}

/// A shell-supplied "there is something new to draw" callback, invoked from a
/// worker thread the moment a finished page is queued. The engine stays
/// windowing-free, so the shell injects one (`window.request_redraw()` on both
/// winit shells — thread-safe there by design) and gets on-demand rendering:
/// the frame loop can idle instead of spinning at refresh rate waiting for a
/// decode it has no other way to learn about.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

struct JobState {
    /// Pending decodes as `(page index, exact target height, lq, thumb)` — the target
    /// is the page's on-screen displayed height; `lq` requests the fast (seeking) tier;
    /// `thumb` marks a whole-volume LQ-tier thumbnail (small target, routed separately).
    jobs: VecDeque<(usize, u32, bool, bool)>,
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
    /// The shell's wake callback (see [`Waker`]), set via `set_waker`. Lives here
    /// rather than being captured per worker so it can be (re)installed on a pool
    /// that is already running — shells rebuild pools at several sites, and the
    /// window the callback belongs to outlives every one of them.
    waker: Option<Waker>,
    running: bool,
}

pub struct DecodePool {
    shared: Arc<(Mutex<JobState>, Condvar)>,
    results: Receiver<Msg>,
    /// Set by the worker that wakes the shell, cleared by `poll()` before it
    /// drains. It coalesces a burst of landings into one wake — see the ordering
    /// invariant on `poll()`.
    wake_pending: Arc<AtomicBool>,
}

/// Lock the job state, recovering from poisoning. The critical sections only
/// touch plain collections (no invariants can be left half-applied by a panic),
/// so a poisoned lock must not cascade panics across the worker pool.
fn lock_jobs(m: &Mutex<JobState>) -> MutexGuard<'_, JobState> {
    m.lock().unwrap_or_else(|e| e.into_inner())
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
                jobs: VecDeque::new(),
                source,
                inflight: HashSet::new(),
                wanted: HashSet::new(),
                waker: None,
                running: true,
            }),
            Condvar::new(),
        ));
        let (tx, rx): (Sender<Msg>, Receiver<Msg>) = channel();
        let wake_pending = Arc::new(AtomicBool::new(false));

        for _ in 0..workers {
            let shared = shared.clone();
            let device = device.clone();
            let queue = queue.clone();
            let tex_pool = tex_pool.clone();
            let tx = tx.clone();
            let wake_pending = wake_pending.clone();
            // Workers are spawned **detached** — no `JoinHandle` is kept. Teardown
            // must never wait out a slow read/decode in flight (see `Drop`), and each
            // worker owns `Arc` clones of everything it touches (device, queue,
            // texture pool, source), so a straggler outliving the pool is safe.
            std::thread::spawn(move || {
                let mut resizer = Resizer::new();
                // Has `index` fallen out of the wanted window? If so, drop it from
                // `inflight` (so it can be re-queued later) and report stale. Called
                // at the read→decode and decode→upload boundaries to abandon work the
                // latest navigation made useless.
                let stale = |index: usize| -> bool {
                    let (m, _) = &*shared;
                    let mut st = lock_jobs(m);
                    if st.wanted.contains(&index) {
                        false
                    } else {
                        st.inflight.remove(&index);
                        true
                    }
                };
                let drop_inflight = |index: usize| {
                    let (m, _) = &*shared;
                    lock_jobs(m).inflight.remove(&index);
                };
                loop {
                    // Wait for and claim the highest-priority job (index + its
                    // exact, per-page decode target height), grabbing the current
                    // source under the same lock so a live `set_source` swap is
                    // picked up on the next claimed job.
                    let (index, th, lq, thumb, source): (usize, u32, bool, bool, Arc<dyn PageSource>) = {
                        let (m, cv) = &*shared;
                        let mut st = lock_jobs(m);
                        loop {
                            if !st.running {
                                return;
                            }
                            if let Some((i, h, lq, thumb)) = st.jobs.pop_front() {
                                st.inflight.insert(i);
                                let source = st.source.clone();
                                break (i, h, lq, thumb, source);
                            }
                            st = cv.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                    };

                    // The whole read → decode → upload body runs under `catch_unwind`:
                    // a panicking decoder (corrupt file) or GPU upload must not kill the
                    // worker — and must not leave `index` stuck in `inflight`, which
                    // would block that page from ever being decoded again (`set_jobs`
                    // filters in-flight indices). `None` = abandoned as stale.
                    let body = std::panic::AssertUnwindSafe(|| -> Option<Msg> {
                        let bytes = match source.read_page(index) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                drop_inflight(index);
                                return Some(Msg::Failed { index, error: format!("read failed: {e}") });
                            }
                        };

                        // Bail before the expensive decode if the jump landed during the read.
                        if stale(index) {
                            return None;
                        }

                        let decoded = decode_page(&bytes, th, lq, &mut resizer);

                        // Bail before upload if the page left the window during the decode —
                        // skips the GPU upload, the texpool/cache churn, and a pointless `Done`.
                        if stale(index) {
                            return None;
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

                        Some(match page {
                            Ok(mut page) => {
                                page.lq = lq;
                                Msg::Done { index, page, thumb }
                            }
                            Err(error) => Msg::Failed { index, error },
                        })
                    });
                    let msg = match std::panic::catch_unwind(body) {
                        Ok(Some(msg)) => msg,
                        Ok(None) => continue, // stale — abandoned mid-pipeline
                        Err(panic) => {
                            drop_inflight(index);
                            // A panic mid-resize can leave the resizer's scratch state
                            // inconsistent; start fresh.
                            resizer = Resizer::new();
                            let what = panic
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| panic.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic".to_string());
                            Msg::Failed { index, error: format!("decode panicked: {what}") }
                        }
                    };
                    if tx.send(msg).is_err() {
                        return; // receiver gone
                    }
                    // Send first, *then* flag, then wake — the order is the whole
                    // mechanism (see `poll`). Only the worker that flips the flag
                    // false→true pays for the wake, so a burst of landings costs
                    // one `request_redraw`, not one per page.
                    if !wake_pending.swap(true, Ordering::AcqRel) {
                        let w = {
                            let (m, _) = &*shared;
                            lock_jobs(m).waker.clone()
                        };
                        if let Some(w) = w {
                            w();
                        }
                    }
                }
            });
        }

        Self { shared, results: rx, wake_pending }
    }

    /// Replace the work list with `desired` (`(index, target_h)`, nearest-first),
    /// skipping pages already in flight. Wakes idle workers.
    pub fn set_jobs(&self, desired: Vec<(usize, u32, bool, bool)>) {
        let (m, cv) = &*self.shared;
        let mut st = lock_jobs(m);
        // `wanted` captures the full window (including in-flight indices that remain
        // in it) so legitimately-running decodes aren't falsely cancelled; the job
        // queue then drops in-flight pages to avoid re-decoding them.
        st.wanted = desired.iter().map(|(i, _, _, _)| *i).collect();
        st.jobs = desired
            .into_iter()
            .filter(|(i, _, _, _)| !st.inflight.contains(i))
            .collect();
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
        lock_jobs(m).source = source;
    }

    /// Install (or clear) the shell's frame-wake callback; see [`Waker`]. A setter
    /// rather than a `new` parameter because pools are rebuilt at several sites
    /// (volume switch, picker, live folder reorder) and the shell's window outlives
    /// all of them — `Reader::set_waker` remembers it and re-applies it to each new
    /// pool.
    pub fn set_waker(&self, w: Option<Waker>) {
        let (m, _) = &*self.shared;
        lock_jobs(m).waker = w;
    }

    /// Drain finished pages (non-blocking).
    ///
    /// **Ordering invariant — do not "optimize" either half.** A worker does
    /// `send → swap(true) → wake`; the main thread does `clear → drain`. Both
    /// halves must stay in that order, because together they guarantee that every
    /// sent message is drained by some frame:
    ///
    /// - If the worker's swap reads `false`, it wakes the shell. The message is
    ///   already in the channel (send came first), so the frame that wake schedules
    ///   drains it.
    /// - If the swap reads `true`, a wake is already pending and the frame it
    ///   scheduled has not run its `clear` yet — the flag would be `false`
    ///   otherwise. That frame's drain therefore happens strictly *after* this
    ///   send, and the channel is FIFO, so it picks this message up too.
    ///
    /// Clearing before draining (rather than after) is what makes the second case
    /// hold: a landing that races the drain re-arms the flag and buys another
    /// frame. The worst case is one spurious wake that finds an empty channel;
    /// a lost result — a page that decoded but never appeared — is impossible.
    pub fn poll(&self) -> Vec<Msg> {
        self.wake_pending.store(false, Ordering::Release);
        let mut out = Vec::new();
        while let Ok(m) = self.results.try_recv() {
            out.push(m);
        }
        out
    }
}

/// Teardown is **signal-only: it never joins a worker.** A join would wait out
/// whatever I/O is in flight — an HDD spin-up, a network share, a RAR entry not yet
/// decompressed — which used to stall app close *and* every volume switch (a switch
/// rebuilds the pool). Instead we tell the workers to stop and walk away: clearing
/// `wanted` makes their existing `stale()` checks abandon the current page at the
/// next yield boundary, and `running = false` retires them at the next job claim.
///
/// Safe because a straggler can't leak results into a new pool: it holds `Arc`
/// clones of everything it touches, and the results `Receiver` drops with the pool,
/// so its `tx.send` fails and it exits. The channel — never the join — was always
/// the guarantee.
impl Drop for DecodePool {
    fn drop(&mut self) {
        let (m, cv) = &*self.shared;
        let mut st = lock_jobs(m);
        st.running = false;
        st.jobs.clear();
        st.wanted.clear();
        // Drop the wake callback with the pool: a straggler still finishing a page
        // must not keep the shell's (possibly already-replaced) window alive through
        // the closure, and its `tx.send` fails anyway, so it would never wake.
        st.waker = None;
        drop(st);
        cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `wake_pending` protocol, exercised on the flag alone: a real
    /// `DecodePool` needs a wgpu `Device`/`Queue`, which a unit test can't assume
    /// (no adapter in CI), so this drives the exact same two operations the worker
    /// (`swap(true, AcqRel)`) and `poll` (`store(false, Release)`) use. It pins the
    /// two properties the decode→UI wakeup rests on: a burst of landings costs one
    /// wake, and the drain re-arms it so the *next* landing wakes again.
    #[test]
    fn wake_pending_coalesces_and_rearms() {
        let flag = Arc::new(AtomicBool::new(false));
        let wake = |f: &AtomicBool| !f.swap(true, Ordering::AcqRel);
        let drain = |f: &AtomicBool| f.store(false, Ordering::Release);

        // First landing of a frame wakes; the rest of the burst rides along.
        assert!(wake(&flag), "first send must wake the loop");
        assert!(!wake(&flag), "second send in the same burst must coalesce");
        assert!(!wake(&flag));

        // The frame those wakes bought clears before draining, so a landing that
        // races the drain arms a fresh wake rather than being swallowed.
        drain(&flag);
        assert!(wake(&flag), "a landing after the clear must wake again");

        // Under real contention: N threads land pages against one flag with no
        // intervening drain — exactly one of them pays for the wake, and it is
        // never zero (which would be a page decoded but never drawn).
        let flag = Arc::new(AtomicBool::new(false));
        let woke = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let (flag, woke) = (flag.clone(), woke.clone());
            handles.push(std::thread::spawn(move || {
                if !flag.swap(true, Ordering::AcqRel) {
                    woke.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(woke.load(Ordering::Relaxed), 1, "exactly one wake per burst");
    }
}

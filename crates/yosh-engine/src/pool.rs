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

/// How many whole-volume thumbnail decodes (the `lq_tail` tier) may run at once.
/// The tail is background filler, so it gets a hard slice of the pool rather than
/// all of it: the *priority* half of the throttle is the claim order (window
/// first), this is the *concurrency* half, which keeps a burst of full-res thumb
/// decodes from occupying every worker on a slow device.
const LQ_CONCURRENCY: usize = 2;

struct JobState {
    /// Pending decodes as `(page index, exact target height, lq, thumb)` — the target
    /// is the page's on-screen displayed height; `lq` requests the fast (seeking) tier;
    /// `thumb` marks a whole-volume LQ-tier thumbnail (small target, routed separately).
    jobs: VecDeque<(usize, u32, bool, bool)>,
    /// Lowest-priority background queue: one tiny thumbnail per page of the whole
    /// volume as `(index, target height)`, nearest-first at build time. Claimed only
    /// when `jobs` is empty, so the HQ window always preempts it. Replaced wholesale
    /// by [`DecodePool::set_lq_tail`] — the reader rebuilds it once per stride of
    /// travel instead of once per landed page, which is what keeps prefetch off the
    /// O(volume)-per-frame path.
    lq_tail: VecDeque<(usize, u32)>,
    /// Tail entries that became pointless since the tail was built (the page got a
    /// real full-res texture). Skipped — and dropped — at claim time, so cancelling
    /// costs a `HashSet` insert instead of a tail rebuild. Cleared by `set_lq_tail`.
    lq_cancel: HashSet<usize>,
    /// Thumbnail decodes currently running, by index; `len()` is the `LQ_CONCURRENCY`
    /// check. **Deliberately separate from `inflight`:** `set_jobs` filters the HQ
    /// window against `inflight`, so a thumb listed there would suppress the *real*
    /// decode of a page the reader just jumped to (and, on completion, would clear
    /// the in-flight marker of a concurrently running HQ decode of the same index).
    /// A set rather than a counter because releasing a slot must be idempotent (the
    /// panic path can run it twice) and because it also keeps one page from being
    /// thumbed twice at once.
    thumbs_inflight: HashSet<usize>,
    /// Worker-thread count, so the notify paths can wake exactly as many workers as
    /// there is claimable work for instead of the whole pool.
    workers: usize,
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

impl JobState {
    /// Claim the next job for a worker, marking it in flight, or `None` when
    /// nothing is claimable right now. Split out of the worker loop so the
    /// two-tier priority rules are unit-testable without a GPU (a real pool needs
    /// a wgpu `Device`).
    ///
    /// Priority, in order:
    /// 1. the HQ prefetch window (`jobs`) — the zero-hitch path — always wins;
    /// 2. otherwise the whole-volume thumbnail tail, and only while fewer than
    ///    [`LQ_CONCURRENCY`] thumbs are running; entries that were cancelled, or
    ///    whose page is already being thumbed, are skipped *and dropped* (a later
    ///    tail rebuild re-adds them if the page still has no preview).
    ///
    /// **`None` is exactly the worker's wait condition**, and every state change
    /// that can turn a `None` into a `Some` must notify the condvar:
    /// `set_jobs` (window jobs appear), `set_lq_tail` (tail entries appear), and
    /// the thumb-slot release in the worker's `drop_inflight` (a slot frees). The
    /// last one is the deadlock to watch: with `LQ_CONCURRENCY` thumbs running and
    /// every other worker parked on a full-slot `None`, nothing but that release
    /// can restart the tail — a missing notify there strands it until the next
    /// navigation.
    fn claim(&mut self) -> Option<(usize, u32, bool, bool)> {
        if let Some((i, h, lq, thumb)) = self.jobs.pop_front() {
            self.inflight.insert(i);
            return Some((i, h, lq, thumb));
        }
        if self.thumbs_inflight.len() >= LQ_CONCURRENCY {
            return None;
        }
        while let Some((i, h)) = self.lq_tail.pop_front() {
            if self.lq_cancel.remove(&i) || self.thumbs_inflight.contains(&i) {
                continue;
            }
            self.thumbs_inflight.insert(i);
            return Some((i, h, true, true));
        }
        None
    }
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
                lq_tail: VecDeque::new(),
                lq_cancel: HashSet::new(),
                thumbs_inflight: HashSet::new(),
                workers,
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
                //
                // **Thumbs are never stale.** A whole-volume thumbnail is position-
                // *independent* work — it is not in `wanted` at all (the tail is a
                // separate queue), and only its scheduling order depends on where the
                // reader is, so abandoning it for a navigation would just throw away a
                // finished decode. (Phase 5's `paused` will be the one exception.)
                // This is also why the thumb slot is released only in `drop_inflight`:
                // a thumb can never leave the pipeline through this path.
                let stale = |index: usize, thumb: bool| -> bool {
                    if thumb {
                        return false;
                    }
                    let (m, _) = &*shared;
                    let mut st = lock_jobs(m);
                    if st.wanted.contains(&index) {
                        false
                    } else {
                        st.inflight.remove(&index);
                        true
                    }
                };
                // Release the job's in-flight marker: the `inflight` set for a window
                // decode, the thumb slot for a tail decode. **Every** post-claim exit
                // path of a thumb runs through here (completed, failed, panicked) —
                // a leaked slot would permanently halve tail throughput — and freeing
                // a slot notifies, because a worker may be parked precisely on
                // "tail work exists but no slot is free" (see `JobState::claim`).
                let drop_inflight = |index: usize, thumb: bool| {
                    let (m, cv) = &*shared;
                    let mut st = lock_jobs(m);
                    if thumb {
                        st.thumbs_inflight.remove(&index);
                        drop(st);
                        cv.notify_one();
                    } else {
                        st.inflight.remove(&index);
                    }
                };
                loop {
                    // Wait for and claim the highest-priority job (index + its
                    // exact, per-page decode target height), grabbing the current
                    // source under the same lock so a live `set_source` swap is
                    // picked up on the next claimed job. `claim` returning `None`
                    // *is* the wait condition — see its doc comment.
                    let (index, th, lq, thumb, source): (usize, u32, bool, bool, Arc<dyn PageSource>) = {
                        let (m, cv) = &*shared;
                        let mut st = lock_jobs(m);
                        loop {
                            if !st.running {
                                return;
                            }
                            if let Some((i, h, lq, thumb)) = st.claim() {
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
                                drop_inflight(index, thumb);
                                return Some(Msg::Failed { index, error: format!("read failed: {e}") });
                            }
                        };

                        // Bail before the expensive decode if the jump landed during the read.
                        if stale(index, thumb) {
                            return None;
                        }

                        let decoded = decode_page(&bytes, th, lq, &mut resizer);

                        // Bail before upload if the page left the window during the decode —
                        // skips the GPU upload, the texpool/cache churn, and a pointless `Done`.
                        if stale(index, thumb) {
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

                        drop_inflight(index, thumb);

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
                            // Idempotent: the panic may have struck *after* the
                            // normal release above, and both markers are sets.
                            drop_inflight(index, thumb);
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
        // One wakeup per queued job, not `notify_all`. A worker only ever blocks
        // after `claim` found nothing claimable, and each woken worker either claims
        // a job or goes straight back to waiting — so waking more workers than there
        // are jobs is pure wasted wakeups (this fires on every navigation, and the
        // pool is 8 threads deep). Workers already running don't need a wakeup: they
        // re-enter `claim` on their own when they finish.
        let wake = st.jobs.len().min(st.workers);
        drop(st);
        for _ in 0..wake {
            cv.notify_one();
        }
    }

    /// Replace the whole-volume thumbnail tail (`(index, target_h)`, nearest-first).
    /// Claimed only when the HQ window queue is empty, so this can never delay a page
    /// flip. Clears the pending cancellations with it — the caller rebuilt the tail
    /// from the current caches, so every entry in it is still wanted.
    pub fn set_lq_tail(&self, tail: Vec<(usize, u32)>) {
        let (m, cv) = &*self.shared;
        let mut st = lock_jobs(m);
        st.lq_tail = tail.into();
        st.lq_cancel.clear();
        // At most `LQ_CONCURRENCY` tail jobs can be claimed at once, so waking more
        // workers than that would be wasted even if the tail is huge.
        let wake = if st.lq_tail.is_empty() {
            0
        } else {
            LQ_CONCURRENCY.min(st.workers)
        };
        drop(st);
        for _ in 0..wake {
            cv.notify_one();
        }
    }

    /// Forget the queued thumbnail for `index` — called when the page lands a real
    /// full-res texture, which makes its preview pointless. Lazy by design: the entry
    /// is skipped and dropped when a worker reaches it (see `JobState::claim`), so a
    /// cancel costs a set insert instead of an O(volume) tail rebuild. No wakeup —
    /// this only ever *removes* claimable work.
    pub fn cancel_lq(&self, index: usize) {
        let (m, _) = &*self.shared;
        lock_jobs(m).lq_cancel.insert(index);
    }

    /// Queued thumbnail count. **Approximate:** entries cancelled since the tail was
    /// built are still counted until a worker walks past them, so this only ever
    /// over-counts. Fine for a progress readout; don't gate logic on it being zero.
    pub fn lq_tail_len(&self) -> usize {
        let (m, _) = &*self.shared;
        lock_jobs(m).lq_tail.len()
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
        // The background tier dies with the pool too, so a straggler that finishes
        // its current thumb can't go on to claim the whole rest of the volume.
        st.lq_tail.clear();
        st.lq_cancel.clear();
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

    /// A `PageSource` that is never read: the claim-order tests exercise
    /// `JobState` alone (a real `DecodePool` needs a wgpu `Device`, which a unit
    /// test can't assume), and `claim` only ever clones this handle.
    struct NoSource;
    impl PageSource for NoSource {
        fn len(&self) -> usize {
            0
        }
        fn name(&self, _index: usize) -> &str {
            ""
        }
        fn read_page(&self, _index: usize) -> std::io::Result<Arc<Vec<u8>>> {
            Err(std::io::Error::other("test stub"))
        }
    }

    fn job_state() -> JobState {
        JobState {
            jobs: VecDeque::new(),
            lq_tail: VecDeque::new(),
            lq_cancel: HashSet::new(),
            thumbs_inflight: HashSet::new(),
            workers: 8,
            source: Arc::new(NoSource),
            inflight: HashSet::new(),
            wanted: HashSet::new(),
            waker: None,
            running: true,
        }
    }

    /// The HQ window preempts the background tail: while any window job is queued,
    /// no worker touches a thumbnail — that is what keeps a page flip from ever
    /// queueing behind a whole-volume fill. Only once `jobs` drains do tail entries
    /// get claimed, and they come out `(lq, thumb) = (true, true)`.
    #[test]
    fn window_jobs_are_claimed_before_the_tail() {
        let mut st = job_state();
        st.jobs = VecDeque::from(vec![(5, 900, false, false), (6, 900, false, false)]);
        st.lq_tail = VecDeque::from(vec![(40, 540), (41, 540)]);

        assert_eq!(st.claim(), Some((5, 900, false, false)));
        assert_eq!(st.claim(), Some((6, 900, false, false)));
        assert!(st.lq_tail.len() == 2, "tail untouched while the window had work");
        // Window empty → the tail is finally eligible, as a thumb job.
        assert_eq!(st.claim(), Some((40, 540, true, true)));
        // A window job arriving mid-fill jumps ahead of the rest of the tail again.
        st.jobs.push_back((7, 900, false, false));
        assert_eq!(st.claim(), Some((7, 900, false, false)));
    }

    /// `cancel_lq` (a page landed a real texture) must stop the queued thumbnail
    /// from ever being claimed — without a tail rebuild. The entry is dropped as the
    /// worker walks past it, and the cancel set doesn't grow unboundedly.
    #[test]
    fn cancelled_tail_entries_are_never_claimed() {
        let mut st = job_state();
        st.lq_tail = VecDeque::from(vec![(40, 540), (41, 540), (42, 540)]);
        st.lq_cancel.insert(41);

        assert_eq!(st.claim(), Some((40, 540, true, true)));
        // 41 is skipped *and* consumed, so the next claim is 42, not 41.
        assert_eq!(st.claim(), Some((42, 540, true, true)));
        assert!(st.lq_tail.is_empty());
        assert!(st.lq_cancel.is_empty(), "a consumed cancel is dropped with its entry");

        // Every entry cancelled → nothing claimable (the worker waits), and the
        // tail is drained rather than re-walked on every claim.
        let mut st = job_state();
        st.lq_tail = VecDeque::from(vec![(7, 540), (8, 540)]);
        st.lq_cancel.extend([7, 8]);
        assert_eq!(st.claim(), None);
        assert!(st.lq_tail.is_empty());
    }

    /// The tail's concurrency half: at most `LQ_CONCURRENCY` thumbnails decode at
    /// once no matter how many workers are idle, and a released slot is what makes
    /// the next one claimable. (The release side lives in the worker's
    /// `drop_inflight`, which notifies — without that notify this exact state is a
    /// deadlock: tail work pending, every worker parked.)
    #[test]
    fn thumb_concurrency_is_capped_and_recovers_on_release() {
        let mut st = job_state();
        st.lq_tail = (0..10).map(|i| (i, 540)).collect();

        let mut claimed = Vec::new();
        for _ in 0..8 {
            // Eight idle workers all try to claim.
            if let Some((i, _, _, thumb)) = st.claim() {
                assert!(thumb);
                claimed.push(i);
            }
        }
        assert_eq!(claimed, vec![0, 1], "at most {LQ_CONCURRENCY} thumbs in flight");
        assert_eq!(st.claim(), None, "slots full → the worker must wait");
        assert_eq!(st.lq_tail.len(), 8, "a blocked claim must not consume the tail");

        // A thumb finishes (worker: `drop_inflight` → remove + notify_one).
        st.thumbs_inflight.remove(&0);
        assert_eq!(st.claim(), Some((2, 540, true, true)));
        assert_eq!(st.claim(), None);

        // A window job is *not* subject to the thumb cap — the HQ path never waits
        // on the background tier.
        st.jobs.push_back((99, 1200, false, false));
        assert_eq!(st.claim(), Some((99, 1200, false, false)));
    }

    /// Thumbs stay out of `inflight`, which is the set `set_jobs` filters the HQ
    /// window against. If a thumb registered there, jumping to a page whose
    /// thumbnail happened to be decoding would suppress its real full-res decode —
    /// and with the LQ epoch gone from `JobsKey` nothing would ever re-queue it, so
    /// the page would sit blurry until the next navigation.
    #[test]
    fn thumbs_do_not_block_the_hq_decode_of_the_same_page() {
        let mut st = job_state();
        st.lq_tail = VecDeque::from(vec![(40, 540)]);
        assert_eq!(st.claim(), Some((40, 540, true, true)));
        assert!(!st.inflight.contains(&40), "a thumb must not claim the inflight slot");
        assert!(st.thumbs_inflight.contains(&40));

        // The reader jumps to 40: the window job survives `set_jobs`' inflight
        // filter and is claimed normally.
        assert!(!st.inflight.contains(&40));
        st.jobs.push_back((40, 1600, false, false));
        assert_eq!(st.claim(), Some((40, 1600, false, false)));
        assert!(st.inflight.contains(&40));

        // …and the thumb finishing must not clear the HQ decode's marker.
        st.thumbs_inflight.remove(&40);
        assert!(st.inflight.contains(&40), "thumb release must not free the HQ marker");
    }

    /// A rebuilt tail can list a page whose thumb is still decoding (it isn't in
    /// `lq_cache` yet); claiming it twice would run the same decode twice and
    /// under-count the concurrency slots.
    #[test]
    fn a_page_is_never_thumbed_twice_concurrently() {
        let mut st = job_state();
        st.lq_tail = VecDeque::from(vec![(40, 540), (40, 540), (41, 540)]);
        assert_eq!(st.claim(), Some((40, 540, true, true)));
        assert_eq!(st.claim(), Some((41, 540, true, true)), "duplicate 40 skipped");
        assert_eq!(st.thumbs_inflight.len(), 2);
    }

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

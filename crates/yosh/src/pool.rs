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

use crate::decode::decode_and_downscale;
use crate::page::{PagePipeline, PageTexture};
use crate::source::PageSource;
use crate::texpool::TexturePool;

pub enum Msg {
    Done { index: usize, page: PageTexture },
    Failed { index: usize },
}

struct JobState {
    jobs: Vec<usize>,
    inflight: HashSet<usize>,
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
        target_h: u32,
        workers: usize,
    ) -> Self {
        let shared = Arc::new((
            Mutex::new(JobState {
                jobs: Vec::new(),
                inflight: HashSet::new(),
                running: true,
            }),
            Condvar::new(),
        ));
        let (tx, rx): (Sender<Msg>, Receiver<Msg>) = channel();
        let mut handles = Vec::new();

        for _ in 0..workers {
            let shared = shared.clone();
            let source = source.clone();
            let device = device.clone();
            let queue = queue.clone();
            let tex_pool = tex_pool.clone();
            let tx = tx.clone();
            handles.push(std::thread::spawn(move || {
                let mut resizer = Resizer::new();
                loop {
                    // Wait for and claim the highest-priority job.
                    let index: usize = {
                        let (m, cv) = &*shared;
                        let mut st = m.lock().unwrap();
                        loop {
                            if !st.running {
                                return;
                            }
                            if !st.jobs.is_empty() {
                                let i = st.jobs.remove(0);
                                st.inflight.insert(i);
                                break i;
                            }
                            st = cv.wait(st).unwrap();
                        }
                    };

                    let result = source
                        .read_page(index)
                        .map_err(|e| e.to_string())
                        .and_then(|bytes| decode_and_downscale(&bytes, target_h, &mut resizer));

                    {
                        let (m, _) = &*shared;
                        m.lock().unwrap().inflight.remove(&index);
                    }

                    let msg = match result {
                        Ok(img) => Msg::Done {
                            index,
                            page: PagePipeline::upload(&device, &queue, &img, &tex_pool),
                        },
                        Err(_) => Msg::Failed { index },
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

    /// Replace the work list with `desired` (nearest-first), skipping pages
    /// already in flight. Wakes idle workers.
    pub fn set_jobs(&self, desired: Vec<usize>) {
        let (m, cv) = &*self.shared;
        let mut st = m.lock().unwrap();
        let filtered: Vec<usize> = desired
            .into_iter()
            .filter(|i| !st.inflight.contains(i))
            .collect();
        st.jobs = filtered;
        drop(st);
        cv.notify_all();
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

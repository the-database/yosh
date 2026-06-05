//! Decode pool: N worker threads that read → decode → downscale → upload a page
//! to a GPU texture, off the main thread. wgpu `Device`/`Queue` are `Send+Sync`,
//! so workers create textures and `write_texture` themselves; the main thread
//! only swaps in finished textures.
//!
//! The job list is rebuilt by the scheduler each navigation (nearest-first), so
//! workers always pick the highest-priority page relative to the latest position.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use fast_image_resize::Resizer;

use crate::decode::{decode_full, decode_page, DecodedPage};
use crate::downscale::Downscaler;
use crate::page::{PagePipeline, PageTexture};
use crate::source::PageSource;
use crate::texpool::{self, TexturePool};

/// GPU-side downscale is disabled: a single bilinear blit can't match the HQ
/// CPU resize (Lanczos / Catmull-Rom + dot-gain). The path below is kept intact
/// behind this flag for a future high-quality GPU rewrite. While `false`, every
/// page goes through the HQ CPU path.
const GPU_DOWNSCALE_ENABLED: bool = false;

pub enum Msg {
    Done { index: usize, page: PageTexture },
    Failed { index: usize, error: String },
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
        downscaler: Arc<Downscaler>,
        gpu_flag: Arc<AtomicBool>,
        target_h: Arc<AtomicU32>,
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
            let downscaler = downscaler.clone();
            let gpu_flag = gpu_flag.clone();
            let target_h = target_h.clone();
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

                    let gpu = gpu_flag.load(Ordering::Relaxed);
                    let th = target_h.load(Ordering::Relaxed);
                    let page: Result<PageTexture, String> = match source.read_page(index) {
                        Ok(bytes) if gpu && GPU_DOWNSCALE_ENABLED => decode_full(&bytes).map(|img| {
                            // Upload full-res, downscale on the GPU into a display texture.
                            let src = tex_pool.get(&device, img.gray, img.w, img.h);
                            texpool::write_pixels(&queue, &src, &img.pixels, img.w, img.h, img.gray);
                            let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
                            let tw = (((img.w as f64) * (th as f64) / (img.h as f64)).round()
                                as u32)
                                .max(1);
                            let dst = tex_pool.get(&device, img.gray, tw, th);
                            let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
                            downscaler.blit(&device, &queue, &src_view, &dst_view, img.gray);
                            drop(src_view);
                            tex_pool.put(src, img.gray, img.w, img.h);
                            PageTexture::from_pooled(dst, tw, th, img.src_w, img.src_h, img.gray, th)
                        }),
                        Ok(bytes) => match decode_page(&bytes, th, &mut resizer) {
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
                        },
                        Err(e) => Err(format!("read failed: {e}")),
                    };

                    {
                        let (m, _) = &*shared;
                        m.lock().unwrap().inflight.remove(&index);
                    }

                    let msg = match page {
                        Ok(page) => Msg::Done { index, page },
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

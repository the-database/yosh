//! End-to-end page throughput through the real engine: a `PageSource` feeding the
//! real `DecodePool` (read → decode → HQ downscale → `write_texture` upload on a real
//! GPU device), with the desktop's High-tier budget. Unlike `crates/decode_bench`
//! (a standalone Phase-0 copy of the decode loop), this measures what the app runs.
//!
//! ```sh
//! cargo run --release -p yosh-engine --example seek_bench -- <folder|.cbz|.zip> [target_h] [runs]
//! ```
//!
//! Two modes, each after one warm-up pass (so the OS file cache is hot):
//! - **bulk**: every page queued at once — the pool's raw production rate.
//! - **seek**: holding "next page" — the reader's own prefetch window around a
//!   cursor that advances the instant the current page lands. This is the
//!   page-turn number the ~83 pages/sec target is about.
//!
//! GPU uploads are queued, not awaited (the app never waits on them either).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use yosh_engine::gpu::GpuContext;
use yosh_engine::pool::{DecodePool, Msg};
use yosh_engine::prefetch::desired_window;
use yosh_engine::reader::{Budget, DeviceTier};
use yosh_engine::source::{FolderSource, PageSource, ZipSource};
use yosh_engine::texpool::TexturePool;

fn open(path: &Path) -> Arc<dyn PageSource> {
    if path.is_dir() {
        Arc::new(FolderSource::new(path).expect("open folder"))
    } else {
        Arc::new(ZipSource::new(path).expect("open zip"))
    }
}

/// Queue every page; time until all have landed.
fn bulk(pool: &DecodePool, n: usize, target_h: u32) -> f64 {
    let t0 = Instant::now();
    pool.set_jobs((0..n).map(|i| (i, target_h, false, false)).collect());
    let mut done = 0;
    while done < n {
        for m in pool.poll() {
            match m {
                Msg::Done { thumb: false, .. } | Msg::Failed { .. } => done += 1,
                _ => {}
            }
        }
        std::thread::yield_now();
    }
    n as f64 / t0.elapsed().as_secs_f64()
}

/// Hold "next page": re-target the prefetch window on every advance, exactly as
/// `Reader::prefetch` does, and step the moment the current page is decoded.
fn seek(pool: &DecodePool, n: usize, target_h: u32, fwd: usize, back: usize) -> f64 {
    let mut have = vec![false; n];
    let mut cur = 0;
    let t0 = Instant::now();
    let jobs = |cur: usize, have: &[bool]| -> Vec<(usize, u32, bool, bool)> {
        desired_window(cur, n, fwd, back)
            .into_iter()
            .filter(|&i| !have[i])
            .map(|i| (i, target_h, false, false))
            .collect()
    };
    pool.set_jobs(jobs(cur, &have));
    while cur < n {
        for m in pool.poll() {
            match m {
                Msg::Done { index, thumb: false, .. } | Msg::Failed { index, .. } => have[index] = true,
                _ => {}
            }
        }
        if have[cur] {
            while cur < n && have[cur] {
                cur += 1;
            }
            if cur < n {
                pool.set_jobs(jobs(cur, &have));
            }
        } else {
            std::thread::yield_now();
        }
    }
    n as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("usage: seek_bench <folder|.cbz> [target_h] [runs]"));
    let target_h: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1440);
    let runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let gpu = GpuContext::create(GpuContext::create_instance(), None, wgpu::PowerPreference::HighPerformance);
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let budget = Budget::for_tier(DeviceTier::High, 8192, cpus);
    let src = open(path);
    let n = src.len();
    println!(
        "{} pages, target_h {target_h}, {} workers, fwd {} back {}, gpu {} ({:?})",
        n, budget.workers, budget.fwd, budget.back, gpu.adapter_info.name, gpu.adapter_info.backend
    );

    // A fresh pool + texture pool per pass, as a volume open gets.
    let fresh = || {
        DecodePool::new(
            src.clone(),
            gpu.device.clone(),
            gpu.queue.clone(),
            Arc::new(TexturePool::with_max_total(budget.texpool_max)),
            budget.workers,
        )
    };
    bulk(&fresh(), n, target_h); // warm-up: OS file cache, shader/driver init
    let mut b: Vec<f64> = (0..runs).map(|_| bulk(&fresh(), n, target_h)).collect();
    let mut s: Vec<f64> = (0..runs).map(|_| seek(&fresh(), n, target_h, budget.fwd, budget.back)).collect();
    b.sort_by(f64::total_cmp);
    s.sort_by(f64::total_cmp);
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" ");
    println!("bulk pps: median {:.0}  [{}]", b[b.len() / 2], fmt(&b));
    println!("seek pps: median {:.0}  [{}]", s[s.len() / 2], fmt(&s));
}

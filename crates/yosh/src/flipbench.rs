//! Frame-timing harness — compiled only with `--features flipbench`, so the shipped
//! binary carries none of it.
//!
//! `yosh --flipbench <report> <book>` flips forward the moment each page is drawn
//! (holding "next" at full speed) and exits at the last page; `yosh --flipbench
//! <report>` with no book redraws the library for [`LIBRARY_FRAMES`] frames. Either
//! way it writes one report of per-frame phase timings, which
//! `scripts/flipbench.ps1` runs and compares across two commits. A run is always
//! portable (see `config::is_portable_dir`): it must neither read the user's
//! settings — runs have to be reproducible — nor write them.
//!
//! Phases, as `render()` stamps them:
//! - `pre`: reader work and quad building, up to the swapchain acquire.
//! - `acquire`: waiting for the swapchain image — where vsync blocks.
//! - `ui`: the egui pass plus tessellation.
//! - `post`: egui texture/buffer upload, encode, submit and present.
//!
//! plus the interval between presents and, in reader mode, displayed pages/sec.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Library-mode run length.
pub const LIBRARY_FRAMES: usize = 600;

struct Config {
    out: PathBuf,
    /// A book was given: flip through it (else: redraw the library).
    reader: bool,
}
static CONFIG: OnceLock<Config> = OnceLock::new();

/// Strip `--flipbench <report>` from the CLI args, so the positional
/// `<path> [start_page]` parse after it is unchanged, and arm the harness.
pub fn init(args: &mut Vec<String>) {
    let Some(i) = args.iter().position(|a| a == "--flipbench") else {
        return;
    };
    args.remove(i);
    let out = if i < args.len() { args.remove(i) } else { "flipbench.txt".to_string() };
    let reader = args.get(1).is_some();
    let _ = CONFIG.set(Config { out: PathBuf::from(out), reader });
}

pub fn active() -> bool {
    CONFIG.get().is_some()
}

/// Reader mode (a book was given) vs library mode.
pub fn reader_mode() -> bool {
    CONFIG.get().is_some_and(|c| c.reader)
}

/// A point in `render()` that [`Frame::mark`] stamps.
#[derive(Clone, Copy)]
pub enum Mark {
    AcquireStart = 1,
    AcquireEnd,
    UiStart,
    UiEnd,
}

/// One frame's phase stamps: index 0 is the frame start, the rest are [`Mark`]s.
pub struct Frame([Instant; 5]);

impl Frame {
    pub fn start() -> Self {
        Self([Instant::now(); 5])
    }
    pub fn mark(&mut self, m: Mark) {
        self.0[m as usize] = Instant::now();
    }
}

#[derive(Default)]
struct Stats {
    rows: Vec<[f64; 5]>,
    intervals: Vec<f64>,
    last: Option<Instant>,
    first_step: Option<Instant>,
    steps: usize,
}
static STATS: Mutex<Option<Stats>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Stats::default))
}

/// Record a presented frame. In reader mode only frames after the first flip count,
/// so the volume open isn't mixed into the page-turn numbers.
pub fn record(frame: Frame) {
    let end = Instant::now();
    let [t0, a0, a1, u0, u1] = frame.0;
    let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f64() * 1e3;
    with(|s| {
        if reader_mode() && s.first_step.is_none() {
            return;
        }
        if let Some(l) = s.last {
            s.intervals.push(ms(l, end));
        }
        s.last = Some(end);
        s.rows.push([ms(t0, a0), ms(a0, a1), ms(u0, u1), ms(u1, end), ms(t0, end)]);
    });
}

/// A page flip happened (reader mode).
pub fn stepped() {
    with(|s| {
        s.first_step.get_or_insert_with(Instant::now);
        s.steps += 1;
    });
}

/// Frames recorded so far.
pub fn frames() -> usize {
    with(|s| s.rows.len())
}

/// Write the report and exit. The format is one `key value…` line per stat, which
/// `scripts/flipbench.ps1` parses.
pub fn finish() -> ! {
    let report = with(|s| {
        let col = |i: usize| s.rows.iter().map(|r| r[i]).collect::<Vec<_>>();
        let mode = if reader_mode() { "reader" } else { "library" };
        let mut r = format!("mode {mode}\nframes {}\nsteps {}\n", s.rows.len(), s.steps);
        if let Some(t) = s.first_step {
            r += &format!("pages_per_sec {:.1}\n", s.steps as f64 / t.elapsed().as_secs_f64());
        }
        for (i, name) in ["pre", "acquire", "ui", "post", "total"].iter().enumerate() {
            r += &format!("{name}_ms {}\n", quantiles(col(i)));
        }
        r += &format!("interval_ms {}\n", quantiles(s.intervals.clone()));
        r
    });
    if let Some(c) = CONFIG.get() {
        let _ = std::fs::write(&c.out, report);
    }
    std::process::exit(0);
}

/// `median p95 p99 max`, in ms.
fn quantiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "0 0 0 0".to_string();
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("{:.3} {:.3} {:.3} {:.3}", q(0.5), q(0.95), q(0.99), v[v.len() - 1])
}

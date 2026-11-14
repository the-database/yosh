//! Tile shapes fitted to a page size, for libaji's still-image ABI.
//!
//! libaji runs a model on fixed-size tiles from one static engine per tile shape, the last tile on
//! each axis pushed flush against the far edge (`tile_origins` in its `core/tiling.cpp`). A square
//! tile that doesn't suit a page's size re-infers much of it: 512 on a 1200x1800 page runs 1.82x the
//! page's pixels. A shape fitted to the page (1216x992 there: two tiles, 1.1x) runs it in about half
//! the time (2x MangaJaNai ESRGAN, RTX 5090, TensorRT-RTX: 122 ms against 242 ms).
//!
//! On TensorRT an even-tiles engine (`AJI_IMG_LOAD_EVEN_TILES`) does as well without fitting:
//! it splits any page into the fewest even tiles of at most its size, at no loss per pixel there
//! (TensorRT-RTX loses 30-50%). With tiles of at most 1024 (2x MangaJaNai ESRGAN, RTX 5090) a
//! 1200x1800 page took 165 ms against 169 fitted and 253 on 512, and ten pages of mixed sizes 36%
//! less time than on 512; larger tiles were no faster. [`even_tile`] picks its size.
//!
//! Tiles are sized to the GPU's free memory ([`MemoryUse`], from libaji's `aji_img_memory_get`):
//! a model's activations grow with its tile's area (TensorRT-RTX: ~1.1 KB per input pixel for 2x
//! ESRGAN, ~4.5 KB for 4x; DirectML about three times that), so one loaded engine predicts
//! another's.
//!
//! This module is the arithmetic ([`fit`], [`even_tile`]) and records of the engines built so far
//! and of each model's memory ([`fitted_known`], [`record_fitted`], [`known_memory`],
//! [`record_memory`]), shared by the apps on libaji (MangaJaNaiCLI and yosh's upscaler) so what
//! one learned the other uses. When to build an engine is the app's call: a build costs seconds
//! (TensorRT-RTX) to minutes (TensorRT).

use std::path::Path;

#[cfg(feature = "aji")]
use crate::aji::Memory;

/// A tile shape, input pixels (width, height).
pub type Shape = (u32, u32);

/// libaji's default overlap between tiles, in input pixels.
pub const DEFAULT_OVERLAP: u32 = 128;

/// Fitted sides are rounded up to this, so pages a few pixels apart share an engine.
const SNAP: u32 = 32;

/// A tile's fixed cost (launch, sync, extract and blend), in pixels of model work. Measured on an
/// RTX 5090 with TensorRT-RTX: about 4 ms per tile against 53 ms per million pixels for 2x
/// MangaJaNai ESRGAN, under 2 ms against 213 for 4x.
const TILE_COST: u64 = 50_000;

/// A fitted shape must save at least this share of the square's work to be worth an engine.
const MIN_GAIN: f64 = 0.15;

/// `1216x992`, or `512` for a square.
pub fn label((w, h): Shape) -> String {
    if w == h { w.to_string() } else { format!("{w}x{h}") }
}

/// How many tiles libaji places along an axis of `len` pixels with tiles of `tile` and an overlap
/// of `overlap` (its `tile_origins`: step `tile - overlap`, the last tile flush with the far edge,
/// a repeated last position dropped; one reflect-padded tile if the axis fits).
pub fn tiles_along(len: u32, tile: u32, overlap: u32) -> u64 {
    if len <= tile {
        return 1;
    }
    let step = tile.saturating_sub(overlap).max(1);
    let n = (len - overlap.min(len)).div_ceil(step).max(1);
    let mut count = 0;
    let mut last = None;
    for i in 0..n {
        let p = (i * step).min(len - tile);
        if last != Some(p) {
            count += 1;
            last = Some(p);
        }
    }
    count
}

/// The work of upscaling a `w`x`h` page with tiles of `shape`, in pixels: tiles x (tile area +
/// [`TILE_COST`]). libaji caps the overlap at half the shorter side.
pub fn cost(w: u32, h: u32, shape: Shape, overlap: u32) -> u64 {
    let o = overlap.min(shape.0.min(shape.1) / 2);
    let n = tiles_along(w, shape.0, o) * tiles_along(h, shape.1, o);
    n * (shape.0 as u64 * shape.1 as u64 + TILE_COST)
}

/// The largest fitted tile: a larger one isn't faster (TensorRT-RTX, 2x ESRGAN: a whole
/// 1200x1800 page in one tile took 119 ms against 117 in two of 1200x964), only larger in memory.
pub const MAX_AREA: u64 = 1_500_000;

/// The largest tile area to fit pages with by the GPU's total memory, for a libaji too old to
/// report memory ([`MemoryUse`] instead when it can). A fitted engine's memory grows with its
/// area: 1216x992 took about 2 GB with 2x MangaJaNai ESRGAN and 6 GB with 4x.
pub fn area_budget(vram_bytes: u64) -> u64 {
    const GIB: u64 = 1 << 30;
    match vram_bytes {
        v if v >= 20 * GIB => 1_500_000,
        v if v >= 11 * GIB => 1_000_000,
        _ => 600_000,
    }
}

/// The tile shape that upscales a `w`x`h` page with the least work, its area at most [`MAX_AREA`],
/// its sides multiples of [`SNAP`] and `allowed` (it fits in memory), if it saves at least
/// [`MIN_GAIN`] of the work of tiles of `square`; with that share saved.
pub fn fit(w: u32, h: u32, overlap: u32, allowed: impl Fn(Shape) -> bool, square: u32) -> Option<(Shape, f64)> {
    let snap = |v: u32| v.div_ceil(SNAP) * SNAP;
    let min_side = (2 * overlap + 64).max(128);
    // The tile sides that split an axis into n tiles, for n = 1, 2, ... while they stay useful.
    let sides = |len: u32| -> Vec<u32> {
        let mut v: Vec<u32> = (1..=32u32)
            .map(|n| snap((len + (n - 1) * overlap).div_ceil(n)).min(4096))
            .take_while(|&s| s >= min_side)
            .collect();
        v.dedup();
        v
    };
    let base = cost(w, h, (square, square), overlap);
    let mut best: Option<(Shape, u64)> = None;
    for tw in sides(w) {
        for th in sides(h) {
            if tw as u64 * th as u64 > MAX_AREA || !allowed((tw, th)) {
                continue;
            }
            let c = cost(w, h, (tw, th), overlap);
            if best.is_none_or(|(_, b)| c < b) {
                best = Some(((tw, th), c));
            }
        }
    }
    let (shape, c) = best?;
    let gain = 1.0 - c as f64 / base as f64;
    (gain >= MIN_GAIN && shape != (square, square)).then_some((shape, gain))
}

/// The tile sides an even-tiles engine is built for, largest first: one engine per model takes
/// pages of any size, and a few steps keep a GPU's free memory, which varies, from asking for a
/// new build every run.
pub const EVEN_TILES: [u32; 3] = [1024, 768, 512];

/// How a model's GPU memory grows with its tile: measured from a loaded engine
/// ([`MemoryUse::measured`]), recorded from an earlier run ([`known_memory`]), or guessed
/// ([`MemoryUse::guess`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryUse {
    pub weights: u64,
    /// Activation bytes per tile input pixel.
    pub activations_per_px: f64,
    /// Tile input/output buffer bytes per tile input pixel.
    pub buffers_per_px: f64,
    /// Activations live in the context's buffer, shared by its models (the CUDA backends).
    pub shared: bool,
    /// The model's scale, for libaji's GPU tiling scratch.
    pub scale: u32,
}

impl MemoryUse {
    /// From libaji's report on a ready model with tiles of `tile` (with even tiles, its largest)
    /// and `scale`; `None` if it reported none.
    #[cfg(feature = "aji")]
    pub fn measured(mem: &Memory, tile: Shape, scale: u32) -> Option<Self> {
        let area = tile.0 as f64 * tile.1 as f64;
        (mem.model_activations > 0 && area > 0.0).then(|| Self {
            weights: mem.model_weights,
            activations_per_px: mem.model_activations as f64 / area,
            buffers_per_px: mem.model_buffers as f64 / area,
            shared: mem.activations_shared,
            scale,
        })
    }

    /// For a model nothing is known about yet: more than any MangaJaNai or IllustrationJaNai model
    /// measured on TensorRT (4x ESRGAN, ~4.5 KB per pixel), so a first build fits.
    pub fn guess() -> Self {
        Self { weights: 200 << 20, activations_per_px: 6000.0, buffers_per_px: 200.0, shared: true, scale: 4 }
    }

    /// The bytes loading an engine with tiles of `tile` would add to a context holding
    /// `ctx_activations` and `ctx_scratch` (libaji's `Memory`), for pages `page` (w, h). On the
    /// CUDA backends the activations and libaji's GPU tiling scratch are
    /// shared by the context's models, so only their growth counts.
    pub fn added(&self, tile: Shape, page: Shape, ctx_activations: u64, ctx_scratch: u64) -> u64 {
        let area = tile.0 as f64 * tile.1 as f64;
        let act = self.activations_per_px * area;
        let (act, scratch) = if self.shared {
            ((act - ctx_activations as f64).max(0.0), scratch(page, tile.1, self.scale).saturating_sub(ctx_scratch) as f64)
        } else {
            (act, 0.0)
        };
        (self.weights as f64 + self.buffers_per_px * area + act + scratch) as u64
    }
}

/// libaji's GPU tiling scratch for a `page` (w, h), taken as 3 channels, with tiles `tile_h` high
/// (aji_image.h: about W*H*C + 5*C*(tile_h*scale)*(W*scale)).
fn scratch((w, h): Shape, tile_h: u32, scale: u32) -> u64 {
    let (w, h, s) = (w as u64, h as u64, scale.max(1) as u64);
    let th = (tile_h as u64).min(h.max(1));
    3 * w * h + 15 * (th * s) * (w * s)
}

/// Whether `added` more bytes fit in `free` (counting every process's use) of a GPU of `total`,
/// keeping back 768 MB or a sixteenth of the GPU (whichever is more) for the desktop and other apps
/// to grow into, and a tenth on top of the estimate.
pub fn fits_memory(added: u64, free: u64, total: u64) -> bool {
    let room = free.saturating_sub((768u64 << 20).max(total / 16));
    added.saturating_add(added / 10) <= room
}

/// The GPU's free memory and a model's memory use at one moment, to check engines against.
#[cfg(feature = "aji")]
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// libaji's report; `None` from a libaji before 0.28.0, which can't say: then `area_budget`
    /// ([`area_budget`], by the GPU's size) instead.
    pub mem: Option<Memory>,
    /// Memory of engines decided on but not loaded yet (building in the background, say).
    pub reserved: u64,
    pub area_budget: u64,
    pub use_: MemoryUse,
}

#[cfg(feature = "aji")]
impl Budget {
    /// Whether an engine with tiles of `tile` fits, for pages of `page` (w, h); with the bytes
    /// it's predicted to add (0 when libaji can't say).
    pub fn fits(&self, tile: Shape, page: Shape) -> (bool, u64) {
        let Some(mem) = &self.mem else {
            return (tile.0 as u64 * tile.1 as u64 <= self.area_budget, 0);
        };
        let added = self.use_.added(tile, page, mem.ctx_activations, mem.ctx_scratch);
        (fits_memory(added, mem.device_free.saturating_sub(self.reserved), mem.device_total), added)
    }
}

/// The largest of [`EVEN_TILES`] whose engine `fits` (see [`MemoryUse::added`] and
/// [`fits_memory`]); the smallest if none does.
pub fn even_tile(fits: impl Fn(u32) -> bool) -> u32 {
    EVEN_TILES.into_iter().find(|&t| fits(t)).unwrap_or(EVEN_TILES[EVEN_TILES.len() - 1])
}

/// The record of each model's memory use, in an engine cache folder: one
/// `model<TAB>backend<TAB>gpu<TAB>weights<TAB>activations/px<TAB>buffers/px<TAB>shared<TAB>scale`
/// line each,
/// so a TensorRT run can size its even-tiles engine before building it.
pub const MEMORY_LIST: &str = "aji-model-memory.txt";

/// `model`'s recorded memory use on `backend` and `gpu` (see [`MEMORY_LIST`]).
pub fn known_memory(cache_dir: &Path, model: &str, backend: &str, gpu: &str) -> Option<MemoryUse> {
    let key = format!("{model}\t{backend}\t{gpu}\t");
    let text = std::fs::read_to_string(cache_dir.join(MEMORY_LIST)).ok()?;
    let line = text.lines().rev().find(|l| l.starts_with(&key))?;
    let mut f = line[key.len()..].split('\t');
    Some(MemoryUse {
        weights: f.next()?.parse().ok()?,
        activations_per_px: f.next()?.parse().ok()?,
        buffers_per_px: f.next()?.parse().ok()?,
        shared: f.next()? == "1",
        scale: f.next()?.parse().ok()?,
    })
}

/// Record `model`'s memory use (see [`known_memory`]), replacing an older line. Best effort.
pub fn record_memory(cache_dir: &Path, model: &str, backend: &str, gpu: &str, m: &MemoryUse) {
    let key = format!("{model}\t{backend}\t{gpu}\t");
    let path = cache_dir.join(MEMORY_LIST);
    let old = std::fs::read_to_string(&path).unwrap_or_default();
    let mut text: String = old.lines().filter(|l| !l.starts_with(&key)).map(|l| format!("{l}\n")).collect();
    text.push_str(&format!(
        "{key}{}\t{:.1}\t{:.1}\t{}\t{}\n",
        m.weights,
        m.activations_per_px,
        m.buffers_per_px,
        if m.shared { 1 } else { 0 },
        m.scale
    ));
    let _ = std::fs::write(&path, text);
}

/// The record of fitted engines built so far, in an engine cache folder: one
/// `model<TAB>backend<TAB>gpu<TAB>shape` line each (`shape` as [`label`] writes it). An app that
/// finds its page size there loads the engine at once instead of first weighing a build. A line
/// whose engine is gone (libaji cleans up engines of an older TensorRT) only means that load
/// builds it again.
pub const FITTED_LIST: &str = "aji-fitted-engines.txt";

fn fitted_line(model: &str, backend: &str, gpu: &str, shape: Shape) -> String {
    format!("{model}\t{backend}\t{gpu}\t{}", label(shape))
}

/// Whether `model`'s (a file name) engine fitted to `shape` was built on `backend` (libaji's label,
/// like `TensorRT-RTX`) and `gpu` (the device name) with engines cached in `cache_dir`.
pub fn fitted_known(cache_dir: &Path, model: &str, backend: &str, gpu: &str, shape: Shape) -> bool {
    let line = fitted_line(model, backend, gpu, shape);
    std::fs::read_to_string(cache_dir.join(FITTED_LIST)).is_ok_and(|s| s.lines().any(|l| l == line))
}

/// Note that `model`'s engine fitted to `shape` is built (see [`fitted_known`]). Best effort: a
/// record that can't be written only costs a later run its head start.
pub fn record_fitted(cache_dir: &Path, model: &str, backend: &str, gpu: &str, shape: Shape) {
    use std::io::Write;
    if fitted_known(cache_dir, model, backend, gpu, shape) {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(cache_dir.join(FITTED_LIST)) {
        let _ = writeln!(f, "{}", fitted_line(model, backend, gpu, shape));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_counts_match_libaji() {
        // 1200x1800 at 512/128 is 3 x 5 (measured: 15 tiles)
        assert_eq!(tiles_along(1200, 512, 128), 3);
        assert_eq!(tiles_along(1800, 512, 128), 5);
        assert_eq!(tiles_along(1200, 1024, 128), 2);
        assert_eq!(tiles_along(1800, 964, 128), 2);
        assert_eq!(tiles_along(1800, 688, 128), 3);
        assert_eq!(tiles_along(1200, 1200, 128), 1);
        assert_eq!(tiles_along(300, 512, 128), 1);
    }

    fn under(max_area: u64) -> impl Fn(Shape) -> bool {
        move |(w, h)| w as u64 * h as u64 <= max_area
    }

    #[test]
    fn a_tall_manga_page_gets_a_fitted_shape() {
        // room for 1216x992 (two tiles); less: 1216x704 (three)
        let (shape, gain) = fit(1200, 1800, 128, under(u64::MAX), 512).unwrap();
        assert_eq!(shape, (1216, 992));
        assert!(gain > 0.45, "{gain}");
        assert_eq!(fit(1200, 1800, 128, under(area_budget(12 << 30)), 512).unwrap().0, (1216, 704));
        let (shape, _) = fit(1200, 1800, 128, under(area_budget(8 << 30)), 512).unwrap();
        assert!(shape.0 as u64 * shape.1 as u64 <= 600_000, "{shape:?}");
    }

    #[test]
    fn a_page_the_square_suits_gets_none() {
        // one tile of the square itself
        assert_eq!(fit(512, 512, 128, under(1_000_000), 512), None);
        // 896x896 is exactly 2 x 2 squares at 512/128
        assert!(fit(896, 896, 128, under(1_000_000), 512).is_none_or(|(_, g)| g < 0.5));
    }

    #[test]
    fn memory_predicts_a_larger_tile_from_a_smaller_one() {
        // TensorRT-RTX, 4x ESRGAN at 512x512: 1176 MB of activations (measured 5382 MB at 1216x992)
        let m = MemoryUse { weights: 38 << 20, activations_per_px: (1176u64 << 20) as f64 / 262_144.0, buffers_per_px: 100.0, shared: true, scale: 4 };
        let mb = |b: u64| b >> 20;
        let act = m.added((1216, 992), (1200, 1800), 0, u64::MAX) - m.weights - (100.0 * 1216.0 * 992.0) as u64;
        assert!((5300..5500).contains(&mb(act)), "{} MB", mb(act));
        // a context already holding that much activation memory adds only the weights and buffers
        let again = m.added((1216, 992), (1200, 1800), 6000 << 20, u64::MAX);
        assert!(mb(again) < 200, "{} MB", mb(again));
        // the scratch for a 1200-wide page with 992-high tiles at 4x: ~270 MB (measured 279)
        assert!((250..300).contains(&mb(scratch((1200, 1800), 992, 4))), "{}", mb(scratch((1200, 1800), 992, 4)));
    }

    #[test]
    fn even_tiles_step_down_to_fit() {
        let m = MemoryUse { weights: 38 << 20, activations_per_px: 4500.0, buffers_per_px: 100.0, shared: true, scale: 4 };
        let gib = |g: u64| g << 30;
        let pick = |free: u64, total: u64| even_tile(|t| fits_memory(m.added((t, t), (1200, 1800), 0, 0), free, total));
        assert_eq!(pick(gib(28), gib(32)), 1024);
        assert_eq!(pick(gib(5), gib(8)), 768);
        assert_eq!(pick(gib(3), gib(4)), 512);
        assert!(fits_memory(gib(1), gib(4), gib(8)));
        assert!(!fits_memory(gib(3), gib(4), gib(8)));
    }

    #[test]
    fn the_memory_record_round_trips() {
        let dir = std::env::temp_dir().join(format!("yosh-image-memory-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (m, b, g) = ("4x_model_fp16.onnx", "TensorRT", "NVIDIA GeForce RTX 5090");
        assert_eq!(known_memory(&dir, m, b, g), None);
        let a = MemoryUse { weights: 40 << 20, activations_per_px: 4486.5, buffers_per_px: 98.0, shared: true, scale: 4 };
        record_memory(&dir, m, b, g, &a);
        record_memory(&dir, "other.onnx", b, g, &MemoryUse { shared: false, ..a });
        let b2 = MemoryUse { activations_per_px: 4500.0, ..a };
        record_memory(&dir, m, b, g, &b2);
        assert_eq!(known_memory(&dir, m, b, g), Some(b2));
        assert_eq!(known_memory(&dir, "other.onnx", b, g).map(|x| x.shared), Some(false));
        assert_eq!(known_memory(&dir, m, "TensorRT-RTX", g), None);
        let text = std::fs::read_to_string(dir.join(MEMORY_LIST)).unwrap();
        assert_eq!(text.lines().count(), 2, "replaced, not appended");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn labels() {
        assert_eq!(label((512, 512)), "512");
        assert_eq!(label((1216, 992)), "1216x992");
    }

    #[test]
    fn the_fitted_record_round_trips() {
        let dir = std::env::temp_dir().join(format!("yosh-image-fitted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (m, b, g) = ("2x_model_fp16.onnx", "TensorRT-RTX", "NVIDIA GeForce RTX 5090");
        assert!(!fitted_known(&dir, m, b, g, (1216, 992)));
        record_fitted(&dir, m, b, g, (1216, 992));
        record_fitted(&dir, m, b, g, (1216, 992));
        assert!(fitted_known(&dir, m, b, g, (1216, 992)));
        assert!(!fitted_known(&dir, m, "TensorRT", g, (1216, 992)), "per backend");
        assert!(!fitted_known(&dir, m, b, g, (1216, 704)), "per shape");
        let text = std::fs::read_to_string(dir.join(FITTED_LIST)).unwrap();
        assert_eq!(text.lines().count(), 1, "recorded once");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

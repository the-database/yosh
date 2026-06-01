// Spike 1 — decode throughput.
//
// Decodes a folder of PNG pages full-res, optionally downscales each to a target
// display height (single-channel-aware), across a hand-rolled fixed-size thread
// pool. Sweeps the thread count and reports pages/sec so we can find the decode
// ceiling and the saturation knee (which sets the real app's worker count and
// ring-buffer depth). Benchmarks zune-png vs the `png` crate.
//
// Output is discarded via black_box; per-thread buffers (decode scratch + resize
// Resizer/dst) are reused across pages to model the real pooled-buffer pipeline.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dec {
    Zune,
    Png,
}

/// Resize quality. `Bilinear` is the old baseline; `Hq` mirrors the real app's
/// content-aware path: Catmull-Rom + a Dot Gain tone LUT round-trip for gray
/// (1ch), Lanczos3 + a loose grayscale-detection scan for color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quality {
    Bilinear,
    Hq,
}

struct Cfg {
    folder: PathBuf,
    target_h: u32,
    decoders: Vec<Dec>,
    resize: bool,
    quality: Quality,
    threads: Option<usize>,
}

/// Loose grayscale detector (mirror of the app's `rgba_is_grayscale`), generic
/// over channel stride so the bench can scan native-channel decode buffers.
fn is_gray_scan(pix: &[u8], ch: usize, t: i32) -> bool {
    let mut diff_sum: u64 = 0;
    let mut non_bw: u64 = 0;
    for px in pix.chunks_exact(ch) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        if (r == 0 && g == 0 && b == 0) || (r == 255 && g == 255 && b == 255) {
            continue;
        }
        non_bw += 1;
        diff_sum += (((r - g).abs() - t).max(0)
            + ((r - b).abs() - t).max(0)
            + ((g - b).abs() - t).max(0)) as u64;
    }
    if non_bw == 0 {
        return false;
    }
    diff_sum as f64 / (non_bw as f64 * 3.0) <= t as f64 / 12.0
}

fn parse() -> Cfg {
    const USAGE: &str =
        "usage: decode_bench <folder> [--target-height H] [--decoder zune|png|both] \
         [--resize on|off] [--quality bilinear|hq] [--threads N]";
    let mut a = std::env::args().skip(1);
    let mut folder = None;
    let mut target_h = 2160u32;
    let mut decoders = vec![Dec::Zune, Dec::Png];
    let mut resize = true;
    let mut quality = Quality::Bilinear;
    let mut threads = None;
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--target-height" => target_h = a.next().expect(USAGE).parse().expect("bad height"),
            "--decoder" => {
                decoders = match a.next().expect(USAGE).as_str() {
                    "zune" => vec![Dec::Zune],
                    "png" => vec![Dec::Png],
                    "both" => vec![Dec::Zune, Dec::Png],
                    other => panic!("unknown decoder {other:?}; {USAGE}"),
                }
            }
            "--resize" => resize = a.next().expect(USAGE) == "on",
            "--quality" => {
                quality = match a.next().expect(USAGE).as_str() {
                    "bilinear" => Quality::Bilinear,
                    "hq" => Quality::Hq,
                    other => panic!("unknown quality {other:?}; {USAGE}"),
                }
            }
            "--threads" => threads = Some(a.next().expect(USAGE).parse().expect("bad N")),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                if folder.is_none() {
                    folder = Some(PathBuf::from(other));
                } else {
                    panic!("unexpected arg {other:?}; {USAGE}");
                }
            }
        }
    }
    Cfg {
        folder: folder.expect(USAGE),
        target_h,
        decoders,
        resize,
        quality,
        threads,
    }
}

/// Parse width/height/color-type straight from the PNG IHDR (bytes 16..26).
fn ihdr(b: &[u8]) -> (u32, u32, u8) {
    let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
    let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
    (w, h, b[25])
}

fn color_name(ct: u8) -> &'static str {
    match ct {
        0 => "gray",
        2 => "rgb",
        3 => "indexed",
        4 => "gray+a",
        6 => "rgba",
        _ => "?",
    }
}

fn pixel_type(ch: usize) -> PixelType {
    match ch {
        1 => PixelType::U8,
        2 => PixelType::U8x2,
        3 => PixelType::U8x3,
        4 => PixelType::U8x4,
        n => panic!("unsupported channel count {n}"),
    }
}

fn main() {
    let cfg = parse();

    // Enumerate PNGs, sorted by filename (zero-padded names => correct page order).
    let mut files: Vec<PathBuf> = std::fs::read_dir(&cfg.folder)
        .expect("read_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .map_or(false, |e| e.eq_ignore_ascii_case("png"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no PNG files in {:?}", cfg.folder);

    // Preload all bytes (so the timed loop is decode-bound, not disk-bound).
    let t = Instant::now();
    let datas: Vec<Vec<u8>> = files
        .iter()
        .map(|p| std::fs::read(p).expect("read file"))
        .collect();
    let read_el = t.elapsed();
    let total: usize = datas.iter().map(|d| d.len()).sum();
    println!(
        "files: {}   total {:.1} MB   read in {:.2?}  ({:.2} GB/s; OS cache likely warm)",
        files.len(),
        total as f64 / 1e6,
        read_el,
        total as f64 / 1e9 / read_el.as_secs_f64()
    );

    // Color-type histogram + dimensions of the set.
    let mut hist: BTreeMap<u8, usize> = BTreeMap::new();
    let mut dims: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut full_mp_total = 0.0f64;
    for d in &datas {
        let (w, h, ct) = ihdr(d);
        *hist.entry(ct).or_default() += 1;
        *dims.entry((w, h)).or_default() += 1;
        full_mp_total += (w as f64 * h as f64) / 1e6;
    }
    print!("colortypes: ");
    for (ct, n) in &hist {
        print!("{}={} ", color_name(*ct), n);
    }
    print!("  dims: ");
    for ((w, h), n) in &dims {
        print!("{w}x{h}={n} ");
    }
    println!(
        "  (avg {:.1} MP/page)",
        full_mp_total / datas.len() as f64
    );

    let sweep: Vec<usize> = match cfg.threads {
        Some(n) => vec![n],
        None => vec![1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    };

    for &dec in &cfg.decoders {
        println!(
            "\n== decoder {:?}   resize={}{} ==",
            dec,
            cfg.resize,
            if cfg.resize {
                format!(
                    " ({:?} -> {}px tall, single-channel-aware)",
                    cfg.quality, cfg.target_h
                )
            } else {
                String::new()
            }
        );
        // Warm caches/allocator with a throwaway pass.
        let _ = run(&datas, 8, &cfg, dec);
        for &n in &sweep {
            let (pps, ms, mp) = run(&datas, n, &cfg, dec);
            println!("  N={n:2}   {pps:8.1} pages/s   {ms:6.2} ms/page   {mp:7.1} full-MP/s");
        }
    }
}

/// Run one timed pass over all pages with `n` worker threads. Returns
/// (pages/sec, ms/page, full-res megapixels/sec).
fn run(datas: &[Vec<u8>], n: usize, cfg: &Cfg, dec: Dec) -> (f64, f64, f64) {
    let counter = AtomicUsize::new(0);
    let len = datas.len();
    let full_mp_total: f64 = datas
        .iter()
        .map(|d| {
            let (w, h, _) = ihdr(d);
            (w as f64 * h as f64) / 1e6
        })
        .sum();

    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                let mut resizer = Resizer::new();
                let opts_bilinear =
                    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
                let opts_catmull =
                    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
                let opts_lanczos =
                    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
                // HQ gray resamples in 16-bit linear light (see decode.rs):
                // 8-bit device -> u16 linear, U16 resize, then a 16-bit -> 8-bit
                // re-encode LUT. Exact curve values don't affect timing, so use a
                // representative gamma curve rather than the real tone tables.
                let mut to_lin = [0u16; 256];
                for (i, v) in to_lin.iter_mut().enumerate() {
                    *v = ((i as f32 / 255.0).powf(2.2) * 65535.0).round() as u16;
                }
                let mut enc = vec![0u8; 65536];
                for (i, e) in enc.iter_mut().enumerate() {
                    *e = ((i as f32 / 65535.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                }
                let mut lin: Vec<u8> = Vec::new();
                let mut dbuf: Vec<u8> = Vec::new();
                // Reused destination, rebuilt only if target dims / pixel type change.
                let mut dst: Option<(Image, u32, PixelType)> = None;

                loop {
                    let i = counter.fetch_add(1, Ordering::Relaxed);
                    if i >= len {
                        break;
                    }
                    let bytes: &[u8] = &datas[i];
                    let (w, h, _ct) = ihdr(bytes);

                    let (pix, ch): (&[u8], usize) = match dec {
                        Dec::Png => {
                            let mut r = png::Decoder::new(std::io::Cursor::new(bytes))
                                .read_info()
                                .expect("png read_info");
                            let size = r.output_buffer_size().expect("png output_buffer_size");
                            if dbuf.len() < size {
                                dbuf.resize(size, 0);
                            }
                            let info = r.next_frame(&mut dbuf).expect("png next_frame");
                            let sz = info.buffer_size();
                            let ch = sz / (w as usize * h as usize);
                            (&dbuf[..sz], ch)
                        }
                        Dec::Zune => {
                            let mut zd = zune_png::PngDecoder::new(std::io::Cursor::new(bytes));
                            zd.decode_headers().expect("zune headers");
                            let comps = zd.colorspace().expect("zune colorspace").num_components();
                            let size = w as usize * h as usize * comps;
                            if dbuf.len() < size {
                                dbuf.resize(size, 0);
                            }
                            zd.decode_into(&mut dbuf[..size]).expect("zune decode_into");
                            (&dbuf[..size], comps)
                        }
                    };

                    if cfg.resize {
                        let tw = ((w as f64) * (cfg.target_h as f64) / (h as f64)).round() as u32;
                        let hq_gray = cfg.quality == Quality::Hq && ch == 1;
                        // Pick pixel type + filter; for HQ color, also pay the
                        // loose grayscale-detection scan (as the real app does).
                        let (pt, opts) = match cfg.quality {
                            Quality::Bilinear => (pixel_type(ch), &opts_bilinear),
                            Quality::Hq if ch == 1 => (PixelType::U16, &opts_catmull),
                            Quality::Hq => {
                                if ch >= 3 {
                                    black_box(is_gray_scan(pix, ch, 12));
                                }
                                (pixel_type(ch), &opts_lanczos)
                            }
                        };
                        // HQ gray: linearize 8-bit device -> 16-bit linear (bytes).
                        let src_pix: &[u8] = if hq_gray {
                            if lin.len() < pix.len() * 2 {
                                lin.resize(pix.len() * 2, 0);
                            }
                            for (c, &s) in lin[..pix.len() * 2].chunks_exact_mut(2).zip(pix) {
                                c.copy_from_slice(&to_lin[s as usize].to_ne_bytes());
                            }
                            &lin[..pix.len() * 2]
                        } else {
                            pix
                        };
                        let src = ImageRef::new(w, h, src_pix, pt).expect("src view");
                        let need_new = match &dst {
                            Some((_, dw, dpt)) => *dw != tw || *dpt != pt,
                            None => true,
                        };
                        if need_new {
                            dst = Some((Image::new(tw, cfg.target_h, pt), tw, pt));
                        }
                        let d = dst.as_mut().unwrap();
                        resizer.resize(&src, &mut d.0, opts).expect("resize");
                        if hq_gray {
                            // Re-encode 16-bit linear -> 8-bit device.
                            let mut acc = 0u64;
                            for c in d.0.buffer().chunks_exact(2) {
                                acc += enc[u16::from_ne_bytes([c[0], c[1]]) as usize] as u64;
                            }
                            black_box(acc);
                        } else {
                            black_box(d.0.buffer());
                        }
                    } else {
                        black_box(pix);
                    }
                }
            });
        }
    });

    let el = t.elapsed().as_secs_f64();
    (
        len as f64 / el,
        el * 1000.0 / len as f64,
        full_mp_total / el,
    )
}

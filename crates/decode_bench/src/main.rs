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

struct Cfg {
    folder: PathBuf,
    target_h: u32,
    decoders: Vec<Dec>,
    resize: bool,
    threads: Option<usize>,
}

fn parse() -> Cfg {
    const USAGE: &str =
        "usage: decode_bench <folder> [--target-height H] [--decoder zune|png|both] \
         [--resize on|off] [--threads N]";
    let mut a = std::env::args().skip(1);
    let mut folder = None;
    let mut target_h = 2160u32;
    let mut decoders = vec![Dec::Zune, Dec::Png];
    let mut resize = true;
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
                format!(" (-> {}px tall, single-channel-aware)", cfg.target_h)
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
                let opts =
                    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
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
                        let pt = pixel_type(ch);
                        let src = ImageRef::new(w, h, pix, pt).expect("src view");
                        let tw =
                            ((w as f64) * (cfg.target_h as f64) / (h as f64)).round() as u32;
                        let need_new = match &dst {
                            Some((_, dw, dpt)) => *dw != tw || *dpt != pt,
                            None => true,
                        };
                        if need_new {
                            dst = Some((Image::new(tw, cfg.target_h, pt), tw, pt));
                        }
                        let d = dst.as_mut().unwrap();
                        resizer.resize(&src, &mut d.0, &opts).expect("resize");
                        black_box(d.0.buffer());
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

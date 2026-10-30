//! Decode-path identity check: decode every page of a folder or archive through
//! the real `decode_page` (HQ and LQ tiers, several target heights) and print a
//! digest of each result. Run it before and after a refactor of the decode path
//! and diff the output — any change in pixels, size, gray/RGBA layout or resize
//! path shows up as a differing line.
//!
//!   cargo run --release -p yosh-engine --example decode_digest -- "<book>" [target_h ...]

use std::hash::{Hash, Hasher};
use std::path::Path;

use fast_image_resize::Resizer;
use yosh_engine::decode::{decode_page, DecodedImage, DecodedPage};
use yosh_engine::source::{FolderSource, PageSource, SevenzSource, ZipSource};

fn digest(img: &DecodedImage, h: &mut std::collections::hash_map::DefaultHasher) {
    (img.w, img.h, img.src_w, img.src_h, img.gray, img.path.label()).hash(h);
    img.pixels.hash(h);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("usage: decode_digest <folder|archive> [target_h ...]"));
    let targets: Vec<u32> = if args.len() > 2 {
        args[2..].iter().map(|a| a.parse().expect("target_h")).collect()
    } else {
        vec![600, 1700, 2600]
    };
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let source: Box<dyn PageSource> = match ext.as_str() {
        "zip" | "cbz" => Box::new(ZipSource::new(path).expect("zip")),
        "7z" | "cb7" => Box::new(SevenzSource::new(path).expect("7z")),
        _ => Box::new(FolderSource::new(path).expect("folder")),
    };
    let mut resizer = Resizer::new();
    for i in 0..source.len() {
        let bytes = source.read_page(i).expect("read");
        for &t in &targets {
            for lq in [false, true] {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                let what = match decode_page(&bytes, t, lq, &mut resizer) {
                    Ok(DecodedPage::Still(img)) => {
                        digest(&img, &mut h);
                        format!("still {}x{}", img.w, img.h)
                    }
                    Ok(DecodedPage::Animated(frames)) => {
                        for (f, d) in &frames {
                            digest(f, &mut h);
                            d.hash(&mut h);
                        }
                        format!("anim x{}", frames.len())
                    }
                    Ok(DecodedPage::Layered(layers)) => {
                        layers.iter().for_each(|l| digest(l, &mut h));
                        format!("layers x{}", layers.len())
                    }
                    Err(e) => {
                        e.hash(&mut h);
                        "error".to_string()
                    }
                };
                println!("{} t={t} lq={lq} {what} {:016x}", source.name(i), h.finish());
            }
        }
    }
}

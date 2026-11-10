//! MangaJaNai's "Auto Adjust Levels on Grayscale Images": stretch a faded gray scan
//! so the paper is white and the ink is black before the model runs. The same
//! levels as MangaJaNaiCLI's `levels.rs` (a port of the 1.x backend's
//! `enhance_contrast`), so both apps make the same of a page.

/// The (black, white) levels of a gray image, from its histogram: each is the
/// dominant peak near its end (bins 0–30 / 255–225), followed toward the middle
/// until the histogram has dropped twice.
pub fn find_levels(gray: impl IntoIterator<Item = u8>) -> (u8, u8) {
    let mut hist = [0u64; 256];
    for v in gray {
        hist[v as usize] += 1;
    }

    let mut black = 0usize;
    let mut max = hist[0];
    for (i, &count) in hist.iter().enumerate().take(31).skip(1) {
        if count > max {
            max = count;
            black = i;
        }
    }
    let mut lower = 0;
    for (i, &count) in hist.iter().enumerate().skip(31) {
        if count > max {
            lower = 0;
            max = count;
            black = i;
        } else if count < max {
            lower += 1;
            if lower > 1 {
                break;
            }
        }
    }

    let mut white = 255usize;
    let mut max = hist[255];
    for i in (225..=254).rev() {
        if hist[i] > max {
            max = hist[i];
            white = i;
        }
    }
    // 1.x continues from 223, skipping bin 224; kept for identical levels.
    let mut lower = 0;
    for i in (0..=223).rev() {
        if hist[i] > max {
            lower = 0;
            max = hist[i];
            white = i;
        } else if hist[i] < max {
            lower += 1;
            if lower > 1 {
                break;
            }
        }
    }
    (black as u8, white as u8)
}

/// The lookup table that maps `black` → 0 and `white` → 255, or `None` for a
/// degenerate range (a flat page), which is left alone.
fn lut(black: u8, white: u8) -> Option<[u8; 256]> {
    if white <= black {
        return None;
    }
    let range = (white - black) as f32;
    let mut lut = [0u8; 256];
    for (v, out) in lut.iter_mut().enumerate() {
        let t = ((v as f32 - black as f32).max(0.0) / range).clamp(0.0, 1.0);
        *out = (t * 255.0).round() as u8;
    }
    Some(lut)
}

/// Auto-adjust the levels of the luma in `pixels` in place: `channels` = 1 (luma)
/// or 2 (luma + straight alpha, which is left as it is). Returns the levels used.
pub fn auto_adjust(pixels: &mut [u8], channels: usize) -> (u8, u8) {
    let n = channels.max(1);
    let (black, white) = find_levels(pixels.iter().step_by(n).copied());
    if let Some(lut) = lut(black, white) {
        for v in pixels.iter_mut().step_by(n) {
            *v = lut[*v as usize];
        }
    }
    (black, white)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(levels: &[(u8, usize)]) -> Vec<u8> {
        levels.iter().flat_map(|&(v, n)| std::iter::repeat_n(v, n)).collect()
    }

    #[test]
    fn faded_scan_levels() {
        // gray ink at 40, yellowed paper at 230, a little noise around both
        let p = page(&[(38, 50), (40, 900), (41, 60), (120, 30), (229, 80), (230, 3000), (231, 90)]);
        assert_eq!(find_levels(p.iter().copied()), (40, 230));
    }

    #[test]
    fn already_clean_page() {
        let p = page(&[(0, 500), (255, 5000)]);
        assert_eq!(find_levels(p), (0, 255));
    }

    #[test]
    fn adjust_stretches_and_clamps() {
        // (40, 230) levels: ink at 40, paper at 230, one mid-gray pixel.
        let mut g = page(&[(40, 900), (230, 3000), (135, 1)]);
        assert_eq!(auto_adjust(&mut g, 1), (40, 230));
        assert!(g[..900].iter().all(|&v| v == 0) && g[900..3900].iter().all(|&v| v == 255));
        assert_eq!(g[3900], 128);
        // Outside the levels clamps (MangaJaNaiCLI's `apply`).
        let map = lut(40, 230).unwrap();
        assert_eq!([20, 40, 135, 230, 250].map(|v| map[v]), [0, 0, 128, 255, 255]);
        let mut flat = vec![100; 4];
        auto_adjust(&mut flat, 1);
        assert_eq!(flat, vec![100; 4]);
        assert!(lut(100, 100).is_none());
    }

    #[test]
    fn adjust_leaves_the_alpha_alone() {
        let luma = page(&[(40, 900), (230, 3000)]);
        let mut la: Vec<u8> = luma.iter().flat_map(|&y| [y, 77]).collect();
        assert_eq!(auto_adjust(&mut la, 2), (40, 230));
        assert!(la.as_chunks::<2>().0.iter().all(|p| p[1] == 77));
        assert_eq!((la[0], la[2 * 3899]), (0, 255));
    }
}

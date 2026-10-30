//! "Auto Adjust Levels on Grayscale Images": a port of run_upscale.py's `enhance_contrast`, which
//! stretches faded scans so the paper is white and the ink is black before the model runs.

/// The (black, white) levels of a gray image, from its histogram: each is the dominant peak near
/// its end (bins 0–30 / 255–225), followed toward the middle until the histogram has dropped twice.
pub fn find_levels(gray: &[u8]) -> (u8, u8) {
    let mut hist = [0u64; 256];
    for &v in gray {
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

/// Stretch `gray` in place so `black` → 0 and `white` → 255. A degenerate range (a flat page) is
/// left alone instead of dividing by zero.
pub fn apply(gray: &mut [u8], black: u8, white: u8) {
    if white <= black {
        return;
    }
    let range = (white - black) as f32;
    let mut lut = [0u8; 256];
    for (v, out) in lut.iter_mut().enumerate() {
        let t = ((v as f32 - black as f32).max(0.0) / range).clamp(0.0, 1.0);
        *out = (t * 255.0).round() as u8;
    }
    for v in gray {
        *v = lut[*v as usize];
    }
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
        assert_eq!(find_levels(&p), (40, 230));
    }

    #[test]
    fn already_clean_page() {
        let p = page(&[(0, 500), (255, 5000)]);
        assert_eq!(find_levels(&p), (0, 255));
    }

    #[test]
    fn apply_stretches_and_clamps() {
        let mut g = vec![20, 40, 135, 230, 250];
        apply(&mut g, 40, 230);
        assert_eq!(g, vec![0, 0, 128, 255, 255]);
        let mut flat = vec![100; 4];
        apply(&mut flat, 100, 100);
        assert_eq!(flat, vec![100; 4]);
    }
}

//! Prefetch window: which page indices to have decoded, ordered nearest-first
//! and forward-biased. (M1.4 widens `fwd` with flip velocity.)

/// Indices to decode around `current`, within `[current-back, current+fwd]`,
/// clamped to `[0, len)`, ordered so the current page is first and forward
/// pages outrank backward pages at equal distance.
pub fn desired_window(current: usize, len: usize, fwd: usize, back: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let lo = current.saturating_sub(back);
    let hi = (current + fwd).min(len - 1);
    let mut v: Vec<usize> = (lo..=hi).collect();
    v.sort_by_key(|&i| {
        if i >= current {
            (i - current) as i64 * 2 // forward weight
        } else {
            (current - i) as i64 * 3 + 1 // backward costs more
        }
    });
    v
}

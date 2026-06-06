//! Page layout: single page vs two-page spread.
//!
//! Spread pairing has a parity `offset` (0 or 1) so a mis-paired volume can be
//! corrected:
//!   offset 0 — page 0 (cover) shown alone, then (1,2), (3,4), …
//!   offset 1 — no cover-single: (0,1), (2,3), (4,5), …
//! The reader's `index` is the anchor (lower page of the current view); RTL/LTR
//! screen ordering is applied at placement time.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Single,
    Spread,
}

impl Layout {
    pub fn label(self) -> &'static str {
        match self {
            Layout::Single => "single",
            Layout::Spread => "spread",
        }
    }
    pub fn toggled(self) -> Self {
        match self {
            Layout::Single => Layout::Spread,
            Layout::Spread => Layout::Single,
        }
    }
}

/// Number of leading single pages before the pairing grid (1 for offset 0 = a
/// cover page, 0 for offset 1).
fn lead(offset: usize) -> usize {
    1 - offset.min(1)
}

/// The anchor (lower page index) of the view containing `index`.
pub fn view_start(layout: Layout, index: usize, offset: usize) -> usize {
    match layout {
        Layout::Single => index,
        Layout::Spread => {
            let l = lead(offset);
            if index < l {
                index
            } else {
                l + ((index - l) / 2) * 2
            }
        }
    }
}

/// The page(s) composing the view containing `index`: `(first, second?)` in
/// ascending order (screen ordering handled by the caller via direction).
pub fn view_pages(
    layout: Layout,
    index: usize,
    len: usize,
    offset: usize,
) -> (usize, Option<usize>) {
    if len == 0 {
        return (0, None);
    }
    let start = view_start(layout, index, offset).min(len - 1);
    match layout {
        Layout::Single => (start, None),
        Layout::Spread => {
            let l = lead(offset);
            if start < l {
                (start, None) // leading single (cover)
            } else if start + 1 < len {
                (start, Some(start + 1))
            } else {
                (start, None) // trailing orphan
            }
        }
    }
}

/// Anchor of the next view after the one containing `index`.
pub fn next_view(layout: Layout, index: usize, len: usize, offset: usize) -> usize {
    if len == 0 {
        return 0;
    }
    match layout {
        Layout::Single => (index + 1).min(len - 1),
        Layout::Spread => {
            let (a, b) = view_pages(layout, index, len, offset);
            let last = b.unwrap_or(a);
            if last + 1 >= len {
                a
            } else {
                view_start(layout, last + 1, offset)
            }
        }
    }
}

/// Anchor of the previous view before the one containing `index`.
pub fn prev_view(layout: Layout, index: usize, len: usize, offset: usize) -> usize {
    match layout {
        Layout::Single => index.saturating_sub(1),
        Layout::Spread => {
            let (a, _) = view_pages(layout, index, len, offset);
            if a == 0 {
                0
            } else {
                view_start(layout, a - 1, offset)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_offset0_cover_single() {
        let l = Layout::Spread;
        assert_eq!(view_pages(l, 0, 10, 0), (0, None));
        assert_eq!(view_pages(l, 1, 10, 0), (1, Some(2)));
        assert_eq!(view_pages(l, 2, 10, 0), (1, Some(2)));
        assert_eq!(view_pages(l, 3, 10, 0), (3, Some(4)));
        assert_eq!(view_pages(l, 9, 10, 0), (9, None)); // trailing orphan
    }

    #[test]
    fn spread_offset1_no_cover() {
        let l = Layout::Spread;
        assert_eq!(view_pages(l, 0, 10, 1), (0, Some(1)));
        assert_eq!(view_pages(l, 1, 10, 1), (0, Some(1)));
        assert_eq!(view_pages(l, 2, 10, 1), (2, Some(3)));
    }

    #[test]
    fn spread_navigation() {
        let l = Layout::Spread;
        // offset 0
        assert_eq!(next_view(l, 0, 10, 0), 1);
        assert_eq!(next_view(l, 1, 10, 0), 3);
        assert_eq!(prev_view(l, 3, 10, 0), 1);
        assert_eq!(prev_view(l, 1, 10, 0), 0);
        // offset 1
        assert_eq!(next_view(l, 0, 10, 1), 2);
        assert_eq!(prev_view(l, 2, 10, 1), 0);
    }
}

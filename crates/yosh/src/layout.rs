//! Page layout: single page vs two-page spread.
//!
//! Spread convention: page 0 (cover) is shown alone, then pages pair up as
//! (1,2), (3,4), (5,6), … The reader's `index` is the anchor (the lower page of
//! the current view); RTL/LTR screen ordering is applied at placement time.

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

/// The anchor (lower page index) of the view containing `index`.
pub fn view_start(layout: Layout, index: usize) -> usize {
    match layout {
        Layout::Single => index,
        Layout::Spread => {
            if index == 0 {
                0
            } else if index % 2 == 1 {
                index
            } else {
                index - 1
            }
        }
    }
}

/// The page(s) composing the view that contains `index`: `(first, second?)` in
/// ascending index order (screen ordering handled by the caller via direction).
pub fn view_pages(layout: Layout, index: usize, len: usize) -> (usize, Option<usize>) {
    let start = view_start(layout, index).min(len.saturating_sub(1));
    match layout {
        Layout::Single => (start, None),
        Layout::Spread => {
            if start == 0 {
                (0, None) // cover alone
            } else if start + 1 < len {
                (start, Some(start + 1))
            } else {
                (start, None) // last page, no partner
            }
        }
    }
}

/// Anchor of the next view after the one containing `index`.
pub fn next_view(layout: Layout, index: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    match layout {
        Layout::Single => (index + 1).min(len - 1),
        Layout::Spread => {
            let start = view_start(layout, index);
            if start == 0 {
                1.min(len - 1)
            } else {
                (start + 2).min(len - 1)
            }
        }
    }
}

/// Anchor of the previous view before the one containing `index`.
pub fn prev_view(layout: Layout, index: usize) -> usize {
    match layout {
        Layout::Single => index.saturating_sub(1),
        Layout::Spread => {
            let start = view_start(layout, index);
            if start <= 1 {
                0
            } else {
                start - 2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_pairs_with_cover_single() {
        let l = Layout::Spread;
        assert_eq!(view_pages(l, 0, 10), (0, None));
        assert_eq!(view_pages(l, 1, 10), (1, Some(2)));
        assert_eq!(view_pages(l, 2, 10), (1, Some(2)));
        assert_eq!(view_pages(l, 3, 10), (3, Some(4)));
        // last page with no partner
        assert_eq!(view_pages(l, 9, 10), (9, None));
    }

    #[test]
    fn spread_navigation() {
        let l = Layout::Spread;
        assert_eq!(next_view(l, 0, 10), 1); // cover -> first pair
        assert_eq!(next_view(l, 1, 10), 3);
        assert_eq!(next_view(l, 2, 10), 3);
        assert_eq!(prev_view(l, 3), 1);
        assert_eq!(prev_view(l, 1), 0); // back to cover
        assert_eq!(prev_view(l, 2), 0);
    }
}

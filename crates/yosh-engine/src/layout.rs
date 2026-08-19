//! Page layout: single page vs two-page spread.
//!
//! Spread pairing has a parity `offset` (0 or 1) so a mis-paired volume can be
//! corrected:
//!   offset 0 — page 0 (cover) shown alone, then (1,2), (3,4), …
//!   offset 1 — no cover-single: (0,1), (2,3), (4,5), …
//! The reader's `index` is the anchor (lower page of the current view); RTL/LTR
//! screen ordering is applied at placement time.
//!
//! **Pre-joined double-page spreads re-phase the pairing.** A landscape page is one
//! *file* holding two *pages*, so it must occupy a view alone — and everything after
//! it shifts by one, which is exactly what keeps real-page parity across it. Pairing
//! therefore can't be pure `(index, offset)` arithmetic; it needs to know which pages
//! are wide ([`WideSet`]), and every consumer goes through [`Grid`] so nothing can
//! compute a pairing the drawing code then disagrees with. That split is what used to
//! swallow the page after a joined spread: the grid paired `(11, 12)`, the draw code
//! dropped 12 because 11 was landscape, and navigation still stepped past both.
//!
//! One consequence worth knowing: the `offset` parity only governs the run *before*
//! the first joined spread. After one, the phase is pinned to `w + 1` — a view can't
//! start on a page that is already inside the wide page's view — so toggling the
//! offset can no longer re-pair that tail. That is inherent to a page that is alone,
//! not a bug to fix with a second offset.

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

/// Which pages are pre-joined double-page spreads (landscape source) and therefore
/// show alone. Learned as pages decode — a page whose size isn't known yet reads as
/// not wide, and the pairing re-phases in place once it lands.
///
/// Deliberately *not* kept in `PageCache`: that cache evicts, and un-learning a
/// joined page would silently re-pair the rest of the volume under the reader. It
/// also keys off the **source** dimensions, not the decoded ones, so a near-square
/// page can't change answer with the decode target.
///
/// Joined spreads are rare, so this is the sorted list of wide indices — a handful
/// of entries, queried by binary search, with an `is_empty` fast path that makes a
/// volume of ordinary pages cost exactly what it did before.
#[derive(Default, Clone)]
pub struct WideSet {
    idx: Vec<usize>,
    epoch: u64,
}

impl WideSet {
    /// Is page `i` known to be a joined double-page spread?
    pub fn contains(&self, i: usize) -> bool {
        self.idx.binary_search(&i).is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    /// The largest known wide page in `lo..index`, if any. This is what makes
    /// [`Grid::view_start`] a closed form: pairing restarts at `w + 1` after a wide
    /// page, so only the *nearest* one below the index matters.
    fn prev_wide(&self, lo: usize, index: usize) -> Option<usize> {
        let p = self.idx.partition_point(|&w| w < index);
        self.idx[..p].last().copied().filter(|&w| w >= lo)
    }

    /// Record what a landed decode says about page `i`; `true` if the set changed.
    /// [`WideSet::epoch`] bumps only on a real change, so the hundreds of thumbnails
    /// in a volume fill don't each invalidate the prefetch job list.
    pub fn set(&mut self, i: usize, wide: bool) -> bool {
        match (self.idx.binary_search(&i), wide) {
            (Err(p), true) => {
                self.idx.insert(p, i);
                self.epoch += 1;
                true
            }
            (Ok(p), false) => {
                self.idx.remove(p);
                self.epoch += 1;
                true
            }
            _ => false,
        }
    }

    /// Content-change counter — differs whenever the pairing may have changed.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn clear(&mut self) {
        if !self.idx.is_empty() {
            self.idx.clear();
            self.epoch += 1;
        }
    }

    /// Re-key every flag through `f` (old index → new index), mirroring
    /// `PageCache::remap` so a folder rescan carries the learned orientations with
    /// their files instead of re-pairing the volume. Entries mapping to `None` (file
    /// gone, or off the end of a shrunken volume) are dropped.
    pub fn remap(&mut self, f: impl Fn(usize) -> Option<usize>) {
        let old = std::mem::take(&mut self.idx);
        self.idx = old.into_iter().filter_map(f).collect();
        self.idx.sort_unstable();
        self.idx.dedup();
        self.epoch += 1;
    }
}

/// The pairing for one volume: the grid parameters plus which pages are joined
/// spreads. Every view-mapping question goes through here — there is deliberately no
/// way to compute a pairing without the wide set.
pub struct Grid<'a> {
    pub layout: Layout,
    pub len: usize,
    pub offset: usize,
    pub wide: &'a WideSet,
}

impl Grid<'_> {
    /// Number of leading single pages before the pairing grid (1 for offset 0 = a
    /// cover page, 0 for offset 1).
    fn lead(&self) -> usize {
        1 - self.offset.min(1)
    }

    /// The anchor (lower page index) of the view containing `index`.
    ///
    /// A closed form rather than a walk from page 0: a wide page always fills a view
    /// by itself, so the next view starts at exactly `w + 1`, and between two wide
    /// pages the stride is a plain 2. `index` is clamped to the volume *first* —
    /// clamping the result instead can land on a page that isn't an anchor.
    pub fn view_start(&self, index: usize) -> usize {
        if self.len == 0 {
            return 0;
        }
        let index = index.min(self.len - 1);
        match self.layout {
            Layout::Single => index,
            Layout::Spread => {
                let l = self.lead();
                if index < l {
                    return index;
                }
                if self.wide.is_empty() {
                    return l + ((index - l) / 2) * 2; // no joined spreads: plain parity grid
                }
                if self.wide.contains(index) {
                    return index;
                }
                let base = self.wide.prev_wide(l, index).map_or(l, |w| w + 1);
                base + ((index - base) / 2) * 2
            }
        }
    }

    /// The page(s) composing the view containing `index`: `(first, second?)` in
    /// ascending order (screen ordering handled by the caller via direction).
    pub fn view_pages(&self, index: usize) -> (usize, Option<usize>) {
        if self.len == 0 {
            return (0, None);
        }
        let start = self.view_start(index);
        match self.layout {
            Layout::Single => (start, None),
            Layout::Spread => {
                // Alone when: it's a leading single (cover); the page itself is a
                // joined spread; there's nothing left to pair with; or the facing
                // slot holds a joined spread, which needs its own view — so this
                // page is orphaned rather than squashed beside a double-wide image.
                if start < self.lead()
                    || self.wide.contains(start)
                    || start + 1 >= self.len
                    || self.wide.contains(start + 1)
                {
                    (start, None)
                } else {
                    (start, Some(start + 1))
                }
            }
        }
    }

    /// Anchor of the next view after the one containing `index`.
    pub fn next_view(&self, index: usize) -> usize {
        if self.len == 0 {
            return 0;
        }
        match self.layout {
            Layout::Single => (index + 1).min(self.len - 1),
            Layout::Spread => {
                let (a, b) = self.view_pages(index);
                let last = b.unwrap_or(a);
                if last + 1 >= self.len {
                    a
                } else {
                    self.view_start(last + 1)
                }
            }
        }
    }

    /// Anchor of the previous view before the one containing `index`.
    pub fn prev_view(&self, index: usize) -> usize {
        match self.layout {
            Layout::Single => index.min(self.len.saturating_sub(1)).saturating_sub(1),
            Layout::Spread => {
                let (a, _) = self.view_pages(index);
                if a == 0 { 0 } else { self.view_start(a - 1) }
            }
        }
    }

    /// Is page `index` actually drawn beside a facing page? The decode target hangs
    /// on this — a pair shares one box, a lone page gets the whole one — and guessing
    /// it from the page's own aspect gets the cover, a trailing orphan and a page
    /// orphaned by a joined neighbour wrong: all three are drawn alone.
    pub fn is_paired(&self, index: usize) -> bool {
        self.view_pages(index).1.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wides(v: &[usize]) -> WideSet {
        let mut w = WideSet::default();
        for &i in v {
            w.set(i, true);
        }
        w
    }

    fn grid(len: usize, offset: usize, wide: &WideSet) -> Grid<'_> {
        Grid { layout: Layout::Spread, len, offset, wide }
    }

    /// Walk forward from page 0, collecting the views in order.
    fn walk(g: &Grid<'_>) -> Vec<(usize, Option<usize>)> {
        let mut out = Vec::new();
        let mut a = 0;
        loop {
            out.push(g.view_pages(a));
            let n = g.next_view(a);
            if n == a {
                break;
            }
            a = n;
        }
        out
    }

    /// The reference model the closed form has to match: step left to right, taking
    /// one page whenever it (or its facing slot) is a joined spread, else two.
    fn greedy(len: usize, offset: usize, wide: &WideSet) -> Vec<(usize, Option<usize>)> {
        let lead = 1 - offset.min(1);
        let mut views = Vec::new();
        let mut a = 0;
        while a < lead.min(len) {
            views.push((a, None));
            a += 1;
        }
        while a < len {
            if wide.contains(a) || a + 1 >= len || wide.contains(a + 1) {
                views.push((a, None));
                a += 1;
            } else {
                views.push((a, Some(a + 1)));
                a += 2;
            }
        }
        views
    }

    fn pages(views: &[(usize, Option<usize>)]) -> Vec<usize> {
        views.iter().flat_map(|&(a, b)| std::iter::once(a).chain(b)).collect()
    }

    #[test]
    fn spread_offset0_cover_single() {
        let w = WideSet::default();
        let g = grid(10, 0, &w);
        assert_eq!(g.view_pages(0), (0, None));
        assert_eq!(g.view_pages(1), (1, Some(2)));
        assert_eq!(g.view_pages(2), (1, Some(2)));
        assert_eq!(g.view_pages(3), (3, Some(4)));
        assert_eq!(g.view_pages(9), (9, None)); // trailing orphan
    }

    #[test]
    fn spread_offset1_no_cover() {
        let w = WideSet::default();
        let g = grid(10, 1, &w);
        assert_eq!(g.view_pages(0), (0, Some(1)));
        assert_eq!(g.view_pages(1), (0, Some(1)));
        assert_eq!(g.view_pages(2), (2, Some(3)));
    }

    #[test]
    fn spread_navigation() {
        let w = WideSet::default();
        let g0 = grid(10, 0, &w);
        assert_eq!(g0.next_view(0), 1);
        assert_eq!(g0.next_view(1), 3);
        assert_eq!(g0.prev_view(3), 1);
        assert_eq!(g0.prev_view(1), 0);
        let g1 = grid(10, 1, &w);
        assert_eq!(g1.next_view(0), 2);
        assert_eq!(g1.prev_view(2), 0);
    }

    /// A volume with no joined spread must pair exactly as the old pure-arithmetic
    /// grid did — the no-regression gate for every ordinary book (and for any page
    /// whose orientation hasn't been learned yet).
    #[test]
    fn spread_unknown_wide_matches_old_grid() {
        let w = WideSet::default();
        for offset in [0, 1] {
            let l = 1 - offset.min(1);
            for len in 1..=20 {
                let g = grid(len, offset, &w);
                for i in 0..len {
                    let start = if i < l { i } else { l + ((i - l) / 2) * 2 };
                    assert_eq!(g.view_start(i), start, "offset {offset}, len {len}, index {i}");
                    let want = if start < l || start + 1 >= len {
                        (start, None)
                    } else {
                        (start, Some(start + 1))
                    };
                    assert_eq!(g.view_pages(i), want, "offset {offset}, len {len}, index {i}");
                }
            }
        }
    }

    /// The closed form equals the greedy scan for *every* arrangement of joined
    /// spreads, exhaustively — equivalence by exhaustion rather than by argument.
    #[test]
    fn spread_closed_form_matches_greedy_scan() {
        for len in 1..=11_usize {
            for mask in 0..(1_u32 << len) {
                let w = wides(&(0..len).filter(|k| mask >> k & 1 == 1).collect::<Vec<_>>());
                for offset in [0, 1] {
                    let g = grid(len, offset, &w);
                    let want = greedy(len, offset, &w);
                    // Every index maps to the view that contains it.
                    for &(a, b) in &want {
                        assert_eq!(g.view_pages(a), (a, b), "len {len} mask {mask} off {offset}");
                        if let Some(bi) = b {
                            assert_eq!(g.view_pages(bi), (a, b), "len {len} mask {mask} off {offset}");
                        }
                    }
                    // …and walking forward reproduces the whole view list, so no page
                    // is skipped (the reported bug) or shown twice.
                    assert_eq!(walk(&g), want, "len {len} mask {mask} off {offset}");
                    assert_eq!(pages(&want), (0..len).collect::<Vec<_>>());
                }
            }
        }
    }

    /// Navigation is reversible, and every anchor it produces is canonical.
    #[test]
    fn spread_wide_navigation_round_trips() {
        for len in 1..=10_usize {
            for mask in 0..(1_u32 << len) {
                let w = wides(&(0..len).filter(|k| mask >> k & 1 == 1).collect::<Vec<_>>());
                for offset in [0, 1] {
                    let g = grid(len, offset, &w);
                    let mut a = 0;
                    loop {
                        assert_eq!(g.view_start(a), a, "len {len} mask {mask}: {a} not an anchor");
                        let n = g.next_view(a);
                        if n == a {
                            break;
                        }
                        assert_eq!(g.prev_view(n), a, "len {len} mask {mask}: {a} → {n} → back");
                        a = n;
                    }
                    // …and backward from the last anchor covers everything too.
                    let mut back = vec![g.view_pages(a)];
                    while a != 0 {
                        a = g.prev_view(a);
                        back.push(g.view_pages(a));
                    }
                    back.reverse();
                    assert_eq!(back, greedy(len, offset, &w), "len {len} mask {mask} off {offset}");
                }
            }
        }
    }

    /// A joined double-page spread fills its own view.
    #[test]
    fn spread_wide_page_shows_alone() {
        let w = wides(&[11]);
        let g = grid(19, 0, &w);
        assert_eq!(g.view_pages(11), (11, None));
        assert_eq!(g.view_pages(12), (12, Some(13)));
        assert!(!g.is_paired(11));
        assert!(g.is_paired(12));
    }

    /// The reported bug: 19 pages, index 11 is a pre-joined spread. Paging forward
    /// used to skip index 12 — the grid paired (11, 12), the draw code dropped 12
    /// because 11 is landscape, and `next_view` still stepped past both to 13.
    #[test]
    fn spread_wide_page_no_skip() {
        let w = wides(&[11]);
        let g = grid(19, 0, &w);
        let views = walk(&g);
        assert_eq!(
            views,
            vec![
                (0, None),
                (1, Some(2)),
                (3, Some(4)),
                (5, Some(6)),
                (7, Some(8)),
                (9, Some(10)),
                (11, None),
                (12, Some(13)),
                (14, Some(15)),
                (16, Some(17)),
                (18, None),
            ]
        );
        assert_eq!(pages(&views), (0..19).collect::<Vec<_>>());
    }

    /// A joined spread landing in the facing slot orphans the page before it rather
    /// than being squashed beside it — what offset 1 did to the same chapter.
    #[test]
    fn spread_wide_second_slot_orphans_first() {
        let w = wides(&[11]);
        let g = grid(19, 1, &w);
        assert_eq!(g.view_pages(10), (10, None));
        assert_eq!(g.next_view(10), 11);
        assert_eq!(g.view_pages(11), (11, None));
        assert_eq!(g.next_view(11), 12);
        assert_eq!(g.view_pages(12), (12, Some(13)));
    }

    #[test]
    fn spread_wide_edges() {
        // A wide *cover* doesn't re-phase: page 0 is already alone at offset 0.
        let w = wides(&[0]);
        assert_eq!(walk(&grid(5, 0, &w)), vec![(0, None), (1, Some(2)), (3, Some(4))]);
        // …but at offset 1 it is the grid's first slot, so it does.
        assert_eq!(walk(&grid(4, 1, &w)), vec![(0, None), (1, Some(2)), (3, None)]);
        // Two adjacent joined spreads.
        let w = wides(&[1, 2]);
        assert_eq!(walk(&grid(5, 0, &w)), vec![(0, None), (1, None), (2, None), (3, Some(4))]);
        // Joined spread as the last page.
        let w = wides(&[3]);
        assert_eq!(walk(&grid(4, 0, &w)), vec![(0, None), (1, Some(2)), (3, None)]);
        // Degenerate volumes.
        let w = wides(&[0]);
        let g = grid(1, 0, &w);
        assert_eq!(g.view_pages(0), (0, None));
        assert_eq!(g.next_view(0), 0);
        assert_eq!(g.prev_view(0), 0);
        assert_eq!(grid(0, 0, &w).view_pages(0), (0, None));
    }

    /// Single-page layout ignores the wide set entirely.
    #[test]
    fn single_layout_ignores_wide() {
        let w = wides(&[0, 3, 4]);
        let g = Grid { layout: Layout::Single, len: 6, offset: 0, wide: &w };
        for i in 0..6 {
            assert_eq!(g.view_pages(i), (i, None));
            assert_eq!(g.view_start(i), i);
        }
        assert_eq!(g.next_view(2), 3);
        assert_eq!(g.prev_view(3), 2);
        assert_eq!(g.next_view(5), 5);
    }

    /// The three kinds of page that are *drawn alone* inside a spread. Decode
    /// targets hang on this: sizing them as half a pair decodes them at half their
    /// drawn height, and the GPU then upscales it back - the one resample the
    /// single-resize invariant forbids.
    #[test]
    fn spread_pages_drawn_alone_are_not_paired() {
        let w = wides(&[11]);
        let g = grid(19, 0, &w);
        assert!(!g.is_paired(0), "cover");
        assert!(!g.is_paired(18), "trailing orphan");
        assert!(!g.is_paired(11), "the joined spread itself");
        // A page whose facing slot holds a joined spread is orphaned, not squashed.
        let w = wides(&[11]);
        let g = grid(19, 1, &w);
        assert!(!g.is_paired(10), "orphaned by a joined neighbour");
        assert!(g.is_paired(9) && g.is_paired(8), "ordinary pairs are unaffected");
    }

    #[test]
    fn wide_set_basics() {
        let mut w = WideSet::default();
        assert!(w.is_empty());
        assert!(w.set(5, true));
        assert!(!w.set(5, true)); // no change, no epoch bump
        let e = w.epoch();
        assert!(!w.set(6, false));
        assert_eq!(w.epoch(), e);
        assert!(w.set(2, true));
        assert!(w.contains(2) && w.contains(5) && !w.contains(3));
        assert_eq!(w.prev_wide(0, 5), Some(2));
        assert_eq!(w.prev_wide(3, 5), None); // below `lo` doesn't count
        assert_eq!(w.prev_wide(0, 2), None);
        assert!(w.set(5, false));
        assert!(!w.contains(5));
        // remap re-keys and drops what maps to None.
        w.set(5, true);
        w.remap(|i| (i != 2).then_some(i + 10));
        assert!(w.contains(15) && !w.contains(12) && !w.contains(5));
        w.clear();
        assert!(w.is_empty());
    }
}

//! Bounded page-texture cache — the "ring buffer" of decoded display-res frames.
//! Eviction is distance-based around the current page (a reader re-reads nearby
//! pages, so keep a window centered on `current`).

use std::collections::HashMap;

use crate::page::PageTexture;

pub struct PageCache {
    map: HashMap<usize, PageTexture>,
    cap: usize,
}

impl PageCache {
    pub fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            cap,
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.map.contains_key(&index)
    }

    pub fn get(&self, index: usize) -> Option<&PageTexture> {
        self.map.get(&index)
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Insert a page, evicting the entries furthest from `current` if over cap.
    pub fn insert(&mut self, index: usize, page: PageTexture, current: usize) {
        self.map.insert(index, page);
        while self.map.len() > self.cap {
            let victim = self
                .map
                .keys()
                .copied()
                .filter(|&k| k != current)
                .max_by_key(|&k| (k as i64 - current as i64).abs());
            match victim {
                Some(k) => {
                    self.map.remove(&k);
                }
                None => break,
            }
        }
    }
}

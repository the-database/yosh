//! Bounded page-texture cache — the "ring buffer" of decoded display-res frames.
//! Eviction is distance-based around the current page (a reader re-reads nearby
//! pages, so keep a window centered on `current`). Evicted textures are returned
//! to the pool for reuse.

use std::collections::HashMap;
use std::sync::Arc;

use crate::page::PageTexture;
use crate::texpool::TexturePool;

pub struct PageCache {
    map: HashMap<usize, PageTexture>,
    cap: usize,
    pool: Arc<TexturePool>,
}

impl PageCache {
    pub fn new(cap: usize, pool: Arc<TexturePool>) -> Self {
        Self {
            map: HashMap::new(),
            cap,
            pool,
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.map.contains_key(&index)
    }

    /// Number of pages currently resident (used for the LQ-tier fill readout).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Eviction ceiling — used by the LQ tier to stop filling (and stop the redraw
    /// loop) once a volume larger than the cache has filled what fits.
    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn get(&self, index: usize) -> Option<&PageTexture> {
        self.map.get(&index)
    }

    /// Indices of pages currently buffered (decoded + GPU-uploaded, ready to draw
    /// instantly) — the set the seekbar's cache bar visualizes. Bounded by the
    /// prefetch window (~`back + fwd + 1` entries clustered around the read position).
    pub fn buffered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.map.keys().copied()
    }

    pub fn clear(&mut self) {
        for (_, page) in self.map.drain() {
            page.recycle(&self.pool);
        }
    }

    /// Insert a page, evicting the entries furthest from `current` if over cap.
    pub fn insert(&mut self, index: usize, page: PageTexture, current: usize) {
        if let Some(old) = self.map.insert(index, page) {
            old.recycle(&self.pool);
        }
        while self.map.len() > self.cap {
            let victim = self
                .map
                .keys()
                .copied()
                .filter(|&k| k != current)
                .max_by_key(|&k| (k as i64 - current as i64).abs());
            match victim {
                Some(k) => {
                    if let Some(page) = self.map.remove(&k) {
                        page.recycle(&self.pool);
                    }
                }
                None => break,
            }
        }
    }
}

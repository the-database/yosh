//! The platform-agnostic reading-state machine.
//!
//! This module will own the reader's view model — navigation, zoom/pan, fit and
//! layout, the continuous-scroll anchor, the decode-view debounce, and the
//! single-resize-invariant draw math — so a shell only has to supply a surface,
//! input, and storage. It is filled in across Phase 2; for now it carries the
//! [`Viewport`], the one piece the shell hands in every frame.

/// The drawable surface size in physical pixels, as the reading math sees it.
///
/// The shell mirrors its `wgpu` surface config into this once per frame (and on
/// resize), so the reader computes exact per-page decode targets and quad
/// placement without depending on any windowing type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Viewport {
    pub w: u32,
    pub h: u32,
}

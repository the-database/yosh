//! AI super-resolution — the opt-in upscaling add-on.
//!
//! The engine only knows the [`Upscaler`] trait: "upscale this 8-bit gray/RGB
//! image by whatever integer factor suits a target scale". Everything heavy (the
//! inference runtime, TensorRT/DirectML, the models) lives in an add-on the user
//! installs separately; without one, no upscaler exists, the pool never spawns
//! its SR lane, and every decode is bit-identical to a build without this module.
//!
//! When an upscaler is installed, the pool runs **one** extra worker (the SR
//! lane) on its own lowest-priority queue: a page is first shown through the
//! normal decode path, then — only if it is displayed *above* its source size,
//! where the GPU would otherwise magnify it — re-decoded through the model and
//! swapped in place. The model output is brought to the exact on-screen size by
//! the same HQ CPU resize as every other page ([`crate::decode::decode_sr`]), so
//! the single-resize invariant holds: the GPU samples the result 1:1.
//!
//! Which model handles a page is data ([`rules`]), mirroring
//! MangaJaNaiConverterGui's workflow chains so both apps pick identically.

pub mod rules;

#[cfg(feature = "sr")]
pub mod aji;

/// What an [`Upscaler`] made of one request.
pub enum SrOutcome {
    /// Upscaled by `scale` (both axes) with `model`: `pixels` has the input's
    /// channel count and `(w * scale) x (h * scale)` tightly packed rows.
    Done { scale: u32, pixels: Vec<u8>, model: std::sync::Arc<str> },
    /// No model applies to this page (no matching rule, or the rule says none).
    Skip,
    /// The model exists but isn't usable yet (e.g. a TensorRT engine is still
    /// building). [`Upscaler::epoch`] changes when retrying may succeed.
    NotReady,
    /// The `cancel` callback asked to stop.
    Cancelled,
    /// The model or runtime failed; the page keeps its normal decode.
    Failed(String),
}

/// Why the SR lane produced no upscaled page (carried by `Msg::SrSkipped`, and
/// surfaced to the user through [`crate::reader::SrViewState`]).
#[derive(Clone, Debug, PartialEq)]
pub enum SrSkip {
    /// Nothing for a model to do: shown at or below source size, an animation,
    /// or transparent (alpha stays on the normal path).
    Unsupported,
    /// No rule maps this page to a model.
    NoModel,
    /// The model isn't usable yet (a TensorRT engine is building); retried when
    /// [`Upscaler::epoch`] moves.
    NotReady,
    /// The model or runtime failed.
    Failed(String),
}

pub trait Upscaler: Send + Sync {
    /// Upscale a `w x h` image with `channels` (1 = gray, 3 = RGB) 8-bit planes
    /// interleaved in tightly packed rows. `target_scale` is displayed height over
    /// source height, which picks the model. `cancel` is polled between tiles.
    fn upscale(
        &self,
        w: u32,
        h: u32,
        channels: u32,
        pixels: &[u8],
        target_scale: f32,
        cancel: &dyn Fn() -> bool,
    ) -> SrOutcome;

    /// Bumps whenever a page that came back [`SrOutcome::NotReady`] might now
    /// succeed, so the reader retries the pages it had skipped.
    fn epoch(&self) -> u64;

    /// The model [`Upscaler::upscale`] would pick for this page, without running
    /// it — lets the pool reuse a kept model output when only the display size
    /// changed (same model ⇒ same output).
    fn model_for(&self, src_h: u32, gray: bool, target_scale: f32) -> Option<std::sync::Arc<str>>;
}

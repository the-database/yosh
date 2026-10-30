//! yosh-image — the GPU-free imaging core shared by the yosh reader
//! (`yosh-engine`) and batch tools built beside it.
//!
//! - [`decode`]: full-resolution decode of any supported format to 8-bit gray or
//!   RGBA, ICC → sRGB, grayscale detection, channel conversions.
//! - [`resize`]: high-quality resampling — linear-light Dot Gain for gray
//!   (screentone-safe), Lanczos3 for color — to any size, up or down.
//! - [`tone`], [`icc`]: the tone curves and color-management helpers behind them.
//! - [`source`]: page sources over folders and archives (zip/cbz, rar/cbr with
//!   the `rar` feature, 7z/cb7), with legacy-codepage entry-name repair.
//!
//! Pure Rust apart from the optional `rar` feature (bundled UnRAR C++, as in the
//! reader) and `avif` (dav1d); no GPU, windowing or UI dependencies.

pub mod decode;
pub mod icc;
pub mod resize;
pub mod source;
pub mod tone;

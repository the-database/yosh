//! yosh-engine — the reusable decode/render core of the yosh manga reader.
//!
//! This crate holds the platform-agnostic pipeline: the parallel decode-ahead
//! `DecodePool`, the HQ CPU decode/downscale path, the page/texture/cache layer,
//! the prefetch + layout math, and the wgpu device/queue context. It carries no
//! windowing (`winit`) or UI (`egui`) types so it can be reused behind a thin
//! per-platform shell. The `yosh` application crate is the desktop shell.
//!
//! Modules are moved here in stages from `crates/yosh`; this file grows its
//! `pub mod`/re-export surface as each lands.

pub mod layout;
pub mod prefetch;

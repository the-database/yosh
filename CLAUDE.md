# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**yosh** is a from-scratch, high-throughput local manga/comic reader in Rust (`winit` + `wgpu` + `egui`).
The single defining feature is **zero-hitch page turning and continuous scrolling**: a parallel
decode-ahead pipeline keeps display-ready GPU textures buffered ahead of the read position, so a page
change is just a texture swap. Performance is the point — treat the decode/render hot path as
load-bearing and don't regress seek throughput (target ≈ 83 pages/sec; HQ path measures ~110+ pps).

Cargo workspace, three crates:
- `crates/yosh` — the application (everything below lives here).
- `crates/decode_bench`, `crates/present_bench` — throwaway Phase-0 spikes that validated the
  throughput ceiling before the real build. See `SPIKES_RESULTS.md`. Not part of the app.

## Commands

```sh
cargo run -p yosh -- "<path>" [start_page]            # dev (debug; keeps a console for logs)
cargo run --release -p yosh -- "<path>" [start_page]  # release (perf-representative; GUI-subsystem)
cargo build --release -p yosh                          # build the shipping binary
cargo check -p yosh                                    # fast type-check while iterating
cargo test  -p yosh                                    # unit tests (live in layout.rs: spread/RTL pairing math)
cargo test  -p yosh spread                             # run a subset of tests by name substring (e.g. spread_navigation)
```

- `<path>` = a folder of images, or a `.cbz/.zip`, `.cbr/.rar`, or `.7z/.cb7` archive. No arg → library grid / keys overlay.
- **The default build needs no C toolchain** — all decoders are pure-Rust (`png`, `jpeg-decoder`,
  `image`, `qcms`), TLS for self-update is `ureq` over rustls+**ring** (not aws-lc). **Preserve this.**
  The one exception is AVIF, gated behind an off-by-default feature:
  `cargo build --release -p yosh --features avif` (needs `nasm` + `dav1d`).
- No rustfmt/clippy config, no toolchain pin; edition 2024 (uses let-chains). Standard `cargo fmt` / `cargo clippy`.
- Release builds are GUI-subsystem on Windows (`#![cfg_attr(not(debug_assertions), windows_subsystem="windows")]`),
  so no console on double-click; `main.rs::reattach_console()` rebinds stdio to the parent console when
  launched from a terminal. Debug builds keep a normal console.

## Architecture (the big picture)

### The decode-ahead pipeline (the heart of the app)
- **`pool.rs` — `DecodePool`**: N worker threads. Each worker reads → decodes → downscales → uploads a
  page to its own GPU texture, **entirely off the main thread** (wgpu `Device`/`Queue` are `Send+Sync`,
  so workers call `write_texture` themselves). The main thread only swaps in finished textures.
  - The scheduler rebuilds the **nearest-first** job list on *every* navigation (`set_jobs`), so workers
    always grab the highest-priority page relative to the latest position. `poll()` drains finished pages.
  - `inflight` dedups; jobs already running are not re-queued.
- **`source/mod.rs` — `PageSource` trait** (`Send + Sync`): the abstraction the pool pulls from —
  `read_page(index) -> bytes`, `len`, `name`, `modified`. Implementations: `FolderSource`, `ZipSource`
  (parallel reads), `RarSource` + `SevenzSource` (sequential formats: a reader thread decompresses into
  an in-memory map, then reads are served from there).
- **Decode + HQ downscale** (`decode.rs` + `tone.rs` + `icc.rs`): magic-byte routing to png / jpeg-decoder
  / image. This is subtle and deliberate:
  - **Grayscale path** downscales in **true linear light** (sRGB → 16-bit linear → Catmull-Rom resample →
    re-encode through the Dot Gain 20% curve in `tone.rs`). This is what kills halftone-screentone moiré;
    resampling in gamma/perceptual space does **not**. Don't "simplify" it back to a gamma-space resize.
  - **Color path**: Lanczos3, plus `qcms` ICC → sRGB color management (`icc.rs`), applied *before* resize.
  - Color decodes that are *visually* grayscale are detected (`rgba_is_grayscale`) and routed to the gray path.
  - **Adaptive decode resolution = on-screen page height.** The pool reads a shared `target_h` atomic and
    decodes each page to ≈ the display height (quantized, capped to source res), so the CPU resize is the
    *only* downscale and the GPU samples ~1:1 — no second, aliasing downscale at draw time. `app.rs`
    debounces `target_h` changes so page-flipping never re-decodes (only resize/zoom does).
  - A GPU-downscale path exists but is **disabled** (`GPU_DOWNSCALE_ENABLED = false` in `pool.rs`) because a
    single bilinear blit can't match the HQ CPU resize. Kept for a future HQ-GPU rewrite.

### Central state and the frame loop
- **`app.rs` (~1.7k lines) — `State`** is the winit `ApplicationHandler` and owns everything: gpu, egui,
  the `PageSource`, the `DecodePool`, the `PageCache`, nav/layout/zoom/pan/scroll state, and persisted settings.
  - **`render()`** is the per-frame heart: drain the pool, recompute+debounce `target_h`, build draw quads,
    draw pages, then the egui chrome. Re-read this before touching frame behavior.
  - Input: keyboard → `Action` enum via `action_from()` → `apply_action()`. The keymap is the source of truth
    for shortcuts (README/F1 help mirror it).
  - **Two reading modes** gated by `scroll_mode`: discrete page-flip (with `single`/two-page-spread `Layout`)
    vs continuous vertical scroll (anchor page + `top_offset`, `normalize()` rolls the anchor across bounds).
- **`layout.rs`**: spread pairing math — single vs two-page, RTL/LTR, and the spread-pairing parity `offset`
  (key `O`). `view_start/next/prev/view_pages`. This is the unit-tested module.
- **Cache & reuse**: `cache.rs` (bounded `PageCache`), `texpool.rs` (`TexturePool` recycles GPU textures keyed
  by gray/w/h), `prefetch.rs`. `prefetch` re-queues a page if **missing OR its decode target is stale** — stale
  pages re-decode *in place* (old texture keeps showing until the new one lands), so zoom/resize never flash black.
- **`ui.rs`**: egui top bar, F1 help, info overlay, loading spinner, library grid. UI sets request flags that
  `app.rs` consumes *after* the egui frame (don't mutate `State` mid-egui).
- **`config.rs`**: settings + per-volume reading position as JSON. Normal location is
  `%APPDATA%\the-database\yosh` (via `directories`); **portable mode** is selected by a `yosh-portable.txt`
  marker next to the exe → config saved as `yosh-state.json` beside the exe.

### Self-update and Windows packaging
- **`update.rs`**: a startup background thread checks the **public** GitHub Releases API of
  `the-database/yosh` (the canonical repo; the old `yosh-rust` name 301-redirects). If newer, the top bar
  offers a one-click in-place update: download the platform asset, **validate it's a real executable
  (MZ/ELF magic) before** `self_replace`, then relaunch. Works for installed and portable builds.
- **Windows icon/identity**: `build.rs` embeds `assets/yosh.ico` via `winresource` (Explorer/installer icon);
  at runtime `app.rs::bind_exe_icon()` pulls that square multi-res icon back out of the exe (`ExtractIconExW`)
  and binds it as the window/taskbar icon — the bundled `yosh.png` is non-square and the taskbar's large slot
  rejects it. `main.rs` sets an explicit AppUserModelID that the installer shortcut must match.
- **Installer**: `crates/yosh/installer/yosh.iss` (Inno Setup, per-user, no admin, optional file associations).

## Release process (project-specific — follow exactly)

- **Commit authorship**: author/committer **must** be
  `the-database <25811902+the-database@users.noreply.github.com>`, and **no `Co-Authored-By` trailer**
  (this overrides the default Claude Code trailer). Use
  `git -c user.name="the-database" -c user.email="25811902+the-database@users.noreply.github.com" commit ...`.
- **To cut a release**: bump `version` in `crates/yosh/Cargo.toml`, commit, then
  `git tag -a vX.Y.Z -m "..."`. Push `main` and the tag as **two separate commands**
  (`git push origin main` then `git push origin vX.Y.Z`). The tag **must** match the
  `Cargo.toml` version — CI fails the release build otherwise (see below), so don't
  tag without bumping.
- **CI** (`.github/workflows/release.yml`) builds + publishes a GitHub Release (Windows installer + portable
  zip + bare exe, and the Linux binary) **only on `v*` tags**. `workflow_dispatch` builds artifacts **without**
  releasing. Pushing to `main` alone runs **no** CI. **Be sparing with CI runs** — don't tag/dispatch to "test";
  validate locally first.
- The version string the app reports (CLI `--version`, F1 help, self-update compare) comes from
  `CARGO_PKG_VERSION`, so the `Cargo.toml` bump is the single source of truth.

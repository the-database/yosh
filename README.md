# yosh

A lean, high-throughput local manga/comic reader in Rust (`winit` + `wgpu` + `egui`). Its defining
feature is zero-hitch page turning and continuous scrolling, built on a parallel decode-ahead pipeline:
worker threads decode pages full-res, downscale (CPU or GPU), upload to GPU textures, and feed a bounded
ring buffer ahead of the read position, so page changes are just texture swaps.

## Build & run

```sh
cargo run --release -p yosh -- "<path>" [start_page]
```

`<path>` is a folder of images, or a `.cbz/.zip`, `.cbr/.rar`, or `.7z/.cb7` archive. With no argument,
yosh opens the library grid (if a library folder was set) or shows the keys overlay.

Default `cargo build` needs no system libraries (pure-Rust decoders).

## Formats
- Sources: image folders, CBZ/ZIP, CBR/RAR (UnRAR), 7z/CB7.
- Images: PNG, JPEG, WebP, GIF, BMP. AVIF is **opt-in** (see below).

## Controls (press <kbd>F1</kbd> in-app for the full list)
| | |
|---|---|
| Flip | ← → (reading-direction aware), ↑ ↓ / Space / PgUp · PgDn, click left/right half, wheel |
| Home / End | first / last page |
| `S` | single ↔ two-page spread |
| `O` | shift spread pairing (fix wrong dual-page offset) |
| `D` | reading direction RTL ↔ LTR |
| `C` | continuous vertical scroll |
| `F` | fit mode (window / width / height; in scroll: width ↔ height fit) |
| `+` `-` `0` | zoom in / out / reset; drag to pan |
| `T` | present mode vsync ↔ turbo (uncapped) |
| `G` | GPU downscale on/off |
| `F11` | fullscreen |

Reading position (per volume), reading direction, fit, layout, spread offset, and mode are persisted.

## AVIF (optional)
AVIF decode uses the `image` crate's native (dav1d) backend, gated behind an off-by-default feature so
the standard build stays toolchain-free:

```sh
cargo build --release -p yosh --features avif   # requires nasm + dav1d (or a vendored build)
```

## Benchmarks
`crates/decode_bench` and `crates/present_bench` are the throwaway spikes that validated the throughput
ceiling; see `SPIKES_RESULTS.md`.

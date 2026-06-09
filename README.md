# yosh

A lean, high-throughput local manga/comic reader in Rust (`winit` + `wgpu` + `egui`). Its defining
feature is zero-hitch page turning and continuous scrolling, built on a parallel decode-ahead pipeline:
worker threads decode pages full-res, downscale (CPU or GPU), upload to GPU textures, and feed a bounded
ring buffer ahead of the read position, so page changes are just texture swaps.

## Install (Windows)
Grab the latest [release](https://github.com/the-database/yosh/releases):
- **`yosh-setup-x64.exe`** — installer (per-user, **no admin**). Adds a Start Menu shortcut and lets
  you tick which file types open in yosh — **comic archives** (`.cbz .cbr .cb7`), **images**
  (`.png .jpg .jpeg .webp .gif .bmp .avif`), or both. For a type another app already owns (e.g. `.png`
  → Photos), Windows keeps that default until you confirm yosh once via right-click → *Open with* →
  *Choose another app* (or Settings → Apps → Default apps) — installers can't change it silently.
- **`yosh-windows-x64-portable.zip`** — portable build. Unzip anywhere and run `yosh.exe`; settings
  live next to it (`yosh-state.json`), so it leaves nothing behind and travels on a USB stick. (The
  bundled `yosh-portable.txt` marker enables this; delete it to use the normal `%APPDATA%` location.)
- **`yosh-windows-x64.exe`** — the bare single executable (settings in `%APPDATA%`).

The installer/exe are unsigned, so SmartScreen may warn on first run — choose *More info → Run anyway*.

## Build & run

```sh
cargo run --release -p yosh -- "<path>" [start_page]
```

`<path>` is a folder of images, or a `.cbz/.zip`, `.cbr/.rar`, or `.7z/.cb7` archive. With no argument,
yosh opens the library grid (if a library folder was set) or shows the keys overlay.

Default `cargo build` needs no system libraries (pure-Rust decoders).

## Formats
- Sources: image folders, CBZ/ZIP, CBR/RAR (UnRAR), 7z/CB7.
- Images: PNG, JPEG, WebP, GIF, BMP, JPEG XL (.jxl), PSD, ICO, TIFF, TGA, DDS, OpenEXR, Radiance
  HDR, QOI, PNM. AVIF is **opt-in** (see below).
  - Animated GIF and animated WebP play back, with a mini panel for play/pause and frame stepping.
  - ICO files step through their contained resolutions ("layers") in that same panel.
  - PSD shows the flattened composite (8-bit RGB); it's browsable in folders/archives but is **not**
    registered as a default `.psd` handler (a "View with yosh" right-click entry is offered instead).

## Controls (press <kbd>F1</kbd> in-app for the full list)
| | |
|---|---|
| Flip | ← → (reading-direction aware), ↑ ↓ / Space / PgUp · PgDn; click a left/right **edge**; wheel |
| `Home` / `End` | first / last page |
| `[` / `]` | previous / next book (folder or archive) |
| Views | `9` fit window · `8` fit width · `0` 100% (1:1) · `7` two-page L→R · `6` two-page R→L |
| `S` / `D` / `C` / `O` | spread ↔ single · direction RTL↔LTR · continuous scroll · shift spread pairing |
| `R` | rotate 90° clockwise (single-page view; resets on opening a new book) |
| `+` `-` | zoom; drag to pan |
| `E` | show in Explorer (opens the folder and selects the file/archive) |
| `I` | image info overlay |
| `B` | bottom seekbar (reveals near the bottom edge; click/drag to jump) |
| `G` | show/hide the animation panel (animated GIF / WebP) |
| double-click | fullscreen (middle of the page); `F11` also toggles |
| `Esc` | quit |

Reading position (per volume), reading direction, fit, layout, spread offset, scroll,
and seekbar visibility are persisted.

## AVIF (optional)
AVIF decode uses the `image` crate's native (dav1d) backend, gated behind an off-by-default feature so
the standard build stays toolchain-free:

```sh
cargo build --release -p yosh --features avif   # requires nasm + dav1d (or a vendored build)
```

## Benchmarks
`crates/decode_bench` and `crates/present_bench` are the throwaway spikes that validated the throughput
ceiling; see `SPIKES_RESULTS.md`.

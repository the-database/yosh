# Phase 0 — validation spike results

Goal was to confirm the architectural ceiling (208 pages in ~2.5s ≈ **83 pages/sec, 12 ms/page**)
before building the real reader. **Both spikes clear the target decisively.** End-to-end ceiling is
`min(decode, present)` and both sit well above 83/sec.

## Hardware
- CPU: i9-13900K — 24 physical / 32 logical cores
- RAM: 31.7 GB
- GPU: RTX 5090 (wgpu picked the **Vulkan** backend under `Backends::PRIMARY`; D3D12 also present)
- Display during the run: ~145 Hz refresh (Fifo locked to 145). Display adapters include a Parsec
  Virtual Display Adapter; user reads on a real monitor.

## Test asset
`D:\file\Boox Manga Upscaled\この音とまれ！ 4x\この音とまれ！ 13-mangajanai\OPS\images`
- **208 PNG pages, all 3056×4800 (14.7 MP)**; **200 are 8-bit grayscale (colortype 0), 8 are RGB**
  (color pages). ~4.5 MB avg, ~947 MB total. MangaJaNai 4× upscaler output.
- Disk read warm ≈ **3 GB/s** (947 MB in ~0.3 s) — not a bottleneck.
- Implication: **no JPEG DCT-scaled-decode shortcut applies**; pages decode full-res then downscale.
  The pipeline can stay **single-channel R8** end-to-end (decode 1 B/px → 1-ch resize → R8 texture →
  broadcast `.r` in shader), ~4× cheaper than the spec's RGBA assumption.

## Spike 1 — decode throughput (`decode_bench`)
Hand-rolled N-thread pool, per-thread reused buffers, full-res decode + single-channel Bilinear
downscale to 2160px tall. Sweep over N. (`crates/decode_bench`)

| decoder | mode | knee (N≈8) | peak (N≈24–32) |
|---|---|---|---|
| **`png` crate** | decode + resize | 288 pps / 3.5 ms | **316 pps / 3.2 ms** |
| **`png` crate** | decode only | 443 pps / 2.3 ms | **634 pps / 1.6 ms** |
| zune-png | decode + resize | 135 pps (plateaus at N≥8) | — |
| zune-png | decode only | 158 pps (plateaus at N≥6) | — |

**Decisions:**
- **Use the `png` crate** (fdeflate fast path). zune-png is ~2× slower *and stops scaling* past ~8
  threads — counter to its usual reputation, but clear here.
- **Worker pool N ≈ 8–12** — at/near the knee (288–300 pps with resize), leaving cores for present/UI.
  Beyond ~16 threads gains <5%.
- **Resize is expensive**: single-channel Bilinear adds ~1.2–1.6 ms/page and roughly *halves* the
  decode-only ceiling (634 → 316 pps at high N). Strong case to do the **downscale on the GPU** in the
  real app (blit/mip on the RTX 5090) and keep CPU cores fully on decode.
- **Ring-buffer depth ≈ 12–16**: production (≥288 pps) ≫ target consumption (83 pps), so the ring only
  needs to cover scheduling jitter, not sustained deficit.
- Result vs target: **316 pps = 3.8× the 83-pps target** with the full decode+resize pipeline;
  decode-only is 634 pps (7.6×). 208 pages decode+resize in ~0.66 s.

## Spike 2 — present ceiling (`present_bench`)
Pre-decoded 30 pages → R8 textures → cycled a full-screen quad, 4 s per present mode.
(`crates/present_bench`) **All present modes were available** (Fifo/FifoRelaxed/Mailbox/Immediate).

| mode | swaps/s | vs target |
|---|---|---|
| Immediate (uncapped) | **4817** | 58× |
| Mailbox | 2794 | 34× |
| Fifo (vsync) | 145 | 1.8× |

**Decisions:**
- **Present is not the bottleneck.** Immediate sustains 4817/s; even Fifo (vsync, tear-free — what you
  want for reading) hits the display refresh, 145/s here = 1.75× target.
- With vsync on, the end-to-end cap is the **refresh rate** (145/s), since decode (316) > refresh. That
  still beats the 83 target and exceeds what's perceivable as distinct pages while reading. Use
  Immediate only if uncapped fast-forward above refresh is ever wanted (introduces tearing).
- Texture upload of R8 display-res frames is trivial (implied by 4817 swaps/s with per-frame binds).

## Net
Target 83 pps cleared on both axes. End-to-end: decode ~316 pps, present 145 (vsync) – 4817
(immediate). **208-page volume blasts in ~0.66–1.4 s**, beating the 2.5 s reference. Architecture
(parallel decode-ahead + ring buffer + GPU present, single-channel R8) is validated; safe to build the
real reader.

## Re-run
```powershell
$imgs = "D:\file\Boox Manga Upscaled\この音とまれ！ 4x\この音とまれ！ 13-mangajanai\OPS\images"
cargo run --release -p decode_bench  -- "$imgs" --decoder both --resize on
cargo run --release -p decode_bench  -- "$imgs" --decoder png  --resize off   # decode-only ceiling
cargo run --release -p present_bench -- "$imgs" --mode all                      # window on physical monitor
```

## Open follow-ups for the real-app plan
- GPU-side downscale (reclaim the ~50% CPU lost to CPU resize).
- Decode straight from CBZ/ZIP entry bytes (per-thread handles) and from a single sequential RAR reader
  for CBR; the `png`-crate decode path is unchanged.
- Confirm the physical monitor's true refresh for the vsync cap (145 Hz observed here).

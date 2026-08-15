# yosh-android

The Android shell for yosh — a `cdylib` that loads as a NativeActivity and drives
the reusable [`yosh-engine`](../yosh-engine) (decode pipeline + reading machine)
through winit + wgpu, the same way the desktop shell (`crates/yosh`) does.

**Status:** feature-complete for reading. The shell drives the engine `Reader` in
the winit frame loop (poll → decode-view debounce → prefetch → build-quads → draw,
same as the desktop shell), rendering real pages sharp (single-resize invariant
holds — verified converging to a crisp 1:1 render on-device), with tap-zones
(left third = previous, right = next), swipe/pinch-zoom, the SAF picker
(content:// → `ZipSource::from_bytes`), egui chrome (seekbar / library browser)
and `KEEP_SCREEN_ON`. Redraws are **on demand** — the frame loop idles on a
settled, decoded page and wakes on input, animation, or pending decode work (see
the redraw guard at the end of `App::render`). Verified on a physical Pixel 9 Pro
XL and a Lenovo TB321FU tablet (wgpu on Vulkan).

RAR/CBR is unavailable on Android (the bundled UnRAR C++ uses `lutimes`, absent
from Bionic libc), so the engine is built `--no-default-features`. Folder / CBZ /
7z page sources work.

## One-time toolchain setup

```powershell
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
# Android NDK r27c + SDK (build-tools;34.0.0, platform-tools, platforms;android-34,
# emulator, a system image) + a JDK 17. Point build-apk.ps1's paths at your install.
```

## Build + run

```powershell
# cross-compile both ABIs, package + sign a universal debug APK:
./build-apk.ps1
# ...and install + launch on a connected device/emulator:
./build-apk.ps1 -Run
# release build (smaller, optimized):
./build-apk.ps1 -Profile release -Run
```

Packaging is Gradle-free: `cargo-ndk` builds the per-ABI `.so`, then
`aapt2 compile`/`link` (manifest + `res/` → base APK) + `jar` (add `lib/<abi>/`) +
`zipalign` + `apksigner` (debug keystore). See `build-apk.ps1`.

The launcher icon (an adaptive icon — the desktop monkey logo on the book's purple,
plus a legacy PNG fallback) lives under `res/mipmap-*/` and is generated from
`../yosh/assets/yosh.ico` by `./gen-icons.ps1` (ImageMagick). The PNGs are committed;
re-run that script only when the logo changes.

A host `cargo build` over the workspace compiles this crate to an empty cdylib
(its deps are gated to `cfg(target_os = "android")`), so it doesn't affect desktop
builds or CI.

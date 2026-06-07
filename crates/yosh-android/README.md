# yosh-android

The Android shell for yosh — a `cdylib` that loads as a NativeActivity and drives
the reusable [`yosh-engine`](../yosh-engine) (decode pipeline + reading machine)
through winit + wgpu, the same way the desktop shell (`crates/yosh`) does.

**Status:** scaffold. It brings up a winit window + the engine's wgpu `GpuContext`
and clears the surface — verified running on a physical Pixel 9 Pro XL (wgpu on
Vulkan). The real reader (engine `Reader` + draw list, touch input, a `content://`
FD → `ZipSource::from_bytes`, suspend/resume → `Gpu::recreate_surface`, egui
chrome) builds on top of this.

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
`aapt2 link` (manifest → base APK) + `jar` (add `lib/<abi>/`) + `zipalign` +
`apksigner` (debug keystore). See `build-apk.ps1`.

A host `cargo build` over the workspace compiles this crate to an empty cdylib
(its deps are gated to `cfg(target_os = "android")`), so it doesn't affect desktop
builds or CI.

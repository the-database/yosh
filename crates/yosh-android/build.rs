// Link the C++ standard library — but only for the Android build with RAR enabled.
//
// unrar_sys compiles its UnRAR C++ with `cpp_link_stdlib(None)` (to dodge a
// windows-gnu issue), so nothing in the tree pulls in libc++. On desktop the system
// toolchain links it implicitly, but Android's Rust target links libc only — so the
// .so fails with undefined std:: symbols. unrar is the tree's *only* C++ consumer, so
// linking libc++ *statically* keeps libyosh_android.so self-contained (no
// libc++_shared.so to bundle) with no ODR concern. These directives come from the root
// cdylib crate, so they land after libunrar in the link order (definition after use).
//
// Gated on android + the `rar` feature, so the host/desktop builds and the default
// (RAR-free) Android build emit nothing and are untouched.
fn main() {
    let android = matches!(std::env::var("CARGO_CFG_TARGET_OS").as_deref(), Ok("android"));
    let rar = std::env::var_os("CARGO_FEATURE_RAR").is_some();
    if android && rar {
        // libc++_static needs libc++abi (and libunwind, already linked by the target).
        println!("cargo:rustc-link-lib=static=c++_static");
        println!("cargo:rustc-link-lib=static=c++abi");
    }
}

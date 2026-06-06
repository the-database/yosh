// The bundled UnRAR C++ (via the `unrar` -> `unrar_sys` crate) calls Win32
// registry + CryptoAPI functions (Reg*/Crypt*) in pathfn.cpp / crypt.cpp but
// does not declare the advapi32 import lib itself. The `yosh` app happens to
// pull advapi32 in transitively (windows-sys), but the engine links standalone
// — e.g. `cargo test -p yosh-engine` — so request it here for Windows targets.
// A duplicate link directive is harmless for the app build. Gated on the
// *target* OS (not the host) so cross-compiles (e.g. to Android) don't emit it.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-lib=dylib=advapi32");
    }
}

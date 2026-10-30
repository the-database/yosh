// libaji's Windows build resolves paths through ANSI APIs (LoadLibraryExA, GetModuleFileNameA, and
// std::string paths for the engine cache), so the process code page must be UTF-8 for an install
// or cache folder under a non-ASCII user profile to work. Long paths are opted into as well.
fn main() {
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        use embed_manifest::manifest::{ActiveCodePage, Setting};
        embed_manifest::embed_manifest(
            embed_manifest::new_manifest("MangaJaNai.Upscale")
                .active_code_page(ActiveCodePage::Utf8)
                .long_path_aware(Setting::Enabled),
        )
        .expect("unable to embed manifest");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

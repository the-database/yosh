//! Build script, Windows only (a no-op elsewhere): embeds the application icon
//! (Explorer, taskbar, title bar) and the application manifest.
//!
//! The manifest is the one MangaJaNaiCLI embeds. Its UTF-8 active code page is
//! what matters: the AI-upscaling add-on's libaji (older releases) and the
//! TensorRT-RTX runtime it loads take paths through the process code page, so
//! without it an add-on folder whose path the system code page can't spell (a
//! non-ASCII user name, say) fails to load. It also opts into long paths,
//! Common Controls v6 (the file dialogs' current look), per-monitor v2 DPI
//! awareness (what winit sets at startup anyway) and the Windows 10/11
//! compatibility entries.

fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/yosh.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=icon embed failed: {e}");
        }

        use embed_manifest::manifest::{ActiveCodePage, Setting};
        embed_manifest::embed_manifest(
            embed_manifest::new_manifest("yosh")
                .active_code_page(ActiveCodePage::Utf8)
                .long_path_aware(Setting::Enabled),
        )
        .expect("unable to embed manifest");
    }
}

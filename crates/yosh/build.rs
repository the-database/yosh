//! Build script: embed the application icon into the Windows executable so it
//! shows in Explorer, the taskbar, and the window title bar. No-op elsewhere.

fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/yosh.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=icon embed failed: {e}");
        }
    }
}

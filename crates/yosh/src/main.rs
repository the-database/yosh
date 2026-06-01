// yosh — high-throughput local manga/comic reader.
// M1.1: app shell (winit + wgpu + egui).
//
// Release builds are GUI-subsystem (no console window on double-click); a startup
// shim reattaches to the parent console when launched from a terminal so CLI
// stdout/stderr still work. Debug keeps a normal console for dev.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod cache;
mod config;
mod decode;
mod downscale;
mod gpu;
mod layout;
mod library;
mod page;
mod pool;
mod prefetch;
mod source;
mod texpool;
mod ui;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    reattach_console();
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("yosh {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let path = std::env::args().nth(1).map(std::path::PathBuf::from);
    let start_index = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let event_loop = EventLoop::new().expect("create event loop");
    // Poll for now (render every frame). Will switch to Wait + EventLoopProxy wake-ups
    // once the async decode pipeline lands (M1.3).
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(path, start_index);
    event_loop.run_app(&mut app).expect("run app");
}

/// On Windows, when the GUI-subsystem exe is launched from a terminal, attach to
/// the parent console and rebind std handles so stdout/stderr appear there.
/// No-op when double-clicked (no parent console) or in debug (already a console).
#[cfg(windows)]
fn reattach_console() {
    use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Console::{
        AttachConsole, GetStdHandle, SetStdHandle, ATTACH_PARENT_PROCESS, STD_ERROR_HANDLE,
        STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    unsafe {
        // If stdout is already connected (redirected to a file/pipe, or an
        // inherited console), don't touch it — preserves `yosh > out.txt`.
        let existing = GetStdHandle(STD_OUTPUT_HANDLE);
        if !existing.is_null() && existing != INVALID_HANDLE_VALUE {
            return;
        }
        if AttachConsole(ATTACH_PARENT_PROCESS) == 0 {
            return; // no parent console (double-clicked) — leave handles as-is
        }
        let open = |name: &str| {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            CreateFileW(
                wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        let out = open("CONOUT$");
        if out != INVALID_HANDLE_VALUE {
            SetStdHandle(STD_OUTPUT_HANDLE, out);
            SetStdHandle(STD_ERROR_HANDLE, out);
        }
        let inp = open("CONIN$");
        if inp != INVALID_HANDLE_VALUE {
            SetStdHandle(STD_INPUT_HANDLE, inp);
        }
    }
}

#[cfg(not(windows))]
fn reattach_console() {}

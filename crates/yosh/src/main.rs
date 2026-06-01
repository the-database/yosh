// yosh — high-throughput local manga/comic reader.
// M1.1: app shell (winit + wgpu + egui).

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

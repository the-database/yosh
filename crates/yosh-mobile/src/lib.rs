//! The mobile shell for yosh, shared by the platform crates (`crates/yosh-android`).
//!
//! Drives the reusable [`yosh_engine::reader::Reader`] in the winit frame loop —
//! the same poll → decode-view debounce → prefetch → build-quads → draw sequence
//! the desktop shell runs — under touch-first egui chrome: seekbar, library
//! browser, options. Renders real manga pages via the engine's decode pool +
//! `PagePipeline`.
//!
//! Storage: a tap in the top strip opens the library browser; the "Open file…"
//! button there launches the system document picker, whose result the platform
//! turns into a page source (Android: a SAF `content://` file's bytes read off its
//! descriptor into `ZipSource::from_bytes`). With nothing open, an empty-state
//! helper explains how to open a comic.
//! Tap left third = previous page, right two-thirds = next.
//!
//! Everything the shell needs from the OS goes through [`Platform`]; a platform
//! crate implements it, builds a [`Shell`] and runs it on its winit event loop.
//! The desktop shell is `crates/yosh`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use yosh_engine::gesture::{GestureCtx, GestureEvent, Phase, TouchGestures};
use yosh_engine::gpu::GpuContext;
use yosh_engine::i18n::{self, tr, Lang};
use yosh_engine::layout::Layout;
use yosh_engine::page::{FitMode, PagePipeline};
use yosh_engine::pool::{DecodePool, Waker};
use yosh_engine::reader::{Budget, DeviceTier, Direction, Reader, Viewport};
use yosh_engine::source::{is_image_ext, FolderSource, PageSource, SevenzSource, ZipSource};
// RAR/CBR is gated behind the off-by-default `rar` feature (Linux/CI builds only —
// see Cargo.toml). The engine only exposes `RarSource` when its `rar` feature is on.
#[cfg(feature = "rar")]
use yosh_engine::source::RarSource;
use yosh_engine::texpool::TexturePool;

mod platform;
mod shell;
mod input;
mod open;
mod persist;
mod prefs;
mod device;
mod fonts;
mod library;
mod app;
mod ui;

use input::*;
use open::*;
use persist::*;
use prefs::*;
use device::*;
use fonts::*;
use library::*;
use app::*;
use ui::*;

pub use platform::{PickedJob, Platform};
pub use shell::Shell;

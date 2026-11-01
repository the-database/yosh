//! The stdout protocol MangaJaNaiConverterGui reads. The `PROGRESS=` and `TOTALZIP=` lines are the
//! ones the 1.x Python backend printed, so the GUI's progress bars work unchanged; everything else
//! is a log line for its console.

use std::io::Write;

fn line(s: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}

/// A single-file upscale finished.
pub fn image_done() {
    line("PROGRESS=postprocess_worker_image");
}

/// One loose image of a folder upscale finished.
pub fn folder_image_done() {
    line("PROGRESS=postprocess_worker_folder");
}

/// An archive is starting: how many entries it has.
pub fn archive_start(entries: usize) {
    line(&format!("TOTALZIP={entries}"));
}

/// One archive entry was written (upscaled or copied).
pub fn archive_entry_done() {
    line("PROGRESS=postprocess_worker_zip_image");
}

/// An archive finished.
pub fn archive_done() {
    line("PROGRESS=postprocess_worker_zip_archive");
}

/// A model's engine is being built (first use of this model and tile on this GPU). The line names
/// the backend ("TensorRT" or "TensorRT-RTX"); the GUI tells them apart by it.
pub fn engine_build(model: &str, tile: u32, backend: &str) {
    let takes = if backend == "TensorRT" { "a few minutes" } else { "a few seconds" };
    line(&format!("ENGINE_BUILD {model} (tile {tile}): building the {backend} engine for this GPU; this takes {takes}, once"));
}

pub fn engine_ready(model: &str) {
    line(&format!("ENGINE_READY {model}"));
}

/// TensorRT-RTX failed and TensorRT runs instead: at start (`model` None, for the whole run) or
/// for one model (the others stay on TensorRT-RTX). The GUI shows it as a warning.
pub fn backend_fallback(model: Option<&str>, why: &str) {
    match model {
        None => line(&format!("BACKEND_FALLBACK TensorRT-RTX failed ({why}); using TensorRT instead")),
        Some(m) => line(&format!("BACKEND_FALLBACK TensorRT-RTX failed for {m} ({why}); using TensorRT for it")),
    }
}

/// A file (or archive entry) that failed; the run continues with the next one.
pub fn error(what: &str, msg: &str) {
    line(&format!("ERROR {what}: {msg}"));
}

pub fn log(msg: &str) {
    line(msg);
}

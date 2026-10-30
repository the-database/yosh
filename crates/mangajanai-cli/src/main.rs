//! mangajanai_upscale — MangaJaNaiConverterGui's upscaling backend (it replaces 1.x's PyTorch
//! backend). Images and archives are decoded, matched to a workflow chain, upscaled through
//! libaji (TensorRT or DirectML), resized to the target and encoded.
//!
//!   mangajanai_upscale --settings <appstate2.json> [options]   the GUI's selected workflow
//!   mangajanai_upscale (-f <file> | -d <folder>) [-o <out>] [-u <1-4>] [options]
//!                                                              the default workflow, from the command line
//!   mangajanai_upscale --list-devices [--backend tensorrt|directml] [--aji-dir <dir>]
//!
//! Options:
//!   --aji-dir <dir>      where aji.dll and its backends are (default: this exe's folder)
//!   --backend <b>        auto (default) | tensorrt | directml
//!   --device <id>        a device id from --list-devices (default: the backend's first device)
//!   --cache-dir <dir>    TensorRT engine cache (default: next to each model)
//!   -m, --models-dir <d> where models are looked up (repeatable; default: ../models)

mod chain;
mod encode;
mod levels;
mod process;
mod progress;
mod run;
mod settings;
mod upscaler;

use std::path::PathBuf;

use settings::{Chain, Workflow};
use upscaler::{BackendChoice, Upscaler};

struct Args {
    settings: Option<PathBuf>,
    file: Option<PathBuf>,
    folder: Option<PathBuf>,
    out: Option<PathBuf>,
    scale: i32,
    list_devices: bool,
    aji_dir: PathBuf,
    backend: BackendChoice,
    device: u64,
    cache_dir: Option<PathBuf>,
    models_dirs: Vec<PathBuf>,
}

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_default()
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        settings: None,
        file: None,
        folder: None,
        out: None,
        scale: 2,
        list_devices: false,
        aji_dir: exe_dir(),
        backend: BackendChoice::Auto,
        device: 0,
        cache_dir: None,
        models_dirs: Vec::new(),
    };
    let mut it = std::env::args_os().skip(1);
    while let Some(arg) = it.next() {
        let arg = arg.to_string_lossy().into_owned();
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--settings" => a.settings = Some(value(&arg)?.into()),
            "-f" | "--file-path" => a.file = Some(value(&arg)?.into()),
            "-d" | "--folder-path" => a.folder = Some(value(&arg)?.into()),
            "-o" | "--output-folder-path" => a.out = Some(value(&arg)?.into()),
            "-u" | "--upscale-factor" => {
                a.scale = value(&arg)?.to_string_lossy().parse().map_err(|_| "--upscale-factor must be 1-4".to_string())?;
            }
            "-m" | "--models-dir" | "--models-directory" => a.models_dirs.push(value(&arg)?.into()),
            "--list-devices" => a.list_devices = true,
            "--aji-dir" => a.aji_dir = value(&arg)?.into(),
            "--backend" => a.backend = BackendChoice::parse(&value(&arg)?.to_string_lossy())?,
            "--device" => a.device = value(&arg)?.to_string_lossy().parse().map_err(|_| "--device must be a device id".to_string())?,
            "--cache-dir" => a.cache_dir = Some(value(&arg)?.into()),
            "--version" | "-V" => {
                println!("mangajanai_upscale {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                let help: Vec<&str> = include_str!("main.rs")
                    .lines()
                    .take_while(|l| l.starts_with("//!"))
                    .map(|l| l.trim_start_matches("//!").strip_prefix(' ').unwrap_or(l.trim_start_matches("//!")))
                    .collect();
                println!("{}", help.join("\n"));
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}' (see --help)")),
        }
    }
    Ok(a)
}

/// The default workflow's chains (MangaJaNaiConverterGui's "Upscale Manga (Default)"), for the
/// command-line mode: IllustrationJaNai for color pages; for gray pages the MangaJaNai model trained
/// nearest the page height, 2x up to a 2x target and 4x above.
fn default_chains() -> Vec<Chain> {
    let chain = |min: &str, max: &str, gray: bool, min_s: i32, max_s: i32, model: String| Chain {
        min_resolution: min.into(),
        max_resolution: max.into(),
        is_grayscale: gray,
        is_color: !gray,
        min_scale_factor: Some(min_s),
        max_scale_factor: Some(max_s),
        model_file_path: model,
        auto_adjust_levels: gray,
        ..Chain::default()
    };
    let mut chains = vec![
        chain("0x0", "0x0", false, 0, 2, "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx".into()),
        chain("0x0", "0x0", false, 2, 0, "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx".into()),
    ];
    for (lo, hi, bucket, k2, k4) in [
        ("0x0", "0x1250", "1200p", "70k", "70k"),
        ("0x1251", "0x1350", "1300p", "75k", "75k"),
        ("0x1351", "0x1450", "1400p", "70k", "105k"),
        ("0x1451", "0x1550", "1500p", "90k", "105k"),
        ("0x1551", "0x1760", "1600p", "90k", "70k"),
        ("0x1761", "0x1984", "1920p", "70k", "105k"),
        ("0x1985", "0x0", "2048p", "95k", "70k"),
    ] {
        chains.push(chain(lo, hi, true, 0, 2, format!("2x_MangaJaNai_{bucket}_V1_ESRGAN_{k2}.onnx")));
        chains.push(chain(lo, hi, true, 2, 0, format!("4x_MangaJaNai_{bucket}_V1_ESRGAN_{k4}.onnx")));
    }
    for (i, c) in chains.iter_mut().enumerate() {
        c.chain_number = (i + 1).to_string();
    }
    chains
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    if args.list_devices {
        use yosh_image::aji::Backend;
        let backend = match args.backend {
            BackendChoice::DirectMl => Backend::DirectMl,
            _ => Backend::TensorRt,
        };
        match upscaler::devices_json(&args.aji_dir, backend) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let mut models_dirs = args.models_dirs.clone();
    let (workflow, input) = if let Some(path) = &args.settings {
        let state = match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|s| settings::AppState::parse(&s)) {
            Ok(s) => s,
            Err(e) => {
                progress::error(&path.display().to_string(), &e);
                std::process::exit(1);
            }
        };
        if let Some(dir) = state.models_directory.as_deref().filter(|d| !d.is_empty()) {
            models_dirs.push(dir.into());
        }
        let wf = match state.current_workflow() {
            Ok(w) => w.clone(),
            Err(e) => {
                progress::error("settings", &e);
                std::process::exit(1);
            }
        };
        let input = match wf.selected_tab_index {
            0 => run::Input::File(wf.input_file_path.clone().into()),
            _ => run::Input::Folder { dir: wf.input_folder_path.clone().into(), images: wf.upscale_images, archives: wf.upscale_archives },
        };
        (wf, input)
    } else if let Some(input) = args
        .file
        .clone()
        .map(run::Input::File)
        .or(args.folder.clone().map(|dir| run::Input::Folder { dir, images: true, archives: true }))
    {
        let wf = Workflow {
            upscale_scale_factor: args.scale.clamp(1, 4),
            output_folder_path: args.out.clone().unwrap_or_else(|| "out".into()).to_string_lossy().into_owned(),
            chains: settings::Values { values: default_chains() },
            ..Workflow::default()
        };
        (wf, input)
    } else {
        eprintln!("nothing to do: give --settings, -f or -d (see --help)");
        std::process::exit(2);
    };
    models_dirs.push(exe_dir().join("..").join("models"));

    let output = run::Output {
        dir: workflow.output_folder_path.clone().into(),
        filename: if workflow.output_filename.is_empty() { "%filename%".into() } else { workflow.output_filename.clone() },
        overwrite: workflow.overwrite_existing_files,
    };
    let job = process::Job {
        chains: workflow.chains.values.clone(),
        threshold: workflow.grayscale_detection_threshold,
        target: workflow.target(),
        format: workflow.format(),
        quality: workflow.quality(),
        lossless: workflow.use_lossless_compression,
    };

    let start = std::time::Instant::now();
    let (aji_dir, backend, device, cache_dir) = (args.aji_dir.clone(), args.backend, args.device, args.cache_dir.clone());
    run::run(input, output, job, move || Upscaler::open(&aji_dir, backend, device, cache_dir, models_dirs));
    progress::log(&format!("Elapsed time: {:.2} seconds", start.elapsed().as_secs_f64()));
}

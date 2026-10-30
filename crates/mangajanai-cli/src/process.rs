//! One image through the pipeline: [`prepare`] (decode, gray detection, chain, pre-resize, levels)
//! → [`infer`] (the model) → [`finish`] (final resize, encode).

use yosh_image::decode::{self, is_grayscale, is_opaque, rgba_to_luma, rgba_to_rgb};
use yosh_image::resize::{self, Resizer};

use crate::chain;
use crate::encode::{self, Out};
use crate::levels;
use crate::settings::{Chain, Format, Target};
use crate::upscaler::Upscaler;

/// The workflow settings every image needs.
pub struct Job {
    pub chains: Vec<Chain>,
    pub threshold: i32,
    pub target: Target,
    pub format: Format,
    pub quality: u8,
    pub lossless: bool,
}

/// An image between stages: 8-bit interleaved, `channels` 1 (gray), 2 (gray + alpha), 3 (RGB) or 4
/// (RGBA, straight alpha).
pub struct Work {
    pub w: u32,
    pub h: u32,
    pub channels: u8,
    pub pixels: Vec<u8>,
    /// The decoded size, before any resize: what the chain and the final size go by.
    pub orig: (u32, u32),
    pub chain: Option<Chain>,
}

fn luma_alpha(rgba: &[u8]) -> Vec<u8> {
    let luma = rgba_to_luma(rgba);
    luma.iter().zip(rgba.as_chunks::<4>().0).flat_map(|(&y, p)| [y, p[3]]).collect()
}

fn describe(c: &Chain) -> String {
    let n = if c.chain_number.is_empty() { String::new() } else { format!(" {}", c.chain_number) };
    format!("chain{n}: {}", if c.model_file_path.is_empty() { "No Model" } else { &c.model_file_path })
}

/// Decode, detect gray, pick the chain, and apply its pre-upscale steps. A gray page handled by a
/// chain becomes one luma channel (plus alpha if it has any); an opaque color page drops alpha.
pub fn prepare(bytes: &[u8], job: &Job, name: &str, resizer: &mut Resizer) -> Result<Work, String> {
    let img = decode::decode(bytes)?;
    let (w, h) = (img.w, img.h);
    let stored_gray = img.channels == 1;
    let opaque = stored_gray || is_opaque(&img.pixels);
    let gray = stored_gray || is_grayscale(&img.pixels, job.threshold);
    let scale = chain::required_scale(job.target, (w, h));
    let chain = chain::select(&job.chains, (w, h), gray, scale).cloned();
    match &chain {
        Some(c) => crate::progress::log(&format!("{name}: {w}x{h} {}, {scale:.3}x, {}", if gray { "gray" } else { "color" }, describe(c))),
        None => crate::progress::log(&format!("{name}: {w}x{h} {}, no chain matches; resizing only", if gray { "gray" } else { "color" })),
    }

    let (channels, mut pixels) = if stored_gray {
        (1, img.pixels)
    } else if chain.is_some() && gray {
        if opaque { (1, rgba_to_luma(&img.pixels)) } else { (2, luma_alpha(&img.pixels)) }
    } else if opaque {
        (3, rgba_to_rgb(&img.pixels))
    } else {
        (4, img.pixels)
    };
    let (mut cw, mut ch) = (w, h);

    if let Some(c) = &chain {
        if let Some((tw, th)) = chain::pre_resize(c, (w, h)) {
            pixels = resize::lanczos(&pixels, channels, (w, h), (tw, th), resizer)?;
            (cw, ch) = (tw, th);
        }
        if c.auto_adjust_levels && channels <= 2 {
            let n = channels as usize;
            let luma: Vec<u8> = pixels.iter().step_by(n).copied().collect();
            let (black, white) = levels::find_levels(&luma);
            crate::progress::log(&format!("Auto adjusted levels: new black level = {black}; new white level = {white}"));
            if n == 1 {
                levels::apply(&mut pixels, black, white);
            } else {
                let mut l = luma;
                levels::apply(&mut l, black, white);
                for (px, v) in pixels.as_chunks_mut::<2>().0.iter_mut().zip(l) {
                    px[0] = v;
                }
            }
        }
    }
    Ok(Work { w: cw, h: ch, channels, pixels, orig: (w, h), chain })
}

/// Run the chain's model, if it has one.
pub fn infer(mut work: Work, up: &mut Upscaler) -> Result<Work, String> {
    let Some(c) = &work.chain else { return Ok(work) };
    let model = c.model_file_path.trim();
    if model.is_empty() || model == "No Model" {
        return Ok(work);
    }
    let path = up.resolve(model).ok_or_else(|| format!("model not found: {model}"))?;
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")) {
        return Err(format!("{model} is not an ONNX model; convert it to ONNX to use it with MangaJaNaiConverterGui 2.0"));
    }
    let tile = chain::tile_size(&c.model_tile_size);
    let (out, scale) = if work.channels == 2 || work.channels == 4 {
        up.upscale_with_alpha(&path, tile, work.w, work.h, work.channels, &work.pixels)?
    } else {
        up.upscale(&path, tile, work.w, work.h, work.channels, &work.pixels)?
    };
    work.w *= scale;
    work.h *= scale;
    work.pixels = out;
    Ok(work)
}

/// Bring the image to its final size and encode it. Gray goes through the linear-light Dot Gain
/// resize (screentone-safe), color through Lanczos3; alpha always through Lanczos3.
pub fn finish(work: Work, job: &Job, resizer: &mut Resizer) -> Result<Vec<u8>, String> {
    let Work { w, h, channels, mut pixels, orig, .. } = work;
    let (mut fw, mut fh) = (w, h);
    if let Some((tw, th)) = chain::final_size(job.target, orig, (w, h)) {
        pixels = match channels {
            1 => resize::gray_linear(&pixels, (w, h), (tw, th), resizer)?,
            2 => {
                let luma: Vec<u8> = pixels.iter().step_by(2).copied().collect();
                let alpha: Vec<u8> = pixels.iter().skip(1).step_by(2).copied().collect();
                let luma = resize::gray_linear(&luma, (w, h), (tw, th), resizer)?;
                let alpha = resize::lanczos(&alpha, 1, (w, h), (tw, th), resizer)?;
                luma.into_iter().zip(alpha).flat_map(|(y, a)| [y, a]).collect()
            }
            c => resize::lanczos(&pixels, c, (w, h), (tw, th), resizer)?,
        };
        (fw, fh) = (tw, th);
    }
    encode::encode(&Out { w: fw, h: fh, channels, pixels }, job.format, job.quality, job.lossless)
}

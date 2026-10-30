//! The workflow settings, read from the GUI's saved state (`appstate2.json`, written by Newtonsoft
//! with `TypeNameHandling.All`: lists are `{"$type": …, "$values": […]}`) — the same keys the 1.x
//! Python backend read. Anything not listed here is ignored.

use serde::Deserialize;

#[derive(Deserialize, Default, Debug, Clone)]
pub struct Values<T> {
    #[serde(rename = "$values", default = "Vec::new")]
    pub values: Vec<T>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
pub struct AppState {
    pub workflows: Values<Workflow>,
    #[serde(default)]
    pub selected_workflow_index: usize,
    #[serde(default)]
    pub models_directory: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "PascalCase", default)]
pub struct Workflow {
    pub webp_selected: bool,
    pub png_selected: bool,
    pub avif_selected: bool,
    pub grayscale_detection_threshold: i32,
    pub mode_scale_selected: bool,
    pub upscale_scale_factor: i32,
    pub mode_width_selected: bool,
    pub resize_width_after_upscale: Option<i32>,
    pub mode_height_selected: bool,
    pub resize_height_after_upscale: Option<i32>,
    pub display_device_width: i32,
    pub display_device_height: i32,
    pub selected_tab_index: i32,
    pub input_file_path: String,
    pub input_folder_path: String,
    pub output_folder_path: String,
    pub output_filename: String,
    pub upscale_images: bool,
    pub upscale_archives: bool,
    pub overwrite_existing_files: bool,
    pub lossy_compression_quality: Option<i32>,
    pub use_lossless_compression: bool,
    pub chains: Values<Chain>,
}

impl Default for Workflow {
    fn default() -> Self {
        Self {
            webp_selected: true,
            png_selected: false,
            avif_selected: false,
            grayscale_detection_threshold: 12,
            mode_scale_selected: true,
            upscale_scale_factor: 4,
            mode_width_selected: false,
            resize_width_after_upscale: None,
            mode_height_selected: false,
            resize_height_after_upscale: None,
            display_device_width: 0,
            display_device_height: 0,
            selected_tab_index: 0,
            input_file_path: String::new(),
            input_folder_path: String::new(),
            output_folder_path: String::new(),
            output_filename: "%filename%-mangajanai".into(),
            upscale_images: true,
            upscale_archives: true,
            overwrite_existing_files: false,
            lossy_compression_quality: Some(80),
            use_lossless_compression: false,
            chains: Values::default(),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "PascalCase", default)]
pub struct Chain {
    pub chain_number: String,
    pub min_resolution: String,
    pub max_resolution: String,
    pub is_grayscale: bool,
    pub is_color: bool,
    pub min_scale_factor: Option<i32>,
    pub max_scale_factor: Option<i32>,
    pub model_file_path: String,
    pub model_tile_size: String,
    pub auto_adjust_levels: bool,
    pub resize_height_before_upscale: Option<i32>,
    pub resize_width_before_upscale: Option<i32>,
    pub resize_factor_before_upscale: Option<f64>,
}

impl Default for Chain {
    fn default() -> Self {
        Self {
            chain_number: String::new(),
            min_resolution: "0x0".into(),
            max_resolution: "0x0".into(),
            is_grayscale: false,
            is_color: false,
            min_scale_factor: Some(0),
            max_scale_factor: Some(0),
            model_file_path: String::new(),
            model_tile_size: "Auto (Estimate)".into(),
            auto_adjust_levels: false,
            resize_height_before_upscale: Some(0),
            resize_width_before_upscale: Some(0),
            resize_factor_before_upscale: Some(100.0),
        }
    }
}

/// Output image format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Webp,
    Png,
    Avif,
    Jpeg,
}

impl Format {
    /// The file extension, as 1.x wrote it (`.jpeg`, so its outputs are still recognized).
    pub fn ext(self) -> &'static str {
        match self {
            Format::Webp => "webp",
            Format::Png => "png",
            Format::Avif => "avif",
            Format::Jpeg => "jpeg",
        }
    }
}

/// What the final image size is relative to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Scale(f64),
    Width(u32),
    Height(u32),
    /// Fit within a display (width, height), orientation already applied by the GUI.
    Fit(u32, u32),
}

impl Workflow {
    pub fn format(&self) -> Format {
        if self.webp_selected {
            Format::Webp
        } else if self.png_selected {
            Format::Png
        } else if self.avif_selected {
            Format::Avif
        } else {
            Format::Jpeg
        }
    }

    pub fn target(&self) -> Target {
        let positive = |v: Option<i32>| v.filter(|&v| v > 0).map(|v| v as u32);
        if self.mode_scale_selected {
            Target::Scale(self.upscale_scale_factor.max(1) as f64)
        } else if self.mode_width_selected {
            positive(self.resize_width_after_upscale).map_or(Target::Scale(1.0), Target::Width)
        } else if self.mode_height_selected {
            positive(self.resize_height_after_upscale).map_or(Target::Scale(1.0), Target::Height)
        } else {
            match (positive(Some(self.display_device_width)), positive(Some(self.display_device_height))) {
                (Some(w), Some(h)) => Target::Fit(w, h),
                _ => Target::Scale(1.0),
            }
        }
    }

    pub fn quality(&self) -> u8 {
        self.lossy_compression_quality.unwrap_or(80).clamp(0, 100) as u8
    }
}

impl AppState {
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("could not read the settings: {e}"))
    }

    pub fn current_workflow(&self) -> Result<&Workflow, String> {
        self.workflows
            .values
            .get(self.selected_workflow_index)
            .ok_or_else(|| format!("no workflow {} in the settings", self.selected_workflow_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_newtonsoft_state() {
        let json = r#"{
            "$type": "MangaJaNaiConverterGui.ViewModels.MainWindowViewModel, MangaJaNaiConverterGui",
            "SelectedWorkflowIndex": 1,
            "ModelsDirectory": "C:\\models",
            "Workflows": { "$type": "Avalonia.Collections.AvaloniaList`1[[...]], Avalonia.Base", "$values": [
                { "$type": "W", "WorkflowName": "a" },
                { "$type": "W", "ModeHeightSelected": true, "ModeScaleSelected": false, "ResizeHeightAfterUpscale": 2160,
                  "PngSelected": true, "WebpSelected": false, "LossyCompressionQuality": null,
                  "Chains": { "$type": "L", "$values": [
                    { "$type": "C", "MinResolution": "0x1251", "MaxResolution": "0x1350", "IsGrayscale": true,
                      "MaxScaleFactor": 2, "ModelFilePath": "2x_m.onnx", "ResizeFactorBeforeUpscale": 100.0 }
                  ] } }
            ] }
        }"#;
        let s = AppState::parse(json).unwrap();
        let w = s.current_workflow().unwrap();
        assert_eq!(w.format(), Format::Png);
        assert_eq!(w.target(), Target::Height(2160));
        assert_eq!(w.quality(), 80);
        assert_eq!(w.chains.values.len(), 1);
        assert_eq!(w.chains.values[0].max_scale_factor, Some(2));
        assert_eq!(w.chains.values[0].min_scale_factor, Some(0), "missing keys take the defaults");
        assert_eq!(s.models_directory.as_deref(), Some("C:\\models"));
    }
}

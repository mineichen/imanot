use std::io;

use log::info;

use super::ImageViewerApp;
use crate::{ImageCallbackMap, app::ToolRegistry};

pub fn run_native(mappers: ImageCallbackMap) -> Result<(), eframe::Error> {
    env_logger::init();

    let config = match std::fs::File::open("config.json") {
        Ok(f) => serde_json::from_reader(f).map_err(|e| eframe::Error::AppCreation(Box::new(e)))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => crate::config::Config::default(),
        Err(e) => return Err(eframe::Error::AppCreation(Box::new(e))),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(config.egui.viewport),
        ..Default::default()
    };

    let image_dir = std::env::args().nth(1).unwrap_or_else(|| {
        config
            .image_dir
            .as_ref()
            .and_then(|s| Some(s.to_str()?.to_string()))
            .unwrap_or_else(|| ".".into())
    });

    info!("Run with config: {config:?}");
    info!("Image directory: {}", image_dir);
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| {
            Ok(Box::new(ImageViewerApp::new(
                Box::new(crate::FileStorage::new(image_dir)),
                ToolRegistry::from(&config),
                super::MaskGenerator::new(mappers),
            )))
        }),
    )
}

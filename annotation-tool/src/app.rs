use std::io;

use crate::storage::Storage;
use egui::{self, InnerResponse, UiBuilder};
use imanot::{AsyncRefTask, AsyncTask, ImageData, ImageViewerInteraction, State, Tools};

use image_selector::ImageSelector;
use tools::ToolRegistry;

mod config;
mod image_selector;
mod mask_generator;
mod menu;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod tools;
#[cfg(target_arch = "wasm32")]
mod web;

pub(crate) use config::Config;
pub(crate) use mask_generator::MaskGenerator;
#[cfg(not(target_arch = "wasm32"))]
pub use native::run_native;
#[cfg(target_arch = "wasm32")]
pub use web::run_web;

type ImageLoadTask = AsyncTask<io::Result<ImageData>>;

pub(crate) struct ImageViewerApp {
    storage: Box<dyn Storage>,
    selector: ImageSelector,
    /// Image selected by the user, handed to `state` as soon as it is loaded
    image_loader: Option<ImageLoadTask>,
    state: State,
    tool_registry: ToolRegistry,
    save_job: AsyncRefTask<Result<(), String>>,
    mask_generator: MaskGenerator,
}
impl ImageViewerApp {
    pub fn new(
        storage: Box<dyn Storage>,
        tool_registry: ToolRegistry,
        mask_generator: MaskGenerator,
    ) -> Self {
        let url_loader = Some(AsyncTask::new(storage.list_images()));
        let state = State::new(Tools::new(
            tool_registry.new_primary(),
            tool_registry.new_secondary(),
        ));

        Self {
            storage,
            selector: ImageSelector::new(url_loader),
            image_loader: None,
            state,
            tool_registry,
            save_job: AsyncRefTask::new_ready(Ok(())),
            mask_generator,
        }
    }
}

impl eframe::App for ImageViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Ctrl/Cmd+Plus/Minus is reserved for image zoom
        // so disable egui's own global UI-scale shortcut on the same keys.
        ui.ctx().options_mut(|o| o.zoom_with_keyboard = false);
        ui.input(|i| {
            if i.modifiers.command {
                if i.key_pressed(egui::Key::Plus) {
                    self.state.viewer.modify_zoom(|x| x / 1.1);
                } else if i.key_pressed(egui::Key::Minus) {
                    self.state.viewer.modify_zoom(|x| x / (1.0 / 1.1));
                }
            }
        });

        ui.heading("Image pixel selector");
        self.menu_ui(ui);
        self.selector.update();

        let response = ui.reserve_bottom_space(80., |ui| self.state.ui(ui));
        if let Some(x) = response.inner {
            let ImageViewerInteraction {
                original_image_size,
                cursor_image_pos,
                ..
            } = x;
            ui.label(format!(
                "Original Size: ({original_image_size:?}), \navail: {:?}, \nspacing: {:?}",
                original_image_size,
                ui.spacing().item_spacing
            ));

            if let Some((x, y)) = cursor_image_pos {
                ui.label(format!("Pixel Coordinates: ({}, {})", x, y,));
            }
        }
    }
}

trait UiExt {
    fn reserve_bottom_space<T>(&mut self, size: f32, inner: impl FnOnce(&mut egui::Ui) -> T) -> T;
}

impl UiExt for egui::Ui {
    fn reserve_bottom_space<T>(&mut self, size: f32, inner: impl FnOnce(&mut egui::Ui) -> T) -> T {
        let mut available = self.available_rect_before_wrap();
        available.max.y = (available.max.y - size).max(0.);

        let r = self.scope_builder(UiBuilder::new().max_rect(available), inner);

        let InnerResponse { inner, .. } = r;
        inner
    }
}

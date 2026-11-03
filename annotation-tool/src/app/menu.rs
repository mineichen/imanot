use egui::Key;
use futures::FutureExt;
use imanot::{
    AsyncRefTask, AsyncTask, ImageState, ImageStateLoaded, MaskActionBuilder, MaskDefaultActions,
};
use log::info;

// const ICON_SAM: &str = "\u{2728}";
const ICON_SAVE: &str = "\u{1F4BE}";

impl crate::app::ImageViewerApp {
    pub(super) fn menu_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let is_image_dirty = matches!(
                &self.state.image_state,
                ImageState::Loaded(ImageStateLoaded {
                    masks,
                    ..
                }) if masks.is_dirty()
            );
            ui.scope(|ui| {
                if is_image_dirty {
                    ui.disable();
                }
                if let Some(id) = self.selector.ui(&*self.storage, ui) {
                    self.image_loader = Some(AsyncTask::new(self.storage.load_image(&id).boxed()));
                }
            });

            // The previous image stays visible until the selected one is loaded
            if let Some(loader) = &mut self.image_loader
                && let Some(loaded) = loader.data()
            {
                self.image_loader = None;
                match loaded {
                    // Explicit navigation must not be held back by a tool's selection
                    Ok(data) => self.state.set_image_now(data),
                    Err(e) => {
                        self.state.image_state = ImageState::Error(format!("IO Error: {e}"));
                    }
                }
            }

            if let (
                Some(last_save),
                ImageState::Loaded(ImageStateLoaded {
                    id, masks, image, ..
                }),
            ) = (self.save_job.data(), &mut self.state.image_state)
            {
                if let Err(e) = last_save {
                    ui.label(e.as_str());
                }
                ui.scope(|ui| {
                    if !is_image_dirty {
                        ui.disable();
                    }
                    if ui
                        .button(ICON_SAVE)
                        .on_hover_text("Save (cmd + S)")
                        .clicked()
                        || ui.input(|i| {
                            i.modifiers.command && i.key_pressed(Key::S) && ui.is_enabled()
                        })
                    {
                        masks.mark_not_dirty();
                        self.save_job = AsyncRefTask::new(
                            self.storage
                                .store_masks(id, masks.subgroups_stack())
                                .map(|x| x.map_err(|e| format!("Error during save: {e}")))
                                .boxed(),
                        );
                    }
                });

                if ui.button("Reset").clicked() {
                    masks.reset();
                }

                if let Some(x) = self.mask_generator.ui(&image.original, ui) {
                    info!("Add {} groups", x.len());
                    for group in x {
                        masks.keep_overlapping(false).add(group);
                    }
                }
            }

            match &mut self.state.image_state {
                ImageState::Loaded(ImageStateLoaded { image, .. }) => {
                    super::tools::ui(ui, image, &mut self.tool_registry, &mut self.state.tools);
                }
                ImageState::Error(error) => {
                    ui.label(format!("Error: {error}"));
                }
                _ => (),
            }
        });
    }
}

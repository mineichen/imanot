use std::{io, num::NonZeroU32};

use egui::{
    self, Color32, ColorImage, ImageSource, TextureHandle, TextureOptions, load::SizedTexture,
};
use futures::FutureExt;

use crate::{
    AsyncTask, History, HistoryStrategy, ImageData, ImageId, ImageLoadOk, MaskImage, Tools,
};

#[expect(
    clippy::large_enum_variant,
    reason = "Loaded holds image, mask and history state"
)]
pub enum ImageState {
    NotLoaded,
    LoadingImageData(AsyncTask<io::Result<ImageData>>),
    Loaded(ImageStateLoaded),
    Error(String),
}

impl ImageState {
    pub fn sources(
        &mut self,
        ctx: &egui::Context,
    ) -> impl Iterator<Item = ImageSource<'static>> + '_ {
        match self {
            ImageState::Loaded(x) => itertools::Either::Left(x.sources(ctx)),
            _ => itertools::Either::Right(std::iter::empty()),
        }
    }
    pub(crate) fn set_image_data(
        &mut self,
        i: ImageData,
        ctx: &egui::Context,
        tools: &mut Tools,
    ) -> Result<(), TextureExceedsLimit> {
        if let ImageState::Loaded(existing) = &mut *self {
            tools.load(&i.image);
            existing.set_image_data(i, ctx)
        } else {
            if i.history_strategy == HistoryStrategy::Keep {
                log::warn!("Replace from other state than loading... Dropped Masks");
            }
            *self = ImageState::new_with_image_data(i);
            Ok(())
        }
    }

    pub fn new_with_image_data(image_data: ImageData) -> Self {
        let fut = std::future::ready(Ok(image_data));
        Self::LoadingImageData(AsyncTask::new(fut.boxed()))
    }

    pub fn update(&mut self, ctx: &egui::Context, mut on_image_load: impl FnMut(&ImageLoadOk)) {
        match self {
            ImageState::NotLoaded => {}
            ImageState::LoadingImageData(t) => {
                if let Some(image_data_result) = t.data() {
                    let load_result = image_data_result
                        .map_err(|e| format!("IO Error: {}", e))
                        .and_then(|i| {
                            ImageStateLoaded::from_image_data(i, ctx).map_err(|e| e.to_string())
                        });
                    *self = match load_result {
                        Ok(loaded) => {
                            on_image_load(&loaded.image);
                            ImageState::Loaded(loaded)
                        }
                        Err(e) => ImageState::Error(e),
                    }
                }
            }
            ImageState::Loaded(ImageStateLoaded { masks, .. }) => {
                masks.handle_events(ctx);
            }
            ImageState::Error(_error) => {}
        }
    }
}

pub struct ImageStateLoaded {
    pub id: ImageId,
    pub texture: (TextureHandle, ImageSource<'static>),
    pub image: ImageLoadOk,
    pub masks: MaskImage,
}

impl ImageStateLoaded {
    pub fn from_image_data(i: ImageData, ctx: &egui::Context) -> Result<Self, TextureExceedsLimit> {
        let (width, height) = i.image.adjust.dimensions();
        let texture = Self::create_texture(&i.image, ctx)?;

        Ok(ImageStateLoaded {
            id: i.id,
            image: i.image,
            texture,
            masks: MaskImage::new(
                [width.get() as usize, height.get() as usize],
                i.masks,
                Default::default(),
            ),
        })
    }
    pub fn sources(
        &mut self,
        ctx: &egui::Context,
    ) -> impl Iterator<Item = ImageSource<'static>> + '_ {
        std::iter::once(self.texture.1.clone()).chain(self.masks.sources(ctx))
    }
    fn set_image_data(
        &mut self,
        i: ImageData,
        ctx: &egui::Context,
    ) -> Result<(), TextureExceedsLimit> {
        self.id = i.id;
        self.texture = Self::create_texture(&i.image, ctx)?;
        self.image = i.image;
        match i.history_strategy {
            HistoryStrategy::Keep => {
                self.masks.replace_base_layers(i.masks);
            }
            HistoryStrategy::Reset => {
                let (width, height) = self.image.adjust.dimensions();
                self.masks = MaskImage::new(
                    [width.get() as _, height.get() as _],
                    i.masks,
                    History::default(),
                );
            }
        }
        Ok(())
    }
    fn create_texture(
        i: &ImageLoadOk,
        ctx: &egui::Context,
    ) -> Result<(TextureHandle, ImageSource<'static>), TextureExceedsLimit> {
        let (width, height) = i.adjust.dimensions();
        let max_texture_side = ctx.input(|i| i.max_texture_side);
        if width.get() as usize > max_texture_side || height.get() as usize > max_texture_side {
            return Err(TextureExceedsLimit::new(width, height, max_texture_side));
        }
        let handle = ctx.load_texture(
            "Overlays",
            ColorImage::new(
                [width.get() as _, height.get() as _],
                i.adjust_pixels()
                    .map(|(_, _, [r, g, b])| Color32::from_rgb(r, g, b))
                    .collect(),
            ),
            TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                ..Default::default()
            },
        );
        let texture = SizedTexture::from_handle(&handle);

        let source = ImageSource::Texture(texture);
        Ok((handle, source))
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "Image too large: {}x{}, max texture side is {}",
    width,
    height,
    max_texture_side
)]
pub struct TextureExceedsLimit {
    width: NonZeroU32,
    height: NonZeroU32,
    max_texture_side: usize,
}

impl TextureExceedsLimit {
    pub fn new(width: NonZeroU32, height: NonZeroU32, max_texture_side: usize) -> Self {
        Self {
            width,
            height,
            max_texture_side,
        }
    }
}

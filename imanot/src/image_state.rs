use std::num::NonZeroU32;

use egui::{
    self, Color32, ColorImage, ImageSource, TextureHandle, TextureOptions, load::SizedTexture,
};

use crate::{History, HistoryStrategy, ImageData, ImageId, ImageLoadOk, MaskImage};

#[expect(
    clippy::large_enum_variant,
    reason = "Loaded holds image, mask and history state"
)]
pub enum ImageState {
    NotLoaded,
    Loaded(ImageStateLoaded),
    Error(String),
}

impl ImageState {
    /// The textures the viewer draws. Masks are not among them, they are
    /// drawn by the GPU backend (see [`GpuBackend`](crate::gpu::GpuBackend)).
    pub fn sources(&self) -> impl Iterator<Item = ImageSource<'static>> + '_ {
        match self {
            ImageState::Loaded(x) => Some(x.texture.1.clone()),
            _ => None,
        }
        .into_iter()
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
    /// Replaces the displayed image. The only fallible step (the texture) runs
    /// first, so on error `self` is left untouched.
    ///
    /// This is where the mask policy for an incoming image lives:
    /// [`HistoryStrategy::Keep`] retains masks and history, but only while the
    /// image dimensions match, because the masks are bound to them.
    pub(crate) fn replace(
        &mut self,
        i: ImageData,
        ctx: &egui::Context,
    ) -> Result<(), TextureExceedsLimit> {
        let texture = Self::create_texture(&i.image, ctx)?;
        let (width, height) = i.image.adjust.dimensions();
        let size = [width.get() as usize, height.get() as usize];
        self.id = i.id;
        self.texture = texture;
        self.image = i.image;
        match i.history_strategy {
            HistoryStrategy::Keep if self.masks.size() == size => {
                self.masks.replace_base_layers(i.masks);
            }
            HistoryStrategy::Keep | HistoryStrategy::Reset => {
                self.masks = MaskImage::new(size, i.masks, History::default());
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

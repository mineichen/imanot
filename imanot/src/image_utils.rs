use std::num::NonZeroU32;

use imbuf::Image;

#[cfg(feature = "image-0_25")]
mod image;

/// Different image formats supported for the original image
#[derive(Clone)]
pub enum OriginalImage {
    Luma8(Image<u8, 1>),
    Luma16(Image<u16, 1>),
    Rgb8(Image<[u8; 3], 1>),
    Rgba8(Image<[u8; 4], 1>),
}

impl OriginalImage {
    pub fn width(&self) -> NonZeroU32 {
        match self {
            OriginalImage::Luma8(img) => img.dimensions().0,
            OriginalImage::Luma16(img) => img.dimensions().0,
            OriginalImage::Rgb8(img) => img.dimensions().0,
            OriginalImage::Rgba8(img) => img.dimensions().0,
        }
    }

    pub fn height(&self) -> NonZeroU32 {
        match self {
            OriginalImage::Luma8(img) => img.dimensions().1,
            OriginalImage::Luma16(img) => img.dimensions().1,
            OriginalImage::Rgb8(img) => img.dimensions().1,
            OriginalImage::Rgba8(img) => img.dimensions().1,
        }
    }
}

/// Represents a loaded image using image-buffer
#[derive(Clone)]
pub struct ImageLoadOk {
    pub original: OriginalImage,
    pub adjust: Image<[u8; 3], 1>,
}

impl ImageLoadOk {
    pub fn adjust_pixels(&self) -> impl Iterator<Item = (u32, u32, [u8; 3])> + '_ {
        let (width, _) = self.adjust.dimensions();
        let width = width.get();
        // flat_buffer() returns &[u8] for RgbImageInterleaved (flattened)
        self.adjust
            .buffer_flat()
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .map(move |(idx, chunk)| {
                let x = (idx % width as usize) as u32;
                let y = (idx / width as usize) as u32;
                (x, y, [chunk[0], chunk[1], chunk[2]])
            })
    }
}

#[cfg(feature = "image-0_25")]
pub use image::load_image;

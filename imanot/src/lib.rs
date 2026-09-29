use std::num::NonZeroU32;
use std::{future::Future, pin::Pin, sync::Arc};

mod async_task;
mod cursor_image;
mod image_state;
mod image_utils;
mod mask;
mod pixel_range;
mod state;
mod tool;
mod tools;
mod viewer;

pub use async_task::*;
pub use cursor_image::*;
pub use image_state::*;
pub use image_utils::*;
pub use imbuf::Image;
pub use state::*;

pub type ToolTask = AsyncRefTask<Result<Box<dyn Tool>, String>>;

pub use mask::*;
pub use pixel_range::*;
pub use tool::*;
pub use tools::*;
pub use viewer::*;

type RgbImageInterleaved<T> = Image<[T; 3], 1>;
type LocalBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(test)]
pub(crate) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(PartialEq, Clone, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ImageId(Arc<str>);

impl<'a> From<&'a str> for ImageId {
    fn from(s: &'a str) -> Self {
        Self(Arc::from(s))
    }
}

impl std::ops::Deref for ImageId {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HistoryStrategy {
    Keep,
    Reset,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct ImageData {
    pub id: ImageId,
    pub image: ImageLoadOk,
    pub masks: PixelAreaStack,
    pub history_strategy: HistoryStrategy,
}

impl ImageData {
    pub fn new(
        id: ImageId,
        image: ImageLoadOk,
        masks: PixelAreaStack,
        history_strategy: HistoryStrategy,
    ) -> Self {
        Self {
            id,
            image,
            masks,
            history_strategy,
        }
    }
    pub fn chessboard() -> impl Iterator<Item = Self> {
        (0..2).map(|i| ImageData {
            id: ImageId::from(format!("image{}", i + 1).as_str()),
            masks: PixelAreaStack::default(),
            image: {
                let width = const { NonZeroU32::new(400).expect("400 is not zero") };
                let height = const { NonZeroU32::new(400).expect("400 is not zero") };
                let square_size = width.get() / 8;
                let (color_1, color_2) = if i == 0 { (0, 255) } else { (255, 0) };

                let image_data = (0..height.get())
                    .flat_map(|y| {
                        (0..width.get()).map(move |x| {
                            let square_x = x / square_size;
                            let square_y = y / square_size;
                            let is_white = (square_x + square_y) % 2 == 0;
                            let color = if is_white { color_1 } else { color_2 };
                            [color, color, color]
                        })
                    })
                    .collect();
                let buffer = RgbImageInterleaved::new_arc(image_data, width, height);
                ImageLoadOk {
                    original: crate::image_utils::OriginalImage::Rgb8(buffer.clone()),
                    adjust: buffer,
                }
            },
            history_strategy: HistoryStrategy::Reset,
        })
    }
}

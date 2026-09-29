use std::{
    num::{NonZeroU32, NonZeroU64, TryFromIntError},
    sync::Arc,
};

use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType};
use imask::NonZeroRange;
use itertools::Itertools;
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::value::{TensorValueType, Value};

use super::RgbImageInterleaved;

impl From<TryFromIntError> for InferenceError {
    fn from(value: TryFromIntError) -> Self {
        Self::Other(Arc::new(value))
    }
}

pub type SamEmbeddings = ResizedImageData<Value<TensorValueType<f32>>>;
pub type SamInputData = ResizedImageData<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>;

#[derive(Debug, thiserror::Error, Clone)]
pub enum InferenceError {
    #[error("Other: {0:?}")]
    Other(Arc<dyn std::error::Error + Send + Sync>),

    #[error("Unexpected network output")]
    UnexpectedOutput(String),
}

pub(super) fn prepare_image_input(
    img: &RgbImageInterleaved<u8>,
) -> Result<SamInputData, InferenceError> {
    let (original_width, original_height) = img.dimensions();
    // Convert RgbImageInterleaved to DynamicImage for proper resizing
    let (width, height) = (original_width.get(), original_height.get());
    let pixels = img.buffer_flat();
    let rgb_image = image::RgbImage::from_raw(width, height, pixels.to_vec()).ok_or_else(|| {
        InferenceError::Other(Arc::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to create RgbImage",
        )))
    })?;
    let dynamic_img = DynamicImage::ImageRgb8(rgb_image);

    // Resize using the same method as the original code
    let img_resized = dynamic_img.resize(1024, 1024, FilterType::CatmullRom);
    let (resized_width, resized_height) = img_resized.dimensions();
    let (resized_width, resized_height) = (
        NonZeroU32::try_from(resized_width)?,
        NonZeroU32::try_from(resized_height)?,
    );

    let mut input = Array::zeros((1, 3, 1024, 1024));
    let rgb = input
        .as_slice_mut()
        .expect("zeros always returns continuous slice");
    let (r, gb) = rgb.split_at_mut(1024 * 1024);
    let (g, b) = gb.split_at_mut(1024 * 1024);

    // Process the resized RGB image - we know it's RGB since input is RgbImageInterleaved
    // Calculate statistics on resized image
    let mut rs = rolling_stats::Stats::new();
    let mut gs = rolling_stats::Stats::new();
    let mut bs = rolling_stats::Stats::new();

    for (_, _, Rgba([r, g, b, _])) in img_resized.pixels() {
        rs.update(r as f32);
        gs.update(g as f32);
        bs.update(b as f32);
    }

    // Fill arrays row-by-row, matching the original code exactly
    for (((input_chunk, r_chunk), g_chunk), b_chunk) in img_resized
        .pixels()
        .chunks(img_resized.width() as _)
        .into_iter()
        .zip(r.as_chunks_mut::<1024>().0)
        .zip(g.as_chunks_mut::<1024>().0)
        .zip(b.as_chunks_mut::<1024>().0)
    {
        for ((((_, _, Rgba([r, g, b, _])), r_dest), g_dest), b_dest) in
            input_chunk.zip(r_chunk).zip(g_chunk).zip(b_chunk)
        {
            *r_dest = (r as f32 - rs.mean) / rs.std_dev;
            *g_dest = (g as f32 - gs.mean) / gs.std_dev;
            *b_dest = (b as f32 - bs.mean) / bs.std_dev;
        }
    }

    Ok(ResizedImageData {
        image_data: input.into_dyn(),
        resized_width,
        resized_height,
        original_width,
        original_height,
    })
}

pub(super) fn extract_pixel_ranges(
    iter: impl Iterator<Item = f32>,
    width: NonZeroU32,
) -> Vec<NonZeroRange<u64>> {
    let mut result = vec![];
    iter.enumerate()
        .filter_map(|(pos, item)| (item > 0.0).then_some(pos as u64))
        .chunk_by(|&x| x / NonZeroU64::from(width))
        .into_iter()
        .for_each(|(_, mut b)| {
            let first = b.next().expect("Doesn't yield if group is empty");
            result.push(NonZeroRange::from_span(first, NonZeroU64::MIN));
            b.fold(first, |last, x| {
                if x - 1 == last {
                    let item = result
                        .last_mut()
                        .expect("result just received a push in this iteration");
                    item.increment_length();
                } else {
                    result.push(NonZeroRange::from_span(x, NonZeroU64::MIN));
                }
                x
            });
        });
    result
}

#[derive(Debug)]
pub struct ResizedImageData<T> {
    pub(super) image_data: T,
    pub(super) original_width: NonZeroU32,
    pub(super) original_height: NonZeroU32,
    pub(super) resized_width: NonZeroU32,
    pub(super) resized_height: NonZeroU32,
}

impl<T> ResizedImageData<T> {
    pub fn map<TNew>(self, x: impl FnOnce(T) -> TNew) -> ResizedImageData<TNew> {
        ResizedImageData {
            image_data: (x)(self.image_data),
            original_width: self.original_width,
            original_height: self.original_height,
            resized_width: self.resized_width,
            resized_height: self.resized_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use super::*;

    const NON_ZERO_3: NonZero<u32> = NonZero::new(3).expect("3 is not zero");

    #[test]
    fn extract_pixel_ranges_summarizes_pixels() {
        assert_eq!(
            vec![NonZeroRange::from_span(0, NON_ZERO_3.into())],
            extract_pixel_ranges([1., 1., 1.].iter().copied(), NON_ZERO_3)
        );
    }
}

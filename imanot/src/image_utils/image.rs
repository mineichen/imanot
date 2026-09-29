use std::num::NonZeroU32;

use image_0_25 as image;
use image_0_25::{DynamicImage, ImageBuffer as ImageImageBuffer, Luma};

use crate::image_utils::{ImageLoadOk, OriginalImage};
use imbuf::Image;

impl OriginalImage {
    pub fn to_dynamic_image(&self) -> DynamicImage {
        match self {
            OriginalImage::Luma8(img) => {
                use image::GrayImage;
                let (width, height) = (img.dimensions().0.get(), img.dimensions().1.get());
                let pixels: Vec<u8> = img.buffer().to_vec();
                let gray =
                    GrayImage::from_raw(width, height, pixels).expect("Failed to create GrayImage");
                DynamicImage::ImageLuma8(gray)
            }
            OriginalImage::Luma16(img) => {
                use image::ImageBuffer;
                let (width, height) = (img.dimensions().0.get(), img.dimensions().1.get());
                let pixels: Vec<u16> = img.buffer().to_vec();
                let gray: ImageBuffer<Luma<u16>, Vec<u16>> =
                    ImageBuffer::from_raw(width, height, pixels)
                        .expect("Failed to create Gray16Image");
                DynamicImage::ImageLuma16(gray)
            }
            OriginalImage::Rgb8(img) => {
                use image::RgbImage;
                let (width, height) = (img.dimensions().0.get(), img.dimensions().1.get());
                let pixels: Vec<u8> = img.buffer_flat().to_vec();
                let rgb =
                    RgbImage::from_raw(width, height, pixels).expect("Failed to create RgbImage");
                DynamicImage::ImageRgb8(rgb)
            }
            OriginalImage::Rgba8(img) => {
                use image::RgbaImage;
                let (width, height) = (img.dimensions().0.get(), img.dimensions().1.get());
                let pixels: Vec<u8> = img.buffer_flat().to_vec();
                let rgba =
                    RgbaImage::from_raw(width, height, pixels).expect("Failed to create RgbaImage");
                DynamicImage::ImageRgba8(rgba)
            }
        }
    }
}

impl ImageLoadOk {
    pub fn to_dynamic_image(&self) -> DynamicImage {
        self.original.to_dynamic_image()
    }
}

pub fn load_image(bytes: &[u8]) -> std::io::Result<ImageLoadOk> {
    let original = image::load_from_memory(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    Ok(match &original {
        DynamicImage::ImageLuma16(i) => {
            let adjust = DynamicImage::ImageLuma16(fix_image_contrast(i));
            let original_buffer = luma16_to_buffer(i)?;
            let adjust_buffer = image_to_rgb_buffer(&adjust)?;
            ImageLoadOk {
                original: OriginalImage::Luma16(original_buffer),
                adjust: adjust_buffer,
            }
        }
        DynamicImage::ImageLuma8(i) => {
            let adjust = DynamicImage::ImageLuma8(fix_image_contrast(i));
            let original_buffer = luma8_to_buffer(i)?;
            let adjust_buffer = image_to_rgb_buffer(&adjust)?;
            ImageLoadOk {
                original: OriginalImage::Luma8(original_buffer),
                adjust: adjust_buffer,
            }
        }
        DynamicImage::ImageRgb8(i) => {
            let original_buffer = rgb8_to_buffer(i)?;
            let adjust_buffer = image_to_rgb_buffer(&original)?;
            ImageLoadOk {
                original: OriginalImage::Rgb8(original_buffer),
                adjust: adjust_buffer,
            }
        }
        _ => {
            let original_buffer = image_to_rgb_buffer(&original)?;
            ImageLoadOk {
                original: OriginalImage::Rgb8(original_buffer.clone()),
                adjust: original_buffer,
            }
        }
    })
}

fn luma8_to_buffer(img: &ImageImageBuffer<Luma<u8>, Vec<u8>>) -> std::io::Result<Image<u8, 1>> {
    let (width, height) = img.dimensions();
    let pixels: Vec<u8> = img.pixels().map(|p| p.0[0]).collect();
    let width = NonZeroU32::new(width).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image width")
    })?;
    let height = NonZeroU32::new(height).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image height")
    })?;
    Ok(Image::new_vec(pixels, width, height))
}

fn luma16_to_buffer(img: &ImageImageBuffer<Luma<u16>, Vec<u16>>) -> std::io::Result<Image<u16, 1>> {
    let (width, height) = img.dimensions();
    let pixels: Vec<u16> = img.pixels().map(|p| p.0[0]).collect();
    let width = NonZeroU32::new(width).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image width")
    })?;
    let height = NonZeroU32::new(height).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image height")
    })?;
    Ok(Image::new_vec(pixels, width, height))
}

fn rgb8_to_buffer(
    img: &ImageImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> std::io::Result<Image<[u8; 3], 1>> {
    let (width, height) = img.dimensions();
    let vec_u8: Vec<u8> = img.pixels().flat_map(|p| p.0).collect();
    // Convert Vec<u8> to Vec<[u8; 3]>
    let vec: Vec<[u8; 3]> = vec_u8.as_chunks::<3>().0.to_vec();
    let width = NonZeroU32::new(width).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image width")
    })?;
    let height = NonZeroU32::new(height).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image height")
    })?;
    Ok(Image::new_vec(vec, width, height))
}

fn image_to_rgb_buffer(img: &DynamicImage) -> std::io::Result<Image<[u8; 3], 1>> {
    let rgba = img.to_rgb8();
    let (width, height) = rgba.dimensions();
    let vec_u8 = rgba.into_vec();
    // Convert Vec<u8> to Vec<[u8; 3]>
    let vec: Vec<[u8; 3]> = vec_u8.as_chunks::<3>().0.to_vec();
    let width = NonZeroU32::new(width).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image width")
    })?;
    let height = NonZeroU32::new(height).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid image height")
    })?;
    Ok(Image::new_vec(vec, width, height))
}

fn fix_image_contrast<T: image::Primitive + Ord>(
    i: &ImageImageBuffer<Luma<T>, Vec<T>>,
) -> ImageImageBuffer<Luma<T>, Vec<T>>
where
    f32: From<T>,
{
    let mut pixels = i.pixels().map(|Luma([p])| p).collect::<Vec<_>>();
    pixels.sort_unstable();
    let five_percent_pos = pixels.len() / 20;
    let lower: f32 = (*pixels[five_percent_pos]).into();
    let upper: f32 = (*pixels[five_percent_pos * 19]).into();
    let max_pixel_value: f32 = T::DEFAULT_MAX_VALUE.into();
    if lower == upper {
        return i.clone();
    }
    let range = max_pixel_value / (upper - lower);
    ImageImageBuffer::from_raw(
        i.width(),
        i.height(),
        i.pixels()
            .map(|Luma([p])| {
                let as_f: f32 = (*p).into();

                num_traits::cast::NumCast::from(
                    ((as_f - lower) * range).clamp(0.0, max_pixel_value),
                )
                .expect("Clamped f32 value fits in the pixel type")
            })
            .collect(),
    )
    .expect("Buffer length matches image dimensions")
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use image::{DynamicImage, ImageBuffer as ImageImageBuffer};
    use imbuf::Image;

    type LumaImage<T> = Image<T, 1>;

    use crate::TestResult;
    use crate::image_utils::OriginalImage;

    use super::*;

    #[test]
    fn fix_image_contrast_all_pixels_same() -> TestResult {
        let image = ImageImageBuffer::from_raw(5, 5, vec![255; 25])
            .ok_or("Buffer length matches image dimensions")?;
        let fixed = fix_image_contrast::<u8>(&image);
        assert_eq!(fixed, image);
        Ok(())
    }

    #[test]
    fn original_image_luma8_to_dynamic_image() -> TestResult {
        let width = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let height = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let pixels: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let luma_img = LumaImage::new_vec(pixels.clone(), width, height);
        let original = OriginalImage::Luma8(luma_img);

        let dyn_img = original.to_dynamic_image();
        match dyn_img {
            DynamicImage::ImageLuma8(img) => {
                assert_eq!(img.dimensions(), (10, 10));
                let converted_pixels: Vec<u8> = img.pixels().map(|&image::Luma([p])| p).collect();
                assert_eq!(converted_pixels, pixels);
                Ok(())
            }
            _ => Err(format!("Expected ImageLuma8, got {dyn_img:?}").into()),
        }
    }

    #[test]
    fn original_image_luma16_to_dynamic_image() -> TestResult {
        let width = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let height = NonZeroU32::new(10).ok_or("10 is not zero")?;
        // Create 16-bit pixels with values that should convert correctly
        let pixels: Vec<u16> = (0..100).map(|i| (i * 257) as u16).collect(); // Use values that span the range
        let luma_img = LumaImage::new_vec(pixels.clone(), width, height);
        let original = OriginalImage::Luma16(luma_img);

        let dyn_img = original.to_dynamic_image();
        match dyn_img {
            DynamicImage::ImageLuma16(img) => {
                assert_eq!(img.dimensions(), (10, 10));
                let converted_pixels: Vec<u16> = img.pixels().map(|&image::Luma([p])| p).collect();
                // Verify that pixels are preserved exactly
                assert_eq!(converted_pixels, pixels);
                Ok(())
            }
            _ => Err(format!("Expected ImageLuma16, got {dyn_img:?}").into()),
        }
    }

    #[test]
    fn original_image_rgb8_to_dynamic_image() -> TestResult {
        let width = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let height = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let pixels: Vec<[u8; 3]> = (0..100)
            .map(|i| [(i * 3) as u8, (i * 3 + 1) as u8, (i * 3 + 2) as u8])
            .collect();
        let rgb_img = Image::new_vec(pixels.clone(), width, height);
        let original = OriginalImage::Rgb8(rgb_img);

        let dyn_img = original.to_dynamic_image();
        match dyn_img {
            DynamicImage::ImageRgb8(img) => {
                assert_eq!(img.dimensions(), (10, 10));
                let converted_pixels: Vec<[u8; 3]> = img.pixels().map(|&image::Rgb(x)| x).collect();
                assert_eq!(converted_pixels, pixels);
                Ok(())
            }
            _ => Err(format!("Expected ImageRgb8, got {dyn_img:?}").into()),
        }
    }

    #[test]
    fn original_image_rgba8_to_dynamic_image() -> TestResult {
        let width = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let height = NonZeroU32::new(10).ok_or("10 is not zero")?;
        let pixels: Vec<[u8; 4]> = (0..100)
            .map(|i| {
                [
                    (i * 4) as u8,
                    (i * 4 + 1) as u8,
                    (i * 4 + 2) as u8,
                    (i * 4 + 3) as u8,
                ]
            })
            .collect();
        let rgba_img = Image::new_vec(pixels.clone(), width, height);
        let original = OriginalImage::Rgba8(rgba_img);

        let dyn_img = original.to_dynamic_image();
        match dyn_img {
            DynamicImage::ImageRgba8(img) => {
                assert_eq!(img.dimensions(), (10, 10));
                let converted_pixels: Vec<[u8; 4]> =
                    img.pixels().map(|&image::Rgba(x)| x).collect();
                assert_eq!(converted_pixels, pixels);
                Ok(())
            }
            _ => Err(format!("Expected ImageRgba8, got {dyn_img:?}").into()),
        }
    }

    #[test]
    fn original_image_luma16_conversion_preserves_range() -> TestResult {
        // Test that Luma16 conversion handles the full 16-bit range correctly
        let width = NonZeroU32::new(5).ok_or("5 is not zero")?;
        let height = NonZeroU32::new(1).ok_or("1 is not zero")?;
        // Test various 16-bit values
        let pixels: Vec<u16> = vec![0, 128, 256, 512, 65535];
        let luma_img = LumaImage::new_vec(pixels.clone(), width, height);
        let original = OriginalImage::Luma16(luma_img);

        let dyn_img = original.to_dynamic_image();
        match dyn_img {
            DynamicImage::ImageLuma16(img) => {
                let converted: Vec<u16> = img.pixels().map(|&image::Luma([p])| p).collect();
                // Verify that all 16-bit values are preserved exactly
                assert_eq!(converted, pixels);
                assert_eq!(converted[0], 0);
                assert_eq!(converted[1], 128);
                assert_eq!(converted[2], 256);
                assert_eq!(converted[3], 512);
                assert_eq!(converted[4], 65535);
                Ok(())
            }
            _ => Err(format!("Expected ImageLuma16, got {dyn_img:?}").into()),
        }
    }
}

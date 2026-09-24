use std::num::NonZeroU16;

use imask::{ImageDimension, SortedRanges, Span};

type Ranges = SortedRanges<u32>;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PixelArea {
    pub pixels: Ranges,
    pub color: [u8; 4],
}

impl PixelArea {
    pub fn new(
        pixels: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
        color: [u8; 4],
    ) -> Option<Self> {
        Some(Self {
            pixels: Self::try_from_spans(pixels)?,
            color,
        })
    }

    pub fn with_black_color(
        pixels: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
    ) -> Option<Self> {
        Some(Self {
            pixels: Self::try_from_spans(pixels)?,
            color: [0, 0, 0, 255],
        })
    }

    pub fn single_pixel_total_color(x: u16, y: u16, len: NonZeroU16, color: [u8; 4]) -> Self {
        Self {
            pixels: Ranges::from(Span::new(x..x + len.get(), y)),
            color,
        }
    }
    #[cfg(test)]
    pub fn single_range_total_black(x: u16, y: u16, len: NonZeroU16) -> Self {
        Self::single_pixel_total_color(x, y, len, [0, 0, 0, 255])
    }

    fn try_from_spans(
        pixels: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
    ) -> Option<Ranges> {
        let iter = pixels.into_iter();
        Ranges::try_from_span_iter(iter).ok()
    }

    pub fn from_ranges(pixels: Ranges, color: [u8; 4]) -> Self {
        Self { pixels, color }
    }

    pub fn range_len(&self) -> usize {
        self.pixels.len()
    }
}

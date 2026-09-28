//! Shared fixtures for the drag module's unit tests.

use std::num::NonZeroU32;

use imask::{ImageDimension, NonZeroRange, Roi, SortedRanges, Span, SpanBoundsBuilder, WithRoi};

use crate::{History, MaskDefaultActions, MaskImage, PixelAreaStack};

pub(crate) const IMG_ROI: Roi<u32> = Roi {
    x: NonZeroRange::<u32>::new_const(0..100),
    y: NonZeroRange::<u32>::new_const(0..100),
};

/// `Vec` adapter: `Vec` is not `ImageDimension`, so tight bounds are tracked
/// natively via `SpanBoundsBuilder` first.
pub(crate) fn ranges_from_spans(spans: Vec<Span<u32>>) -> Option<SortedRanges<u32>> {
    let tight = spans
        .iter()
        .copied()
        .collect::<SpanBoundsBuilder<u32>>()
        .build()
        .ok()?;
    SortedRanges::try_from_span_iter(WithRoi::new(spans.into_iter(), tight)).ok()
}

pub(crate) fn layer_pixels(masks: &MaskImage) -> Option<SortedRanges<u32>> {
    masks.subgroups_stack().get(0).map(|a| a.pixels.clone())
}

pub(crate) fn outsider_block_ok(masks: &MaskImage) -> bool {
    layer_pixels(masks).is_some_and(|p| {
        let rows: Vec<Span<u32>> = p
            .spans::<u32>()
            .filter(|s| (30..35).contains(&s.y))
            .collect();
        rows.len() == 5 && rows.iter().all(|s| s.x.start <= 40 && 50 <= s.x.end)
    })
}

/// Fresh image holding a single layer with `ranges`.
pub(crate) fn mask(
    ranges: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
) -> MaskImage {
    let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
    let ranges = SortedRanges::try_from_span_iter(ranges).expect("Valid ranges");
    masks.add(ranges);
    masks
}

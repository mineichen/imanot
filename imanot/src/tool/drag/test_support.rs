//! Shared fixtures for the drag module's unit tests.

use std::num::NonZeroU32;

use imask::{Roi, SortedRanges, Span, SpanBoundsBuilder, WithRoi};

use crate::{History, MaskDefaultActions, MaskImage, PixelAreaStack};

pub(crate) fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

pub(crate) fn img_roi() -> Roi<u32> {
    Roi::from_dimensions(nz(100), nz(100))
}

pub(crate) fn rect_ranges(x: u32, y: u32, w: NonZeroU32, h: NonZeroU32) -> SortedRanges<u32> {
    SortedRanges::try_from_span_iter(Roi::new(x..x + w.get(), y..y + h.get()).into_spans()).unwrap()
}

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
pub(crate) fn mask(ranges: SortedRanges<u32>) -> MaskImage {
    let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
    masks.add(ranges);
    masks
}

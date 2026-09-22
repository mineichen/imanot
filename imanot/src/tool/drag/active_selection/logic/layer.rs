use std::iter::FusedIterator;

use imask::{ImageDimension, ImaskSet, PipelineError, SortedRanges, SortedRangesSpanBuilder, Span};

/// One selected layer: the pristine `original` ranges (snapshotted at
/// selection time, never modified) plus the `committed` ranges (what the last
/// commit wrote to history, `None` if currently moved fully out of the
/// image). Every preview and every commit is computed as
/// `transform(original, total)`, so chained gestures never accumulate
/// rasterization loss within one selection lifetime. `background` holds the
/// non-selected pixels (`layer − snapshot` at selection time, shrunk on
/// shift-add): every commit re-adds the background under the cleared
/// footprint, so pixels outside the original selection always remain
/// unchanged — even ones a previous commit overlapped.
pub(super) struct LayerSelection {
    pub(super) original: SortedRanges<u32>,
    pub(super) committed: SortedRanges<u32>,
    pub(super) background: Option<SortedRanges<u32>>,
}

impl LayerSelection {
    /// Fresh snapshot: nothing transformed yet, so placed pixels equal the
    /// pristine original.
    pub(super) fn fresh(ranges: SortedRanges<u32>, background: Option<SortedRanges<u32>>) -> Self {
        Self {
            original: ranges.clone(),
            committed: ranges,
            background,
        }
    }

    pub(super) fn update(mut self, ranges: &SortedRanges<u32>) -> Option<Self> {
        union_ranges(&self.original, ranges)
            .zip(union_ranges(&self.committed, ranges))
            .map(|(original, committed)| {
                self.background = self
                    .background
                    .take()
                    .and_then(|bg| subtract_ranges(&bg, ranges.spans()));
                self.original = original;
                self.committed = committed;
                self
            })
    }

    /// Background pixels under the committed footprint, for re-adding in the
    /// same hook as the Clear. `None` when nothing overlaps (the common
    /// case) — then commit stays a Clear + Add pair.
    pub(super) fn restore(&self) -> Option<impl Iterator<Item = Span<u32>> + ImageDimension> {
        let bg = self.background.as_ref()?;
        let under = bg.spans::<u32>().intersect(self.committed.spans()).ok()?;
        Some(under)
    }
}

/// Union of two range sets. Returns `None` only if both are empty (cannot
/// happen for snapshot content, which is never empty). Builds straight from
/// the union stream; `minbounds` guarantees tight bounds either way.
fn union_ranges(a: &SortedRanges<u32>, b: &SortedRanges<u32>) -> Option<SortedRanges<u32>> {
    SortedRanges::try_from_span_iter_minbounds(a.spans::<u32>().union(b.spans())).ok()
}

/// `a` minus `b` as tight ranges. `None` when empty.
pub(crate) fn subtract_ranges(
    a: &SortedRanges<u32>,
    b: impl Iterator<Item = Span<u32>>,
) -> Option<SortedRanges<u32>> {
    SortedRanges::try_from_span_iter_minbounds(a.spans::<u32>().subtract(b)).ok()
}

pub(crate) fn subtract_ranges_collect_subtrahend(
    a: &SortedRanges<u32>,
    b: impl FusedIterator<Item = Span<u32>> + ImageDimension,
) -> (
    Option<SortedRanges<u32>>,
    Result<SortedRanges<u32>, PipelineError>,
) {
    let span_builder = SortedRangesSpanBuilder::new(b.roi());
    let mut b = b.fold_inline(span_builder, |b, n| {
        b.add(*n);
    });

    let r = SortedRanges::try_from_span_iter_minbounds(a.spans::<u32>().subtract(&mut b)).ok();
    (r, b.finish_all().build())
}

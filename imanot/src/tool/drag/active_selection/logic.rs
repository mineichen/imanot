use std::{collections::BTreeMap, iter::FusedIterator};

use imask::{ImageDimension, ImaskSet, PipelineError, SortedRanges, SortedRangesSpanBuilder, Span};
use nalgebra::Matrix3;

use super::super::frame::Frame;
use crate::HistoryAction;

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
pub(crate) struct LayerSelection {
    pub(crate) original: SortedRanges<u32>,
    pub(crate) committed: SortedRanges<u32>,
    pub(crate) background: Option<SortedRanges<u32>>,
}

impl LayerSelection {
    /// Fresh snapshot: nothing transformed yet, so placed pixels equal the
    /// pristine original.
    pub(crate) fn fresh(ranges: SortedRanges<u32>, background: Option<SortedRanges<u32>>) -> Self {
        Self {
            original: ranges.clone(),
            committed: ranges,
            background,
        }
    }

    /// Background pixels under the committed footprint, for re-adding in the
    /// same hook as the Clear. `None` when nothing overlaps (the common
    /// case) — then commit stays a Clear + Add pair.
    pub(crate) fn restore(&self) -> Option<SortedRanges<u32>> {
        match self.background.as_ref() {
            Some(bg) => {
                let under = bg
                    .spans::<u32>()
                    .intersect(self.committed.spans())
                    .ok()?;
                SortedRanges::try_from_span_iter_minbounds(under).ok()
            }
            _ => None,
        }
    }
}

/// Pure selection logic: snapshot content, accumulated transform and frame.
/// GPU-free on purpose, so unit tests stay context-free. The live preview
/// lives in the owning [`super::ActiveSelection`] wrapper next to this, which
/// clears it whenever the logic mutates underneath a live texture.
pub(crate) struct ActiveSelectionLogic {
    pub(crate) layers: BTreeMap<usize, LayerSelection>,
    /// Accumulated gesture transform, applied uniformly to every layer's
    /// pristine `original`. Only ever extended by gesture deltas, reset to
    /// identity on rebase (see `rebase`).
    pub(crate) total: Matrix3<f64>,
    /// Current selection geometry for overlay, anchors, pivots and
    /// hit-testing. Advanced by the same gesture deltas as `total`, so the two
    /// can never drift apart. Never derived from bounds.
    pub(crate) frame: Frame,
    /// Last history action seen (at selection or after an own commit). If the
    /// history tip differs, something else changed the mask (push, undo,
    /// redo, another tool) and the snapshot is stale.
    pub(crate) tip: Option<HistoryAction>,
}

impl ActiveSelectionLogic {
    /// Fresh (replacing) single-layer selection. `original == committed`.
    pub(crate) fn fresh_single(
        idx: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
        tip: Option<HistoryAction>,
    ) -> Self {
        Self {
            total: Matrix3::identity(),
            frame: Frame::around(ranges.bounds()),
            layers: std::iter::once((idx, LayerSelection::fresh(ranges, background))).collect(),
            tip,
        }
    }

    /// Add `ranges` on layer `idx`, unioning into the existing entry when the
    /// layer is already selected. Unioning is idempotent, so re-adding the
    /// same pixels is a no-op — and required: separate entries for one layer
    /// would clear/add the same layer twice per commit, subtracting or
    /// duplicating content on resize/move. `background` is the layer's
    /// non-selected content for a not-yet-selected layer; entries that
    /// already exist shrink their own background instead.
    ///
    /// Unioning changes the snapshot content the preview is rasterized from,
    /// so callers must clear the preview afterwards (see
    /// [`super::ActiveSelection::merge_layer`]).
    pub(crate) fn merge_layer(
        &mut self,
        layer_id: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
    ) {
        self.frame.expand_to_cover(ranges.bounds());
        if let Some(entry) = self.layers.get_mut(&layer_id) {
            entry.background = entry
                .background
                .take()
                .and_then(|bg| subtract_ranges(&bg, ranges.spans()));
            if let (Some(original), Some(committed)) = (
                union_ranges(&entry.original, &ranges),
                union_ranges(&entry.committed, &ranges),
            ) {
                entry.original = original;
                entry.committed = committed;
            }
        } else {
            let fresh = LayerSelection::fresh(ranges, background);
            self.layers.insert(layer_id, fresh);
        }
    }
}

/// Union of two range sets. Returns `None` only if both are empty (cannot
/// happen for snapshot content, which is never empty). Builds straight from
/// the union stream; `minbounds` guarantees tight bounds either way.
pub(crate) fn union_ranges(
    a: &SortedRanges<u32>,
    b: &SortedRanges<u32>,
) -> Option<SortedRanges<u32>> {
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
    let span_builder = SortedRangesSpanBuilder::new(b.bounds());
    let mut b = b.fold_inline(span_builder, |b, n| {
        b.add(*n);
    });

    let r = SortedRanges::try_from_span_iter_minbounds(a.spans::<u32>().subtract(&mut b)).ok();
    (r, b.finish_all().build())
}

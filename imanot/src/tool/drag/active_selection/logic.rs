use std::{collections::BTreeMap, iter::FusedIterator};

use imask::{
    ImageDimension, ImaskSet, PipelineError, Roi, SortedRanges, SortedRangesSpanBuilder, Span,
};
use nalgebra::Matrix3;

use super::super::frame::{Frame, union_bounds};
use super::super::transform::{push_add, push_clear, transform_layer};
use crate::{HistoryAction, MaskImage};

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
struct LayerSelection {
    original: SortedRanges<u32>,
    committed: SortedRanges<u32>,
    background: Option<SortedRanges<u32>>,
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
    pub(crate) fn restore(&self) -> Option<impl Iterator<Item = Span<u32>> + ImageDimension> {
        let bg = self.background.as_ref()?;
        let under = bg.spans::<u32>().intersect(self.committed.spans()).ok()?;
        Some(under)
    }
}

/// Pure selection logic: snapshot content, accumulated transform and frame.
/// GPU-free on purpose, so unit tests stay context-free. The live preview
/// lives in the owning [`super::ActiveSelection`] wrapper next to this, which
/// clears it whenever the logic mutates underneath a live texture.
pub(crate) struct ActiveSelectionLogic {
    layers: BTreeMap<usize, LayerSelection>,
    /// Accumulated gesture transform, applied uniformly to every layer's
    /// pristine `original`. Only ever extended by gesture deltas, reset to
    /// identity on rebase (see `rebase`).
    total: Matrix3<f64>,
    /// Current selection geometry for overlay, anchors, pivots and
    /// hit-testing. Advanced by the same gesture deltas as `total`, so the two
    /// can never drift apart. Never derived from bounds.
    frame: Frame,
    /// Last history action seen (at selection or after an own commit). If the
    /// history tip differs, something else changed the mask (push, undo,
    /// redo, another tool) and the snapshot is stale.
    tip: Option<HistoryAction>,
}

impl ActiveSelectionLogic {
    pub(crate) fn rebase(&mut self, tip: Option<HistoryAction>) {
        self.layers.values_mut().for_each(|l| {
            l.original = l.committed.clone();
        });
        self.total = Matrix3::identity();
        self.tip = tip;
    }
    /// Fresh (replacing) single-layer selection. `original == committed`.
    pub(crate) fn fresh_single(
        idx: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
        tip: Option<HistoryAction>,
    ) -> Self {
        Self {
            total: Matrix3::identity(),
            frame: Frame::around(ranges.roi()),
            layers: BTreeMap::from([(idx, LayerSelection::fresh(ranges, background))]),
            tip,
        }
    }

    /// Fresh multi-layer selection from rect-clipped triples. Returns `None`
    /// when empty so callers drop instead of showing an empty box. The frame
    /// tightly covers the contained pixels, never the queried boxes.
    pub(crate) fn fresh_from_clipped(
        parts: impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)>,
        tip: Option<HistoryAction>,
    ) -> Option<Self> {
        let layers = parts
            .map(|(layer, ranges, background)| (layer, LayerSelection::fresh(ranges, background)))
            .collect::<BTreeMap<_, _>>();
        let content = union_bounds(layers.values().map(|x| x.original.roi()))?;
        Some(Self {
            total: Matrix3::identity(),
            frame: Frame::around(content),
            layers,
            tip,
        })
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
    /// [`super::ActiveSelection::merge_layers`]).
    pub fn merge_layer(
        &mut self,
        layer_id: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
    ) {
        self.frame.expand_to_cover(ranges.roi());
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

    /// Fresh selection from raw per-layer span streams carrying their own
    /// bounds: rebuilds tight ranges per layer (attached bounds may cover
    /// the whole image, e.g. layers loaded from storage, and must not size
    /// the selection or the transform output). Layers without visible pixels
    /// are skipped; `None` when empty so callers drop instead of showing an
    /// empty box.
    pub(crate) fn fresh_from_spans<S>(
        parts: impl Iterator<Item = (usize, S)>,
        tip: Option<HistoryAction>,
    ) -> Option<Self>
    where
        S: Iterator<Item = Span<u32>> + ImageDimension,
    {
        Self::fresh_from_clipped(
            parts.filter_map(|(idx, spans)| {
                SortedRanges::try_from_span_iter_minbounds(spans)
                    .ok()
                    .map(|ranges| (idx, ranges, None))
            }),
            tip,
        )
    }

    /// Shift-add batch: bake placed pixels into the snapshot and reset the
    /// transform ([`Self::rebase`], so later gestures transform old and new
    /// pixels uniformly), then union every new part ([`Self::merge_layer`]).
    /// Takes the new history tip so the staleness guard is always re-armed
    /// together with the rebase — callers must not set `tip` separately.
    /// Consumes the parts lazily; callers pass `std::iter::once(first).chain(rest)`
    /// after peeking non-emptiness, so no intermediate collection is needed.
    /// Unioning changes the snapshot content the preview is rasterized from,
    /// so callers must clear the preview afterwards (see
    /// [`super::ActiveSelection::merge_layers`]).
    pub(crate) fn merge_layers(
        &mut self,
        parts: impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)>,
        tip: Option<HistoryAction>,
    ) {
        self.rebase(tip);
        for (layer_id, ranges, background) in parts {
            self.merge_layer(layer_id, ranges, background);
        }
    }

    /// Current frame by value (`Frame` is `Copy`): overlay, anchors,
    /// hit-testing. Read-only; mutation goes through [`Self::set_transform`].
    pub(crate) fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Current accumulated transform by value. Read-only; mutation goes
    /// through [`Self::set_transform`], `rebase` or construction.
    pub(crate) fn total(&self) -> Matrix3<f64> {
        self.total
    }

    /// Atomic `frame ↔ total` update for gesture progress / cancel. The only
    /// way to move one without the other, so overlay and rasterization can
    /// never drift apart.
    pub(crate) fn set_transform(&mut self, frame: Frame, total: Matrix3<f64>) {
        self.frame = frame;
        self.total = total;
    }

    /// Paired snapshot for gesture `base` state.
    pub(crate) fn snapshot_transform(&self) -> (Frame, Matrix3<f64>) {
        (self.frame, self.total)
    }

    /// Whether the snapshot is stale against the current history tip.
    pub(crate) fn is_stale(&self, current: Option<HistoryAction>) -> bool {
        self.tip != current
    }

    /// Whether `layer`'s currently placed pixels cover `(x, y)`.
    pub(crate) fn covers_on_layer(&self, layer: usize, x: u32, y: u32) -> bool {
        self.layers
            .get(&layer)
            .is_some_and(|l| l.committed.contains(x, y))
    }

    /// Pristine originals for preview rasterization (read-only refs: the
    /// snapshot content can only change via `merge_layer`/`rebase`/ctor).
    pub(crate) fn originals(&self) -> impl Iterator<Item = &SortedRanges<u32>> {
        self.layers.values().map(|l| &l.original)
    }

    /// Commit the current transform: Clear previously committed ranges, Add
    /// freshly transformed originals. First action is `tracked`, the rest are
    /// not, so one ctrl-Z reverts the whole gesture across all layers.
    /// Re-arms `tip` exactly when history was written; a no-op leaves history
    /// and `tip` untouched.
    pub(crate) fn commit(mut self, masks: &mut MaskImage, img_roi: Roi<u32>) -> Option<Self> {
        let matrix = self.total;
        let mut should_abort = true;
        let computed = self
            .layers
            .iter_mut()
            .map(|(idx, ls)| {
                let new = transform_layer(&ls.original, &matrix, img_roi);
                should_abort &= new.as_ref() == Some(&ls.committed);
                (idx, ls, new)
            })
            .collect::<Vec<_>>();
        if should_abort {
            return Some(self);
        }
        let mut first = true;
        let mut clear = true;
        for (&layer, ls, new) in computed {
            let restore = ls
                .restore()
                .and_then(|i| SortedRanges::try_from_span_iter(i).ok());
            push_clear(masks, layer, ls.committed.clone(), first);
            first = false;
            if let Some(new) = new {
                push_add(masks, layer, new.clone(), false);
                clear = false;
                ls.committed = new;
                if let Some(restore) = restore {
                    push_add(masks, layer, restore, false);
                }
            }
        }
        self.tip = masks.last_history_action();
        (!clear).then_some(self)
    }

    /// Delete path: Clear every layer's placed ranges (re-adding absorbed
    /// background in the same hook), then drain the layers. First action is
    /// `tracked` so one ctrl-Z reverts the whole delete.
    pub(crate) fn delete_all(&mut self, masks: &mut MaskImage) {
        let mut first = true;
        for (layer, ls) in std::mem::take(&mut self.layers).into_iter() {
            let restore = ls
                .restore()
                .and_then(|i| SortedRanges::try_from_span_iter(i).ok());
            push_clear(masks, layer, ls.committed, first);
            first = false;
            if let Some(restore) = restore {
                push_add(masks, layer, restore, false);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn total_ref(&self) -> &Matrix3<f64> {
        &self.total
    }

    #[cfg(test)]
    pub(crate) fn tip_ref(&self) -> &Option<HistoryAction> {
        &self.tip
    }

    #[cfg(test)]
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[cfg(test)]
    pub(crate) fn has_layer(&self, layer: usize) -> bool {
        self.layers.contains_key(&layer)
    }

    #[cfg(test)]
    pub(crate) fn layer_ids(&self) -> Vec<usize> {
        self.layers.keys().copied().collect()
    }

    #[cfg(test)]
    pub(crate) fn original_of(&self, layer: usize) -> Option<&SortedRanges<u32>> {
        self.layers.get(&layer).map(|l| &l.original)
    }

    #[cfg(test)]
    pub(crate) fn committed_of(&self, layer: usize) -> Option<&SortedRanges<u32>> {
        self.layers.get(&layer).map(|l| &l.committed)
    }

    #[cfg(test)]
    pub(crate) fn first_original_cloned(&self) -> Option<SortedRanges<u32>> {
        self.layers.values().next().map(|l| l.original.clone())
    }

    #[cfg(test)]
    pub(crate) fn first_committed_cloned(&self) -> Option<SortedRanges<u32>> {
        self.layers.values().next().map(|l| l.committed.clone())
    }

    /// Pixel area of the first layer's committed content (test helper with
    /// selection-lifetime semantics: asserts an active single-entry selection).
    #[cfg(test)]
    pub(crate) fn first_committed_area(&self) -> usize {
        self.layers
            .values()
            .next()
            .expect("Has at least one layer")
            .committed
            .spans::<u32>()
            .map(|s| (s.x.end - s.x.start) as usize)
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn first_original_bounds(&self) -> Roi<u32> {
        self.layers
            .values()
            .next()
            .expect("Has at least one layer")
            .original
            .roi()
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
    let span_builder = SortedRangesSpanBuilder::new(b.roi());
    let mut b = b.fold_inline(span_builder, |b, n| {
        b.add(*n);
    });

    let r = SortedRanges::try_from_span_iter_minbounds(a.spans::<u32>().subtract(&mut b)).ok();
    (r, b.finish_all().build())
}

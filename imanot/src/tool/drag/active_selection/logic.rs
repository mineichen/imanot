use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use imask::{
    AffineTransformHeap, ImageDimension, ImaskSet, PipelineError, Roi, SortedRanges, Span, UnionAll,
};
use nalgebra::Matrix3;

use super::super::frame::Frame;
use super::super::transform::transform_layer;
use crate::{
    AffectedLayer, HistoryAction, HistoryActionAdd, HistoryActionClear, HistoryActionKind,
    MaskImage,
};

mod layer;

pub(in crate::tool::drag) use layer::LayerSelection;

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
        self.layers.values_mut().for_each(|l| l.rebase());
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

    pub(crate) fn fresh_from_sorted_ranges_iter(
        first: (usize, LayerSelection),
        parts: impl Iterator<Item = (usize, LayerSelection)>,
        tip: Option<HistoryAction>,
    ) -> Self {
        let mut content = first.1.original_roi();
        let layers = std::iter::once(first)
            .chain(parts.inspect(|x| {
                content = content.union(&x.1.original_roi());
            }))
            .collect::<BTreeMap<_, _>>();

        Self {
            total: Matrix3::identity(),
            frame: Frame::around(content),
            layers,
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
    /// [`super::ActiveSelection::merge_layers`]).
    fn merge_layer(
        &mut self,
        layer_id: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
    ) {
        self.frame.expand_to_cover(ranges.roi());
        match self.layers.entry(layer_id) {
            Entry::Vacant(x) => {
                x.insert(LayerSelection::fresh(ranges, background));
            }
            Entry::Occupied(x) => {
                let r = x.remove();
                if let Some(new_val) = r.update(&ranges) {
                    self.layers.insert(layer_id, new_val);
                }
            }
        }
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

    pub(super) fn transformed(
        &self,
        img_roi: Roi<u32>,
    ) -> Result<impl Iterator<Item = Span<u32>> + ImageDimension, PipelineError> {
        let matrix = self.total();
        UnionAll::new(
            self.layers
                .values()
                .map(LayerSelection::original)
                .filter_map(|original| {
                    AffineTransformHeap::new(original.spans::<u32>(), &matrix)
                        .ok()?
                        .clip(img_roi)
                        .ok()
                }),
        )
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
                let new = transform_layer(ls.original(), &matrix, img_roi);
                should_abort &= new.as_ref() == Some(&ls.committed);
                (idx, ls, new)
            })
            .collect::<Vec<_>>();
        if should_abort {
            return Some(self);
        }

        let actions = computed.into_iter().flat_map(|(&layer, ls, new)| {
            let restore = ls
                .restore()
                .and_then(|i| SortedRanges::try_from_span_iter(i).ok());
            let clear = build_clear_untracked(layer, ls.committed.clone());
            let add = new
                .map(move |new| {
                    let add = build_add_untracked(layer, new.clone());
                    ls.committed = new;
                    std::iter::once(add).chain(restore.map(|r| build_add_untracked(layer, r)))
                })
                .into_iter()
                .flatten();
            std::iter::once(clear).chain(add)
        });

        let r = add_history_actions(masks, actions);

        self.tip = masks.last_history_action();
        r.then_some(self)
    }

    /// Delete path: Clear every layer's placed ranges (re-adding absorbed
    /// background in the same hook), then drain the layers. First action is
    /// `tracked` so one ctrl-Z reverts the whole delete.
    pub(crate) fn delete_all(self, masks: &mut MaskImage) {
        add_history_actions(
            masks,
            self.layers.into_iter().flat_map(|(layer, ls)| {
                let restore = ls
                    .restore()
                    .and_then(|i| SortedRanges::try_from_span_iter(i).ok());
                let clear = build_clear_untracked(layer, ls.committed);
                let add = restore.map(|r| build_add_untracked(layer, r));
                std::iter::once(clear).chain(add)
            }),
        );
    }
}

fn add_history_actions(
    masks: &mut MaskImage,
    mut actions: impl Iterator<Item = HistoryAction>,
) -> bool {
    let r = actions.next().map_or(false, |mut first| {
        first.tracked = true;
        masks.add_history_action(first);
        for action in actions {
            masks.add_history_action(action);
        }
        true
    });
    r
}

fn build_add_untracked(layer: usize, pixel_area: SortedRanges<u32>) -> HistoryAction {
    HistoryAction {
        kind: HistoryActionKind::Add(HistoryActionAdd { pixel_area }),
        layer: AffectedLayer::Layer(layer),
        tracked: false,
    }
}

fn build_clear_untracked(layer: usize, ranges: SortedRanges<u32>) -> HistoryAction {
    HistoryAction {
        kind: HistoryActionKind::Clear(HistoryActionClear { ranges }),
        layer: AffectedLayer::Layer(layer),
        tracked: false,
    }
}
#[cfg(test)]
mod tests {
    use imask::ImaskSet;

    use nalgebra::{Point2, Vector2};

    use super::super::super::test_support::*;
    use super::*;

    fn mask_with_rect(x: u32, y: u32) -> (MaskImage, SortedRanges<u32>) {
        let original = rect_ranges(x, y, nz(5), nz(5));
        (mask(original.clone()), original)
    }

    #[test]
    fn commit_moves_content_and_frame() {
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut logic =
            ActiveSelectionLogic::fresh_single(0, original, None, masks.last_history_action());
        // Advance frame and matrix together, as a finished Move gesture would.
        let (frame, total) = logic.snapshot_transform();
        let delta = Vector2::new(5.0, 0.0);
        logic.set_transform(
            Frame {
                center: frame.center + delta,
                ..frame
            },
            total * Matrix3::new_translation(&delta),
        );
        let logic = logic
            .commit(&mut masks, img_roi())
            .expect("moved commit survives");
        assert_eq!(
            layer_pixels(&masks),
            Some(rect_ranges(15, 10, nz(5), nz(5)))
        );
        assert_eq!(logic.frame().center, Point2::new(17.5, 12.5));
        assert_eq!(logic.frame().half, Vector2::new(2.5, 2.5));
    }

    #[test]
    fn commit_offscreen_doesnt_drop_empty_selection() {
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut logic =
            ActiveSelectionLogic::fresh_single(0, original, None, masks.last_history_action());
        // Move fully out of the image: pixels are cleared, but the commit
        // still lands in history, so the logic survives (the tool drops its
        // selection only when `commit` returns `None`).
        let (frame, total) = logic.snapshot_transform();
        let delta = Vector2::new(-50.0, 0.0);
        logic.set_transform(
            Frame {
                center: frame.center + delta,
                ..frame
            },
            total * Matrix3::new_translation(&delta),
        );
        let tip_before = masks.last_history_action();
        assert!(logic.commit(&mut masks, img_roi()).is_some());
        assert_eq!(layer_pixels(&masks), None);
        assert_ne!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn shift_add_rebases_snapshot_without_history_write() {
        // Shift-add batch: the placed (committed) pixels are baked into a
        // pristine `original` and `total` resets, so later gestures transform
        // old and new content uniformly. No history write: the caller re-arms
        // `tip` together with the rebase.
        let (mut masks, original) = mask_with_rect(0, 0);
        let mut logic =
            ActiveSelectionLogic::fresh_single(0, original, None, masks.last_history_action());
        // Transform + commit: `original` stays pristine, `committed` moves.
        let (frame, total) = logic.snapshot_transform();
        let delta = Vector2::new(5.0, 0.0);
        logic.set_transform(
            Frame {
                center: frame.center + delta,
                ..frame
            },
            total * Matrix3::new_translation(&delta),
        );
        let mut logic = logic.commit(&mut masks, img_roi()).unwrap();
        let added = rect_ranges(5, 3, nz(2), nz(1));
        logic.merge_layers(
            std::iter::once((0, added, None)),
            masks.last_history_action(),
        );
        // Rebase: the placed content plus the added cluster form the new
        // pristine original; `total` is back to identity, `tip` re-armed.
        let entry = &logic.layers[&0];
        let placed = rect_ranges(5, 0, nz(5), nz(1));
        let union = SortedRanges::<u32>::try_from_span_iter(
            placed.spans::<u32>().union(entry.committed.spans()),
        )
        .unwrap();
        assert_eq!(entry.original(), &union);
        assert_eq!(logic.total, Matrix3::identity());
        assert_eq!(logic.tip, masks.last_history_action());
    }
}

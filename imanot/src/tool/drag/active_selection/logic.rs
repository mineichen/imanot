use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use imask::{ImageDimension, Roi, SortedRanges};
use nalgebra::Matrix3;

use super::super::frame::Frame;
use super::super::transform::transform_layer;
use crate::tool::drag::transform::{build_add_untracked, build_clear_untracked};
use crate::{HistoryAction, MaskImage};

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

    pub(crate) fn fresh_from_sorted_ranges_iter(
        first: (usize, LayerSelection),
        parts: impl Iterator<Item = (usize, LayerSelection)>,
        tip: Option<HistoryAction>,
    ) -> Self {
        let mut content = first.1.original.roi();
        let layers = std::iter::once(first)
            .chain(parts.inspect(|x| {
                content = content.union(&x.1.original.roi());
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use nalgebra::{Point2, Vector2};

    use super::*;
    use crate::{History, MaskDefaultActions, PixelAreaStack};

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    fn img_roi() -> Roi<u32> {
        Roi::from_dimensions(nz(100), nz(100))
    }

    fn rect_ranges(x: u32, y: u32, w: NonZeroU32, h: NonZeroU32) -> SortedRanges<u32> {
        SortedRanges::try_from_span_iter(Roi::new(x..x + w.get(), y..y + h.get()).into_spans())
            .unwrap()
    }

    fn mask_with_rect(x: u32, y: u32) -> (MaskImage, SortedRanges<u32>) {
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        let original = rect_ranges(x, y, nz(5), nz(5));
        masks.add(original.clone());
        (masks, original)
    }

    fn layer_pixels(masks: &MaskImage) -> Option<SortedRanges<u32>> {
        masks.subgroups_stack().get(0).map(|a| a.pixels.clone())
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
}

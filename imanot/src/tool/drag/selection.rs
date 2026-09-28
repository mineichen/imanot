use std::iter::once;

use egui::Pos2;
use imask::{ImaskSet, Roi, SortedRanges};

use crate::{
    AffectedLayer, DragTool, MaskImage,
    tool::drag::{
        LayerSelection,
        active_selection::{ActiveSelection, ActiveSelectionLogic},
        layer_ranges,
        transform::{clamp_pixel, cluster_at},
    },
};

impl DragTool {
    /// Delete all ranges in the current selection: Clear the currently placed
    /// ranges on every selected layer, then drop the selection.
    /// First action is `tracked`, the rest are not, so one ctrl-Z reverts the
    /// whole delete across all layers. Uncommitted gesture deltas are
    /// discarded — the mask itself is never touched during a gesture, so
    /// there is nothing to undo there.
    pub fn delete_selection(&mut self, masks: &mut MaskImage) {
        if let Some(sel) = self.selection.take() {
            self.gesture = None;
            sel.delete_all(masks);
        };
    }

    /// Click: select the cluster under the cursor (8-connected component of
    /// the topmost *affected* layer at the pixel), or deselect on empty
    /// (no affected layer covers the pixel). With `additive`
    /// (Shift held), the cluster is added to the existing selection — across
    /// layers — instead of replacing it; clicking an already-selected cluster
    /// is a no-op.
    pub fn select_pos(
        &mut self,
        masks: &mut MaskImage,
        pointer: Pos2,
        img_roi: Roi<u32>,
        additive: bool,
    ) {
        let img_w = img_roi.width().get() as usize;
        let img_h = img_roi.height().get() as usize;
        let (x, y) = clamp_pixel(pointer, img_w, img_h);
        // Topmost layer at the pixel restricted to this tool's `AffectedLayer`:
        // a click is "empty space" when no *affected* layer covers it, even if
        // an unaffected layer has a pixel there — then a non-additive click
        // clears the selection below.
        let all = masks
            .subgroups_stack()
            .iter_filtered(self.layer)
            .rev()
            // Might be ineffective
            .filter(|(_, area)| area.pixels.contains(x, y))
            .find_map(|(i, area)| {
                let selection = cluster_at(&area.pixels, x, y)?;
                Some((i, area, selection))
            });
        if all.is_none() && additive {
            return;
        }
        let layers = layer_ranges(all.into_iter());
        self.select_internal(masks, additive, layers)
    }

    /// Build a selection from a finished rect selection: all pixels inside the
    /// rect, independent of layer, restricted to the tool's `AffectedLayer`.
    /// With `additive` (Shift held), the rect's pixels are unioned into the
    /// existing selection instead of replacing it; an empty rect then keeps
    /// the selection unchanged.
    pub fn select_rect(&mut self, masks: &MaskImage, roi: Roi<u32>, additive: bool) {
        // `RectSelection` is shared tool infra still on `Rect`; convert at
        // the boundary — everything inside the drag tool uses `Roi`.
        let clipped_selected = masks
            .subgroups_stack()
            .iter_filtered(self.layer)
            .filter_map(move |(idx, area)| {
                let clipped = area.pixels.spans::<u32>().clip(roi).ok()?;
                Some((idx, area, clipped))
            });
        let layers = layer_ranges(clipped_selected);
        self.select_internal(masks, additive, layers)
    }

    /// Programmatically select whole mask layers: all pixels of every layer
    /// matched by `layer` (e.g. `2` or `0..3`), gathered from `masks` itself —
    /// unlike the old raw-span interface, no pixels can be named that have no
    /// corresponding ranges in the mask. Behaves like a fresh selection: tight
    /// ranges are rebuilt per layer and snapshotted as pristine originals,
    /// the frame tightly covers all selected pixels and any in-progress
    /// gesture is dropped. Layers without visible pixels are skipped; if
    /// nothing matches, the selection is dropped (an empty box is never
    /// shown). Whole layers are selected, so there is no non-selected
    /// remainder to restore: the background is always `None`. Unlike
    /// click/rect selection this does not intersect with the tool's own
    /// `AffectedLayer` filter — the caller names the layers explicitly.
    pub fn select_layers(&mut self, masks: &MaskImage, layer: impl Into<AffectedLayer>) {
        let selected = masks
            .subgroups_stack()
            .iter_filtered(layer.into())
            .map(|(idx, area)| (idx, area.pixels.clone(), None));

        self.select_internal(masks, false, selected);
        self.gesture = None;
    }

    fn select_internal<'m>(
        &mut self,
        masks: &MaskImage,
        additive: bool,
        mut layers: impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)> + 'm,
    ) {
        if let Some(first) = layers.next() {
            if additive && let Some(sel) = self.selection.as_mut() {
                sel.merge_layers(once(first).chain(layers), masks.last_history_action());
            } else {
                self.selection = Some(ActiveSelection::from_logic(
                    ActiveSelectionLogic::fresh_from_sorted_ranges_iter(
                        (first.0, LayerSelection::fresh(first.1, first.2)),
                        layers.map(|(idx, r, bg)| (idx, LayerSelection::fresh(r, bg))),
                        masks.last_history_action(),
                    ),
                ));
            }
        } else {
            // No box left to show; the preview dies with the selection.
            self.selection = None;
        }
    }
}

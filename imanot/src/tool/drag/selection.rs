use std::iter::{FusedIterator, once};

use egui::Pos2;
use imask::{
    ImageDimension, ImaskSet, Roi, SortedRanges, SortedRangesTightSpanBuilder, Span, SpanCluster,
};

use crate::{
    AffectedLayer, DragTool, HistoryAction, MaskImage, PixelArea,
    tool::drag::{
        DragToolSettings, LayerSelection,
        active_selection::{ActiveSelection, ActiveSelectionLogic},
        transform::clamp_pixel,
    },
};

/// Selection change computed by a `calc_select_*` method, without touching
/// the tool's state: [`SelectionUpdate::apply`] turns the current selection
/// into the next one.
pub(super) struct SelectionUpdate {
    /// Per-layer tight ranges and non-selected background, see
    /// [`layer_ranges`].
    layers: Vec<(usize, SortedRanges<u32>, Option<SortedRanges<u32>>)>,
    /// Merge into the current selection (if any) instead of replacing it.
    additive: bool,
    tip: Option<HistoryAction>,
}

impl SelectionUpdate {
    /// The selection following `current`: the layers are merged into it
    /// (`additive`) or replace it. No layers keep it when `additive`, else
    /// drop it (no box left to show; the preview dies with the selection).
    pub(super) fn apply(self, current: Option<ActiveSelection>) -> Option<ActiveSelection> {
        let mut layers = self.layers.into_iter();
        let Some(first) = layers.next() else {
            return current.filter(|_| self.additive);
        };
        match current {
            Some(mut sel) if self.additive => {
                sel.merge_layers(once(first).chain(layers), self.tip);
                Some(sel)
            }
            _ => Some(ActiveSelection::from_logic(
                ActiveSelectionLogic::fresh_from_sorted_ranges_iter(
                    (first.0, LayerSelection::fresh(first.1, first.2)),
                    layers.map(|(idx, r, bg)| (idx, LayerSelection::fresh(r, bg))),
                    self.tip,
                ),
            )),
        }
    }
}

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

    /// Click: select the cluster under the cursor (see
    /// [`DragTool::calc_select_pos`]).
    pub fn select_at(
        &mut self,
        masks: &mut MaskImage,
        pointer: Pos2,
        img_roi: Roi<u32>,
        additive: bool,
    ) {
        let update = self
            .settings
            .calc_select_at(masks, pointer, img_roi, additive);
        self.selection = update.apply(self.selection.take());
    }

    /// Build a selection from a finished rect selection (see
    /// [`DragTool::calc_select_rect`]).
    pub fn select_rect(&mut self, masks: &MaskImage, roi: Roi<u32>, additive: bool) {
        let update = self.settings.calc_select_rect(masks, roi, additive);
        self.selection = update.apply(self.selection.take());
    }

    /// Programmatically select whole mask layers (see
    /// [`DragTool::calc_select_layers`]); any in-progress gesture is dropped.
    pub fn select_layers(&mut self, masks: &MaskImage, layer: impl Into<AffectedLayer>) {
        let update = self.settings.calc_select_layers(masks, layer);
        self.selection = update.apply(self.selection.take());
    }
}
impl DragToolSettings {
    /// Click: select the cluster under the cursor (8-connected component of
    /// the topmost *affected* layer at the pixel), or deselect on empty
    /// (no affected layer covers the pixel). With `additive`
    /// (Shift held), the cluster is added to the existing selection — across
    /// layers — instead of replacing it; clicking an already-selected cluster
    /// is a no-op, as is an additive click on empty space.
    pub(super) fn calc_select_at(
        &self,
        masks: &MaskImage,
        pointer: Pos2,
        img_roi: Roi<u32>,
        additive: bool,
    ) -> SelectionUpdate {
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
        SelectionUpdate {
            layers: layer_ranges(all.into_iter()).collect(),
            additive,
            tip: masks.last_history_action(),
        }
    }
    /// Build a selection from a finished rect selection: all pixels inside the
    /// rect, independent of layer, restricted to the tool's `AffectedLayer`.
    /// With `additive` (Shift held), the rect's pixels are unioned into the
    /// existing selection instead of replacing it; an empty rect then keeps
    /// the selection unchanged.
    pub(super) fn calc_select_rect(
        &self,
        masks: &MaskImage,
        roi: Roi<u32>,
        additive: bool,
    ) -> SelectionUpdate {
        // `RectSelection` is shared tool infra still on `Rect`; convert at
        // the boundary — everything inside the drag tool uses `Roi`.
        let clipped_selected = masks
            .subgroups_stack()
            .iter_filtered(self.layer)
            .filter_map(move |(idx, area)| {
                let clipped = area.pixels.spans::<u32>().clip(roi).ok()?;
                Some((idx, area, clipped))
            });
        SelectionUpdate {
            layers: layer_ranges(clipped_selected).collect(),
            additive,
            tip: masks.last_history_action(),
        }
    }
    /// Programmatically select whole mask layers: all pixels of every layer
    /// matched by `layer` (e.g. `2` or `0..3`), gathered from `masks` itself —
    /// unlike the old raw-span interface, no pixels can be named that have no
    /// corresponding ranges in the mask. Behaves like a fresh selection: tight
    /// ranges are rebuilt per layer and snapshotted as pristine originals and
    /// the frame tightly covers all selected pixels. Layers without visible
    /// pixels are skipped; if nothing matches, the selection is dropped (an
    /// empty box is never shown). Whole layers are selected, so there is no
    /// non-selected remainder to restore: the background is always `None`.
    /// Unlike click/rect selection this does not intersect with the tool's
    /// own `AffectedLayer` filter — the caller names the layers explicitly.
    pub(super) fn calc_select_layers(
        &self,
        masks: &MaskImage,
        layer: impl Into<AffectedLayer>,
    ) -> SelectionUpdate {
        let layers = masks
            .subgroups_stack()
            .iter_filtered(layer.into())
            .map(|(idx, area)| (idx, area.pixels.clone(), None))
            .collect();
        SelectionUpdate {
            layers,
            additive: false,
            tip: masks.last_history_action(),
        }
    }
}

/// Build tight per-layer selection content from pre-filtered layer spans:
/// each item carries the layer index, the layer's full [`PixelArea`] and the
/// selected span stream (already clipped by the caller when selecting a
/// sub-region, e.g. [`DragTool::select_rect`]). Returns the rebuilt tight
/// ranges plus the layer's non-selected remainder as background (layer minus
/// snapshot, so later commits can restore outsiders under cleared
/// footprints) — `None` when everything was selected. Layers whose stream is
/// empty are skipped.
///
/// Single pass over the selected spans: they are fed into the tight ranges
/// builder inline ([`ImaskSet::fold_inline`]) while `subtract` consumes them
/// for the background, instead of building the ranges first and re-walking
/// them inside `subtract`.
fn layer_ranges<'m, S>(
    layers: impl Iterator<Item = (usize, &'m PixelArea, S)> + 'm,
) -> impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)> + 'm
where
    S: Iterator<Item = Span<u32>> + ImageDimension + FusedIterator,
{
    layers.filter_map(|(idx, area, selected)| {
        let span_builder = SortedRangesTightSpanBuilder::new(selected.roi(), &selected);
        let mut selected = selected.fold_inline(span_builder, |b, s| b.add(*s));
        // Materialize the background before `finish_all` drains the selected
        // stream's remainder into the builder; an empty background is `None`,
        // not a reason to skip the layer.
        let background = SortedRanges::try_from_span_iter_minbounds(
            area.pixels.spans::<u32>().subtract(&mut selected),
        )
        .ok();
        let ranges = selected.finish_all().build().ok()?;
        Some((idx, ranges, background))
    })
}

/// Find the 8-connected cluster of `ranges` containing pixel `(x, y)`.
/// Returns `None` if no span covers the pixel.
fn cluster_at(ranges: &SortedRanges<u32>, x: u32, y: u32) -> Option<SpanCluster<u32>> {
    ranges.spans::<u32>().cluster().find(|cluster| {
        cluster.roi().contains(&x, &y)
            && cluster
                .clone()
                .skip_while(|s| s.y < y)
                .take_while(|s| s.y == y && s.x.start <= x)
                .any(|s| x < s.x.end)
    })
}

#[cfg(test)]
mod tests {
    use crate::tool::drag::test_support::ranges_from_spans;

    use super::*;

    #[test]
    fn cluster_at_finds_covering_cluster_only() {
        let ranges = disjoint_rects();
        let left = cluster_at(&ranges, 0, 0).unwrap();
        assert_eq!(left.roi(), Roi::new(0..2, 0..1));
        assert_eq!(left.clone().count(), 1);
        let right = cluster_at(&ranges, 6, 3).unwrap();
        assert_eq!(right.roi(), Roi::new(5..7, 3..4));
        // Pixels between/outside clusters select nothing.
        assert!(cluster_at(&ranges, 3, 0).is_none());
        assert!(cluster_at(&ranges, 0, 5).is_none());
    }

    #[test]
    fn cluster_connects_diagonally() {
        // 8-connectivity: diagonally touching pixels form one cluster.
        let ranges = ranges_from_spans(&[Span::new(0..1, 0u32), Span::new(1..2, 1u32)]);
        let cluster = cluster_at(&ranges, 0, 0).unwrap();
        assert_eq!(cluster.count(), 2);
    }

    fn disjoint_rects() -> SortedRanges<u32> {
        ranges_from_spans(&[Span::new(0..2, 0u32), Span::new(5..7, 3u32)])
    }
}

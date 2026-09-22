use imask::{AffineTransformHeap, ImageDimension, Roi, SortedRanges, Span, UnionAll};
use nalgebra::Matrix3;

use crate::{HistoryAction, ImagePainter, MaskImage};

use super::frame::Frame;
use super::{overlay::draw_overlay, transform::clip_heap_to_image};

pub(crate) mod logic;
pub(crate) mod preview;

pub(crate) use logic::ActiveSelectionLogic;
pub(in crate::tool::drag) use logic::LayerSelection;

/// A live selection: pure snapshot logic plus its small preview texture. The
/// preview starts hidden (the first render uploads it) and dies with the
/// selection, so dropping or replacing a selection needs no explicit
/// invalidation; in-place mutations hide it via
/// [`ActiveSelection::merge_layers`] (rebase + union changed the snapshot
/// content the preview is rasterized from) while `DragTool` hides it for
/// gesture restore and the offscreen case.
/// Commits need no invalidation: the placed pixels are exactly what the
/// preview already shows.
pub(crate) struct ActiveSelection {
    logic: ActiveSelectionLogic,
    preview: preview::PreviewState,
}

impl ActiveSelection {
    /// Fresh single-layer selection with a hidden preview (see
    /// [`ActiveSelectionLogic::fresh_single`]).
    pub(crate) fn fresh_single(
        idx: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
        tip: Option<HistoryAction>,
    ) -> Self {
        Self::from_logic(ActiveSelectionLogic::fresh_single(
            idx, ranges, background, tip,
        ))
    }
    /// Render an idle (no active transform) selection: the black overlay of
    /// the actually selected pixels plus the frame overlay. The frame alone
    /// is not enough — e.g. a rect selection only covers the dragged box,
    /// not every pixel inside it. Paints the uploaded pixels when they are
    /// still valid, so idle frames cost a single texture draw and no
    /// rasterization.

    pub(crate) fn render_selection(
        &mut self,
        egui_ctx: &egui::Context,
        painter: &mut ImagePainter,
        img_roi: Roi<u32>,
    ) {
        if self.preview.is_visible() {
            self.preview.paint(painter);
            draw_overlay(painter, self.logic.frame());
        } else {
            self.render_transform(egui_ctx, painter, img_roi, false);
        }
    }

    /// Wrap finished logic with a hidden preview; the next render uploads it.
    pub(crate) fn from_logic(logic: ActiveSelectionLogic) -> Self {
        Self {
            logic,
            preview: preview::PreviewState::new(),
        }
    }

    /// Shift-add batch with a single preview invalidation (see
    /// [`ActiveSelectionLogic::merge_layers`]).
    pub(crate) fn merge_layers(
        &mut self,
        parts: impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)>,
        tip: Option<HistoryAction>,
    ) {
        self.logic.merge_layers(parts, tip);
        self.preview.hide();
    }

    /// See [`ActiveSelectionLogic::frame`].
    pub(crate) fn frame(&self) -> &Frame {
        self.logic.frame()
    }

    /// See [`ActiveSelectionLogic::snapshot_transform`].
    pub(crate) fn snapshot_transform(&self) -> (Frame, Matrix3<f64>) {
        self.logic.snapshot_transform()
    }

    /// Live-gesture update without preview invalidation (see
    /// [`ActiveSelectionLogic::set_transform`]).
    pub(crate) fn set_transform(&mut self, frame: Frame, total: Matrix3<f64>) {
        self.logic.set_transform(frame, total);
    }

    /// Cancelled-gesture restore: revert `frame`+`total` and invalidate the
    /// uploaded pixels, which were rasterized from the discarded state.
    pub(crate) fn restore_gesture(&mut self, frame: Frame, total: Matrix3<f64>) {
        self.logic.set_transform(frame, total);
        self.preview.hide();
    }

    /// See [`ActiveSelectionLogic::is_stale`].
    pub(crate) fn is_stale(&self, current: Option<HistoryAction>) -> bool {
        self.logic.is_stale(current)
    }

    /// See [`ActiveSelectionLogic::covers_on_layer`].
    pub(crate) fn covers_on_layer(&self, layer: usize, x: u32, y: u32) -> bool {
        self.logic.covers_on_layer(layer, x, y)
    }

    /// Commit without preview invalidation (see
    /// [`ActiveSelectionLogic::commit`]).
    pub(crate) fn commit_transform(
        mut self,
        masks: &mut MaskImage,
        img_roi: Roi<u32>,
    ) -> Option<Self> {
        self.logic = self.logic.commit(masks, img_roi)?;
        Some(self)
    }

    /// See [`ActiveSelectionLogic::delete_all`].
    pub(crate) fn delete_all(self, masks: &mut MaskImage) {
        self.logic.delete_all(masks);
    }

    /// Render preview texture + oriented overlay for an active gesture. The
    /// overlay draws the frame itself, so it always matches the selection
    /// exactly, at any rotation.
    ///
    /// Pure moves (`allow_reposition`) only shift the already-uploaded pixels
    /// to the new bounds origin — no rasterization, no upload. Anything else
    /// re-rasterizes the transformed snapshot into the small texture.
    pub(crate) fn render_transform(
        &mut self,
        egui_ctx: &egui::Context,
        painter: &mut ImagePainter,
        img_roi: imask::Roi<u32>,
        allow_reposition: bool,
    ) {
        let matrix = self.logic.total();
        // One lazy iterator per layer (analytic transform ∩ image), merged
        // into a single ordered, self-bounded span stream: no span is ever
        // collected.
        let chains = self.logic.originals().filter_map(|original| {
            let heap = AffineTransformHeap::new(original.spans::<u32>(), &matrix).ok()?;
            clip_heap_to_image(heap, img_roi)
        });

        match UnionAll::new(chains) {
            Ok(all) => {
                self.preview.show(egui_ctx, painter, all, allow_reposition);
            }
            Err(_) => {
                self.preview.hide();
            }
        }
        draw_overlay(painter, self.logic.frame());
    }

    #[cfg(test)]
    pub(crate) fn total_for_test(&self) -> Matrix3<f64> {
        *self.logic.total_ref()
    }

    #[cfg(test)]
    pub(crate) fn tip_for_test(&self) -> Option<HistoryAction> {
        self.logic.tip_ref().clone()
    }

    #[cfg(test)]
    pub(crate) fn layer_count_for_test(&self) -> usize {
        self.logic.layer_count()
    }

    #[cfg(test)]
    pub(crate) fn has_layer_for_test(&self, layer: usize) -> bool {
        self.logic.has_layer(layer)
    }

    #[cfg(test)]
    pub(crate) fn layer_ids_for_test(&self) -> Vec<usize> {
        self.logic.layer_ids()
    }

    #[cfg(test)]
    pub(crate) fn original_of_for_test(&self, layer: usize) -> Option<SortedRanges<u32>> {
        self.logic.original_of(layer).cloned()
    }

    #[cfg(test)]
    pub(crate) fn committed_of_for_test(&self, layer: usize) -> Option<SortedRanges<u32>> {
        self.logic.committed_of(layer).cloned()
    }

    #[cfg(test)]
    pub(crate) fn first_original_for_test(&self) -> Option<SortedRanges<u32>> {
        self.logic.first_original_cloned()
    }

    #[cfg(test)]
    pub(crate) fn first_committed_for_test(&self) -> Option<SortedRanges<u32>> {
        self.logic.first_committed_cloned()
    }

    #[cfg(test)]
    pub(crate) fn first_original_bounds_for_test(&self) -> imask::Roi<u32> {
        self.logic.first_original_bounds()
    }

    #[cfg(test)]
    pub(crate) fn preview_visible(&self) -> bool {
        self.preview.is_visible()
    }

    /// Pixel area of the first layer's committed content (single-entry
    /// selection lifetime helper for tests).
    #[cfg(test)]
    pub(crate) fn first_committed_area(&self) -> usize {
        self.logic.first_committed_area()
    }
}

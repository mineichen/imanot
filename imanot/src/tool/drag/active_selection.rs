use imask::{Roi, SortedRanges};
use nalgebra::{Matrix3, Point2};

use crate::{HistoryAction, ImagePainter, MaskImage};

use super::frame::Frame;
use super::gesture::TransformGesture;
use overlay::draw_overlay;

pub(crate) mod logic;
mod overlay;
pub(crate) mod preview;

pub(crate) use logic::ActiveSelectionLogic;
pub(in crate::tool::drag) use logic::LayerSelection;
pub(super) use overlay::*;

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

    /// Follow a running transform gesture to `pointer`, without preview
    /// invalidation (see [`TransformGesture::apply`]).
    pub(super) fn apply_gesture(
        &mut self,
        gesture: &TransformGesture,
        pointer: Point2<f64>,
        shift: bool,
    ) {
        let (frame, total) = gesture.apply(pointer, shift);
        self.logic.set_transform(frame, total);
    }

    /// Cancel a transform gesture: revert to the frame and matrix it started
    /// from and invalidate the uploaded pixels, which were rasterized from
    /// the discarded state.
    pub(super) fn cancel_gesture(&mut self, gesture: TransformGesture) {
        let (frame, total) = gesture.base_state();
        self.logic.set_transform(frame, total);
        self.preview.hide();
    }

    /// See [`ActiveSelectionLogic::is_stale`].
    pub(crate) fn is_stale(&self, current: Option<HistoryAction>) -> bool {
        self.logic.is_stale(current)
    }

    #[cfg(test)]
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
        if let Ok(all) = self.logic.transformed(img_roi) {
            self.preview.show(egui_ctx, painter, all, allow_reposition);
        } else {
            self.preview.hide();
        }
        draw_overlay(painter, self.logic.frame());
    }
}

#[cfg(test)]
mod tests {
    use imask::ImaskSet;

    use nalgebra::Vector2;

    use super::super::test_support::*;
    use super::*;

    /// Layer with a 10x5 block at (10,10) plus a disjoint 10x5 outsider
    /// block at (40,30).
    fn mask_with_outsiders() -> (MaskImage, SortedRanges<u32>, SortedRanges<u32>) {
        let block = rect_ranges(10, 10, nz(10), nz(5));
        let outsiders = rect_ranges(40, 30, nz(10), nz(5));
        let combined =
            SortedRanges::try_from_span_iter(block.spans::<u32>().union(outsiders.spans()))
                .unwrap();
        (mask(combined), block, outsiders)
    }

    #[test]
    fn commit_keeps_preview_alive() {
        // Dropping the selection at a new position must not re-rasterize:
        // the placed pixels are exactly what the preview already shows.
        let (mut masks, block, outsiders) = mask_with_outsiders();
        let mut selection =
            ActiveSelection::from_logic(ActiveSelectionLogic::fresh_from_sorted_ranges_iter(
                (0, LayerSelection::fresh(block, Some(outsiders))),
                std::iter::empty(),
                masks.last_history_action(),
            ));
        let ctx = egui::Context::default();
        let screen =
            egui::Rect::from_min_max(egui::Pos2::new(0.0, 0.0), egui::Pos2::new(100.0, 100.0));
        let mut painter =
            ImagePainter::new(ctx.layer_painter(egui::LayerId::background()), screen, 1.0);
        selection.render_transform(&ctx, &mut painter, img_roi(), false);
        assert!(selection.preview.is_visible());
        // Move by 5px, as a finished Move gesture would.
        let (frame, total) = selection.snapshot_transform();
        let press = frame.center;
        let gesture = TransformGesture::begin(HoverPart::Inside, press, (frame, total)).unwrap();
        selection.apply_gesture(&gesture, press + Vector2::new(5.0, 0.0), false);
        let selection = selection
            .commit_transform(&mut masks, img_roi())
            .expect("moved commit survives");
        // Pixels landed (moved block + untouched outsiders), and the preview
        // survived the drop. (Span comparison: the mask keeps the coordinate
        // frame's bounds, not tight ones.)
        let expected = SortedRanges::<u32>::try_from_span_iter(
            rect_ranges(15, 10, nz(10), nz(5))
                .spans::<u32>()
                .union(rect_ranges(40, 30, nz(10), nz(5)).spans()),
        )
        .unwrap();
        assert_eq!(
            layer_pixels(&masks).map(|p| p.spans::<u32>().collect::<Vec<_>>()),
            Some(expected.spans::<u32>().collect())
        );
        assert!(outsider_block_ok(&masks));
        assert!(selection.preview.is_visible());
    }
}

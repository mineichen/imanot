use std::sync::Arc;

use egui::{CursorIcon, Pos2, Vec2};
use futures::FutureExt;
use imask::Roi;
use nalgebra::Point2;

use crate::{
    AffectedLayer, PanTool, RectSelection, Tool, ToolContext, ToolFactory,
    tool::drag::active_selection::{HoverGesture, LayerSelection},
};

mod active_selection;
mod frame;
mod gesture;
mod selection;
mod transform;

#[cfg(test)]
mod test_support;

use active_selection::ActiveSelection;
use gesture::{Gesture, TransformGesture};
use transform::clamp_pixel;

/// Drag-select + transform tool. See `DRAG_TOOL_REFINED.md` for the full plan.
///
/// This is deliberately *not* a `DrawTool`: it never adds or removes pixels
/// itself (commits only move already-selected pixels), so there is no
/// insert/clear `Mode` to switch. It only owns the `AffectedLayer` filter.
#[derive(Default)]
#[non_exhaustive]
pub struct DragTool {
    selection: Option<ActiveSelection>,
    gesture: Option<Gesture>,
    settings: DragToolSettings,
}

impl DragTool {
    pub fn set_layer(&mut self, layer: impl Into<AffectedLayer>) -> &mut Self {
        self.settings.layer = layer.into();
        self
    }

    pub fn set_pan_on_drag(&mut self, pan_on_drag: bool) -> &mut Self {
        self.settings.pan_on_drag = pan_on_drag;
        self
    }

    pub fn create_factory() -> ToolFactory {
        Arc::new(|_| async { Ok(Box::new(DragTool::default()) as Box<dyn Tool>) }.boxed_local())
    }

    pub fn create_factory_with(modifier: impl Fn(&mut DragTool) + 'static) -> ToolFactory {
        Arc::new(move |_| {
            let mut tool = DragTool::default();
            modifier(&mut tool);
            async { Ok(Box::new(tool) as Box<dyn Tool>) }.boxed_local()
        })
    }
}

impl Tool for DragTool {
    fn handle_interaction(&mut self, ctx: ToolContext) {
        (self.selection, self.gesture) = self.settings.handle_interaction_immutable(
            ctx,
            self.selection.take(),
            self.gesture.take(),
        );
    }
}

#[derive(Default)]
struct DragToolSettings {
    layer: AffectedLayer,
    /// Dragging on empty space pans instead of rect-selecting.
    pub pan_on_drag: bool,
}

impl DragToolSettings {
    fn handle_interaction_immutable(
        &self,
        mut ctx: ToolContext,
        mut selection: Option<ActiveSelection>,
        mut gesture: Option<Gesture>,
    ) -> (Option<ActiveSelection>, Option<Gesture>) {
        if let Some(sel) = selection.take() {
            *ctx.postpone_new_images = true;
            const DELETE_KEYS: [egui::Key; 2] = [egui::Key::Delete, egui::Key::Backspace];
            let delete = ctx
                .egui
                .input(|i| DELETE_KEYS.iter().any(|k| i.key_pressed(*k)));
            // A stale selection (history changed underneath: undo, redo or
            // another tool's commit) no longer matches the mask, so it is dropped
            // together with its gesture — before Delete could clear pixels from
            // the outdated snapshot. Delete removes a live selection.
            if !sel.is_stale(ctx.image.masks.last_history_action()) {
                if delete {
                    sel.delete_all(&mut ctx.image.masks);
                } else {
                    selection = Some(sel);
                }
            }
        };
        // Escape cancels a running gesture, or drops the idle selection. A
        // cancelled transform reverts to the pre-gesture frame and matrix;
        // the mask was never touched, so there is nothing to undo there.
        // Other gestures just end.
        if ctx.egui.input(|i| i.key_pressed(egui::Key::Escape))
            && let (Some(gesture), Some(mut sel)) = (gesture.take(), selection.take())
        {
            if let Gesture::Transform(t) = gesture {
                sel.cancel_gesture(t);
            }
            selection = Some(sel);
        }

        let img_roi = {
            let (w, h) = ctx.image.image.adjust.dimensions();
            Roi::from_dimensions(w, h)
        };
        let pointer_screen = ctx
            .response
            .interact_pointer_pos()
            .or_else(|| ctx.response.hover_pos());
        let pointer = pointer_screen.map(|p| ctx.painter.screen_to_image(p));

        (selection, gesture) = match gesture {
            Some(Gesture::Rect(rect)) => self.step_rect(rect, selection, &mut ctx),
            Some(Gesture::Transform(transform)) => {
                let pointer = pointer.map(|p| Point2::new(p.x as f64, p.y as f64));
                step_transform(transform, selection, &mut ctx, pointer, img_roi)
            }
            None => self.step_idle(selection, &mut ctx, pointer, pointer_screen, img_roi),
        };

        if let Some(s) = selection.as_mut()
            && matches!(&gesture, Some(Gesture::Rect(_)) | None)
        {
            s.render_selection(ctx.egui, &mut *ctx.painter, img_roi)
        }

        if self.pan_on_drag && gesture.is_none() && ctx.response.dragged() {
            PanTool::default().handle_interaction(ctx);
        }
        (selection, gesture)
    }

    /// Rect-select step: selects on release (Shift adds instead of
    /// replacing) and ends; otherwise keeps the rubber band running.
    fn step_rect(
        &self,
        mut rect: RectSelection,
        selection: Option<ActiveSelection>,
        ctx: &mut ToolContext,
    ) -> (Option<ActiveSelection>, Option<Gesture>) {
        if let Some(result) = rect.drag_finished(ctx) {
            let additive = ctx.egui.input(|i| i.modifiers.shift);
            if let Ok(roi) = result.rect() {
                let update = self.calc_select_rect(&ctx.image.masks, roi.into(), additive);
                return (update.apply(selection), None);
            } else {
                return (None, None);
            }
        }
        if !ctx.response.dragged() {
            return (selection, None);
        }

        (selection, Some(Gesture::Rect(rect)))
    }

    /// No gesture running: a drag start begins one, a click (Shift adds)
    /// selects the cluster under the pointer.
    fn step_idle(
        &self,
        selection: Option<ActiveSelection>,
        ctx: &mut ToolContext,
        pointer: Option<Pos2>,
        pointer_screen: Option<Pos2>,
        img_roi: Roi<u32>,
    ) -> (Option<ActiveSelection>, Option<Gesture>) {
        let img_w = img_roi.width().get() as usize;
        let img_h = img_roi.height().get() as usize;
        let gesture = ctx
            .response
            .drag_started()
            .then(|| self.begin_gesture(ctx, selection.as_ref(), img_w, img_h))
            .flatten();
        if ctx.response.clicked()
            && !ctx.response.drag_stopped()
            && let Some(p) = pointer
        {
            let additive = ctx.egui.input(|i| i.modifiers.shift);
            let update = self.calc_select_at(&ctx.image.masks, p, img_roi, additive);
            (update.apply(selection), gesture)
        } else {
            self.hover_cursor(ctx, selection.as_ref(), gesture.as_ref(), pointer_screen);
            (selection, gesture)
        }
    }

    /// Decide what a fresh drag does on `selection`.
    /// Hit-testing and gesture origins use the PRESS position: with click +
    /// drag sensing, `drag_started()` only fires after the pointer moved past
    /// `max_click_dist`, so the current position may already have left small
    /// hit targets (anchors, rotate handle).
    fn begin_gesture(
        &self,
        ctx: &mut ToolContext,
        selection: Option<&ActiveSelection>,
        img_w: usize,
        img_h: usize,
    ) -> Option<Gesture> {
        let press_screen = ctx
            .response
            .interact_pointer_pos()
            .map(|cur| cur - ctx.response.total_drag_delta().unwrap_or(Vec2::ZERO))?;
        let pointer = ctx.painter.screen_to_image(press_screen);
        let r = selection.and_then(|s| {
            let press = Point2::new(pointer.x as f64, pointer.y as f64);
            let part = active_selection::hit_test(ctx, press_screen, s);
            TransformGesture::begin(part, press, s.snapshot_transform())
        });
        match r {
            Some(transform) => Some(Gesture::Transform(transform)),
            None => self.start_empty_space_gesture(ctx, pointer, img_w, img_h),
        }
    }

    /// Empty-space interaction: no gesture (the drag pans, see
    /// `handle_interaction`) when `pan_on_drag` is set and no layer covers
    /// the cursor, else start a rect selection.
    fn start_empty_space_gesture(
        &self,
        ctx: &mut ToolContext,
        pointer: Pos2,
        img_w: usize,
        img_h: usize,
    ) -> Option<Gesture> {
        if self.pan_on_drag {
            let (x, y) = clamp_pixel(pointer, img_w, img_h);
            // No layer under the cursor: no gesture, pan.
            ctx.image.masks.find_layer_at((x, y))?;
        }
        let mut selection = RectSelection::default();
        let _ = selection.drag_finished(ctx);
        Some(Gesture::Rect(selection))
    }

    /// Cursor for the hover state, or for a `gesture` that just began.
    fn hover_cursor(
        &self,
        ctx: &ToolContext,
        selection: Option<&ActiveSelection>,
        gesture: Option<&Gesture>,
        pointer_screen: Option<Pos2>,
    ) {
        let is_hovered_affected =
            || (ctx.image.masks.hover_layer()).is_some_and(|l| self.layer.affects(l));
        let icon = match (gesture, selection, pointer_screen) {
            (Some(Gesture::Transform(t)), Some(sel), _) => t.cursor(sel.frame()),
            (Some(Gesture::Rect(_)), _, _) => CursorIcon::Crosshair,
            // Not on a drag start: `begin_gesture` already hit-tested.
            (None, sel, Some(p)) if !ctx.response.drag_started() => {
                match sel.map(|s| (s, active_selection::hit_test(ctx, p, s))) {
                    Some((s, HoverGesture::Resize(a))) => {
                        active_selection::resize_cursor(s.frame(), a)
                    }
                    Some((_, HoverGesture::Rotate)) => CursorIcon::Grab,
                    Some((_, HoverGesture::Move)) => CursorIcon::Move,
                    _ if is_hovered_affected() => CursorIcon::PointingHand,
                    _ => return,
                }
            }
            _ => return,
        };
        ctx.egui.set_cursor_icon(icon);
    }
}

/// Transform step: follow the pointer, and commit on mouseup (or when
/// the pointer is gone), which ends the gesture. Without a selection the
/// gesture just ends.
fn step_transform(
    transform: TransformGesture,
    selection: Option<ActiveSelection>,
    ctx: &mut ToolContext,
    pointer: Option<Point2<f64>>,
    img_roi: Roi<u32>,
) -> (Option<ActiveSelection>, Option<Gesture>) {
    let Some(mut sel) = selection else {
        return (None, None);
    };
    // Shift alters the active gesture: exact unsnapped rotation angles,
    // or a corner resize free of the aspect-ratio lock.
    let shift = ctx.egui.input(|i| i.modifiers.shift);
    if let Some(p) = pointer {
        sel.apply_gesture(&transform, p, shift);
    }
    if ctx.response.drag_stopped() || pointer.is_none() {
        // See [`ActiveSelection::commit_transform`]. If nothing remains
        // visible, the selection is dropped — an empty box is never
        // shown.
        (sel.commit_transform(&mut ctx.image.masks, img_roi), None)
    } else {
        sel.render_transform(ctx.egui, &mut *ctx.painter, img_roi, transform.is_move());
        ctx.egui.set_cursor_icon(transform.cursor(sel.frame()));
        (Some(sel), Some(Gesture::Transform(transform)))
    }
}

#[cfg(test)]
mod tests {
    use imask::{ImaskSet, SortedRanges, Span};
    use nalgebra::{Matrix3, Vector2};

    use super::frame::Anchor;
    use super::test_support::*;
    use super::transform::transform_layer;
    use super::*;
    use crate::{MaskDefaultActions, MaskImage, TestResult};

    #[test]
    fn fractional_move_snaps_to_whole_pixels() -> TestResult {
        // App drag deltas are fractional, but masks live on whole pixels: the
        // move delta snaps so the rasterization stays pixel-exact. Without
        // snapping, half-integer matrices make the span rasterizer emit
        // fringe rows outside its analytic bounds; those fringe spans enter
        // `committed`, and the *next* commit's Clear erases outsider pixels
        // sharing the rows.
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(12.0, 12.0), IMG_ROI, false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(44.5, 32.5))?;
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert_eq!(
            sel.snapshot_transform().1,
            Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        assert_eq!(sel.frame().center, Point2::new(45.0, 32.5));
        tool.commit(&mut masks, IMG_ROI);
        // Committed content is pixel-exact: the moved block lands exactly on
        // the outsider block, no fringe rows anywhere. (Span comparison: the
        // mask keeps the coordinate frame's bounds, not tight ones.)
        assert_eq!(
            layer_pixels(&masks).map(|p| p.spans::<u32>().collect::<Vec<_>>()),
            Some(Roi::new(40..50, 30..35).into_spans().collect())
        );
        // Second fractional move: still exact, only it and the outsiders
        // remain.
        drag_move(&mut tool, Point2::new(44.5, 32.5), Point2::new(20.4, 22.6))?;
        tool.commit(&mut masks, IMG_ROI);
        let selection = tool.selection.as_ref().ok_or("has a selection")?;
        assert_eq!(
            selection.snapshot_transform().1,
            Matrix3::new_translation(&Vector2::new(-24.0, -10.0))
                * Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        let expected = SortedRanges::<u32>::try_from_span_iter(
            Roi::new(16u16..26, 20..25)
                .into_spans()
                .union(Roi::new(40..50, 30..35).into_spans()),
        )?;
        assert_eq!(
            layer_pixels(&masks).map(|p| p.spans::<u32>().collect::<Vec<_>>()),
            Some(expected.spans::<u32>().collect())
        );
        Ok(())
    }

    #[test]
    fn delete_restores_absorbed_outsiders() -> TestResult {
        // Move the block exactly onto the outsider block (union), then press
        // Delete: selection content goes, outsiders come back byte-identical.
        let mut masks = mask_blocks();
        let outsider_before: Vec<Span<u32>> = layer_pixels(&masks)
            .ok_or("Test setup painted a block")?
            .spans::<u32>()
            .filter(|s| (30..35).contains(&s.y))
            .collect();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(12.0, 12.0), IMG_ROI, false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(45.0, 32.5))?;
        tool.commit(&mut masks, IMG_ROI);
        tool.delete_selection(&mut masks);
        assert!(tool.selection.is_none());
        // Only the outsiders remain: selection content (original spot and
        // moved union) is gone, outsiders byte-identical.
        let rem = layer_pixels(&masks).ok_or("Outsider block survives the delete")?;
        assert_eq!(rem.spans::<u32>().collect::<Vec<_>>(), outsider_before);
        Ok(())
    }

    fn mask_with_rect(x: u32, y: u32) -> TestResult<MaskImage> {
        Ok(mask(SortedRanges::try_from_span_iter(
            Roi::new(x..x + 5, y..y + 5).into_spans(),
        )?))
    }

    /// Simulate a Move gesture from `from` to `to` through the real update
    /// path (frame and matrix stay in sync by construction).
    impl DragTool {
        /// Commit the current transform, as a finished transform gesture
        /// does (see [`DragTool::step_transform`]).
        fn commit(&mut self, masks: &mut MaskImage, img_roi: Roi<u32>) {
            self.selection = self
                .selection
                .take()
                .and_then(|sel| sel.commit_transform(masks, img_roi));
        }
    }

    fn drag_move(tool: &mut DragTool, from: Point2<f64>, to: Point2<f64>) -> TestResult {
        drag(tool, HoverGesture::Move, from, to, false)
    }

    /// Run a transform gesture on `part` from `from` to `to` through the
    /// real update path (frame and matrix stay in sync by construction).
    fn drag(
        tool: &mut DragTool,
        gesture: HoverGesture,
        from: Point2<f64>,
        to: Point2<f64>,
        shift: bool,
    ) -> TestResult {
        let sel = tool.selection.as_mut().ok_or("has a selection")?;
        let gesture = TransformGesture::begin(gesture, from, sel.snapshot_transform())
            .ok_or("Test gesture starts inside the frame")?;
        sel.apply_gesture(&gesture, to, shift);
        Ok(())
    }

    #[test]
    fn commit_keeps_selection_after_move() -> TestResult {
        // Mouseup commits the transform and keeps the selection idle for
        // further gestures. (The gesture itself ends because the transform
        // step returns `None` on mouseup.)
        let mut masks = mask_with_rect(10, 10)?;
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(10..10 + 5, 10..10 + 5), false);
        drag_move(&mut tool, Point2::new(12.5, 12.5), Point2::new(17.5, 12.5))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(
            tool.selection.is_some(),
            "selection stays idle after a successful move"
        );
        Ok(())
    }

    #[test]
    fn no_op_writes_no_history() -> TestResult {
        // Neither an unchanged commit nor a selection-less delete may touch
        // history.
        let mut masks = mask_with_rect(10, 10)?;
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(10..10 + 5, 10..10 + 5), false);
        let tip_before = masks.last_history_action();
        tool.commit(&mut masks, IMG_ROI);
        assert_eq!(masks.last_history_action(), tip_before);
        let mut tool = DragTool::default();
        tool.delete_selection(&mut masks);
        assert_eq!(masks.last_history_action(), tip_before);
        Ok(())
    }

    #[test]
    fn delete_selection_clears_ranges_and_drops_selection() -> TestResult {
        let mut masks = mask_with_rect(10, 10)?;
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(10..10 + 5, 10..10 + 5), false);
        let tip_before = masks.last_history_action();
        tool.delete_selection(&mut masks);
        assert_eq!(layer_pixels(&masks), None);
        assert!(tool.selection.is_none());
        assert_ne!(masks.last_history_action(), tip_before);
        Ok(())
    }

    fn mask_with_two_clusters() -> MaskImage {
        mask(ranges_from_spans(&[
            Span::new(0..2, 0u32),
            Span::new(5..7, 3u32),
        ]))
    }

    #[test]
    fn shift_click_accumulates_clusters_same_layer() -> TestResult {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        let sel = tool.selection.as_ref().ok_or("has selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        tool.select_at(&mut masks, Pos2::new(6.0, 3.0), IMG_ROI, true);
        let sel = tool.selection.as_ref().ok_or("has selection")?;
        // Same layer unions into a single entry (like rect-select): both
        // clusters are covered by the one selection.
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(0, 6, 3));
        // Frame expanded to contain both clusters.
        assert!(sel.frame().half.x >= 3.5);
        Ok(())
    }

    #[test]
    fn shift_click_covered_or_empty_is_noop() -> TestResult {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        tool.select_at(&mut masks, Pos2::new(6.0, 3.0), IMG_ROI, true);
        let tip_before = masks.last_history_action();
        // Re-clicking covered pixels is a no-op: no duplicate entry, no
        // history write, no rebase. Clicking empty space keeps the selection.
        tool.select_at(&mut masks, Pos2::new(1.0, 0.0), IMG_ROI, true);
        tool.select_at(&mut masks, Pos2::new(50.0, 50.0), IMG_ROI, true);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        assert_eq!(masks.last_history_action(), tip_before);
        Ok(())
    }

    #[test]
    fn shift_click_adds_other_layer() -> TestResult {
        let mut masks = mask(Roi::new(0..2, 0..2).into_spans());
        masks.add(Roi::new(50u16..52, 50..52).into());
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        tool.select_at(&mut masks, Pos2::new(51.0, 50.0), IMG_ROI, true);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(1, 51, 50));
        // Layer 2 stays unselected.
        assert!(!sel.covers_on_layer(2, 0, 50));
        Ok(())
    }

    #[test]
    fn unaffected_layer_click_clears_only_without_shift() -> TestResult {
        // Tool restricted to layer 0: a pixel covered only by layer 1 is
        // empty space for the tool — a plain click clears the selection,
        // a Shift-click keeps it.
        let mut masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.set_layer(0);
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        assert!(tool.selection.is_some());
        tool.select_at(&mut masks, Pos2::new(51.0, 0.0), IMG_ROI, false);
        assert!(tool.selection.is_none());
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        tool.select_at(&mut masks, Pos2::new(51.0, 0.0), IMG_ROI, true);
        let selection = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(selection.covers_on_layer(0, 0, 0));
        Ok(())
    }

    #[test]
    fn shift_add_rebases_transform_without_history_write() -> TestResult {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        // Transform + commit.
        drag_move(&mut tool, Point2::new(1.0, 0.0), Point2::new(6.0, 0.0))?;
        tool.commit(&mut masks, IMG_ROI);
        // Shift-add bakes committed into original and resets the matrix,
        // without touching history. (Snapshot semantics asserted directly in
        // the `logic` unit tests.)
        let tip_before = masks.last_history_action();
        tool.select_at(&mut masks, Pos2::new(6.0, 3.0), IMG_ROI, true);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert_eq!(masks.last_history_action(), tip_before);
        assert_eq!(sel.snapshot_transform().1, Matrix3::identity());
        // The next commit moves placed pixels and new cluster uniformly:
        // both travel by the same delta, nothing is left behind.
        drag_move(&mut tool, Point2::new(6.0, 3.0), Point2::new(16.0, 13.0))?;
        tool.commit(&mut masks, IMG_ROI);
        let expected = SortedRanges::<u32>::try_from_span_iter(
            Roi::new(15u16..17, 10..11)
                .into_spans()
                .union(Roi::new(15..17, 13..14).into_spans()),
        )?;
        assert_eq!(layer_pixels(&masks), Some(expected));
        Ok(())
    }

    #[test]
    fn shift_rect_unions_same_layer() -> TestResult {
        let masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(0..3, 0..1), false);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(!sel.covers_on_layer(0, 6, 3));
        // Frame hugs the contained cluster (0..2, 0), not the marquee.
        assert_eq!(sel.frame().center, Point2::new(1.0, 0.5));
        assert_eq!(sel.frame().half, Vector2::new(1.0, 0.5));
        // Shift-rect over the second cluster unions: the frame hugs the
        // combined content (0..7, 0..4).
        tool.select_rect(&masks, Roi::new(4..8, 2..4), true);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 6, 3));
        assert_eq!(sel.frame().center, Point2::new(3.5, 2.0));
        assert_eq!(sel.frame().half, Vector2::new(3.5, 2.0));
        Ok(())
    }

    #[test]
    fn shift_rect_on_empty_space_keeps_selection() -> TestResult {
        let masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(0..3, 0..1), false);
        tool.select_rect(&masks, Roi::new(80..90, 80..90), true);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        Ok(())
    }

    #[test]
    fn shift_click_union_then_resize_commits_both() -> TestResult {
        // Regression: two shift-clicked areas on the same layer must resize
        // together. Separate per-cluster entries used to clear/add the same
        // layer twice per commit, subtracting one area or unioning stale
        // content.
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        tool.select_at(&mut masks, Pos2::new(6.0, 3.0), IMG_ROI, true);
        // The selection content is exactly the union of the two clusters.
        let union = ranges_from_spans(&[Span::new(0..2, 0u32), Span::new(5..7, 3u32)]);
        let selection = tool.selection.as_ref().ok_or("has a selection")?;
        let (frame, _) = selection.snapshot_transform();
        let east = frame.point(Vector2::new(frame.half.x, 0.0));
        drag(
            &mut tool,
            HoverGesture::Resize(Anchor::E),
            east,
            Point2::new(east.x + frame.half.x, east.y),
            false,
        )?;
        let selection = tool.selection.as_ref().ok_or("has a selection")?;
        let total = selection.snapshot_transform().1;
        let expected =
            transform_layer(union.spans(), &total, IMG_ROI).ok_or("visible after resize")?;
        tool.commit(&mut masks, IMG_ROI);
        assert_eq!(layer_pixels(&masks), Some(expected));
        // Both areas survived the resize (no subtraction of the old area),
        // and the selection stays alive for further gestures.
        let pixels = layer_pixels(&masks).ok_or("Resized content stays visible")?;
        assert!(pixels.spans::<u32>().any(|s| s.y == 0));
        assert!(tool.selection.is_some());
        Ok(())
    }

    fn mask_with_three_layers() -> MaskImage {
        let mut masks = mask(Roi::new(0..2, 0..2).into_spans());
        masks.add(Roi::new(50u16..52, 0..2).into());
        masks.add(Roi::new(0u16..2, 50..52).into());
        masks
    }

    #[test]
    fn select_layers_selects_whole_matched_layers() -> TestResult {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 0..2);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        // Both matched layers selected with their full pixels, layer 2 not.
        assert!(sel.covers_on_layer(0, 0, 0) && sel.covers_on_layer(0, 1, 1));
        assert!(sel.covers_on_layer(1, 50, 0) && sel.covers_on_layer(1, 51, 1));
        assert!(!sel.covers_on_layer(2, 0, 50));
        assert_eq!(sel.snapshot_transform().1, Matrix3::identity());
        // Frame tightly covers both layers, nothing else.
        assert_eq!(sel.frame().center, Point2::new(26.0, 1.0));
        assert_eq!(sel.frame().half, Vector2::new(26.0, 1.0));
        // Staleness tip armed against the current history.
        assert!(!sel.is_stale(masks.last_history_action()));
        Ok(())
    }

    #[test]
    fn select_layers_accepts_layer_shorthand() -> TestResult {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 2);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(2, 0, 50) && sel.covers_on_layer(2, 1, 51));
        // Only layer 2: the others stay unselected.
        assert!(!sel.covers_on_layer(0, 0, 0));
        Ok(())
    }

    #[test]
    fn select_layers_without_match_drops_selection() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, ..);
        assert!(tool.selection.is_some());
        tool.select_layers(&masks, 5..8);
        assert!(tool.selection.is_none());
    }

    #[test]
    fn select_layers_replaces_existing_selection() -> TestResult {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, ..);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(1, 50, 0));
        assert!(sel.covers_on_layer(2, 0, 50));
        tool.select_layers(&masks, 1..2);
        let sel = tool.selection.as_ref().ok_or("has a selection")?;
        assert!(sel.covers_on_layer(1, 50, 0));
        assert!(!sel.covers_on_layer(0, 0, 0));
        assert!(!sel.covers_on_layer(2, 0, 50));
        Ok(())
    }

    #[test]
    fn select_layers_then_gesture_moves_all_layers() -> TestResult {
        let mut masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 0..3);
        drag_move(&mut tool, Point2::new(1.0, 1.0), Point2::new(4.0, 5.0))?;
        tool.commit(&mut masks, IMG_ROI);
        assert_eq!(layer_pixels(&masks), Some(Roi::new(3u16..5, 4..6).into()));
        let second = masks.subgroups_stack().get(1).map(|a| a.pixels.clone());
        assert_eq!(second, Some(Roi::new(53..55, 4..6).into()));
        Ok(())
    }

    fn mask_blocks() -> MaskImage {
        // 10x5 selection block at (10,10), 10x5 outsider block at (40,30).
        let spans = (10..15)
            .map(|y| Span::new(10..20, y))
            .chain((30..35).map(|y| Span::new(40..50, y)))
            .collect::<Vec<_>>();
        mask(ranges_from_spans(&spans))
    }

    /// Click-selected 10x5 block on a layer that also holds a disjoint
    /// 10x5 outsider block.
    fn fresh_block_selection() -> (MaskImage, DragTool) {
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(12.0, 12.0), IMG_ROI, false);
        (masks, tool)
    }

    #[test]
    fn move_preserves_pixels_outside_selection() -> TestResult {
        // Small overlapping move.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(14.0, 12.0))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));

        // Two consecutive moves.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(17.0, 12.0))?;
        tool.commit(&mut masks, IMG_ROI);
        drag_move(&mut tool, Point2::new(17.0, 12.0), Point2::new(22.0, 12.0))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));

        // Move and back.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(17.0, 12.0))?;
        tool.commit(&mut masks, IMG_ROI);
        drag_move(&mut tool, Point2::new(17.0, 12.0), Point2::new(12.0, 12.0))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));

        // Shift-add second cluster, then move both: nothing outside the
        // union may change.
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_at(&mut masks, Pos2::new(0.0, 0.0), IMG_ROI, false);
        drag_move(&mut tool, Point2::new(1.0, 0.0), Point2::new(6.0, 0.0))?;
        tool.commit(&mut masks, IMG_ROI);
        let before = layer_pixels(&masks).ok_or("Moved block stays visible")?;
        tool.select_at(&mut masks, Pos2::new(6.0, 3.0), IMG_ROI, true);
        drag_move(&mut tool, Point2::new(6.0, 3.0), Point2::new(16.0, 13.0))?;
        tool.commit(&mut masks, IMG_ROI);
        let after = layer_pixels(&masks).ok_or("Moved block stays visible")?;
        assert_eq!(after.spans::<u32>().count(), before.spans::<u32>().count());

        // Rect-select, then two fractional moves.
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(8..22, 8..16), false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(19.7, 14.8))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));
        drag_move(&mut tool, Point2::new(19.7, 14.8), Point2::new(16.2, 19.3))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));

        // 2D partial overlap: 10x5 block moved to straddle the outsider block.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(43.0, 31.5))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));

        // 2D full cover: moved block lands exactly on the outsider block.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(45.0, 32.5))?;
        tool.commit(&mut masks, IMG_ROI);
        assert!(outsider_block_ok(&masks));
        Ok(())
    }
}

use std::{
    iter::{FusedIterator, once},
    sync::Arc,
};

use egui::{CursorIcon, Pos2, Vec2};
use futures::FutureExt;
use imask::{ImageDimension, ImaskSet, Roi, SortedRanges, SortedRangesTightSpanBuilder, Span};
use nalgebra::Point2;

use crate::{
    AffectedLayer, MaskImage, PixelArea, RectSelection, Tool, ToolContext, ToolFactory,
    tool::drag::active_selection::{ActiveSelectionLogic, HoverPart, LayerSelection},
};

mod active_selection;
mod frame;
mod gesture;
mod transform;

#[cfg(test)]
mod test_support;

use active_selection::ActiveSelection;
use gesture::{Gesture, TransformGesture};
use transform::{clamp_pixel, cluster_at};

/// Drag-select + transform tool. See `DRAG_TOOL_REFINED.md` for the full plan.
///
/// This is deliberately *not* a `DrawTool`: it never adds or removes pixels
/// itself (commits only move already-selected pixels), so there is no
/// insert/clear `Mode` to switch. It only owns the `AffectedLayer` filter.
#[derive(Default)]
#[non_exhaustive]
pub struct DragTool {
    layer: AffectedLayer,
    /// Dragging on empty space pans instead of rect-selecting.
    pub pan_on_drag: bool,
    selection: Option<ActiveSelection>,
    gesture: Option<Gesture>,
}

impl DragTool {
    pub fn set_layer(&mut self, layer: impl Into<AffectedLayer>) -> &mut Self {
        self.layer = layer.into();
        self
    }

    pub fn set_pan_on_drag(&mut self, pan_on_drag: bool) -> &mut Self {
        self.pan_on_drag = pan_on_drag;
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

    /// Empty-space interaction: pan when `pan_on_drag` is set and no layer
    /// covers the cursor, else start a rect selection.
    fn start_empty_space_gesture(
        &self,
        ctx: &mut ToolContext,
        pointer: Pos2,
        img_w: usize,
        img_h: usize,
    ) -> Gesture {
        if self.pan_on_drag {
            let (x, y) = clamp_pixel(pointer, img_w, img_h);
            if ctx.image.masks.find_layer_at((x, y)).is_none() {
                return Gesture::Pan;
            }
        }
        let mut selection = RectSelection::default();
        let _ = selection.drag_finished(ctx);
        Gesture::Rect(selection)
    }

    /// Click: select the cluster under the cursor (8-connected component of
    /// the topmost *affected* layer at the pixel), or deselect on empty
    /// (no affected layer covers the pixel). With `additive`
    /// (Shift held), the cluster is added to the existing selection — across
    /// layers — instead of replacing it; clicking an already-selected cluster
    /// is a no-op.
    fn click_select(
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
    fn select_rect(&mut self, masks: &MaskImage, roi: Roi<u32>, additive: bool) {
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

    /// Decide what a fresh drag does.
    /// Hit-testing and gesture origins use the PRESS position: with click +
    /// drag sensing, `drag_started()` only fires after the pointer moved past
    /// `max_click_dist`, so the current position may already have left small
    /// hit targets (anchors, rotate handle).
    fn begin_gesture(&self, ctx: &mut ToolContext, img_w: usize, img_h: usize) -> Option<Gesture> {
        let press_screen = ctx
            .response
            .interact_pointer_pos()
            .map(|cur| cur - ctx.response.total_drag_delta().unwrap_or(Vec2::ZERO))?;
        let pointer = ctx.painter.screen_to_image(press_screen);
        let press = Point2::new(pointer.x as f64, pointer.y as f64);
        let Some(s) = self.selection.as_ref() else {
            // No selection: rect-select, or pan on empty space.
            return Some(self.start_empty_space_gesture(ctx, pointer, img_w, img_h));
        };
        let part = active_selection::hit_test(&*ctx.painter, press_screen, s.frame());
        Some(
            match TransformGesture::begin(part, press, s.snapshot_transform()) {
                Some(transform) => Gesture::Transform(transform),
                None => self.start_empty_space_gesture(ctx, pointer, img_w, img_h),
            },
        )
    }

    /// Commit the current transform (see [`ActiveSelection::commit_transform`]).
    /// If nothing remains visible, the selection is dropped — an empty box is
    /// never shown.
    fn commit(&mut self, masks: &mut MaskImage, img_roi: Roi<u32>) {
        self.selection = self
            .selection
            .take()
            .and_then(|sel| sel.commit_transform(masks, img_roi));
    }

    /// Escape on a running gesture: a cancelled transform reverts to the
    /// pre-gesture frame and matrix; the mask was never touched, so there is
    /// nothing to undo there. Other gestures just end.
    fn cancel(&mut self, gesture: Gesture) {
        if let (Gesture::Transform(t), Some(sel)) = (gesture, self.selection.as_mut()) {
            sel.cancel_gesture(t);
        }
    }

    /// Rect-select step: selects on release (Shift adds instead of
    /// replacing) and ends; otherwise keeps the rubber band running.
    fn step_rect(
        &mut self,
        mut rect: RectSelection,
        ctx: &mut ToolContext,
        img_roi: Roi<u32>,
    ) -> Option<Gesture> {
        if let Some(result) = rect.drag_finished(ctx) {
            let additive = ctx.egui.input(|i| i.modifiers.shift);
            self.select_rect(&ctx.image.masks, Roi::from(result.rect()), additive);
            return None;
        }
        if !ctx.response.dragged() {
            return None;
        }
        // A shift-held rubber band keeps the existing selection: keep
        // showing its highlight underneath (cheap repaint of the live
        // texture; nothing to show after a replacing drag dropped it).
        if let Some(s) = self.selection.as_mut() {
            s.render_selection(ctx.egui, ctx.painter, img_roi);
        }
        Some(Gesture::Rect(rect))
    }

    /// Transform step: follow the pointer, and commit on mouseup (or when
    /// the pointer is gone), which ends the gesture.
    fn step_transform(
        &mut self,
        transform: TransformGesture,
        ctx: &mut ToolContext,
        pointer: Option<Point2<f64>>,
        img_roi: Roi<u32>,
    ) -> Option<Gesture> {
        *ctx.postpone_new_images = true;
        let sel = self.selection.as_mut()?;
        // Shift alters the active gesture: exact unsnapped rotation angles,
        // or a corner resize free of the aspect-ratio lock.
        let shift = ctx.egui.input(|i| i.modifiers.shift);
        if let Some(p) = pointer {
            sel.apply_gesture(&transform, p, shift);
        }
        if ctx.response.drag_stopped() || pointer.is_none() {
            self.commit(&mut ctx.image.masks, img_roi);
            return None;
        }
        sel.render_transform(ctx.egui, &mut *ctx.painter, img_roi, transform.is_move());
        ctx.egui.set_cursor_icon(transform.cursor(sel.frame()));
        Some(Gesture::Transform(transform))
    }

    /// No gesture running: a drag start begins one, a click (Shift adds)
    /// selects the cluster under the pointer.
    fn step_idle(
        &mut self,
        ctx: &mut ToolContext,
        pointer: Option<Pos2>,
        pointer_screen: Option<Pos2>,
        img_roi: Roi<u32>,
    ) -> Option<Gesture> {
        let img_w = img_roi.width().get() as usize;
        let img_h = img_roi.height().get() as usize;
        let gesture = ctx
            .response
            .drag_started()
            .then(|| self.begin_gesture(ctx, img_w, img_h))
            .flatten();
        if ctx.response.clicked()
            && !ctx.response.drag_stopped()
            && let Some(p) = pointer
        {
            let additive = ctx.egui.input(|i| i.modifiers.shift);
            self.click_select(&mut ctx.image.masks, p, img_roi, additive);
        } else {
            self.hover_cursor(ctx, gesture.as_ref(), pointer_screen);
        }
        gesture
    }

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

    /// Cursor for the hover state, or for a `gesture` that just began.
    fn hover_cursor(
        &self,
        ctx: &ToolContext,
        gesture: Option<&Gesture>,
        pointer_screen: Option<Pos2>,
    ) {
        let icon = match (gesture, self.selection.as_ref(), pointer_screen) {
            (Some(Gesture::Transform(t)), Some(sel), _) => t.cursor(sel.frame()),
            (Some(Gesture::Pan), _, _) => CursorIcon::AllScroll,
            (Some(Gesture::Rect(_)), _, _) => CursorIcon::Crosshair,
            (None, Some(sel), Some(p)) => {
                match active_selection::hit_test(&*ctx.painter, p, sel.frame()) {
                    HoverPart::Outside => return,
                    HoverPart::Inside => CursorIcon::Move,
                    HoverPart::Anchor(a) => active_selection::resize_cursor(sel.frame(), a),
                    HoverPart::Rotate => CursorIcon::Grab,
                }
            }
            _ => return,
        };
        ctx.egui.set_cursor_icon(icon);
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

impl Tool for DragTool {
    fn handle_interaction(&mut self, mut ctx: ToolContext) {
        // Escape cancels a running gesture, or drops the idle selection.
        let escape = ctx.egui.input(|i| i.key_pressed(egui::Key::Escape));
        let gesture = match self.gesture.take() {
            gesture if !escape => gesture,
            Some(gesture) => {
                self.cancel(gesture);
                None
            }
            None => {
                self.selection = None;
                None
            }
        };

        // Delete removes the selection; a stale one (history changed
        // underneath) ends the gesture that started from it.
        const DELETE_KEYS: [egui::Key; 2] = [egui::Key::Delete, egui::Key::Backspace];
        let delete = ctx
            .egui
            .input(|i| DELETE_KEYS.iter().any(|k| i.key_pressed(*k)));
        let gesture = if delete && let Some(sel) = self.selection.take() {
            sel.delete_all(&mut ctx.image.masks);
            None
        } else if let Some(sel) = &self.selection {
            *ctx.postpone_new_images = true;
            gesture.filter(|_| !sel.is_stale(ctx.image.masks.last_history_action()))
        } else {
            gesture
        };

        let img_roi = {
            let (w, h) = ctx.image.image.adjust.dimensions();
            Roi::from_dimensions(w, h)
        };
        let pointer_screen = ctx
            .response
            .interact_pointer_pos()
            .or_else(|| ctx.response.hover_pos());
        let pointer = pointer_screen.map(|p| ctx.painter.screen_to_image(p));

        let gesture = match gesture {
            Some(Gesture::Pan) => (!ctx.response.drag_stopped()).then_some(Gesture::Pan),
            Some(Gesture::Rect(rect)) => self.step_rect(rect, &mut ctx, img_roi),
            Some(Gesture::Transform(transform)) => {
                let pointer = pointer.map(|p| Point2::new(p.x as f64, p.y as f64));
                self.step_transform(transform, &mut ctx, pointer, img_roi)
            }
            None => self.step_idle(&mut ctx, pointer, pointer_screen, img_roi),
        };

        if gesture.is_none()
            && let Some(s) = self.selection.as_mut()
        {
            s.render_selection(ctx.egui, &mut *ctx.painter, img_roi);
        }
        self.gesture = gesture;
    }
}

#[cfg(test)]
mod tests {
    use imask::Span;
    use nalgebra::{Matrix3, Vector2};

    use super::frame::Anchor;
    use super::test_support::*;
    use super::transform::transform_layer;
    use super::*;
    use crate::MaskDefaultActions;

    #[test]
    fn fractional_move_snaps_to_whole_pixels() {
        // App drag deltas are fractional, but masks live on whole pixels: the
        // move delta snaps so the rasterization stays pixel-exact. Without
        // snapping, half-integer matrices make the span rasterizer emit
        // fringe rows outside its analytic bounds; those fringe spans enter
        // `committed`, and the *next* commit's Clear erases outsider pixels
        // sharing the rows.
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 12.0, 12.0, false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(44.5, 32.5));
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(
            sel.snapshot_transform().1,
            Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        assert_eq!(sel.frame().center, Point2::new(45.0, 32.5));
        tool.commit(&mut masks, img_roi());
        // Committed content is pixel-exact: the moved block lands exactly on
        // the outsider block, no fringe rows anywhere. (Span comparison: the
        // mask keeps the coordinate frame's bounds, not tight ones.)
        assert_eq!(
            layer_pixels(&masks).map(|p| p.spans::<u32>().collect::<Vec<_>>()),
            Some(rect_ranges(40, 30, nz(10), nz(5)).spans::<u32>().collect())
        );
        // Second fractional move: still exact, only it and the outsiders
        // remain.
        drag_move(&mut tool, Point2::new(44.5, 32.5), Point2::new(20.4, 22.6));
        tool.commit(&mut masks, img_roi());
        assert_eq!(
            tool.selection.as_ref().unwrap().snapshot_transform().1,
            Matrix3::new_translation(&Vector2::new(-24.0, -10.0))
                * Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        let expected = SortedRanges::<u32>::try_from_span_iter(
            rect_ranges(16, 20, nz(10), nz(5))
                .spans::<u32>()
                .union(rect_ranges(40, 30, nz(10), nz(5)).spans()),
        )
        .unwrap();
        assert_eq!(
            layer_pixels(&masks).map(|p| p.spans::<u32>().collect::<Vec<_>>()),
            Some(expected.spans::<u32>().collect())
        );
    }

    #[test]
    fn delete_restores_absorbed_outsiders() {
        // Move the block exactly onto the outsider block (union), then press
        // Delete: selection content goes, outsiders come back byte-identical.
        let mut masks = mask_blocks();
        let outsider_before: Vec<Span<u32>> = layer_pixels(&masks)
            .unwrap()
            .spans::<u32>()
            .filter(|s| (30..35).contains(&s.y))
            .collect();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 12.0, 12.0, false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(45.0, 32.5));
        tool.commit(&mut masks, img_roi());
        tool.delete_selection(&mut masks);
        assert!(tool.selection.is_none());
        // Only the outsiders remain: selection content (original spot and
        // moved union) is gone, outsiders byte-identical.
        let remaining: Vec<Span<u32>> = layer_pixels(&masks).unwrap().spans::<u32>().collect();
        assert_eq!(remaining, outsider_before);
    }

    fn mask_with_rect(x: u32, y: u32) -> MaskImage {
        mask(rect_ranges(x, y, nz(5), nz(5)))
    }

    /// Rect-select the 5x5 block placed by [`mask_with_rect`]: the whole
    /// layer, so the background is empty — same shape the `select_layer`
    /// helper used to build by hand.
    fn select_rect_block(tool: &mut DragTool, masks: &MaskImage, x: u32, y: u32) {
        tool.select_rect(masks, Roi::new(x..x + 5, y..y + 5), false);
    }

    /// Simulate a Move gesture from `from` to `to` through the real update
    /// path (frame and matrix stay in sync by construction).
    fn drag_move(tool: &mut DragTool, from: Point2<f64>, to: Point2<f64>) {
        drag(tool, HoverPart::Inside, from, to, false);
    }

    /// Run a transform gesture on `part` from `from` to `to` through the
    /// real update path (frame and matrix stay in sync by construction).
    fn drag(tool: &mut DragTool, part: HoverPart, from: Point2<f64>, to: Point2<f64>, shift: bool) {
        let sel = tool.selection.as_mut().unwrap();
        let gesture = TransformGesture::begin(part, from, sel.snapshot_transform()).unwrap();
        sel.apply_gesture(&gesture, to, shift);
    }

    #[test]
    fn commit_keeps_selection_after_move() {
        // Mouseup commits the transform and keeps the selection idle for
        // further gestures. (The gesture itself ends because the transform
        // step returns `None` on mouseup.)
        let mut masks = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_rect_block(&mut tool, &masks, 10, 10);
        drag_move(&mut tool, Point2::new(12.5, 12.5), Point2::new(17.5, 12.5));
        tool.commit(&mut masks, img_roi());
        assert!(
            tool.selection.is_some(),
            "selection stays idle after a successful move"
        );
    }

    #[test]
    fn no_op_writes_no_history() {
        // Neither an unchanged commit nor a selection-less delete may touch
        // history.
        let mut masks = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_rect_block(&mut tool, &masks, 10, 10);
        let tip_before = masks.last_history_action();
        tool.commit(&mut masks, img_roi());
        assert_eq!(masks.last_history_action(), tip_before);
        let mut tool = DragTool::default();
        tool.delete_selection(&mut masks);
        assert_eq!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn delete_selection_clears_ranges_and_drops_selection() {
        let mut masks = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_rect_block(&mut tool, &masks, 10, 10);
        let tip_before = masks.last_history_action();
        tool.delete_selection(&mut masks);
        assert_eq!(layer_pixels(&masks), None);
        assert!(tool.selection.is_none());
        assert_ne!(masks.last_history_action(), tip_before);
    }

    fn mask_with_two_clusters() -> MaskImage {
        mask(ranges_from_spans(vec![Span::new(0..2, 0u32), Span::new(5..7, 3u32)]).unwrap())
    }

    fn click(tool: &mut DragTool, masks: &mut MaskImage, x: f32, y: f32, additive: bool) {
        tool.click_select(masks, Pos2::new(x, y), img_roi(), additive);
    }

    #[test]
    fn shift_click_accumulates_clusters_same_layer() {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 0, 0));
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        let sel = tool.selection.as_ref().unwrap();
        // Same layer unions into a single entry (like rect-select): both
        // clusters are covered by the one selection.
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(0, 6, 3));
        // Frame expanded to contain both clusters.
        assert!(sel.frame().half.x >= 3.5);
    }

    #[test]
    fn shift_click_covered_or_empty_is_noop() {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        let tip_before = masks.last_history_action();
        // Re-clicking covered pixels is a no-op: no duplicate entry, no
        // history write, no rebase. Clicking empty space keeps the selection.
        click(&mut tool, &mut masks, 1.0, 0.0, true);
        click(&mut tool, &mut masks, 50.0, 50.0, true);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 0, 0));
        assert_eq!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn shift_click_adds_other_layer() {
        let mut masks = mask(rect_ranges(0, 0, nz(2), nz(2)));
        masks.add(rect_ranges(50, 50, nz(2), nz(2)));
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        click(&mut tool, &mut masks, 51.0, 50.0, true);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(1, 51, 50));
        // Layer 2 stays unselected.
        assert!(!sel.covers_on_layer(2, 0, 50));
    }

    #[test]
    fn unaffected_layer_click_clears_only_without_shift() {
        // Tool restricted to layer 0: a pixel covered only by layer 1 is
        // empty space for the tool — a plain click clears the selection,
        // a Shift-click keeps it.
        let mut masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.set_layer(0);
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        assert!(tool.selection.is_some());
        click(&mut tool, &mut masks, 51.0, 0.0, false);
        assert!(tool.selection.is_none());
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        click(&mut tool, &mut masks, 51.0, 0.0, true);
        assert!(tool.selection.unwrap().covers_on_layer(0, 0, 0));
    }

    #[test]
    fn shift_add_rebases_transform_without_history_write() {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        // Transform + commit.
        drag_move(&mut tool, Point2::new(1.0, 0.0), Point2::new(6.0, 0.0));
        tool.commit(&mut masks, img_roi());
        // Shift-add bakes committed into original and resets the matrix,
        // without touching history. (Snapshot semantics asserted directly in
        // the `logic` unit tests.)
        let tip_before = masks.last_history_action();
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(masks.last_history_action(), tip_before);
        assert_eq!(sel.snapshot_transform().1, Matrix3::identity());
        // The next commit moves placed pixels and new cluster uniformly:
        // both travel by the same delta, nothing is left behind.
        drag_move(&mut tool, Point2::new(6.0, 3.0), Point2::new(16.0, 13.0));
        tool.commit(&mut masks, img_roi());
        let expected = SortedRanges::<u32>::try_from_span_iter(
            rect_ranges(15, 10, nz(2), nz(1))
                .spans::<u32>()
                .union(rect_ranges(15, 13, nz(2), nz(1)).spans()),
        )
        .unwrap();
        assert_eq!(layer_pixels(&masks), Some(expected));
    }

    #[test]
    fn shift_rect_unions_same_layer() {
        let masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(0..3, 0..1), false);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(!sel.covers_on_layer(0, 6, 3));
        // Frame hugs the contained cluster (0..2, 0), not the marquee.
        assert_eq!(sel.frame().center, Point2::new(1.0, 0.5));
        assert_eq!(sel.frame().half, Vector2::new(1.0, 0.5));
        // Shift-rect over the second cluster unions: the frame hugs the
        // combined content (0..7, 0..4).
        tool.select_rect(&masks, Roi::new(4..8, 2..4), true);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 6, 3));
        assert_eq!(sel.frame().center, Point2::new(3.5, 2.0));
        assert_eq!(sel.frame().half, Vector2::new(3.5, 2.0));
    }

    #[test]
    fn shift_click_union_then_resize_commits_both() {
        // Regression: two shift-clicked areas on the same layer must resize
        // together. Separate per-cluster entries used to clear/add the same
        // layer twice per commit, subtracting one area or unioning stale
        // content.
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        // The selection content is exactly the union of the two clusters.
        let union = ranges_from_spans(vec![Span::new(0..2, 0u32), Span::new(5..7, 3u32)]).unwrap();
        let (frame, _) = tool.selection.as_ref().unwrap().snapshot_transform();
        let east = frame.point(Vector2::new(frame.half.x, 0.0));
        drag(
            &mut tool,
            HoverPart::Anchor(Anchor::E),
            east,
            Point2::new(east.x + frame.half.x, east.y),
            false,
        );
        let total = tool.selection.as_ref().unwrap().snapshot_transform().1;
        let expected = transform_layer(&union, &total, img_roi()).unwrap();
        tool.commit(&mut masks, img_roi());
        assert_eq!(layer_pixels(&masks), Some(expected));
        // Both areas survived the resize (no subtraction of the old area),
        // and the selection stays alive for further gestures.
        let pixels = layer_pixels(&masks).unwrap();
        assert!(pixels.spans::<u32>().any(|s| s.y == 0));
        assert!(tool.selection.is_some());
    }

    fn mask_with_three_layers() -> MaskImage {
        let mut masks = mask(rect_ranges(0, 0, nz(2), nz(2)));
        masks.add(rect_ranges(50, 0, nz(2), nz(2)));
        masks.add(rect_ranges(0, 50, nz(2), nz(2)));
        masks
    }

    #[test]
    fn select_layers_selects_whole_matched_layers() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 0..2);
        let sel = tool.selection.as_ref().unwrap();
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
    }

    #[test]
    fn select_layers_accepts_layer_shorthand() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 2);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(2, 0, 50) && sel.covers_on_layer(2, 1, 51));
        // Only layer 2: the others stay unselected.
        assert!(!sel.covers_on_layer(0, 0, 0));
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
    fn select_layers_replaces_existing_selection() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, ..);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(0, 0, 0));
        assert!(sel.covers_on_layer(1, 50, 0));
        assert!(sel.covers_on_layer(2, 0, 50));
        tool.select_layers(&masks, 1..2);
        let sel = tool.selection.as_ref().unwrap();
        assert!(sel.covers_on_layer(1, 50, 0));
        assert!(!sel.covers_on_layer(0, 0, 0));
        assert!(!sel.covers_on_layer(2, 0, 50));
    }

    #[test]
    fn select_layers_then_gesture_moves_all_layers() {
        let mut masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, 0..3);
        drag_move(&mut tool, Point2::new(1.0, 1.0), Point2::new(4.0, 5.0));
        tool.commit(&mut masks, img_roi());
        assert_eq!(layer_pixels(&masks), Some(rect_ranges(3, 4, nz(2), nz(2))));
        let second = masks.subgroups_stack().get(1).map(|a| a.pixels.clone());
        assert_eq!(second, Some(rect_ranges(53, 4, nz(2), nz(2))));
    }

    fn mask_blocks() -> MaskImage {
        // 10x5 selection block at (10,10), 10x5 outsider block at (40,30).
        let spans = (10..15)
            .map(|y| Span::new(10..20, y))
            .chain((30..35).map(|y| Span::new(40..50, y)))
            .collect();
        mask(ranges_from_spans(spans).unwrap())
    }

    /// Click-selected 10x5 block on a layer that also holds a disjoint
    /// 10x5 outsider block.
    fn fresh_block_selection() -> (MaskImage, DragTool) {
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 12.0, 12.0, false);
        (masks, tool)
    }

    #[test]
    fn move_preserves_pixels_outside_selection() {
        // Small overlapping move.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(14.0, 12.0));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));

        // Two consecutive moves.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(17.0, 12.0));
        tool.commit(&mut masks, img_roi());
        drag_move(&mut tool, Point2::new(17.0, 12.0), Point2::new(22.0, 12.0));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));

        // Move and back.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(17.0, 12.0));
        tool.commit(&mut masks, img_roi());
        drag_move(&mut tool, Point2::new(17.0, 12.0), Point2::new(12.0, 12.0));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));

        // Shift-add second cluster, then move both: nothing outside the
        // union may change.
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        drag_move(&mut tool, Point2::new(1.0, 0.0), Point2::new(6.0, 0.0));
        tool.commit(&mut masks, img_roi());
        let before = layer_pixels(&masks).unwrap();
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        drag_move(&mut tool, Point2::new(6.0, 3.0), Point2::new(16.0, 13.0));
        tool.commit(&mut masks, img_roi());
        let after = layer_pixels(&masks).unwrap();
        assert_eq!(after.spans::<u32>().count(), before.spans::<u32>().count());

        // Rect-select, then two fractional moves.
        let mut masks = mask_blocks();
        let mut tool = DragTool::default();
        tool.select_rect(&masks, Roi::new(8..22, 8..16), false);
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(19.7, 14.8));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));
        drag_move(&mut tool, Point2::new(19.7, 14.8), Point2::new(16.2, 19.3));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));

        // 2D partial overlap: 10x5 block moved to straddle the outsider block.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(43.0, 31.5));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));

        // 2D full cover: moved block lands exactly on the outsider block.
        let (mut masks, mut tool) = fresh_block_selection();
        drag_move(&mut tool, Point2::new(15.0, 12.5), Point2::new(45.0, 32.5));
        tool.commit(&mut masks, img_roi());
        assert!(outsider_block_ok(&masks));
    }
}

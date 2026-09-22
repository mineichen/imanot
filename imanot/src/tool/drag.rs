use std::sync::Arc;

use egui::{CursorIcon, Pos2, Vec2};
use futures::FutureExt;
use imask::{ImageDimension, ImaskSet, Roi, SortedRanges, Span};
#[cfg(test)]
use imask::{SpanBoundsBuilder, WithRoi};
use nalgebra::Point2;

use crate::{
    AffectedLayer, MaskImage, RectSelection, RectSelectionResult, Tool, ToolContext, ToolFactory,
    tool::drag::active_selection::logic::subtract_ranges_collect_subtrahend,
};

mod active_selection;
mod frame;
mod gesture;
mod overlay;
mod transform;

use active_selection::{ActiveSelection, subtract_ranges};
use gesture::{Gesture, GestureMove, GestureResize, GestureRotate};
use overlay::{HoverPart, hit_test, resize_cursor};
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

    /// Drop any in-progress gesture. Selection replacement and drops need no
    /// explicit preview handling: the preview lives inside `ActiveSelection`
    /// and dies or starts fresh with it. In-place selection mutations reset
    /// it at the mutation point (`merge_layers`).
    fn settle(&mut self) {
        self.gesture = None;
    }

    /// Drop the whole selection (no box left to show). The preview dies with
    /// it; any in-progress gesture is dropped too.
    fn drop_selection(&mut self) {
        self.selection = None;
        self.settle();
    }

    /// Is pixel `(x, y)` already covered by `layer`'s currently placed
    /// selection pixels?
    fn covers_on_layer(&self, layer: usize, x: u32, y: u32) -> bool {
        self.selection
            .as_ref()
            .is_some_and(|sel| sel.covers_on_layer(layer, x, y))
    }

    /// Drop the selection if the history tip changed underneath it (external
    /// push, undo, redo or another tool's commit).
    fn check_stale(&mut self, masks: &MaskImage) {
        if self
            .selection
            .as_ref()
            .is_some_and(|sel| sel.is_stale(masks.last_history_action()))
        {
            self.drop_selection();
        }
    }

    /// Empty-space interaction: pan when `pan_on_drag` is set and no layer
    /// covers the cursor, else start a rect selection. Returns true when
    /// panning, in which case the caller must delegate to `PanTool`.
    fn start_empty_space_gesture(
        &mut self,
        ctx: &mut ToolContext,
        pointer: Pos2,
        img_w: usize,
        img_h: usize,
    ) -> bool {
        if self.pan_on_drag {
            let (x, y) = clamp_pixel(pointer, img_w, img_h);
            if ctx.image.masks.find_layer_at((x, y)).is_none() {
                self.gesture = Some(Gesture::Pan);
                return true;
            }
        }
        let mut selection = RectSelection::default();
        let _ = selection.drag_finished(ctx);
        self.gesture = Some(Gesture::Rect(selection));
        false
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
        let Some(idx) = masks.subgroups_stack().iter().rev().find_map(|(i, area)| {
            (self.layer.affects(i) && area.pixels.contains(x, y)).then_some(i)
        }) else {
            if !additive {
                self.drop_selection();
            }
            return;
        };
        // Additive re-click of an already-selected cluster is a no-op (avoids
        // duplicate entries which would clear/add the same pixels twice per
        // commit). A non-additive click replaces the selection, so it falls
        // through to the fresh selection below even when covered.
        if additive && self.covers_on_layer(idx, x, y) {
            return;
        }
        let Some(area) = masks.subgroups_stack().get(idx) else {
            return;
        };
        let Some(cluster) = cluster_at(&area.pixels, x, y) else {
            return;
        };
        // Non-selected layer content: re-added under every cleared footprint
        // so later moves cannot erase it.

        let (background, original) = subtract_ranges_collect_subtrahend(&area.pixels, cluster);
        let Ok(original) = original else { return };

        if additive && let Some(sel) = &mut self.selection {
            sel.merge_layers(
                std::iter::once((idx, original, background)),
                masks.last_history_action(),
            );
        } else {
            // New click without Shift replaces the old selection.
            self.selection = Some(ActiveSelection::fresh_single(
                idx,
                original,
                background,
                masks.last_history_action(),
            ));
        }
    }

    /// Build a selection from a finished rect selection: all pixels inside the
    /// rect, independent of layer, restricted to the tool's `AffectedLayer`.
    /// With `additive` (Shift held), the rect's pixels are unioned into the
    /// existing selection instead of replacing it; an empty rect then keeps
    /// the selection unchanged.
    fn select_rect(&mut self, masks: &MaskImage, result: &RectSelectionResult, additive: bool) {
        // `RectSelection` is shared tool infra still on `Rect`; convert at
        // the boundary — everything inside the drag tool uses `Roi`.
        let roi = Roi::from(result.rect());
        let layer = self.layer;
        let mut layers = masks.subgroups_stack().iter().filter_map(|(idx, area)| {
            if !layer.affects(idx) {
                return None;
            }
            // `minbounds` shrinks the (possibly whole-image) layer bounds to
            // the clipped content in a single pass — no intermediate `Vec`.
            // The background (layer minus snapshot) travels along so later
            // commits can restore outsiders under cleared footprints.
            let clipped = area.pixels.spans::<u32>().clip(roi).ok()?;
            let ranges = SortedRanges::try_from_span_iter_minbounds(clipped).ok()?;
            let background = subtract_ranges(&area.pixels, ranges.spans());
            Some((idx, ranges, background))
        });
        if let Some(first) = layers.next() {
            if additive && let Some(sel) = self.selection.as_mut() {
                sel.merge_layers(
                    std::iter::once(first).chain(layers),
                    masks.last_history_action(),
                );
                return;
            }
            self.select_rect_fresh(masks, std::iter::once(first).chain(layers));
        } else {
            // Nothing selected: no box to show, drop any previous selection.
            self.drop_selection();
        }
    }

    /// Fresh (replacing) selection from rect-clipped `(layer, ranges,
    /// background)` triples with `original == committed`. The frame hugs the
    /// contained pixels — the dragged box may cover empty areas, which must
    /// not size anchors and pivots.
    fn select_rect_fresh(
        &mut self,
        masks: &MaskImage,
        layers: impl Iterator<Item = (usize, SortedRanges<u32>, Option<SortedRanges<u32>>)>,
    ) {
        // Each snapshot already carries tight bounds (`minbounds` over the
        // imask-clipped spans); the frame hugs the contained pixels, never
        // the queried boxes. `None` means no pixels: no box to show.
        self.selection = ActiveSelection::fresh_from_clipped(layers, masks.last_history_action());
        if self.selection.is_none() {
            self.settle();
        }
    }

    /// Programmatically replace the selection from raw per-layer span streams:
    /// `(layer index, spans carrying their own bounds)`, e.g.
    /// `masks.subgroups_stack().iter().map(|(i, a)| (i, a.pixels.spans()))`.
    /// Behaves like a fresh selection: tight ranges are rebuilt per layer and
    /// snapshotted as pristine originals, the frame tightly covers all
    /// selected pixels and any in-progress gesture is dropped. Layers without
    /// visible pixels are skipped; if nothing matches, the selection is
    /// dropped (an empty box is never shown). Unlike click/rect selection this
    /// does not intersect with the tool's own `AffectedLayer` filter — the
    /// caller names the layers explicitly.
    pub fn select_layers<S>(&mut self, masks: &MaskImage, layers: impl Iterator<Item = (usize, S)>)
    where
        S: Iterator<Item = Span<u32>> + ImageDimension,
    {
        self.selection = ActiveSelection::fresh_from_spans(layers, masks.last_history_action());
        self.settle();
    }
    /// Decide what a fresh drag does and store it in `self.gesture`.
    /// Hit-testing and gesture origins use the PRESS position: with click +
    /// drag sensing, `drag_started()` only fires after the pointer moved past
    /// `max_click_dist`, so the current position may already have left small
    /// hit targets (anchors, rotate handle).
    /// Returns true if the caller must delegate the whole interaction to
    /// `PanTool` (pan gesture, consumes `ctx`).
    fn begin_gesture(&mut self, ctx: &mut ToolContext, img_w: usize, img_h: usize) -> bool {
        let press_screen = ctx
            .response
            .interact_pointer_pos()
            .map(|cur| cur - ctx.response.total_drag_delta().unwrap_or(Vec2::ZERO));
        let Some(press_screen) = press_screen else {
            return false;
        };
        let pointer = ctx.painter.screen_to_image(press_screen);
        let press = Point2::new(pointer.x as f64, pointer.y as f64);
        let Some((frame, total)) = self.selection.as_ref().map(|s| s.snapshot_transform()) else {
            // No selection: rect-select, or pan on empty space.
            return self.start_empty_space_gesture(ctx, pointer, img_w, img_h);
        };
        match hit_test(&*ctx.painter, press_screen, &frame) {
            HoverPart::Outside => {
                // New box selection without Shift replaces the old selection:
                // drop it as the rubber-band starts so no stale box stays
                // visible during the drag. With Shift the old selection is
                // kept for union on release.
                if !ctx.egui.input(|i| i.modifiers.shift) {
                    self.drop_selection();
                }
                self.start_empty_space_gesture(ctx, pointer, img_w, img_h)
            }
            HoverPart::Inside => {
                self.gesture = Some(Gesture::Move(GestureMove {
                    start: press,
                    base: frame,
                    base_total: total,
                }));
                false
            }
            HoverPart::Anchor(anchor) => {
                self.gesture = Some(Gesture::Resize(GestureResize {
                    anchor,
                    start: press,
                    base: frame,
                    base_total: total,
                }));
                false
            }
            HoverPart::Rotate => {
                self.gesture = Some(Gesture::Rotate(GestureRotate {
                    start_angle: (press.y - frame.center.y).atan2(press.x - frame.center.x),
                    base: frame,
                    base_total: total,
                }));
                false
            }
        }
    }

    /// Recompute `selection.frame` and `selection.total` from the current
    /// pointer for an active transform gesture. Pointer in image coordinates.
    /// Thin state wrapper around [`apply`]: the geometry itself lives in
    /// `gesture.rs` as pure functions, so overlay and rasterization can never
    /// drift apart.
    fn update_gesture_frame(&mut self, pointer: Point2<f64>, shift: bool) {
        let gesture = self.gesture.as_mut();
        let Some(sel) = self.selection.as_mut() else {
            return;
        };
        if let Some((frame, total)) = gesture.and_then(|g| g.apply(pointer, shift)) {
            sel.set_transform(frame, total);
        }
    }

    /// Commit the current transform (see [`ActiveSelection::commit_transform`]).
    /// If nothing remains visible, the selection is dropped — an empty box is
    /// never shown. Either way the in-progress gesture ends on mouseup, so it
    /// is always settled: otherwise the tool stays stuck in the transform
    /// branch and re-commits the old gesture every frame.
    fn commit(&mut self, masks: &mut MaskImage, img_roi: Roi<u32>) {
        if let Some(sel) = self.selection.take() {
            if let Some(sel) = sel.commit_transform(masks, img_roi) {
                self.selection = Some(sel);
            }
            self.settle();
        };
    }

    /// Delete all ranges in the current selection: Clear the currently placed
    /// ranges on every selected layer, then drop the selection.
    /// First action is `tracked`, the rest are not, so one ctrl-Z reverts the
    /// whole delete across all layers. Uncommitted gesture deltas are
    /// discarded — the mask itself is never touched during a gesture, so
    /// there is nothing to undo there.
    fn delete_selection(&mut self, masks: &mut MaskImage) {
        if let Some(sel) = self.selection.take() {
            self.settle();
            sel.delete_all(masks);
        };
    }

    /// Cursor for the current hover/gesture state.
    fn hover_cursor(&self, ctx: &ToolContext, pointer_screen: Option<Pos2>) {
        let icon = match (&self.gesture, self.selection.as_ref(), pointer_screen) {
            (Some(Gesture::Move(_)), _, _) => CursorIcon::Grabbing,
            (Some(Gesture::Resize(g)), Some(sel), _) => resize_cursor(sel.frame(), g.anchor),
            (Some(Gesture::Rotate(_)), _, _) => CursorIcon::Grabbing,
            (Some(Gesture::Pan), _, _) => CursorIcon::AllScroll,
            (Some(Gesture::Rect(_)), _, _) => CursorIcon::Crosshair,
            (None, Some(sel), Some(p)) => match hit_test(&*ctx.painter, p, &sel.frame()) {
                HoverPart::Outside => return,
                HoverPart::Inside => CursorIcon::Move,
                HoverPart::Anchor(a) => resize_cursor(&sel.frame(), a),
                HoverPart::Rotate => CursorIcon::Grab,
            },
            _ => return,
        };
        ctx.egui.set_cursor_icon(icon);
    }
}

impl Tool for DragTool {
    fn handle_interaction(&mut self, mut ctx: ToolContext) {
        if ctx.egui.input(|i| i.key_pressed(egui::Key::Escape)) {
            if let Some(gesture) = self.gesture.take() {
                // Cancelled gestures revert to the pre-gesture frame and
                // matrix; the mask was never touched, so there is nothing to
                // undo there.
                if let (Some((base, base_total)), Some(sel)) =
                    (gesture.base_state(), self.selection.as_mut())
                {
                    // Restored frame and matrix invalidate the uploaded pixels.
                    sel.restore_gesture(base, base_total);
                }
            } else {
                self.drop_selection();
            }
        }
        self.check_stale(&ctx.image.masks);
        if ctx
            .egui
            .input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            self.delete_selection(&mut ctx.image.masks);
        }
        // Keep the image still while a selection is active: postpone loading
        // of new images so the selection is not lost underneath the tool.
        if self.selection.is_some() {
            *ctx.postpone_new_images = true;
        }

        let (w_nz, h_nz) = ctx.image.image.adjust.dimensions();
        let img_w = w_nz.get() as usize;
        let img_h = h_nz.get() as usize;
        let img_roi = Roi::from_dimensions(w_nz, h_nz);
        let pointer_screen = ctx
            .response
            .interact_pointer_pos()
            .or_else(|| ctx.response.hover_pos());
        let pointer = pointer_screen.map(|p| ctx.painter.screen_to_image(p));

        match &mut self.gesture {
            Some(Gesture::Pan) => {
                if ctx.response.drag_stopped() {
                    self.gesture = None;
                }
                return;
            }
            Some(Gesture::Rect(rect_selection)) => {
                let result = rect_selection.drag_finished(&mut ctx);
                if let Some(result) = result {
                    // Shift on release adds to the selection instead of
                    // replacing it.
                    let additive = ctx.egui.input(|i| i.modifiers.shift);
                    self.select_rect(&ctx.image.masks, &result, additive);
                    self.settle();
                } else if !ctx.response.dragged() {
                    self.gesture = None;
                }
                // A shift-held rubber band keeps the existing selection: keep
                // showing its highlight underneath (cheap repaint of the live
                // texture; nothing to show after a replacing drag dropped it).
                if let Some(s) = self.selection.as_mut() {
                    s.render_selection(ctx.egui, &mut *ctx.painter, img_roi);
                }
            }
            Some(Gesture::Move(_) | Gesture::Resize(_) | Gesture::Rotate(_)) => {
                *ctx.postpone_new_images = true;
                // Shift alters the active gesture: exact unsnapped rotation
                // angles, or a corner resize free of the aspect-ratio lock.
                let shift = ctx.egui.input(|i| i.modifiers.shift);
                if let Some(p) = pointer {
                    self.update_gesture_frame(Point2::new(p.x as f64, p.y as f64), shift);
                }
                if ctx.response.drag_stopped() || pointer.is_none() {
                    self.commit(&mut ctx.image.masks, img_roi);
                } else if let Some(sel) = self.selection.as_mut() {
                    let moved = matches!(self.gesture, Some(Gesture::Move(_)));
                    sel.render_transform(ctx.egui, &mut *ctx.painter, img_roi, moved);
                }
                self.hover_cursor(&ctx, pointer_screen);
            }
            None => {
                if ctx.response.drag_started() && self.begin_gesture(&mut ctx, img_w, img_h) {
                    return;
                } else if ctx.response.clicked()
                    && !ctx.response.drag_stopped()
                    && let Some(p) = pointer
                {
                    // Shift-click adds the cluster to the selection instead of
                    // replacing it.
                    let additive = ctx.egui.input(|i| i.modifiers.shift);
                    self.click_select(&mut ctx.image.masks, p, img_roi, additive);
                } else {
                    self.hover_cursor(&ctx, pointer_screen);
                }
            }
        }

        if self.gesture.is_none()
            && let Some(s) = self.selection.as_mut()
        {
            s.render_selection(ctx.egui, &mut *ctx.painter, img_roi);
        }
        if self.selection.is_some() {
            *ctx.postpone_new_images = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use imask::ImageDimension;
    use nalgebra::{Matrix3, Vector2};

    use super::frame::Anchor;
    use super::transform::transform_layer;
    use super::*;
    use crate::ImagePainter;

    /// Span streams of the `masks` stack layers matched by `which`
    /// (any `impl Into<AffectedLayer>`): the caller-side half of
    /// [`DragTool::select_layers`] — filtering stays with the caller, tight
    /// rebuilding inside the tool.
    fn stack_spans(
        masks: &MaskImage,
        which: impl Into<AffectedLayer>,
    ) -> impl Iterator<Item = (usize, impl Iterator<Item = Span<u32>> + ImageDimension + '_)> + '_
    {
        let which = which.into();
        masks
            .subgroups_stack()
            .iter()
            .filter(move |(i, _)| which.affects(*i))
            .map(|(i, a)| (i, a.pixels.spans::<u32>()))
    }

    /// Test-only `Vec` adapter: `Vec` is not `ImageDimension`, so tight
    /// bounds are tracked natively via `SpanBoundsBuilder` first.
    fn ranges_from_spans(spans: Vec<Span<u32>>) -> Option<SortedRanges<u32>> {
        let tight = spans
            .iter()
            .copied()
            .collect::<SpanBoundsBuilder<u32>>()
            .build()
            .ok()?;
        SortedRanges::try_from_span_iter(WithRoi::new(spans.into_iter(), tight)).ok()
    }

    fn rect_ranges(x: u32, y: u32, w: NonZeroU32, h: NonZeroU32) -> SortedRanges<u32> {
        SortedRanges::try_from_span_iter(Roi::new(x..x + w.get(), y..y + h.get()).into_spans())
            .unwrap()
    }

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    fn img_roi() -> Roi<u32> {
        Roi::from_dimensions(nz(100), nz(100))
    }

    /// Pixel area of the first layer's committed selection content.
    fn committed_area(tool: &DragTool) -> usize {
        tool.selection
            .as_ref()
            .expect("Selection is active")
            .first_committed_area()
    }

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
            sel.total_for_test(),
            Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        assert_eq!(sel.frame().center, Point2::new(45.0, 32.5));
        tool.commit(&mut masks, img_roi());
        // Committed content is pixel-exact: original 50px, no fringe.
        assert_eq!(committed_area(&tool), 50);
        assert!(outsider_block_ok(&masks));
        // Second fractional move: still exact, outsiders still intact.
        drag_move(&mut tool, Point2::new(44.5, 32.5), Point2::new(20.4, 22.6));
        tool.commit(&mut masks, img_roi());
        assert_eq!(
            tool.selection.as_ref().unwrap().total_for_test(),
            Matrix3::new_translation(&Vector2::new(-24.0, -10.0))
                * Matrix3::new_translation(&Vector2::new(30.0, 20.0))
        );
        assert_eq!(committed_area(&tool), 50);
        assert!(outsider_block_ok(&masks));
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

    fn mask_with_rect(x: u32, y: u32) -> (MaskImage, SortedRanges<u32>) {
        use crate::{History, MaskDefaultActions, PixelAreaStack};
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        let original = rect_ranges(x, y, nz(5), nz(5));
        masks.add(original.clone());
        (masks, original)
    }

    fn select_layer(tool: &mut DragTool, masks: &MaskImage, original: SortedRanges<u32>) {
        tool.selection = Some(ActiveSelection::fresh_single(
            0,
            original,
            None,
            masks.last_history_action(),
        ));
    }

    /// Simulate a Move gesture from `from` to `to` through the real update
    /// path (frame and matrix stay in sync by construction).
    fn drag_move(tool: &mut DragTool, from: Point2<f64>, to: Point2<f64>) {
        let (frame, total) = {
            let sel = tool.selection.as_ref().unwrap();
            sel.snapshot_transform()
        };
        tool.gesture = Some(Gesture::Move(GestureMove {
            start: from,
            base: frame,
            base_total: total,
        }));
        tool.update_gesture_frame(to, false);
        tool.settle();
    }

    fn layer_pixels(masks: &MaskImage) -> Option<SortedRanges<u32>> {
        masks.subgroups_stack().get(0).map(|a| a.pixels.clone())
    }

    #[test]
    fn commit_moves_content_and_frame() {
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_layer(&mut tool, &masks, original);
        drag_move(&mut tool, Point2::new(12.5, 12.5), Point2::new(17.5, 12.5));
        tool.commit(&mut masks, img_roi());
        assert_eq!(
            layer_pixels(&masks),
            Some(rect_ranges(15, 10, nz(5), nz(5)))
        );
        let frame = tool.selection.as_ref().unwrap().frame();
        assert_eq!(frame.center, Point2::new(17.5, 12.5));
        assert_eq!(frame.half, Vector2::new(2.5, 2.5));
    }

    #[test]
    fn commit_drops_gesture_after_mouseup() {
        // Mouseup ends the transform gesture: the commit keeps the selection
        // (idle, for further gestures) but must drop the in-progress gesture.
        // Otherwise the tool stays stuck in the Move branch and re-commits or
        // re-renders the old gesture on every following frame.
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_layer(&mut tool, &masks, original);
        let (frame, total) = {
            let sel = tool.selection.as_ref().unwrap();
            sel.snapshot_transform()
        };
        tool.gesture = Some(Gesture::Move(GestureMove {
            start: Point2::new(12.5, 12.5),
            base: frame,
            base_total: total,
        }));
        tool.update_gesture_frame(Point2::new(17.5, 12.5), false);
        tool.commit(&mut masks, img_roi());
        assert!(
            tool.gesture.is_none(),
            "gesture must be dropped after mouseup commit"
        );
        assert!(
            tool.selection.is_some(),
            "selection stays idle after a successful move"
        );
    }

    #[test]
    fn commit_offscreen_doesnt_drop_empty_selection() {
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_layer(&mut tool, &masks, original);
        // Move fully out of the image: pixels cleared, selection dropped so
        // no empty box is shown. The clears stay one undo step.
        drag_move(&mut tool, Point2::new(12.5, 12.5), Point2::new(-37.5, 12.5));
        let tip_before = masks.last_history_action();
        tool.commit(&mut masks, img_roi());
        assert_eq!(layer_pixels(&masks), None);
        assert!(tool.selection.is_some());
        assert_ne!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn commit_keeps_preview_alive() {
        // Dropping the selection at a new position must not re-rasterize:
        // the placed pixels are exactly what the preview already shows.
        let (mut masks, mut tool) = fresh_block_selection();
        let ctx = egui::Context::default();
        let screen =
            egui::Rect::from_min_max(egui::Pos2::new(0.0, 0.0), egui::Pos2::new(100.0, 100.0));
        let mut painter =
            ImagePainter::new(ctx.layer_painter(egui::LayerId::background()), screen, 1.0);
        tool.selection
            .as_mut()
            .unwrap()
            .render_transform(&ctx, &mut painter, img_roi(), false);
        assert!(tool.selection.as_ref().unwrap().preview_visible());
        drag_move(&mut tool, Point2::new(12.0, 12.0), Point2::new(17.0, 12.0));
        tool.commit(&mut masks, img_roi());
        // Pixels landed, outsiders intact, and the preview survived the drop.
        assert_eq!(
            tool.selection.as_ref().unwrap().first_committed_for_test(),
            Some(rect_ranges(15, 10, nz(10), nz(5)))
        );
        assert!(outsider_block_ok(&masks));
        assert!(tool.selection.as_ref().unwrap().preview_visible());
    }

    #[test]
    fn no_op_writes_no_history() {
        // Neither an unchanged commit nor a selection-less delete may touch
        // history.
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_layer(&mut tool, &masks, original);
        let tip_before = masks.last_history_action();
        tool.commit(&mut masks, img_roi());
        assert_eq!(masks.last_history_action(), tip_before);
        let mut tool = DragTool::default();
        tool.delete_selection(&mut masks);
        assert_eq!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn delete_selection_clears_ranges_and_drops_selection() {
        let (mut masks, original) = mask_with_rect(10, 10);
        let mut tool = DragTool::default();
        select_layer(&mut tool, &masks, original);
        let tip_before = masks.last_history_action();
        tool.delete_selection(&mut masks);
        assert_eq!(layer_pixels(&masks), None);
        assert!(tool.selection.is_none());
        assert_ne!(masks.last_history_action(), tip_before);
    }

    fn mask_with_two_clusters() -> MaskImage {
        use crate::{History, MaskDefaultActions, PixelAreaStack};
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        masks.add(ranges_from_spans(vec![Span::new(0..2, 0u32), Span::new(5..7, 3u32)]).unwrap());
        masks
    }

    fn click(tool: &mut DragTool, masks: &mut MaskImage, x: f32, y: f32, additive: bool) {
        tool.click_select(masks, Pos2::new(x, y), img_roi(), additive);
    }

    #[test]
    fn shift_click_accumulates_clusters_same_layer() {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        assert_eq!(tool.selection.as_ref().unwrap().layer_count_for_test(), 1);
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        let sel = tool.selection.as_ref().unwrap();
        // Same layer unions into a single entry (like rect-select).
        assert_eq!(sel.layer_count_for_test(), 1);
        assert_eq!(sel.original_of_for_test(0).unwrap().len(), 2);
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
        assert_eq!(sel.layer_count_for_test(), 1);
        assert_eq!(masks.last_history_action(), tip_before);
    }

    #[test]
    fn shift_click_adds_other_layer() {
        use crate::{History, MaskDefaultActions, PixelAreaStack};
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        masks.add(rect_ranges(0, 0, nz(2), nz(2)));
        masks.add(rect_ranges(50, 50, nz(2), nz(2)));
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        click(&mut tool, &mut masks, 51.0, 50.0, true);
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 2);
        let mut layers: Vec<usize> = sel.layer_ids_for_test();
        layers.sort_unstable();
        assert_eq!(layers, vec![0, 1]);
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
        assert_eq!(tool.selection.as_ref().unwrap().layer_count_for_test(), 1);
    }

    #[test]
    fn shift_add_rebases_transform_without_history_write() {
        let mut masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        click(&mut tool, &mut masks, 0.0, 0.0, false);
        // Transform + commit: entry original stays pristine, committed moves.
        drag_move(&mut tool, Point2::new(1.0, 0.0), Point2::new(6.0, 0.0));
        tool.commit(&mut masks, img_roi());
        // Shift-add bakes committed into original and resets the matrix,
        // without touching history.
        let tip_before = masks.last_history_action();
        click(&mut tool, &mut masks, 6.0, 3.0, true);
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(masks.last_history_action(), tip_before);
        assert_eq!(sel.total_for_test(), Matrix3::identity());
        assert_eq!(sel.layer_count_for_test(), 1);
        // Rebaked original is the union of the placed pixels (moved cluster A)
        // and the newly added cluster B.
        let moved_a = transform_layer(
            &ranges_from_spans(vec![Span::new(0..2, 0u32)]).unwrap(),
            &Matrix3::new_translation(&Vector2::new(5.0, 0.0)),
            img_roi(),
        )
        .unwrap();
        let cluster_b = ranges_from_spans(vec![Span::new(5..7, 3u32)]).unwrap();
        let combined = moved_a.spans::<u32>().union(cluster_b.spans());
        let expected = SortedRanges::try_from_span_iter(combined).unwrap();
        assert_eq!(sel.original_of_for_test(0), Some(expected.clone()));
        assert_eq!(sel.committed_of_for_test(0), Some(expected));
    }

    #[test]
    fn shift_rect_unions_same_layer() {
        let masks = mask_with_two_clusters();
        let mut tool = DragTool::default();
        let w = nz(100);
        let a = RectSelectionResult::new(0, 0, 3, 1, w, w).unwrap();
        tool.select_rect(&masks, &a, false);
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 1);
        // Frame hugs the contained cluster (0..2, 0), not the marquee.
        assert_eq!(sel.frame().center, Point2::new(1.0, 0.5));
        assert_eq!(sel.frame().half, Vector2::new(1.0, 0.5));
        // Shift-rect over the second cluster unions into the same entry.
        let b = RectSelectionResult::new(4, 2, 8, 4, w, w).unwrap();
        tool.select_rect(&masks, &b, true);
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 1);
        let bounds = sel.first_original_bounds_for_test();
        assert_eq!((bounds.x.start, bounds.y.start), (0, 0));
        assert_eq!((bounds.x.end, bounds.y.end), (7, 4));
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
        let union_original = tool
            .selection
            .as_ref()
            .unwrap()
            .first_original_for_test()
            .unwrap();
        assert_eq!(union_original.len(), 2);
        let (frame, _) = {
            let sel = tool.selection.as_ref().unwrap();
            sel.snapshot_transform()
        };
        let east = frame.point(Vector2::new(frame.half.x, 0.0));
        drag_resize(
            &mut tool,
            Anchor::E,
            east,
            Point2::new(east.x + frame.half.x, east.y),
            false,
        );
        let total = tool.selection.as_ref().unwrap().total_for_test();
        let expected = transform_layer(&union_original, &total, img_roi()).unwrap();
        tool.commit(&mut masks, img_roi());
        assert_eq!(layer_pixels(&masks), Some(expected));
        // Both areas survived the resize (no subtraction of the old area).
        let pixels = layer_pixels(&masks).unwrap();
        assert!(pixels.spans::<u32>().any(|s| s.y == 0));
        assert_eq!(tool.selection.as_ref().unwrap().layer_count_for_test(), 1);
    }

    /// Simulate a Resize gesture from `from` to `to` through the real update
    /// path (frame and matrix stay in sync by construction). `shift`
    /// mirrors holding Shift: exact angles, and corner resizes free of the
    /// aspect-ratio lock.
    fn drag_resize(
        tool: &mut DragTool,
        anchor: Anchor,
        from: Point2<f64>,
        to: Point2<f64>,
        shift: bool,
    ) {
        let (frame, total) = {
            let sel = tool.selection.as_ref().unwrap();
            sel.snapshot_transform()
        };
        tool.gesture = Some(Gesture::Resize(GestureResize {
            anchor,
            start: from,
            base: frame,
            base_total: total,
        }));
        tool.update_gesture_frame(to, shift);
        tool.settle();
    }

    fn mask_with_three_layers() -> MaskImage {
        use crate::{History, MaskDefaultActions, PixelAreaStack};
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        masks.add(rect_ranges(0, 0, nz(2), nz(2)));
        masks.add(rect_ranges(50, 0, nz(2), nz(2)));
        masks.add(rect_ranges(0, 50, nz(2), nz(2)));
        masks
    }

    #[test]
    fn select_layers_selects_whole_matched_layers() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, stack_spans(&masks, 0..2));
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 2);
        assert_eq!(
            sel.original_of_for_test(0),
            Some(rect_ranges(0, 0, nz(2), nz(2)))
        );
        assert_eq!(
            sel.original_of_for_test(1),
            Some(rect_ranges(50, 0, nz(2), nz(2)))
        );
        assert_eq!(sel.total_for_test(), Matrix3::identity());
        // Frame tightly covers both layers, nothing else.
        assert_eq!(sel.frame().center, Point2::new(26.0, 1.0));
        assert_eq!(sel.frame().half, Vector2::new(26.0, 1.0));
        // Staleness tip armed against the current history.
        assert_eq!(sel.tip_for_test(), masks.last_history_action());
    }

    #[test]
    fn select_layers_accepts_layer_shorthand() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, stack_spans(&masks, 2));
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 1);
        assert_eq!(
            sel.original_of_for_test(2),
            Some(rect_ranges(0, 50, nz(2), nz(2)))
        );
    }

    #[test]
    fn select_layers_without_match_drops_selection() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, stack_spans(&masks, ..));
        assert!(tool.selection.is_some());
        tool.select_layers(&masks, stack_spans(&masks, 5..8));
        assert!(tool.selection.is_none());
    }

    #[test]
    fn select_layers_replaces_existing_selection() {
        let masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, stack_spans(&masks, ..));
        assert_eq!(tool.selection.as_ref().unwrap().layer_count_for_test(), 3);
        tool.select_layers(&masks, stack_spans(&masks, 1..2));
        let sel = tool.selection.as_ref().unwrap();
        assert_eq!(sel.layer_count_for_test(), 1);
        assert!(sel.has_layer_for_test(1));
    }

    #[test]
    fn select_layers_then_gesture_moves_all_layers() {
        let mut masks = mask_with_three_layers();
        let mut tool = DragTool::default();
        tool.select_layers(&masks, stack_spans(&masks, 0..3));
        drag_move(&mut tool, Point2::new(1.0, 1.0), Point2::new(4.0, 5.0));
        tool.commit(&mut masks, img_roi());
        assert_eq!(layer_pixels(&masks), Some(rect_ranges(3, 4, nz(2), nz(2))));
        let second = masks.subgroups_stack().get(1).map(|a| a.pixels.clone());
        assert_eq!(second, Some(rect_ranges(53, 4, nz(2), nz(2))));
    }

    fn mask_blocks() -> MaskImage {
        use crate::{History, MaskDefaultActions, PixelAreaStack};
        let mut masks = MaskImage::new([100, 100], PixelAreaStack::default(), History::default());
        // 10x5 selection block at (10,10), 10x5 outsider block at (40,30).
        let mut spans = Vec::new();
        for y in 10..15 {
            spans.push(Span::new(10..20, y));
        }
        for y in 30..35 {
            spans.push(Span::new(40..50, y));
        }
        masks.add(ranges_from_spans(spans).unwrap());
        masks
    }

    fn outsider_block_ok(masks: &MaskImage) -> bool {
        layer_pixels(masks).is_some_and(|p| {
            let rows: Vec<Span<u32>> = p
                .spans::<u32>()
                .filter(|s| (30..35).contains(&s.y))
                .collect();
            rows.len() == 5 && rows.iter().all(|s| s.x.start <= 40 && 50 <= s.x.end)
        })
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
        let w = nz(100);
        let r = RectSelectionResult::new(8, 8, 22, 16, w, w).unwrap();
        tool.select_rect(&masks, &r, false);
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

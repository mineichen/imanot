use std::ops::{Deref, DerefMut};

use imask::{AffineTransformHeap, ImageDimension, Rect, SortedRanges, UnionAll};
use nalgebra::Matrix3;

use crate::{HistoryAction, ImagePainter};

use super::{overlay::draw_overlay, transform::clip_heap_to_image};

pub(crate) mod logic;
pub(crate) mod preview;

pub(crate) use logic::{ActiveSelectionLogic, LayerSelection, subtract_ranges};
pub(crate) use preview::PreviewState;

/// A live selection: pure snapshot logic plus its small preview texture. The
/// preview starts hidden (the first render uploads it) and dies with the
/// selection, so dropping or replacing a selection needs no explicit
/// invalidation; in-place mutations hide it via
/// [`ActiveSelection::merge_layer`] (union changed the snapshot) while
/// `DragTool` hides it for `rebase`, gesture restore and the offscreen case.
/// Commits need no invalidation: the placed pixels are exactly what the
/// preview already shows.
pub(crate) struct ActiveSelection {
    pub(crate) logic: ActiveSelectionLogic,
    pub(crate) preview: PreviewState,
}

impl ActiveSelection {
    /// Fresh (replacing) single-layer selection with a hidden preview.
    /// `original == committed`.
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
        img_rect: Rect<u32>,
    ) {
        if self.preview.is_visible() {
            self.preview.paint(painter);
            draw_overlay(painter, &self.logic.frame);
        } else {
            self.render_transform(egui_ctx, painter, img_rect, false);
        }
    }

    /// Wrap finished logic with a hidden preview; the next render uploads it.
    pub(crate) fn from_logic(logic: ActiveSelectionLogic) -> Self {
        Self {
            logic,
            preview: PreviewState::new(),
        }
    }

    /// Add `ranges` on layer `idx` (see [`ActiveSelectionLogic::merge_layer`])
    /// and hide the preview, which was rasterized from the old snapshot.
    pub(crate) fn merge_layer(
        &mut self,
        layer_id: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
    ) {
        self.logic.merge_layer(layer_id, ranges, background);
        self.preview.hide();
    }

    pub(crate) fn rebase(&mut self, tip: Option<HistoryAction>) {
        self.logic.rebase(tip);
        // Rebasing swaps the snapshot content and the accumulated matrix
        // the preview is rasterized from, so the uploaded pixels are stale.
        self.preview.hide();
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
        img_rect: imask::Rect<u32>,
        allow_reposition: bool,
    ) {
        let matrix = self.logic.total;
        // One lazy iterator per layer (analytic transform ∩ image), merged
        // into a single ordered, self-bounded span stream: no span is ever
        // collected.
        let chains = self.logic.layers.values().filter_map(|ls| {
            let heap = AffineTransformHeap::new(ls.original.spans::<u32>(), &matrix).ok()?;
            clip_heap_to_image(heap, img_rect)
        });

        match UnionAll::new(chains) {
            Ok(all) => {
                self.preview.show(egui_ctx, painter, all, allow_reposition);
            }
            Err(_) => {
                self.preview.hide();
            }
        }
        draw_overlay(painter, &self.logic.frame);
    }
}

// impl Deref for ActiveSelection {
//     type Target = ActiveSelectionLogic;

//     fn deref(&self) -> &Self::Target {
//         &self.logic
//     }
// }

// impl DerefMut for ActiveSelection {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.logic
//     }
// }

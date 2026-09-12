use std::ops::{Deref, DerefMut};

use imask::SortedRanges;

use crate::HistoryAction;

pub(crate) mod logic;
pub(crate) mod preview;

pub(crate) use logic::{ActiveSelectionLogic, LayerSelection, subtract_ranges};
pub(crate) use preview::PreviewState;

/// A live selection: pure snapshot logic plus its eagerly allocated GPU
/// preview. The preview dies with the selection, so dropping or replacing a
/// selection needs no explicit invalidation; in-place mutations clear it via
/// [`ActiveSelection::merge_layer`] (union changed the snapshot) while
/// `DragTool` clears it for `rebase`, `commit`, gesture restore and the
/// offscreen case. Fresh selections start non-live.
pub(crate) struct ActiveSelection {
    pub(crate) logic: ActiveSelectionLogic,
    pub(crate) preview: PreviewState,
}

impl ActiveSelection {
    /// Fresh (replacing) single-layer selection with a blank preview.
    /// `original == committed`.
    pub(crate) fn fresh_single(
        idx: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
        tip: Option<HistoryAction>,
        ctx: &egui::Context,
        img_rect: imask::Rect<u32>,
    ) -> Self {
        Self::from_logic(
            ActiveSelectionLogic::fresh_single(idx, ranges, background, tip),
            ctx,
            img_rect,
        )
    }

    /// Wrap finished logic with a blank preview texture.
    pub(crate) fn from_logic(
        logic: ActiveSelectionLogic,
        ctx: &egui::Context,
        img_rect: imask::Rect<u32>,
    ) -> Self {
        Self {
            logic,
            preview: PreviewState::new(ctx, img_rect),
        }
    }

    /// Add `ranges` on layer `idx` (see [`ActiveSelectionLogic::merge_layer`])
    /// and clear the preview, which was rasterized from the old snapshot.
    pub(crate) fn merge_layer(
        &mut self,
        layer_id: usize,
        ranges: SortedRanges<u32>,
        background: Option<SortedRanges<u32>>,
    ) {
        self.logic.merge_layer(layer_id, ranges, background);
        self.preview.clear();
    }
}

impl Deref for ActiveSelection {
    type Target = ActiveSelectionLogic;

    fn deref(&self) -> &Self::Target {
        &self.logic
    }
}

impl DerefMut for ActiveSelection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logic
    }
}

use std::sync::Arc;

use futures::FutureExt;
use imanot::{AsyncRefTask, MaskActionBuilder, RectSelection, Tool, ToolContext, ToolFactory};
use imask::{ImaskSet, SortedRanges};
use imbuf::Image;

use inference::{InferenceError, SamEmbeddings};

mod inference;
mod native_ort;

pub use native_ort::SamSession;

type RgbImageInterleaved<T> = imbuf::Image<[T; 3], 1>;

pub struct SamTool {
    embeddings: AsyncRefTask<Result<SamEmbeddings, InferenceError>>,
    session: SamSession,
    rect_selection: RectSelection,
    // If selection starts, before embeddings are ready
    last_pos: Option<[[usize; 2]; 2]>,
}

impl SamTool {
    pub fn new(session: SamSession, img: Image<[u8; 3], 1>) -> Self {
        Self {
            embeddings: AsyncRefTask::new(session.get_image_embeddings(img).boxed()),
            session,
            rect_selection: RectSelection::default(),
            last_pos: None,
        }
    }
    pub fn create_factory(session: SamSession) -> ToolFactory {
        Arc::new(move |img| {
            let tool = SamTool::new(session.clone(), img.adjust.clone());
            async move { Ok(Box::new(tool) as Box<dyn Tool>) }.boxed_local()
        })
    }
}

impl Tool for SamTool {
    fn handle_interaction(&mut self, mut ctx: ToolContext) {
        if let Some(rect_result) = self.rect_selection.drag_finished(&mut ctx) {
            self.last_pos = Some(rect_result.bounds());
        }
        if let (Some([[top_x, top_y], [bottom_x, bottom_y]]), Some(Ok(loaded_embeddings))) =
            (self.last_pos, self.embeddings.data())
        {
            let new_mask = self
                .session
                .decode_prompt(
                    top_x as f32,
                    top_y as f32,
                    bottom_x as f32,
                    bottom_y as f32,
                    loaded_embeddings,
                )
                .expect("Decoder runs on embeddings produced by the encoder");

            let width = ctx.image.image.original.width();
            let height = ctx.image.image.original.height();
            let new_mask = new_mask.with_bounds(width, height);
            if let Ok(ranges) = SortedRanges::try_from_ordered_iter(new_mask) {
                ctx.image.masks.keep_overlapping(false).add(ranges);
            }
            self.last_pos = None;

            // if let Some(x) = ctx.app.selector.current() {
            //     x.has_masks = true;
            // } else {
            //     warn!("Couldn't mark URL as containing masks")
            // }
        }
    }
}

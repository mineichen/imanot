use std::num::{NonZeroU32, NonZeroUsize, TryFromIntError};

use egui::Pos2;
use imask::Roi;

use crate::ToolContext;

pub struct RectSelectionResult {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

impl RectSelectionResult {
    pub fn new(
        min_x: usize,
        min_y: usize,
        max_x: usize,
        max_y: usize,
        image_width: NonZeroU32,
        image_height: NonZeroU32,
    ) -> Option<Self> {
        (max_x > min_x
            && max_y > min_y
            && max_x < usize::try_from(image_width.get()).ok()?
            && max_y < usize::try_from(image_height.get()).ok()?)
        .then_some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    pub fn width(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.max_x - self.min_x + 1)
            .expect("Width should always be non-zero due to validation in new()")
    }

    pub fn height(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.max_y - self.min_y + 1)
            .expect("Height should always be non-zero due to validation in new()")
    }

    pub fn bounds(&self) -> [[usize; 2]; 2] {
        [[self.min_x, self.min_y], [self.max_x, self.max_y]]
    }

    pub fn rect(&self) -> Result<Roi<u16>, TryFromIntError> {
        Ok(Roi::new(
            u16::try_from(self.min_x)?..u16::try_from(self.max_x + 1)?,
            u16::try_from(self.min_y)?..u16::try_from(self.max_y + 1)?,
        ))
    }
}

#[derive(Default)]
pub struct RectSelection {
    drag_start_image: Option<Pos2>,
}

impl RectSelection {
    pub fn drag_finished(&mut self, ctx: &mut ToolContext) -> Option<RectSelectionResult> {
        if ctx.response.drag_started() {
            let drag_delta = ctx.response.drag_delta();
            self.drag_start_image = ctx
                .response
                .interact_pointer_pos()
                .map(|screen_pos| ctx.painter.screen_to_image(screen_pos - drag_delta));
        }

        if ctx.response.dragged()
            && let (Some(start_image), Some(current_screen)) =
                (self.drag_start_image, ctx.response.interact_pointer_pos())
        {
            let start_screen = ctx.painter.image_to_screen(start_image);
            ctx.painter.draw_dotted_rect(start_screen, current_screen);
        }

        if ctx.response.drag_stopped() {
            if let (Some(start_image), Some((end_x, end_y))) =
                (self.drag_start_image, ctx.cursor_image_pos())
            {
                let start_x = start_image.x as usize;
                let start_y = start_image.y as usize;
                self.drag_start_image = None;

                // The mask texture is sized from `adjust`, not `original`, so we
                // must clip against the adjusted dimensions to avoid creating
                // ranges that exceed the mask buffer.
                let (image_width, image_height) = ctx.image.image.adjust.dimensions();
                let min_x = start_x.min(end_x);
                let min_y = start_y.min(end_y);
                let max_x = start_x
                    .max(end_x)
                    .min(usize::try_from(image_width.get()).expect("Width < usize::MAX") - 1);
                let max_y = start_y
                    .max(end_y)
                    .min(usize::try_from(image_height.get()).expect("Height < usize::MAX") - 1);

                RectSelectionResult::new(min_x, min_y, max_x, max_y, image_width, image_height)
            } else {
                self.drag_start_image = None;
                None
            }
        } else {
            None
        }
    }
}

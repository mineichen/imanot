mod brush;
mod drag;
mod pan;
mod rect;
mod rect_selection;

pub use brush::*;
pub use drag::*;
pub use pan::*;
pub use rect::*;
pub use rect_selection::*;

use std::any::Any;
use std::num::NonZeroU32;

use crate::geometry::scalar_from_f32;
use crate::{
    AffectedLayer, CursorImageSystem, ImageLoadOk, ImageStateLoaded, ImageViewer, ImgPoint, U20F12,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Insert,
    Clear,
}

pub trait Tool: Any {
    fn handle_interaction(&mut self, ctx: ToolContext);

    /// Called whenever a new image is shown, for the active and the inactive
    /// tool alike, and when a tool is set while an image is already shown.
    ///
    /// Tools live across images, so they keep their configuration (e.g. mode,
    /// brush size) and may keep state. Override this to drop what is only
    /// valid for the previous image. Most tools don't care.
    fn update_image(&mut self, _image: &ImageLoadOk) {}
}

#[derive(Default)]
pub struct DrawTool {
    mode: Mode,
    layer: AffectedLayer,
}

impl DrawTool {
    pub fn set_layer(&mut self, layer: impl Into<AffectedLayer>) -> &mut Self {
        self.layer = layer.into();
        self
    }
    pub fn set_mode(&mut self, mode: Mode) -> &mut Self {
        self.mode = mode;
        self
    }
}

/// Painter for tools that allows drawing on the image canvas.
/// Provides methods for both screen-space (zoom-independent) and
/// image-space (zoom-dependent) drawing.
pub struct ImagePainter {
    painter: egui::Painter,
    image_rect: egui::Rect,
    render_scale: f32,
}

/// Length of each dash in the dotted pattern
const DASH_LENGTH: f32 = 4.0;
/// Gap between dashes
const GAP_LENGTH: f32 = 3.0;

impl ImagePainter {
    pub fn new(painter: egui::Painter, image_rect: egui::Rect, render_scale: f32) -> Self {
        Self {
            painter,
            image_rect,
            render_scale,
        }
    }

    /// Get the underlying egui Painter for direct drawing
    pub fn painter(&self) -> &egui::Painter {
        &self.painter
    }

    /// Get the image rect in screen coordinates
    pub fn image_rect(&self) -> egui::Rect {
        self.image_rect
    }

    /// Get the render scale (screen pixels per image pixel)
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// Convert image coordinates to screen coordinates.
    ///
    /// Image coordinates are [`ImgPoint`]s in the continuous convention
    /// described at [`ImageSpace`](crate::ImageSpace). This is the only
    /// place image coordinates become egui screen coordinates.
    pub fn image_to_screen(&self, image_pos: ImgPoint) -> egui::Pos2 {
        self.image_rect.min
            + egui::Vec2::new(image_pos.x.to_num::<f32>(), image_pos.y.to_num::<f32>())
                * self.render_scale
    }

    /// Convert screen coordinates to continuous image coordinates
    /// (an [`ImgPoint`]; see
    /// [`top_left_from_index`](crate::top_left_from_index)).
    pub fn screen_to_image(&self, screen_pos: egui::Pos2) -> ImgPoint {
        let image_pos = (screen_pos - self.image_rect.min) / self.render_scale;
        ImgPoint::new(scalar_from_f32(image_pos.x), scalar_from_f32(image_pos.y))
    }

    /// The pixel under a continuous image-space position: its `floor`,
    /// clamped to the image bounds. Pixel `(x, y)` covers `[x, x+1)`, so
    /// the floor is exact — see [`ImageSpace`](crate::ImageSpace).
    pub fn pixel_at(pos: ImgPoint, width: NonZeroU32, height: NonZeroU32) -> (u32, u32) {
        let max_x = U20F12::saturating_from_num(width.get() - 1);
        let max_y = U20F12::saturating_from_num(height.get() - 1);
        let x = pos.x.clamp(U20F12::ZERO, max_x).floor().to_num::<u32>();
        let y = pos.y.clamp(U20F12::ZERO, max_y).floor().to_num::<u32>();
        (x, y)
    }

    /// Like [`ImagePainter::pixel_at`], but `None` for a position outside of
    /// the image instead of its nearest edge pixel. The viewer also reacts to
    /// pointers beside the image (e.g. in the bars next to a tall one), which
    /// are not on any pixel.
    pub fn try_pixel_at(
        pos: ImgPoint,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Option<(u32, u32)> {
        let x_range = U20F12::ZERO..U20F12::saturating_from_num(width.get());
        let y_range = U20F12::ZERO..U20F12::saturating_from_num(height.get());
        (x_range.contains(&pos.x) && y_range.contains(&pos.y))
            .then_some((pos.x.to_num::<u32>(), pos.y.to_num::<u32>()))
    }

    /// Draw a black-white dotted rectangle from corner to corner (screen coordinates)
    pub fn draw_dotted_rect(&self, start: egui::Pos2, end: egui::Pos2) {
        let rect = egui::Rect::from_two_pos(start, end);

        // Draw each side of the rectangle with alternating black/white dashes
        self.draw_dotted_line(rect.left_top(), rect.right_top());
        self.draw_dotted_line(rect.right_top(), rect.right_bottom());
        self.draw_dotted_line(rect.right_bottom(), rect.left_bottom());
        self.draw_dotted_line(rect.left_bottom(), rect.left_top());
    }

    /// Draw a dotted line with alternating black and white segments (screen coordinates)
    pub fn draw_dotted_line(&self, start: egui::Pos2, end: egui::Pos2) {
        let direction = end - start;
        let length = direction.length();
        if length < 0.001 {
            return;
        }
        let unit = direction / length;
        let segment_length = DASH_LENGTH + GAP_LENGTH;

        let mut offset = 0.0;
        let mut is_black = true;

        while offset < length {
            let dash_end = (offset + DASH_LENGTH).min(length);
            let p1 = start + unit * offset;
            let p2 = start + unit * dash_end;

            let color = if is_black {
                egui::Color32::BLACK
            } else {
                egui::Color32::WHITE
            };

            self.painter
                .line_segment([p1, p2], egui::Stroke::new(1.5, color));

            offset += segment_length;
            is_black = !is_black;
        }
    }
}

#[non_exhaustive]
pub struct ToolContext<'a> {
    pub image: &'a mut ImageStateLoaded,
    pub response: &'a egui::Response,
    pub egui: &'a egui::Context,
    pub painter: &'a mut ImagePainter,
    pub viewer: &'a mut ImageViewer,
    pub cursor_image: &'a mut CursorImageSystem,
    /// Set to `true` while the tool holds state tied to the current image
    /// (e.g. a selection or a running gesture). The current frame then keeps
    /// back any image passed to [`State::set_image`](crate::State::set_image)
    /// instead of replacing it, and does not process it in any way until a
    /// frame in which nobody sets it anymore. It starts out `false` each frame,
    /// and the pointer being down on the viewer holds the image without tool
    /// help.
    pub postpone_new_images: &'a mut bool,
}

impl<'a> ToolContext<'a> {
    pub fn new(
        image: &'a mut ImageStateLoaded,
        response: &'a egui::Response,
        egui: &'a egui::Context,
        painter: &'a mut ImagePainter,
        viewer: &'a mut ImageViewer,
        cursor_image: &'a mut CursorImageSystem,
        postpone_new_images: &'a mut bool,
    ) -> Self {
        Self {
            image,
            response,
            egui,
            painter,
            viewer,
            cursor_image,
            postpone_new_images,
        }
    }

    /// Get the cursor position in image coordinates (pixels)
    /// Returns None if the cursor is not over the image
    pub fn cursor_image_pos(&self) -> Option<(usize, usize)> {
        self.response.interact_pointer_pos().map(|screen_pos| {
            let image_pos = self.painter.screen_to_image(screen_pos);
            let (x, y) = ImagePainter::pixel_at(
                image_pos,
                self.image.image.original.width(),
                self.image.image.original.height(),
            );
            (x as usize, y as usize)
        })
    }

    /// Get the cursor position in continuous image coordinates (see
    /// [`top_left_from_index`](crate::top_left_from_index)):
    /// subpixel-accurate, with pixel `(x, y)` covering `[x, x+1)` so its
    /// center is at `(x + 0.5, y + 0.5)`. Not clamped to the image.
    /// Returns None if the cursor is not over the widget.
    pub fn cursor_image_posf(&self) -> Option<ImgPoint> {
        self.response
            .interact_pointer_pos()
            .map(|screen_pos| self.painter.screen_to_image(screen_pos))
    }

    /// The pixel of the shown image under the cursor. `None` if the cursor is
    /// not over the widget (see [`ToolContext::cursor_image_posf`]) or beside
    /// the image (see [`ImagePainter::try_pixel_at`]).
    pub fn try_pixel_at(&self) -> Option<(u32, u32)> {
        let (width, height) = self.image.image.adjust.dimensions();
        ImagePainter::try_pixel_at(self.cursor_image_posf()?, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f32, y: f32) -> ImgPoint {
        ImgPoint::new(U20F12::from_num(x), U20F12::from_num(y))
    }

    fn dims(n: u32) -> (NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(n).expect("n > 0"),
            NonZeroU32::new(n).expect("n > 0"),
        )
    }

    #[test]
    fn pixel_at_floors_to_the_pixel_under_the_cursor() {
        let (w, h) = dims(10);
        // Pixel 5 covers [5, 6): its left edge at 5.0 already belongs to it.
        assert_eq!(ImagePainter::pixel_at(pt(5.4, 3.2), w, h), (5, 3));
        assert_eq!(ImagePainter::pixel_at(pt(5.0, 3.0), w, h), (5, 3));
        assert_eq!(ImagePainter::pixel_at(pt(6.0, 4.0), w, h), (6, 4));
        // Sub-pixel positions stay inside the pixel of their floor.
        assert_eq!(ImagePainter::pixel_at(pt(5.99, 0.99), w, h), (5, 0));
    }

    #[test]
    fn pixel_at_clamps_to_image_bounds() {
        let (w, h) = dims(10);
        // The image's top-left corner is the origin (0, 0).
        assert_eq!(ImagePainter::pixel_at(pt(0.0, 0.0), w, h), (0, 0));
        assert_eq!(ImagePainter::pixel_at(pt(15.0, 9.49), w, h), (9, 9));
        // Beyond the image: clamped to the last pixel.
        assert_eq!(ImagePainter::pixel_at(pt(100.0, 200.0), w, h), (9, 9));
    }

    #[test]
    fn try_pixel_at_is_none_outside_of_the_image() {
        let (w, h) = dims(10);
        assert_eq!(ImagePainter::try_pixel_at(pt(0.0, 0.0), w, h), Some((0, 0)));
        assert_eq!(
            ImagePainter::try_pixel_at(pt(9.99, 9.99), w, h),
            Some((9, 9))
        );
        // The far edge belongs to no pixel, nor does anything beyond it
        assert_eq!(ImagePainter::try_pixel_at(pt(10.0, 5.0), w, h), None);
        assert_eq!(ImagePainter::try_pixel_at(pt(5.0, 10.0), w, h), None);
    }

    #[test]
    fn image_screen_conversion_roundtrips() {
        let ctx = egui::Context::default();
        let rect =
            egui::Rect::from_min_size(egui::Pos2::new(10.0, 20.0), egui::Vec2::new(300.0, 300.0));
        let painter = ImagePainter::new(ctx.layer_painter(egui::LayerId::background()), rect, 2.0);
        // All values exactly representable in `U20F12`.
        let image_pos = pt(3.25, 1.75);
        let screen_pos = painter.image_to_screen(image_pos);
        let back = painter.screen_to_image(screen_pos);
        assert_eq!(back, image_pos);
    }

    #[test]
    fn image_to_screen_anchors_pixel_corners() {
        let ctx = egui::Context::default();
        let rect =
            egui::Rect::from_min_size(egui::Pos2::new(10.0, 20.0), egui::Vec2::new(300.0, 300.0));
        let painter = ImagePainter::new(ctx.layer_painter(egui::LayerId::background()), rect, 2.0);
        // Pixel (0, 0)'s top-left corner is the image origin (0, 0): the
        // image rect's min.
        assert_eq!(
            painter.image_to_screen(crate::top_left_of_pixel(0usize, 0usize)),
            rect.min
        );
        // Pixel centers sit at +0.5: pixel (1, 1)'s center is at (1.5, 1.5),
        // so it lands 1.5 · render_scale = 3 screen px from the rect's min.
        let center = painter.image_to_screen(pt(1.5, 1.5));
        assert_eq!(center, rect.min + egui::Vec2::splat(3.0));
    }
}

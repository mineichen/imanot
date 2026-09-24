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

use crate::{AffectedLayer, CursorImageSystem, ImageStateLoaded, ImageViewer};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Insert,
    Clear,
}

pub trait Tool: Any {
    fn handle_interaction(&mut self, ctx: ToolContext);
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

    /// Convert image coordinates to screen coordinates
    pub fn image_to_screen(&self, image_pos: egui::Pos2) -> egui::Pos2 {
        self.image_rect.min + (image_pos.to_vec2() * self.render_scale)
    }

    /// Convert screen coordinates to image coordinates
    pub fn screen_to_image(&self, screen_pos: egui::Pos2) -> egui::Pos2 {
        ((screen_pos - self.image_rect.min) / self.render_scale).to_pos2()
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
            (image_pos.x as usize, image_pos.y as usize)
        })
    }
}

use std::sync::Arc;

use egui::Vec2;
use futures::FutureExt;

use crate::{Tool, ToolContext, ToolFactory};

/// Pan tool for moving the viewport around the image
#[derive(Default)]
#[non_exhaustive]
pub struct PanTool;

impl PanTool {
    pub fn create_factory() -> ToolFactory {
        Arc::new(|_| async { Ok(Box::new(PanTool::default()) as Box<dyn Tool>) }.boxed_local())
    }
}

impl Tool for PanTool {
    fn handle_interaction(&mut self, ctx: ToolContext) {
        // Panning logic will be moved here from ImageViewer
        let viewer = ctx.viewer;
        let response = &ctx.response;

        let drag_delta = response.drag_delta();
        let drag_delta = (drag_delta.x.abs() > f32::EPSILON || drag_delta.y.abs() > f32::EPSILON)
            .then_some(drag_delta);
        let zoom = response.hover_pos().and_then(|hover| {
            let speed = ctx.egui.options(|o| o.input_options.scroll_zoom_speed);
            let factor = ctx
                .egui
                .input(|i| i.zoom_delta() * (i.smooth_scroll_delta.y * speed).exp());
            ((factor - 1.0).abs() > f32::EPSILON).then_some((hover, factor))
        });

        if drag_delta.is_none() && zoom.is_none() {
            return;
        }

        let original_image_size = Vec2::new(
            ctx.image.image.original.width().get() as f32,
            ctx.image.image.original.height().get() as f32,
        );
        let viewport_rect = response.rect;
        let viewport_size = viewport_rect.size();
        let fit_scale =
            (viewport_size.x / original_image_size.x).min(viewport_size.y / original_image_size.y);

        if let Some((hover, factor)) = zoom {
            let cursor_img = ctx.painter.screen_to_image(hover);
            viewer.modify_zoom(|x| x / factor);
            let render_scale = fit_scale / viewer.zoom();

            // Keep the image point under the cursor fixed while zooming
            let center = cursor_img.to_vec2() + (viewport_rect.center() - hover) / render_scale;
            viewer.set_pan_offset(center / original_image_size);
        }

        if let Some(drag_delta) = drag_delta {
            let render_scale = fit_scale / viewer.zoom();

            let delta_norm = drag_delta / (render_scale * original_image_size);
            let mut new_offset = viewer.pan_offset() - delta_norm;

            let (min_pan, max_pan) =
                viewer.pan_bounds(original_image_size, viewport_size, render_scale);

            let move_left = viewer.pan_offset().x < new_offset.x;
            let move_top = viewer.pan_offset().y < new_offset.y;

            if !move_left && new_offset.x < min_pan.x {
                new_offset.x = viewer.pan_offset().x.min(min_pan.x);
            }

            if move_left && new_offset.x > max_pan.x {
                new_offset.x = viewer.pan_offset().x.max(max_pan.x);
            }

            if !move_top && new_offset.y < min_pan.y {
                new_offset.y = viewer.pan_offset().y.min(min_pan.y);
            }

            if move_top && new_offset.y > max_pan.y {
                new_offset.y = viewer.pan_offset().y.max(max_pan.y);
            }

            viewer.set_pan_offset(new_offset);
        }
    }
}

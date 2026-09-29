use egui::{
    self, ImageSource, InnerResponse, Pos2, Rect, Sense, TextureOptions, Vec2,
    load::{SizedTexture, TexturePoll},
};

use crate::ImagePainter;

/// Largest fraction of the viewport's smaller side a single image pixel may
/// cover on screen. Defines the deepest possible zoom-in.
const MAX_PIXEL_VIEWPORT_FRACTION: f32 = 1.0 / 10.0;

pub struct ImageViewer {
    // Raw zoom level, may hold values outside the valid range
    // (min_zoom..1.0). It is clamped whenever it is applied, see `zoom()`.
    // 1.0 means, that image width or height fits the viewport and the other dimension is smaller than the viewport
    // The minimum depends on the image resolution: one image pixel can grow
    // to at most `MAX_PIXEL_VIEWPORT_FRACTION` of the viewport's smaller side.
    zoom: f32,
    // Normalized image coordinate of the viewport center per axis in [0, 1]:
    // 0.0 = left/top image edge is at the viewport center
    // 0.5 = image center is at the viewport center (fully centered)
    // 1.0 = right/bottom image edge is at the viewport center
    pan_offset: Vec2,
}

impl ImageViewer {
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan_offset = Vec2::splat(0.5);
    }

    /// Zoom level clamped to the valid range `min_zoom..1.0`.
    /// The stored value may be out of range; it is clamped only here,
    /// when it is applied.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(0., 1.);
    }

    pub fn modify_zoom(&mut self, zoom: impl Fn(f32) -> f32) {
        self.zoom = zoom(self.zoom.clamp(0., 1.));
    }

    /// Deepest allowed zoom-in for the given image and viewport sizes.
    /// `render_scale = fit_scale / zoom` must not exceed
    /// `min(viewport) * MAX_PIXEL_VIEWPORT_FRACTION`, i.e. a single image
    /// pixel covers at most that fraction of the viewport's smaller side.
    fn compute_min_zoom(fit_scale: f32, viewport_size: Vec2) -> f32 {
        (fit_scale / (viewport_size.min_elem() * MAX_PIXEL_VIEWPORT_FRACTION)).min(1.0)
    }

    pub fn pan_offset(&self) -> Vec2 {
        self.pan_offset
    }

    pub fn set_pan_offset(&mut self, offset: Vec2) {
        self.pan_offset = offset;
    }

    pub fn pan_bounds(
        &self,
        original_image_size: Vec2,
        viewport_size_px: Vec2,
        render_scale: f32,
    ) -> (Vec2, Vec2) {
        // Half viewport size measured in original image pixels
        let half_viewport_in_image = viewport_size_px / (2.0 * render_scale);

        let min_pan = (half_viewport_in_image / original_image_size)
            .clamp(Vec2::splat(0.0), Vec2::splat(0.5));

        (min_pan, Vec2::splat(1.0) - min_pan)
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        sources: impl Iterator<Item = ImageSource<'static>>,
        sense: Option<Sense>,
    ) -> InnerResponse<Option<ImageViewerInteraction>> {
        let available_size = ui.available_size();
        let viewport_rect = ui.available_rect_before_wrap();

        let mut iter = sources.map(|i| {
            egui::Image::new(i)
                .maintain_aspect_ratio(true)
                // Important for Texture-ImageSources
                .fit_to_exact_size(available_size)
                .texture_options(TextureOptions {
                    magnification: egui::TextureFilter::Nearest,
                    ..Default::default()
                })
        });
        fn next_loaded(
            iter: impl Iterator<Item = egui::Image<'static>>,
            ui: &egui::Ui,
        ) -> Option<(SizedTexture, egui::Image<'static>)> {
            iter.filter_map(|image| {
                let tlr = image.load_for_size(ui.ctx(), ui.available_size());
                match tlr {
                    Ok(TexturePoll::Ready { texture }) => Some((texture, image)),
                    _ => None,
                }
            })
            .next()
        }

        let Some((first_texture, _image)) = next_loaded(&mut iter, ui) else {
            return InnerResponse {
                inner: None,
                response: ui.response(),
            };
        };

        let original_image_size = first_texture.size;
        let my_sense = Sense::hover().union(Sense::drag());
        let combined_sense = sense.map(|s| s.union(my_sense)).unwrap_or(my_sense);

        let response = ui.allocate_rect(viewport_rect, combined_sense);
        let p = ui.painter().with_clip_rect(viewport_rect);
        // p.rect(
        //     viewport_rect,
        //     10.0,
        //     egui::Color32::WHITE,
        //     egui::Stroke::NONE,
        //     egui::StrokeKind::Inside,
        // );

        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));

        // Compute scale so that at zoom=1.0 the whole image fits the viewport (letterboxed/pillarboxed)
        let viewport_size = viewport_rect.size();
        let fit_scale =
            (viewport_size.x / original_image_size.x).min(viewport_size.y / original_image_size.y);

        // The deepest zoom-in depends on the image resolution: zooming in
        // further would make a single image pixel larger than
        // `MAX_PIXEL_VIEWPORT_FRACTION` of the viewport's smaller side.
        let min_zoom = Self::compute_min_zoom(fit_scale, viewport_size);
        self.zoom = self.zoom().clamp(min_zoom, 1.);

        let cursor_image_pos = {
            let render_scale = fit_scale / self.zoom;

            let is_zoomed_out = (self.zoom - 1.0).abs() <= f32::EPSILON;
            if is_zoomed_out {
                self.pan_offset = Vec2::splat(0.5);
            }

            response.hover_pos().and_then(|hover| {
                // Where to place the image so that the image point at `pan_offset`
                // (normalized) appears at the viewport center.
                let center_img_px = self.pan_offset * original_image_size;
                let pixel_offset = viewport_size * 0.5 - center_img_px * render_scale;
                let screen_rel = (hover - viewport_rect.min.to_vec2()).to_vec2();

                // Use zoom relative to fit, so p stays constant in original image space
                let rel_zoom = self.zoom / fit_scale;
                let p = (screen_rel - pixel_offset) * rel_zoom;

                // log::info!(
                //     "Hover: {:?}, pan_offset: {:?}, zoom: {:?}, pixel_offset: {:?}, rel: {:?}",
                //     (p.x, p.y),
                //     (self.pan_offset.x, self.pan_offset.y),
                //     self.zoom,
                //     pixel_offset,
                //     screen_rel,
                // );
                if p.x < 0.0
                    || p.y < 0.0
                    || p.x > original_image_size.x
                    || p.y > original_image_size.y
                {
                    None
                } else {
                    Some((p.x as _, p.y as _))
                }
            })
        };

        let render_scale = fit_scale / self.zoom;
        let image_size_px = original_image_size * render_scale;
        let pixel_offset = viewport_size * 0.5 - self.pan_offset * image_size_px;

        let image_rect_unclipped =
            Rect::from_min_size(viewport_rect.min + pixel_offset, image_size_px);

        p.image(
            first_texture.id,
            image_rect_unclipped,
            uv,
            egui::Color32::WHITE,
        );
        while let Some((texture, _)) = next_loaded(&mut iter, ui) {
            p.image(texture.id, image_rect_unclipped, uv, egui::Color32::WHITE);
        }

        let image_painter = ImagePainter::new(p, image_rect_unclipped, render_scale);

        let interaction = ImageViewerInteraction {
            original_image_size,
            cursor_image_pos,
            image_painter,
        };

        InnerResponse {
            inner: Some(interaction),
            response,
        }
    }
}

impl Default for ImageViewer {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_offset: Vec2::splat(0.5),
        }
    }
}

pub struct ImageViewerInteraction {
    pub original_image_size: Vec2,
    /// Cursor position relative to image (in image pixels)
    pub cursor_image_pos: Option<(usize, usize)>,
    /// Allows painting stuff on the image with image coordinates
    pub image_painter: ImagePainter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_zoom_caps_pixel_size_at_viewport_fraction() {
        // 4K image fitting a 4K viewport: fit_scale = 1, max render_scale = 2160/20
        let min_zoom = ImageViewer::compute_min_zoom(1.0, Vec2::new(3840.0, 2160.0));
        assert!((min_zoom - 20.0 / 2160.0).abs() < 1e-6);
        assert!(1.0 / min_zoom <= 2160.0 / 20.0 + 1e-6);

        // 400x400 image in a 1000x800 viewport: fit_scale = 2, max render_scale = 40
        let min_zoom = ImageViewer::compute_min_zoom(2.0, Vec2::new(1000.0, 800.0));
        assert!((min_zoom - 0.05).abs() < 1e-6);
    }

    #[test]
    fn min_zoom_never_exceeds_fit() {
        // Image smaller than viewport/20: even at fit a pixel exceeds the
        // fraction, so the limit saturates at 1.0 (no zooming in past fit).
        let min_zoom = ImageViewer::compute_min_zoom(100.0, Vec2::new(1000.0, 1000.0));
        assert_eq!(min_zoom, 1.0);
    }

    #[test]
    fn zoom_is_stored_raw_and_clamped_on_render() {
        let mut viewer = ImageViewer {
            ..Default::default()
        };

        viewer.set_zoom(42.0);
        assert_eq!(viewer.zoom, 1.0);
        assert_eq!(viewer.zoom(), 1.0);

        viewer.set_zoom(0.1);
        assert_eq!(viewer.zoom, 0.1);
        assert_eq!(viewer.zoom(), 0.1);

        viewer.modify_zoom(|z| z * 10.0);
        assert_eq!(viewer.zoom, 1.0);
        assert_eq!(viewer.zoom(), 1.0);
    }
}

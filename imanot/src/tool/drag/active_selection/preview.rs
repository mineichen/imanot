use egui::{Color32, ColorImage, Pos2, Rect as EguiRect, TextureHandle, TextureOptions};
use imask::{Rect, Span};

use crate::ImagePainter;

/// Color of the live transform preview (same look as the brush preview).
const PREVIEW_COLOR: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 128);

/// Live preview texture. The mask itself is never touched during a gesture;
/// the transformed snapshot is rasterized into this image-sized texture
/// (partial uploads only, mirroring the brush preview).
///
/// The texture is created eagerly with the selection (every selection has
/// pixels, so every selection needs its preview) and dies with it. Only
/// `last_bounds` stays optional: it tracks whether the texture currently
/// shows the selection and provides the dirty region for partial uploads.
pub(crate) struct PreviewState {
    texture: TextureHandle,
    last_bounds: Option<Rect<u32>>,
}

impl PreviewState {
    /// Allocate a blank image-sized texture. Starts non-live: the first
    /// `update` uploads exactly the new bounds.
    pub(crate) fn new(ctx: &egui::Context, img_rect: Rect<u32>) -> Self {
        let img_w = img_rect.width.get() as usize;
        let img_h = img_rect.height.get() as usize;
        Self {
            texture: ctx.load_texture(
                "drag_preview",
                ColorImage::new([img_w, img_h], vec![Color32::TRANSPARENT; img_w * img_h]),
                TextureOptions::NEAREST,
            ),
            last_bounds: None,
        }
    }

    /// Erase any preview pixels from the GPU texture and forget the dirty
    /// region. Must run whenever the preview becomes invalid while the same
    /// handle is reused (in-place mutation, gesture restore, offscreen
    /// selection), otherwise stale pixels stay visible. Dropping or replacing
    /// the whole selection needs no `clear`: the handle dies with it.
    pub(crate) fn clear(&mut self) {
        if let Some(bounds) = self.last_bounds.take() {
            let w = bounds.width.get() as usize;
            let h = bounds.height.get() as usize;
            self.texture.set_partial(
                [bounds.x as usize, bounds.y as usize],
                ColorImage::new([w, h], vec![Color32::TRANSPARENT; w * h]),
                TextureOptions::NEAREST,
            );
        }
    }

    /// Whether the texture currently shows the selection (`update` ran since
    /// the last `clear`). A live texture is always up to date and can just be
    /// repainted without re-rasterizing.
    pub(crate) fn is_live(&self) -> bool {
        self.last_bounds.is_some()
    }

    /// Paint the live texture over the image without re-uploading.
    pub(crate) fn paint(&self, painter: &mut ImagePainter) {
        painter.painter().image(
            self.texture.id(),
            painter.image_rect(),
            EguiRect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    /// Rasterize already-clipped `spans` (tightly covered by `bounds`) into
    /// the texture and paint it over the image. Only the union of the previous
    /// and the new bounds, clamped to the image, is re-uploaded.
    pub(crate) fn update(
        &mut self,
        painter: &mut ImagePainter,
        spans: impl Iterator<Item = Span<u32>>,
        bounds: Rect<u32>,
        img_rect: Rect<u32>,
    ) {
        let dirty = self
            .last_bounds
            .map_or(bounds, |p| p.union(&bounds))
            .intersection(&img_rect);
        if let Some(dirty) = dirty {
            let dx = dirty.x as usize;
            let dy = dirty.y as usize;
            let dw = dirty.width.get() as usize;
            let dh = dirty.height.get() as usize;
            let mut pixels = vec![Color32::TRANSPARENT; dw * dh];
            for span in spans {
                let y = span.y as usize;
                if y < dy || y >= dy + dh {
                    continue;
                }
                let xs = (span.x.start as usize).max(dx);
                let xe = (span.x.end as usize).min(dx + dw);
                if xs < xe {
                    let fill_range = (y - dy) * dw + (xs - dx)..(y - dy) * dw + (xe - dx);
                    pixels[fill_range].fill(PREVIEW_COLOR);
                }
            }
            self.texture.set_partial(
                [dx, dy],
                ColorImage::new([dw, dh], pixels),
                TextureOptions::NEAREST,
            );
        }
        self.last_bounds = Some(bounds);

        self.paint(painter);
    }
}

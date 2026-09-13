use std::sync::Arc;

use egui::{Color32, ColorImage, Pos2, Rect as EguiRect, TextureHandle, TextureOptions, Vec2};
use imask::{ImageDimension, Rect, Span};

use crate::ImagePainter;

/// Color of the live transform preview (same look as the brush preview).
const PREVIEW_COLOR: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 128);

/// Live preview: a small texture holding exactly the selection's current
/// pixels (`bounds.w × bounds.h`), painted at `origin` in image coordinates.
///
/// The mask itself is never touched during a gesture; the transformed
/// snapshot is rasterized here instead.
///
/// Moves never touch the texture — only the paint position changes (see
/// `try_reposition`). Anything that changes the pixels (resize, rotate, added
/// layers, …) goes through `show`, which re-rasterizes into the reusable
/// staging `buffer` and re-uploads the whole (small) texture at once.
/// `hide` drops the handle but keeps the buffer capacity; the next `show`
/// recreates the handle.
///
/// `buffer` is shared with egui as an `Arc`: `set` takes
/// `impl Into<ImageData>` and `ImageData::Color` holds an `Arc<ColorImage>`,
/// so each upload clones the `Arc` (cheap refcount bump) while we keep ours.
/// As long as egui dropped its clone after the previous frame (the common
/// case — at most one upload per frame), the next staging reuses the same
/// allocation via `Arc::get_mut` instead of allocating a fresh vector.
pub(crate) struct PreviewState {
    texture: Option<TextureHandle>,
    origin: [u32; 2],
    size: [usize; 2],
    buffer: Arc<ColorImage>,
}

impl PreviewState {
    pub(crate) fn new() -> Self {
        Self {
            texture: None,
            origin: [0, 0],
            size: [0, 0],
            buffer: Arc::new(ColorImage::new([0, 0], Vec::new())),
        }
    }

    /// Whether there are uploaded pixels to paint.
    pub(crate) fn is_visible(&self) -> bool {
        self.texture.is_some()
    }

    /// Rasterize the ordered `spans` (which carry their own bounds) into the
    /// staging buffer and upload the whole texture, then paint it at the
    /// bounds origin. Recreates the texture handle when there is none;
    /// resizes the GPU texture in place otherwise.
    pub(crate) fn show(
        &mut self,
        ctx: &egui::Context,
        painter: &mut ImagePainter,
        spans: impl Iterator<Item = Span<u32>> + ImageDimension,
    ) {
        let bounds = spans.bounds();
        let w = bounds.width.get() as usize;
        let h = bounds.height.get() as usize;
        let img = self.stage(w, h);
        fill_spans(&mut img.pixels, spans, bounds);
        match self.texture.as_mut() {
            Some(handle) => handle.set(self.buffer.clone(), TextureOptions::NEAREST),
            None => {
                self.texture = Some(ctx.load_texture(
                    "drag_preview",
                    self.buffer.clone(),
                    TextureOptions::NEAREST,
                ));
            }
        }
        self.origin = [bounds.x, bounds.y];
        self.size = [w, h];
        self.paint(painter);
    }

    /// Move the already-uploaded pixels to the `bounds` origin without
    /// re-rasterizing. Returns `false` (painting nothing) when the uploaded
    /// pixels don't cover `bounds` — the caller must `show` instead.
    ///
    /// Only valid for pure translations: same size means same pixels.
    pub(crate) fn try_reposition(&mut self, painter: &mut ImagePainter, bounds: Rect<u32>) -> bool {
        if !self.texture.is_some()
            || self.size != [bounds.width.get() as usize, bounds.height.get() as usize]
        {
            return false;
        }
        self.origin = [bounds.x, bounds.y];
        self.paint(painter);
        true
    }

    /// Drop the uploaded pixels. Painting afterwards draws nothing until the
    /// next `show`.
    pub(crate) fn hide(&mut self) {
        self.texture = None;
        self.size = [0, 0];
    }

    /// Paint the uploaded pixels at their current origin. Draws nothing when
    /// hidden.
    pub(crate) fn paint(&self, painter: &mut ImagePainter) {
        let Some(texture) = &self.texture else {
            return;
        };
        let min = painter.image_to_screen(Pos2::new(self.origin[0] as f32, self.origin[1] as f32));
        let max = painter.image_to_screen(Pos2::new(
            (self.origin[0] as usize + self.size[0]) as f32,
            (self.origin[1] as usize + self.size[1]) as f32,
        ));
        painter.painter().image(
            texture.id(),
            EguiRect::from_min_max(min, max),
            EguiRect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    /// Stage a `w × h` image for the next upload, reusing the previous
    /// allocation whenever we are its sole owner. Returns it for span-filling
    /// via `fill_spans` (which clears every cell the spans don't cover) —
    /// callers upload via `self.buffer.clone()`. The staged pixels are stale
    /// until filled; fill-before-clone ordering is load-bearing: cloning
    /// first would mutate the image egui may not have consumed yet.
    fn stage(&mut self, w: usize, h: usize) -> &mut ColorImage {
        let n = w * h;
        // Count first so no borrow is held across the realloc branch below.
        if Arc::strong_count(&self.buffer) == 1 {
            let img = Arc::get_mut(&mut self.buffer).expect("counted a single owner");
            img.size = [w, h];
            img.source_size = Vec2::new(w as f32, h as f32);
            // No clearing: `fill_spans` owns the full content and clears
            // every uncovered cell, so only growth needs fresh elements.
            if img.pixels.len() != n {
                img.pixels.resize(n, Color32::TRANSPARENT);
            }
            log::debug!(
                "drag preview: reuse buffer (capacity {}, need {n}px)",
                img.pixels.capacity()
            );
            img
        } else {
            let shared = Arc::strong_count(&self.buffer);
            log::debug!("drag preview: realloc ({n}px, still shared by {shared})");
            self.buffer = Arc::new(ColorImage::filled([w, h], Color32::TRANSPARENT));
            Arc::get_mut(&mut self.buffer).expect("fresh Arc is solely owned")
        }
    }
}

/// Fill `pixels` (a `bounds.w × bounds.h` row-major image) from the ordered
/// `spans`, resetting every uncovered cell in a single linear pass: cells
/// before each span go `TRANSPARENT`, covered cells `PREVIEW_COLOR`, and the
/// tail after the last span `TRANSPARENT`. The incoming content is
/// irrelevant — stale staging memory is fine.
///
/// Spans must come pre-clipped to `bounds` in non-decreasing cell order
/// (guaranteed when the stream carries its own `ImageDimension`, e.g. a
/// `UnionAll` over clipped chains): every span lands inside the image.
fn fill_spans(pixels: &mut [Color32], spans: impl Iterator<Item = Span<u32>>, bounds: Rect<u32>) {
    let w = bounds.width.get() as usize;
    let h = bounds.height.get() as usize;
    debug_assert_eq!(pixels.len(), w * h);
    let mut cursor = 0;
    for span in spans {
        let y = span.y - bounds.y;
        debug_assert!(y < bounds.height.get());
        debug_assert!(span.x.start >= bounds.x && span.x.end <= bounds.x + bounds.width.get());
        let start = y as usize * w + (span.x.start - bounds.x) as usize;
        let end = y as usize * w + (span.x.end - bounds.x) as usize;
        debug_assert!(start >= cursor && end <= pixels.len());
        pixels[cursor..start].fill(Color32::TRANSPARENT);
        pixels[start..end].fill(PREVIEW_COLOR);
        cursor = end;
    }
    pixels[cursor..].fill(Color32::TRANSPARENT);
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn bounds(x: u32, y: u32, w: u32, h: u32) -> Rect<u32> {
        Rect::new(
            x,
            y,
            NonZeroU32::new(w).unwrap(),
            NonZeroU32::new(h).unwrap(),
        )
    }

    #[test]
    fn fill_spans_maps_to_local_coords() {
        let mut pixels = vec![Color32::TRANSPARENT; 4 * 3];
        fill_spans(
            &mut pixels,
            vec![Span::new(11..13, 21)].into_iter(),
            bounds(10, 20, 4, 3),
        );
        // (11, 21) and (12, 21) land at local (1, 1) and (2, 1).
        assert_eq!(pixels[5], PREVIEW_COLOR);
        assert_eq!(pixels[6], PREVIEW_COLOR);
        assert_eq!(pixels[0], Color32::TRANSPARENT);
        assert_eq!(pixels[2 * 4 + 3], Color32::TRANSPARENT);
    }

    #[test]
    fn fill_spans_clears_stale_cells() {
        // Stale staging content (everything set) must not leak through:
        // only covered cells stay set.
        let mut pixels = vec![PREVIEW_COLOR; 2 * 2];
        fill_spans(
            &mut pixels,
            vec![Span::new(10..11, 20)].into_iter(),
            bounds(10, 20, 2, 2),
        );
        assert_eq!(pixels[0], PREVIEW_COLOR);
        assert!(pixels[1..].iter().all(|&px| px == Color32::TRANSPARENT));
    }

    #[test]
    fn staging_buffer_reused_when_sole_owner() {
        let mut preview = PreviewState::new();
        preview.stage(10, 5);
        let ptr_before = Arc::as_ptr(&preview.buffer);
        let cap_before = preview.buffer.pixels.capacity();
        assert!(cap_before >= 50);
        // Smaller restage while solely owned: same Arc, same capacity.
        // Content is stale until `fill_spans` runs — only the allocation
        // matters here.
        preview.stage(8, 4);
        assert_eq!(ptr_before, Arc::as_ptr(&preview.buffer));
        assert_eq!(cap_before, preview.buffer.pixels.capacity());
        assert_eq!(preview.buffer.size, [8, 4]);
    }

    #[test]
    fn staging_buffer_reallocs_when_shared() {
        let mut preview = PreviewState::new();
        preview.stage(10, 5);
        // Simulate egui still holding the previous upload.
        let held = preview.buffer.clone();
        preview.stage(10, 5);
        assert!(!Arc::ptr_eq(&held, &preview.buffer));
        assert_eq!(preview.buffer.size, [10, 5]);
    }

    #[test]
    fn show_reposition_hide() {
        let ctx = egui::Context::default();
        let screen = EguiRect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 100.0));
        let mut painter =
            ImagePainter::new(ctx.layer_painter(egui::LayerId::background()), screen, 1.0);
        let mut preview = PreviewState::new();
        assert!(!preview.is_visible());

        preview.show(&ctx, &mut painter, bounds(10, 20, 4, 3).into_spans());
        assert!(preview.is_visible());
        assert_eq!(preview.origin, [10, 20]);

        // Same-size move: only the origin changes.
        assert!(preview.try_reposition(&mut painter, bounds(12, 21, 4, 3)));
        assert_eq!(preview.origin, [12, 21]);
        // Different size: the caller must `show` again.
        assert!(!preview.try_reposition(&mut painter, bounds(12, 21, 5, 3)));

        preview.hide();
        assert!(!preview.is_visible());
        assert!(!preview.try_reposition(&mut painter, bounds(10, 20, 4, 3)));
    }
}

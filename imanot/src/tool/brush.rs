use std::{
    num::{NonZero, NonZeroU16},
    ops::{Deref, DerefMut},
    sync::Arc,
};

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use futures::FutureExt;
use imask::{BitmapToSpanIter, Rect, SortedRanges, WithRoi};

use crate::{
    AffectedLayer, DrawTool, ImagePainter, MaskActionBuilder, MaskDefaultActions, Mode, Tool,
    ToolContext, ToolFactory,
};

struct StrokeState {
    width: usize,
    height: usize,
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
    mask: Vec<bool>,
    last_pos: Option<(usize, usize)>,
    texture: Option<TextureHandle>,
    dirty: Option<Rect<usize>>,
}

impl StrokeState {
    fn new(width: usize, height: usize, x: usize, y: usize) -> Self {
        Self {
            width,
            height,
            min_x: x,
            max_x: x,
            min_y: y,
            max_y: y,
            mask: vec![false; width * height],
            last_pos: None,
            texture: None,
            dirty: None,
        }
    }

    fn bounded_spans(
        &self,
        image_width: NonZero<u32>,
        image_height: NonZero<u32>,
    ) -> WithRoi<BitmapToSpanIter<std::iter::Copied<std::slice::Iter<'_, bool>>>> {
        let bounds = Rect::new(
            self.min_x as u32,
            self.min_y as u32,
            NonZero::new((self.max_x - self.min_x + 1) as u32).unwrap(),
            NonZero::new((self.max_y - self.min_y + 1) as u32).unwrap(),
        );
        WithRoi::new(
            BitmapToSpanIter::from_bool_iter(self.mask.iter().copied(), image_width, image_height),
            bounds,
        )
    }

    fn stamp_square(&mut self, cx: usize, cy: usize, half_size: usize) {
        let x_min = cx.saturating_sub(half_size);
        let x_max = (cx + half_size).min(self.width - 1);
        let y_min = cy.saturating_sub(half_size);
        let y_max = (cy + half_size).min(self.height - 1);

        for py in y_min..=y_max {
            let row = py * self.width;
            for px in x_min..=x_max {
                unsafe {
                    *self.mask.get_unchecked_mut(row + px) = true;
                }
            }
        }

        self.min_x = self.min_x.min(x_min);
        self.max_x = self.max_x.max(x_max);
        self.min_y = self.min_y.min(y_min);
        self.max_y = self.max_y.max(y_max);

        let stamp_rect = Rect::new(
            x_min,
            y_min,
            NonZero::new(x_max - x_min + 1).unwrap(),
            NonZero::new(y_max - y_min + 1).unwrap(),
        );
        self.dirty = Some(match self.dirty {
            Some(d) => d.union(&stamp_rect),
            None => stamp_rect,
        });
    }

    fn stamp_line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, half_size: usize) {
        let dx = x1 as f64 - x0 as f64;
        let dy = y1 as f64 - y0 as f64;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1.0 {
            return;
        }
        let step = (half_size as f64).max(1.0);
        let steps = (dist / step).ceil() as usize;

        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = ((x0 as f64 + dx * t).round() as usize).min(self.width - 1);
            let y = ((y0 as f64 + dy * t).round() as usize).min(self.height - 1);
            self.stamp_square(x, y, half_size);
        }
    }

    fn stamp_to(&mut self, x: usize, y: usize, half_size: usize) {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        if let Some((lx, ly)) = self.last_pos {
            self.stamp_line(lx, ly, x, y, half_size);
        } else {
            self.stamp_square(x, y, half_size);
        }
        self.last_pos = Some((x, y));
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let Some(dirty) = self.dirty.take() else {
            return;
        };

        if self.texture.is_none() {
            let pixels = vec![Color32::TRANSPARENT; self.width * self.height];
            let image = ColorImage::new([self.width, self.height], pixels);
            self.texture = Some(ctx.load_texture("brush_preview", image, TextureOptions::NEAREST));
        }

        let handle = self.texture.as_mut().unwrap();
        let dx = dirty.x;
        let dy = dirty.y;
        let dw = dirty.width.get();
        let dh = dirty.height.get();
        let mask = &self.mask;
        let width = self.width;
        let pixels = (dy..dy + dh)
            .flat_map(move |py| {
                (dx..dx + dw).map(move |px| {
                    if mask[py * width + px] {
                        Color32::from_rgba_premultiplied(0, 0, 0, 128)
                    } else {
                        Color32::TRANSPARENT
                    }
                })
            })
            .collect();
        let image = ColorImage::new([dw, dh], pixels);
        handle.set_partial([dx, dy], image, TextureOptions::NEAREST);
    }

    fn render(&self, painter: &mut ImagePainter) {
        if let Some(handle) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::Pos2::new(0.0, 0.0), egui::Pos2::new(1.0, 1.0));
            painter
                .painter()
                .image(handle.id(), painter.image_rect(), uv, egui::Color32::WHITE);
        }
    }
}

fn draw_brush_outline(
    painter: &ImagePainter,
    ix: usize,
    iy: usize,
    half_size: usize,
    width: usize,
    height: usize,
) {
    let x_min = ix.saturating_sub(half_size);
    let x_max = (ix + half_size).min(width - 1);
    let y_min = iy.saturating_sub(half_size);
    let y_max = (iy + half_size).min(height - 1);

    let top_left = painter.image_to_screen(egui::Pos2::new(x_min as f32, y_min as f32));
    let bottom_right =
        painter.image_to_screen(egui::Pos2::new((x_max + 1) as f32, (y_max + 1) as f32));

    painter.draw_dotted_rect(top_left, bottom_right);
}

const DEFAULT_BRUSH_SIZE: NonZeroU16 = NonZeroU16::new(10).unwrap();

#[non_exhaustive]
pub struct BrushTool {
    draw_tool: DrawTool,
    pub brush_size: NonZeroU16,
    stroke: Option<StrokeState>,
}

impl DerefMut for BrushTool {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.draw_tool
    }
}

impl Deref for BrushTool {
    type Target = DrawTool;

    fn deref(&self) -> &Self::Target {
        &self.draw_tool
    }
}

impl Default for BrushTool {
    fn default() -> Self {
        Self {
            brush_size: DEFAULT_BRUSH_SIZE,
            draw_tool: DrawTool::default(),
            stroke: None,
        }
    }
}

impl BrushTool {
    pub fn set_brush_size(&mut self, size: NonZeroU16) -> &mut Self {
        self.brush_size = size;
        self
    }

    pub fn create_factory() -> ToolFactory {
        Arc::new(|_| async { Ok(Box::new(BrushTool::default()) as Box<dyn Tool>) }.boxed_local())
    }

    pub fn create_factory_with(modifier: impl Fn(&mut BrushTool) + 'static) -> ToolFactory {
        Arc::new(move |_| {
            let mut tool = BrushTool::default();
            modifier(&mut tool);
            async { Ok(Box::new(tool) as Box<dyn Tool>) }.boxed_local()
        })
    }
}

impl Tool for BrushTool {
    fn handle_interaction(&mut self, ctx: ToolContext) {
        let image_width = ctx.image.image.original.width();
        let image_height = ctx.image.image.original.height();
        let width = image_width.get() as usize;
        let height = image_height.get() as usize;
        let half_size = self.brush_size.get() as usize;

        let cursor_pos = ctx
            .response
            .interact_pointer_pos()
            .or_else(|| ctx.response.hover_pos())
            .map(|screen_pos| {
                let image_pos = ctx.painter.screen_to_image(screen_pos);
                let x = image_pos.x.round().clamp(0.0, (width - 1) as f32) as usize;
                let y = image_pos.y.round().clamp(0.0, (height - 1) as f32) as usize;
                (x, y)
            });

        if ctx.response.drag_started() {
            self.stroke = cursor_pos.map(|(x, y)| {
                let mut stroke = StrokeState::new(width, height, x, y);
                stroke.stamp_to(x, y, half_size);
                stroke
            });
        }

        if ctx.response.dragged() {
            if let Some(stroke) = &mut self.stroke {
                if let Some((x, y)) = cursor_pos {
                    stroke.stamp_to(x, y, half_size);
                }
            }
        }

        if let Some(stroke) = &mut self.stroke {
            stroke.update_texture(ctx.egui);
            stroke.render(ctx.painter);
        }

        if let Some((x, y)) = cursor_pos {
            draw_brush_outline(ctx.painter, x, y, half_size, width, height);
        }

        if ctx.response.drag_stopped() {
            if !ctx.egui.input(|i| i.modifiers.command || i.modifiers.ctrl) {
                if let Some(stroke) = self.stroke.take() {
                    let spans = stroke.bounded_spans(image_width, image_height);
                    match self.mode {
                        Mode::Insert => {
                            if let Ok(pixel_area) = SortedRanges::try_from_span_iter(spans) {
                                ctx.image
                                    .masks
                                    .on_layer(self.layer)
                                    .keep_overlapping(!matches!(
                                        self.layer,
                                        AffectedLayer::Unspecified
                                    ))
                                    .add(pixel_area);
                            }
                        }
                        Mode::Clear => {
                            ctx.image.masks.on_layer(self.layer).clear(spans);
                        }
                    }
                }
            } else {
                self.stroke = None;
            }
        } else if ctx.response.clicked() {
            if let Some((x, y)) = cursor_pos {
                let mut stroke = StrokeState::new(width, height, x, y);
                stroke.stamp_to(x, y, half_size);
                let spans = stroke.bounded_spans(image_width, image_height);
                match self.mode {
                    Mode::Insert => {
                        if let Ok(pixel_area) = SortedRanges::try_from_span_iter(spans) {
                            ctx.image
                                .masks
                                .on_layer(self.layer)
                                .keep_overlapping(false)
                                .add(pixel_area);
                        }
                    }
                    Mode::Clear => {
                        ctx.image.masks.on_layer(self.layer).clear(spans);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state(w: usize, h: usize) -> StrokeState {
        StrokeState::new(w, h, w / 2, h / 2)
    }

    fn count_mask_true(mask: &[bool]) -> usize {
        mask.iter().filter(|&&b| b).count()
    }

    fn mask_at(stroke: &StrokeState, x: usize, y: usize) -> bool {
        stroke.mask[y * stroke.width + x]
    }

    #[test]
    fn stamp_square_center() {
        let mut s = new_state(20, 20);
        s.stamp_square(10, 10, 3);
        assert!(mask_at(&s, 7, 7));
        assert!(mask_at(&s, 13, 13));
        assert!(!mask_at(&s, 6, 10));
        assert!(!mask_at(&s, 10, 14));
        assert_eq!(count_mask_true(&s.mask), 7 * 7);
    }

    #[test]
    fn stamp_square_clamps_top_left() {
        let mut s = new_state(20, 20);
        s.stamp_square(1, 1, 5);
        assert!(mask_at(&s, 0, 0));
        assert!(mask_at(&s, 6, 6));
        assert!(!mask_at(&s, 7, 1));
        assert_eq!(count_mask_true(&s.mask), 7 * 7);
    }

    #[test]
    fn stamp_square_clamps_bottom_right() {
        let mut s = new_state(20, 20);
        s.stamp_square(18, 18, 5);
        assert!(mask_at(&s, 13, 13));
        assert!(mask_at(&s, 19, 19));
        assert!(!mask_at(&s, 12, 18));
        assert_eq!(count_mask_true(&s.mask), 7 * 7);
    }

    #[test]
    fn stamp_square_half_size_zero_single_pixel() {
        let mut s = new_state(10, 10);
        s.stamp_square(5, 5, 0);
        assert!(mask_at(&s, 5, 5));
        assert_eq!(count_mask_true(&s.mask), 1);
    }

    #[test]
    fn stamp_square_exact_corner() {
        let mut s = new_state(10, 10);
        s.stamp_square(0, 0, 3);
        assert!(mask_at(&s, 0, 0));
        assert!(mask_at(&s, 3, 3));
        assert!(!mask_at(&s, 4, 0));
        assert_eq!(count_mask_true(&s.mask), 4 * 4);
    }

    #[test]
    fn stamp_line_horizontal() {
        let mut s = new_state(20, 20);
        s.stamp_line(5, 10, 15, 10, 2);
        assert!(mask_at(&s, 5, 10));
        assert!(mask_at(&s, 15, 10));
        assert!(mask_at(&s, 10, 10));
    }

    #[test]
    fn stamp_line_vertical() {
        let mut s = new_state(20, 20);
        s.stamp_line(10, 5, 10, 15, 2);
        assert!(mask_at(&s, 10, 5));
        assert!(mask_at(&s, 10, 15));
        assert!(mask_at(&s, 10, 10));
    }

    #[test]
    fn stamp_line_same_position_returns_early() {
        let mut s = new_state(10, 10);
        s.stamp_line(5, 5, 5, 5, 2);
        assert_eq!(count_mask_true(&s.mask), 0);
    }

    #[test]
    fn stamp_line_diagonal_covers_path() {
        let mut s = new_state(20, 20);
        s.stamp_line(0, 0, 10, 10, 1);
        assert!(mask_at(&s, 0, 0));
        assert!(mask_at(&s, 5, 5));
        assert!(mask_at(&s, 10, 10));
    }

    #[test]
    fn stamp_to_first_call_stamps_square() {
        let mut s = new_state(20, 20);
        s.stamp_to(10, 10, 2);
        assert!(mask_at(&s, 8, 8));
        assert!(mask_at(&s, 12, 12));
        assert_eq!(s.last_pos, Some((10, 10)));
    }

    #[test]
    fn stamp_to_second_call_uses_line() {
        let mut s = new_state(20, 20);
        s.stamp_to(5, 10, 2);
        s.stamp_to(15, 10, 2);
        assert!(mask_at(&s, 10, 10));
    }

    #[test]
    fn into_pixel_area_empty_mask_returns_none() {
        let s = new_state(10, 10);
        let result = s.mask.iter().find(|x| **x);
        assert!(result.is_none());
    }

    #[test]
    fn into_pixel_area_single_pixel() {
        let mut s = new_state(10, 10);
        s.stamp_square(5, 5, 0);
        let result = s.mask.iter().find(|x| **x);
        assert!(result.is_some());
    }

    #[test]
    fn into_pixel_area_disconnected_spans() {
        let mut s = new_state(10, 3);
        s.stamp_square(3, 1, 1);
        s.stamp_square(6, 0, 1);
        let result = s.mask.iter().copied().collect::<Vec<_>>();
        #[rustfmt::skip]
        assert_eq!(result, vec![
            false, false, true,  true,  true, true,  true,  true,  false, false,
            false, false, true,  true,  true, true,  true,  true,  false, false,
            false, false, true,  true,  true, false, false, false, false, false,
        ]);
    }

    #[test]
    fn stamp_square_overlapping_idempotent() {
        let mut s = new_state(20, 20);
        s.stamp_square(10, 10, 3);
        let count1 = count_mask_true(&s.mask);
        s.stamp_square(10, 10, 3);
        let count2 = count_mask_true(&s.mask);
        assert_eq!(count1, count2);
    }
}

use std::ops::{Deref, DerefMut};

use imask::{SortedRanges, Span};

use crate::{
    AffectedLayer, CursorImage, DrawTool, MaskActionBuilder, MaskDefaultActions, Mode,
    RectSelection, Tool, ToolContext,
};

const RECT_CURSOR_IMAGE: CursorImage = CursorImage {
    bytes: "iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAABhWlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9bpUWqDnZQcchQneyiooJLqWIRLJS2QqsOJpd+QZOGJMXFUXAtOPixWHVwcdbVwVUQBD9A3AUnRRcp8X9JoUWMB8f9eHfvcfcO8DYqTDG6ooCimnoqHhOyuVXB/4oAhtCHGcyJzNAS6cUMXMfXPTx8vYvwLPdzf45eOW8wwCMQR5mmm8QbxNObpsZ5nzjESqJMfE48rtMFiR+5Ljn8xrlos5dnhvRMap44RCwUO1jqYFbSFeIp4rCsqJTvzTosc97irFRqrHVP/sJgXl1Jc53mCOJYQgJJCJBQQxkVmIjQqpJiIEX7MRf/sO1PkksiVxmMHAuoQoFo+8H/4He3RmFywkkKxoDuF8v6GAX8u0Czblnfx5bVPAF8z8CV2vZXG8DsJ+n1thY+Avq3gYvrtibtAZc7wOCTJuqiLfloegsF4P2MvikHDNwCPWtOb619nD4AGepq+QY4OATGipS97vLuQGdv/55p9fcDUmtzAIjlR5QAAAAGYktHRAAAAAAAAPlDu38AAAAJcEhZcwAADdcAAA3XAUIom3gAAAAHdElNRQfpCBkPAB2IpJjaAAABr0lEQVRIx+3WPUiVYRQH8F8ZDYFGViDUZJlLNCW4REtJRBASZGtTe5CLLg1BH9DY0ORtdmhoKkSyxCLRagiKHHKIiEtGF0HNbi0neLB73/vxXofAAy/n5fA/n895/8/LljQp2+rEdeMkjqMLHSjhK+bwFO/zFHIWUyjjd43nJS422lEHxjCY2N5hBp/wA+04iH4cS2JN4BKKtbrYjTdRZRkFHK3hcwT38DP8PsaIM2U8wEsYaHDU/fgS/pNZO9AXoF841eS59mE14pypBrobgEc5t3ks4hT+GrZvAPSEfpYz0VRydhUTrYfemzNRe+jlaolehT6PnTlIYCje56uBDmAl5juLb/iOJzhRZ6JryUL1ZAFHq3z5azid4bcLdxL89VoV3QzgAs5FJ4/D9qLCmHoxgsUkyf0Kx/KPFAL8ILFdCdt6sMZzvI3Rpl0XA1uXDCWOs8HMaxlkWsZrXMWeRkn1BoaxI7EVg507Y/1LQTfzsTBNywCmk8onWnkR7sMtfN4wnlIQZkvkUIUEH4L+D7eym4cRfBGXsX+z/h8WItGFzQjelrx3Rhe346r+P+UPJi6EyWu6XtcAAAAASUVORK5CYII=",
    offset_x: 10,
    offset_y: 10,
};

#[derive(Default)]
#[non_exhaustive]
pub struct RectTool {
    draw_tool: DrawTool,
    rect_selection: RectSelection,
}

impl Deref for RectTool {
    type Target = DrawTool;

    fn deref(&self) -> &Self::Target {
        &self.draw_tool
    }
}

impl DerefMut for RectTool {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.draw_tool
    }
}

impl Tool for RectTool {
    fn handle_interaction(&mut self, mut ctx: ToolContext) {
        ctx.cursor_image.set(RECT_CURSOR_IMAGE);

        let selection = self.rect_selection.drag_finished(&mut ctx);
        if let Some(rect_ranges) = selection.and_then(|x| Some(x.rect().ok()?.into())) {
            match self.mode {
                Mode::Insert => {
                    ctx.image
                        .masks
                        .on_layer(self.layer)
                        .keep_overlapping(!matches!(self.layer, AffectedLayer::Unspecified))
                        .add(rect_ranges);
                }
                Mode::Clear => {
                    ctx.image.masks.on_layer(self.layer).clear(rect_ranges);
                }
            }
        } else if ctx.response.clicked()
            // The viewer also reacts to clicks beside the image. They are not
            // clicks on the image and must not end up as its nearest edge pixel.
            && let Some((x, y)) = ctx.try_pixel_at()
        {
            let Ok((x, y)) = u16::try_from(x)
                .and_then(|x| Ok((x, u16::try_from(y)?)))
                .inspect_err(|e| log::warn!("Cannot to u16: {e:?}"))
            else {
                return;
            };
            let span = Span::new(x..x + 1, y);
            let ranges = SortedRanges::<u32>::from(span);
            match self.mode {
                Mode::Insert => {
                    ctx.image
                        .masks
                        .on_layer(self.layer)
                        .keep_overlapping(false)
                        .add(ranges);
                }
                Mode::Clear => {
                    ctx.image.masks.on_layer(self.layer).clear(ranges);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroU32,
        sync::{Arc, Mutex},
    };

    use super::*;
    use crate::{
        HistoryStrategy, ImageData, ImageId, ImageLoadOk, ImageState, OriginalImage, PanTool,
        PixelAreaStack, RgbImageInterleaved, State, Tools,
    };

    /// A real [`RectTool`] which also tells where the image is drawn on screen
    struct PlacementSpy {
        rect: RectTool,
        image_rect: Arc<Mutex<Option<egui::Rect>>>,
    }

    impl Tool for PlacementSpy {
        fn handle_interaction(&mut self, ctx: ToolContext) {
            *self.image_rect.lock().unwrap() = Some(ctx.painter.image_rect());
            self.rect.handle_interaction(ctx);
        }
    }

    /// The viewer reacts to clicks next to the image as well (e.g. in the bars
    /// beside a tall image), but such a click is not on the image. The tool
    /// clamps it to the nearest edge pixel and paints that: a click right of the
    /// image leaves a stray pixel in its last column, which later turns a
    /// rectangular mask into "a rectangle and one pixel".
    #[test]
    fn click_next_to_the_image_paints_nothing() {
        let width = NonZeroU32::new(60).expect("60 is not zero");
        let height = NonZeroU32::new(100).expect("100 is not zero");
        let buffer = RgbImageInterleaved::new_arc(vec![[0u8; 3]; 60 * 100].into(), width, height)
            .expect("Buffer matches dimensions");
        let image_rect = Arc::new(Mutex::new(None));
        let mut state = State::new::<crate::gpu::Headless>(Tools::new(
            Box::new(PlacementSpy {
                rect: RectTool::default(),
                image_rect: image_rect.clone(),
            }),
            Box::new(PanTool::default()),
        ));
        state.set_image(ImageData::new(
            ImageId::from("a"),
            ImageLoadOk {
                original: OriginalImage::Rgb8(buffer.clone()),
                adjust: buffer,
            },
            PixelAreaStack::default(),
            HistoryStrategy::Reset,
        ));

        let egui = egui::Context::default();
        let mut frame = |events: Vec<egui::Event>| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                events,
                ..Default::default()
            };
            egui.run_ui(input, |ui| {
                state.ui(ui);
            })
            .drop_without_applying_deltas();
        };
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };

        // Installs the image, the second frame is the first one the tool sees it in
        frame(vec![]);
        frame(vec![]);
        let image_rect = image_rect.lock().unwrap().expect("The tool ran");
        let beside = egui::pos2(image_rect.right() + 40.0, image_rect.center().y);
        assert!(beside.x < 800.0, "Test needs room right of {image_rect:?}");

        frame(vec![egui::Event::PointerMoved(beside)]);
        frame(vec![button(beside, true)]);
        frame(vec![button(beside, false)]);
        frame(vec![]);

        let ImageState::Loaded(loaded) = &state.image_state else {
            panic!("The image is shown");
        };
        let painted = loaded
            .masks
            .subgroups_stack()
            .iter()
            .flat_map(|(layer, area)| {
                area.pixels
                    .spans::<u32>()
                    .map(move |s| (layer, s.x.start..s.x.end, s.y))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            painted,
            [],
            "Clicking at {beside:?}, right of the image at {image_rect:?}, painted (layer, x, y)"
        );
    }
}

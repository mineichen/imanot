use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use futures::FutureExt;
use imask::{Roi, SortedRanges, Span};

use crate::{
    AffectedLayer, CursorImage, DrawTool, MaskActionBuilder, MaskDefaultActions, Mode,
    RectSelection, Tool, ToolContext, ToolFactory,
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

impl RectTool {
    pub fn create_factory() -> ToolFactory {
        Arc::new(|_| async { Ok(Box::new(RectTool::default()) as Box<dyn Tool>) }.boxed_local())
    }

    pub fn create_factory_with(modifier: impl Fn(&mut RectTool) + 'static) -> ToolFactory {
        Arc::new(move |_| {
            let mut tool = RectTool::default();
            modifier(&mut tool);
            async { Ok(Box::new(tool) as Box<dyn Tool>) }.boxed_local()
        })
    }
}

impl Tool for RectTool {
    fn handle_interaction(&mut self, mut ctx: ToolContext) {
        ctx.cursor_image.set(RECT_CURSOR_IMAGE);

        let selection = self.rect_selection.drag_finished(&mut ctx);
        if let Some(rect_result) = selection {
            match self.mode {
                Mode::Insert => {
                    // let color = self.color(&ctx);
                    if let Ok(pixel_area) =
                        SortedRanges::try_from_span_iter(rect_result.rect().into_spans())
                    {
                        ctx.image
                            .masks
                            .on_layer(self.layer)
                            .keep_overlapping(!matches!(self.layer, AffectedLayer::Unspecified))
                            .add(pixel_area);
                    }
                }
                Mode::Clear => {
                    ctx.image
                        .masks
                        .on_layer(self.layer)
                        .clear(rect_result.rect().into_spans());
                }
            }
        } else if ctx.response.clicked()
            && let Some((x, y)) = ctx.cursor_image_pos()
        {
            let (image_width, image_height) = ctx.image.image.adjust.dimensions();

            let coords = u16::try_from(image_width.get() - 1).and_then(|width| {
                let height = u16::try_from(image_height.get() - 1)?;
                let x = u16::try_from(x)?;
                let y = u16::try_from(y)?;

                Ok((x.min(width), y.min(height)))
            });
            let (x, y) = match coords {
                Ok(x) => x,
                Err(e) => {
                    log::warn!("Cannot to u16: {e:?}");
                    return;
                }
            };
            let span = Span::new(x..x + 1, y);
            match self.mode {
                Mode::Insert => {
                    let ranges = SortedRanges::<u32>::from(span);
                    ctx.image
                        .masks
                        .on_layer(self.layer)
                        .keep_overlapping(false)
                        .add(ranges);
                }
                Mode::Clear => {
                    let rect = Roi::from(Span::<u32>::from(span));
                    ctx.image
                        .masks
                        .on_layer(self.layer)
                        .clear(rect.into_spans());
                }
            }
        }
    }
}

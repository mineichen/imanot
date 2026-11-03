use crate::{ImageLoadOk, PanTool, Tool};

/// Core tool management without UI concerns.
///
/// Holds one tool per slot (primary/secondary). Tools are plain values: they
/// are created without an image and keep their state and configuration when
/// the image changes. They are told about every new image through
/// [`Tool::update_image`], whether they are the active one or not.
/// The registry of available tools and their display names lives in the
/// embedding application.
pub struct Tools {
    primary: Box<dyn Tool>,
    secondary: Box<dyn Tool>,
}

pub struct ToolHandle<'a> {
    tool: &'a mut Box<dyn Tool>,
}

impl<'a> ToolHandle<'a> {
    /// Replaces the slot's tool. If an image is already shown, the new tool
    /// receives it through [`Tool::update_image`] right away, otherwise with
    /// the first image that is shown.
    pub fn set(&mut self, tool: Box<dyn Tool>, img: Option<&ImageLoadOk>) {
        log::info!("[tools] tool set, image already shown: {}", img.is_some());
        *self.tool = tool;
        if let Some(img) = img {
            self.tool.update_image(img);
        }
    }

    pub fn tool(&mut self) -> &mut dyn Tool {
        &mut **self.tool
    }
}

impl Default for Tools {
    fn default() -> Self {
        Self::new(Box::new(NopTool), Box::new(PanTool::default()))
    }
}

impl Tools {
    /// Create a new Tools instance with the given primary and secondary tools.
    /// They get their first image through [`Tool::update_image`] once one is shown.
    pub fn new(primary: Box<dyn Tool>, secondary: Box<dyn Tool>) -> Self {
        Self { primary, secondary }
    }

    /// Hands a newly shown image to both tools, the active and the inactive one.
    pub(crate) fn update_image(&mut self, i: &ImageLoadOk) {
        self.primary.update_image(i);
        self.secondary.update_image(i);
    }

    pub fn primary(&mut self) -> ToolHandle<'_> {
        let [p, _s] = self.handles();
        p
    }
    pub fn secondary(&mut self) -> ToolHandle<'_> {
        let [_p, s] = self.handles();
        s
    }
    pub fn handles(&mut self) -> [ToolHandle<'_>; 2] {
        [
            ToolHandle {
                tool: &mut self.primary,
            },
            ToolHandle {
                tool: &mut self.secondary,
            },
        ]
    }
}

/// A no-operation tool used as placeholder
struct NopTool;
impl Tool for NopTool {
    fn handle_interaction(&mut self, _ctx: crate::ToolContext) {
        log::debug!("NopTool was called");
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::*;
    use crate::BrushTool;

    #[test]
    fn boxed_tools_can_still_be_downcast_to_their_type() {
        let boxed: Box<dyn Tool> = Box::new(BrushTool::default());
        let mut tools = Tools::new(boxed, Box::new(PanTool::default()));
        let boxed: Box<dyn Tool> = Box::new(BrushTool::default());
        tools.secondary().set(boxed, None);

        let [mut primary, mut secondary] = tools.handles();
        let any: &mut dyn Any = primary.tool();
        assert!(any.downcast_mut::<BrushTool>().is_some(), "Tools::new");
        let any: &mut dyn Any = secondary.tool();
        assert!(any.downcast_mut::<BrushTool>().is_some(), "ToolHandle::set");
    }
}

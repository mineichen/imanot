use std::num::NonZeroU16;

use std::any::Any;

use imanot::{BrushTool, DragTool, ImageLoadOk, Mode, PanTool, RectTool, ToolFactory};

mod registry;
#[cfg(feature = "sam")]
mod sam;

pub use registry::ToolRegistry;

#[allow(unused_variables)]
pub fn default_tools(config: &crate::config::Config) -> Vec<(String, ToolFactory)> {
    #[cfg(feature = "sam")]
    let session = {
        eprintln!(
            "[DEBUG] Creating SAM session from path: {:?}",
            config.sam_path
        );
        match sam::SamSession::new(&config.sam_path) {
            Ok(s) => {
                eprintln!("[DEBUG] SAM session created successfully");
                s
            }
            Err(e) => {
                eprintln!("[DEBUG] SAM session FAILED: {:?}", e);
                panic!("SAM session error: {:?}", e);
            }
        }
    };
    vec![
        ("Pan".to_string(), PanTool::create_factory()),
        ("Pan2".to_string(), PanTool::create_factory()),
        #[cfg(feature = "sam")]
        ("SAM".to_string(), sam::SamTool::create_factory(session)),
        ("Rect".to_string(), RectTool::create_factory()),
        (
            "PanDrag".to_string(),
            DragTool::create_factory_with(|t| {
                t.set_pan_on_drag(true);
            }),
        ),
        (
            "RectDrag".to_string(),
            DragTool::create_factory_with(|t| {
                t.set_pan_on_drag(false);
            }),
        ),
        ("Brush".to_string(), BrushTool::create_factory()),
        (
            "Brush clear".to_string(),
            BrushTool::create_factory_with(|t| {
                t.set_mode(Mode::Clear);
            }),
        ),
        (
            "Rect clear".to_string(),
            RectTool::create_factory_with(|t| {
                t.set_mode(Mode::Clear);
            }),
        ),
    ]
}

pub(super) fn ui(
    ui: &mut egui::Ui,
    img: &ImageLoadOk,
    registry: &mut ToolRegistry,
    core: &mut imanot::Tools,
) {
    let infos = [
        (registry.primary_idx(), "Primary:"),
        (registry.secondary_idx(), "Secondary (CTRL):"),
    ];
    for (i, (mut tool, (mut active_idx, label))) in
        core.handles().into_iter().zip(infos).enumerate()
    {
        ui.horizontal(|ui| {
            ui.label(label);
            egui::ComboBox::from_id_salt(label)
                .selected_text(registry.name(active_idx))
                .show_ui(ui, |ui| {
                    for (i, name) in registry.names().enumerate() {
                        ui.selectable_value(&mut active_idx, i, name);
                    }
                });

            let changed = match i {
                0 => registry.set_primary_idx(active_idx),
                1 => registry.set_secondary_idx(active_idx),
                _ => unreachable!("Borrowchecker limitation made it impossible to use callback fn(&mut ToolRegistry, usize)"),
            };

            if let Some(factory) = changed {
                tool.set_factory(factory, img);
            }
            if let Some(Ok(tool)) = tool.data() {
                let any: &mut dyn Any = &mut **tool;
                if let Some(brush) = any.downcast_mut::<BrushTool>() {
                    ui.horizontal(|ui| {
                        ui.label("Brush Size:");
                        let mut raw = brush.brush_size.get();
                        if ui
                            .add(egui::Slider::new(&mut raw, 1..=100).step_by(1.0))
                            .changed()
                        {
                            brush.brush_size = NonZeroU16::new(raw).unwrap();
                        }
                    });
                }
            }
        });
    }
}

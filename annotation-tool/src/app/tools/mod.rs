use std::num::NonZeroU16;

use std::any::Any;

use imanot::{BrushTool, DragTool, ImageLoadOk, Mode, PanTool, RectTool};

mod registry;
#[cfg(feature = "sam")]
mod sam;

pub use registry::{ToolFactory, ToolRegistry};

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
        ("Pan".to_string(), creator(PanTool::default)),
        ("Pan2".to_string(), creator(PanTool::default)),
        #[cfg(feature = "sam")]
        (
            "SAM".to_string(),
            creator(move || sam::SamTool::new(session.clone())),
        ),
        ("Rect".to_string(), creator(RectTool::default)),
        (
            "PanDrag".to_string(),
            creator(|| {
                let mut tool = DragTool::default();
                tool.set_pan_on_drag(true);
                tool
            }),
        ),
        (
            "RectDrag".to_string(),
            creator(|| {
                let mut tool = DragTool::default();
                tool.set_pan_on_drag(false);
                tool
            }),
        ),
        ("Brush".to_string(), creator(BrushTool::default)),
        (
            "Brush clear".to_string(),
            creator(|| {
                let mut tool = BrushTool::default();
                tool.set_mode(Mode::Clear);
                tool
            }),
        ),
        (
            "Rect clear".to_string(),
            creator(|| {
                let mut tool = RectTool::default();
                tool.set_mode(Mode::Clear);
                tool
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

            if let Some(new_tool) = changed {
                tool.set(new_tool, Some(img));
            }
            let any: &mut dyn Any = tool.tool();
            if let Some(brush) = any.downcast_mut::<BrushTool>() {
                ui.horizontal(|ui| {
                    ui.label("Brush Size (% of viewport):");
                    const NZ100: NonZeroU16 = NonZeroU16::new(100).unwrap();
                    ui.add(egui::Slider::new(&mut brush.brush_size, NonZeroU16::MIN..=NZ100).step_by(1.0));
                });
            }
        });
    }
}

fn creator<T: imanot::Tool>(f: impl Fn() -> T + 'static) -> ToolFactory {
    std::rc::Rc::new(move || Box::new(f()) as Box<dyn imanot::Tool>)
}

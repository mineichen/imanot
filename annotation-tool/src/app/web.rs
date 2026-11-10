use log::info;

use crate::app::ToolRegistry;
use crate::{ImageCallbackMap, config::Config};
use imanot::CursorImage;

pub fn run_web(mappers: ImageCallbackMap) {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("image_viewer")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        log::info!("About to start eframe");
        let start_result = eframe::WebRunner::new()
            .start(
                canvas.clone(),
                web_options,
                Box::new(|cc| {
                    let mut app = crate::app::ImageViewerApp::new(
                        cc,
                        Box::new(crate::InMemoryStorage::chessboard()),
                        ToolRegistry::from(&Config::default()),
                        super::MaskGenerator::new(mappers),
                    );
                    app.state.cursor_image.enable_web(canvas);
                    Ok(Box::new(app))
                }),
            )
            .await;
        log::info!("ended eframe");

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

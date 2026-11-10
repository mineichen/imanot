use egui::{InnerResponse, Sense};

use crate::{
    CursorImage, CursorImageSystem, HistoryStrategy, ImageData, ImageState, ImageStateLoaded,
    ImageViewer, ImageViewerInteraction, Tools,
    gpu::{GpuBackend, GpuMasks, MaskPainter},
};

/// State container for handling tool interactions with the image viewer.
/// Contains all the necessary components to process tool events and render tools on the image.
///
/// # Replacing the image
///
/// Loaders just call [`State::set_image`]; the `State` decides when the image
/// is actually taken over. There is exactly one place for that: the end of
/// [`State::ui`], once the pointer and the tools have had their say. While the
/// pointer is down on the viewer, or a tool asks for it (see
/// [`ToolContext::postpone_new_images`](crate::ToolContext)), the image is kept
/// back. Only the latest one is retained, so a producer may keep delivering at
/// any rate without memory growing, and nothing of a kept-back image is
/// processed (no texture, masks or tools) until the hold is over.
///
/// A taken-over image is painted by the next frame, which `ui` requests. The
/// textures it replaces are only released after the frame that still
/// references them has been painted.
///
/// # Reacting to the image becoming visible
///
/// Whatever is derived from an incoming image (a threshold layer, statistics
/// drawn next to it) should not be computed while the image is merely pending:
/// most pending images are replaced before they are ever shown. Hand a callback
/// along with the image ([`PendingImage::with_callback`]). It runs when the
/// image becomes the loaded state, before its first paint, and is simply
/// dropped with the image if that never happens.
#[non_exhaustive]
pub struct State {
    pub image_state: crate::ImageState,
    pub viewer: ImageViewer,
    pub tools: Tools,
    pub cursor_image: CursorImageSystem,
    pub config: StateConfig,
    /// Whether the current frame holds the image back: set by the pointer
    /// being down on the viewer, or by a tool while it handles its interaction.
    /// Starts every frame as `false`.
    postpone_new_images: bool,
    /// The latest image handed to `set_image`, waiting for the intake.
    pending_image: PendingImageUpdate,
    /// Draws the masks with the backend chosen in [`State::new`]
    mask_painter: Box<dyn MaskPainter>,
}

/// An image waiting to be shown by a [`State`], optionally with a callback.
/// It only exists until the image is taken over or dropped, the callback is not
/// part of the image data.
pub struct PendingImage {
    data: ImageData,
    callback: Option<ShownCallback>,
}

/// The image a [`State`] waits to show, and whether a hold may keep it back.
#[derive(Default)]
enum PendingImageUpdate {
    #[default]
    None,
    /// Ignores a hold, see [`State::set_image_now`].
    Forced(PendingImage),
    /// Kept back while the pointer or a tool holds the current image, see [`State::set_image`].
    Proposed(PendingImage),
}

type ShownCallback = Box<dyn FnOnce(&mut ImageStateLoaded)>;

impl PendingImage {
    pub fn new(data: ImageData) -> Self {
        Self {
            data,
            callback: None,
        }
    }

    /// Runs `callback` with the loaded state once the image is shown, e.g. to
    /// add a layer computed from [`ImageStateLoaded::image`]. If the image is
    /// replaced or fails to load, the callback is dropped without running.
    /// Replaces a previously set callback.
    #[must_use]
    pub fn with_callback(mut self, callback: impl FnOnce(&mut ImageStateLoaded) + 'static) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }
}

impl From<ImageData> for PendingImage {
    fn from(data: ImageData) -> Self {
        Self::new(data)
    }
}

#[derive(Default)]
pub struct StateConfig {
    /// Reset zoom and pan whenever an image with a different [`ImageId`](crate::ImageId)
    /// (or the first image) is shown. A stream of images with a constant id keeps the viewport.
    pub reset_viewport_on_image_load: bool,
}

impl State {
    /// Create a new State with the given tools, drawing with the renderer
    /// eframe uses (see [`gpu`](crate::gpu)), e.g.
    /// `State::new::<imanot::gpu::Glow>(tools)`.
    pub fn new<B: GpuBackend>(tools: Tools) -> Self {
        Self {
            image_state: crate::ImageState::NotLoaded,
            viewer: ImageViewer::default(),
            tools,
            cursor_image: CursorImageSystem::from(Box::new(|_: Option<&CursorImage>| {
                #[cfg(target_arch = "wasm32")]
                log::warn!(
                    "WebCursors have to be enabled manually with `state.cursor_image.enable_web(canvas), probably in your egui::Webrunner::start() callback`"
                );
            })),
            config: StateConfig::default(),
            postpone_new_images: false,
            pending_image: PendingImageUpdate::None,
            mask_painter: Box::new(GpuMasks::<B>::new()),
        }
    }
    /// Shows `image` as soon as neither the pointer nor a tool holds the
    /// current image (see the type docs). The first image, or one replacing an
    /// error, is shown right away. Each call overrides a not yet shown image.
    ///
    /// This only stores the image; the caller is responsible for requesting a
    /// repaint if no frame is going to run anyway. It's intended use case is
    /// for ImageStreams with fast refresh rates
    pub fn set_image(&mut self, image: impl Into<PendingImage>) {
        self.pending_image = PendingImageUpdate::Proposed(image.into());
    }

    /// Like [`State::set_image`], but not kept back by the pointer or a tool.
    /// For explicit user navigation, where a tool's uncommitted state
    /// (e.g. a selection) must not silently block the switch.
    ///
    /// This is intended for usage with buttons and key bindings
    pub fn set_image_now(&mut self, image: impl Into<PendingImage>) {
        self.pending_image = PendingImageUpdate::Forced(image.into());
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> InnerResponse<Option<ImageViewerInteraction>> {
        let ctx = ui.ctx().clone();
        if let ImageState::Loaded(image) = &mut self.image_state {
            image.masks.handle_events(&ctx);
        }

        let InnerResponse { inner, response } =
            self.viewer
                .ui(ui, self.image_state.sources(), Some(Sense::click()));
        // Right above the image, below whatever the tools draw. Filled once the
        // tools ran, so the masks show this very frame's hover and edits.
        let mask_slot = inner
            .as_ref()
            .map(|r| r.image_painter.painter().add(egui::Shape::Noop));

        // Reset before the tools are called, so that only the pointer or a tool
        // actively setting it to `true` holds the next image back.
        self.postpone_new_images = false;
        let inner = inner.map(|mut r| {
            if let ImageState::Loaded(image) = &mut self.image_state {
                image.masks.update_hover_layer(r.cursor_image_pos);
            }
            self.postpone_new_images = response.is_pointer_button_down_on();
            self.handle_tool_interaction(&response, &ctx, &mut r.image_painter);
            r
        });

        // Before the intake: this frame shows the image the masks belong to
        if let (Some(slot), Some(r), ImageState::Loaded(image)) =
            (mask_slot, &inner, &self.image_state)
        {
            let painter = &r.image_painter;
            self.mask_painter
                .paint(painter.painter(), slot, painter.image_rect(), &image.masks);
        }

        // The one place a new image is taken over. Only now is the hold of this
        // very frame known, and the old textures have been painted already.
        if self.take_over_pending_image(&ctx) {
            // Painted by the next frame, which egui only runs on request
            ctx.request_repaint();
        }

        self.cursor_image
            .apply(inner.as_ref().and_then(|r| r.cursor_image_pos).is_some());

        InnerResponse { inner, response }
    }

    /// Installs the pending image unless it is held back. Returns whether one
    /// was installed.
    ///
    /// Nothing is held while no image is shown: the viewer reports no
    /// interaction then, so neither the pointer nor a tool can set the flag.
    fn take_over_pending_image(&mut self, ctx: &egui::Context) -> bool {
        let pending = match std::mem::take(&mut self.pending_image) {
            PendingImageUpdate::None => return false,
            held @ PendingImageUpdate::Proposed(_) if self.postpone_new_images => {
                self.pending_image = held;
                return false;
            }
            PendingImageUpdate::Proposed(pending) | PendingImageUpdate::Forced(pending) => pending,
        };
        self.install(pending, ctx);
        true
    }

    /// Applies `pending` as the displayed image. Both ways an image gets shown
    /// (first image and replacement) end up here. A failing image drops its
    /// callback unrun.
    fn install(&mut self, pending: PendingImage, ctx: &egui::Context) {
        let PendingImage { data, callback, .. } = pending;
        let is_new_image = !matches!(&self.image_state, ImageState::Loaded(l) if l.id == data.id);
        let result = match &mut self.image_state {
            ImageState::Loaded(existing) => existing.replace(data, ctx),
            state => {
                if data.history_strategy == HistoryStrategy::Keep {
                    log::debug!("HistoryStrategy::Keep without a shown image: no history to keep");
                }
                ImageStateLoaded::from_image_data(data, ctx).map(|loaded| {
                    *state = ImageState::Loaded(loaded);
                })
            }
        };
        if let Err(e) = result {
            self.image_state = ImageState::Error(e.to_string());
        } else if let ImageState::Loaded(loaded) = &mut self.image_state {
            if self.config.reset_viewport_on_image_load && is_new_image {
                self.viewer.reset();
            }
            // The image is painted by the next frame, so this is still before its first paint
            if let Some(callback) = callback {
                callback(loaded);
            }
            // Last, so tools are only told about an image that was shown
            self.tools.update_image(&loaded.image);
        }
    }

    /// Handle tool interaction based on user input.
    /// This method checks which tool should be active (primary or secondary based on modifier keys),
    /// and delegates the interaction handling to the appropriate tool.
    pub fn handle_tool_interaction(
        &mut self,
        response: &egui::Response,
        ctx: &egui::Context,
        tool_painter: &mut crate::ImagePainter,
    ) {
        if let crate::ImageState::Loaded(image) = &mut self.image_state {
            // Check if CTRL is pressed to determine which tool to use
            let use_secondary = ctx.input(|i| i.modifiers.ctrl || i.modifiers.command);
            let mut handle = if use_secondary {
                self.tools.secondary()
            } else {
                self.tools.primary()
            };

            handle.tool().handle_interaction(crate::ToolContext::new(
                image,
                response,
                ctx,
                tool_painter,
                &mut self.viewer,
                &mut self.cursor_image,
                &mut self.postpone_new_images,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroU32,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use imask::SortedRanges;

    use super::*;
    use crate::{
        ImageId, ImageLoadOk, MaskDefaultActions, OriginalImage, PanTool, PixelArea,
        PixelAreaStack, RgbImageInterleaved, Tool, ToolContext,
    };

    /// Mimics a tool with uncommitted state (like DragTool with a selection).
    struct HoldTool {
        hold: Arc<AtomicBool>,
        updates: Arc<AtomicUsize>,
    }

    impl Tool for HoldTool {
        fn handle_interaction(&mut self, ctx: ToolContext) {
            if self.hold.load(Ordering::Relaxed) {
                *ctx.postpone_new_images = true;
            }
        }

        fn update_image(&mut self, _image: &ImageLoadOk) {
            self.updates.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Runs `State::ui` headless, one frame at a time.
    struct Harness {
        ctx: egui::Context,
        state: State,
        /// While `true`, the primary tool holds the image
        tool_hold: Arc<AtomicBool>,
        /// How often the primary tool was told about an image, i.e. how often an image was installed
        tool_loads: Arc<AtomicUsize>,
    }

    impl Harness {
        const CENTER: egui::Pos2 = egui::pos2(400.0, 300.0);

        fn new() -> Self {
            let tool_hold = Arc::new(AtomicBool::new(false));
            let tool_loads = Arc::new(AtomicUsize::new(0));
            let primary = HoldTool {
                hold: tool_hold.clone(),
                updates: tool_loads.clone(),
            };
            Self {
                ctx: egui::Context::default(),
                state: State::new::<crate::gpu::Headless>(Tools::new(
                    Box::new(primary),
                    Box::new(PanTool::default()),
                )),
                tool_hold,
                tool_loads,
            }
        }

        /// Runs a frame and returns whether it asked egui for another one.
        fn frame_with(&mut self, events: Vec<egui::Event>) -> bool {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                events,
                ..Default::default()
            };
            let output = self.ctx.run_ui(input, |ui| {
                self.state.ui(ui);
            });
            let repaint = output
                .viewport_output
                .get(&egui::ViewportId::ROOT)
                .is_some_and(|v| v.repaint_delay.is_zero());
            output.drop_without_applying_deltas();
            repaint
        }

        fn frame(&mut self) -> bool {
            self.frame_with(Vec::new())
        }

        fn pointer(pressed: bool) -> egui::Event {
            egui::Event::PointerButton {
                pos: Self::CENTER,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            }
        }

        /// Shows `id` and makes sure the viewer has a widget the pointer can interact with
        fn show(&mut self, id: &str) {
            self.state.set_image(image(id, 10, HistoryStrategy::Reset));
            self.frame();
            self.frame_with(vec![egui::Event::PointerMoved(Self::CENTER)]);
            assert_eq!(self.shown_id().as_deref(), Some(id));
        }

        fn shown_id(&self) -> Option<String> {
            match &self.state.image_state {
                ImageState::Loaded(loaded) => Some(loaded.id.to_string()),
                _ => None,
            }
        }

        fn installs(&self) -> usize {
            self.tool_loads.load(Ordering::Relaxed)
        }

        fn hold_by_tool(&self, hold: bool) {
            self.tool_hold.store(hold, Ordering::Relaxed);
        }
    }

    fn image(id: &str, side: u32, history_strategy: HistoryStrategy) -> ImageData {
        let width = NonZeroU32::new(side).expect("Side is not zero");
        // A single row is enough, the tests only care about the dimensions
        let buffer = RgbImageInterleaved::new_arc(
            vec![[0u8; 3]; side as usize].into(),
            width,
            NonZeroU32::MIN,
        )
        .expect("Buffer matches dimensions");
        ImageData::new(
            ImageId::from(id),
            ImageLoadOk {
                original: OriginalImage::Rgb8(buffer.clone()),
                adjust: buffer,
            },
            PixelAreaStack::default(),
            history_strategy,
        )
    }

    #[test]
    fn first_image_is_installed_by_the_next_frame_which_asks_for_a_repaint() {
        let mut h = Harness::new();
        h.state.set_image(image("a", 10, HistoryStrategy::Reset));

        let repaint = h.frame();

        assert_eq!(h.shown_id().as_deref(), Some("a"));
        assert_eq!(h.installs(), 1);
        assert!(repaint, "Painting the new image needs another frame");
    }

    #[test]
    fn tool_hold_keeps_the_image_back_and_unprocessed_until_released() {
        let mut h = Harness::new();
        h.show("a");
        assert_eq!(h.installs(), 1);

        h.hold_by_tool(true);
        // Lets egui settle, it asks for frames of its own after the pointer moved
        h.frame();
        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        let repaint_while_held = h.frame();

        assert_eq!(h.shown_id().as_deref(), Some("a"));
        assert_eq!(h.installs(), 1, "A held image must not reach the tools");
        assert!(matches!(
            h.state.pending_image,
            PendingImageUpdate::Proposed(_)
        ));
        assert!(!repaint_while_held, "Nothing to do until the hold ends");

        // The very frame in which the hold ends takes the image over
        h.hold_by_tool(false);
        let repaint_on_release = h.frame();
        assert_eq!(h.shown_id().as_deref(), Some("b"));
        assert_eq!(h.installs(), 2);
        assert!(matches!(h.state.pending_image, PendingImageUpdate::None));
        assert!(repaint_on_release);
    }

    #[test]
    fn only_the_latest_of_many_held_images_is_ever_processed() {
        let mut h = Harness::new();
        h.show("a");
        h.hold_by_tool(true);

        for id in ["b", "c", "d"] {
            h.state.set_image(image(id, 10, HistoryStrategy::Reset));
            h.frame();
        }
        assert_eq!(h.installs(), 1);

        h.hold_by_tool(false);
        h.frame();

        assert_eq!(h.shown_id().as_deref(), Some("d"));
        assert_eq!(h.installs(), 2, "b and c were never installed");
    }

    #[test]
    fn pointer_down_on_the_viewer_holds_the_image_until_release() {
        let mut h = Harness::new();
        h.show("a");

        h.frame_with(vec![Harness::pointer(true)]);
        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        h.frame();
        assert_eq!(h.shown_id().as_deref(), Some("a"));
        assert!(matches!(
            h.state.pending_image,
            PendingImageUpdate::Proposed(_)
        ));

        // Released: the hold is over within this frame, which takes the image over
        let repaint = h.frame_with(vec![Harness::pointer(false)]);
        assert_eq!(h.shown_id().as_deref(), Some("b"));
        assert!(repaint);
    }

    #[test]
    fn a_press_in_the_frame_an_image_is_waiting_still_holds_it() {
        let mut h = Harness::new();
        h.show("a");

        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        h.frame_with(vec![Harness::pointer(true)]);

        assert_eq!(h.shown_id().as_deref(), Some("a"));
        assert_eq!(h.installs(), 1);
    }

    #[test]
    fn press_and_release_within_one_frame_leave_no_hold_behind() {
        let mut h = Harness::new();
        h.show("a");

        h.frame_with(vec![Harness::pointer(true), Harness::pointer(false)]);
        assert!(!h.state.postpone_new_images);
        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        h.frame();

        assert_eq!(h.shown_id().as_deref(), Some("b"));
    }

    #[test]
    fn set_image_now_ignores_the_hold() {
        let mut h = Harness::new();
        h.show("a");
        h.hold_by_tool(true);

        h.state
            .set_image_now(image("b", 10, HistoryStrategy::Reset));
        h.frame();

        assert_eq!(h.shown_id().as_deref(), Some("b"));
        assert_eq!(h.installs(), 2);
    }

    #[test]
    fn an_image_exceeding_the_texture_limit_errors_without_touching_the_tools() {
        let mut h = Harness::new();
        h.show("a");

        h.state
            .set_image(image("wide", 1 << 20, HistoryStrategy::Reset));
        h.frame();

        assert!(matches!(h.state.image_state, ImageState::Error(_)));
        assert_eq!(h.installs(), 1);
    }

    struct CountTool(Arc<AtomicUsize>);

    impl Tool for CountTool {
        fn handle_interaction(&mut self, _ctx: ToolContext) {}

        fn update_image(&mut self, _image: &ImageLoadOk) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn every_new_image_reaches_the_active_and_the_inactive_tool() {
        let (primary, secondary) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let mut h = Harness::new();
        h.state.tools = Tools::new(
            Box::new(CountTool(primary.clone())),
            Box::new(CountTool(secondary.clone())),
        );

        h.show("a");
        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        h.frame();

        assert_eq!(primary.load(Ordering::Relaxed), 2);
        assert_eq!(secondary.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_tool_set_while_an_image_is_shown_gets_it_at_once() {
        let mut h = Harness::new();
        h.show("a");
        let count = Arc::new(AtomicUsize::new(0));
        let ImageState::Loaded(loaded) = &h.state.image_state else {
            panic!("Image a is shown");
        };

        h.state
            .tools
            .primary()
            .set(Box::new(CountTool(count.clone())), Some(&loaded.image));
        assert_eq!(count.load(Ordering::Relaxed), 1);

        // Without an image, it is told about the first one that arrives
        let late = Arc::new(AtomicUsize::new(0));
        h.state
            .tools
            .secondary()
            .set(Box::new(CountTool(late.clone())), None);
        assert_eq!(late.load(Ordering::Relaxed), 0);
        h.state.set_image(image("b", 10, HistoryStrategy::Reset));
        h.frame();
        assert_eq!(late.load(Ordering::Relaxed), 1);
    }

    /// Adds a mask layer in the first frame it runs and always draws an overlay
    struct EditTool {
        edited: bool,
    }

    impl Tool for EditTool {
        fn handle_interaction(&mut self, ctx: ToolContext) {
            if !self.edited {
                self.edited = true;
                ctx.image
                    .masks
                    .add(SortedRanges::from(imask::Span::new(0..5, 0)));
            }
            ctx.painter
                .painter()
                .circle_filled(Harness::CENTER, 3.0, egui::Color32::RED);
        }
    }

    #[test]
    fn masks_are_drawn_between_image_and_tools_including_the_frames_edits() {
        let ctx = egui::Context::default();
        let tool = EditTool { edited: false };
        let mut state = State::new::<crate::gpu::Headless>(Tools::new(
            Box::new(tool),
            Box::new(PanTool::default()),
        ));
        state.set_image(image("a", 10, HistoryStrategy::Reset));
        let mut drawn = || {
            let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                state.ui(ui);
            });
            let drawn = output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::Mesh(_) => Some("image"),
                    egui::Shape::Callback(_) => Some("masks"),
                    egui::Shape::Circle(_) => Some("tool"),
                    _ => None,
                })
                .collect::<Vec<_>>();
            output.drop_without_applying_deltas();
            drawn
        };
        assert_eq!(
            drawn(),
            [] as [&str; 0],
            "Image is installed after the frame"
        );
        assert_eq!(drawn(), ["image", "masks", "tool"]);
    }

    #[test]
    fn keep_retains_masks_only_while_the_dimensions_match() {
        let mut h = Harness::new();
        h.show("a");
        let ImageState::Loaded(loaded) = &mut h.state.image_state else {
            panic!("Image a is shown");
        };
        loaded.masks.reset();
        assert!(loaded.masks.is_dirty());

        h.state.set_image(image("b", 10, HistoryStrategy::Keep));
        h.frame();
        let ImageState::Loaded(loaded) = &h.state.image_state else {
            panic!("Image b is shown");
        };
        assert_eq!(loaded.id.to_string(), "b");
        assert!(loaded.masks.is_dirty(), "Same dimensions keep the history");

        h.state.set_image(image("c", 20, HistoryStrategy::Keep));
        h.frame();
        let ImageState::Loaded(loaded) = &h.state.image_state else {
            panic!("Image c is shown");
        };
        assert_eq!(loaded.id.to_string(), "c");
        assert_eq!(loaded.masks.size(), [20, 1]);
        assert!(
            !loaded.masks.is_dirty(),
            "Masks of another size can't be kept"
        );
    }

    type Log = std::rc::Rc<std::cell::RefCell<Vec<String>>>;

    /// An image whose callback records `<name>:<shown id>`
    fn logging(name: &'static str, id: &str, log: &Log) -> PendingImage {
        let log = log.clone();
        PendingImage::new(image(id, 10, HistoryStrategy::Reset)).with_callback(move |loaded| {
            log.borrow_mut().push(format!("{name}:{}", &*loaded.id));
        })
    }

    #[test]
    fn callback_runs_once_when_the_image_is_installed_and_may_edit_it() {
        let mut h = Harness::new();
        h.show("a");

        let hits = std::rc::Rc::new(std::cell::Cell::new(0));
        let seen = hits.clone();
        h.state.set_image(
            PendingImage::new(image("b", 10, HistoryStrategy::Reset)).with_callback(
                move |loaded| {
                    // It is the shown image, with everything known about it
                    assert_eq!(&*loaded.id, "b");
                    assert_eq!(loaded.masks.size(), [10, 1]);
                    loaded.masks.set_base_layer(
                        1,
                        Some(PixelArea::from_ranges(
                            SortedRanges::try_from_span_iter(
                                imask::Roi::new(0u16..2, 0..1).into_spans(),
                            )
                            .expect("Valid ranges"),
                            [255, 0, 0, 255],
                        )),
                    );
                    seen.set(seen.get() + 1);
                },
            ),
        );
        h.frame();
        assert_eq!(hits.get(), 1, "Done by the frame that installs the image");
        h.frame();
        h.frame();
        assert_eq!(hits.get(), 1);

        let ImageState::Loaded(loaded) = &h.state.image_state else {
            panic!("Image b is shown");
        };
        assert!(loaded.masks.subgroups_stack().get(1).is_some());
    }

    #[test]
    fn only_the_callback_of_the_image_that_gets_shown_runs() {
        let log = Log::default();
        let mut h = Harness::new();
        h.show("a");
        h.hold_by_tool(true);

        h.state.set_image(logging("b", "b", &log));
        h.state.set_image(logging("c", "c", &log));
        h.frame();
        assert!(log.borrow().is_empty(), "Nothing runs while held");

        h.hold_by_tool(false);
        h.frame();
        assert_eq!(*log.borrow(), ["c:c"]);
    }

    #[test]
    fn callback_of_an_image_failing_to_load_does_not_run() {
        let log = Log::default();
        let mut h = Harness::new();
        h.show("a");

        let log2 = log.clone();
        h.state.set_image(
            PendingImage::new(image("wide", 1 << 20, HistoryStrategy::Reset))
                .with_callback(move |_| log2.borrow_mut().push("ran".into())),
        );
        h.frame();

        assert!(matches!(h.state.image_state, ImageState::Error(_)));
        assert!(log.borrow().is_empty());
    }
}

//! Drawing on the GPU, with the renderer eframe uses.
//!
//! egui only knows textured triangles. Whatever goes beyond that, like
//! drawing the masks straight from their ranges without a texture, is a
//! backend specific paint callback. A [`State`](crate::State) is created for
//! one [`GpuBackend`] and only draws with it:
//!
//! ```ignore
//! // eframe's `glow` renderer (the default)
//! let state = State::new::<imanot::gpu::Glow>(tools);
//! ```
//!
//! Each backend is behind the cargo feature of the same name. Several can be
//! enabled at once, the [`State`](crate::State) decides which one it uses.

use std::sync::Arc;

#[cfg(feature = "glow")]
mod glow;
#[cfg_attr(
    not(any(feature = "glow", feature = "wgpu")),
    allow(dead_code, reason = "Only read by the backends")
)]
mod mask;
#[cfg(feature = "wgpu")]
mod wgpu;

#[cfg(feature = "glow")]
pub use self::glow::Glow;
#[cfg(feature = "wgpu")]
pub use self::wgpu::Wgpu;

pub(crate) use mask::{GpuMasks, MaskPainter};

/// A renderer imanot can draw with: [`Glow`] or [`Wgpu`], matching the
/// renderer of eframe. Only implemented within imanot.
pub trait GpuBackend: sealed::Sealed + 'static {
    /// What a [`State`](crate::State) keeps on the GPU to draw its masks:
    /// the buffer holding the ranges and whatever the backend needs to draw
    /// them. Created empty, the GPU objects are created by the first paint.
    type MaskBuffer: Default + Send + Sync + 'static;

    /// The paint callback drawing `frame` with `buffer`.
    fn mask_callback(buffer: Arc<Self::MaskBuffer>, frame: mask::MaskFrame) -> egui::PaintCallback;
}

mod sealed {
    pub trait Sealed {}
}

/// Draws nothing: for tests, which never paint.
#[cfg(test)]
pub(crate) struct Headless;

#[cfg(test)]
impl sealed::Sealed for Headless {}

#[cfg(test)]
impl GpuBackend for Headless {
    type MaskBuffer = ();

    fn mask_callback(_buffer: Arc<()>, frame: mask::MaskFrame) -> egui::PaintCallback {
        egui::PaintCallback {
            rect: frame.rect,
            callback: Arc::new(()),
        }
    }
}

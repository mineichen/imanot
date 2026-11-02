//! Continuous image-space geometry: positions and rectangles in the
//! pixel-grid convention shared by all tools.

pub use fixed::types::U20F12;

/// Unit marker for continuous image coordinates: the image's top-left
/// corner is at `(0, 0)` and pixel `(x, y)` covers the half-open square
/// `[x, x+1) x [y, y+1)` — so the center of the top-left pixel is at
/// `(0.5, 0.5)` and every pixel center sits on `(x + 0.5, y + 0.5)`.
/// Converting between continuous coordinates and pixel indices is a plain
/// `floor`.
///
/// Image coordinates ([`ImgPoint`] / [`ImgRect`]) never live in egui types
/// ([`egui::Pos2`] / [`egui::Rect`] are screen space); the unit marker makes
/// that unrepresentable, and the only conversions happen inside
/// [`ImagePainter`](crate::ImagePainter)'s `image_to_screen` /
/// `screen_to_image`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageSpace;

/// A continuous image-space position (see [`ImageSpace`]).
pub type ImgPoint = euclid::Point2D<ImgScalar, ImageSpace>;

pub type ImgScalar = U20F12;

/// A continuous image-space rectangle (see [`ImageSpace`]).
pub type ImgRect = euclid::Rect<U20F12, ImageSpace>;

/// Index types accepted by [`top_left_from_index`]: the unsigned integers
/// pixel grids are addressed with.
#[allow(private_bounds)]
pub trait PixelIndex: Sealed {
    /// The index as an image scalar.
    fn to_fixed(self) -> U20F12;
}

/// Seals [`PixelIndex`] to the integer types implemented below.
trait Sealed {}

macro_rules! pixel_index {
    ($($t:ty),* $(,)?) => {
        $(
            impl Sealed for $t {}
            impl PixelIndex for $t {
                fn to_fixed(self) -> U20F12 {
                    U20F12::saturating_from_num(self)
                }
            }
        )*
    };
}

pixel_index!(u8, u16, u32, u64, usize);

/// The image coordinate of the pixel-grid corner at index `(x, y)`: the
/// top-left corner of pixel `(x, y)`. The bottom-right corner of pixel `x`
/// is `top_left_from_index(x + 1, ..)`; the center of pixel `(x, y)` is
/// half a pixel down-right of it, at `(x + 0.5, y + 0.5)`.
pub fn top_left_of_pixel<I: PixelIndex>(x: I, y: I) -> ImgPoint {
    ImgPoint::new(x.to_fixed(), y.to_fixed())
}

/// Lossy `f32` → image scalar: non-finite values become zero, out-of-range
/// values saturate. Cursor-derived positions can lie far outside the image
/// (or degenerate to NaN at degenerate viewport sizes), where no exact grid
/// value would be meaningful.
pub(crate) fn scalar_from_f32(value: f32) -> U20F12 {
    if value.is_finite() {
        U20F12::saturating_from_num(value)
    } else {
        U20F12::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: i64, y: i64) -> ImgPoint {
        ImgPoint::new(U20F12::from_num(x), U20F12::from_num(y))
    }

    #[test]
    fn top_left_from_index_names_pixel_corners() {
        // Index (x, y) names the top-left corner of pixel (x, y): the image
        // corner is the origin, no offset.
        assert_eq!(top_left_of_pixel(2usize, 3usize), pt(2, 3));
        assert_eq!(top_left_of_pixel(2u8, 3u8), pt(2, 3));
        assert_eq!(top_left_of_pixel(2u16, 3u16), pt(2, 3));
        assert_eq!(top_left_of_pixel(2u32, 3u32), pt(2, 3));
        assert_eq!(top_left_of_pixel(2u64, 3u64), pt(2, 3));
        // Pixel (0, 0)'s corner is the image origin; its center is at
        // (0.5, 0.5) — exactly representable in `U20F12`.
        assert_eq!(top_left_of_pixel(0u32, 0u32), pt(0, 0));
        assert_eq!(
            pt(0, 0) + euclid::vec2(U20F12::from_num(0.5), U20F12::from_num(0.5)),
            ImgPoint::new(U20F12::from_num(0.5), U20F12::from_num(0.5))
        );
        // The bottom-right corner of pixel x is the top-left corner of x+1.
        assert_eq!(top_left_of_pixel(4u16, 7u16), pt(4, 7));
    }

    #[test]
    fn scalar_from_f32_saturates_and_zeroes() {
        assert_eq!(scalar_from_f32(2.5), U20F12::from_num(2.5));
        // Unsigned scalar: negative saturates to zero.
        assert_eq!(scalar_from_f32(-3.0), U20F12::ZERO);
        assert_eq!(scalar_from_f32(f32::NEG_INFINITY), U20F12::ZERO);
        // Out of range saturates instead of panicking.
        assert_eq!(scalar_from_f32(1e9), U20F12::MAX);
        // Non-finite is meaningless: zero.
        assert_eq!(scalar_from_f32(f32::NAN), U20F12::ZERO);
    }
}

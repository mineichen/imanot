use imask::Rect;
use nalgebra::{Matrix3, Point2, Rotation2, Translation2, Vector2};

/// Rotation snaps to multiples of this (absolute angle, not delta), unless
/// Shift is held for the exact value.
pub(crate) const ROTATE_SNAP_DEG: f64 = 1.0;

/// Resize anchors of the bounding box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Anchor {
    Nw,
    N,
    Ne,
    E,
    Se,
    S,
    Sw,
    W,
}

impl Anchor {
    pub(crate) const ALL: [Self; 8] = [
        Self::Nw,
        Self::N,
        Self::Ne,
        Self::E,
        Self::Se,
        Self::S,
        Self::Sw,
        Self::W,
    ];

    /// Side signs in frame space as a vector, each component in `{-1, 0, 1}`.
    pub(crate) fn sides(self) -> Vector2<f64> {
        match self {
            Self::Nw => Vector2::new(-1.0, -1.0),
            Self::N => Vector2::new(0.0, -1.0),
            Self::Ne => Vector2::new(1.0, -1.0),
            Self::E => Vector2::new(1.0, 0.0),
            Self::Se => Vector2::new(1.0, 1.0),
            Self::S => Vector2::new(0.0, 1.0),
            Self::Sw => Vector2::new(-1.0, 1.0),
            Self::W => Vector2::new(-1.0, 0.0),
        }
    }

    /// Which axes this anchor scales: `(scale_u, scale_v)`.
    pub(crate) fn axes(self) -> (bool, bool) {
        match self {
            Self::Nw | Self::Ne | Self::Se | Self::Sw => (true, true),
            Self::E | Self::W => (true, false),
            Self::N | Self::S => (false, true),
        }
    }
}

/// Oriented selection frame in image coordinates: center, half-sizes and
/// rotation angle. Half-sizes are signed: dragging an anchor past the opposite
/// edge flips the sign, which mirrors the content (negative scale in the
/// rasterization matrix). The overlay draws the same box for `±half`, with the
/// dragged anchor staying under the cursor and the pivot fixed.
/// This is the explicitly tracked selection geometry — it is
/// never derived from `ImageDimension` bounds, so rotated shapes keep their
/// exact size and orientation instead of inflating through nested AABBs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub(crate) center: Point2<f64>,
    pub(crate) half: Vector2<f64>,
    pub(crate) angle: f64,
}

impl Frame {
    /// Unrotated frame tightly around integer content `bounds`. Snapshots
    /// always carry tight bounds, so this is O(1) with no span iteration.
    pub(crate) fn around(bounds: Rect<u32>) -> Self {
        let x0 = f64::from(bounds.x);
        let y0 = f64::from(bounds.y);
        let x1 = f64::from(bounds.x + bounds.width.get());
        let y1 = f64::from(bounds.y + bounds.height.get());
        Self {
            center: Point2::new((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            half: Vector2::new((x1 - x0) / 2.0, (y1 - y0) / 2.0),
            angle: 0.0,
        }
    }

    /// Rotation of the frame (identity at angle 0).
    pub(crate) fn rotation(&self) -> Rotation2<f64> {
        Rotation2::new(self.angle)
    }

    /// Orthonormal frame axes (u right, v down at angle 0) in image coords.
    pub(crate) fn axes(&self) -> (Vector2<f64>, Vector2<f64>) {
        let (sin, cos) = self.angle.sin_cos();
        (Vector2::new(cos, sin), Vector2::new(-sin, cos))
    }

    /// Frame-space offset in image coordinates.
    pub(crate) fn point(&self, local: Vector2<f64>) -> Point2<f64> {
        self.center + self.rotation() * local
    }

    /// Grow the frame (keeping angle) to cover integer content `bounds`,
    /// using their tight extent.
    pub(crate) fn expand_to_cover(&mut self, bounds: Rect<u32>) {
        self.expand_to_include(
            Point2::new(f64::from(bounds.x), f64::from(bounds.y)),
            Point2::new(
                f64::from(bounds.x + bounds.width.get()),
                f64::from(bounds.y + bounds.height.get()),
            ),
        );
    }

    /// Grow the frame (keeping angle) to include an image-space box. The
    /// frame stays the minimal same-angle box containing both the old frame
    /// and the new box: the center shifts so opposite sides don't grow
    /// unnecessarily.
    pub(crate) fn expand_to_include(&mut self, min: Point2<f64>, max: Point2<f64>) {
        let (u, v) = self.axes();
        // Signed halves (mirrored frames) cover the same `±half` extent.
        let ah = Vector2::new(self.half.x.abs(), self.half.y.abs());
        let mut min_u = -ah.x;
        let mut max_u = ah.x;
        let mut min_v = -ah.y;
        let mut max_v = ah.y;
        for corner in [
            min,
            Point2::new(max.x, min.y),
            max,
            Point2::new(min.x, max.y),
        ] {
            let d: Vector2<f64> = corner - self.center;
            let lu = d.dot(&u);
            let lv = d.dot(&v);
            min_u = min_u.min(lu);
            max_u = max_u.max(lu);
            min_v = min_v.min(lv);
            max_v = max_v.max(lv);
        }
        let off = u * ((min_u + max_u) / 2.0) + v * ((min_v + max_v) / 2.0);
        self.center += off;
        self.half = Vector2::new((max_u - min_u) / 2.0, (max_v - min_v) / 2.0);
    }
}

pub(crate) fn rotate_about(center: Point2<f64>, angle: f64) -> Matrix3<f64> {
    Translation2::new(center.x, center.y).to_homogeneous()
        * Rotation2::new(angle).to_homogeneous()
        * Translation2::new(-center.x, -center.y).to_homogeneous()
}

/// Scale about a world-space pivot, along axes rotated by `angle`.
pub(crate) fn scale_about_frame(
    pivot: Point2<f64>,
    angle: f64,
    scale: Vector2<f64>,
) -> Matrix3<f64> {
    Translation2::new(pivot.x, pivot.y).to_homogeneous()
        * Rotation2::new(angle).to_homogeneous()
        * Matrix3::new(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, 1.0)
        * Rotation2::new(-angle).to_homogeneous()
        * Translation2::new(-pivot.x, -pivot.y).to_homogeneous()
}

/// Clamp a signed half-size to ≥1px magnitude, preserving a mirror flip.
/// Dragging an anchor onto (or past) the opposite edge must never leave a
/// degenerate zero frame behind.
pub(crate) fn clamp_half(raw: f64) -> f64 {
    if raw >= 0.0 {
        raw.max(0.5)
    } else {
        raw.min(-0.5)
    }
}

/// Snap an absolute angle to `ROTATE_SNAP_DEG` increments.
pub(crate) fn snap_angle(angle: f64) -> f64 {
    ((angle.to_degrees() / ROTATE_SNAP_DEG).round() * ROTATE_SNAP_DEG).to_radians()
}

/// Union of content bounds. `None` when empty.
pub(crate) fn union_bounds(bounds: impl IntoIterator<Item = Rect<u32>>) -> Option<Rect<u32>> {
    bounds.into_iter().reduce(|a, b| a.union(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[test]
    fn frame_point_roundtrip() {
        let frame = Frame {
            center: Point2::new(5.0, 5.0),
            half: Vector2::new(2.0, 3.0),
            angle: 0.0,
        };
        assert_eq!(frame.point(Vector2::new(2.0, -3.0)), Point2::new(7.0, 2.0));
        assert_eq!(frame.point(Vector2::new(2.0, 3.0)), Point2::new(7.0, 8.0));
    }

    #[test]
    fn snap_angle_rounds_to_degree_steps() {
        assert_eq!(snap_angle(0.4f64.to_radians()), 0.0);
        assert_eq!(snap_angle(0.6f64.to_radians()), 1.0f64.to_radians());
        assert_eq!(snap_angle(1.0f64.to_radians()), 1.0f64.to_radians());
        assert_eq!(snap_angle((-2.4f64).to_radians()), (-2.0f64).to_radians());
    }

    #[test]
    fn anchor_geometry() {
        // (anchor, frame-space sides, scaled axes).
        let cases = [
            (Anchor::Nw, Vector2::new(-1.0, -1.0), (true, true)),
            (Anchor::N, Vector2::new(0.0, -1.0), (false, true)),
            (Anchor::Ne, Vector2::new(1.0, -1.0), (true, true)),
            (Anchor::E, Vector2::new(1.0, 0.0), (true, false)),
            (Anchor::Se, Vector2::new(1.0, 1.0), (true, true)),
            (Anchor::S, Vector2::new(0.0, 1.0), (false, true)),
            (Anchor::Sw, Vector2::new(-1.0, 1.0), (true, true)),
            (Anchor::W, Vector2::new(-1.0, 0.0), (true, false)),
        ];
        for (anchor, sides, axes) in cases {
            assert_eq!(anchor.sides(), sides, "{anchor:?}");
            assert_eq!(anchor.axes(), axes, "{anchor:?}");
        }
    }

    #[test]
    fn expand_covers_mirrored_frame() {
        let mut frame = Frame {
            center: Point2::new(7.5, 12.5),
            half: Vector2::new(-2.5, 2.5),
            angle: 0.0,
        };
        frame.expand_to_include(Point2::new(10.0, 10.0), Point2::new(15.0, 15.0));
        // Old mirrored extent (5..10) plus new box (10..15) → 5..15.
        assert!((frame.center.x - 10.0).abs() < 1e-9, "{frame:?}");
        assert!((frame.half.x - 5.0).abs() < 1e-9, "{frame:?}");
    }

    #[test]
    fn expand_to_cover_wraps_content_bounds() {
        // Frame hugs the contained content, not the queried box: the frame
        // shrinks to the union of its extent and the content bounds.
        let mut frame = Frame::around(Rect::new(
            5,
            5,
            NonZeroU32::new(4).unwrap(),
            NonZeroU32::new(2).unwrap(),
        ));
        frame.expand_to_cover(Rect::new(
            0,
            0,
            NonZeroU32::new(2).unwrap(),
            NonZeroU32::new(2).unwrap(),
        ));
        // Old extent (5..9)x(5..7) plus content (0..2)x(0..2) → (0..9)x(0..7).
        assert_eq!(frame.center, Point2::new(4.5, 3.5));
        assert_eq!(frame.half, Vector2::new(4.5, 3.5));
    }
}

use nalgebra::{Matrix3, Point2, Vector2};

use crate::RectSelection;

use super::frame::{Anchor, Frame, clamp_half, rotate_about, scale_about_frame, snap_angle};

/// Move gesture state: press origin plus the frame/matrix it started from.
#[derive(Clone, Copy)]
pub(crate) struct GestureMove {
    pub(crate) start: Point2<f64>,
    pub(crate) base: Frame,
    pub(crate) base_total: Matrix3<f64>,
}

impl GestureMove {
    /// Snap the drag delta to whole pixels so the rasterization stays
    /// pixel-exact. A fractional translation makes the span rasterizer emit
    /// fringe rows outside its analytic bounds; those fringe spans would
    /// enter `committed` and the *next* commit's Clear would erase outsider
    /// pixels sharing the rows. Snapping the gesture-total delta (not
    /// per-frame increments) keeps frame and matrix in sync.
    pub(crate) fn apply(&self, pointer: Point2<f64>) -> (Frame, Matrix3<f64>) {
        let delta = pointer - self.start;
        let snapped = Vector2::new(delta.x.round(), delta.y.round());
        let frame = Frame {
            center: self.base.center + snapped,
            ..self.base
        };
        let total = Matrix3::new_translation(&snapped) * self.base_total;
        (frame, total)
    }

    pub(crate) fn base_state(&self) -> (Frame, Matrix3<f64>) {
        (self.base, self.base_total)
    }
}

/// Resize gesture state: the dragged anchor, press origin and the
/// frame/matrix it started from.
#[derive(Clone, Copy)]
pub(crate) struct GestureResize {
    pub(crate) anchor: Anchor,
    pub(crate) start: Point2<f64>,
    pub(crate) base: Frame,
    pub(crate) base_total: Matrix3<f64>,
}

impl GestureResize {
    /// Resize along the frame's own axes, so rotated boxes resize correctly.
    /// `shift` (Shift held) frees two-axis (corner) anchors from the
    /// aspect-ratio lock.
    pub(crate) fn apply(&self, pointer: Point2<f64>, shift: bool) -> (Frame, Matrix3<f64>) {
        let base = self.base;
        let (u, v) = base.axes();
        let delta: Vector2<f64> = pointer - self.start;
        let d = Vector2::new(delta.dot(&u), delta.dot(&v));
        let s = self.anchor.sides();
        let (scale_u, scale_v) = self.anchor.axes();
        // Negative direction allowed: dragging past the opposite edge flips
        // the sign, mirroring the shape about the pivot. Keep ≥1px magnitude
        // so a resting frame never divides by zero on the next gesture.
        let mut half = base.half;
        if scale_u {
            half.x = clamp_half(base.half.x + s.x * d.x / 2.0);
        }
        if scale_v {
            half.y = clamp_half(base.half.y + s.y * d.y / 2.0);
        }
        if !shift && scale_u && scale_v {
            half = locked_halves(base.half, s, d);
        }
        let mut center = base.center;
        if scale_u {
            center += u * (s.x * (half.x - base.half.x));
        }
        if scale_v {
            center += v * (s.y * (half.y - base.half.y));
        }
        let scale = Vector2::new(half.x / base.half.x, half.y / base.half.y);
        let frame = Frame {
            center,
            half,
            angle: base.angle,
        };
        let pivot = base.point(Vector2::new(-s.x * base.half.x, -s.y * base.half.y));
        let total = scale_about_frame(pivot, base.angle, scale) * self.base_total;
        (frame, total)
    }

    pub(crate) fn base_state(&self) -> (Frame, Matrix3<f64>) {
        (self.base, self.base_total)
    }
}

/// Rotate gesture state: the pointer angle at press plus the frame/matrix it
/// started from.
#[derive(Clone, Copy)]
pub(crate) struct GestureRotate {
    pub(crate) start_angle: f64,
    pub(crate) base: Frame,
    pub(crate) base_total: Matrix3<f64>,
}

impl GestureRotate {
    /// Rotate about the frame center. Unless `shift` is held, the absolute
    /// angle snaps to whole degrees.
    pub(crate) fn apply(&self, pointer: Point2<f64>, shift: bool) -> (Frame, Matrix3<f64>) {
        let base = self.base;
        let delta = (pointer.y - base.center.y).atan2(pointer.x - base.center.x) - self.start_angle;
        let angle = base.angle + delta;
        let frame = Frame {
            center: base.center,
            half: base.half,
            angle: if shift { angle } else { snap_angle(angle) },
        };
        let total = rotate_about(base.center, frame.angle - base.angle) * self.base_total;
        (frame, total)
    }

    pub(crate) fn base_state(&self) -> (Frame, Matrix3<f64>) {
        (self.base, self.base_total)
    }
}

/// Corner anchors preserve the aspect ratio with a single magnitude, picked
/// so the cursor stays on a visible edge segment: the larger absolute
/// per-axis factor wins, with each axis keeping its own sign (the side of
/// the pivot the pointer is on). Either factor alone would leave the cursor
/// floating off the box on the axis extension whenever the pointer leaves
/// that axis' edge — shrinking with the dominant (most-moved) factor even
/// lands outside the segment. The max-magnitude choice puts the pointer on
/// the moving u edge (or v edge, or the corner when equal); independent
/// signs keep it there even when the pointer crosses the pivot in one axis
/// only, by mirroring just that axis. Shift frees the axes entirely.
/// Single-axis (edge) anchors are unaffected.
fn locked_halves(base_half: Vector2<f64>, s: Vector2<f64>, d: Vector2<f64>) -> Vector2<f64> {
    let s_u = (base_half.x + s.x * d.x / 2.0) / base_half.x;
    let s_v = (base_half.y + s.y * d.y / 2.0) / base_half.y;
    let mut mag = s_u.abs().max(s_v.abs());
    // Keep ≥1px magnitude on both axes without breaking the aspect ratio
    // (per-axis clamping would square tiny boxes).
    let min_mag = (0.5 / base_half.x.abs()).max(0.5 / base_half.y.abs());
    mag = mag.max(min_mag);
    let sign_u = s_u.signum();
    let sign_v = s_v.signum();
    Vector2::new(base_half.x * mag * sign_u, base_half.y * mag * sign_v)
}

/// In-progress pointer gesture. Pointer positions are image coordinates.
/// Transform gestures carry both the frame and the matrix they started from,
/// so cancelling restores both consistently. The per-variant state lives in
/// `GestureMove`/`GestureResize`/`GestureRotate`; this enum only tags which
/// gesture is active and forwards to its methods.

pub(crate) enum Gesture {
    Move(GestureMove),
    Resize(GestureResize),
    Rotate(GestureRotate),
    /// Rect-select in progress (state lives in `RectSelection`).
    Rect(RectSelection),
    /// Pan in progress (delegated to `PanTool`).
    Pan,
}

impl Gesture {
    /// Recompute `(frame, total)` for an active transform gesture from the
    /// current pointer (image coordinates). Pure: both state halves derive
    /// from the same gesture parameters, so overlay and rasterization can
    /// never drift apart. Returns `None` for non-transform gestures (`Rect`,
    /// `Pan`).
    pub(crate) fn apply(&self, pointer: Point2<f64>, shift: bool) -> Option<(Frame, Matrix3<f64>)> {
        match self {
            Self::Move(g) => Some(g.apply(pointer)),
            Self::Resize(g) => Some(g.apply(pointer, shift)),
            Self::Rotate(g) => Some(g.apply(pointer, shift)),
            Self::Rect(_) | Self::Pan => None,
        }
    }

    /// Frame and matrix to restore if a transform gesture is cancelled.
    pub(crate) fn base_state(self) -> Option<(Frame, Matrix3<f64>)> {
        match self {
            Self::Move(g) => Some(g.base_state()),
            Self::Resize(g) => Some(g.base_state()),
            Self::Rotate(g) => Some(g.base_state()),
            Self::Rect(_) | Self::Pan => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn square() -> Frame {
        Frame {
            center: Point2::new(12.5, 12.5),
            half: Vector2::new(2.5, 2.5),
            angle: 0.0,
        }
    }

    fn wide() -> Frame {
        Frame {
            center: Point2::new(15.0, 12.5),
            half: Vector2::new(5.0, 2.5),
            angle: 0.0,
        }
    }

    fn pt(x: f64, y: f64) -> Point2<f64> {
        Point2::new(x, y)
    }

    #[test]
    fn move_snaps_to_whole_pixels() {
        // App drag deltas are fractional, but masks live on whole pixels.
        let gesture = GestureMove {
            start: pt(15.0, 12.5),
            base: square(),
            base_total: Matrix3::identity(),
        };
        let (frame, total) = gesture.apply(pt(44.5, 32.5));
        assert_eq!(total, Matrix3::new_translation(&Vector2::new(30.0, 20.0)));
        assert_eq!(frame.center, pt(42.5, 32.5));
    }

    #[test]
    fn resize_mirror_and_minimum_magnitude() {
        // Drag the E anchor (15, 12.5) west past the W edge (10) to (5, 12.5):
        // half flips sign, pivot stays, dragged anchor follows the cursor.
        let gesture = GestureResize {
            anchor: Anchor::E,
            start: pt(15.0, 12.5),
            base: square(),
            base_total: Matrix3::identity(),
        };
        let (frame, total) = gesture.apply(pt(5.0, 12.5), false);
        assert_eq!(frame.half.x, -2.5);
        assert_eq!(frame.center.x, 7.5);
        // Pivot (W edge) fixed by the total matrix, dragged edge under cursor.
        let p = total * Vector3::new(10.0, 12.5, 1.0);
        assert!((p.x - 10.0).abs() < 1e-9, "{p:?}");
        assert!((p.y - 12.5).abs() < 1e-9, "{p:?}");
        let q = total * Vector3::new(15.0, 12.5, 1.0);
        assert!((q.x - 5.0).abs() < 1e-9, "{q:?}");
        assert!((q.y - 12.5).abs() < 1e-9, "{q:?}");

        // Drag E exactly onto the W edge: raw half 0 snaps to +0.5 (1px);
        // a touch further flips to -0.5, never a degenerate zero frame.
        for (to, half) in [(10.0, 0.5), (9.0, -0.5)] {
            let gesture = GestureResize {
                anchor: Anchor::E,
                start: pt(15.0, 12.5),
                base: square(),
                base_total: Matrix3::identity(),
            };
            let (frame, _) = gesture.apply(pt(to, 12.5), false);
            assert_eq!(frame.half.x, half);
        }
    }

    #[test]
    fn corner_resize_locks_aspect_ratio() {
        // 10x5 content (center (15, 12.5), half (5, 2.5)): dragging the SE
        // corner scales both axes by one factor, picked so the cursor stays
        // on a visible edge segment.
        for (to, half, center) in [
            ((25.0, 15.0), (7.5, 3.75), (17.5, 13.75)),
            ((22.5, 20.0), (10.0, 5.0), (20.0, 15.0)),
            ((24.0, 17.5), (7.5, 3.75), (17.5, 13.75)),
        ] {
            let gesture = GestureResize {
                anchor: Anchor::Se,
                start: pt(20.0, 15.0),
                base: wide(),
                base_total: Matrix3::identity(),
            };
            let (frame, _) = gesture.apply(pt(to.0, to.1), false);
            assert_eq!(frame.half, Vector2::new(half.0, half.1));
            assert_eq!(frame.center, pt(center.0, center.1));
        }
        // Pivot (NW corner) fixed by the total matrix.
        let gesture = GestureResize {
            anchor: Anchor::Se,
            start: pt(20.0, 15.0),
            base: wide(),
            base_total: Matrix3::identity(),
        };
        let (_, total) = gesture.apply(pt(25.0, 15.0), false);
        let p = total * Vector3::new(10.0, 10.0, 1.0);
        assert!((p.x - 10.0).abs() < 1e-9, "{p:?}");
        assert!((p.y - 10.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn shift_corner_resize_destroys_aspect_ratio() {
        // Holding Shift frees the corner anchors: each axis follows the
        // pointer independently, like edge anchors always do.
        let gesture = GestureResize {
            anchor: Anchor::Se,
            start: pt(20.0, 15.0),
            base: wide(),
            base_total: Matrix3::identity(),
        };
        let (frame, _) = gesture.apply(pt(25.0, 15.0), true);
        assert_eq!(frame.half, Vector2::new(7.5, 2.5));
        assert_eq!(frame.center, pt(17.5, 12.5));
    }
}

use egui::CursorIcon;
use nalgebra::{Matrix3, Point2, Vector2};

use crate::RectSelection;

use super::active_selection::{HoverPart, resize_cursor};
use super::frame::{Anchor, Frame, clamp_half, rotate_about, scale_about_frame, snap_angle};

/// In-progress pointer gesture. Pointer positions are image coordinates.
pub(super) enum Gesture {
    /// Rect-select in progress (state lives in `RectSelection`).
    Rect(RectSelection),
    /// Pan in progress (delegated to `PanTool`).
    Pan,
    /// Move/resize/rotate of the active selection.
    Transform(TransformGesture),
}

/// Transform gesture: what is being dragged plus the frame and matrix it
/// started from, so cancelling restores both consistently.
pub(super) struct TransformGesture {
    kind: TransformKind,
    base: Frame,
    base_total: Matrix3<f64>,
}

enum TransformKind {
    Move {
        start: Point2<f64>,
    },
    Resize {
        anchor: Anchor,
        start: Point2<f64>,
    },
    /// Pointer angle at press.
    Rotate {
        start_angle: f64,
    },
}

impl TransformGesture {
    /// Start the gesture matching the hovered frame part at the press
    /// position; `None` when the press is outside the frame.
    pub(super) fn begin(
        part: HoverPart,
        press: Point2<f64>,
        (base, base_total): (Frame, Matrix3<f64>),
    ) -> Option<Self> {
        let kind = match part {
            HoverPart::Outside => return None,
            HoverPart::Inside => TransformKind::Move { start: press },
            HoverPart::Anchor(anchor) => TransformKind::Resize {
                anchor,
                start: press,
            },
            HoverPart::Rotate => TransformKind::Rotate {
                start_angle: (press.y - base.center.y).atan2(press.x - base.center.x),
            },
        };
        Some(Self {
            kind,
            base,
            base_total,
        })
    }

    /// Recompute `(frame, total)` from the current pointer (image
    /// coordinates). Pure: both state halves derive from the same gesture
    /// parameters, so overlay and rasterization can never drift apart.
    /// `shift` gives exact unsnapped rotation angles, or a corner resize
    /// free of the aspect-ratio lock.
    pub(super) fn apply(&self, pointer: Point2<f64>, shift: bool) -> (Frame, Matrix3<f64>) {
        match self.kind {
            TransformKind::Move { start } => self.apply_move(start, pointer),
            TransformKind::Resize { anchor, start } => {
                self.apply_resize(anchor, start, pointer, shift)
            }
            TransformKind::Rotate { start_angle } => self.apply_rotate(start_angle, pointer, shift),
        }
    }

    /// Frame and matrix to restore if the gesture is cancelled.
    pub(super) fn base_state(&self) -> (Frame, Matrix3<f64>) {
        (self.base, self.base_total)
    }

    /// Pure moves only shift the already rasterized pixels.
    pub(super) fn is_move(&self) -> bool {
        matches!(self.kind, TransformKind::Move { .. })
    }

    /// Cursor while the gesture runs on the selection's current `frame`.
    pub(super) fn cursor(&self, frame: &Frame) -> CursorIcon {
        match self.kind {
            TransformKind::Move { .. } | TransformKind::Rotate { .. } => CursorIcon::Grabbing,
            TransformKind::Resize { anchor, .. } => resize_cursor(frame, anchor),
        }
    }

    /// Snap the drag delta to whole pixels so the rasterization stays
    /// pixel-exact. A fractional translation makes the span rasterizer emit
    /// fringe rows outside its analytic bounds; those fringe spans would
    /// enter `committed` and the *next* commit's Clear would erase outsider
    /// pixels sharing the rows. Snapping the gesture-total delta (not
    /// per-frame increments) keeps frame and matrix in sync.
    fn apply_move(&self, start: Point2<f64>, pointer: Point2<f64>) -> (Frame, Matrix3<f64>) {
        let delta = pointer - start;
        let snapped = Vector2::new(delta.x.round(), delta.y.round());
        let frame = Frame {
            center: self.base.center + snapped,
            ..self.base
        };
        let total = Matrix3::new_translation(&snapped) * self.base_total;
        (frame, total)
    }

    /// Resize along the frame's own axes, so rotated boxes resize correctly.
    /// `shift` (Shift held) frees two-axis (corner) anchors from the
    /// aspect-ratio lock.
    fn apply_resize(
        &self,
        anchor: Anchor,
        start: Point2<f64>,
        pointer: Point2<f64>,
        shift: bool,
    ) -> (Frame, Matrix3<f64>) {
        let base = self.base;
        let (u, v) = base.axes();
        let delta = pointer - start;
        let d = Vector2::new(delta.dot(&u), delta.dot(&v));
        let s = anchor.sides();
        let (scale_u, scale_v) = anchor.axes();
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

    /// Rotate about the frame center. Unless `shift` is held, the absolute
    /// angle snaps to whole degrees.
    fn apply_rotate(
        &self,
        start_angle: f64,
        pointer: Point2<f64>,
        shift: bool,
    ) -> (Frame, Matrix3<f64>) {
        let base = self.base;
        let delta = (pointer.y - base.center.y).atan2(pointer.x - base.center.x) - start_angle;
        let angle = base.angle + delta;
        let frame = Frame {
            angle: if shift { angle } else { snap_angle(angle) },
            ..base
        };
        let total = rotate_about(base.center, frame.angle - base.angle) * self.base_total;
        (frame, total)
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

    fn resize(anchor: Anchor, start: Point2<f64>, base: Frame) -> TransformGesture {
        let part = HoverPart::Anchor(anchor);
        TransformGesture::begin(part, start, (base, Matrix3::identity())).unwrap()
    }

    #[test]
    fn move_snaps_to_whole_pixels() {
        // App drag deltas are fractional, but masks live on whole pixels.
        let base = (square(), Matrix3::identity());
        let gesture = TransformGesture::begin(HoverPart::Inside, pt(15.0, 12.5), base).unwrap();
        let (frame, total) = gesture.apply(pt(44.5, 32.5), false);
        assert_eq!(total, Matrix3::new_translation(&Vector2::new(30.0, 20.0)));
        assert_eq!(frame.center, pt(42.5, 32.5));
    }

    #[test]
    fn resize_mirror_and_minimum_magnitude() {
        // Drag the E anchor (15, 12.5) west past the W edge (10) to (5, 12.5):
        // half flips sign, pivot stays, dragged anchor follows the cursor.
        let gesture = resize(Anchor::E, pt(15.0, 12.5), square());
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
            let gesture = resize(Anchor::E, pt(15.0, 12.5), square());
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
            let gesture = resize(Anchor::Se, pt(20.0, 15.0), wide());
            let (frame, _) = gesture.apply(pt(to.0, to.1), false);
            assert_eq!(frame.half, Vector2::new(half.0, half.1));
            assert_eq!(frame.center, pt(center.0, center.1));
        }
        // Pivot (NW corner) fixed by the total matrix.
        let gesture = resize(Anchor::Se, pt(20.0, 15.0), wide());
        let (_, total) = gesture.apply(pt(25.0, 15.0), false);
        let p = total * Vector3::new(10.0, 10.0, 1.0);
        assert!((p.x - 10.0).abs() < 1e-9, "{p:?}");
        assert!((p.y - 10.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn shift_corner_resize_destroys_aspect_ratio() {
        // Holding Shift frees the corner anchors: each axis follows the
        // pointer independently, like edge anchors always do.
        let gesture = resize(Anchor::Se, pt(20.0, 15.0), wide());
        let (frame, _) = gesture.apply(pt(25.0, 15.0), true);
        assert_eq!(frame.half, Vector2::new(7.5, 2.5));
        assert_eq!(frame.center, pt(17.5, 12.5));
    }
}

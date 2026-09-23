use egui::{Color32, CursorIcon, Pos2, Rect as EguiRect, Stroke, Vec2};
use nalgebra::{Point2, Vector2};

use crate::ImagePainter;

use super::super::frame::{Anchor, Frame};

/// Size of resize anchors and the rotate handle in screen pixels.
pub(crate) const ANCHOR_SIZE_PX: f32 = 9.0;
/// Hit-test half-size of anchors in screen pixels. Deliberately larger than
/// the drawn size so grabs stay comfortable.
pub(crate) const ANCHOR_HIT_HALF_PX: f32 = 7.0;
/// Distance of the rotate handle above the bbox top edge in screen pixels.
pub(crate) const ROTATE_HANDLE_GAP_PX: f32 = 22.0;

/// Which part of the selection overlay the pointer is over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HoverPart {
    Outside,
    Inside,
    Anchor(Anchor),
    Rotate,
}

/// Frame corners in screen coordinates: [nw, ne, se, sw].
pub(crate) fn frame_corners_screen(painter: &ImagePainter, frame: &Frame) -> [Pos2; 4] {
    let half = frame.half;
    [
        Vector2::new(-half.x, -half.y),
        Vector2::new(half.x, -half.y),
        Vector2::new(half.x, half.y),
        Vector2::new(-half.x, half.y),
    ]
    .map(|local| {
        let p = frame.point(local);
        painter.image_to_screen(Pos2::new(p.x as f32, p.y as f32))
    })
}

/// Rotate-handle position in image coordinates: outward from the frame's top
/// edge midpoint by `ROTATE_HANDLE_GAP_PX` screen pixels, converted to image
/// units via the render scale so the gap is zoom-independent.
pub(crate) fn rotate_handle_image(painter: &ImagePainter, frame: &Frame) -> Point2<f64> {
    let top_mid = frame.point(Vector2::new(0.0, -frame.half.y));
    let dir: Vector2<f64> = top_mid - frame.center;
    let len = dir.norm();
    let out = if len > f64::EPSILON {
        dir / len
    } else {
        Vector2::new(0.0, -1.0)
    };
    top_mid + out * (f64::from(ROTATE_HANDLE_GAP_PX) / f64::from(painter.render_scale()))
}

/// Hit-test the overlay, entirely in image coordinates: anchors in frame
/// space (so rotated boxes hit-test exactly), handle and anchors with
/// screen-pixel thresholds converted via the render scale.
pub(crate) fn hit_test(painter: &ImagePainter, screen: Pos2, frame: &Frame) -> HoverPart {
    let img = painter.screen_to_image(screen);
    let p = Point2::new(f64::from(img.x), f64::from(img.y));
    let scale = f64::from(painter.render_scale());
    if (p - rotate_handle_image(painter, frame)).norm() <= f64::from(ANCHOR_SIZE_PX) / scale {
        return HoverPart::Rotate;
    }
    let (u, v) = frame.axes();
    let d = p - frame.center;
    let (lu, lv) = (d.dot(&u), d.dot(&v));
    let hit = f64::from(ANCHOR_HIT_HALF_PX) / scale;
    for anchor in Anchor::ALL {
        let s = anchor.sides();
        if (lu - s.x * frame.half.x).abs() <= hit && (lv - s.y * frame.half.y).abs() <= hit {
            return HoverPart::Anchor(anchor);
        }
    }
    if lu.abs() <= frame.half.x.abs() && lv.abs() <= frame.half.y.abs() {
        HoverPart::Inside
    } else {
        HoverPart::Outside
    }
}

/// Resize cursor for an anchor from its world-space direction, so rotated
/// boxes still show a sensible cursor. Mirrored halves flip the drawn
/// direction, so the sign of each half is folded in — otherwise a mirrored
/// corner keeps the pre-mirror icon (e.g. NwSe instead of NeSw).
pub(crate) fn resize_cursor(frame: &Frame, anchor: Anchor) -> CursorIcon {
    let (u, v) = frame.axes();
    let s = anchor.sides();
    let dir = u * (s.x * frame.half.x.signum()) + v * (s.y * frame.half.y.signum());
    let mut deg = dir.y.atan2(dir.x).to_degrees() % 180.0;
    if deg < 0.0 {
        deg += 180.0;
    }
    if deg < 22.5 || deg >= 157.5 {
        CursorIcon::ResizeHorizontal
    } else if deg < 67.5 {
        CursorIcon::ResizeNwSe
    } else if deg < 112.5 {
        CursorIcon::ResizeVertical
    } else {
        CursorIcon::ResizeNeSw
    }
}

/// Draw the oriented dotted frame with anchors and the rotate handle.
pub(crate) fn draw_overlay(painter: &ImagePainter, frame: &Frame) {
    let [nw, ne, se, sw] = frame_corners_screen(painter, frame);
    painter.draw_dotted_line(nw, ne);
    painter.draw_dotted_line(ne, se);
    painter.draw_dotted_line(se, sw);
    painter.draw_dotted_line(sw, nw);

    let egui_painter = painter.painter();
    let half = frame.half;
    for anchor in Anchor::ALL {
        let s = anchor.sides();
        let p = frame.point(Vector2::new(s.x * half.x, s.y * half.y));
        let c = painter.image_to_screen(Pos2::new(p.x as f32, p.y as f32));
        let r = EguiRect::from_center_size(c, Vec2::splat(ANCHOR_SIZE_PX));
        egui_painter.rect_filled(r, 1.0, Color32::WHITE);
        egui_painter.rect_stroke(
            r,
            1.0,
            Stroke::new(1.0, Color32::BLACK),
            egui::StrokeKind::Inside,
        );
    }
    let handle = {
        let p = rotate_handle_image(painter, frame);
        painter.image_to_screen(Pos2::new(p.x as f32, p.y as f32))
    };
    let top_mid = {
        let p = frame.point(Vector2::new(0.0, -half.y));
        painter.image_to_screen(Pos2::new(p.x as f32, p.y as f32))
    };
    egui_painter.line_segment([top_mid, handle], Stroke::new(1.5, Color32::WHITE));
    egui_painter.circle_filled(handle, ANCHOR_SIZE_PX / 2.0, Color32::WHITE);
    egui_painter.circle_stroke(
        handle,
        ANCHOR_SIZE_PX / 2.0,
        Stroke::new(1.0, Color32::BLACK),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    #[test]
    fn mirrored_corner_cursor_follows_visual_direction() {
        // Regression: after mirroring, the resize cursor must match the
        // anchor's drawn (visual) direction, not its logical identity.
        // A horizontal mirror (half.x < 0, e.g. E dragged past W / Y-axis
        // mirror) draws Nw where Ne was, so Nw must show NeSw and vice versa.
        let plain = Frame {
            center: Point2::new(15.0, 12.5),
            half: Vector2::new(5.0, 2.5),
            angle: 0.0,
        };
        assert_eq!(resize_cursor(&plain, Anchor::Se), CursorIcon::ResizeNwSe);
        assert_eq!(resize_cursor(&plain, Anchor::Sw), CursorIcon::ResizeNeSw);

        // Same logical anchors on a Y-axis mirrored frame: diagonal cursors
        // swap, edge anchors stay symmetric.
        let mirrored = Frame {
            center: Point2::new(7.5, 12.5),
            half: Vector2::new(-5.0, 2.5),
            angle: 0.0,
        };
        assert!(mirrored.half.x < 0.0, "{mirrored:?}");
        assert_eq!(resize_cursor(&mirrored, Anchor::Se), CursorIcon::ResizeNeSw);
        assert_eq!(resize_cursor(&mirrored, Anchor::Sw), CursorIcon::ResizeNwSe);
        assert_eq!(resize_cursor(&mirrored, Anchor::Nw), CursorIcon::ResizeNeSw);
        assert_eq!(resize_cursor(&mirrored, Anchor::Ne), CursorIcon::ResizeNwSe);
        assert_eq!(
            resize_cursor(&mirrored, Anchor::E),
            CursorIcon::ResizeHorizontal
        );
        assert_eq!(
            resize_cursor(&mirrored, Anchor::W),
            CursorIcon::ResizeHorizontal
        );
    }
}

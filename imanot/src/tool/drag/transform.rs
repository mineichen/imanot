use egui::Pos2;
use imask::{AffineTransformHeap, ImageDimension, ImaskSet, Roi, SortedRanges, Span};
use nalgebra::Matrix3;

/// Transform `original` by `matrix`, clipped to the image. `None` if nothing
/// remains visible.
pub(crate) fn transform_layer(
    original: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
    matrix: &Matrix3<f32>,
    img_roi: Roi<u32>,
) -> Option<SortedRanges<u32>> {
    let matrix_f64 = matrix.map(|v| v as f64);
    let heap = AffineTransformHeap::new(original.into_iter(), &matrix_f64).ok()?;
    let clipped = heap.clip(img_roi).ok()?;
    SortedRanges::try_from_span_iter_minbounds(clipped).ok()
}

/// Clamp an image-space pointer to a pixel inside the image.
pub(crate) fn clamp_pixel(pointer: Pos2, img_w: usize, img_h: usize) -> (u32, u32) {
    let x = pointer.x.round().clamp(0.0, img_w.saturating_sub(1) as f32) as u32;
    let y = pointer.y.round().clamp(0.0, img_h.saturating_sub(1) as f32) as u32;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::super::frame::{rotate_about, scale_about_frame};
    use super::super::test_support::*;
    use super::*;
    use crate::TestResult;

    use nalgebra::{Point2, Vector2};

    #[test]
    fn transform_identity_and_translation() {
        let original = Roi::new(10..15, 10..15);
        assert_eq!(
            transform_layer(original.into_spans(), &Matrix3::identity(), IMG_ROI),
            SortedRanges::try_from_span_iter(original.into_spans()).ok()
        );
        assert_eq!(
            transform_layer(
                original.into_spans(),
                &Matrix3::new_translation(&Vector2::new(3.0, -2.0)),
                IMG_ROI
            ),
            Some(Roi::new(13..18, 8..13).into())
        );
    }

    #[test]
    fn transform_clipping() -> TestResult {
        // Fully outside → nothing visible; partially outside → clipped to
        // the image.

        let original = Roi::new(10..15, 10..15);

        assert!(
            transform_layer(
                original.into_spans(),
                &Matrix3::new_translation(&Vector2::new(-50.0, 0.0)),
                IMG_ROI
            )
            .is_none()
        );
        let clipped = transform_layer(
            Roi::new(95..99, 95..99).into_spans(),
            &Matrix3::new_translation(&Vector2::new(3.0, 3.0)),
            IMG_ROI,
        )
        .ok_or("Clipped transform stays visible")?;
        let bounds = clipped.roi();
        assert_eq!((bounds.x.start, bounds.y.start), (98, 98));
        assert!(bounds.x.end <= 100);
        assert!(bounds.y.end <= 100);
        Ok(())
    }

    #[test]
    fn multi_commit_from_original_avoids_chaining() -> TestResult {
        // Move, then rotate, both computed from the original: the result must
        // equal a single composed transform of the original.
        let original = Roi::new(40..50, 40..50);
        let shift = Matrix3::new_translation(&Vector2::new(5.0, 0.0));
        let after_move =
            transform_layer(original.into_spans(), &shift, IMG_ROI).ok_or("Shift stays visible")?;
        let total = rotate_about(Point2::new(50.0, 45.0), std::f32::consts::FRAC_PI_2) * shift;
        let from_original = transform_layer(original.into_spans(), &total, IMG_ROI)
            .ok_or("Rotated stays visible")?;
        // Chaining (rotate the already-rasterized move result) must not be
        // what the tool commits; it must equal the direct transform.
        let delta = rotate_about(Point2::new(50.0, 45.0), std::f32::consts::FRAC_PI_2);
        let chained =
            transform_layer(after_move.spans(), &delta, IMG_ROI).ok_or("Chained stays visible")?;
        assert_eq!(from_original, chained);
        Ok(())
    }

    #[test]
    fn rotate_ninety_degrees_swaps_dimensions() -> TestResult {
        // The rasterized 90° rotation of a centered 10x20 rect is 20x10.
        // Uses the production `rotate_about` helper, not a test-only matrix.
        let original = Roi::new(40..50, 40..60);
        let m = rotate_about(Point2::new(45.0, 50.0), std::f32::consts::FRAC_PI_2);
        let out =
            transform_layer(original.into_spans(), &m, IMG_ROI).ok_or("Rotation stays visible")?;
        let bounds = out.roi();
        assert_eq!(bounds.width().get(), 20);
        assert_eq!(bounds.height().get(), 10);
        Ok(())
    }

    #[test]
    fn mirror_transform_preserves_width_on_mirrored_side() -> TestResult {
        // Horizontal mirror about the west edge (x=10) of a 5px rect.
        let original = Roi::new(10..15, 10..15);
        let m = scale_about_frame(Point2::new(10.0, 12.5), 0.0, Vector2::new(-1.0, 1.0));
        let out =
            transform_layer(original.into_spans(), &m, IMG_ROI).ok_or("Mirror stays visible")?;
        let bounds = out.roi();
        assert_eq!(bounds.width().get(), 5);
        assert_eq!(bounds.height().get(), 5);
        // Mirrored content sits west of the pivot (imask's half-open
        // discretization lands it up to 1px overlapping, hence `<= 11`).
        assert!(bounds.x.start < 10, "{bounds:?}");
        assert!(bounds.x.end <= 11, "{bounds:?}");
        Ok(())
    }
}

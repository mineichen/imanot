use imask::{AffineTransformHeap, ImageDimension, ImaskSet, Roi, SortedRanges, Span};
use nalgebra::{Matrix3, Vector2};

/// Re-base a frame-space matrix into imask's span-rasterizer space.
///
/// Our geometry (see [`ImageSpace`](crate::ImageSpace)) is corner-origin:
/// pixel `(x, y)` covers `[x, x+1)²`, its center at `(x + 0.5, y + 0.5)`.
/// imask's [`AffineTransformHeap`] instead evaluates matrices in
/// center-on-integer coordinates: pixel `(x, y)` is `[x-0.5, x+0.5]²`.
/// The two agree on translations (snapped to whole pixels) but differ by
/// the half pixel everywhere else — a rotation or mirror applied un-rebased
/// rasterizes ~1px off its pivot. Converting via
/// `M_span = T(-0.5) · M · T(+0.5)` re-expresses the same physical mapping
/// in imask's coordinates.
pub(crate) fn to_span_space(matrix: &Matrix3<f32>) -> Matrix3<f64> {
    let half = Vector2::new(0.5f64, 0.5);
    Matrix3::new_translation(&-half) * matrix.map(|v| v as f64) * Matrix3::new_translation(&half)
}

/// Transform `original` by `matrix`, clipped to the image. `None` if nothing
/// remains visible. The matrix is in frame space (see [`to_span_space`]).
pub(crate) fn transform_layer(
    original: impl IntoIterator<Item = Span<u32>, IntoIter: ImageDimension>,
    matrix: &Matrix3<f32>,
    img_roi: Roi<u32>,
) -> Option<SortedRanges<u32>> {
    let heap = AffineTransformHeap::new(original.into_iter(), &to_span_space(matrix)).ok()?;
    let clipped = heap.clip(img_roi).ok()?;
    SortedRanges::try_from_span_iter_minbounds(clipped).ok()
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
        // The rasterized 90° rotation of a centered 10x20 rect is a 20x10
        // rect, pixel-exact at the rotated extent [35..55] x [45..55].
        // Uses the production `rotate_about` helper, not a test-only matrix.
        let original = Roi::new(40..50, 40..60);
        let m = rotate_about(Point2::new(45.0, 50.0), std::f32::consts::FRAC_PI_2);
        let out =
            transform_layer(original.into_spans(), &m, IMG_ROI).ok_or("Rotation stays visible")?;
        let bounds = out.roi();
        assert_eq!((bounds.x.start, bounds.x.end), (35, 55));
        assert_eq!((bounds.y.start, bounds.y.end), (45, 55));
        Ok(())
    }

    #[test]
    fn mirror_transform_preserves_width_on_mirrored_side() -> TestResult {
        // Horizontal mirror about the west edge (x=10) of a 5px rect: the
        // content lands pixel-exact at 5..10, sharing the pivot edge.
        let original = Roi::new(10..15, 10..15);
        let m = scale_about_frame(Point2::new(10.0, 12.5), 0.0, Vector2::new(-1.0, 1.0));
        let out =
            transform_layer(original.into_spans(), &m, IMG_ROI).ok_or("Mirror stays visible")?;
        let bounds = out.roi();
        assert_eq!(bounds.width().get(), 5);
        assert_eq!(bounds.height().get(), 5);
        assert_eq!((bounds.x.start, bounds.x.end), (5, 10));
        assert_eq!((bounds.y.start, bounds.y.end), (10, 15));
        Ok(())
    }
}

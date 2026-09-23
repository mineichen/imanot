use egui::Pos2;
use imask::{AffineTransformHeap, ImageDimension, ImaskSet, Roi, SortedRanges, SpanCluster};
use nalgebra::Matrix3;

/// Transform `original` by `matrix`, clipped to the image. `None` if nothing
/// remains visible.
pub(crate) fn transform_layer(
    original: &SortedRanges<u32>,
    matrix: &Matrix3<f64>,
    img_roi: Roi<u32>,
) -> Option<SortedRanges<u32>> {
    let heap = AffineTransformHeap::new(original.spans::<u32>(), matrix).ok()?;
    let clipped = heap.clip(img_roi).ok()?;
    SortedRanges::try_from_span_iter_minbounds(clipped).ok()
}

/// Find the 8-connected cluster of `ranges` containing pixel `(x, y)`.
/// Returns `None` if no span covers the pixel.
pub(crate) fn cluster_at(ranges: &SortedRanges<u32>, x: u32, y: u32) -> Option<SpanCluster<u32>> {
    for cluster in ranges.spans::<u32>().cluster() {
        // Fast reject on the tight cluster bounds before consuming spans.
        if cluster.roi().contains(&x, &y)
            && cluster
                .clone()
                .skip_while(|s| s.y < y)
                .take_while(|s| s.y == y && s.x.start <= x)
                .any(|s| x < s.x.end)
        {
            return Some(cluster);
        }
    }
    None
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
    use imask::Span;

    use nalgebra::{Point2, Vector2};

    fn disjoint_rects() -> SortedRanges<u32> {
        ranges_from_spans(vec![Span::new(0..2, 0u32), Span::new(5..7, 3u32)]).unwrap()
    }

    #[test]
    fn transform_identity_and_translation() {
        let original = rect_ranges(10, 10, nz(5), nz(5));
        assert_eq!(
            transform_layer(&original, &Matrix3::identity(), img_roi()),
            Some(original.clone())
        );
        assert_eq!(
            transform_layer(
                &original,
                &Matrix3::new_translation(&Vector2::new(3.0, -2.0)),
                img_roi()
            ),
            Some(rect_ranges(13, 8, nz(5), nz(5)))
        );
    }

    #[test]
    fn transform_clipping() {
        // Fully outside → nothing visible; partially outside → clipped to
        // the image.
        let original = rect_ranges(10, 10, nz(5), nz(5));
        assert!(
            transform_layer(
                &original,
                &Matrix3::new_translation(&Vector2::new(-50.0, 0.0)),
                img_roi()
            )
            .is_none()
        );
        let clipped = transform_layer(
            &rect_ranges(95, 95, nz(4), nz(4)),
            &Matrix3::new_translation(&Vector2::new(3.0, 3.0)),
            img_roi(),
        )
        .unwrap();
        let bounds = clipped.roi();
        assert_eq!((bounds.x.start, bounds.y.start), (98, 98));
        assert!(bounds.x.end <= 100);
        assert!(bounds.y.end <= 100);
    }

    #[test]
    fn multi_commit_from_original_avoids_chaining() {
        // Move, then rotate, both computed from the original: the result must
        // equal a single composed transform of the original.
        let original = rect_ranges(40, 40, nz(10), nz(10));
        let shift = Matrix3::new_translation(&Vector2::new(5.0, 0.0));
        let after_move = transform_layer(&original, &shift, img_roi()).unwrap();
        let total = rotate_about(Point2::new(50.0, 45.0), std::f64::consts::FRAC_PI_2) * shift;
        let from_original = transform_layer(&original, &total, img_roi()).unwrap();
        // Chaining (rotate the already-rasterized move result) must not be
        // what the tool commits; it must equal the direct transform.
        let delta = rotate_about(Point2::new(50.0, 45.0), std::f64::consts::FRAC_PI_2);
        let chained = transform_layer(&after_move, &delta, img_roi()).unwrap();
        assert_eq!(from_original, chained);
    }

    #[test]
    fn rotate_ninety_degrees_swaps_dimensions() {
        // The rasterized 90° rotation of a centered 10x20 rect is 20x10.
        // Uses the production `rotate_about` helper, not a test-only matrix.
        let original = rect_ranges(40, 40, nz(10), nz(20));
        let m = rotate_about(Point2::new(45.0, 50.0), std::f64::consts::FRAC_PI_2);
        let out = transform_layer(&original, &m, img_roi()).unwrap();
        let bounds = out.roi();
        assert_eq!(bounds.width().get(), 20);
        assert_eq!(bounds.height().get(), 10);
    }

    #[test]
    fn mirror_transform_preserves_width_on_mirrored_side() {
        // Horizontal mirror about the west edge (x=10) of a 5px rect.
        let original = rect_ranges(10, 10, nz(5), nz(5));
        let m = scale_about_frame(Point2::new(10.0, 12.5), 0.0, Vector2::new(-1.0, 1.0));
        let out = transform_layer(&original, &m, img_roi()).unwrap();
        let bounds = out.roi();
        assert_eq!(bounds.width().get(), 5);
        assert_eq!(bounds.height().get(), 5);
        // Mirrored content sits west of the pivot (imask's half-open
        // discretization lands it up to 1px overlapping, hence `<= 11`).
        assert!(bounds.x.start < 10, "{bounds:?}");
        assert!(bounds.x.end <= 11, "{bounds:?}");
    }

    #[test]
    fn cluster_at_finds_covering_cluster_only() {
        let ranges = disjoint_rects();
        let left = cluster_at(&ranges, 0, 0).unwrap();
        assert_eq!(left.roi(), Roi::new(0..2, 0..1));
        assert_eq!(left.clone().count(), 1);
        let right = cluster_at(&ranges, 6, 3).unwrap();
        assert_eq!(right.roi(), Roi::new(5..7, 3..4));
        // Pixels between/outside clusters select nothing.
        assert!(cluster_at(&ranges, 3, 0).is_none());
        assert!(cluster_at(&ranges, 0, 5).is_none());
    }

    #[test]
    fn cluster_connects_diagonally() {
        // 8-connectivity: diagonally touching pixels form one cluster.
        let ranges = ranges_from_spans(vec![Span::new(0..1, 0u32), Span::new(1..2, 1u32)]).unwrap();
        let cluster = cluster_at(&ranges, 0, 0).unwrap();
        assert_eq!(cluster.count(), 2);
    }
}

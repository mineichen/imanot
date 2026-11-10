use std::{
    collections::BinaryHeap,
    iter::FusedIterator,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    sync::atomic::{AtomicU64, Ordering},
};

use imask::{
    CreateRange, ImageDimension, ImaskSet, NonZeroRange, SortedRanges, SortedRangesIter,
    SortedRangesSpanIter, Span,
};
use log::{debug, info};

use crate::PixelArea;

mod history;
mod pixel_area_stack;
mod random_color;

pub use history::*;
pub use pixel_area_stack::*;
pub use random_color::random_color_from_seed;

#[derive(Debug, Eq, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct MaskSettings {
    pub default_opacity: u8,
    pub active_opacity: u8,
}

impl MaskSettings {
    pub fn opacity(&self, active: bool) -> u8 {
        if active {
            self.active_opacity
        } else {
            self.default_opacity
        }
    }
}

impl Default for MaskSettings {
    fn default() -> Self {
        Self {
            default_opacity: 128,
            active_opacity: 200,
        }
    }
}
pub struct MaskImage {
    size: [usize; 2],
    // Often the only reference.
    base: PixelAreaStack,
    applied: AppliedPixelAreaStack,
    history: History,
    /// Toggled with Ctrl/Cmd+D
    visible: bool,
    settings: MaskSettings,
    active_layer: Option<usize>,
    hover_layer: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AffectedLayer {
    #[default]
    Unspecified,
    Layer(usize),
    /// Layers in the half-open range `start..end`, or `start..` when `end` is `None`.
    Range(usize, Option<usize>),
}

impl AffectedLayer {
    pub fn affects(&self, layer: usize) -> bool {
        match self {
            AffectedLayer::Unspecified => true,
            AffectedLayer::Layer(idx) => *idx == layer,
            AffectedLayer::Range(start, end) => match end {
                Some(end) => *start <= layer && layer < *end,
                None => *start <= layer,
            },
        }
    }
}

impl From<usize> for AffectedLayer {
    fn from(layer: usize) -> Self {
        AffectedLayer::Layer(layer)
    }
}

impl From<Range<usize>> for AffectedLayer {
    fn from(range: Range<usize>) -> Self {
        AffectedLayer::Range(range.start, Some(range.end))
    }
}

impl From<RangeFrom<usize>> for AffectedLayer {
    fn from(range: RangeFrom<usize>) -> Self {
        AffectedLayer::Range(range.start, None)
    }
}

impl From<RangeInclusive<usize>> for AffectedLayer {
    fn from(range: RangeInclusive<usize>) -> Self {
        let (start, end) = range.into_inner();
        AffectedLayer::Range(start, end.checked_add(1))
    }
}

impl From<RangeTo<usize>> for AffectedLayer {
    fn from(range: RangeTo<usize>) -> Self {
        AffectedLayer::Range(0, Some(range.end))
    }
}

impl From<RangeToInclusive<usize>> for AffectedLayer {
    fn from(range: RangeToInclusive<usize>) -> Self {
        AffectedLayer::Range(0, range.end.checked_add(1))
    }
}

impl From<RangeFull> for AffectedLayer {
    fn from(_: RangeFull) -> Self {
        AffectedLayer::Range(0, None)
    }
}

impl MaskImage {
    pub fn new(size: [usize; 2], base: impl Into<PixelAreaStack>, history: History) -> Self {
        let base = base.into();
        let applied = AppliedPixelAreaStack::new(&base, &history);
        Self {
            size,
            base: base.clone(),
            applied,
            history,
            visible: true,
            settings: MaskSettings::default(),
            active_layer: None,
            hover_layer: None,
        }
    }

    /// Image dimensions `[width, height]` these masks are bound to.
    pub(crate) fn size(&self) -> [usize; 2] {
        self.size
    }

    pub fn set_settings(&mut self, settings: MaskSettings) {
        self.settings = settings;
    }

    pub(crate) fn settings(&self) -> &MaskSettings {
        &self.settings
    }

    /// Whether the masks are drawn at all (Ctrl/Cmd+D toggles it).
    pub(crate) fn is_visible(&self) -> bool {
        self.visible
    }

    /// Changes whenever the applied layers change (pixels or colors), and is
    /// unique across all `MaskImage`s, so whatever is derived from the layers
    /// can be cached by it. Active layer, settings and visibility don't count:
    /// they are applied when drawing.
    pub(crate) fn revision(&self) -> u64 {
        self.applied.revision
    }

    pub fn random_seed(&self) -> u16 {
        (self.base.max_layer() as u16).wrapping_add(self.history.random_seed())
    }

    pub fn next_color(&self) -> [u8; 4] {
        random_color_from_seed(self.random_seed())
    }

    pub fn is_dirty(&self) -> bool {
        self.history.is_dirty()
    }

    pub fn mark_not_dirty(&mut self) {
        self.history.mark_not_dirty();
    }

    pub fn take_dirty(&mut self) -> Option<AffectedLayer> {
        self.history.take_dirty()
    }

    /// The currently applied (tip) history action, or `None` if history is
    /// empty / fully undone. Tools holding snapshots of mask ranges compare
    /// this against the action they last saw to detect that the mask changed
    /// underneath them (push, undo, redo or another tool's commit).
    pub(crate) fn last_history_action(&self) -> Option<HistoryAction> {
        self.history.iter().next_back().cloned()
    }

    pub fn add_history_action(&mut self, action: HistoryAction) {
        #[cfg(debug_assertions)]
        {
            let max_x = self.size[0] as u32;
            let max_y = self.size[1] as u32;
            let ranges_bounds = match &action.kind {
                HistoryActionKind::Add(add) => Some(add.pixel_area.roi()),
                HistoryActionKind::Clear(clear) => Some(clear.ranges.roi()),
                HistoryActionKind::Replace(replace) => Some(replace.pixel_area.roi()),
                HistoryActionKind::Reset => None,
            };
            if let Some(b) = ranges_bounds {
                assert!(
                    b.x.end <= max_x && b.y.end <= max_y,
                    "Tool wrote SortedRanges at {:?} exceeding image bounds {}x{}",
                    b,
                    max_x,
                    max_y
                );
            }
        }
        self.history.push(action);
        self.applied.mark_redraw(&self.base, &self.history);
    }

    pub fn handle_events(&mut self, ctx: &egui::Context) {
        let (shift_pressed, cmd_z_pressed, cmd_d_pressed) = ctx.input(|i| {
            (
                i.modifiers.shift,
                i.key_pressed(egui::Key::Z) && i.modifiers.command,
                i.key_pressed(egui::Key::D) && i.modifiers.command,
            )
        });

        if cmd_z_pressed {
            let require_redraw = if shift_pressed {
                info!("Redo");
                self.history.redo().is_some()
            } else {
                info!("Undo");
                self.history.undo().is_some()
            };
            if require_redraw {
                self.applied.mark_redraw(&self.base, &self.history);
            };
        }
        if cmd_d_pressed {
            self.visible = !self.visible;
        }
    }

    #[deprecated = "Use Self::active_layer"]
    pub fn active_subgroup(&self) -> Option<usize> {
        self.active_layer()
    }

    pub fn active_layer(&self) -> Option<usize> {
        self.active_layer
    }

    pub fn hover_layer(&self) -> Option<usize> {
        self.hover_layer
    }

    #[deprecated = "Use Self::active_layer"]
    pub fn set_active_subgroup(&mut self, index: Option<usize>) {
        self.set_active_layer(index);
    }
    /// The active layer is drawn with [`MaskSettings::active_opacity`]. Cheap:
    /// the layers themselves don't change.
    pub fn set_active_layer(&mut self, index: Option<usize>) {
        self.active_layer = index;
    }
    pub(crate) fn update_hover_layer(&mut self, pos: Option<(usize, usize)>) {
        let cursor_pos_u32 = pos.and_then(|cursor_pos| {
            let x = u32::try_from(cursor_pos.0).ok()?;
            let y = u32::try_from(cursor_pos.1).ok()?;
            Some((x, y))
        });
        self.hover_layer = if let Some(pos) = cursor_pos_u32 {
            let new_active = self.find_layer_at(pos);
            self.set_active_layer(new_active);
            new_active
        } else {
            let mouse_just_left_canvas = self.hover_layer.is_some();
            if mouse_just_left_canvas {
                self.set_active_layer(None);
            }
            None
        };
    }
    #[deprecated = "Use active_layer_at instead"]
    pub fn active_subgroup_at(&self, p: (u32, u32)) -> Option<usize> {
        self.find_layer_at(p)
    }
    pub fn find_layer_at(&self, (x, y): (u32, u32)) -> Option<usize> {
        self.subgroups_stack().iter().rev().find_map(|(i, area)| {
            (area.pixels.roi().contains(&x, &y)
                && area
                    .pixels
                    .spans::<u32>()
                    .any(|s| s.y == y && s.x.contains(&(x))))
            .then_some(i)
        })
    }

    pub fn subgroups_stack(&self) -> &PixelAreaStack {
        &self.applied.stack
    }

    pub(crate) fn replace_base_layers(&mut self, mut x: PixelAreaStack) -> PixelAreaStack {
        if self.base != x {
            std::mem::swap(&mut self.base, &mut x);
            self.applied.mark_redraw(&self.base, &self.history);
        }
        x
    }

    /// Returns the old value
    pub fn set_base_layer(&mut self, index: usize, area: Option<PixelArea>) -> Option<PixelArea> {
        let old = self.base.set_layer(index, area);
        self.applied.mark_redraw(&self.base, &self.history);
        old
    }

    pub fn set_layer_color(&mut self, index: usize, color: [u8; 4]) {
        self.base.set_layer_color(index, color);
        self.applied.mark_redraw(&self.base, &self.history);
    }

    pub fn layer_color(&self, index: usize) -> Option<[u8; 4]> {
        self.base.layer_color(index)
    }
    fn subgroups_ordered_spans(&self) -> impl FusedIterator<Item = (usize, Span<u32>)> + '_ {
        struct HeapItem<T>(Span<u32>, usize, T);

        impl<T> Eq for HeapItem<T> {}
        impl<T> PartialEq for HeapItem<T> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl<T> PartialOrd for HeapItem<T> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<T> Ord for HeapItem<T> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                (self.0.y, self.0.x.start)
                    .cmp(&(other.0.y, other.0.x.start))
                    .reverse()
            }
        }

        type SpanIter<'a> = SortedRangesSpanIter<
            SortedRangesIter<
                std::iter::Copied<std::slice::Iter<'a, u32>>,
                std::iter::Copied<std::slice::Iter<'a, u32>>,
                NonZeroRange<u32>,
            >,
        >;
        type GroupHeap<'a> = BinaryHeap<HeapItem<SpanIter<'a>>>;
        struct GroupIterator<'a>(GroupHeap<'a>);

        let x: BinaryHeap<_> = self
            .subgroups_stack()
            .iter()
            .filter_map(|(group_id, x)| {
                let mut iter = x.pixels.spans::<u32>();
                Some(HeapItem(iter.next()?, group_id, iter))
            })
            .collect();

        impl<'a> Iterator for GroupIterator<'a> {
            type Item = (usize, Span<u32>);

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(HeapItem(subgroup, group_id, mut rest)) = self.0.pop() {
                    if let Some(x) = rest.next() {
                        self.0.push(HeapItem(x, group_id, rest));
                    }
                    Some((group_id, subgroup))
                } else {
                    None
                }
            }
        }
        impl<'a> FusedIterator for GroupIterator<'a> {}
        GroupIterator(x)
    }
}

struct AppliedPixelAreaStack {
    /// See [`MaskImage::revision`]
    revision: u64,
    stack: PixelAreaStack,
}

impl AppliedPixelAreaStack {
    fn new(base: &PixelAreaStack, history: &History) -> Self {
        static NEXT_REVISION: AtomicU64 = AtomicU64::new(0);
        let base = base.to_layer_vec();
        Self {
            revision: NEXT_REVISION.fetch_add(1, Ordering::Relaxed),
            stack: PixelAreaStack::from_layer_vec(history.iter().fold(base, |acc, r| r.apply(acc))),
        }
    }
    fn mark_redraw(&mut self, base: &PixelAreaStack, history: &History) {
        *self = Self::new(base, history);
    }
}

pub struct AddAction {
    overlapping: bool,
}

pub struct HistoryActionBuilder<'a, A = ()> {
    mask: &'a mut MaskImage,
    layer: AffectedLayer,
    tracked: bool,
    action: A,
}

pub trait MaskActionBuilder<'a>: Sized {
    type Builder;
    fn on_layer(self, layer: AffectedLayer) -> Self::Builder;
    fn without_tracking(self) -> Self::Builder;
    fn keep_overlapping(self, overlapping: bool) -> HistoryActionBuilder<'a, AddAction>;
}

pub trait MaskDefaultActions: Sized {
    fn add(self, ranges: SortedRanges<u32>);
    fn clear(self, ranges: SortedRanges<u32>);
    fn reset(self);
}

impl<'a> MaskActionBuilder<'a> for &'a mut MaskImage {
    type Builder = HistoryActionBuilder<'a>;

    fn on_layer(self, layer: AffectedLayer) -> HistoryActionBuilder<'a> {
        HistoryActionBuilder {
            mask: self,
            layer,
            tracked: true,
            action: (),
        }
    }

    fn without_tracking(self) -> HistoryActionBuilder<'a> {
        HistoryActionBuilder {
            mask: self,
            layer: AffectedLayer::Unspecified,
            tracked: false,
            action: (),
        }
    }

    fn keep_overlapping(self, overlapping: bool) -> HistoryActionBuilder<'a, AddAction> {
        HistoryActionBuilder {
            mask: self,
            layer: AffectedLayer::Unspecified,
            tracked: true,
            action: AddAction { overlapping },
        }
    }
}

impl<'a, A> MaskActionBuilder<'a> for HistoryActionBuilder<'a, A> {
    type Builder = Self;

    fn on_layer(mut self, layer: AffectedLayer) -> Self {
        self.layer = layer;
        self
    }

    fn without_tracking(mut self) -> Self {
        self.tracked = false;
        self
    }

    fn keep_overlapping(self, overlapping: bool) -> HistoryActionBuilder<'a, AddAction> {
        HistoryActionBuilder {
            mask: self.mask,
            layer: self.layer,
            tracked: self.tracked,
            action: AddAction { overlapping },
        }
    }
}

impl MaskDefaultActions for &mut MaskImage {
    fn add(self, subgroups: SortedRanges<u32>) {
        self.keep_overlapping(true).add(subgroups);
    }

    fn clear(self, ranges: SortedRanges<u32>) {
        HistoryActionBuilder {
            mask: self,
            layer: AffectedLayer::Unspecified,
            tracked: true,
            action: (),
        }
        .clear(ranges);
    }

    fn reset(self) {
        HistoryActionBuilder {
            mask: self,
            layer: AffectedLayer::Unspecified,
            tracked: true,
            action: (),
        }
        .reset();
    }
}

impl<'a> MaskDefaultActions for HistoryActionBuilder<'a, ()> {
    fn add(self, subgroups: SortedRanges<u32>) {
        self.keep_overlapping(true).add(subgroups);
    }

    fn clear(self, ranges: SortedRanges<u32>) {
        self.mask.add_history_action(HistoryAction {
            kind: HistoryActionKind::Clear(HistoryActionClear { ranges }),
            layer: self.layer,
            tracked: self.tracked,
        });
    }

    fn reset(self) {
        self.mask.add_history_action(HistoryAction {
            kind: HistoryActionKind::Reset,
            layer: self.layer,
            tracked: self.tracked,
        });
    }
}

impl HistoryActionBuilder<'_, AddAction> {
    pub fn add(self, subgroups: SortedRanges<u32>) {
        let subgroups = if self.action.overlapping {
            subgroups
        } else {
            let reduced = subgroups.map_span_inplace(|new_spans| {
                // Todo: Workaround, as map_span_inplace only works with u64 at the time of writing this
                let b_spans = self.mask.subgroups_ordered_spans().map(|(_layer, x)| Span {
                    y: x.y.into(),
                    x: NonZeroRange::new_debug_checked_zeroable(x.x.start.into(), x.x.end.into()),
                });
                new_spans.subtract(b_spans)
            });
            let Some(x) = reduced else {
                debug!("All Pixels are in a other subgroup already");
                return;
            };
            x
        };
        self.mask.add_history_action(HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: subgroups,
            }),
            layer: self.layer,
            tracked: self.tracked,
        });
    }
}

fn prepare_layer_space(layers: &mut Vec<Layer>, idx: usize) -> &mut Layer {
    while layers.len() <= idx {
        let i = layers.len();
        layers.push(Layer::Empty(crate::random_color_from_seed(i as u16)));
    }

    &mut layers[idx]
}

#[cfg(test)]
mod tests {
    use imask::{ImaskSet, Roi};

    use super::*;
    use crate::TestResult;
    use std::num::NonZero;

    const WIDTH_10: NonZero<u32> = NonZero::new(10).expect("10 is not zero");

    fn mask_10(init: impl Into<PixelAreaStack>) -> MaskImage {
        MaskImage::new([10, 10], init, History::default())
    }

    impl MaskImage {
        fn subgroup_spans_flat(&self) -> impl Iterator<Item = (usize, Span<u32>)> {
            self.subgroups_stack()
                .iter()
                .map(|(i, x)| (i, &x.pixels))
                .flat_map(|(i, x)| x.spans::<u32>().map(move |s| (i, s)))
        }
    }

    #[test]
    fn affected_layer_affects() {
        assert!(AffectedLayer::Unspecified.affects(0));
        assert!(AffectedLayer::Unspecified.affects(10));
        assert!(AffectedLayer::Layer(2).affects(2));
        assert!(!AffectedLayer::Layer(2).affects(3));
        assert!(AffectedLayer::Range(1, Some(4)).affects(1));
        assert!(AffectedLayer::Range(1, Some(4)).affects(3));
        assert!(!AffectedLayer::Range(1, Some(4)).affects(4));
        assert!(!AffectedLayer::Range(1, Some(4)).affects(0));
        assert!(AffectedLayer::Range(2, None).affects(2));
        assert!(AffectedLayer::Range(2, None).affects(100));
        assert!(!AffectedLayer::Range(2, None).affects(1));
    }

    #[test]
    fn add_to_range_of_layers() {
        let mut mask_image = mask_10(Vec::new());
        mask_image
            .on_layer(AffectedLayer::Range(0, Some(2)))
            .add(SortedRanges::from(Span::new(1..3, 0)));
        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(1..3, 0)), (1, Span::new(1..3, 0)),]
        );
    }

    #[test]
    fn add_to_open_ended_range_of_layers() {
        let mut mask_image = mask_10(Vec::new());
        mask_image.add(SortedRanges::from(Span::new(1..3, 0)));
        mask_image.add(SortedRanges::from(Span::new(4..6, 0)));
        mask_image
            .on_layer(AffectedLayer::Range(1, None))
            .add(SortedRanges::from(Span::new(2..5, 0)));
        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(1..3, 0)), (1, Span::new(2..6, 0)),]
        );
    }

    #[test]
    fn clear_range_of_layers() {
        let mut mask_image = mask_10(Vec::new());
        mask_image.add(SortedRanges::from(Span::new(1..9, 0)));
        mask_image.add(SortedRanges::from(Span::new(2..8, 0)));
        mask_image
            .on_layer(AffectedLayer::Range(0, Some(2)))
            .clear(Roi::new(0..4, 0..1).into());
        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(4..9, 0)), (1, Span::new(4..8, 0)),]
        );
    }

    #[test]
    fn reset_range_of_layers() {
        let mut mask_image = mask_10(Vec::new());
        mask_image.add(SortedRanges::from(Span::new(1..9, 0)));
        mask_image.add(SortedRanges::from(Span::new(2..8, 0)));
        mask_image.add(SortedRanges::from(Span::new(3..7, 0)));
        mask_image
            .on_layer(AffectedLayer::Range(0, Some(2)))
            .reset();
        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(2, Span::new(3..7, 0))]
        );
    }

    #[test]
    fn add_area_with_overlap() {
        let mut mask_image = mask_10(Vec::new());
        mask_image.add(SortedRanges::from(Span::new(1..5, 0)));
        mask_image.add(SortedRanges::from(Span::new(2..4, 0)));
        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(1..5, 0)), (1, Span::new(2..4, 0)),]
        );
    }

    #[test]
    fn add_area_non_overlapping_parts_remove_completely() {
        let mut mask_image = mask_10(Vec::new());

        let ranges = SortedRanges::from(Span::new(1..5, 0));
        mask_image.keep_overlapping(false).add(ranges);
        let ranges = SortedRanges::from(Span::new(2..4, 0));
        mask_image.keep_overlapping(false).add(ranges);

        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(1..5, 0)),]
        );
    }
    #[test]
    fn add_area_non_overlapping_parts_remove_partially() {
        let mut mask_image = mask_10(Vec::new());

        let ranges = SortedRanges::from(Span::new(1..5, 0));
        mask_image.keep_overlapping(false).add(ranges);
        let ranges = SortedRanges::from(Span::new(2..6, 0));
        mask_image.keep_overlapping(false).add(ranges);

        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(1..5, 0)), (1, Span::new(5..6, 0)),]
        );
    }

    #[test]
    fn clear_should_remove_multiple_overlapping_areas_start() {
        let mut mask_image = mask_10(Vec::new());

        let ranges = SortedRanges::from(Span::new(1..9, 0));
        mask_image.add(ranges);
        let ranges = SortedRanges::from(Span::new(2..8, 0));
        mask_image.add(ranges);

        mask_image.clear(Roi::new(0..4, 0..1).into());

        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(4..9, 0)), (1, Span::new(4..8, 0)),]
        );
    }

    #[test]
    fn clear_should_remove_multiple_overlapping_areas_end() {
        let mut mask_image = mask_10(Vec::new());

        let ranges = SortedRanges::from(Span::new(2..8, 0));
        mask_image.add(ranges);
        let ranges = SortedRanges::from(Span::new(1..9, 0));
        mask_image.add(ranges);

        mask_image.clear(Roi::new(5..10, 0..1).into());

        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![(0, Span::new(2..5, 0)), (1, Span::new(1..5, 0)),]
        );
    }

    #[test]
    fn clear_should_remove_multiple_overlapping_areas_within() {
        let mut mask_image = mask_10(Vec::new());

        let ranges = SortedRanges::from(Span::new(1..9, 0));
        mask_image.add(ranges);
        let ranges = SortedRanges::from(Span::new(2..8, 0));
        mask_image.add(ranges);

        mask_image.clear(Roi::new(4..6, 0..1).into());

        assert_eq!(
            mask_image.subgroup_spans_flat().collect::<Vec<_>>(),
            vec![
                (0, Span::new(1..4, 0)),
                (0, Span::new(6..9, 0)),
                (1, Span::new(2..4, 0)),
                (1, Span::new(6..8, 0)),
            ]
        );
    }

    #[test]
    fn iter_sorted() -> TestResult {
        let history = History::default();
        let mut x = MaskImage::new(
            [10, 10],
            vec![
                PixelArea::with_black_color(
                    [Span::new(2..7, 0), Span::new(2..7, 1)].with_bounds(WIDTH_10, WIDTH_10),
                )
                .ok_or("Test spans are valid")?,
                PixelArea {
                    pixels: SortedRanges::from(Span::new(2..7, 3)),
                    color: [0, 0, 0, 255],
                },
            ],
            history,
        );
        x.add(SortedRanges::try_from_span_iter(
            [
                Span::new(2u32..9, 2),
                Span::new(9..10, 3),
                Span::new(2..9, 4),
            ]
            .with_bounds(WIDTH_10, WIDTH_10),
        )?);
        let group_sequence: Vec<_> = x
            .subgroups_ordered_spans()
            .map(|(group_id, _)| group_id)
            .collect();
        assert_eq!(group_sequence, vec![0, 0, 2, 1, 2, 2]);
        Ok(())
    }

    #[test]
    fn non_overlapping_no_pixel_overlap_with_multiple_layers() -> TestResult {
        let mut mask_image = MaskImage::new([10, 10], vec![], History::default());

        let layer0 = SortedRanges::try_from_span_iter(
            [Span::new(0u32..5, 0), Span::new(2..5, 1)].with_bounds(WIDTH_10, WIDTH_10),
        )?;
        mask_image.add(layer0);

        let spans1 = [Span::new(0u32..5, 3), Span::new(0..3, 5)].with_bounds(WIDTH_10, WIDTH_10);
        let layer1 = SortedRanges::try_from_span_iter(spans1)?;
        mask_image.add(layer1);

        let new_mask = SortedRanges::try_from_span_iter(
            [
                Span::new(3u32..8, 0),
                Span::new(8..10, 2),
                Span::new(0..3, 3),
                Span::new(9..10, 4),
                Span::new(0..4, 5),
            ]
            .with_bounds(WIDTH_10, WIDTH_10),
        )?;
        mask_image.keep_overlapping(false).add(new_mask);

        let subgroups = mask_image.subgroups_stack();
        assert_eq!(subgroups.max_layer(), 3);

        let mut subgroups_iter = subgroups.iter();
        let all_existing_pixels: Vec<u64> = (&mut subgroups_iter)
            .take(2)
            .flat_map(|(_i, area)| {
                area.pixels
                    .iter_roi::<std::ops::Range<u64>>()
                    .flat_map(|r| r.start..r.end)
            })
            .collect();

        let new_layer_pixels: Vec<u64> = subgroups_iter
            .next()
            .ok_or("Test setup adds three layers")?
            .1
            .pixels
            .iter_roi::<std::ops::Range<u64>>()
            .flat_map(|r| r.start..r.end)
            .collect();

        for pixel in &new_layer_pixels {
            assert!(
                !all_existing_pixels.contains(pixel),
                "Pixel {pixel} exists in both new layer and existing layers"
            );
        }
        Ok(())
    }

    #[test]
    fn add_to_existing_overlapping_doesnt_fail() {
        let mut history = History::default();
        history.push(HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: SortedRanges::from(Span::new(0..2, 0)),
            }),
            layer: AffectedLayer::Unspecified,
            tracked: true,
        });
        history.push(HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: SortedRanges::from(Span::new(1..5, 0)),
            }),
            layer: AffectedLayer::Unspecified,
            tracked: true,
        });
        let mut x = MaskImage::new([10, 10], vec![], history);
        x.keep_overlapping(false)
            .add(SortedRanges::from(Span::new(2..6, 0)));
        let group_sequence: Vec<_> = x
            .subgroups_ordered_spans()
            .map(|(group_id, _)| group_id)
            .collect();
        assert_eq!(group_sequence, vec![0, 1, 2]);
    }
}

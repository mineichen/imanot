//! History is a stack of actions that can be aplied to Vec<SubGroups>.
//! There is no undo on Vec<SubGroups>, but the original Vec<SubGroup> can be converted multiple times to get the Aggregated result.
//! This way, a we don't need to implement undo, which would require additional infos in HistoryAction
use imask::{ImaskSet, SortedRanges};

use crate::{AffectedLayer, PixelArea};

use super::pixel_area_stack::Layer;

fn take_layer_ranges(
    layers: &mut Vec<Layer>,
    idx: usize,
) -> (&mut Layer, Option<SortedRanges<u32>>) {
    let target = super::prepare_layer_space(layers, idx);
    let ranges = target.clear_ranges();
    (target, ranges)
}

fn apply_add(layers: &mut Vec<Layer>, idx: usize, add: &HistoryActionAdd) {
    let (target, maybe_existing) = take_layer_ranges(layers, idx);
    if let Some(existing) = maybe_existing {
        let new_spans = add.pixel_area.spans::<u64>();
        let mapped = existing.map_span_inplace(|existing_spans| existing_spans.union(new_spans));
        if let Some(new_area) = mapped {
            target.set_ranges(new_area);
        }
    } else {
        target.set_ranges(add.pixel_area.clone());
    };
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct HistoryActionAdd {
    pub pixel_area: SortedRanges<u32>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct HistoryActionClear {
    pub ranges: SortedRanges<u32>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct HistoryActionReplace {
    pub pixel_area: SortedRanges<u32>,
}

fn apply_replace(
    layers: &mut Vec<Layer>,
    idxs: impl IntoIterator<Item = usize>,
    replace: &HistoryActionReplace,
) {
    let source = replace.pixel_area.clone();
    let mut idxs = idxs.into_iter();
    let Some(first) = idxs.next() else {
        return;
    };
    idxs.for_each(|idx| {
        super::prepare_layer_space(layers, idx).set_ranges(source.clone());
    });
    super::prepare_layer_space(layers, first).set_ranges(source);
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum HistoryActionKind {
    Add(HistoryActionAdd),
    Reset,
    Clear(HistoryActionClear),
    Replace(HistoryActionReplace),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct HistoryAction {
    pub kind: HistoryActionKind,
    pub layer: AffectedLayer,
    pub tracked: bool,
}

impl HistoryAction {
    pub fn layer(&self) -> AffectedLayer {
        self.layer
    }
    pub(in crate::mask) fn apply(&self, mut rest: Vec<Layer>) -> Vec<Layer> {
        match &self.kind {
            HistoryActionKind::Add(add) => match self.layer {
                AffectedLayer::Unspecified => {
                    let color = crate::random_color_from_seed(rest.len() as u16);
                    let pixel_area = PixelArea::from_ranges(add.pixel_area.clone(), color);
                    rest.push(Layer::Filled(pixel_area));
                    rest
                }
                AffectedLayer::Layer(idx) => {
                    apply_add(&mut rest, idx, add);
                    rest
                }
                AffectedLayer::Range(start, Some(end)) => {
                    for idx in start..end {
                        apply_add(&mut rest, idx, add);
                    }
                    rest
                }
                AffectedLayer::Range(start, None) => {
                    for idx in start..rest.len() {
                        apply_add(&mut rest, idx, add);
                    }
                    rest
                }
            },
            HistoryActionKind::Reset => match self.layer {
                AffectedLayer::Unspecified => {
                    rest.clear();
                    rest
                }
                affected => {
                    for (idx, target) in rest.iter_mut().enumerate() {
                        if affected.affects(idx) {
                            target.clear_ranges();
                        }
                    }
                    rest
                }
            },
            HistoryActionKind::Clear(clear) => {
                for (idx, target) in rest.iter_mut().enumerate() {
                    if self.layer.affects(idx)
                        && let Some(ranges) = target.clear_ranges()
                        && let Some(new_area) = ranges.map_span_inplace(|existing_spans| {
                            existing_spans.subtract(clear.ranges.spans::<u64>())
                        })
                    {
                        target.set_ranges(new_area);
                    }
                }
                rest
            }
            HistoryActionKind::Replace(replace) => match self.layer {
                AffectedLayer::Unspecified => {
                    let idxs = 0..rest.len();
                    apply_replace(&mut rest, idxs, replace);
                    rest
                }
                AffectedLayer::Layer(idx) => {
                    apply_replace(&mut rest, std::iter::once(idx), replace);
                    rest
                }
                AffectedLayer::Range(start, Some(end)) => {
                    apply_replace(&mut rest, start..end, replace);
                    rest
                }
                AffectedLayer::Range(start, None) => {
                    let idxs = start..rest.len();
                    apply_replace(&mut rest, idxs, replace);
                    rest
                }
            },
        }
    }
}

pub struct History {
    actions: Vec<HistoryAction>,
    end: usize,
    not_dirty_pos: Option<usize>,
    undo_redo_layer: Option<AffectedLayer>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            actions: Default::default(),
            end: Default::default(),
            not_dirty_pos: Some(0),
            undo_redo_layer: None,
        }
    }
}

impl History {
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &'_ HistoryAction> {
        self.actions.iter().take(self.end)
    }

    pub(crate) fn random_seed(&self) -> u16 {
        self.end as u16
    }

    pub fn is_dirty(&self) -> bool {
        self.not_dirty_pos != Some(self.end)
    }
    pub fn take_dirty(&mut self) -> Option<AffectedLayer> {
        if self.is_dirty() {
            self.mark_not_dirty();
            self.undo_redo_layer
                .or_else(|| self.iter().rev().next().map(|a| a.layer))
        } else {
            None
        }
    }

    pub fn mark_not_dirty(&mut self) {
        self.not_dirty_pos = Some(self.end);
    }

    pub fn push(&mut self, new_action: HistoryAction) {
        self.undo_redo_layer = None;

        let last_action = self.end.checked_sub(1).and_then(|i| self.actions.get(i));
        if let Some(i) = last_action
            && i.kind == HistoryActionKind::Reset
            && i.kind == new_action.kind
            && i.layer == new_action.layer
        {
            return;
        }

        match &mut self.not_dirty_pos {
            Some(pos) if *pos > self.end => {
                self.not_dirty_pos = None;
            }
            _ => (),
        }

        self.actions.truncate(self.end);
        self.actions.push(new_action);
        self.end = self.actions.len();
    }

    pub fn redo(&mut self) -> Option<&HistoryAction> {
        let tracked_idx = (self.end..self.actions.len()).find(|&i| self.actions[i].tracked)?;
        let mut new_end = tracked_idx + 1;
        while new_end < self.actions.len() && !self.actions[new_end].tracked {
            new_end += 1;
        }
        self.end = new_end;
        self.undo_redo_layer = Some(self.actions[tracked_idx].layer);
        Some(&self.actions[tracked_idx])
    }
    pub fn undo(&mut self) -> Option<&HistoryAction> {
        let tracked_idx = (0..self.end).rev().find(|&i| self.actions[i].tracked)?;
        let action = &self.actions[tracked_idx];
        self.end = tracked_idx;
        self.undo_redo_layer = Some(action.layer);
        Some(action)
    }
}

#[cfg(test)]
mod tests {
    use imask::Span;

    use super::*;
    use crate::PixelAreaStack;

    fn tracked_add(x: u16) -> HistoryAction {
        HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: Span::new(x..x + 1, 0).into(),
            }),
            layer: AffectedLayer::Unspecified,
            tracked: true,
        }
    }

    fn untracked_add(x: u16) -> HistoryAction {
        HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: Span::new(x..x + 1, 0).into(),
            }),
            layer: AffectedLayer::Unspecified,
            tracked: false,
        }
    }

    #[test]
    fn undo_empty_returns_none() {
        let mut history = History::default();
        assert_eq!(None, history.undo());
    }

    #[test]
    fn insert_undo_and_redo() {
        let mut history = History::default();
        let item = tracked_add(0);
        history.push(item.clone());
        assert_eq!(history.undo(), Some(&item));
        assert_eq!(history.undo(), None);
        assert_eq!(history.redo(), Some(&item));
    }

    #[test]
    fn push_after_undo() {
        let mut history = History::default();
        let item = tracked_add(0);
        let item2 = tracked_add(10);
        history.push(item.clone());
        assert_eq!(history.undo(), Some(&item));
        assert_eq!(history.undo(), None);
        history.push(item2);
        assert_eq!(None, history.redo());
    }

    #[test]
    fn undo_redo_group_tracked_and_untracked() {
        let mut history = History::default();
        let tracked = tracked_add(0);
        let untracked = untracked_add(10);

        history.push(tracked.clone());
        history.push(untracked.clone());

        assert_eq!(history.end, 2);

        history.undo();
        assert_eq!(history.end, 0);

        history.redo();
        assert_eq!(history.end, 2);
    }

    #[test]
    fn undo_redo_multiple_groups() {
        let mut history = History::default();
        let a = tracked_add(0);
        let b = untracked_add(10);
        let c = tracked_add(20);
        let d = untracked_add(30);

        history.push(a.clone());
        history.push(b.clone());
        history.push(c.clone());
        history.push(d.clone());
        assert_eq!(history.end, 4);

        history.undo();
        assert_eq!(history.end, 2);

        history.undo();
        assert_eq!(history.end, 0);

        history.redo();
        assert_eq!(history.end, 2);

        history.redo();
        assert_eq!(history.end, 4);
    }

    #[test]
    fn redo_stops_at_next_tracked() {
        let mut history = History::default();
        let a = tracked_add(0);
        let b = untracked_add(10);

        history.push(a.clone());
        history.push(b.clone());

        history.undo();
        assert_eq!(history.end, 0);

        history.redo();
        assert_eq!(history.end, 2);

        history.undo();
        assert_eq!(history.end, 0);
    }

    #[test]
    fn only_untracked_actions_cannot_undo() {
        let mut history = History::default();
        history.push(untracked_add(0));
        history.push(untracked_add(10));

        assert_eq!(history.end, 2);

        assert_eq!(None, history.undo());
        assert_eq!(history.end, 2);
    }

    #[test]
    fn consecutive_resets_different_layers_are_not_deduped() {
        let mut history = History::default();
        history.push(HistoryAction {
            kind: HistoryActionKind::Reset,
            layer: AffectedLayer::Layer(0),
            tracked: false,
        });
        history.push(HistoryAction {
            kind: HistoryActionKind::Reset,
            layer: AffectedLayer::Layer(1),
            tracked: false,
        });

        assert_eq!(history.actions.len(), 2);
    }

    #[test]
    fn undo_affected_layer() {
        let mut history = History::default();
        history.push(HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: Span::new(0u16..1, 0).into(),
            }),
            layer: AffectedLayer::Layer(0),
            tracked: true,
        });
        history.push(HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: Span::new(1u16..2, 0).into(),
            }),
            layer: AffectedLayer::Layer(1),
            tracked: true,
        });

        assert_eq!(history.take_dirty(), Some(AffectedLayer::Layer(1)));
        history.undo();
        assert_eq!(history.take_dirty(), Some(AffectedLayer::Layer(1)));
    }

    fn untracked_add_area(x: u16, y: u16) -> HistoryAction {
        HistoryAction {
            kind: HistoryActionKind::Add(HistoryActionAdd {
                pixel_area: Span::new(x..x + 2, y).into(),
            }),
            layer: AffectedLayer::Unspecified,
            tracked: false,
        }
    }

    fn replace_action(pixel_area: SortedRanges<u32>, layer: AffectedLayer) -> HistoryAction {
        HistoryAction {
            kind: HistoryActionKind::Replace(HistoryActionReplace { pixel_area }),
            layer,
            tracked: true,
        }
    }

    fn applied_spans(history: &History) -> Vec<(usize, Vec<Span<u32>>)> {
        PixelAreaStack::from_layer_vec(
            history
                .iter()
                .fold(Vec::new(), |acc: Vec<Layer>, r| r.apply(acc)),
        )
        .iter()
        .map(|(i, area)| (i, area.pixels.spans::<u32>().collect()))
        .collect()
    }

    #[test]
    fn replace_specific_layer() {
        let mut history = History::default();
        history.push(untracked_add_area(0, 0));
        history.push(untracked_add_area(0, 1));
        history.push(replace_action(
            Span::new(4u16..6, 3).into(),
            AffectedLayer::Layer(1),
        ));

        assert_eq!(
            applied_spans(&history),
            vec![(0, vec![Span::new(0..2, 0)]), (1, vec![Span::new(4..6, 3)]),]
        );
    }

    #[test]
    fn replace_unspecified_replaces_all_existing_layers() {
        let mut history = History::default();
        history.push(untracked_add_area(0, 0));
        history.push(untracked_add_area(0, 1));
        history.push(replace_action(
            Span::new(4u16..6, 3).into(),
            AffectedLayer::Unspecified,
        ));

        assert_eq!(
            applied_spans(&history),
            vec![(0, vec![Span::new(4..6, 3)]), (1, vec![Span::new(4..6, 3)]),]
        );
    }

    #[test]
    fn replace_range_of_layers() {
        let mut history = History::default();
        history.push(untracked_add_area(0, 0));
        history.push(untracked_add_area(0, 1));
        history.push(untracked_add_area(0, 2));
        history.push(replace_action(
            Span::new(4u16..6, 3).into(),
            AffectedLayer::Range(1, Some(3)),
        ));

        assert_eq!(
            applied_spans(&history),
            vec![
                (0, vec![Span::new(0..2, 0)]),
                (1, vec![Span::new(4..6, 3)]),
                (2, vec![Span::new(4..6, 3)]),
            ]
        );
    }
}

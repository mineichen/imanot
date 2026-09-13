use std::sync::Arc;

use imask::SortedRanges;

use crate::PixelArea;

// Keep the enum, so we can iter over &PixelArea, which would not be possible for struct { color: [u8;4], ranges: Option<Ranges> }
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Layer {
    Empty([u8; 4]),
    Filled(PixelArea),
}

impl Layer {
    pub(in crate::mask) fn into_filled(self) -> Option<PixelArea> {
        match self {
            Layer::Filled(a) => Some(a),
            _ => None,
        }
    }
    pub(in crate::mask) fn as_filled(&self) -> Option<&PixelArea> {
        match self {
            Layer::Empty(_) => None,
            Layer::Filled(a) => Some(a),
        }
    }
    pub fn color(&self) -> [u8; 4] {
        match self {
            Layer::Empty(c) => *c,
            Layer::Filled(pixel_area) => pixel_area.color,
        }
    }
    pub fn set_color(&mut self, color: [u8; 4]) {
        match self {
            Layer::Empty(x) => *x = color,
            Layer::Filled(pixel_area) => pixel_area.color = color,
        }
    }
    pub fn set_ranges(&mut self, pixels: SortedRanges<u32>) {
        let color = self.color();
        *self = Layer::Filled(PixelArea { pixels, color })
    }

    pub fn clear_ranges(&mut self) -> Option<SortedRanges<u32>> {
        let color = self.color();
        let mut new_value = Layer::Empty(color);
        std::mem::swap(self, &mut new_value);
        new_value.into_filled().map(|x| x.pixels)
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct PixelAreaStack {
    areas: Arc<Vec<Layer>>,
}

impl From<Vec<PixelArea>> for PixelAreaStack {
    fn from(areas: Vec<PixelArea>) -> Self {
        Self {
            areas: Arc::new(areas.into_iter().map(Layer::Filled).collect()),
        }
    }
}

impl PixelAreaStack {
    pub fn get(&self, i: usize) -> Option<&PixelArea> {
        self.areas.get(i).and_then(Layer::as_filled)
    }
    pub fn from_iter(areas: impl IntoIterator<Item = (usize, PixelArea)>) -> Self {
        let areas = areas.into_iter();
        let mut all: Vec<Layer> = Vec::with_capacity(areas.size_hint().0);
        for (i, area) in areas {
            while all.len() <= i {
                let idx = all.len();
                all.push(Layer::Empty(crate::random_color_from_seed(idx as u16)));
            }
            all[i] = Layer::Filled(area);
        }
        Self {
            areas: Arc::new(all),
        }
    }

    pub fn max_layer(&self) -> usize {
        self.areas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.areas.iter().all(|l| matches!(l, Layer::Empty(_)))
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (usize, &'_ PixelArea)> {
        PixelAreaStackIter {
            index: 0,
            end_index: self.areas.len(),
            inner: self.areas.iter().map(|l| match l {
                Layer::Filled(a) => Some(a),
                Layer::Empty(_) => None,
            }),
        }
    }

    pub fn set_layer_color(&mut self, index: usize, color: [u8; 4]) {
        let inner = Arc::make_mut(&mut self.areas);
        let target = super::prepare_layer_space(inner, index);
        target.set_color(color);
    }

    pub fn layer_color(&self, index: usize) -> Option<[u8; 4]> {
        self.areas.get(index).map(|l| match l {
            Layer::Empty(c) => *c,
            Layer::Filled(a) => a.color,
        })
    }

    pub(in crate::mask) fn set_layer(
        &mut self,
        index: usize,
        area: Option<PixelArea>,
    ) -> Option<PixelArea> {
        let inner = Arc::make_mut(&mut self.areas);
        while inner.len() <= index {
            let i = inner.len();
            inner.push(Layer::Empty(crate::random_color_from_seed(i as u16)));
        }
        let old = match &inner[index] {
            Layer::Filled(old) => Some(old.clone()),
            _ => None,
        };
        match area {
            Some(new) => inner[index] = Layer::Filled(new),
            None => inner[index] = Layer::Empty(crate::random_color_from_seed(index as u16)),
        }
        old
    }

    pub(in crate::mask) fn to_layer_vec(&self) -> Vec<Layer> {
        (*self.areas).clone()
    }
    pub(in crate::mask) fn from_layer_vec(areas: Vec<Layer>) -> Self {
        Self {
            areas: Arc::new(areas),
        }
    }
}

impl IntoIterator for PixelAreaStack {
    type Item = (usize, PixelArea);

    type IntoIter = PixelAreaStackIter<std::vec::IntoIter<Option<PixelArea>>>;

    fn into_iter(mut self) -> Self::IntoIter {
        let extracted = std::sync::Arc::make_mut(&mut self.areas);

        let vec: Vec<Option<PixelArea>> = std::mem::take(extracted)
            .into_iter()
            .map(|l| match l {
                Layer::Filled(a) => Some(a),
                Layer::Empty(_) => None,
            })
            .collect();

        PixelAreaStackIter {
            index: 0,
            end_index: vec.len(),
            inner: vec.into_iter(),
        }
    }
}

pub struct PixelAreaStackIter<T> {
    index: usize,
    end_index: usize,
    inner: T,
}

impl<T: Iterator<Item = Option<TItem>>, TItem> Iterator for PixelAreaStackIter<T> {
    type Item = (usize, TItem);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.inner.next()?;
            let cur_index = self.index;
            self.index += 1;
            if let Some(x) = next {
                return Some((cur_index, x));
            }
        }
    }
}

impl<T: DoubleEndedIterator<Item = Option<TItem>>, TItem> DoubleEndedIterator
    for PixelAreaStackIter<T>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.inner.next_back()?;
            self.end_index -= 1;

            if let Some(x) = next {
                return Some((self.end_index, x));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;

    use imask::{SortedRanges, Span};

    use super::*;
    const NON_ZERO_10: NonZeroU16 = NonZeroU16::new(10).unwrap();
    #[test]
    fn allow_unordered() {
        let stack = PixelAreaStack::from_iter([
            (10, PixelArea::single_range_total_black(0, 0, NON_ZERO_10)),
            (1, PixelArea::single_range_total_black(0, 0, NON_ZERO_10)),
        ]);
        assert!(stack.get(1).is_some());
        assert!(stack.get(10).is_some());
    }

    #[test]
    fn test_end_index() {
        let ranges = SortedRanges::from(Span::new(0u16..10, 0));
        let example = PixelArea::from_ranges(ranges, [0, 0, 0, 255]);
        let x =
            PixelAreaStack::from_iter([(1, example.clone()), (3, example.clone()), (5, example)]);

        let iter = x.iter().map(|(i, _)| i);
        assert_eq!(vec![1, 3, 5], iter.collect::<Vec<_>>());
    }

    #[test]
    fn set_color_auto_fills_gaps() {
        let mut stack = PixelAreaStack::default();
        stack.set_layer_color(5, [255, 0, 0, 255]);
        assert_eq!(stack.max_layer(), 6);
        assert!(stack.is_empty());
        assert_eq!(stack.layer_color(5), Some([255, 0, 0, 255]));
    }
}

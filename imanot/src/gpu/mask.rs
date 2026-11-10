//! Masks drawn straight from their ranges, the part all backends share.
//!
//! No texture is involved, the pixels of each layer are drawn as runs: a
//! rectangle one pixel high. A [`SortedRanges`](imask::SortedRanges) range
//! runs along the image rows and may wrap, so it becomes one run per row it
//! touches (see [`range_runs`]). Two runs of a layer are one [`Instance`] of
//! 12 bytes, 6 per run, drawn as one triangle strip. The vertex shader
//! computes the corners from the vertex number, there is no vertex or index
//! buffer.
//! - One draw call for all layers: the instances of each layer follow those
//!   of the layer below. GPUs blend the triangles of a draw call in order and
//!   the runs of a layer never overlap, so this gives the same source-over
//!   result as compositing the layers first.
//! - What the instances of a layer share is not in the instances: its color
//!   and where its instances end are uniforms (see [`LayerUniform`]), the
//!   vertex shader finds the layer of an instance by binary search. The active
//!   layer and the opacity are uniforms as well. Instances and layers are only
//!   rebuilt and uploaded when the layers change, see [`MaskImage::revision`].

use std::{mem::offset_of, num::NonZeroU32, ops::Range, sync::Arc};

use egui::{Rect, epaint::ViewportInPixels, layers::ShapeIdx};

use crate::MaskImage;

use super::GpuBackend;

/// Two runs of a layer: the pixels `x[i][0]..x[i][1]` of row `y[i]`. As the
/// vertex shaders read it: attribute 0 is `x`, 1 is `y`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Instance {
    /// In image pixels, the ends exclusive. `u16` is plenty, no GPU takes
    /// textures that large.
    x: [[u16; 2]; 2],
    y: [u16; 2],
}

impl Instance {
    /// From runs `[x0, x1, y]` within the image, whose size fits `u16`.
    /// Without `second`, the second run is empty and draws nothing: drawing
    /// `first` twice would blend it twice.
    fn new(first: [u32; 3], second: Option<[u32; 3]>) -> Self {
        let [[x0, x1, y], [second_x0, second_x1, second_y]] =
            [first, second.unwrap_or_default()].map(|run| run.map(|value| value as u16));
        Self {
            x: [[x0, x1], [second_x0, second_x1]],
            y: [y, second_y],
        }
    }
}

/// Bytes of one [`Instance`]
pub(super) const INSTANCE_SIZE: usize = size_of::<Instance>();
// The vertex attributes of the backends rely on this layout
const _: () =
    assert!(offset_of!(Instance, x) == 0 && offset_of!(Instance, y) == 8 && INSTANCE_SIZE == 12);
/// Vertices per instance: two rectangles as one triangle strip, joined by
/// triangles without area (vertices 4 and 5 repeat the corners next to them)
pub(super) const VERTICES_PER_INSTANCE: u32 = 10;
/// Bytes of the `MaskUniforms` block of the shaders before the layers, see
/// [`MaskFrame::header`].
pub(super) const HEADER_SIZE: usize = 32;
/// Layers drawn at most: as many as fit into the 16 KiB of uniforms WebGL2
/// guarantees. Two per array element in the shaders.
pub(super) const MAX_LAYERS: usize = 2044;
/// Bytes of the `MaskUniforms` block of the shaders (std140 / WGSL layout),
/// the header followed by the layers.
pub(super) const UNIFORM_SIZE: usize = HEADER_SIZE + size_of::<LayerUniform>() * MAX_LAYERS;
const _: () = assert!(UNIFORM_SIZE == 16 * 1024 && MAX_LAYERS.is_multiple_of(2));

/// What the shaders know of a drawn layer, two per element of the layer array
/// of `MaskUniforms`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LayerUniform {
    /// RGBA, not premultiplied. Read as one `u32` with red in the low byte.
    color: [u8; 4],
    /// Where the instances of the layer end. They start where those of the
    /// layer below end, those of the first layer at 0.
    end: u32,
}

/// Draws the masks of a [`State`](crate::State). Type-erased, so the state
/// isn't generic over its backend.
pub(crate) trait MaskPainter {
    /// Puts the masks into `slot`, a shape reserved right above the image.
    fn paint(
        &mut self,
        painter: &egui::Painter,
        slot: ShapeIdx,
        image_rect: Rect,
        masks: &MaskImage,
    );
}

/// What the backends upload when the layers change.
#[derive(Default)]
struct LayerData {
    /// Those of each drawn layer one after the other, in stack order
    instances: Vec<Instance>,
    /// Each drawn layer, in stack order. The index is the layer as the shaders
    /// know it. At most [`MAX_LAYERS`], none without instances.
    layers: Vec<LayerUniform>,
}

pub(crate) struct GpuMasks<B: GpuBackend> {
    buffer: Arc<B::MaskBuffer>,
    /// Built for the [`MaskImage::revision`] `revision`. Shared with the
    /// frames, see [`GpuMasks::update_data`].
    data: Arc<LayerData>,
    /// The stack index of each drawn layer, by its index in the shaders
    stack_indices: Vec<usize>,
    revision: Option<u64>,
}

impl<B: GpuBackend> GpuMasks<B> {
    pub(crate) fn new() -> Self {
        Self {
            buffer: Arc::default(),
            data: Arc::default(),
            stack_indices: Vec::new(),
            revision: None,
        }
    }

    /// Rebuilds the instances and layers if the layers changed. The frames of
    /// earlier paints are usually dropped by then, so the allocations are
    /// reused. Otherwise those frames keep the data they were made with.
    fn update_data(&mut self, masks: &MaskImage) {
        let revision = masks.revision();
        if self.revision == Some(revision) {
            return;
        }
        self.revision = Some(revision);
        self.stack_indices.clear();
        if let Some(data) = Arc::get_mut(&mut self.data) {
            data.instances.clear();
            data.layers.clear();
            build_layers(masks, data, &mut self.stack_indices);
        } else {
            let mut data = LayerData {
                instances: Vec::with_capacity(self.data.instances.len()),
                layers: Vec::with_capacity(self.data.layers.len()),
            };
            build_layers(masks, &mut data, &mut self.stack_indices);
            self.data = Arc::new(data);
        }
    }

    /// The active layer as the shaders know it, `u32::MAX` if it isn't drawn.
    fn active_layer(&self, masks: &MaskImage) -> u32 {
        masks
            .active_layer()
            .and_then(|active| self.stack_indices.binary_search(&active).ok())
            // Below MAX_LAYERS
            .map_or(u32::MAX, |index| index as u32)
    }
}

impl<B: GpuBackend> MaskPainter for GpuMasks<B> {
    fn paint(
        &mut self,
        painter: &egui::Painter,
        slot: ShapeIdx,
        image_rect: Rect,
        masks: &MaskImage,
    ) {
        if !masks.is_visible() {
            return;
        }
        self.update_data(masks);
        // The backend's viewport: no larger than needed, as zooming in makes
        // the image rect huge.
        let rect = painter.clip_rect().intersect(image_rect);
        if self.data.instances.is_empty() || !rect.is_positive() {
            return;
        }
        let [width, height] = masks.size();
        let settings = masks.settings();
        let frame = MaskFrame {
            revision: masks.revision(),
            data: self.data.clone(),
            rect,
            image_rect,
            image_size: [width as f32, height as f32],
            active_layer: self.active_layer(masks),
            opacity: [settings.default_opacity, settings.active_opacity],
        };
        painter.set(slot, B::mask_callback(self.buffer.clone(), frame));
    }
}

/// Appends the instances and uniforms of all visible layers in stack order,
/// and to `stack_indices` the stack index of each.
fn build_layers(masks: &MaskImage, data: &mut LayerData, stack_indices: &mut Vec<usize>) {
    let [width, height] = masks.size();
    let (Some(row), Ok(rows)) = (
        u16::try_from(width)
            .ok()
            .and_then(|width| NonZeroU32::new(width.into())),
        u16::try_from(height),
    ) else {
        log::error!("Masks of {width}x{height} pixels can't be drawn");
        return;
    };
    // Fits, 65535² < 2³²
    let pixel_count = row.get() * u32::from(rows);
    for (layer, area) in masks.subgroups_stack().iter() {
        if area.color[3] == 0 {
            continue;
        }
        let first = data.instances.len();
        // The first run of the next instance
        let mut unpaired = None;
        for Range { start, end } in area.pixels.iter_global_with::<Range<u32>>(row) {
            // Ranges outside of the image would be drawn next to it
            let end = end.min(pixel_count);
            if start >= end {
                log::warn!("Layer {layer} has pixels outside of the {width}x{height} image");
                break;
            }
            for run in range_runs(start, end, row.get()) {
                match unpaired.take() {
                    Some(previous) => data.instances.push(Instance::new(previous, Some(run))),
                    None => unpaired = Some(run),
                }
            }
        }
        if let Some(run) = unpaired {
            data.instances.push(Instance::new(run, None));
        }
        if data.instances.len() == first {
            continue;
        }
        if data.layers.len() == MAX_LAYERS {
            data.instances.truncate(first);
            log::error!("Only {MAX_LAYERS} layers can be drawn, layer {layer} and above aren't");
            break;
        }
        data.layers.push(LayerUniform {
            color: area.color,
            end: data.instances.len() as u32,
        });
        stack_indices.push(layer);
    }
}

/// The runs `[x0, x1, y]` covering the pixels `start..end` (indices
/// `y * width + x`, `start < end`), one per row, `x1` exclusive.
fn range_runs(start: u32, end: u32, width: u32) -> impl Iterator<Item = [u32; 3]> {
    (start / width..=(end - 1) / width).map(move |y| {
        let row = y * width;
        [start.max(row) - row, end.min(row + width) - row, y]
    })
}

/// What a backend needs to draw the masks of one frame.
pub struct MaskFrame {
    /// [`MaskImage::revision`] of the instances and layers: upload them only
    /// if it changed.
    pub(super) revision: u64,
    data: Arc<LayerData>,
    /// Where to draw, the paint callback's rect: the visible part of the image.
    pub(super) rect: Rect,
    /// Where the whole image is, in points.
    image_rect: Rect,
    /// In image pixels
    image_size: [f32; 2],
    /// Drawn with the active opacity, as the shaders know it. `u32::MAX` for
    /// none.
    active_layer: u32,
    /// Default and active opacity
    opacity: [u8; 2],
}

impl MaskFrame {
    /// See [`Instance`], never empty. Native endian: the GPU reads them like
    /// the CPU writes them.
    pub(super) fn instances(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data.instances)
    }

    /// The instances of all layers, drawn by one draw call of
    /// [`VERTICES_PER_INSTANCE`] vertices each.
    pub(super) fn instance_count(&self) -> u32 {
        self.data.instances.len() as u32
    }

    /// The layers of the `MaskUniforms` block, at [`HEADER_SIZE`]: only those
    /// drawn, the shaders read no others. See [`LayerUniform`].
    pub(super) fn layers(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data.layers)
    }

    /// The `MaskUniforms` block of the shaders up to the layers, for the
    /// viewport the backend draws into (callback rect, rounded to pixels and
    /// clamped to the screen by egui). `None` if it has no area.
    pub(super) fn header(
        &self,
        viewport: &ViewportInPixels,
        pixels_per_point: f32,
    ) -> Option<[u8; HEADER_SIZE]> {
        if viewport.width_px <= 0 || viewport.height_px <= 0 {
            return None;
        }
        let viewport_size = [viewport.width_px as f32, viewport.height_px as f32];
        // Image in pixels of the render target, relative to the viewport
        let image_min = [
            self.image_rect.min.x * pixels_per_point - viewport.left_px as f32,
            self.image_rect.min.y * pixels_per_point - viewport.top_px as f32,
        ];
        let image_pixel = [
            self.image_rect.width() * pixels_per_point / self.image_size[0],
            self.image_rect.height() * pixels_per_point / self.image_size[1],
        ];
        // Clip space: x to the right and y upwards, both in -1..1
        let scale = [
            2.0 * image_pixel[0] / viewport_size[0],
            -2.0 * image_pixel[1] / viewport_size[1],
        ];
        let offset = [
            2.0 * image_min[0] / viewport_size[0] - 1.0,
            1.0 - 2.0 * image_min[1] / viewport_size[1],
        ];
        let words: [u32; HEADER_SIZE / 4] = [
            scale[0].to_bits(),
            scale[1].to_bits(),
            offset[0].to_bits(),
            offset[1].to_bits(),
            self.active_layer,
            (f32::from(self.opacity[0]) / 255.0).to_bits(),
            (f32::from(self.opacity[1]) / 255.0).to_bits(),
            // At most MAX_LAYERS. The layers start right after, 16 byte aligned
            self.data.layers.len() as u32,
        ];
        Some(bytemuck::cast(words))
    }
}

#[cfg(test)]
mod tests {
    use egui::{pos2, vec2};
    use imask::{ImaskSet, Roi, SortedRanges, Span};

    use super::*;
    use crate::{History, MaskDefaultActions, PixelArea, PixelAreaStack};

    #[test]
    fn range_runs_cover_exactly_the_range() {
        for width in 1..5 {
            for start in 0..20 {
                for end in start + 1..21 {
                    let mut covered = vec![false; 25];
                    for [x0, x1, y] in range_runs(start, end, width) {
                        assert!(
                            x0 < x1 && x1 <= width,
                            "{start}..{end} in {width}: {x0}..{x1}"
                        );
                        for x in x0..x1 {
                            let index = (y * width + x) as usize;
                            assert!(!covered[index], "{start}..{end} in {width}: {x},{y} twice");
                            covered[index] = true;
                        }
                    }
                    let expected: Vec<_> = (0..25).map(|i| (start..end).contains(&i)).collect();
                    assert_eq!(covered, expected, "{start}..{end} in width {width}");
                }
            }
        }
    }

    /// The vertices of the shaders: both runs exactly, the triangles joining
    /// them without area.
    #[test]
    fn the_strip_draws_each_run_once() {
        // Runs in different rows and columns, so any wrong corner has an area
        let runs = [[2, 5, 1], [7, 9, 4]];
        // As in `mask.vert` and `mask.wgsl`
        let corner = |vertex: u32| {
            let second = vertex >= 5;
            let corner = if second {
                vertex.max(6) - 6
            } else {
                vertex.min(3)
            };
            let [x0, x1, y] = runs[usize::from(second)];
            let x = if corner & 1 == 0 { x0 } else { x1 };
            let y = if corner & 2 == 0 { y } else { y + 1 };
            [x as f32, y as f32]
        };
        let corners: Vec<_> = (0..VERTICES_PER_INSTANCE).map(corner).collect();
        let areas: Vec<f32> = corners
            .windows(3)
            .map(|t| {
                let ([ax, ay], [bx, by], [cx, cy]) = (t[0], t[1], t[2]);
                ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs() / 2.0
            })
            .collect();
        // Each run is 1 pixel high: two triangles of half its width each
        assert_eq!(areas, [1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            corners[..4],
            [[2.0, 1.0], [5.0, 1.0], [2.0, 2.0], [5.0, 2.0]]
        );
        assert_eq!(
            corners[6..],
            [[7.0, 4.0], [9.0, 4.0], [7.0, 5.0], [9.0, 5.0]]
        );
    }

    /// The layer of each instance as the shaders find it: the first one
    /// ending after it.
    #[test]
    fn instances_find_their_layer() {
        // As in `mask.vert` and `mask.wgsl`
        let layer_of = |instance: u32, layers: &[LayerUniform]| {
            let (mut low, mut high) = (0, layers.len() - 1);
            while low < high {
                let middle = (low + high) / 2;
                if layers[middle].end > instance {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            low
        };
        // Every count of up to 9 layers, with 1 to 3 instances each
        for count in 1..10 {
            for seed in 0..27 {
                let mut end = 0;
                let layers: Vec<_> = (0..count)
                    .map(|layer| {
                        end += 1 + (seed + layer) % 3;
                        LayerUniform {
                            color: [255; 4],
                            end,
                        }
                    })
                    .collect();
                let mut start = 0;
                for (layer, uniform) in layers.iter().enumerate() {
                    for instance in start..uniform.end {
                        assert_eq!(layer_of(instance, &layers), layer, "{layers:?}");
                    }
                    start = uniform.end;
                }
            }
        }
    }

    #[test]
    fn shaders_match_the_constants() {
        let elements = MAX_LAYERS / 2;
        let vert = include_str!("shader/mask.vert");
        assert!(vert.contains(&format!("u_layers[{elements}]")));
        let wgsl = include_str!("shader/mask.wgsl");
        assert!(wgsl.contains(&format!("array<vec4<u32>, {elements}>")));
    }

    #[test]
    fn instances_follow_the_stack_and_skip_transparent_layers() {
        let [red, blue] = [[255, 0, 0, 255], [0, 0, 255, 128]];
        let stack = PixelAreaStack::from(vec![
            PixelArea {
                pixels: SortedRanges::from(Span::new(1..3, 0)),
                color: red,
            },
            PixelArea {
                pixels: SortedRanges::from(Span::new(4..6, 0)),
                color: [0, 255, 0, 0],
            },
            // On top, one range over rows 2 and 3
            PixelArea {
                pixels: Roi::new(0..10, 2..4).into(),
                color: blue,
            },
        ]);
        let masks = MaskImage::new([10, 10], stack, History::default());

        let (mut data, mut stack_indices) = (LayerData::default(), Vec::new());
        build_layers(&masks, &mut data, &mut stack_indices);
        assert_eq!(
            data.instances,
            [
                Instance {
                    x: [[1, 3], [0, 0]],
                    y: [0, 0],
                },
                Instance {
                    x: [[0, 10], [0, 10]],
                    y: [2, 3],
                },
            ]
        );
        assert_eq!(
            data.layers,
            [
                LayerUniform { color: red, end: 1 },
                LayerUniform {
                    color: blue,
                    end: 2
                },
            ]
        );
        assert_eq!(stack_indices, [0, 2]);
    }

    #[test]
    fn runs_pair_up_within_their_layer() {
        let [width, height] = [10, 40].map(|v| NonZeroU32::new(v).expect("Not zero"));
        let spans = [
            Span::new(1u32..3, 0),
            Span::new(2..4, 16),
            Span::new(0..5, 31),
        ];
        let pixels = SortedRanges::try_from_span_iter(spans.with_bounds(width, height))
            .expect("Spans are sorted and within bounds");
        let stack = PixelAreaStack::from(vec![
            PixelArea {
                pixels,
                color: [255; 4],
            },
            PixelArea {
                pixels: SortedRanges::from(Span::new(6..7, 5)),
                color: [255; 4],
            },
        ]);
        let masks = MaskImage::new([10, 40], stack, History::default());

        let (mut data, mut stack_indices) = (LayerData::default(), Vec::new());
        build_layers(&masks, &mut data, &mut stack_indices);
        assert_eq!(
            data.instances,
            [
                Instance {
                    x: [[1, 3], [2, 4]],
                    y: [0, 16],
                },
                // The odd ones out of each layer, with an empty run
                Instance {
                    x: [[0, 5], [0, 0]],
                    y: [31, 0],
                },
                Instance {
                    x: [[6, 7], [0, 0]],
                    y: [5, 0],
                },
            ]
        );
        let ends: Vec<_> = data.layers.iter().map(|layer| layer.end).collect();
        assert_eq!(ends, [2, 3]);
    }

    #[test]
    fn layers_beyond_the_uniforms_are_not_drawn() {
        let stack = PixelAreaStack::from(
            (0..MAX_LAYERS + 2)
                .map(|_| PixelArea {
                    pixels: SortedRanges::from(Span::new(1..3, 0)),
                    color: [255; 4],
                })
                .collect::<Vec<_>>(),
        );
        let masks = MaskImage::new([10, 10], stack, History::default());

        let (mut data, mut stack_indices) = (LayerData::default(), Vec::new());
        build_layers(&masks, &mut data, &mut stack_indices);
        assert_eq!(data.layers.len(), MAX_LAYERS);
        assert_eq!(data.instances.len(), MAX_LAYERS);
        assert_eq!(
            data.layers.last().map(|layer| layer.end),
            Some(MAX_LAYERS as u32)
        );
        assert_eq!(stack_indices.last(), Some(&(MAX_LAYERS - 1)));
    }

    #[test]
    fn the_active_layer_is_the_index_of_its_color() {
        let area = |pixels, alpha| PixelArea {
            pixels,
            color: [255, 255, 255, alpha],
        };
        let stack = PixelAreaStack::from(vec![
            area(SortedRanges::from(Span::new(1..3, 0)), 255),
            area(SortedRanges::from(Span::new(4..6, 0)), 0),
            area(Roi::new(2..4, 2..4).into(), 255),
        ]);
        let mut masks = MaskImage::new([10, 10], stack, History::default());
        let mut gpu = GpuMasks::<crate::gpu::Headless>::new();
        gpu.update_data(&masks);

        let mut active = |layer| {
            masks.set_active_layer(layer);
            gpu.active_layer(&masks)
        };
        assert_eq!(active(Some(2)), 1);
        assert_eq!(active(Some(0)), 0);
        assert_eq!(active(Some(1)), u32::MAX, "Transparent, so not drawn");
        assert_eq!(active(Some(3)), u32::MAX);
        assert_eq!(active(None), u32::MAX);
    }

    #[test]
    fn rebuilt_data_reuses_its_allocation_but_spares_unpainted_frames() {
        let masks_up_to = |end| {
            let stack = PixelAreaStack::from(vec![PixelArea {
                pixels: SortedRanges::from(Span::new(0..end, 0)),
                color: [255; 4],
            }]);
            MaskImage::new([10, 10], stack, History::default())
        };
        let end_of_first = |data: &LayerData| data.instances[0].x[0][1];
        let mut gpu = GpuMasks::<crate::gpu::Headless>::new();
        gpu.update_data(&masks_up_to(2));

        // Held by a frame which isn't painted yet
        let frame = gpu.data.clone();
        gpu.update_data(&masks_up_to(3));
        assert_eq!(end_of_first(&frame), 2, "The frame keeps what it shows");
        assert_eq!(end_of_first(&gpu.data), 3);

        let allocation = gpu.data.instances.as_ptr();
        gpu.update_data(&masks_up_to(4));
        assert_eq!(end_of_first(&gpu.data), 4);
        assert_eq!(gpu.data.instances.as_ptr(), allocation, "Rebuilt in place");
    }

    #[test]
    fn hover_keeps_the_instances_but_edits_rebuild_them() {
        let mut masks = MaskImage::new([10, 10], PixelAreaStack::default(), History::default());
        let revision = masks.revision();
        masks.set_active_layer(Some(0));
        assert_eq!(masks.revision(), revision);
        masks.add(SortedRanges::from(Span::new(1..3, 0)));
        assert_ne!(masks.revision(), revision);
    }

    #[test]
    fn header_maps_the_image_onto_its_rect() {
        let image_rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(100.0, 50.0));
        let layer = LayerUniform {
            color: [255; 4],
            end: 1,
        };
        let frame = MaskFrame {
            revision: 0,
            data: Arc::new(LayerData {
                instances: Vec::new(),
                layers: vec![layer; 2],
            }),
            rect: image_rect,
            image_rect,
            image_size: [20.0, 10.0],
            active_layer: 3,
            opacity: [0, 255],
        };
        // Viewport of a callback covering the image, with 2 pixels per point
        let viewport = ViewportInPixels {
            left_px: 20,
            top_px: 40,
            from_bottom_px: 0,
            width_px: 200,
            height_px: 100,
        };
        let header: [u32; HEADER_SIZE / 4] =
            bytemuck::cast(frame.header(&viewport, 2.0).expect("Viewport has an area"));
        let [scale_x, scale_y, offset_x, offset_y] =
            [0, 1, 2, 3].map(|i| f32::from_bits(header[i]));
        let clip = |x: f32, y: f32| (x * scale_x + offset_x, y * scale_y + offset_y);
        assert_eq!(clip(0.0, 0.0), (-1.0, 1.0));
        assert_eq!(clip(20.0, 10.0), (1.0, -1.0));
        assert_eq!(header[4], 3);
        assert_eq!(f32::from_bits(header[5]), 0.0);
        assert_eq!(f32::from_bits(header[6]), 1.0);
        assert_eq!(header[7], 2, "Layer count");

        let empty = ViewportInPixels {
            width_px: 0,
            ..viewport
        };
        assert_eq!(frame.header(&empty, 2.0), None);
    }
}

// Masks drawn straight from their ranges, see `gpu/mask.rs`. Keep in sync
// with `mask.vert`.

struct MaskUniforms {
    // Image pixel -> clip space: clip = pixel * scale + offset
    scale: vec2<f32>,
    offset: vec2<f32>,
    // Layer drawn with `active_opacity`, the others with `default_opacity`
    active_layer: u32,
    default_opacity: f32,
    active_opacity: f32,
    // Layers drawn, at least one
    layer_count: u32,
    // Of each layer, see `LayerUniform` in `gpu/mask.rs`: RGBA, not
    // premultiplied, red in the low byte, and where its instances end. Two
    // per element, uniform arrays have a stride of 16 bytes.
    // 1022 = MAX_LAYERS / 2
    layers: array<vec4<u32>, 1022>,
}

@group(0) @binding(0) var<uniform> uniforms: MaskUniforms;

// Color and end of the instances of layer `index`
fn layer_at(index: u32) -> vec2<u32> {
    let pair = uniforms.layers[index >> 1u];
    return select(pair.zw, pair.xy, (index & 1u) == 0u);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Premultiplied, in gamma space like everything egui draws
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
    // Two runs of a layer per instance, see `Instance` in `gpu/mask.rs`:
    // x0, x1 (exclusive) of the first and of the second run, in image pixels
    @location(0) runs_x: vec4<u32>,
    // Rows of the first and of the second run
    @location(1) runs_y: vec2<u32>,
) -> VertexOutput {
    // One draw call for all layers: the instances of each layer follow those
    // of the layer below. The layer of this instance is the first one ending
    // after it.
    var low = 0u;
    var high = uniforms.layer_count - 1u;
    while (low < high) {
        let middle = (low + high) / 2u;
        if (layer_at(middle).y > instance) {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }
    let layer = low;

    // One triangle strip: vertices 0-3 are the first run, 4 repeats its last
    // corner and 5 the first corner of the second run, which is 6-9. The
    // triangles joining both have no area, so nothing is drawn twice.
    let second = vertex >= 5u;
    // (x0, y) (x1, y) (x0, y + 1) (x1, y + 1)
    let corner = select(min(vertex, 3u), max(vertex, 6u) - 6u, second);
    let run = select(runs_x.xy, runs_x.zw, second);
    let row = select(runs_y.x, runs_y.y, second);
    let x = select(run.x, run.y, (corner & 1u) != 0u);
    let y = select(row, row + 1u, (corner & 2u) != 0u);

    let rgba = layer_at(layer).x;
    let color = vec4<f32>((vec4<u32>(rgba) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    let opacity = select(uniforms.default_opacity, uniforms.active_opacity, layer == uniforms.active_layer);
    let alpha = color.a * opacity;

    var output: VertexOutput;
    output.position = vec4<f32>(vec2<f32>(f32(x), f32(y)) * uniforms.scale + uniforms.offset, 0.0, 1.0);
    output.color = vec4<f32>(color.rgb * alpha, alpha);
    return output;
}

@fragment
fn fs_main_gamma_framebuffer(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// For sRGB render targets, which convert to gamma themselves (as egui does).
// Blending then happens in linear space, like for egui's own shapes.
@fragment
fn fs_main_linear_framebuffer(input: VertexOutput) -> @location(0) vec4<f32> {
    let cutoff = input.color.rgb < vec3<f32>(0.04045);
    let lower = input.color.rgb / vec3<f32>(12.92);
    let higher = pow((input.color.rgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return vec4<f32>(select(higher, lower, cutoff), input.color.a);
}

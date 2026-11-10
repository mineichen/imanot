// Masks drawn straight from their ranges, see `gpu/mask.rs`. Keep in sync
// with `mask.wgsl`. The `#version` line is prepended at runtime.

layout(std140) uniform MaskUniforms {
    // Image pixel -> clip space: clip = pixel * u_scale + u_offset
    vec2 u_scale;
    vec2 u_offset;
    // Layer drawn with `u_active_opacity`, the others with `u_default_opacity`
    uint u_active_layer;
    float u_default_opacity;
    float u_active_opacity;
    // Layers drawn, at least one
    uint u_layer_count;
    // Of each layer, see `LayerUniform` in `gpu/mask.rs`: RGBA, not
    // premultiplied, red in the low byte, and where its instances end. Two
    // per element, std140 pads each to 16 bytes. 1022 = MAX_LAYERS / 2
    uvec4 u_layers[1022];
};

// Two runs of a layer per instance, see `Instance` in `gpu/mask.rs`:
// x0, x1 (exclusive) of the first and of the second run, in image pixels
in uvec4 a_runs_x;
// Rows of the first and of the second run
in uvec2 a_runs_y;

out vec4 v_color;

// Color and end of the instances of layer `index`
uvec2 layer_at(uint index) {
    uvec4 pair = u_layers[index >> 1u];
    return (index & 1u) == 0u ? pair.xy : pair.zw;
}

void main() {
    // One draw call for all layers: the instances of each layer follow those
    // of the layer below. The layer of this instance is the first one ending
    // after it.
    uint instance = uint(gl_InstanceID);
    uint low = 0u;
    uint high = u_layer_count - 1u;
    while (low < high) {
        uint middle = (low + high) / 2u;
        if (layer_at(middle).y > instance) {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }
    uint layer = low;

    // One triangle strip: vertices 0-3 are the first run, 4 repeats its last
    // corner and 5 the first corner of the second run, which is 6-9. The
    // triangles joining both have no area, so nothing is drawn twice.
    uint vertex = uint(gl_VertexID);
    bool second = vertex >= 5u;
    // (x0, y) (x1, y) (x0, y + 1) (x1, y + 1)
    uint corner = second ? max(vertex, 6u) - 6u : min(vertex, 3u);
    uvec2 run = second ? a_runs_x.zw : a_runs_x.xy;
    uint row = second ? a_runs_y.y : a_runs_y.x;
    uint x = (corner & 1u) == 0u ? run.x : run.y;
    uint y = (corner & 2u) == 0u ? row : row + 1u;
    gl_Position = vec4(vec2(x, y) * u_scale + u_offset, 0.0, 1.0);

    uint rgba = layer_at(layer).x;
    vec4 color = vec4((uvec4(rgba) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u)) / 255.0;
    float alpha = color.a * (layer == u_active_layer ? u_active_opacity : u_default_opacity);
    // Premultiplied, in gamma space like everything egui draws
    v_color = vec4(color.rgb * alpha, alpha);
}

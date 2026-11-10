// See `mask.vert`. The `#version` line is prepended at runtime.

in vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}

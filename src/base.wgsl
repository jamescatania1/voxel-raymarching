struct Camera {
    view_proj_matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Model {
    matrix: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> model: Model;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) color: vec3<f32>,
}


@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.Position = camera.view_proj_matrix * model.matrix * position;
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}

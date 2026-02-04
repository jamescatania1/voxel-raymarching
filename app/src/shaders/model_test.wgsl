struct Camera {
    view_proj_matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
}
@group(1) @binding(0) var<uniform> model: Model;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) color: vec3<f32>,
}


@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    // out.Position = camera.view_proj_matrix *  vec4(position, 1.0);

    let albedo = vec3(1.0);
    let ws_normal = normalize(model.normal_matrix * normal);


    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));
    let diff = max(dot(normal, ws_light_dir), 0.0);
    let color = albedo * (diff * 0.8 + 0.2);

    var out: VertexOutput;
    out.Position = camera.view_proj_matrix * model.matrix * vec4(position, 1.0);
    // out.color = ws_normal;
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}

struct VoxelSceneMetadata {
    size: vec3<u32>,
    bounding_size: u32,
    probe_size: vec3<u32>,
    index_levels: u32,
    index_chunk_count: u32,
}
@group(0) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(0) @binding(1) var tex_depth: texture_2d<f32>;

struct PostFxSettings {
    fxaa_enabled: u32,
    exposure: f32,
    tonemapping: u32,
}
struct Environment {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    shadow_bias: f32,
    skybox_rotation: vec2<f32>,
    camera: Camera,
    prev_camera: Camera,
    shadow_spread: f32,
    filter_shadows: u32,
    shadow_filter_radius: f32,
    max_ambient_distance: u32,
    smooth_normal_factor: f32,
    roughness_multiplier: f32,
    indirect_sky_intensity: f32,
    debug_view: u32,
}
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    jitter: vec2<f32>,
    far: f32,
    fov: f32,
}
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
    inv_normal_transform: mat3x3<f32>,
}
@group(1) @binding(0) var<uniform> environment: Environment;
@group(1) @binding(1) var<uniform> frame: FrameMetadata;
@group(1) @binding(2) var<uniform> model: Model;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) depth: f32,
}

const PROBE_SCALE: f32 = 32.0;
const DOT_SIZE: f32 = 6.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(tex_depth).xy);
    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );

    let probe_grid = vec3<u32>(
        instance_id % scene.probe_size.x,
        (instance_id / scene.probe_size.x) % scene.probe_size.y,
        instance_id / (scene.probe_size.x * scene.probe_size.y),
    );
    let probe_pos = (vec3<f32>(probe_grid) + 0.5) * PROBE_SCALE;

    let world_pos = model.transform * vec4(probe_pos, 1.0);
    let clip = environment.camera.view_proj * world_pos;

    let pos = corners[vertex_id] * DOT_SIZE * texel_size / clamp(clip.w, 0.15, 0.5);

    var color = vec3<f32>(probe_grid % 10) / 10.0;
    color.r = 1.0;

    let cam_local = (model.inv_transform * vec4(environment.camera.ws_position, 1.0)).xyz;

    var out: VertexOutput;
    out.position = clip + vec4(pos, 0.0, 0.0);
    out.color = color;
    out.depth = length(probe_pos.xyz - cam_local);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureLoad(tex_depth, vec2<u32>(in.position.xy), 0).r;
    if in.depth > depth + 4.5 {
        discard;
    }
    return vec4(in.color * 3.0, 1.0);
}
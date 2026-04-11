struct ChunkLightingSample {
    // for irradiance
    m1: u32,
    m2: u32,
    // pad (2) | shadow sum (10) | shadow samples (10) | irradiance samples (10)
    shadow_counts: u32,
}
struct ChunkLighting {
    m1: f32,
    m2: f32,
    history_length: u32,
}
@group(0) @binding(0) var<storage, read> visible_chunks: array<u32>;
@group(0) @binding(1) var<storage, read> cur_chunk_lighting: array<ChunkLightingSample>;
@group(0) @binding(2) var<storage, read_write> acc_chunk_lighting: array<ChunkLighting>;

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

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const MAX_HISTORY_LENGTH: u32 = 32;
const MIN_MOMENTS_ALPHA: f32 = 0.2;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let ci = visible_chunks[in.id.x];
    if ci == 0u {
        return;
    }
    let cur_packed = cur_chunk_lighting[ci];

    let cur_count = (cur_packed.shadow_counts >> 10u) & 0x3FFu;
    let cur_samples = (cur_packed.shadow_counts >> 20u) & 0x3FFu;
    if cur_count == 0 {
        return;
    }
    let cur_shadow = f32(cur_samples) / f32(cur_count);

    let m1 = cur_shadow;
    let m2 = cur_shadow * cur_shadow;

    var acc = acc_chunk_lighting[ci];

    acc.history_length = min(MAX_HISTORY_LENGTH, acc.history_length + 1);
    let alpha = max(MIN_MOMENTS_ALPHA, 1.0 / f32(acc.history_length));

    acc.m1 = mix(acc.m1, m1, alpha);
    acc.m2 = mix(acc.m2, m2, alpha);

    acc_chunk_lighting[ci] = acc;
}
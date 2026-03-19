@group(0) @binding(0) var<storage, read> voxel_map: array<u32>; // voxel hashmap, two words (key, value) per entry
@group(0) @binding(1) var<storage, read> cur_voxel_lighting: array<u32>;
@group(0) @binding(2) var<storage, read_write> acc_voxel_lighting: array<u32>;

struct Environment {
    sun_direction: vec3<f32>,
    shadow_bias: f32,
    camera: Camera,
    prev_camera: Camera,
    shadow_spread: f32,
    filter_shadows: u32,
    shadow_filter_radius: f32,
    max_ambient_distance: u32,
    smooth_normal_factor: f32,
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
    fxaa_enabled: u32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
    inv_normal_transform: mat3x3<f32>,
}
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

const MAX_HISTORY_LENGTH: u32 = 2048u;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let key = voxel_map[in.id.x << 1u];
    if key == 0u {
        return;
    }

    let index = voxel_map[(in.id.x << 1u) | 1u];
    if index == 0u {
        return;
    }

    let cur_shadow = cur_voxel_lighting[index];
    var cur_visible_count = cur_shadow & 0xFFFFu;
    var cur_shadow_count = min(cur_visible_count, cur_shadow >> 16u);

    if cur_visible_count >= MAX_HISTORY_LENGTH {
        let diff = cur_visible_count - MAX_HISTORY_LENGTH;
        let shadow = f32(cur_shadow_count) / f32(cur_visible_count);
        cur_shadow_count -= u32(round(shadow * f32(diff)));
        cur_visible_count -= diff;
    }

    let acc = acc_voxel_lighting[key];
    var acc_length = acc & 0xFFFFu;
    var acc_shadow_count = min(acc_length, acc >> 16u);

    if acc_length >= MAX_HISTORY_LENGTH {
        let diff = acc_length - MAX_HISTORY_LENGTH + cur_visible_count;
        let shadow = f32(acc_shadow_count) / f32(acc_length);
        acc_shadow_count -= u32(round(shadow * f32(diff)));
        acc_length -= diff;
    }
    acc_length += cur_visible_count;
    acc_shadow_count += cur_shadow_count;

    acc_voxel_lighting[key] = (acc_shadow_count << 16u) | acc_length;

    // acc_voxel_lighting[key] = acc_voxel_lighting[key] *(1.0 - ACC_ALPHA) + shadow *ACC_ALPHA;

    // cur_voxel_lighting[cur_index] = bitcast<u32>(shadow);
}
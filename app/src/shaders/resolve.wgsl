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

const MAX_HISTORY_LENGTH: u32 = 64u;

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

    let cur = cur_voxel_lighting[index];
    let cur_visible = cur & 0xFFFFu;
    if cur_visible == 0u {
        return;
    }
    let cur_shadow_count = min(cur_visible, cur >> 16u);
    let cur_shadow = f32(cur_shadow_count) / f32(cur_visible);

    let acc = acc_voxel_lighting[key];
    let acc_shadow = f32(acc >> 8u) / 16777215.0;
    let history_len = min(MAX_HISTORY_LENGTH, (acc & 0xFFu) + 1u);

    let alpha = 1.0 / f32(history_len);
    let res_shadow = mix(acc_shadow, cur_shadow, alpha);

    let res = ((u32(res_shadow * 16777215.0) & 0xFFFFFFu) << 8u) | history_len;
    acc_voxel_lighting[key] = res;
    // let res_length = min(MAX_HISTORY_LENGTH, acc_length + cur_length);
    // let res_shadow = f32(cur_shadow_count + acc_shadow_count) / f32(cur_length + acc_length);
    // let res_shadow_count = u32(ceil(res_shadow * f32(res_length)));
    // if acc_length + cur_length > MAX_HISTORY_LENGTH {
    //     return;
    // }

    // let res_length = min(MAX_HISTORY_LENGTH, acc_length + cur_length);


    // if cur_visible_count >= MAX_HISTORY_LENGTH {
    //     let diff = cur_visible_count - MAX_HISTORY_LENGTH;
    //     let shadow = f32(cur_shadow_count) / f32(cur_visible_count);
    //     cur_shadow_count -= u32(round(shadow * f32(diff)));
    //     cur_visible_count -= diff;
    // }


    // if acc_length + cur_visible_count >= MAX_HISTORY_LENGTH {
    //     let diff = acc_length - MAX_HISTORY_LENGTH + cur_visible_count;
    //     let shadow = f32(acc_shadow_count) / f32(acc_length);
    //     acc_shadow_count -= u32(round(shadow * f32(diff)));
    //     acc_length -= diff;
    // }
    // acc_length += cur_visible_count;
    // acc_shadow_count += cur_shadow_count;

}

struct VisibleVoxel {
    data: u32,
    leaf_index: u32,
    pos: array<u32, 2>,
}
struct VoxelLighting {
    irradiance: vec3<f32>,
    shadow: f32,
    variance: f32,
    history_length: u32,
}
// struct ChunkLighting {
//     shadow_m1: f32,
//     shadow_m2: f32,
//     irradiance_m1: f32,
//     irradiance_m2: f32,
// }
@group(0) @binding(0) var<storage, read> visible_voxels: array<VisibleVoxel>;
@group(0) @binding(1) var<storage, read> cur_voxel_lighting: array<array<u32, 3>>;
@group(0) @binding(2) var<storage, read_write> acc_voxel_lighting: array<array<u32, 3>>;

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

const MAX_HISTORY_LENGTH: u32 = 1023u;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let visible = visible_voxels[in.id.x];
    if visible.data == 0u {
        return;
    }

    var cur = unpack_voxel_lighting(cur_voxel_lighting[in.id.x]);
    var acc = unpack_voxel_lighting(acc_voxel_lighting[visible.leaf_index]);

    let cur_luma = combined_luminance(cur);
    let acc_luma = combined_luminance(acc);

    // if acc.history_length == 0u {
    //     acc.history_length = 1u;
    //     acc.shadow = cur.shadow;
    //     acc.irradiance = cur.irradiance;
    //     acc.variance = cur_luma * cur_luma;
    //     acc_voxel_lighting[visible.leaf_index] = pack_voxel_lighting(acc);
    //     return;
    // }

    let moment_alpha = max(1.0 / f32(acc.history_length + 1u), 0.01);
    let m1 = mix(acc_luma, cur_luma, moment_alpha);
    let m2 = mix(acc.variance, cur_luma * cur_luma, moment_alpha);

    let variance = max(m2 - m1 * m1, 0.0);
    let rel_variance = variance / max(m1 * m1, 0.01);

    let prev_luma = acc_luma;
    let max_history_len = u32(mix(512.0, 8.0, saturate(rel_variance * 4.0)));

    // acc.history_length = min(acc.history_length + 1u, max_history_len);
    acc.variance = m2;

    acc.history_length = min(MAX_HISTORY_LENGTH, acc.history_length + 1u);

    let alpha = 1.0 / f32(acc.history_length);
    acc.shadow = mix(acc.shadow, cur.shadow, alpha);
    acc.irradiance = mix(acc.irradiance, cur.irradiance, alpha);

    acc_voxel_lighting[visible.leaf_index] = pack_voxel_lighting(acc);
}

fn combined_luminance(lighting: VoxelLighting) -> f32 {
    // return lighting.shadow + dot(lighting.irradiance, vec3(0.2126, 0.7152, 0.0722));
    return lighting.shadow;
}

fn pack_voxel_lighting(value: VoxelLighting) -> array<u32, 3> {
    return array<u32, 3>(
        pack2x16float(value.irradiance.rg),
        pack2x16float(vec2(value.irradiance.b, value.variance)),
        (u32(saturate(value.shadow) * 65535.0 + 0.9 * f32(frame.frame_id & 1u)) << 16u) | (value.history_length & 0xFFFFu),
    );
}
fn unpack_voxel_lighting(packed: array<u32, 3>) -> VoxelLighting {
    let irr_rg = unpack2x16float(packed[0]);
    let irr_b_variance = unpack2x16float(packed[1]);

    var res: VoxelLighting;
    res.irradiance = vec3(irr_rg, irr_b_variance.x);
    res.variance = irr_b_variance.y;
    res.shadow = f32(packed[2] >> 16u) / 65535.0;
    res.history_length = packed[2] & 0xFFFFu;
    return res;
}
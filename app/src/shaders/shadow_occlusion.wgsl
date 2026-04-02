struct VoxelSceneMetadata {
    size: vec3<u32>,
    bounding_size: u32,
    probe_size: vec3<u32>,
    probe_scale: f32,
    index_levels: u32,
    index_chunk_count: u32,
}
struct IndexChunk {
    child_index: u32,
    mask: array<u32, 2>,
}
@group(0) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(0) @binding(1) var<storage, read> index_chunks: array<IndexChunk>;
@group(0) @binding(2) var<storage, read_write> shadow_mask: array<atomic<u32>>;

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
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) thread_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let light_dir = normalize(model.inv_normal_transform * (-environment.sun_direction));
    let step_dir = vec3<i32>(step(vec3(0.0), light_dir) * 2.0 - 1.0);

    let group_size = max(vec3(2), (scene.size + 3) / 4);
    let size_padded = group_size * 4;
    let base = vec3<f32>(vec3<u32>(max(-step_dir, vec3(0))) * size_padded);

    let face_groups_xy = group_size.x * group_size.y;
    let face_groups_xz = group_size.x * group_size.z;
    let face_groups_yz = group_size.y * group_size.z;

    let group_index = in.group_id.x + in.group_id.y * in.num_workgroups.x;
    let thread_offset = in.thread_id.xy / 2;
    let voxel_offset = (vec2<f32>(in.thread_id.xy & vec2(1u)) + 0.5) * 0.5;

    var pos: vec3<f32>;
    if group_index < face_groups_xy {
        let id = group_index;
        let uv = vec2<f32>(vec2(id % group_size.x, id / group_size.x) * 4 + thread_offset) + voxel_offset;
        pos = vec3(uv, base.z);
    } else if group_index < (face_groups_xy + face_groups_xz) {
        let id = group_index - face_groups_xy;
        let uv = vec2<f32>(vec2(id % group_size.x, id / group_size.x) * 4 + thread_offset) + voxel_offset;
        pos = vec3(uv.x, base.y, uv.y);
    } else {
        let id = group_index - (face_groups_xy + face_groups_xz);
        let uv = vec2<f32>(vec2(id % group_size.y, id / group_size.y) * 4 + thread_offset) + voxel_offset;
        pos = vec3(base.x, uv);
    }
    if any(vec3<u32>(pos) > size_padded) {
        return;
    }

    var ray: Ray;
    ray.origin = pos;
    ray.direction = light_dir;
    raymarch(ray, in.local_index);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn raymarch(ray: Ray, local_index: u32) {
    var dir = ray.direction;
    let inv_dir = -1.0 / max(abs(dir), vec3(1e-8));

    let scale = 1.0 / f32(scene.bounding_size);
    var origin = ray.origin * scale + 1.0;
    origin = mirrored_pos(origin, dir);

    var pos = clamp(origin, vec3(1.0), vec3(1.9999999));

    var mirror_mask = 0u;
    mirror_mask |= select(0u, 3u, dir.x > 0.0) << 0u;
    mirror_mask |= select(0u, 3u, dir.y > 0.0) << 4u;
    mirror_mask |= select(0u, 3u, dir.z > 0.0) << 2u;

    var scale_exp = 21u;
    var side_distance: vec3<f32>;

    var ci = 0u;
    var chunk = index_chunks[ci];
    // var hit_depth = -1.0;

    var i = 0u;
    for (i = 0u; i < 2048; i++) {
        var child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;

        while chunk_contains_child(chunk.mask, child_offset) && !chunk_is_leaf(chunk.child_index) && scale_exp >= 2u {
            stack[local_index][scale_exp >> 1u] = ci;
            ci = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);
            chunk = index_chunks[ci];

            scale_exp -= 2u;
            child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;
        }

        if chunk_contains_child(chunk.mask, child_offset) && chunk_is_leaf(chunk.child_index) {
            pos = mirrored_pos_unchecked(pos, dir);

            let tmax = min(min(side_distance.x, side_distance.y), side_distance.z);

            let leaf_index = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);
            set_shadow_mask(leaf_index);

            return;
        }

        var adv_scale_exp = scale_exp;

        let snapped_idx = child_offset & 0x2Au;
        let sub_chunk_empty = ((chunk.mask[snapped_idx >> 5u] >> (snapped_idx & 31u)) & 0x00330033u) == 0u;
        adv_scale_exp += u32(sub_chunk_empty);

        let edge_pos = floor_scale(pos, adv_scale_exp);

        side_distance = (edge_pos - origin) * inv_dir;
        let t_max = min(min(side_distance.x, side_distance.y), side_distance.z);

        let max_sibling_bounds: vec3<i32> = bitcast<vec3<i32>>(edge_pos) + select(vec3(i32(1u << adv_scale_exp) - 1), vec3(-1), side_distance == vec3(t_max));
        pos = min(origin - abs(dir) * t_max, bitcast<vec3<f32>>(max_sibling_bounds));

        // carry bit tells us which node to go up to
        let diff_pos: vec3<u32> = bitcast<vec3<u32>>(pos) ^ bitcast<vec3<u32>>(edge_pos);
        let diff_exp = firstLeadingBit((diff_pos.x | diff_pos.y | diff_pos.z) & 0xFFAAAAAAu);

        // ascend
        if i32(diff_exp) > i32(scale_exp) {
            scale_exp = diff_exp;
            if diff_exp > 21 {
                break;
            }

            ci = stack[local_index][scale_exp >> 1u];
            chunk = index_chunks[ci];
        }
    }
}

// write unshadowed (1 bit) into shadow occlusion mask
fn set_shadow_mask(leaf_index: u32) {
    atomicOr(&shadow_mask[leaf_index >> 5u], 1u << (leaf_index & 31u));
}

/// ------------------------------------------------------
/// ---------------- tree traversal utils ----------------

fn mirrored_pos(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var mirrored: vec3<f32> = bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) ^ vec3(0x7FFFFFu));

    if any(pos < vec3<f32>(1.0)) || any(pos >= vec3<f32>(2.0)) {
        mirrored = 3.0 - pos;
    }
    return select(pos, mirrored, dir > vec3(0.0));
}

fn mirrored_pos_unchecked(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let mirrored: vec3<f32> = bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) ^ vec3(0x7FFFFFu));
    return select(pos, mirrored, dir > vec3(0.0));
}

/// computes floor(pos / scale) * scale
fn floor_scale(pos: vec3<f32>, scale_exp: u32) -> vec3<f32> {
    let mask = ~0u << scale_exp;
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) & vec3(mask));
}

fn chunk_offset(pos: vec3<f32>, scale_exp: u32) -> u32 {
    let chunk_pos = (bitcast<vec3<u32>>(pos) >> vec3(scale_exp)) & vec3(3u);
    return (chunk_pos.y << 4u) | (chunk_pos.z << 2u) | chunk_pos.x;
}

fn chunk_contains_child(mask: array<u32, 2>, offset: u32) -> bool {
    let half_mask = select(mask[0], mask[1], offset >= 32u);
    return (half_mask & (1u << (offset & 31u))) != 0u;
}

/// given mask and index i, gets packed offset based on count of 1s in mask for 0 <= j < i
fn mask_packed_offset(mask: array<u32, 2>, i: u32) -> u32 {
    return select(
        countOneBits(mask[0]) + countOneBits(mask[1] & ~(0xffffffffu << (i - 32u))),
        countOneBits(mask[0] & ~(0xffffffffu << i)),
        i < 32u
    );
}

fn chunk_is_leaf(child_index: u32) -> bool {
    return (child_index & 1u) == 1u;
}
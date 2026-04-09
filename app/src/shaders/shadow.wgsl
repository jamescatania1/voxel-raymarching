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
struct VisibleVoxel {
    data: u32,
    leaf_index: u32,
    pos: array<u32, 2>,
}
@group(0) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(0) @binding(1) var<storage, read> index_chunks: array<IndexChunk>;
@group(0) @binding(2) var tex_noise: texture_3d<f32>;
@group(0) @binding(3) var sampler_noise: sampler;

@group(0) @binding(4) var<storage, read> visible_voxels: array<VisibleVoxel>;

// current frame per-voxel shadow values
@group(0) @binding(5) var<storage, read_write> voxel_lighting: array<array<u32, 3>>;

struct VisibilityInfo {
    voxel_count: atomic<u32>,
    chunk_count: atomic<u32>,
    failed_to_add: atomic<u32>,
}
struct ChunkLightingSample {
    // for irradiance
    m1: atomic<u32>,
    m2: atomic<u32>,
    // pad (2) | shadow sum (10) | shadow samples (10) | irradiance samples (10)
    shadow_counts: atomic<u32>,
}
@group(0) @binding(6) var<storage, read_write> visibility: VisibilityInfo;
@group(0) @binding(7) var<storage, read_write> chunk_visibility_mask: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> visible_chunks: array<u32>;
@group(0) @binding(9) var<storage, read_write> cur_chunk_lighting: array<ChunkLightingSample>;

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
    @builtin(local_invocation_index) local_index: u32,
}

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let visible = visible_voxels[in.id.x];
    if visible.data == 0u {
        return;
    }

    let voxel = unpack_voxel(visible.data);
    let voxel_pos = unpack_voxel_pos(visible.pos);
    let voxel_center = vec3<f32>(voxel_pos) + 0.5;

    let result = trace_shadow(voxel_pos, voxel_center, in.local_index);
    voxel_lighting[in.id.x] = pack_visible_lighting(result);

    let ci = get_containing_index_chunk_id(voxel_pos);
    atomicAdd(&cur_chunk_lighting[ci].shadow_counts, (u32(result) << 20u) | (1u << 10u));

    // add the chunk to the visible list if not there yet
    // same as the voxels, but simpler since memory is cheap when you divide everything by 64
    if (atomicOr(&chunk_visibility_mask[ci >> 5u], 1u << (ci & 31u)) & (1u << (ci & 31u))) == 0 {
        let i = atomicAdd(&visibility.chunk_count, 1u);
        visible_chunks[i] = ci;
    }
}

fn pack_visible_lighting(shadow: bool) -> array<u32, 3> {
    return array<u32, 3>(
        0u,
        0u,
        u32(shadow),
    );
}

// gets the first parent 4x4x4 index chunk's id for the given voxel position
// if this is slow we can store it explicitly in the future
fn get_containing_index_chunk_id(voxel_pos: vec3<u32>) -> u32 {
    var ci = 0u;
    var chunk = index_chunks[ci];
    var scale_exp = scene.index_levels * 2u - 2u;

    while scale_exp > 0u {
        let cx = (voxel_pos.x >> scale_exp) & 3u;
        let cy = (voxel_pos.y >> scale_exp) & 3u;
        let cz = (voxel_pos.z >> scale_exp) & 3u;

        let child_offset = cx | (cz << 2u) | (cy << 4u);

        if chunk_is_leaf(chunk.child_index) {
            // return ci;
            break;
        }
        let next = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);

        ci = next;
        chunk = index_chunks[ci];
        scale_exp -= 2u;
    }

    return ci;
    // return 0; // won't happen
}

fn trace_shadow(voxel_pos: vec3<u32>, ls_pos: vec3<f32>, local_index: u32) -> bool {
    let light_dir = normalize(model.inv_normal_transform * environment.sun_direction);

    let light_tangent = normalize(cross(light_dir, vec3(0.0, 0.0, 1.0)));
    let light_bitangent = normalize(cross(light_tangent, light_dir));

    var dir = blue_noise(voxel_pos);
    dir = align_direction(dir, light_dir);

    var ray: Ray;
    ray.origin = ls_pos + dir * environment.shadow_bias;
    ray.direction = dir;

    return raymarch_shadow(ray, local_index);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn raymarch_shadow(ray: Ray, local_index: u32) -> bool {
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

    var stack: array<u32, 11>;
    var ci = 0u;
    var chunk = index_chunks[ci];
    var skip_next_hit = true;

    var i = 0u;
    for (i = 0u; i < 256; i++) {
        var child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;

        while chunk_contains_child(chunk.mask, child_offset) && !chunk_is_leaf(chunk.child_index) && scale_exp >= 2u {
            stack[scale_exp >> 1u] = ci;
            ci = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);
            chunk = index_chunks[ci];

            scale_exp -= 2u;
            child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;
        }

        // if we reached a leaf, we're done
        if !skip_next_hit && chunk_contains_child(chunk.mask, child_offset) && chunk_is_leaf(chunk.child_index) {
            return true;
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

            ci = stack[scale_exp >> 1u];
            chunk = index_chunks[ci];
        }
        skip_next_hit = false;
    }

    return false;
}

fn blue_noise(voxel_pos: vec3<u32>) -> vec3<f32> {
    let x = (voxel_pos.x ^ (voxel_pos.z << 4)) & 127;
    let y = (voxel_pos.y ^ (voxel_pos.z >> 4)) & 127;
    let z = frame.frame_id & 63u;
    let dir = textureLoad(tex_noise, vec3(x, y, z), 0).rgb * 2.0 - 1.0;

    let s = 1.0 - environment.shadow_spread;
    let u_1 = 1.0 - dir.z * dir.z;
    let len = sqrt(max(1e-8, u_1));

    let cos_t = 1.0 - u_1 * environment.shadow_spread;
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let cos_phi = dir.x / len;
    let sin_phi = dir.y / len;

    return vec3(sin_t * cos_phi, sin_t * sin_phi, cos_t);
}

fn rand_hemisphere_direction(noise: vec2<f32>, spread: f32) -> vec3<f32> {
    let xy = (noise * 2.0 - 1.0) * spread;
    let z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
    return vec3(xy, z);
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

fn primary_ray(uv: vec2<f32>) -> Ray {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = environment.camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = environment.camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    var ray: Ray;
    ray.origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
}

// aligns dir to n's tangent space
fn align_direction(dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var tangent = vec3<f32>(0.0);
    var bitangent = vec3<f32>(0.0);
    if n.z < 0.0 {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        tangent = vec3(1.0 - n.x * n.x * a, -b, n.x);
        bitangent = vec3(b, n.y * n.y * a - 1.0, -n.y);
    }
    else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        tangent = vec3(1.0 - n.x * n.x * a, b, -n.x);
        bitangent = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    return normalize(tangent * dir.x + bitangent * dir.y + n * dir.z);
}

struct Voxel {
    ws_normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ls_hit_normal: vec3<f32>,
    is_emissive: bool,
    emissive_intensity: f32,
}
fn unpack_voxel(packed: u32) -> Voxel {
    let is_dialetric = ((packed >> 10u) & 1u) == 0u;
    let emissive_flag = ((packed >> 6u) & 1u) == 1u;

    var res: Voxel;
    res.ws_normal = decode_normal_octahedral(packed >> 11u);
    if is_dialetric {
        res.metallic = 0.0;
        res.roughness = f32((packed >> 6u) & 15u) / 16.0;
    } else if emissive_flag {
        res.metallic = 0.0;
        res.roughness = 1.0;
        res.is_emissive = true;
        res.emissive_intensity = f32((packed >> 7u) & 7u) / 7.0;
    } else {
        res.metallic = 1.0;
        res.roughness = f32((packed >> 7u) & 7u) / 7.0;
    }

    let hit_mask = decode_hit_mask((packed >> 3u) & 7u);
    let ray_dir_sign = vec3<f32>(decode_hit_mask(packed & 7u)) * 2.0 - 1.0;
    res.ls_hit_normal = normalize(-ray_dir_sign * vec3<f32>(hit_mask));

    return res;
}

fn decode_hit_mask(packed: u32) -> vec3<bool> {
    let mask = vec3<u32>(
        (packed >> 2u) & 1u,
        (packed >> 1u) & 1u,
        packed & 1u,
    );
    return vec3<bool>(mask);
}

/// decodes world space normal from lower 21 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn decode_normal_octahedral(packed: u32) -> vec3<f32> {
    let x = f32((packed >> 11u) & 0x3ffu) / 1023.0;
    let y = f32((packed >> 1u) & 0x3ffu) / 1023.0;
    let sgn = f32(packed & 1u) * 2.0 - 1.0;
    var res = vec3<f32>(0.);
    res.x = x - y;
    res.y = x + y - 1.0;
    res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
    return normalize(res);
}

/// ------------------------------------------------------
/// -------------------- map utils -----------------------

fn unpack_voxel_pos(packed: array<u32, 2>) -> vec3<u32> {
    return vec3<u32>(
        packed[1] & 0xFFFFFu,
        ((packed[0] & 0x3FFu) << 10u) | (packed[1] >> 20u),
        packed[0] >> 10u,
    );
}

fn hash_noise(voxel_id: u32) -> vec2<f32> {
    let p = f32(voxel_id);
    var p3 = fract(vec3(p, p, p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
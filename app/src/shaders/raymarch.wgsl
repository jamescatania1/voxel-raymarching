@group(0) @binding(0) var tex_out_albedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_out_velocity: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var tex_out_id: texture_storage_2d<r32uint, write>;

@group(1) @binding(0) var tex_out_normal: texture_storage_2d<r32uint, write>;
@group(1) @binding(1) var tex_out_depth: texture_storage_2d<r32float, write>;

struct VoxelSceneMetadata {
    size: vec3<u32>,
    bounding_size: u32,
    probe_size: vec3<u32>,
    probe_scale: f32,
    index_levels: u32,
    index_chunk_count: u32,
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
struct IndexChunk {
    child_index: u32,
    mask: array<u32, 2>,
}
@group(2) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(1) var<uniform> palette: Palette;
@group(2) @binding(2) var<storage, read> index_chunks: array<IndexChunk>;
@group(2) @binding(3) var<storage, read> leaf_chunks: array<u32>;
@group(2) @binding(4) var tex_noise: texture_3d<f32>;
@group(2) @binding(5) var sampler_noise: sampler;
struct VoxelInfo {
    visible_count: atomic<u32>,
    failed_to_add: atomic<u32>,
}
struct VisibleVoxel {
    data: u32,
    leaf_index: u32,
    pos: array<u32, 2>,
}

/// allocator info for visible voxels
@group(2) @binding(6) var<storage, read_write> voxel_info: VoxelInfo;

/// two words (key, value) per entry. keyed by the leaf index, and values are indices into the visible_voxels
/// each value for the current fragment is also written to the red channel of voxel_id in the gbuffer
@group(2) @binding(7) var<storage, read_write> voxel_map: array<atomic<u32>>;
@group(2) @binding(8) var<storage, read_write> visible_voxels: array<VisibleVoxel>;

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
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

const DDA_MAX_STEPS: u32 = 300u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let res = trace_scene(pos, in.local_index);

    textureStore(tex_out_albedo, pos, vec4(res.albedo, 1.0));
    textureStore(tex_out_id, pos, vec4(res.voxel_id, 0, 0, 0));
    textureStore(tex_out_normal, pos, vec4(res.normal, 0, 0, 0));
    textureStore(tex_out_depth, pos, vec4(res.depth, 0.0, 0.0, 1.0));
    textureStore(tex_out_velocity, pos, vec4(res.velocity, 0.0, 1.0));
}

struct SceneResult {
    albedo: vec3<f32>,
    normal: u32,
    depth: f32,
    velocity: vec2<f32>,
    voxel_id: u32,
    ao: f32,
}

fn trace_scene(pos: vec2<i32>, local_index: u32) -> SceneResult {
    let dimensions = vec2<i32>(textureDimensions(tex_out_albedo).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let uv_jittered = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;
    let ray_uv = select(uv_jittered, uv, frame.taa_enabled == 0u);

    let ray = start_ray(ray_uv);
    let hit = raymarch(start_ray(ray_uv), local_index);

    if !hit.hit {
        var res: SceneResult;
        res.voxel_id = 0;
        res.albedo = vec3(1.0, 0.0, 0.0);
        res.normal = 0u;
        res.depth = -1.0;
        res.velocity = vec2(0.0);
        return res;
    }

    var voxel = unpack_leaf_voxel(leaf_chunks[hit.leaf_index]);

    let ws_pos_h = model.transform * vec4<f32>(hit.local_pos, 1.0);
    let ws_pos = ws_pos_h.xyz;
    let depth = hit.depth;

    let cs_pos = environment.camera.view_proj * ws_pos_h;
    let ndc = cs_pos.xyz / cs_pos.w;
    let cur_uv = ndc.xy * vec2(0.5, -0.5) + 0.5;

    let prev_cs_pos = environment.prev_camera.view_proj * ws_pos_h;
    let prev_ndc = prev_cs_pos.xy / prev_cs_pos.w;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

    let velocity = cur_uv - prev_uv;

    let albedo = palette_color(voxel.palette_index);
    voxel.roughness *= environment.roughness_multiplier;

    let ls_normal = align_per_voxel_normal(hit.hit_normal, voxel.normal, voxel.roughness);
    let ws_normal = normalize(model.normal_transform * ls_normal);

    let packed = repack_voxel(ws_normal, voxel.metallic, voxel.roughness, hit.hit_mask, ray.direction, voxel.is_emissive, voxel.emissive_intensity);

    let map_result = map_insert(hit.leaf_index);
    if map_result.inserted {
        var visible: VisibleVoxel;
        visible.data = packed;
        visible.leaf_index = hit.leaf_index;
        visible.pos = pack_voxel_pos(vec3<u32>(hit.local_pos));
        visible_voxels[map_result.visible_index] = visible;
    }

    var res: SceneResult;
    res.albedo = albedo;
    res.voxel_id = hit.leaf_index;
    res.normal = packed;
    res.depth = depth;
    res.velocity = velocity;
    return res;
}

struct Ray {
    ls_origin: vec3<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_start: f32,
    in_bounds: bool,
}

struct RaymarchResult {
    hit: bool,
    leaf_index: u32,
    hit_normal: vec3<f32>,
    local_pos: vec3<f32>,
    depth: f32,
    hit_mask: vec3<bool>,
}

fn raymarch(ray: Ray, local_index: u32) -> RaymarchResult {
    if !ray.in_bounds {
        return RaymarchResult();
    }

    var dir = ray.direction;
    let inv_dir = -1.0 / max(abs(dir), vec3(1e-8));

    let scale = 1.0 / f32(scene.bounding_size);
    var origin = ray.ls_origin * scale + (ray.t_start * ray.direction) * scale + 1.0;
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

    var i = 0u;
    for (i = 0u; i < 256; i++) {
        var child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;

        while (scale_exp >= 2) & chunk_contains_child(chunk.mask, child_offset) & (!chunk_is_leaf(chunk.child_index)) {
            stack[local_index][scale_exp >> 1u] = ci;
            ci = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);
            chunk = index_chunks[ci];

            scale_exp -= 2u;
            child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;
        }

        // if we reached a leaf, we're done
        if chunk_contains_child(chunk.mask, child_offset) && chunk_is_leaf(chunk.child_index) {
            pos = mirrored_pos_unchecked(pos, dir);

            let leaf_index = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);

            let t_max = min(min(side_distance.x, side_distance.y), side_distance.z);
            let t_total = ray.t_start + f32(scene.bounding_size) * t_max;

            let local_pos = ray.ls_origin + dir * t_total;

            let mask = vec3(t_max) >= side_distance;
            let hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));

            var res: RaymarchResult;
            res.hit = true;
            res.leaf_index = leaf_index;
            res.hit_normal = hit_normal;
            res.depth = t_total;
            res.local_pos = ray.ls_origin + dir * t_total;
            res.hit_mask = mask;

            // res.voxel.palette_index = i;
            return res;
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

    return RaymarchResult();
}

/// ------------------------------------------------------
/// -------------------- map utils -----------------------

struct MapResult {
    inserted: bool,
    visible_index: u32,
}

fn map_insert(leaf_index: u32) -> MapResult {
    let n = arrayLength(&voxel_map);

    /// note - this will stop working past 1 billion leaf_index (# leaf index chunks * 64)
    let key = (leaf_index << 2u) | (frame.frame_id & 3u);
    var index = hash_murmur3(leaf_index) % n;

    for (var _i = 0u; _i < 10u; _i++) {
        let cur = atomicLoad(&voxel_map[index]);

        if cur == key {
            return MapResult();
        }
        if (cur == 0u) || ((cur & 3u) != (frame.frame_id & 3u)) {
            let res = atomicCompareExchangeWeak(&voxel_map[index], cur, key);

            if res.exchanged {
                var res: MapResult;
                res.inserted = true;
                res.visible_index = atomicAdd(&voxel_info.visible_count, 1u);
                return res;
            }
            if res.old_value == key {
                return MapResult();
            }
        }
        index = (index + 3u) % n;
    }

    // for tracking
    atomicAdd(&voxel_info.failed_to_add, 1u);
    return MapResult();
}

fn pack_voxel_pos(pos: vec3<u32>) -> array<u32, 2> {
    return array<u32, 2>(
        ((pos.z & 0xFFFFFu) << 10u) | ((pos.y >> 10u) & 0x3FFu),
        ((pos.y & 0x3FFu) << 20u) | (pos.x & 0xFFFFFu),
    );
}

fn unpack_voxel_pos(packed: array<u32, 2>) -> vec3<u32> {
    return vec3<u32>(
        packed[1] & 0xFFFFFu,
        ((packed[0] & 0x3FFu) << 10u) | (packed[1] >> 20u),
        packed[0] >> 10u,
    );
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

fn start_ray(uv: vec2<f32>) -> Ray {
    let ndc = vec2<f32>(uv.x, 1.0 - uv.y) * 2.0 - 1.0;

    let ts_near = environment.camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = environment.camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = (model.inv_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let bd_max = (model.inv_transform * vec4(vec3<f32>(scene.bounding_size), 1.0)).xyz;

    let inv_dir = safe_inverse(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    let t_start = max(0.0, t_near + 1e-7);

    var ray: Ray;
    ray.origin = ls_origin + t_start * ls_direction;
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    ray.t_start = t_start;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    return ray;
}

fn safe_inverse(v: vec3<f32>) -> vec3<f32> {
    return vec3(
        select(1.0 / v.x, 1e10, v.x == 0.0),
        select(1.0 / v.y, 1e10, v.y == 0.0),
        select(1.0 / v.z, 1e10, v.z == 0.0),
    );
}

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
}

// clamps per-voxel normal to cone aligned with the hit normal
// https://www.desmos.com/3d/cnbvln5rz6
fn align_per_voxel_normal(n_hit: vec3<f32>, n_surface: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smooth_factor = smoothstep(0.0, 1.0, roughness) * environment.smooth_normal_factor;

    let t = clamp(1.0 - smooth_factor * 2.0, -0.999, 0.999);

    let d = dot(n_hit, n_surface);
    if d > t {
        return n_surface;
    }

    return t * n_hit + sqrt((1 - t * t) / (1 - d * d)) * (n_surface - d * n_hit);
}

/// encode hit mask into lower 3 bits of u32
/// this encodes the hit normal, as we recover the ray direction from depth
fn encode_hit_mask(mask: vec3<bool>) -> u32 {
    return (u32(mask.x) << 2u) | (u32(mask.y) << 1u) | u32(mask.z);
}

fn decode_hit_mask(packed: u32) -> vec3<bool> {
    let mask = vec3<u32>(
        (packed >> 2u) & 1u,
        (packed >> 1u) & 1u,
        packed & 1u,
    );
    return vec3<bool>(mask);
}

fn repack_voxel(ws_normal: vec3<f32>, metallic: f32, roughness: f32, hit_mask: vec3<bool>, ray_dir: vec3<f32>, is_emissive: bool, emissive_intensity: f32) -> u32 {
    let n = encode_normal_octahedral(ws_normal);
    let hm = encode_hit_mask(hit_mask) & 7u;
    let dir_mask = encode_hit_mask(vec3<bool>(ray_dir >= vec3(0.0))) & 7u;
    var m: u32;
    var r: u32;
    if is_emissive {
        m = 1u;
        r = (u32(emissive_intensity * 7.0 + 0.5) << 1u) | 1u;
    } else if metallic > 0.5 {
        m = 1u;
        r = u32(roughness * 7.0 + 0.5) << 1u;
    } else {
        m = 0u;
        r = u32(roughness * 15.0 + 0.5);
    }
    return (n << 11u) | (m << 10u) | (r << 6u) | (hm << 3u) | dir_mask;
}

struct LeafVoxel {
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    palette_index: u32,
    is_emissive: bool,
    emissive_intensity: f32,
}

fn unpack_leaf_voxel(packed: u32) -> LeafVoxel {
    let is_dialetric = ((packed >> 14u) & 1u) == 0u;
    let emissive_flag = ((packed >> 10u) & 1u) == 1u;

    var res: LeafVoxel;
    res.palette_index = packed & 0x3ffu;
    res.normal = decode_normal_octahedral_leaf(packed >> 15u);
    if is_dialetric {
        res.metallic = 0.0;
        res.roughness = f32((packed >> 10u) & 0xfu) / 15.0;
    } else if emissive_flag {
        res.metallic = 0.0;
        res.roughness = 1.0;
        res.is_emissive = true;
        res.emissive_intensity = f32((packed >> 11u) & 7u) / 7.0;
    } else {
        res.metallic = 1.0;
        res.roughness = f32((packed >> 11u) & 7u) / 7.0;
    }

    return res;
}

/// decodes world space normal from lower 17 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn decode_normal_octahedral_leaf(packed: u32) -> vec3<f32> {
    let x = f32((packed >> 9u) & 0xffu) / 255.0;
    let y = f32((packed >> 1u) & 0xffu) / 255.0;
    let sgn = f32(packed & 1u) * 2.0 - 1.0;
    var res = vec3<f32>(0.);
    res.x = x - y;
    res.y = x + y - 1.0;
    res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
    return normalize(res);
}

/// encodes world space normal in lower 21 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn encode_normal_octahedral(normal: vec3<f32>) -> u32 {
    var n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var nrm = vec2<f32>(0.);
    nrm.y = n.y * 0.5 + 0.5;
    nrm.x = n.x * 0.5 + nrm.y;
    nrm.y = n.x * -0.5 + nrm.y;
    let sgn = select(0u, 1u, n.z >= 0.0);
    let res = (u32(nrm.x * 1023.0 + 0.5) << 11u) | (u32(nrm.y * 1023.0 + 0.5) << 1u) | sgn;
    return res;
}

// from https://github.com/aappleby/smhasher
fn hash_murmur3(seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51u;
    const C2: u32 = 0x1b873593u;

    var h = 0u;
    var k = seed;

    k *= C1;
    k = (k << 15u) | (k >> 17u);
    k *= C2;

    h ^= k;
    h = (h << 13u) | (h >> 19u);
    h = h * 5u + 0xe6546b64u;
    h ^= 4u;

    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

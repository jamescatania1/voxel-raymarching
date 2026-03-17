@group(0) @binding(0) var tex_voxel_id: texture_storage_2d<r32uint, read>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;

struct VoxelSceneMetadata {
    bounding_size: u32,
    index_levels: u32,
    index_chunk_count: u32,
}
struct IndexChunk {
    child_index: u32,
    mask: array<u32, 2>,
}
@group(2) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(1) var<storage, read> index_chunks: array<IndexChunk>;
@group(2) @binding(2) var<storage, read> index_leaf_positions: array<vec2<u32>>;
@group(2) @binding(3) var tex_noise: texture_3d<f32>;
@group(2) @binding(4) var sampler_noise: sampler;
@group(2) @binding(5) var<storage, read> voxel_map: array<u32>; // voxel hashmap, two words (key, value) per entry
@group(2) @binding(6) var<storage, read_write> voxel_lighting: array<atomic<u32>>; // one per voxel entry right now

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
    @builtin(local_invocation_index) local_index: u32,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

@compute @workgroup_size(8, 4, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_depth).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let uv_jittered = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

    let ray_length = textureLoad(tex_depth, pos).r;
    if ray_length < 0.0 {
        // primary ray missed
        return;
    }

    let voxel_id = textureLoad(tex_voxel_id, pos).r;
    let map_val = map_get(voxel_id);
    if !map_val.exists {
        return;
    }
    let voxel_pos = vec3<f32>(get_voxel_position(voxel_id)) + 0.5;

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    let packed = textureLoad(tex_normal, pos).r;
    let voxel = unpack_voxel(packed);

    let ls_normal = normalize(model.inv_normal_transform * voxel.ws_normal);
    let ls_hit_normal = normalize(-vec3<f32>(sign(ray.direction)) * vec3<f32>(voxel.hit_mask));
    let ls_pos = ray.origin + ray.direction * ray_length;

    let noise = blue_noise(in.id.xy);

    if trace_shadow(pos, noise, voxel_pos, in.local_index) {
        atomicStore(&voxel_lighting[map_val.value], 1u);
    }
    // if res {
    //     let index = (7 - in.thread_id.x) + (in.thread_id.y << 3u);
    //     atomicOr(&results, 1u << index);
    // }

    // workgroupBarrier();

    // if in.local_index == 0u {
    //     let mask = atomicLoad(&results);
    //     textureStore(tex_out, in.group_id.xy, vec4<u32>(mask, 0u, 0u, 0u));
    // }
}

fn trace_shadow(pos: vec2<i32>, noise: vec3<f32>, ls_pos: vec3<f32>, local_index: u32) -> bool {
    let light_dir = normalize(model.inv_normal_transform * environment.sun_direction);

    let light_tangent = normalize(cross(light_dir, vec3(0.0, 0.0, 1.0)));
    let light_bitangent = normalize(cross(light_tangent, light_dir));

    let disk_point = (noise.xy * 2.0 - 1.0) * environment.shadow_spread * 0.0001;
    let dir = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);

    // var dir = rand_hemisphere_direction(noise.xy, environment.shadow_spread);
    // dir = align_direction(dir, light_dir);

    var ray: Ray;
    ray.origin = ls_pos;
    ray.direction = dir;

    let occluded = raymarch_shadow(ray, local_index);
    // let occluded = false;
    return !occluded;
    // return select(1.0, 0.0, occluded);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn raymarch_shadow(ray: Ray, local_index: u32) -> bool {
    var dir = ray.direction;
    // let inv_dir = sign(dir) / max(vec3(1e-7), abs(dir));
    let inv_dir = 1.0 / -abs(dir);

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
    var skip_next_hit = true;

    var i = 0u;
    for (i = 0u; i < 256; i++) {
        var child_offset = chunk_offset(pos, scale_exp) ^ mirror_mask;

        while chunk_contains_child(chunk.mask, child_offset) && !chunk_is_leaf(chunk.child_index) && scale_exp >= 2u {
            stack[local_index][scale_exp >> 1u] = ci;
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
        if ((chunk.mask[snapped_idx >> 5u] >> (snapped_idx & 31u)) & 0x00330033u) == 0u {
            adv_scale_exp++;
        }
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
        skip_next_hit = false;
    }

    return false;
}

// noise from https://github.com/electronicarts/fastnoise/blob/main/FastNoiseDesign.md
fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    const FRACT_PHI: f32 = 0.61803398875;
    const FRACT_SQRT_2: f32 = 0.41421356237;
    const OFFSET: vec2<f32> = vec2<f32>(FRACT_PHI, FRACT_SQRT_2);

    let frame_offset_seed = (frame.frame_id >> 5u) & 0xffu;
    let frame_offset = vec2<u32>(OFFSET * 128.0 * f32(frame_offset_seed));

    let id = pos + frame_offset;
    let sample_pos = vec3<u32>(
        id.x & 0x7fu,
        id.y & 0x7fu,
        frame.frame_id & 0x1fu,
    );
    let noise = textureLoad(tex_noise, sample_pos, 0).rgb;
    return noise;
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
    if i < 32u {
        return countOneBits(mask[0] & ~(0xffffffffu << i));
    } else {
        return countOneBits(mask[0]) + countOneBits(mask[1] & ~(0xffffffffu << (i - 32u)));
    }
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
    hit_mask: vec3<bool>,
}
fn unpack_voxel(packed: u32) -> Voxel {
    var res: Voxel;
    res.ws_normal = decode_normal_octahedral(packed >> 11u);
    res.metallic = f32((packed >> 10u) & 1u);
    res.roughness = f32((packed >> 6u) & 15u) / 16.0;
    res.hit_mask = decode_hit_mask((packed >> 3u) & 7u);
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

struct MapResult {
    exists: bool,
    value: u32,
}

fn map_get(id: u32) -> MapResult {
    let n = arrayLength(&voxel_map) >> 1u;

    var key = hash_murmur3(id) % n;
    for (var i = 0u; i < 4u; i++) {
        if voxel_map[key << 1u] == id {
            var res: MapResult;
            res.exists = true;
            res.value = voxel_map[(key << 1u) + 1u];
            return res;
        }
        key += 1u;
        if key >= n {
            key = 0u;
        }
    }

    var res: MapResult;
    res.exists = false;
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

fn pack_voxel_pos(pos: vec3<u32>) -> vec2<u32> {
    return vec2<u32>(
        ((pos.z & 0xFFFFFu) << 10u) | ((pos.y >> 10u) & 0x3FFu),
        ((pos.y & 0x3FFu) << 20u) | (pos.x & 0xFFFFFu),
    );
}

fn unpack_voxel_pos(packed: vec2<u32>) -> vec3<u32> {
    return vec3<u32>(
        packed.y & 0xFFFFFu,
        ((packed.x & 0x3FFu) << 10u) | (packed.y >> 20u),
        packed.x >> 10u,
    );
}

fn get_voxel_position(map_val: u32) -> vec3<u32> {
    let ci = map_val >> 6u;

    var voxel_pos = unpack_voxel_pos(index_leaf_positions[ci]);
    voxel_pos.x += map_val & 3u;
    voxel_pos.z += (map_val >> 2u) & 3u;
    voxel_pos.y += (map_val >> 4u) & 3u;

    return voxel_pos;
}
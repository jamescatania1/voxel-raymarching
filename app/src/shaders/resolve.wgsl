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
struct ChunkLighting {
    m1: f32,
    m2: f32,
    history_length: u32,
}
struct ChunkLightingSample {
    // for irradiance
    m1: u32,
    m2: u32,
    // pad (2) | shadow sum (10) | shadow samples (10) | irradiance samples (10)
    shadow_counts: u32,
}
@group(0) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(0) @binding(1) var<storage, read> index_chunks: array<IndexChunk>;
@group(0) @binding(2) var<storage, read> voxel_map: array<u32>;
@group(0) @binding(3) var<storage, read> visible_voxels: array<VisibleVoxel>;
@group(0) @binding(4) var<storage, read> cur_voxel_lighting: array<array<u32, 3>>;
@group(0) @binding(5) var<storage, read_write> acc_voxel_lighting: array<array<u32, 3>>;
@group(0) @binding(6) var<storage, read> cur_chunk_lighting: array<ChunkLightingSample>;
@group(0) @binding(7) var<storage, read> acc_chunk_lighting: array<ChunkLighting>;

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

const MAX_HISTORY_LENGTH: u32 = 128;
const SHADOW_VARIANCE_SENSITIVITY: f32 = 38.0;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let visible = visible_voxels[in.id.x];
    if visible.data == 0u {
        return;
    }
    // let voxel = unpack_voxel(visible.data);

    var cur = unpack_visible_lighting(cur_voxel_lighting[in.id.x]);

    let shadow = f32(cur.shadow);
    var irradiance = cur.irradiance;
    var weight = 8.0;

    let voxel_pos = unpack_voxel_pos(visible.pos);

    const NEIGHBORS: array<vec4<i32>, 26> = array<vec4<i32>, 26>(
        vec4<i32>(-1, -1, -1, 1), vec4<i32>(0, -1, -1, 2), vec4<i32>(1, -1, -1, 1),
        vec4<i32>(-1, 0, -1, 2), vec4<i32>(0, 0, -1, 4), vec4<i32>(1, 0, -1, 2),
        vec4<i32>(-1, 1, -1, 1), vec4<i32>(0, 1, -1, 2), vec4<i32>(1, 1, -1, 1),
        vec4<i32>(-1, -1, 0, 2), vec4<i32>(0, -1, 0, 4), vec4<i32>(1, -1, 0, 2),
        vec4<i32>(-1, 0, 0, 4), vec4<i32>(1, 0, 0, 4),
        vec4<i32>(-1, 1, 0, 2), vec4<i32>(0, 1, 0, 4), vec4<i32>(1, 1, 0, 2),
        vec4<i32>(-1, -1, 1, 1), vec4<i32>(0, -1, 1, 2), vec4<i32>(1, -1, 1, 1),
        vec4<i32>(-1, 0, 1, 2), vec4<i32>(0, 0, 1, 4), vec4<i32>(1, 0, 1, 2),
        vec4<i32>(-1, 1, 1, 1), vec4<i32>(0, 1, 1, 2), vec4<i32>(1, 1, 1, 1),
    );

    for (var i = 0u; i < 26u; i++) {
        break;
        let n = NEIGHBORS[i];

        let n_pos = n.xyz + vec3<i32>(voxel_pos);

        if any(n_pos < vec3<i32>(0)) {
            continue;
        }
        let n_pos_u = vec3<u32>(n_pos);
        if any(n_pos_u >= scene.size) {
            continue;
        }

        var w = f32(n.w);

        let n_leaf = lookup_leaf(n_pos_u);
        if !n_leaf.found {
            continue;
        }
        // let n_lighting = unpack_voxel_lighting(acc_voxel_lighting[n_leaf.leaf_index]);
        let n_visible = map_get(n_leaf.leaf_index);
        if !n_visible.valid {
            continue;
        }
        let neighbor = visible_voxels[n_visible.visible_index];
        let n_voxel = unpack_voxel(neighbor.data);
        let n_lighting = unpack_visible_lighting(cur_voxel_lighting[n_visible.visible_index]);

        let ndl = max(dot(n_voxel.ls_hit_normal, n_lighting.direction), 0.0);
        w *= ndl;

        irradiance += n_lighting.irradiance * w;
        weight += w;
    }

    irradiance /= weight;

    let luma = dot(irradiance, vec3(0.2126, 0.7152, 0.0722));

    var acc = unpack_voxel_lighting(acc_voxel_lighting[visible.leaf_index]);

    let ci = get_containing_index_chunk_id(voxel_pos);
    let chunk = acc_chunk_lighting[ci];

    let cur_chunk_packed = cur_chunk_lighting[ci];
    let chunk_sample_count = (cur_chunk_packed.shadow_counts >> 10u) & 0x3FFu;

    let shadow_floor = chunk.m1 * (1.0 - chunk.m1) / f32(chunk_sample_count);
    var variance_shadow = max(0.0, chunk.m2 - chunk.m1 * chunk.m1);
    // variance_shadow = max(0.0, variance_shadow - shadow_floor);

    let shadow_confidence = saturate(variance_shadow * SHADOW_VARIANCE_SENSITIVITY);

    let max_shadow_history = u32(mix(f32(MAX_HISTORY_LENGTH), 8.0, shadow_confidence));
    acc.history_length = min(max_shadow_history, acc.history_length + 1);

    // let cur_luma = combined_luminance(cur);
    // let acc_luma = combined_luminance(acc);

    // if acc.history_length == 0u {
    //     acc.history_length = 1u;
    //     acc.shadow = cur.shadow;
    //     acc.irradiance = cur.irradiance;
    //     acc.variance = cur_luma * cur_luma;
    //     acc_voxel_lighting[visible.leaf_index] = pack_voxel_lighting(acc);
    //     return;
    // }

    // let moment_alpha = max(1.0 / f32(acc.history_length + 1u), 0.01);
    // let m1 = mix(acc_luma, cur_luma, moment_alpha);
    // let m2 = mix(acc.variance, cur_luma * cur_luma, moment_alpha);

    // let variance = max(m2 - m1 * m1, 0.0);
    // let rel_variance = variance / max(m1 * m1, 0.01);

    // let prev_luma = acc_luma;
    // let max_history_len = u32(mix(512.0, 8.0, saturate(rel_variance * 4.0)));

    // acc.history_length = min(acc.history_length + 1u, max_history_len);
    // acc.variance = m2;
    // acc.history_length = min(MAX_HISTORY_LENGTH, acc.history_length + 1u);

    let alpha = 1.0 / f32(acc.history_length);
    acc.shadow = mix(acc.shadow, shadow, alpha);
    // acc.shadow = shadow_confidence;
    acc.irradiance = mix(acc.irradiance, irradiance, alpha);
    // acc.irradiance = irradiance;

    // {
    //     let vox = map_get(visible.leaf_index);
    //     let pos = unpack_voxel_pos(visible.pos);
    //     let l = lookup_leaf(pos);
    //     if !l.found {
    //         acc.irradiance = vec3(1.0, 0.0, 0.0);
    //     } else {
    //         // acc.irradiance = vec3(0.0, 1.0, 0.0);
    //         acc.irradiance = vec3(fract(abs(f32(visible.leaf_index) - f32(l.leaf_index))), 0.0, 0.0);
    //     }

    //     var h = vec3(hash_murmur3(pos.x), hash_murmur3(pos.y), hash_murmur3(pos.z));
    //     var n = fract(vec3<f32>(pos & vec3(0xFFFFu)) * 2.1398571);
    //     // acc.irradiance = n;
    //     // acc.irradiance = vec3<f32>(abs(f32(in.id.x) - f32(vox.visible_index)));
    // }

    acc_voxel_lighting[visible.leaf_index] = pack_voxel_lighting(acc);
}

// fn combined_luminance(lighting: VoxelLighting) -> f32 {
//     // return lighting.shadow + dot(lighting.irradiance, vec3(0.2126, 0.7152, 0.0722));
//     return lighting.shadow;
// }

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

struct MapResult {
    valid: bool,
    visible_index: u32,
}

fn map_get(leaf_index: u32) -> MapResult {
    let n = arrayLength(&voxel_map) >> 1u;
    // let key = (leaf_index << 2u) | (frame.frame_id & 3u);
    let key = leaf_index;
    var index = hash_murmur3(leaf_index) % n;

    for (var _i = 0u; _i < 10u; _i++) {
        let cur = voxel_map[index << 1u];

        if cur == key {
            var res: MapResult;
            res.valid = true;
            res.visible_index = voxel_map[(index << 1u) + 1u];
            return res;
        }
        if cur == 0u {
            break;
        }
        index = (index + 3u) % n;
    }
    return MapResult();
}

fn unpack_voxel_pos(packed: array<u32, 2>) -> vec3<u32> {
    return vec3<u32>(
        packed[1] & 0xFFFFFu,
        ((packed[0] & 0x3FFu) << 10u) | ((packed[1] >> 20u) & 0x3FFu),
        (packed[0] >> 10u) & 0xFFFFFu,
    );
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

struct VisibleLighting {
    irradiance: vec3<f32>,
    shadow: bool,
    direction: vec3<f32>,
}
fn unpack_visible_lighting(packed: array<u32, 3>) -> VisibleLighting {
    var res: VisibleLighting;
    res.shadow = bool(packed[2] & 1u);
    res.direction = decode_normal_octahedral(packed[2] >> 1u);
    res.irradiance = vec3<f32>(unpack2x16float(packed[0]).xy, unpack2x16float(packed[1]).x);
    return res;
}
fn pack_visible_lighting(val: VisibleLighting) -> array<u32, 3> {
    var res: array<u32, 3>;
    res[0] = pack2x16float(val.irradiance.rg);
    res[1] = pack2x16float(vec2<f32>(val.irradiance.b, 0.0));
    res[2] = (encode_normal_octahedral(val.direction) << 1u) | select(0u, 1u, val.shadow);
    return res;
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

struct LookupResult {
    found: bool,
    leaf_index: u32,
}

/// ------------------------------------------------------
/// ---------------- tree traversal utils ----------------

fn lookup_leaf(pos: vec3<u32>) -> LookupResult {
    var ci = 0u;
    var chunk = index_chunks[ci];
    var scale_exp = scene.index_levels * 2u - 2u;

    while scale_exp >= 0u {
        let cx = (pos.x >> scale_exp) & 3u;
        let cy = (pos.y >> scale_exp) & 3u;
        let cz = (pos.z >> scale_exp) & 3u;

        let child_offset = cx | (cz << 2u) | (cy << 4u);

        if !chunk_contains_child(chunk.mask, child_offset) {
            return LookupResult();
        }

        let next = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);

        if chunk_is_leaf(chunk.child_index) {
            var r: LookupResult;
            r.found = true;
            r.leaf_index = next;
            return r;
        }

        ci = next;
        chunk = index_chunks[ci];
        scale_exp -= 2u;
    }

    return LookupResult();
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
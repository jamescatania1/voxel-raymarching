struct VoxelSceneMetadata {
    bounding_size: u32,
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
@group(0) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(0) @binding(1) var<uniform> palette: Palette;
@group(0) @binding(2) var<storage, read> index_chunks: array<IndexChunk>;
@group(0) @binding(3) var<storage, read> leaf_chunks: array<u32>;
@group(0) @binding(4) var tex_noise: texture_3d<f32>;
@group(0) @binding(5) var sampler_noise: sampler;
@group(0) @binding(6) var sampler_linear: sampler;
@group(0) @binding(7) var tex_skybox: texture_cube<f32>;

@group(0) @binding(8) var<storage, read> visible_voxels: array<VisibleVoxel>;

// current frame per-voxel shadow values
@group(0) @binding(9) var<storage, read_write> voxel_lighting: array<array<u32, 3>>;
@group(0) @binding(10) var<storage, read> acc_voxel_lighting: array<array<u32, 3>>;

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

    // let noise = blue_noise(vec2(voxel_pos.xy));
    let noise = hash_noise(visible.leaf_index);

    var res = unpack_voxel_lighting(voxel_lighting[in.id.x]);

    let trace = trace_ambient(noise, voxel_center, in.local_index, voxel.ls_hit_normal);
    res.irradiance = trace.irradiance;

    voxel_lighting[in.id.x] = pack_voxel_lighting(res);
}

struct AmbientResult {
    irradiance: vec3<f32>,
}

fn trace_ambient(noise: vec2<f32>, ls_pos: vec3<f32>, local_index: u32, ls_normal: vec3<f32>) -> AmbientResult {
    let light_dir = normalize(model.inv_normal_transform * environment.sun_direction);

    let u = f32(reverseBits(frame.frame_id)) * 2.3283064365386963e-10;
    let v = fract(f32(frame.frame_id) * 0.61803398875);
    var sample = fract(vec2(u, v) + noise);

    let r = sqrt(sample.x);
    let t = 2.0 * 3.14159265359 * sample.y;
    var dir = vec3<f32>(r * cos(t), r * sin(t), sqrt(max(0.0, 1.0 - sample.x)));
    dir = align_direction(dir, ls_normal);

    var ray: Ray;
    ray.origin = ls_pos + ls_normal * environment.shadow_bias;
    ray.direction = dir;

    let hit = raymarch(ray);

    let ws_ray_dir = normalize((model.transform * vec4(dir, 0.0)).xyz);
    let rot_dir = vec3(
        ws_ray_dir.x * environment.skybox_rotation.x - ws_ray_dir.y * environment.skybox_rotation.y,
        ws_ray_dir.x * environment.skybox_rotation.y + ws_ray_dir.y * environment.skybox_rotation.x,
        ws_ray_dir.z,
    );
    var sky_color = textureSampleLevel(tex_skybox, sampler_linear, rot_dir.xzy, 0.0).rgb;
    sky_color = min(sky_color, vec3(10.0)) * environment.indirect_sky_intensity;

    if hit.hit {
        let secondary = unpack_leaf_voxel(leaf_chunks[hit.leaf_index]);
        let albedo = palette_color(secondary.palette_index);

        var lighting = unpack_voxel_lighting(acc_voxel_lighting[hit.leaf_index]);
        if lighting.history_length == 0u {
            // lighting.irradiance = textureSampleLevel(tex_skybox, sampler_linear, secondary.normal.xzy, 0.0).rgb;
            // lighting.irradiance = min(lighting.irradiance, vec3(15.0)) * environment.indirect_sky_intensity;
            lighting.shadow = 0.0;
        }

        let ndl = max(dot(secondary.normal, environment.sun_direction), 0.0);

        let direct = environment.sun_color * environment.sun_intensity * ndl * (1.0 - lighting.shadow);
        let indirect = lighting.irradiance;

        var emissive = vec3(0.0);
        if secondary.is_emissive {
            emissive = albedo * secondary.emissive_intensity * 5.0;
        }

        let diffuse = (direct + lighting.irradiance) * albedo + emissive;

        // var ao_weight = saturate(hit.depth / f32(environment.max_ambient_distance));
        // ao_weight *= ao_weight;
        // let radiance = mix(diffuse, sky_color, ao_weight);

        let irradiance = diffuse;

        return AmbientResult(irradiance);
    } else {
        return AmbientResult(sky_color);
    }
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct RaymarchResult {
    hit: bool,
    id: u32,
    leaf_index: u32,
    depth: f32,
}

fn raymarch(ray: Ray) -> RaymarchResult {
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
            let leaf_index = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);

            let t_max = min(min(side_distance.x, side_distance.y), side_distance.z);
            let t_total = f32(scene.bounding_size) * t_max;

            // let mask = vec3(t_max) >= side_distance;

            var res: RaymarchResult;
            res.hit = true;
            res.leaf_index = leaf_index;
            res.depth = t_total;
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

            ci = stack[scale_exp >> 1u];
            chunk = index_chunks[ci];
        }
        skip_next_hit = false;
    }

    return RaymarchResult();
}

// noise from https://github.com/electronicarts/fastnoise/blob/main/FastNoiseDesign.md
fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    const FRACT_PHI: f32 = 0.61803398875;
    const FRACT_SQRT_2: f32 = 0.41421356237;
    const OFFSET: vec2<f32> = vec2<f32>(FRACT_PHI, FRACT_SQRT_2);

    let frame_offset_seed = (frame.frame_id >> 5u) & 0xffu;
    let frame_offset = vec2<u32>(OFFSET * 128.0 * f32(frame_offset_seed));

    let id = pos + frame_offset;
    // let id = pos;
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

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
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

fn hash_noise(voxel_id: u32) -> vec2<f32> {
    let p = f32(voxel_id);
    var p3 = fract(vec3(p, p, p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
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

fn pack_voxel_lighting(value: VoxelLighting) -> array<u32, 3> {
    return array<u32, 3>(
        pack2x16float(value.irradiance.rg),
        pack2x16float(vec2(value.irradiance.b, value.variance)),
        (u32(saturate(value.shadow) * 65535.0 + 0.5) << 16u) | (value.history_length & 0xFFFFu),
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
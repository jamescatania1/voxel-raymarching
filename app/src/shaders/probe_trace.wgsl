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
@group(0) @binding(4) var<storage, read> shadow_mask: array<u32>;
@group(0) @binding(5) var sampler_linear: sampler;
@group(0) @binding(6) var tex_skybox: texture_cube<f32>;
@group(0) @binding(7) var tex_probe_irradiance: texture_2d<f32>;
@group(0) @binding(8) var tex_probe_depth: texture_2d<f32>;
@group(0) @binding(9) var tex_probe_rays: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var<storage, read> probes: array<vec3<f32>>;

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
    ambient_filter_scale: f32,
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
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

const RAY_COUNT: u32 = 128;
const PROBE_DEPTH_SCALE: f32 = 1.0 / 40.0;

@compute @workgroup_size(128, 1, 1)
fn compute_main(in: ComputeIn) {
    let dir = generate_sample_direction(f32(in.local_index), f32(RAY_COUNT));

    let probe_id = get_probe_id(in.workgroup_id);
    let probe_pos = probes[probe_id];

    let trace = trace(probe_pos, dir);

    // here, we get the coordnate in the ray results texture
    // the results texture is organized as 16x8 tiles
    // with 128 tiles width (2048 pixels) x however many necessary height
    // this will change if we want more/less rays per probe update

    let out_tile_base = vec2(
        (probe_id & 127u) << 4,
        (probe_id >> 7u) << 3,
    );
    let out_tile_offset = vec2(
        in.local_index & 15,
        in.local_index >> 4,
    );
    let out_pos = out_tile_base + out_tile_offset;
    textureStore(tex_probe_rays, out_pos, vec4(trace.radiance, trace.depth * PROBE_DEPTH_SCALE));
}

struct Trace {
    radiance: vec3<f32>,
    depth: f32,
}
fn trace(pos: vec3<f32>, dir: vec3<f32>) -> Trace {
    var ray: Ray;
    ray.origin = pos;
    ray.direction = dir;

    let hit = raymarch(ray);

    if hit.hit {
        let hit_voxel = unpack_leaf_voxel(leaf_chunks[hit.leaf_index]);
        let hit_albedo = palette_color(hit_voxel.palette_index);

        let shadow_occluded = (shadow_mask[hit.leaf_index >> 5u] & (1u << (hit.leaf_index & 31u))) == 1u;
        let hit_shadow = select(0.0, 1.0, shadow_occluded);

        let ls_normal = hit.hit_normal;
        // let ls_normal = align_per_voxel_normal(hit.hit_normal, hit_voxel.normal, max(hit_voxel.roughness, 0.8));
        let ws_normal = normalize(model.normal_transform * ls_normal);

        let hit_irradiance = sample_irradiance(hit.pos, ls_normal);
        var hit_emissive = vec3(0.0);
        if hit_voxel.is_emissive {
            hit_emissive = hit_albedo * hit_voxel.emissive_intensity * 5.0;
        }

        let ndl = max(dot(ws_normal, environment.sun_direction), 0.0);
        let direct = environment.sun_color * environment.sun_intensity * ndl * hit_shadow;

        let radiance = (direct + hit_irradiance) * hit_albedo + hit_emissive;

        // we omit the depth samples if the ray traveled farther than the main diagonal of the grid cell
        var depth = hit.depth;
        if hit.depth > scene.probe_scale * 1.0 * 1.41421356237 {
            depth = -1; // don't accumulate
        }

        return Trace(radiance, hit.depth);
    } else {
        let ws_ray_dir = normalize((model.transform * vec4(dir, 0.0)).xyz);
        let rot_dir = vec3(
            ws_ray_dir.x * environment.skybox_rotation.x - ws_ray_dir.y * environment.skybox_rotation.y,
            ws_ray_dir.x * environment.skybox_rotation.y + ws_ray_dir.y * environment.skybox_rotation.x,
            ws_ray_dir.z,
        );
        var sky_color = textureSampleLevel(tex_skybox, sampler_linear, rot_dir.xzy, 0.0).rgb;
        sky_color = min(sky_color, vec3(10.0)) * environment.indirect_sky_intensity;

        return Trace(sky_color, -1.0);
    }
}

fn generate_sample_direction(i: f32, n: f32) -> vec3<f32> {
    // TODO rotate per frame
    const GOLDEN: f32 = 1.61803398875;
    const PI: f32 = 3.14159265359;

    let phi = 2.0 * PI * fract(i * (GOLDEN - 1.0));
    let cos_t = 1.0 - (2.0 * i + 1.0) * (1.0 / n);
    let sin_t = sqrt(saturate(1.0 - cos_t * cos_t));

    // return vec3(
    //     cos(phi) * sin_t,
    //     sin(phi) * sin_t,
    //     cos_t,
    // );
    var dir = vec3(
        cos(phi) * sin_t,
        sin(phi) * sin_t,
        cos_t,
    );

    let angle = f32(reverseBits(frame.frame_id)) * 2.3283064365386963e-10 * 6.28318530718;
    let ca = cos(angle);
    let sa = sin(angle);
    return vec3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
}

fn sample_irradiance(pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // let grid_cur = vec3<u32>(pos / scene.probe_scale);
    // let cur_probe_id = get_probe_id(grid_cur);
    // let cur_pos = probes[cur_probe_id];

    // alot of the improvements by Dominik Rohacek are used here
    // https://cescg.org/wp-content/uploads/2022/04/Rohacek-Improving-Probes-in-Dynamic-Diffuse-Global-Illumination.pdf

    let grid_base = vec3<i32>(floor(pos / scene.probe_scale - 0.5));
    let grid_base_pos = (vec3<f32>(grid_base) + 0.5) * scene.probe_scale;

    let alpha = saturate((pos - grid_base_pos) / scene.probe_scale);

    var irradiance = vec3(0.0);
    var weight = 0.0;

    for (var i = 0u; i < 8; i++) {
        let grid_offset = vec3<i32>(vec3(i, i >> 1, i >> 2) & vec3(1));
        let grid_pos = grid_base + grid_offset;

        let probe = get_probe(grid_pos);
        if !probe.exists {
            continue;
        }

        // let probe_offset = (probe.pos - grid_base_pos) / scene.probe_scale;
        // let probe_shift = probe_offset - vec3<f32>(grid_offset);
        // let scaled_offset = (alpha - probe_shift) / max(vec3(1e-4), 1.0 - probe_shift);
        let trilinear = mix(1.0 - alpha, alpha, vec3<f32>(grid_offset));

        // let offset = (probe.pos - grid_base_pos) / scene.probe_scale;
        // let scaled_offset = (vec3<f32>(grid_offset) - offset) / (1.0 - offset);

        var dir = probe.pos - (pos + 0.2 * normal);
        let probe_distance = length(dir) * PROBE_DEPTH_SCALE;
        dir = normalize(dir);

        let sample = sample_probe(probe.id, dir, normal);
        var w = 1.0;

        // most of this math is from Majercik et. al's original impl
        // their code is pretty simple, worth a look
        // https://www.jcgt.org/published/0008/02/01/paper-lowres.pdf
        let dir_unbiased = normalize(probe.pos - pos);
        w *= pow(max(1e-4, (dot(dir_unbiased, normal) + 1.0) * 0.5), 2.0) + 0.05;

        let diff = max(probe_distance - sample.depth_mean, 0.0);
        var chebyshev_weight = sample.depth_variance / (sample.depth_variance + diff * diff);
        chebyshev_weight = max(pow(chebyshev_weight, 3.0), 0.0);
        if probe_distance > sample.depth_mean {
            w *= max(1e-6, chebyshev_weight);
        }

        const DAMPEN_THREHSOLD: f32 = 0.2;
        if w < DAMPEN_THREHSOLD {
            w *= (w * w) / (DAMPEN_THREHSOLD * DAMPEN_THREHSOLD);
        }

        w *= trilinear.x * trilinear.y * trilinear.z;

        irradiance += w * sample.irradiance;
        weight += w;
    }

    return irradiance / max(1e-6, weight);
}

struct ProbeSample {
    irradiance: vec3<f32>,
    depth_mean: f32,
    depth_variance: f32,
}
fn sample_probe(probe_id: u32, dir: vec3<f32>, normal: vec3<f32>) -> ProbeSample {
    let irradiance_texel_size = 1.0 / vec2<f32>(textureDimensions(tex_probe_irradiance).xy);
    let depth_texel_size = 1.0 / vec2<f32>(textureDimensions(tex_probe_depth).xy);

    let dir_uv = encode_octahedral(dir) * 0.5 + 0.5;
    let normal_uv = encode_octahedral(normal) * 0.5 + 0.5;

    let irradiance_base = vec2<f32>(probe_atlas_offset_irradiance(probe_id)) * irradiance_texel_size;
    let irradiance_offset = (normal_uv * 6.0 + 1.0) * irradiance_texel_size;
    let irradiance_uv = irradiance_base + irradiance_offset;

    let depth_base = vec2<f32>(probe_atlas_offset_depth(probe_id)) * depth_texel_size;
    let depth_offset = (dir_uv * 14.0 + 1.0) * depth_texel_size;
    let depth_uv = depth_base + depth_offset;

    let irradiance = textureSampleLevel(tex_probe_irradiance, sampler_linear, irradiance_uv, 0.0).rgb;
    let moments = textureSampleLevel(tex_probe_depth, sampler_linear, depth_uv, 0.0).rg;

    var res: ProbeSample;
    res.irradiance = irradiance;
    res.depth_mean = moments.x;
    res.depth_variance = max(0.0, moments.y - moments.x * moments.x);
    return res;
}

struct Probe {
    exists: bool,
    id: u32,
    pos: vec3<f32>,
}
fn get_probe(grid_pos: vec3<i32>) -> Probe {
    if any(grid_pos < vec3(0)) || any(grid_pos >= vec3<i32>(scene.probe_size)) {
        let pos = (vec3<f32>(grid_pos) + 0.5) * scene.probe_scale;
        return Probe(false, 0, pos);
    }
    let p = vec3<u32>(grid_pos);
    let id = p.x + p.y * scene.probe_size.x + p.z * scene.probe_size.x * scene.probe_size.y;;
    let pos = probes[id];
    return Probe(true, id, pos);
}

// flat probe index from grid coordinates
fn get_probe_id(grid_pos: vec3<u32>) -> u32 {
    return grid_pos.x + grid_pos.y * scene.probe_size.x + grid_pos.z * scene.probe_size.x * scene.probe_size.y;
}

// get the actual position of the probe from grid coordinates
// pretty simple since it's all uniform atm
fn get_probe_pos(grid_pos: vec3<u32>) -> vec3<f32> {
    return (vec3<f32>(grid_pos) + 0.5) * scene.probe_scale;
}

// atlas always has width 2048
// each row contains 256 tiles since each is 8x8 incl. padding
fn probe_atlas_offset_irradiance(probe_id: u32) -> vec2<u32> {
    return vec2(
        (probe_id & 255) << 3,
        (probe_id >> 8u) << 3,
    );
}

// atlas always has width 2048
// each row contains 128 tiles since each is 16x16 incl. padding
fn probe_atlas_offset_depth(probe_id: u32) -> vec2<u32> {
    return vec2(
        (probe_id & 127) << 4,
        (probe_id >> 7u) << 4,
    );
}

fn encode_octahedral(direction: vec3<f32>) -> vec2<f32> {
    let l1 = dot(abs(direction), vec3(1.0));
    var uv = direction.xy * (1.0 / l1);
    if direction.z < 0.0 {
        uv = (1.0 - abs(uv.yx)) * select(vec2(-1.0), vec2(1.0), uv.xy >= vec2(0.0));
    }
    return uv;
}

fn decode_octahedral(uv: vec2<f32>) -> vec3<f32> {
    var v = vec3(uv, 1.0 - abs(uv.x) - abs(uv.y));
    if v.z < 0.0 {
        v = vec3((1.0 - abs(v.yx)) * select(vec2(-1.0), vec2(1.0), uv.xy >= vec2(0.0)), v.z);
    }
    return normalize(v);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct RaymarchResult {
    hit: bool,
    leaf_index: u32,
    depth: f32,
    pos: vec3<f32>,
    hit_normal: vec3<f32>,
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
            let depth = f32(scene.bounding_size) * t_max;

            let mask = vec3(t_max) >= side_distance;
            let hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));

            var res: RaymarchResult;
            res.hit = true;
            res.leaf_index = leaf_index;
            res.depth = depth;
            res.pos = ray.origin + dir * depth;
            res.hit_normal = hit_normal;
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

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
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

// clamps per-voxel normal to cone aligned with the hit normal
// https://www.desmos.com/3d/cnbvln5rz6
fn align_per_voxel_normal(n_hit: vec3<f32>, n_surface: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smooth_factor = smoothstep(0.0, 1.0, roughness) * environment.smooth_normal_factor;

    let t = clamp(1.0 - smooth_factor * 2.0, -0.999, 0.999);

    let d = dot(n_hit, n_surface);
    if d > t {
        return n_surface;
    }

    return normalize(t * n_hit + sqrt((1 - t * t) / (1 - d * d)) * (n_surface - d * n_hit));
}

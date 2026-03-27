@group(0) @binding(0) var tex_out_specular: texture_storage_2d<rgba16float, write>;
// @group(0) @binding(1) var tex_out_specular_velocity: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_out_specular_dir_pdf: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;

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
    ao: f32,
    history_length: u32,
}
@group(2) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(1) var<uniform> palette: Palette;
@group(2) @binding(2) var<storage, read> index_chunks: array<IndexChunk>;
@group(2) @binding(3) var<storage, read> leaf_chunks: array<u32>;
@group(2) @binding(4) var tex_noise: texture_3d<f32>;
@group(2) @binding(5) var sampler_noise: sampler;
@group(2) @binding(6) var sampler_linear: sampler;
@group(2) @binding(7) var tex_skybox: texture_cube<f32>;
@group(2) @binding(8) var<storage, read> acc_voxel_lighting: array<array<u32, 3>>;

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
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const PI = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_out_specular).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let uv_jittered = (vec2<f32>(pos) + environment.camera.jitter * 0.5) * texel_size;

    // let quadrant_depth = textureGather
    let ray_length = textureLoad(tex_depth, pos * 2).r;
    if ray_length < 0.0 {
        // primary ray missed
        textureStore(tex_out_specular, pos, vec4(1.0));
        return;
    }

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    let packed = textureLoad(tex_normal, pos * 2).r;
    let voxel = unpack_voxel(packed);

    let ls_normal = normalize(model.inv_normal_transform * voxel.ws_normal);
    let ls_pos = ray.origin + ray.direction * ray_length;

    let noise = blue_noise(in.id.xy);

    var trace = trace_specular(pos, noise, ls_pos, ls_normal, voxel.ls_hit_normal, voxel.roughness);

    textureStore(tex_out_specular, pos, vec4(trace.specular, select(1.0, -1.0, trace.hit)));
    textureStore(tex_out_specular_dir_pdf, pos, vec4(trace.dir, trace.pdf));
    // if !trace.valid {
    //     textureStore(tex_out_specular_velocity, pos, vec4(0.0));
    //     return;
    // }

    // var prev_cs_pos: vec4<f32>;
    // if trace.hit {
    //     prev_cs_pos = environment.prev_camera.view_proj * vec4<f32>(trace.hit_pos, 1.0);
    // } else {
    //     prev_cs_pos = environment.prev_camera.view_proj * vec4<f32>(trace.dir, 0.0);
    // }
    // let prev_ndc = prev_cs_pos.xy / prev_cs_pos.w;
    // let prev_uv = prev_ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;

    // let velocity = uv - prev_uv;
    // let hit_flag = select(-1.0, 1.0, trace.hit);
    // textureStore(tex_out_specular_velocity, pos, vec4<f32>(velocity, hit_flag, 0.0));
}

struct TraceResult {
    specular: vec3<f32>,
    hit_pos: vec3<f32>,
    dir: vec3<f32>,
    pdf: f32,
    hit: bool,
    valid: bool,
}

fn trace_specular(pos: vec2<i32>, noise: vec3<f32>, ls_pos: vec3<f32>, ls_normal: vec3<f32>, ls_hit_normal: vec3<f32>, roughness: f32) -> TraceResult {
    let camera_pos = (model.inv_transform * vec4<f32>(environment.camera.ws_position, 1.0)).xyz;
    let wi = normalize(camera_pos - ls_pos);

    // stole this from the bf1 talk https://www.youtube.com/watch?v=ncUNLDQZMzQ
    // we "cut off" the vndf lobe edges which cuts down the noise on rougher surfaces significantly
    // technically messes with the importance sampling but the spatial filter fixes it mostly
    let trace_roughness = max(0.02, roughness * 0.5);

    const HALTON_16: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2(0.500000, 0.333333),
        vec2(0.250000, 0.666667),
        vec2(0.750000, 0.111111),
        vec2(0.125000, 0.444444),
    );

    // here, we importance sample the visible ggx smith distribution up to 4 times
    // reflected sample directions aren't guaranteed to be above the horizon,
    // so we "reroll" if invalid, as in the SEED presentation
    // https://media.contentapi.ea.com/content/dam/ea/seed/presentations/cedec2018-towards-effortless-photorealism-through-real-time-raytracing.pdf
    // since i only want to trace specular rays for low roughness values, this isn't really all that contentious
    var dir: vec3<f32>;
    var pdf: f32;
    var above_horizon = false;
    for (var i = 0u; i < 4u; i++) {
        let sample = fract(HALTON_16[i] + noise.xy);

        let H = sample_vndf_ggx(sample, wi, ls_normal, trace_roughness);

        dir = normalize(reflect(-wi, H));
        if dot(ls_normal, dir) > 0.0 {
            let ndh = max(dot(ls_normal, H), 0.0);
            pdf = vndf_ggx_pdf(wi, H, ls_normal, trace_roughness);

            above_horizon = true;
            break;
        }
    }
    if !above_horizon {
        // couldn't find one, just return 0 here
        // shouldn't happen enough to be noticable
        var res: TraceResult;
        res.pdf = -1.0;
        res.specular = vec3(1.0, 0.0, 0.0);
        return res;
    }

    var in: Ray;
    in.origin = ls_pos + environment.shadow_bias * ls_hit_normal;
    in.direction = dir;

    let hit = raymarch(in);

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

        let ls_hit_pos = in.origin + in.direction * hit.depth;
        let ws_hit_pos = (model.transform * vec4(ls_hit_pos, 1.0)).xyz;

        var res: TraceResult;
        res.valid = true;
        res.specular = diffuse;
        res.hit = true;
        res.hit_pos = ws_hit_pos;
        res.dir = dir;
        res.pdf = pdf;
        return res;
    } else {
        let ws_ray_dir = normalize((model.transform * vec4(dir, 0.0)).xyz);
        let rot_dir = vec3(
            ws_ray_dir.x * environment.skybox_rotation.x - ws_ray_dir.y * environment.skybox_rotation.y,
            ws_ray_dir.x * environment.skybox_rotation.y + ws_ray_dir.y * environment.skybox_rotation.x,
            ws_ray_dir.z,
        );
        var sky_color = textureSampleLevel(tex_skybox, sampler_linear, rot_dir.xzy, 0.0).rgb;
        sky_color = min(sky_color, vec3(15.0)) * environment.indirect_sky_intensity;

        var res: TraceResult;
        res.valid = true;
        res.specular = sky_color;
        res.hit = false;
        res.dir = dir;
        res.pdf = pdf;
        return res;
    }
}

// importance samples the visible portion of the ggx distribution
// successor to Heitz' VNDF for isotropic materials, which is all i've got rn
// https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0#file-samplevndf_ggx-cpp
fn sample_vndf_ggx(noise: vec2<f32>, wi: vec3<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;

    let wi_local = to_tangent(wi, N);
    let V = normalize(vec3(a * wi_local.xy, wi_local.z));

    let phi = (2.0 * noise.x - 1.0) * PI;
    let z = (1.0 - noise.y) * (1.0 + V.z) - V.z;
    let sin_t = sqrt(saturate(1.0 - z * z));
    let x = sin_t * cos(phi);
    let y = sin_t * sin(phi);

    let h_std = vec3(x, y, z) + V;
    let h_local = normalize(vec3(a * h_std.xy, max(h_std.z, 0.0)));

    return align_direction(h_local, N);
}

fn vndf_ggx_pdf(V: vec3<f32>, H: vec3<f32>, N: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a_sq = a * a;

    let ndh = max(dot(N, H), 0.0);
    let ndv = max(dot(N, V), 1e-6);

    let d = ndh * ndh * (a_sq - 1.0) + 1.0;
    let ndf_ggx = a_sq / (PI * d * d);

    let g1 = 2.0 * ndv / (ndv + sqrt(a_sq + (1.0 - a_sq) * ndv * ndv));

    return g1 * ndf_ggx / (4.0 * ndv);
}

fn to_tangent(v: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var T: vec3<f32>;
    var B: vec3<f32>;
    if N.z < 0.0 {
        let a = 1.0 / (1.0 - N.z);
        let b = N.x * N.y * a;
        T = vec3(1.0 - N.x * N.x * a, -b, N.x);
        B = vec3(b, N.y * N.y * a - 1.0, -N.y);
    } else {
        let a = 1.0 / (1.0 + N.z);
        let b = -N.x * N.y * a;
        T = vec3(1.0 - N.x * N.x * a, b, -N.x);
        B = vec3(b, 1.0 - N.y * N.y * a, -N.y);
    }
    // dot with each basis vector = project into tangent frame
    return vec3(dot(v, T), dot(v, B), dot(v, N));
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

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
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

fn unpack_voxel_lighting(packed: array<u32, 3>) -> VoxelLighting {
    let irr_rg = unpack2x16float(packed[0]);
    let irr_b_shadow = unpack2x16float(packed[1]);

    var res: VoxelLighting;
    res.irradiance = vec3(irr_rg, irr_b_shadow.r);
    res.shadow = irr_b_shadow.y;
    res.ao = f32(packed[2] >> 16u) / 65535.0;
    res.history_length = packed[2] & 0xFFFFu;
    return res;
}

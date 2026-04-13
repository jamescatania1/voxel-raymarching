override min_roughness: f32 = 0.001;
override dielectric_specular: f32 = 0.04;

struct VoxelLighting {
    irradiance: vec3<f32>,
    shadow: f32,
    variance: f32,
    history_length: u32,
}

@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_voxel_id: texture_storage_2d<r32uint, read>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(2) var tex_specular: texture_storage_2d<rgba16float, read>;
@group(1) @binding(3) var tex_radiance_sh_r: texture_2d<f32>;
@group(1) @binding(4) var tex_radiance_sh_g: texture_2d<f32>;
@group(1) @binding(5) var tex_radiance_sh_b: texture_2d<f32>;
@group(1) @binding(6) var tex_radiance_history: texture_storage_2d<r32uint, read>;

@group(2) @binding(0) var sampler_linear: sampler;
@group(2) @binding(1) var sampler_noise: sampler;
@group(2) @binding(2) var tex_skybox: texture_cube<f32>;
@group(2) @binding(3) var tex_irradiance: texture_cube<f32>;
@group(2) @binding(4) var tex_prefilter: texture_cube<f32>;
@group(2) @binding(5) var tex_brdf_lut: texture_2d<f32>;
@group(2) @binding(6) var<storage, read> voxel_lighting: array<array<u32,3>>;
@group(2) @binding(7) var<storage, read> voxel_shadow_mask: array<u32>;

struct VoxelSceneMetadata {
    size: vec3<u32>,
    bounding_size: u32,
    probe_size: vec3<u32>,
    probe_scale: f32,
    index_levels: u32,
    index_chunk_count: u32,
}
@group(2) @binding(8) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(9) var tex_probe_irradiance: texture_2d<f32>;
@group(2) @binding(10) var tex_probe_depth: texture_2d<f32>;
@group(2) @binding(11) var<storage, read> probes: array<vec3<f32>>;

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

var<private> sky_ray_dir: vec3<f32>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

fn evaluate_irradiance(uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    let sh_r = textureSampleLevel(tex_radiance_sh_r, sampler_linear, uv, 0.0);
    let sh_g = textureSampleLevel(tex_radiance_sh_g, sampler_linear, uv, 0.0);
    let sh_b = textureSampleLevel(tex_radiance_sh_b, sampler_linear, uv, 0.0);

    let basis = vec4(
        0.886226925,
        1.023326708 * normal.y,
        1.023326708 * normal.z,
        1.023326708 * normal.x,
    );

    let res = vec3(
        dot(sh_r, basis),
        dot(sh_g, basis),
        dot(sh_b, basis)
    );
    return max(res, vec3(0.0));
}

fn evaluate_irradiance_zh3(uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {

    // debug show disocclusions
    let pos = vec2<u32>(vec2<f32>(textureDimensions(tex_radiance_history).xy) * uv);
    let history_len = textureLoad(tex_radiance_history, pos).r;
    if history_len <= 1u {
        // return vec3(1.0, 0.0, 0.0);
    }

    let sh_r = textureSampleLevel(tex_radiance_sh_r, sampler_linear, uv, 0.0);
    let sh_g = textureSampleLevel(tex_radiance_sh_g, sampler_linear, uv, 0.0);
    let sh_b = textureSampleLevel(tex_radiance_sh_b, sampler_linear, uv, 0.0);

    const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma_l1 = vec3(
        dot(vec3(sh_r.w, sh_g.w, sh_b.w), LUMA_WEIGHTS),
        dot(vec3(sh_r.y, sh_g.y, sh_b.y), LUMA_WEIGHTS),
        dot(vec3(sh_r.z, sh_g.z, sh_b.z), LUMA_WEIGHTS),
    );
    let zonal_axis = normalize(luma_l1);

    let l1_r = vec3(sh_r.w, sh_r.y, sh_r.z);
    let l1_g = vec3(sh_g.w, sh_g.y, sh_g.z);
    let l1_b = vec3(sh_b.w, sh_b.y, sh_b.z);

    var ratio = vec3(
        abs(dot(l1_r, zonal_axis)),
        abs(dot(l1_g, zonal_axis)),
        abs(dot(l1_b, zonal_axis)),
    );
    let dc = vec3(sh_r.x, sh_g.x, sh_b.x);
    ratio /= dc;

    let zonal_l2_coeff = dc * (0.08 * ratio + 0.6 * ratio * ratio);
    let f_z = dot(zonal_axis, normal);
    let zh_dir = sqrt(5.0 / (16.0 * PI)) * (3.0 * f_z * f_z - 1.0);

    let basis = vec4(
        0.886226925,
        1.023326708 * normal.y,
        1.023326708 * normal.z,
        1.023326708 * normal.x,
    );

    var res = vec3(
        dot(sh_r, basis),
        dot(sh_g, basis),
        dot(sh_b, basis)
    );
    res += 0.25 * zonal_l2_coeff * zh_dir;

    return max(res, vec3(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_albedo).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let uv_jittered = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    sky_ray_dir = vec3(
        ray.ws_direction.x * environment.skybox_rotation.x - ray.ws_direction.y * environment.skybox_rotation.y,
        ray.ws_direction.x * environment.skybox_rotation.y + ray.ws_direction.y * environment.skybox_rotation.x,
        ray.ws_direction.z,
    );

    let depth = textureLoad(tex_depth, pos).r;
    if depth < 0.0 {
        var sky = textureSampleLevel(tex_skybox, sampler_linear, sky_ray_dir.xzy, 0.0).rgb;
        textureStore(out_color, vec2<i32>(in.id.xy), vec4(sky, 1.0));
        return;
    }

    let voxel_id = textureLoad(tex_voxel_id, pos).r;

    let lighting = unpack_voxel_lighting(voxel_lighting[voxel_id]);

    let velocity = textureLoad(tex_velocity, pos).rg;
    let packed = textureLoad(tex_normal, pos).r;
    let voxel = unpack_voxel(packed);

    let albedo_sample = textureLoad(tex_albedo, pos);
    let albedo = albedo_sample.rgb;
    let specular = textureLoad(tex_specular, pos).rgb;

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    let shadow_occluded = (voxel_shadow_mask[voxel_id >> 5u] & (1u << (voxel_id & 31u))) != 0u;

    let ls_normal = normalize(model.inv_normal_transform * voxel.ws_normal);
    // let irradiance = sample_irradiance(ls_pos, ls_normal);
    // let irradiance = sample_irradiance(ls_pos, voxel.ls_hit_normal);
    let irradiance = evaluate_irradiance_zh3(uv, voxel.ws_normal);

    var surface: PbrInput;
    surface.uv = uv;
    surface.ws_pos = ws_pos;
    surface.ws_normal = voxel.ws_normal;
    surface.albedo = albedo * select(1.0, voxel.emissive_intensity * 5.0, voxel.is_emissive);
    surface.metallic = voxel.metallic;
    surface.roughness = max(voxel.roughness, min_roughness);
    // surface.shadow = lighting.shadow;
    surface.shadow = select(0.0, 1.0, shadow_occluded);
    // surface.irradiance = lighting.irradiance;
    surface.irradiance = irradiance;
    surface.specular = specular;
    var color = pbr(surface);

    switch environment.debug_view {
        case 1u: {
            color = surface.albedo;
        }
        case 2u {
            color = vec3(depth) * 0.005;
        }
        case 3u {
            color = voxel.ls_hit_normal;
        }
        case 4u {
            color = voxel.ws_normal;
        }
        case 5u {
            color = vec3(surface.roughness);
        }
        case 6u {
            color = vec3(surface.metallic);
        }
        case 7u {
            color = vec3(surface.shadow);
        }
        case 8u {
            color = surface.irradiance * 0.1;
        }
        case 9u {
            // color = vec3(surface.specular);
            // color = sample_irradiance(ls_pos, voxel.ls_hit_normal);
            color = evaluate_irradiance(uv, voxel.ws_normal) * 0.1;
        }
        case 10u {
            // color = vec3(abs(velocity), 0.0);
            // color = sample_irradiance_debug();
            let pos = vec2<u32>(vec2<f32>(textureDimensions(tex_radiance_history).xy) * uv);
            let history_len = textureLoad(tex_radiance_history, pos).r;
            color = vec3(f32(history_len) / f32(128.0));
        }
        case 11u {
            color = textureSampleLevel(tex_skybox, sampler_linear, sky_ray_dir, 0.0).rgb;
        }
        case 12u {
            // color = textureSampleLevel(tex_irradiance, sampler_linear, sky_ray_dir, 0.0).rgb;
            color = vec3(select(0.0, 1.0, shadow_occluded));
        }
        case 13u {
            let t = cos(f32(frame.frame_id) / 300.0) * 0.5 + 0.5;
            color = textureSampleLevel(tex_prefilter, sampler_linear, sky_ray_dir, t * 5.0).rgb;
        }
        default {}
    }

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

struct PbrInput {
    uv: vec2<f32>,
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    specular: vec3<f32>,
    shadow: f32,
    irradiance: vec3<f32>,
    // ao: f32,
}

fn pbr(in: PbrInput) -> vec3<f32> {
    let N = in.ws_normal;
    let V = normalize(environment.camera.ws_position - in.ws_pos);
    let L = normalize(environment.sun_direction);
    let H = normalize(V + L);
    let R = reflect(-V, N);

    var direct: vec3<f32>;
    {
        let f_0 = mix(vec3(0.04), in.albedo, in.metallic);
        let k_s = fresnel(f_0, H, V);
        let k_d = (1.0 - k_s) * (1.0 - in.metallic);

        let ndf = normal_distribution(in.roughness, N, H);
        let geom = geom_smith(in.roughness, N, V, L);

        let ndv = max(dot(N, V), 0.0);
        let ndl = max(dot(N, L), 0.0);

        let diffuse = k_d * in.albedo / PI;

        let specular = ndf * geom * k_s / max(4.0 * ndv * ndl, 0.000001);

        let brdf = diffuse + specular;
        direct = brdf * environment.sun_color * environment.sun_intensity * in.shadow * ndl;
    }
    var indirect: vec3<f32>;
    {
        let f_0 = mix(vec3(0.04), in.albedo, in.metallic);
        let k_s = fresnel_roughness(f_0, N, V, in.roughness);
        let k_d = (1.0 - k_s) * (1.0 - in.metallic);

        let ndv = clamp(dot(N, V), 0.0001, 1.0);
        let vdh = saturate(dot(V, H));

        let sky_reflect_dir = vec3(
            R.x * environment.skybox_rotation.x - R.y * environment.skybox_rotation.y,
            R.x * environment.skybox_rotation.y + R.y * environment.skybox_rotation.x,
            R.z,
        );
        let sky_prefilter = textureSampleLevel(tex_prefilter, sampler_linear, sky_reflect_dir.xzy, in.roughness * 4.0).rgb * environment.indirect_sky_intensity;
        let sky_brdf = textureSampleLevel(tex_brdf_lut, sampler_linear, vec2(ndv, in.roughness), 0.0).rg;

        let diffuse = k_d * in.irradiance * in.albedo / PI;

        // let smoothing = exp2(-16.0 * in.roughness - 1.0);
        // let specular_occlusion = saturate(pow(ndv + 1.0 - in.ao, smoothing) - 1.0 + (1.0 - in.ao));
        // let ibl_occluded = specular_occlusion * sky_prefilter;

        let specular = sky_prefilter * 0.0 * (f_0 * sky_brdf.x + sky_brdf.y);
        // let specular = in.specular * (f_0 * sky_brdf.x + sky_brdf.y);

        indirect = diffuse + specular;
    }
    let res = direct + indirect;
    return res;
}

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz
fn normal_distribution(r: f32, N: vec3<f32>, H: vec3<f32>) -> f32 {
    let ndh = max(dot(N, H), 0.0);
    let d = ndh * ndh * (r * r - 1.0) + 1.0;

    return pow(r, 2.0) / max(PI * d * d, 0.000001);
}

// Smith model
fn geom_smith(r: f32, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> f32 {
    return geom_ggx(r, N, V) * geom_ggx(r, N, L);
}

// Schlick/Beckman geometry shadowing
fn geom_ggx(r: f32, N: vec3<f32>, x: vec3<f32>) -> f32 {
    let ndx = max(dot(N, x), 0.0);

    let k = r / 2.0;
    let den = ndx * (1.0 - k) + k;

    return ndx / max(den, 0.000001);
}

// Fresnel/Schick approximation
fn fresnel(f0: vec3<f32>, H: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    let ndv = max(dot(H, V), 0.0);
    return f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
}

fn fresnel_roughness(f0: vec3<f32>, N: vec3<f32>, V: vec3<f32>, roughness: f32) -> vec3<f32> {
    let ndv = max(dot(N, V), 0.01);
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - ndv, 5.0);
}

struct Ray {
    ls_origin: vec3<f32>,
    ws_direction: vec3<f32>,
    direction: vec3<f32>,
};

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
    ray.ws_direction = ws_direction;
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
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

/// ------------------------------------------------------
/// --------------- irradiance probe utils ---------------

const PROBE_DEPTH_SCALE: f32 = 1.0 / 40.0;

fn sample_irradiance_debug() -> vec3<f32> {
    let pos = (model.inv_transform * vec4(environment.camera.ws_position, 1.0)).xyz;
    // samples the nearest irradiance probe as if it is the skybox
    let grid_pos = vec3<i32>(round(pos / scene.probe_scale));
    let probe = get_probe(grid_pos);
    if !probe.exists {
        return vec3(1.0, 0.0, 0.0);
    }

    let ws_pos = (model.transform * vec4(probe.pos, 1.0)).xyz;
    let ls_dir = normalize(sky_ray_dir.xzy);
    let val = sample_probe(probe.id, ls_dir, ls_dir).irradiance;

    // let grid_base_pos = (vec3<f32>(grid_base) + 0.5) * scene.probe_scale;

    return val;
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
        // w *= max(dot(dir_unbiased, normal), 0.0);
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

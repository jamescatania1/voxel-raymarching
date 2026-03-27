@group(0) @binding(0) var tex_specular: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var tex_direction_pdf: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;

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
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
    @builtin(local_invocation_id) thread_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
}

// shout out the goat once again
// https://www.youtube.com/watch?v=MyTOGHqyquU

// this is generated from taking a blue noise texture, partitioning it into four ranges [0.0 < 0.25 < 0.5 < 0.75 < 1.0]
// and then using the 16 closest offsets from the center for each as a poisson kernel, getting us 4 disjoint kernels
// these offsets correspond with the kernel offsets of the downsampled specular rays
// since we're sampling a quarter res texture, each upsampled texel in the 2x2 uses a separate kernel
//
// these are also sorted nearest first, so if we ever do anything adaptive just take the first N
const SPATIAL_KERNELS: array<array<vec2<i32>, 16>, 4> = array<array<vec2<i32>, 16>, 4>(
    array<vec2<i32>, 16>(
        vec2(1, -1), vec2(1, 1), vec2(0, 2), vec2(0, -2), vec2(-2, 1), vec2(-2, -2), vec2(-3, 1), vec2(-3, -1), vec2(3, -1), vec2(2, 3), vec2(3, 2), vec2(0, -4), vec2(-1, 4), vec2(-3, 3), vec2(2, -4), vec2(4, -2)
    ),
    array<vec2<i32>, 16>(
        vec2(1, 0), vec2(0, -1), vec2(-1, 1), vec2(-2, 0), vec2(-2, -1), vec2(2, 1), vec2(0, 3), vec2(3, 0), vec2(1, -3), vec2(1, 3), vec2(-3, 2), vec2(-2, -3), vec2(0, 4), vec2(4, 1), vec2(-1, -4), vec2(1, -4)
    ),
    array<vec2<i32>, 16>(
        vec2(0, 0), vec2(-1, 0), vec2(-1, 2), vec2(1, -2), vec2(2, -1), vec2(2, 2), vec2(0, -3), vec2(-3, 0), vec2(-1, 3), vec2(2, -3), vec2(4, 0), vec2(-4, -1), vec2(3, 3), vec2(-4, 2), vec2(-2, 4), vec2(-2, -4)
    ),
    array<vec2<i32>, 16>(
        vec2(0, 1), vec2(-1, -1), vec2(2, 0), vec2(1, 2), vec2(-1, -2), vec2(2, -2), vec2(-2, 2), vec2(3, 1), vec2(-1, -3), vec2(-3, -2), vec2(-2, 3), vec2(3, -2), vec2(-4, 0), vec2(1, 4), vec2(-4, 1), vec2(2, 4)
    ),
);
const PI: f32 = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let res = filter_spatial(in);
    textureStore(tex_out, in.id.xy, vec4<f32>(res, 1.0));
}

fn filter_spatial(in: ComputeIn) -> vec3<f32> {
    let dimensions = textureDimensions(tex_depth);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let pos = vec2<i32>(in.id.xy);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let jitter = texel_size * select(vec2(0.0), environment.camera.jitter - 0.5, frame.taa_enabled != 0u);

    let surface = gather_surface(uv + jitter, pos);
    if surface.is_sky {
        return vec3(0.0);
    }
    let V = normalize(environment.camera.ws_position - surface.ws_pos);
    let N = surface.ws_normal;
    let ndv = max(dot(N, V), 1e-4);

    let half_pos = pos / 2;
    let kernel = SPATIAL_KERNELS[((in.id.x & 1u) << 1u) | (in.id.y & 1u)];

    var res = vec3(0.0);
    var weight_sum = 0.0;

    for (var i = 0u; i < 16u; i++) {
        let sample_pos = half_pos + kernel[i];

        let specular = textureLoad(tex_specular, sample_pos).rgb;

        let dir_pdf = textureLoad(tex_direction_pdf, sample_pos);
        let ls_dir = dir_pdf.xyz;
        let dir = normalize((model.normal_transform * ls_dir));
        let pdf = dir_pdf.w;

        if pdf <= 0.0 {
            continue; // only happens if getting a sample failed
        }

        let H = normalize(V + dir);
        let ndh = max(dot(N, H), 0.0);
        let ndl = max(dot(N, dir), 0.0);
        let vdh = max(dot(V, H), 0.0);

        if ndl <= 0.0 {
            continue;
        }

        let brdf = brdf(ndh, ndv, ndl, surface.roughness);

        let weight = min(brdf * ndl / pdf, 1.0);
        res += specular * weight;
        weight_sum += weight;
    }
    res /= max(weight_sum, 1e-6);

    // some bs math to trust the raw result for lower roughness
    // otherwise low roughness surfaces have no reflections visible, since
    // the pdf's go towards 0 for smooth surfaces
    let center = textureLoad(tex_specular, half_pos).rgb;
    let confidence = saturate(weight_sum);
    let blend = confidence * smoothstep(0.05, 0.25, surface.roughness);
    res = mix(center, res, blend);

    return res;
}

fn brdf(ndh: f32, ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a_sq = a * a;

    let d = ndh * ndh * (a_sq - 1.0) + 1.0;
    let ndf = a_sq / max(PI * d * d, 1e-6);

    let geom = geom_ggx(ndv, a_sq) * geom_ggx(ndl, a_sq);

    return ndf * geom / max(4.0 * ndv * ndl, 1e-6);
}
fn geom_ggx(ndx: f32, a_sq: f32) -> f32 {
    return 2.0 * ndx / max(ndx + sqrt(a_sq + (1.0 - a_sq) * ndx * ndx), 1e-6);
}

struct SurfaceData {
    is_sky: bool,
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    ws_hit_normal: vec3<f32>,
    roughness: f32,
}
fn gather_surface(uv: vec2<f32>, pos: vec2<i32>) -> SurfaceData {
    let depth = textureLoad(tex_depth, pos).r;
    let packed = textureLoad(tex_normal, pos).r;

    if depth < 0.0 {
        var res: SurfaceData;
        res.is_sky = true;
        return res;
    }

    let ray = primary_ray(uv, environment.camera.inv_view_proj);

    let voxel = unpack_voxel(packed);

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    let ws_hit_normal = normalize(model.normal_transform * voxel.ls_hit_normal);

    var res: SurfaceData;
    res.ws_pos = ws_pos;
    res.ws_normal = voxel.ws_normal;
    res.ws_hit_normal = ws_hit_normal;
    res.roughness = voxel.roughness;
    return res;
}

struct Ray {
    ls_origin: vec3<f32>,
    ws_direction: vec3<f32>,
    direction: vec3<f32>,
};

fn primary_ray(uv: vec2<f32>, inv_view_proj: mat4x4<f32>) -> Ray {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
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

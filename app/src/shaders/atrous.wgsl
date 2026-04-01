override g_phi_color: f32 = 10.0;
override g_phi_normal: f32 = 128.0;

struct FilterData {
    step_size: u32,
}
@group(0) @binding(0) var<uniform> filter_data: FilterData;
@group(0) @binding(1) var tex_out_illum: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var tex_in_illum: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_history_len: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(2) var tex_depth: texture_storage_2d<r32float, read>;

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
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_in_illum).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let jitter = texel_size * select(vec2(0.0), environment.camera.jitter - 0.5, frame.taa_enabled != 0u);

    let center_val = textureLoad(tex_in_illum, pos);
    let center_luma = center_val.r;

    let center_depth = textureLoad(tex_depth, pos).r;
    let center_packed = textureLoad(tex_normal, pos).r;

    let center = gather_surface(uv + jitter, environment.camera.inv_view_proj, center_depth, center_packed);
    if center.is_sky {
        textureStore(tex_out_illum, pos, center_val);
        return;
    }

    let center_history_len = textureLoad(tex_history_len, pos).r;
    let center_variance = gather_variance(pos);

    let phi_luma = g_phi_color * sqrt(max(0.0, center_variance + 1e-10));
    let phi_depth = f32(filter_data.step_size) * max(center_depth, 1e-8);

    var res_luma = 1.0;
    var res_val = center_val;

    const KERNEL: array<f32, 3> = array<f32, 3>(1.0, 2.0 / 3.0, 1.0 / 6.0);

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let sample_pos = pos + vec2<i32>(x, y) * i32(filter_data.step_size);
            if any(sample_pos < vec2(0)) || any(sample_pos >= dimensions) || (x == 0 && y == 0) {
                continue;
            }

            let kernel = KERNEL[abs(x)] * KERNEL[abs(y)];

            let sample_val = textureLoad(tex_in_illum, sample_pos);
            let sample_luma = sample_val.r;

            let sample_depth = textureLoad(tex_depth, sample_pos).r;
            let sample_packed = textureLoad(tex_normal, sample_pos).r;

            let sample_uv = (vec2<f32>(sample_pos) + 0.5) * texel_size;
            let sample = gather_surface(sample_uv + jitter, environment.camera.inv_view_proj, sample_depth, sample_packed);

            if sample.is_sky {
                continue;
            }

            let weight = kernel * edge_weights(center, sample, center_luma, sample_luma, phi_luma);

            res_luma += weight;
            res_val += sample_val * vec4<f32>(vec3(weight), weight * weight);
        }
    }

    let res = res_val / vec4<f32>(vec3<f32>(res_luma), res_luma * res_luma);
    textureStore(tex_out_illum, pos, res);
}

fn gather_variance(pos: vec2<i32>) -> f32 {
    const KERNEL: array<array<f32, 2>, 2> = array<array<f32, 2>, 2>(
        array<f32, 2>(1.0 / 4.0, 1.0 / 8.0),
        array<f32, 2>(1.0 / 8.0, 1.0 / 16.0)
    );
    var res = 0.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sample_pos = pos + vec2<i32>(x, y);
            let k = KERNEL[abs(x)][abs(y)];

            res += textureLoad(tex_in_illum, sample_pos).a * k;
        }
    }

    return res;
}

fn edge_weights(center: SurfaceData, sample: SurfaceData, center_luma: f32, sample_luma: f32, phi_luma: f32) -> f32 {
    let weight_normal = pow(saturate(dot(center.ws_hit_normal, sample.ws_hit_normal)), g_phi_normal);
    // let weight_z 
    let weight_luma = abs(center_luma - sample_luma) / phi_luma;

    let res = exp(0.0 - max(weight_luma, 0.0)) * weight_normal;
    // let res = 1.0;

    return res;
}

struct SurfaceData {
    is_sky: bool,
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    ws_hit_normal: vec3<f32>,
}

fn gather_surface(uv: vec2<f32>, inv_view_proj: mat4x4<f32>, depth: f32, packed: u32) -> SurfaceData {
    if depth < 0.0 {
        var res: SurfaceData;
        res.is_sky = true;
        return res;
    }

    let ray = primary_ray(uv, inv_view_proj);

    let voxel = unpack_voxel(packed);

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    let ls_hit_normal = normalize(-vec3<f32>(sign(ray.direction)) * vec3<f32>(voxel.hit_mask));
    let ws_hit_normal = normalize(model.normal_transform * ls_hit_normal);

    var res: SurfaceData;
    res.ws_pos = ws_pos;
    res.ws_normal = voxel.ws_normal;
    res.ws_hit_normal = ws_hit_normal;
    return res;
}

struct Ray {
    ls_origin: vec3<f32>,
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
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
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

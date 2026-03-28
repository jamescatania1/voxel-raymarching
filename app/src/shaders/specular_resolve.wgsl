@group(0) @binding(0) var main_sampler: sampler;
@group(0) @binding(1) var tex_cur: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_specular_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_velocity: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var tex_acc: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(3) var tex_prev_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(4) var tex_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(5) var tex_prev_depth: texture_storage_2d<r32float, read>;

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

const MAX_HISTORY_LENGTH: u32 = 200;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
    @builtin(local_invocation_id) thread_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let cur_pos = vec2<i32>(in.id.xy);
    let dimensions = textureDimensions(tex_depth);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let cur_uv = (vec2<f32>(cur_pos) + 0.5) * texel_size;

    let acc = reproject(cur_pos);

    var res_color: vec3<f32>;
    var history_len: f32;

    let cur_color = textureLoad(tex_cur, cur_pos).rgb;

    if acc.reject_history {
        res_color = cur_color;
        history_len = 1.0;
    } else {
        history_len = min(f32(MAX_HISTORY_LENGTH), acc.history_len + 1.0);

        let history_ratio = history_len / f32(MAX_HISTORY_LENGTH);
        let k = mix(0.5, 3.0, history_ratio);

        let acc_sample = acc.color;

        let alpha_color = 1.0 / f32(history_len);
        res_color = mix(acc_sample, cur_color, alpha_color);
    }

    textureStore(tex_out, cur_pos, vec4<f32>(res_color, history_len));
}

struct ReprojectResult {
    reject_history: bool,
    history_len: f32,
    color: vec3<f32>,
    moments: vec2<f32>,
}

fn reproject(cur_pos: vec2<i32>) -> ReprojectResult {
    let dimensions = vec2<i32>(textureDimensions(tex_depth).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let cur_jitter = texel_size * select(vec2(0.0), environment.camera.jitter - 0.5, frame.taa_enabled != 0u);
    let acc_jitter = texel_size * select(vec2(0.0), environment.prev_camera.jitter - 0.5, frame.taa_enabled != 0u);

    let cur_uv = (vec2<f32>(cur_pos) + 0.5) * texel_size;

    let cur = gather_surface(cur_uv + cur_jitter, cur_pos, false);

    var res: ReprojectResult;
    res.reject_history = true;
    res.history_len = 0.0;
    res.color = vec3(0.0);
    res.moments = vec2(0.0);

    if cur.is_sky {
        return res;
    }

    let virtual_velocity = textureLoad(tex_specular_velocity, cur_pos / 2).rg;
    let surface_velocity = textureLoad(tex_velocity, cur_pos).rg;

    // blend between standard diffuse velocity -> parallax specular velocity based on roughness (1-0)
    // stepping roughness seems to work alright here
    // it's all a big old approximation anyways
    let roughness_weight = select(0.0, 1.0, cur.roughness > 0.3);
    // let velocity = mix(virtual_velocity, surface_velocity, roughness_weight);
    let velocity = surface_velocity;

    let acc_uv = cur_uv - velocity;
    let acc_pos = acc_uv * vec2<f32>(dimensions);

    {
        // start with 2x2 bilinear filter

        const BILINEAR_OFFSET: array<vec2<i32>, 4> = array<vec2<i32>, 4>(
            vec2<i32>(0, 0),
            vec2<i32>(1, 0),
            vec2<i32>(0, 1),
            vec2<i32>(1, 1),
        );

        let f = fract(acc_pos - 0.5);
        let w = array<f32, 4>(
            (1.0 - f.x) * (1.0 - f.y),
            f.x * (1.0 - f.y),
            (1.0 - f.x) * f.y,
            f.x * f.y,
        );

        var w_sum = 0.0;
        var acc_sum = vec3(0.0);
        var acc_moments = vec2(0.0);

        for (var i = 0u; i < 4u; i++) {
            let offset = vec2<f32>(BILINEAR_OFFSET[i]);

            let sample_pos = acc_pos + offset - 0.5;
            let sample_texel = vec2<i32>(floor(sample_pos));
            let sample_uv = (vec2<f32>(sample_texel) + 0.5) * texel_size;

            if any(sample_pos < vec2(0.0)) || any(sample_pos >= vec2<f32>(dimensions)) {
                continue;
            }

            let acc = gather_surface(sample_uv + acc_jitter, sample_texel, true);

            if is_reprojection_valid(cur, acc) {
                let weight = w[i];

                acc_sum += weight * textureLoad(tex_acc, sample_texel).rgb;
                acc_moments += weight * textureLoad(tex_acc, sample_texel).rg;
                w_sum += weight;
            }
        }

        if w_sum > 0.001 {
            res.reject_history = false;
            res.color = acc_sum / w_sum;
            res.moments = acc_moments / w_sum;
        }
    }

    if !res.reject_history {
        res.history_len = textureLoad(tex_acc, vec2<i32>(acc_pos)).a;
    }

    return res;
}

fn is_reprojection_valid(cur: SurfaceData, acc: SurfaceData) -> bool {
    return true;
    if acc.is_sky {
        return false;
    }

    let plane_distance = abs(dot(cur.ws_pos - acc.ws_pos, cur.ws_hit_normal));
    if plane_distance > 0.15 {
        return false;
    }

    if pow(abs(dot(cur.ws_hit_normal, acc.ws_hit_normal)), 2.0) < 1.0 {
        return false;
    }

    return true;
}

struct SurfaceData {
    is_sky: bool,
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    ws_hit_normal: vec3<f32>,
    roughness: f32,
}

fn gather_surface(uv: vec2<f32>, pos: vec2<i32>, is_prev: bool) -> SurfaceData {
    var depth: f32;
    var packed: u32;
    var inv_view_proj: mat4x4<f32>;
    if is_prev {
        depth = textureLoad(tex_prev_depth, pos).r;
        packed = textureLoad(tex_prev_normal, pos).r;
        inv_view_proj = environment.prev_camera.inv_view_proj;
    } else {
        depth = textureLoad(tex_depth, pos).r;
        packed = textureLoad(tex_normal, pos).r;
        inv_view_proj = environment.camera.inv_view_proj;
    }

    if depth < 0.0 {
        var res: SurfaceData;
        res.is_sky = true;
        return res;
    }

    let ray = primary_ray(uv, inv_view_proj);

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

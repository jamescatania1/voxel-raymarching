@group(0) @binding(0) var main_sampler: sampler;
@group(0) @binding(1) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_cur_illum: texture_storage_2d<r32uint, read>;

@group(1) @binding(0) var tex_out_illum: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var tex_acc_illum: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var tex_out_moments: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var tex_acc_moments: texture_storage_2d<rgba16float, read>;
@group(1) @binding(4) var tex_out_history_len: texture_storage_2d<r32uint, write>;
@group(1) @binding(5) var tex_acc_history_len: texture_storage_2d<r32uint, read>;
@group(1) @binding(6) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(7) var tex_prev_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(8) var tex_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(9) var tex_prev_depth: texture_storage_2d<r32float, read>;

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
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
    @builtin(local_invocation_id) thread_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
}

var<workgroup> horizontal_masks: array<array<u32, 24>, 8>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let cur_pos = vec2<i32>(in.id.xy);
    let dimensions = textureDimensions(tex_velocity);

    if in.thread_id.y < 6u {
        let base = in.group_id.y << 1u;
        let tile = i32(base + in.thread_id.y) - 2;

        let slice = gather_horizontal_masks(in.id.x, u32(tile));

        for (var y = 0u; y < 4u; y++) {
            horizontal_masks[in.thread_id.x][(in.thread_id.y << 2u) + y] = slice[y];
        }
    }

    workgroupBarrier();

    if environment.filter_shadows == 0u {
        let shadow = (horizontal_masks[in.thread_id.x][in.thread_id.y + 8u] >> 8u) & 1u;
        textureStore(tex_out_illum, cur_pos, vec4<f32>(f32(shadow), 1.0, 1.0, 1.0));
        return;
    }

    var cur = 0.0;
    var mu = 0.0;

    for (var i = 0u; i < 17u; i++) {
        let nbr_mask = horizontal_masks[in.thread_id.x][i + in.thread_id.y];
        let nbr_count = countOneBits(nbr_mask);

        mu += f32(nbr_count) / 17.0;

        if i > 6u && i < 10u {
            let cur_mask = (nbr_mask >> 7u) & 7u;
            let cur_count = countOneBits(cur_mask);
            cur += f32(cur_count) / 3.0;
        }
    }
    cur /= 3.0;
    mu /= 17.0;

    let sigma = sqrt(mu * (1.0 - mu));

    let cur_color = vec3<f32>(cur);
    var cur_moments: vec2<f32>;

    cur_moments.r = cur_color.r;
    cur_moments.g = cur_moments.r * cur_moments.r;

    let acc = reproject(cur_pos);

    var res_color: vec3<f32>;
    var res_moments: vec2<f32>;
    var history_len: u32;

    if acc.reject_history {
        res_color = cur_color;
        res_moments = cur_moments;
        history_len = 1u;
    } else {
        history_len = min(32u, acc.history_len + 1u);

        let history_ratio = f32(history_len) / 32.0;
        let k = mix(0.5, 3.0, history_ratio);

        let acc_min = mu - k * sigma;
        let acc_max = mu + k * sigma;

        let acc_sample = clamp(acc.color, vec3(acc_min), vec3(acc_max));

        // TODO add clamping parameters to customize each alpha
        let alpha_moments = 1.0 / f32(history_len);
        res_moments = mix(acc.moments, cur_moments, alpha_moments);

        let alpha_color = 1.0 / f32(history_len);
        res_color = mix(acc_sample, cur_color, alpha_color);
    }

    // let variance = max(0.0, res_moments.g - res_moments.r * res_moments.r);
    // textureStore(tex_out_moments, cur_pos, vec4<f32>(res_moments, 0.0, 1.0));

    textureStore(tex_out_illum, cur_pos, vec4<f32>(res_color, 1.0));
    textureStore(tex_out_history_len, cur_pos, vec4<u32>(history_len, 0, 0, 0));
}

fn gather_horizontal_masks(pos_x: u32, tile_y: u32) -> array<u32, 4> {
    let dimensions = textureDimensions(tex_velocity);

    let base = vec2(pos_x >> 3u, tile_y);

    let x = pos_x & 7u;

    let mask_l = textureLoad(tex_cur_illum, base - vec2(1, 0)).r;
    let mask_c = textureLoad(tex_cur_illum, base).r;
    let mask_r = textureLoad(tex_cur_illum, base + vec2(1, 0)).r;

    var res: array<u32, 4>;

    for (var y = 0u; y < 4u; y++) {
        let shift = y << 3u;

        let l = (mask_l >> shift) & 0xFFu;
        let c = (mask_c >> shift) & 0xFFu;
        let r = (mask_r >> shift) & 0xFFu;

        let mask = (l << 16u) | (c << 8u) | r;
        let window = (mask >> (7 - x)) & 0x1FFFFu;

        res[y] = window;
    }
    return res;
}

struct ReprojectResult {
    reject_history: bool,
    history_len: u32,
    color: vec3<f32>,
    moments: vec2<f32>,
}

fn reproject(cur_pos: vec2<i32>) -> ReprojectResult {
    let dimensions = vec2<i32>(textureDimensions(tex_velocity).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let cur_jitter = texel_size * select(vec2(0.0), environment.camera.jitter - 0.5, frame.taa_enabled != 0u);
    let acc_jitter = texel_size * select(vec2(0.0), environment.prev_camera.jitter - 0.5, frame.taa_enabled != 0u);

    let cur_uv = (vec2<f32>(cur_pos) + 0.5) * texel_size;

    let cur_depth = textureLoad(tex_depth, cur_pos).r;
    let cur_packed = textureLoad(tex_normal, cur_pos).r;
    let cur = gather_surface(cur_uv + cur_jitter, environment.camera.inv_view_proj, cur_depth, cur_packed);

    let velocity = textureLoad(tex_velocity, cur_pos).rg;

    let acc_uv = cur_uv - velocity;
    let acc_pos = acc_uv * vec2<f32>(dimensions);

    var res: ReprojectResult;
    res.reject_history = true;
    res.history_len = 0u;
    res.color = vec3(0.0);
    res.moments = vec2(0.0);

    if cur.is_sky {
        return res;
    }

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

            let acc_depth = textureLoad(tex_prev_depth, sample_texel).r;
            let acc_packed = textureLoad(tex_prev_normal, sample_texel).r;
            let acc = gather_surface(sample_uv + acc_jitter, environment.prev_camera.inv_view_proj, acc_depth, acc_packed);

            if is_reprojection_valid(cur, acc) {
                let weight = w[i];

                acc_sum += weight * textureLoad(tex_acc_illum, sample_texel).rgb;
                acc_moments += weight * textureLoad(tex_acc_moments, sample_texel).rg;
                w_sum += weight;
            }
        }

        if w_sum > 0.001 {
            res.reject_history = false;
            res.color = acc_sum / w_sum;
            res.moments = acc_moments / w_sum;
        }
    }
    // if res.reject_history {
    //     // now we fall back to cross-bilateral

    //     var w_sum = 0.0;
    //     var acc_sum = vec4(0.0);

    //     for (var y = -1; y <= 1; y++) {
    //         for (var x = -1; x <= 1; x++) {
    //             let sample_pos = acc_pos + vec2(f32(x), f32(y)) - 0.5;
    //             let sample_texel = vec2<i32>(sample_pos);
    //             let sample_uv = (vec2<f32>(sample_texel) - 0.5) * texel_size;

    //             if any(sample_pos < vec2(0.0)) || any(sample_pos >= vec2<f32>(dimensions)) {
    //                 continue;
    //             }

    //             let acc_depth = textureLoad(tex_prev_depth, sample_texel).r;
    //             let acc_packed = textureLoad(tex_prev_normal, sample_texel).r;
    //             let acc = gather_surface(sample_uv + acc_jitter, environment.prev_camera.inv_view_proj, acc_depth, acc_packed);

    //             if is_reprojection_valid(cur, acc) {
    //                 acc_sum += textureLoad(tex_acc_illum, sample_texel, 0);
    //                 w_sum += 1.0;
    //             }
    //         }
    //     }

    //     if (w_sum > 0.001) {
    //         res.reject_history = false;
    //         res.color = acc_sum / w_sum;
    //     }
    // }

    if !res.reject_history {
        res.history_len = textureLoad(tex_acc_history_len, vec2<i32>(acc_pos)).r;
    }

    return res;
}

fn is_reprojection_valid(cur: SurfaceData, acc: SurfaceData) -> bool {
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

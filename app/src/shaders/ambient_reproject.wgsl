@group(0) @binding(0) var tex_sh_r: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var tex_sh_g: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_sh_b: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(4) var tex_out_sh_r: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(5) var tex_out_sh_g: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(6) var tex_out_sh_b: texture_storage_2d<rgba16float, read_write>;

@group(1) @binding(0) var tex_out_history: texture_storage_2d<r32uint, write>;
@group(1) @binding(1) var tex_prev_sh_r: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var tex_prev_sh_g: texture_storage_2d<rgba16float, read>;
@group(1) @binding(3) var tex_prev_sh_b: texture_storage_2d<rgba16float, read>;
@group(1) @binding(4) var tex_prev_history: texture_storage_2d<r32uint, read>;
@group(1) @binding(5) var tex_cur_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(6) var tex_cur_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(7) var tex_prev_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(8) var tex_prev_depth: texture_storage_2d<r32float, read>;

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

const PI: f32 = 3.14159265359;

const MAX_HISTORY_LEN: u32 = 64;
const PLANE_DISTANCE_THRESHOLD: f32 = 0.1;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

var<private> dimensions: vec2<i32>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    dimensions = vec2<i32>(textureDimensions(tex_sh_r).xy);

    let cur_half_pos = vec2<i32>(in.id.xy);
    if any(cur_half_pos >= dimensions) {
        return;
    }

    var sh_r = textureLoad(tex_sh_r, cur_half_pos);
    var sh_g = textureLoad(tex_sh_g, cur_half_pos);
    var sh_b = textureLoad(tex_sh_b, cur_half_pos);

    // var min_r = sh_r;
    // var max_r = sh_r;
    // var min_g = sh_g;
    // var max_g = sh_g;
    // var min_b = sh_b;
    // var max_b = sh_b;
    // for (var dy = -2; dy <= 2; dy++) {
    //     for (var dx = -2; dx <= 2; dx++) {
    //         if dx == 0 && dy == 0 {
    //             continue;
    //         }
    //         let p = cur_half_pos + vec2(dx, dy);
    //         if any(p < vec2(0)) || any(p >= dimensions_half) {
    //             continue;
    //         }

    //         // TODO use shared memory
    //         let n_r = textureLoad(tex_sh_r, p);
    //         let n_g = textureLoad(tex_sh_g, p);
    //         let n_b = textureLoad(tex_sh_b, p);

    //         min_r = min(min_r, n_r); max_r = max(max_r, n_r);
    //         min_g = min(min_g, n_g); max_g = max(max_g, n_g);
    //         min_b = min(min_b, n_b); max_b = max(max_b, n_b);
    //     }
    // }
    // // relaxing the clamp on the L1 range seems to improve things
    // // i'm not smart enough to give a real reason
    // let diff_r = vec4(0.0, max_r.yzw - min_r.yzw) * 0.25;
    // let diff_g = vec4(0.0, max_g.yzw - min_g.yzw) * 0.25;
    // let diff_b = vec4(0.0, max_b.yzw - min_b.yzw) * 0.25;
    // min_r -= diff_r; max_r += diff_r;
    // min_g -= diff_g; max_g += diff_g;
    // min_b -= diff_b; max_b += diff_b;

    var prev = reproject(cur_half_pos);

    if !prev.valid {
        textureStore(tex_out_sh_r, cur_half_pos, sh_r);
        textureStore(tex_out_sh_g, cur_half_pos, sh_g);
        textureStore(tex_out_sh_b, cur_half_pos, sh_b);
        textureStore(tex_out_history, cur_half_pos, vec4(1));
        return;
    }
    // prev.sh_r = clamp(prev.sh_r, min_r, max_r);
    // prev.sh_g = clamp(prev.sh_g, min_g, max_g);
    // prev.sh_b = clamp(prev.sh_b, min_b, max_b);

    let history_len = min(MAX_HISTORY_LEN, prev.history_len + 1);
    let alpha = 1.0 / f32(history_len);

    sh_r = mix(prev.sh_r, sh_r, alpha);
    sh_g = mix(prev.sh_g, sh_g, alpha);
    sh_b = mix(prev.sh_b, sh_b, alpha);

    textureStore(tex_out_sh_r, cur_half_pos, sh_r);
    textureStore(tex_out_sh_g, cur_half_pos, sh_g);
    textureStore(tex_out_sh_b, cur_half_pos, sh_b);
    textureStore(tex_out_history, cur_half_pos, vec4(history_len));
}

struct ReprojectResult {
    valid: bool,
    history_len: u32,
    sh_r: vec4<f32>,
    sh_g: vec4<f32>,
    sh_b: vec4<f32>,
}

const OFFSETS: array<vec2<i32>, 4> = array<vec2<i32>, 4>(
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
);

fn reproject(cur_half_pos: vec2<i32>) -> ReprojectResult {
    let texel_size_half = 1.0 / vec2<f32>(dimensions);
    let texel_size_full = texel_size_half * 0.5;

    let cur_jitter = texel_size_full * select(vec2(0.0), environment.camera.jitter - 0.5, frame.taa_enabled != 0u);
    let prev_jitter = texel_size_full * select(vec2(0.0), environment.prev_camera.jitter - 0.5, frame.taa_enabled != 0u);

    let cur_base_full_pos = cur_half_pos * 2;
    var cur_pos_full = cur_base_full_pos + OFFSETS[frame.frame_id & 3u];
    // {
    //     // get full res texel with closest depth to camera
    //     var closest_depth = -1.0;
    //     for (var i = 0; i < 4; i++) {
    //         let sample_full_pos = cur_base_full_pos + OFFSETS[i];
    //         let depth = textureLoad(tex_cur_depth, sample_full_pos).r;

    //         if depth >= 0.0 && (closest_depth < 0.0 || depth < closest_depth) {
    //             closest_depth = depth;
    //             cur_pos_full = sample_full_pos;
    //         }
    //     }
    //     if closest_depth < 0.0 { // sky
    //         return ReprojectResult();
    //     }
    // }

    let velocity = textureLoad(tex_velocity, cur_pos_full).rg;

    // let cur_uv = (vec2<f32>(cur_half_pos) + 0.5) * texel_size_half;
    let cur_uv = (vec2<f32>(cur_pos_full) + 0.5) * texel_size_full;

    let cur = gather_surface(cur_uv + cur_jitter, cur_pos_full, false);
    if cur.sky {
        return ReprojectResult();
    }

    let prev_uv = cur_uv - velocity;
    let prev_origin = vec2<i32>(floor(prev_uv * vec2<f32>(dimensions) - 0.5));

    let f = fract(prev_uv * vec2<f32>(dimensions) - 0.5);
    let w = array<f32,4>(
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y,
    );

    var weight_sum = 0.0;
    var valid_count = 0u;
    var prev_history_len = 0u;
    var prev_sum_r = vec4(0.0);
    var prev_sum_g = vec4(0.0);
    var prev_sum_b = vec4(0.0);
    for (var i = 0; i < 4; i++) {
        let offset = OFFSETS[i];

        let sample_pos = prev_origin + offset;
        if any(sample_pos < vec2(0)) || any(sample_pos >= dimensions) {
            continue;
        }

        var valid = false;
        for (var j = 0; j < 4; j++) {
            let sample_pos_full = sample_pos * 2 + OFFSETS[j];
            // let sample_uv = (vec2<f32>(sample_pos) + 0.5) * texel_size_half;
            let sample_uv = (vec2<f32>(sample_pos_full) + 0.5) * texel_size_full;

            let sample = gather_surface(sample_uv + prev_jitter, sample_pos_full, true);

            if is_reprojection_valid(cur, sample) {
                valid = true;
                break;
            }
        }
        if !valid {
            continue;
        }

        let weight = w[i];
        let history_len = textureLoad(tex_prev_history, sample_pos).r;
        if history_len == 0 {
            continue;
        }
        valid_count += 1u;
        // prev_history_len = max(prev_history_len, history_len);
        prev_history_len += history_len;
        prev_sum_r += weight * textureLoad(tex_prev_sh_r, sample_pos);
        prev_sum_g += weight * textureLoad(tex_prev_sh_g, sample_pos);
        prev_sum_b += weight * textureLoad(tex_prev_sh_b, sample_pos);
        weight_sum += weight;
    }

    if weight_sum < 0.001 {
        return ReprojectResult();
    }
    var res: ReprojectResult;
    res.valid = true;
    res.history_len = u32(ceil(f32(prev_history_len) / f32(valid_count)));
    // res.history_len = prev_history_len;
    res.sh_r = prev_sum_r / weight_sum;
    res.sh_g = prev_sum_g / weight_sum;
    res.sh_b = prev_sum_b / weight_sum;
    return res;
}

fn is_reprojection_valid(cur: Surface, prev: Surface) -> bool {
    if prev.sky {
        return false;
    }

    let cur_distance = length(cur.pos - environment.camera.ws_position);
    let plane_dist = abs(dot(cur.pos - prev.pos, cur.hit_normal));
    if plane_dist > PLANE_DISTANCE_THRESHOLD * cur_distance {
        return false;
    }

    // if dot(cur.normal, prev.normal) < NORMAL_THRESHOLD {
    //     return false;
    // }

    return true;
}

struct Surface {
    sky: bool,
    pos: vec3<f32>,
    normal: vec3<f32>,
    hit_normal: vec3<f32>,
}

fn gather_surface(uv: vec2<f32>, pos: vec2<i32>, prev: bool) -> Surface {
    var depth: f32;
    var packed: u32;
    var inv_view_proj: mat4x4<f32>;
    if prev {
        depth = textureLoad(tex_prev_depth, pos).r;
        packed = textureLoad(tex_prev_normal, pos).r;
        inv_view_proj = environment.prev_camera.inv_view_proj;
    } else {
        depth = textureLoad(tex_cur_depth, pos).r;
        packed = textureLoad(tex_cur_normal, pos).r;
        inv_view_proj = environment.camera.inv_view_proj;
    }

    if depth < 0.0 {
        var res: Surface;
        res.sky = true;
        return res;
    }

    let ray = primary_ray(uv, inv_view_proj);

    let voxel = unpack_voxel(packed);

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    let ws_hit_normal = normalize(model.normal_transform * voxel.ls_hit_normal);

    var res: Surface;
    res.pos = ws_pos;
    res.normal = voxel.ws_normal;
    res.hit_normal = ws_hit_normal;
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
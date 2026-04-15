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
@group(0) @binding(0) var<uniform> environment: Environment;
@group(0) @binding(1) var<uniform> frame: FrameMetadata;
@group(0) @binding(2) var<uniform> model: Model;

@group(1) @binding(0) var main_sampler: sampler;
@group(1) @binding(1) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var tex_color: texture_2d<f32>;

@group(2) @binding(0) var tex_acc: texture_2d<f32>;
@group(2) @binding(1) var out_color: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var tex_depth: texture_2d<f32>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const ACC_ALPHA: f32 = 0.1;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = vec2<i32>(textureDimensions(tex_color).xy);
    let pos = vec2<i32>(in.id.xy);

    var color = vec3<f32>();
    if frame.taa_enabled == 0u || frame.frame_id == 0u {
        color = textureLoad(tex_color, pos, 0).rgb;
    } else {
        color = taa(in);
    }

    textureStore(out_color, pos, vec4(color, 1.0));
}

fn taa(in: ComputeIn) -> vec3<f32> {
    let dimensions = vec2<i32>(textureDimensions(tex_color).xy);
    let pos = vec2<i32>(in.id.xy);

    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;

    var near_depth = 100000.0;
    var near_depth_pos = pos;
    for (var x = -1; x <= 1; x += 1) {
        for (var y = -1; y <= 1; y += 1) {
            let sample_pos = clamp(pos + vec2(x, y), vec2(0), dimensions);

            let depth = textureLoad(tex_depth, sample_pos, 0).r;
            if depth < near_depth && depth >= 0.0 {
                near_depth = depth;
                near_depth_pos = sample_pos;
            }
        }
    }

    let velocity = textureLoad(tex_velocity, near_depth_pos).rg;
    var cur_color = textureLoad(tex_color, pos, 0).rgb;

    let acc_uv = uv - velocity;
    if any(acc_uv < vec2(0.0)) || any(acc_uv >= vec2(1.0)) {
        return cur_color;
    }

    cur_color = rgb_to_ycocg(cur_color);

    var min_color = cur_color;
    var max_color = cur_color;
    var avg_color = vec3(0.0);
    for (var x = -1; x <= 1; x += 1) {
        for (var y = -1; y <= 1; y += 1) {
            let sample_pos = clamp(pos + vec2(x, y), vec2(0), dimensions);

            let sample = rgb_to_ycocg(textureLoad(tex_color, sample_pos, 0).rgb);
            min_color = min(min_color, sample);
            max_color = max(max_color, sample);
            avg_color += sample;
        }
    }
    avg_color /= 9.0;

    var acc_color = sample_catmull_rom_5(acc_uv, vec2<f32>(dimensions));
    acc_color = rgb_to_ycocg(acc_color);
    acc_color = clamp(acc_color, min_color, max_color);

    let res = mix(acc_color, cur_color, ACC_ALPHA);
    return ycocg_to_rgb(res);
}

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    const R1: vec3<f32> = vec3<f32>(0.25, 0.5, 0.25);
    const R2: vec3<f32> = vec3<f32>(0.5, 0.0, -0.5);
    const R3: vec3<f32> = vec3<f32>(-0.25, 0.5, -0.25);
    return vec3<f32>(
        dot(rgb, R1),
        dot(rgb, R2),
        dot(rgb, R3),
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    const R1: vec3<f32> = vec3<f32>(1.0, 1.0, -1.0);
    const R2: vec3<f32> = vec3<f32>(1.0, 0.0, 1.0);
    const R3: vec3<f32> = vec3<f32>(1.0, -1.0, -1.0);
    return vec3<f32>(
        dot(ycocg, R1),
        dot(ycocg, R2),
        dot(ycocg, R3),
    );
}

fn clip_aabb(bds_min: vec3<f32>, bds_max: vec3<f32>, p: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
    let r_min = bds_min - p;
    let r_max = bds_max - p;

    const EPSILON: f32 = 1.0e-8;
    var r = q - p;

    if r.x > r_max.x {
        r *= r_max.x / (r.x + EPSILON);
    }
    if r.y > r_max.y {
        r *= r_max.y / (r.y + EPSILON);
    }
    if r.z > r_max.z {
        r *= r_max.z / (r.z + EPSILON);
    }

    if r.x < r_min.x {
        r *= r_min.x / (r.x + EPSILON);
    }
    if r.y < r_min.y {
        r *= r_min.y / (r.y + EPSILON);
    }
    if r.z < r_min.z {
        r *= r_min.z / (r.z + EPSILON);
    }

    return p + r;
}

// 5-tap approximation of of Catmull-Rom filter
// very similar results to 9-tap
// from https://advances.realtimerendering.com/s2016/Filmic%20SMAA%20v7.pptx
fn sample_catmull_rom_5(uv: vec2<f32>, dimensions: vec2<f32>) -> vec3<f32> {
    let texel_size = 1.0 / dimensions;
    let pos = uv * dimensions;
    let center_pos = floor(pos - 0.5) + 0.5;
    let f = pos - center_pos;
    let f2 = f * f;
    let f3 = f * f2;

    const SHARPNESS: f32 = 0.4;
    let c = SHARPNESS;
    let w0 = -c * f3 + 2.0 * c * f2 - c * f;
    let w1 = (2.0 - c) * f3 - (3.0 - c) * f2 + 1.0;
    let w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c) * f2 + c * f;
    let w3 = c * f3 - c * f2;

    let w12 = w1 + w2;
    let tc12 = texel_size * (center_pos + w2 / w12);
    let center_color = textureSampleLevel(tex_acc, main_sampler, tc12.xy, 0.0).rgb;

    let tc0 = texel_size * (center_pos - 1.0);
    let tc3 = texel_size * (center_pos + 2.0);

    var color = vec4(0.0);
    color += vec4(textureSampleLevel(tex_acc, main_sampler, vec2(tc12.x, tc0.y), 0.0).rgb, 1.0) * (w12.x * w0.y);
    color += vec4(textureSampleLevel(tex_acc, main_sampler, vec2(tc0.x, tc12.y), 0.0).rgb, 1.0) * (w0.x * w12.y);
    color += vec4(center_color, 1.0) * (w12.x * w12.y);
    color += vec4(textureSampleLevel(tex_acc, main_sampler, vec2(tc3.x, tc12.y), 0.0).rgb, 1.0) * (w3.x * w12.y);
    color += vec4(textureSampleLevel(tex_acc, main_sampler, vec2(tc12.x, tc3.y), 0.0).rgb, 1.0) * (w12.x * w3.y);

    let res = color.rgb / color.a;
    return max(res, vec3(0.0));
}

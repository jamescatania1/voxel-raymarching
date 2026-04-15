@group(0) @binding(0) var tex_sh_r: texture_2d<f32>;
@group(0) @binding(1) var tex_sh_g: texture_2d<f32>;
@group(0) @binding(2) var tex_sh_b: texture_2d<f32>;

@group(1) @binding(0) var tex_out_sh_r: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var tex_out_sh_g: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var tex_out_sh_b: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var tex_history_len: texture_storage_2d<r32uint, read>;
@group(1) @binding(4) var tex_depth: texture_2d<f32>;

@group(2) @binding(0) var sampler_linear: sampler;
@group(2) @binding(1) var tex_noise: texture_3d<f32>;

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
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const PI: f32 = 3.14159265359;

const POISSON_KERNEL: array<vec2<f32>, 8> = array(
    vec2(0.000, 0.000),
    vec2(0.527, 0.154),
    vec2(0.135, 0.593),
    vec2(-0.440, 0.413),
    vec2(-0.626, -0.083),
    vec2(-0.189, -0.627),
    vec2(0.368, -0.551),
    vec2(0.787, -0.239),
);

const OFFSETS: array<vec2<i32>, 4> = array<vec2<i32>, 4>(
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
);

// based on fast denoising w/ self stabilizing recurrent blurs,
// see https://developer.nvidia.com/gtc/2020/video/s22699-vid

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = textureDimensions(tex_out_sh_r).xy;
    if any(in.id.xy >= dimensions) {
        return;
    }

    var is_sky = true;
    var cur_depth = 9999999.0;
    for (var i = 0u; i < 4u; i++) {
        let depth = textureLoad(tex_depth, pos * 2 + OFFSETS[i], 0).r;
        if depth > 0.0 {
            is_sky = false;
            cur_depth = min(cur_depth, depth);
        }
    }
    if is_sky {
        return;
    }

    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;

    let history_length = textureLoad(tex_history_len, pos).r;

    // clamped how small the kernel can shrink
    // looks better in my case, might revisit though
    var history_weight = 1.0 / (1.0 + min(f32(history_length) * 0.25, 8.0));

    var radius = texel_size * environment.ambient_filter_scale;
    radius *= history_weight;

    let noise = textureLoad(tex_noise, vec3(in.id.xy % vec2(128), frame.frame_id % 64), 0).r;
    let t = noise * 2.0 * PI;
    let cos_t = cos(t);
    let sin_t = sin(t);

    var weight = 0.0;
    var sh_r = vec4(0.0);
    var sh_g = vec4(0.0);
    var sh_b = vec4(0.0);

    for (var i = 0; i < 8; i++) {
        var offset = POISSON_KERNEL[i] * radius;
        offset = vec2(
            cos_t * offset.x - sin_t * offset.y,
            sin_t * offset.x + cos_t * offset.y
        );

        let sample_uv = uv + offset;
        if any(sample_uv < vec2(0.0)) || any(sample_uv > vec2(1.0)) {
            continue;
        }

        var w = 1.0;

        let depths = textureGather(0, tex_depth, sampler_linear, sample_uv);
        let diffs = abs(depths - cur_depth);
        var depth_weight = min(min(min(diffs.x, diffs.y), diffs.z), diffs.w);
        depth_weight = saturate(1.0 - depth_weight * 50.0);

        // w *= depth_weight;

        weight += w;
        sh_r += w * textureSampleLevel(tex_sh_r, sampler_linear, sample_uv, 0.0);
        sh_g += w * textureSampleLevel(tex_sh_g, sampler_linear, sample_uv, 0.0);
        sh_b += w * textureSampleLevel(tex_sh_b, sampler_linear, sample_uv, 0.0);
    }

    if weight > 0.001 {
        sh_r /= weight;
        sh_g /= weight;
        sh_b /= weight;
    }
    textureStore(tex_out_sh_r, pos, sh_r);
    textureStore(tex_out_sh_g, pos, sh_g);
    textureStore(tex_out_sh_b, pos, sh_b);
}
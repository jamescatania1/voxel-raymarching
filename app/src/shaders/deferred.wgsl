@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_2d<f32>;
@group(0) @binding(2) var tex_normal: texture_2d<f32>;
@group(0) @binding(3) var tex_depth: texture_2d<f32>;
@group(0) @binding(4) var tex_velocity: texture_2d<f32>;
@group(0) @binding(5) var main_sampler: sampler;

// struct Environment {
//     sun_direction: vec3<f32>,
//     shadow_bias: f32,
//     camera: Camera,
//     prev_camera: Camera,
// }
// struct Camera {
//     view_proj: mat4x4<f32>,
//     inv_view_proj: mat4x4<f32>,
//     ws_position: vec3<f32>,
//     forward: vec3<f32>,
//     near: f32,
//     far: f32,
//     fov: f32,
// }
// @group(2) @binding(0) var<uniform> environment: Environment;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const ACC_ALPHA: f32 = 0.2;

const AMBIENT_INTENSITY: f32 = 0.2;
const DIRECTIONAL_INTENSITY: f32 = 0.8;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_normal).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = vec2<f32>(in.id.xy) * texel_size;

    let albedo_sample = textureSampleLevel(tex_albedo, main_sampler, uv, 0.0);
    let albedo = albedo_sample.rgb;
    let shadow_factor = albedo_sample.a;
    let normal = normalize(textureSampleLevel(tex_normal, main_sampler, uv, 0.0).rgb);
    let depth = textureSampleLevel(tex_depth, main_sampler, uv, 0.0).rgb;
    let velocity = textureSampleLevel(tex_velocity, main_sampler, uv, 0.0).rg;

    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

    let diff = max(dot(normal, ws_light_dir), 0.0);

    var color = albedo * (diff * DIRECTIONAL_INTENSITY * (1.0 - shadow_factor) + AMBIENT_INTENSITY);
    // color *= 0.000001;
    // color += normal;
    // if shadow_factor > 0.1 {
    //     color.r = 1.0;
    // }

    // var src_sample_total = vec3<f32>(0.0);
    // var src_weight = 0.0;
    // var neighborhood_min = vec3<f32>(1000.0);
    // var neighborhood_max = vec3<f32>(-1000.0);
    // var m1 = vec3<f32>(0.0);
    // var m2 = vec3<f32>(0.0);
    // var closest_depth = 0.0;
    // var closest_depth_pixel = vec2<i32>(0);
    // for (var dx = -1; dx >= 1; dx += 1) {
    //     for (var dy = -1; dy >= 1; dy += 1) {
    //         for (var dz = -1; dz >= 1; dz += 1) {
    //             let pixel_pos = clamp(vec2<i32>(in.id.xy) + vec2<i32>(dx, dy), vec2(0), vec2<i32>(dimensions) - 1);

    //             let neighbor = max(vec3(0.0),
    //         }
    //     }
    // }

    // let prev_color = textureSampleLevel(acc_color, main_sampler, uv, 0.0).rgb;
    // color = color * ACC_ALPHA + prev_color * (1.0 - ACC_ALPHA);

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

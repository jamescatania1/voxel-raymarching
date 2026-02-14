struct Environment {
    sun_direction: vec3<f32>,
    shadow_bias: f32,
    camera: Camera,
    prev_camera: Camera,
}
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
}
struct FrameMetadata {
    frame_id: u32,
}
@group(0) @binding(0) var<uniform> environment: Environment;
@group(0) @binding(1) var<uniform> frame: FrameMetadata;

@group(1) @binding(0) var main_sampler: sampler;
@group(1) @binding(1) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(1) @binding(2) var tex_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(3) var tex_color: texture_storage_2d<rgba16float, read>;

@group(2) @binding(0) var tex_acc: texture_2d<f32>;
@group(2) @binding(1) var out_color: texture_storage_2d<rgba16float, write>;


struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const ACC_ALPHA: f32 = 0.05;


@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = vec2<i32>(textureDimensions(tex_acc).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = vec2<f32>(in.id.xy) * texel_size;

    // let albedo_sample = textureSampleLevel(tex_albedo, main_sampler, uv, 0.0);
    // let albedo = albedo_sample.rgb;
    // let shadow_factor = albedo_sample.a;
    // let normal = normalize(textureSampleLevel(tex_normal, main_sampler, uv, 0.0).rgb);
    // let depth = textureSampleLevel(tex_depth, main_sampler, uv, 0.0).rgb;
    // let velocity = textureSampleLevel(tex_velocity, main_sampler, uv, 0.0).rg;

    // let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

    // let diff = max(dot(normal, ws_light_dir), 0.0);

    // var color = albedo * (diff * DIRECTIONAL_INTENSITY * (1.0 - shadow_factor) + AMBIENT_INTENSITY);
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
    let pos = vec2<i32>(in.id.xy);

    if frame.frame_id == 0u {
        let cur_color = textureLoad(tex_color, vec2<i32>(in.id.xy)).rgb;
        textureStore(out_color, pos, vec4(cur_color, 1.0));
        return;
    }

    var src_sample_total = vec3<f32>(0.0);
    var src_sample_weight = 0.0;
    var neighborhood_min = vec3<f32>(9999.0);
    var neighborhood_max = vec3<f32>(-9999.0);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var closest_depth = 1.0;
    var closest_depth_pixel = vec2<i32>(0);

    for (var dx = -1; dx <= 1; dx += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            let pixel_offset = vec2<i32>(dx, dy);
            let pixel_pos = clamp(pos + pixel_offset, vec2(0), dimensions - 1);

            let neighbor = max(vec3(0.0), textureLoad(tex_color, pixel_pos).rgb);
            let sub_sample_distance = length(vec2<f32>(pixel_offset));
            let sub_sample_weight = filter_mitchell(sub_sample_distance);

            src_sample_total += neighbor * sub_sample_weight;
            src_sample_weight += sub_sample_weight;

            neighborhood_min = min(neighborhood_min, neighbor);
            neighborhood_max = max(neighborhood_max, neighbor);

            m1 += neighbor;
            m2 += neighbor * neighbor;

            let cur_depth = textureLoad(tex_depth, pixel_pos).r;
            if cur_depth < closest_depth {
                closest_depth = cur_depth;
                closest_depth_pixel = pixel_pos;
            }
        }
    }

    let velocity = textureLoad(tex_velocity, closest_depth_pixel).xy * 0.5;
    let acc_pixel_pos = vec2<f32>(pos) - velocity;
    var src_sample = src_sample_total / src_sample_weight;

    if any(acc_pixel_pos < vec2(0.0)) || any(acc_pixel_pos > vec2<f32>(dimensions)) {
        textureStore(out_color, pos, vec4(1.0, 0.0, 0.0, 1.0));
        // textureStore(out_color, pos, vec4(src_sample, 1.0));
        return;
    }

    // TODO: catmull rom filter
    var acc_sample = textureSampleLevel(tex_acc, main_sampler, acc_pixel_pos / vec2<f32>(dimensions), 0.0).rgb;

    let gamma = 1.0;
    let mu = m1 / 9.0;
    let sigma = sqrt(abs((m2 / 9.0) - (mu * mu)));
    let min_c = mu - gamma * sigma;
    let max_c = mu + gamma * sigma;

    // acc_sample = clip_aabb(min_c, max_c, clamp(acc_sample, neighborhood_min, neighborhood_max));

    let luma_src = luminance(src_sample);
    let luma_acc = luminance(acc_sample);

    let src_weight = ACC_ALPHA / (1.0 + luma_src);
    let acc_weight = (1.0 - ACC_ALPHA) / (1.0 + luma_acc);

    let color = (src_sample * src_weight + acc_sample * acc_weight) / max(src_weight + acc_weight, 1e-5);

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

fn filter_mitchell(d: f32) -> f32 {
    const B: f32 = 1.0 / 3.0;
    const C: f32 = 1.0 / 3.0;
    let x = d * 1.0;

    let x2 = x * x;
    let x3 = x * x * x;
    if x < 1.0 {
        return (12.0 - 9.0 * B - 6.0 * C) * x3 + (-18.0 + 12.0 * B + 6.0 * C) * x2 + (6.0 - 2.0 * B);
    } else {
        return (-B - 6.0 * C) * x3 + (6.0 * B + 30.0 * C) * x2 + (-12.0 * B - 48.0 * C) * x + (8.0 * B + 24.0 * C);
    }
}

fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, prev_sample: vec3<f32>) -> vec3<f32> {
    // 1. Find the center of the AABB (this effectively creates a ray from center to history)
    let p_clip = 0.5 * (aabb_max + aabb_min);

    // 2. Get the half-extents of the box
    // Add a tiny epsilon to avoid division by zero if min == max
    let e_clip = 0.5 * (aabb_max - aabb_min) + vec3<f32>(0.00000001);

    // 3. Calculate the vector from the center to the history sample
    let v_clip = prev_sample - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);

    // 4. Determine the "furthest" dimension relative to the box size
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    // 5. If we are outside the box (ratio > 1.0), scale the vector back to the edge
    if (ma_unit > 1.0) {
        return p_clip + v_clip / ma_unit;
    } else {
        return prev_sample; // Inside the box, keep it
    }
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// fn sample_texture_catmull_rom(
//     tex: texture_2d<f32>,
//     samp: sampler,
//     uv: vec2<f32>,
//     tex_size: vec2<f32>
// ) -> vec3<f32> {
//     // We work in unnormalized coordinates to make math easier
//     let sample_pos = uv * tex_size;
//     let tex_pos1 = floor(sample_pos - 0.5) + 0.5;

//     // Compute weights
//     let f = sample_pos - tex_pos1;
//     let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
//     let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
//     let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
//     let w3 = f * f * (-0.5 + 0.5 * f);

//     // Compute weighted coordinates
//     let w12 = w1 + w2;
//     let offset12 = w2 / (w1 + w2);

//     let tex_pos0 = tex_pos1 - 1.0;
//     let tex_pos3 = tex_pos1 + 2.0;
//     let tex_pos12 = tex_pos1 + offset12;

//     // Convert back to UVs
//     let inv_tex_size = 1.0 / tex_size;
//     let uv0 = vec2<f32>(tex_pos1.x - 1.0, tex_pos1.y - 1.0) * inv_tex_size; // Top-left is actually irrelevant for 5-tap if optimized further, but here's the 5-tap logic:

//     // 5-Tap Approximation Strategy:
//     // We usually combine weights to reduce samples.
//     // However, a full separated axis implementation is often cleaner in code.
//     // Below is the specific 5-tap bilinear-exploitation method adapted for WGSL.

//     // Recalculate for 5-tap specifically:
//     let tc = floor(sample_pos - 0.5) + 0.5;
//     let f = sample_pos - tc;

//     let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
//     let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
//     let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
//     let w3 = f * f * (-0.5 + 0.5 * f);

//     let w12 = w1 + w2;
//     let offset12 = w2 / w12;

//     let tc0 = (tc - 1.0) * inv_tex_size;
//     let tc3 = (tc + 2.0) * inv_tex_size;
//     let tc12 = (tc + offset12) * inv_tex_size;

//     // Sample using linear sampler
//     // Center rows
//     let result =
//         textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y) +
//         textureSampleLevel(tex, samp, vec2<f32>(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y) +
//         textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y) + // Center
//         textureSampleLevel(tex, samp, vec2<f32>(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y) +
//         textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);

//     // Normalizing the result isn't strictly necessary if weights sum to 1,
//     // but Catmull-Rom weights usually do.
//     // Note: This 5-tap is an approximation.

//     return max(result, vec3<f32>(0.0)); // Prevent negative ringing
// }

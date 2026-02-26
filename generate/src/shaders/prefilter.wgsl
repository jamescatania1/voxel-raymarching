struct PrefilterData {
    roughness: f32,
    sample_count: u32,
};

@group(0) @binding(0) var tex_in: texture_cube<f32>;
@group(0) @binding(1) var tex_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var sampler_main: sampler;
@group(0) @binding(3) var<uniform> data: PrefilterData;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
};

const PI = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let cube_dim: vec2<u32> = textureDimensions(tex_in);
    let prefilter_dim: vec2<u32> = textureDimensions(tex_out);

    let pos: vec3<f32> = world_pos(in.id, vec2<f32>(prefilter_dim));
    let n: vec3<f32> = normalize(pos);
    var r: vec3<f32> = n;
    var v: vec3<f32> = r;

    var weight: f32 = 0.0;
    var prefiltered_color = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < data.sample_count; i++) {
        let x_i: vec2<f32> = hammersley(i);
        let h: vec3<f32> = importance_sample_ggx(x_i, n);
        let l: vec3<f32> = normalize(2.0 * dot(v, h) * h - v);
        let ndl: f32 = max(dot(n, l), 0.0);
        if (ndl > 0.0) {
            let sample = textureSampleLevel(tex_in, sampler_main, l, 0.0).rgb;
            prefiltered_color += sample * ndl;
            weight += ndl;
        }
    }
    prefiltered_color /= weight;

    textureStore(tex_out, vec2<i32>(in.id.xy), in.id.z, vec4<f32>(prefiltered_color, 1.0));
}

fn rad_inverse_vdc(x: u32) -> f32 {
    var z = x;
    z = (z << 16u) | (z >> 16u);
    z = ((z & 0x55555555u) << 1u) | ((z & 0xAAAAAAAAu) >> 1u);
    z = ((z & 0x33333333u) << 2u) | ((z & 0xCCCCCCCCu) >> 2u);
    z = ((z & 0x0F0F0F0Fu) << 4u) | ((z & 0xF0F0F0F0u) >> 4u);
    z = ((z & 0x00FF00FFu) << 8u) | ((z & 0xFF00FF00u) >> 8u);
    return f32(z) * 2.3283064365386963e-10;
}

fn hammersley(i: u32) -> vec2<f32> {
    return vec2<f32>(f32(i)/f32(data.sample_count), rad_inverse_vdc(i));
}

fn importance_sample_ggx(x_i: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let a: f32 = data.roughness * data.roughness;
    let phi: f32 = 2.0 * PI * x_i.x;
    let cos_theta: f32 = sqrt((1.0 - x_i.y) / (1.0 + (a * a - 1.0) * x_i.y));
    let sin_theta: f32 = sqrt(1.0 - cos_theta * cos_theta);
    let h = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    var up: vec3<f32>;
    if (abs(n.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent_x: vec3<f32> = normalize(cross(up, n));
    let tangent_y: vec3<f32> = cross(n, tangent_x);
    return tangent_x * h.x + tangent_y * h.y + n * h.z;
}

fn world_pos(cubemap_pos: vec3<u32>, cubemap_dim: vec2<f32>) -> vec3<f32> {
    let u: f32 = (f32(cubemap_pos.x) / f32(cubemap_dim.x)) * 2.0 - 1.0;
    let v: f32 = (f32(cubemap_pos.y) / f32(cubemap_dim.y)) * 2.0 - 1.0;
    switch cubemap_pos.z {
        case 0u: { // right
            return vec3<f32>(1.0, -v, -u);
        }
        case 1u: { // left
            return vec3<f32>(-1.0, -v, u);
        }
        case 2u: { // top
            return vec3<f32>(u, 1.0, v);
        }
        case 3u: { // bottom
            return vec3<f32>(u, -1.0, -v);
        }
        case 4u: { // back
            return vec3<f32>(u, -v, 1.0);
        }
        case 5u: { // front
            return vec3<f32>(-u, -v, -1.0);
        }
        default: {
            return vec3<f32>(0.0);
        }
    }
}

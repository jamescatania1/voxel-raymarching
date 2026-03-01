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

    let N: vec3<f32> = normalize(pos);
    var R: vec3<f32> = N;
    var V: vec3<f32> = R;

    var weight: f32 = 0.0;
    var res = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < data.sample_count; i++) {
        let x_i: vec2<f32> = hammersley(i);
        
        let H: vec3<f32> = importance_sample_ggx(x_i, N, data.roughness);
        let vdh = dot(V, H);
        let ndh = saturate(vdh);

        let L = 2.0 * vdh * H - V;
        let ndl = saturate(dot(N, L));

        if (ndl > 0.0) {
            // TODO: generate mips for hdr texture and sample based on that
            // should fix up some artifacts
            // https://bruop.github.io/ibl/

            // let pdf = max(distribution_ggx(ndh, data.roughness) / 4.0, 0.001);
            // let omega = vec2(
            //     1.0 / f32(data.sample_count),
            //     4.0 * PI / (6.0 * )
            // )
            // let s = 1.0 / f32(data.sample_count);
            // let p = 4.0 * PI / (6.0 * f32(prefilter_dim.x * prefilter_dim.x));
            // let mip_level = max(0.5 * log2(s / p), 0.0);

            let sample = textureSampleLevel(tex_in, sampler_main, L, 0.0).rgb;
            res += sample * ndl;
            weight += ndl;
        }
    }
    res /= max(weight, 1.0);

    textureStore(tex_out, vec2<i32>(in.id.xy), in.id.z, vec4<f32>(res, 1.0));
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

fn importance_sample_ggx(x_i: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a: f32 = roughness * roughness;
    
    let phi: f32 = 2.0 * PI * x_i.x;
    let cos_t: f32 = sqrt((1.0 - x_i.y) / (1.0 + (a * a - 1.0) * x_i.y));
    let sin_t: f32 = sqrt(1.0 - cos_t * cos_t);
    
    let h = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);

    var up: vec3<f32>;
    if (abs(N.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent_x: vec3<f32> = normalize(cross(up, N));
    let tangent_y: vec3<f32> = cross(N, tangent_x);

    return tangent_x * h.x + tangent_y * h.y + N * h.z;
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

override sample_count: u32 = 2048;

@group(0) @binding(0) var tex_out: texture_storage_2d<rgba16float, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
};

const PI = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_out);
    var uv: vec2<f32> = (vec2<f32>(in.id.xy) + 0.5) / vec2<f32>(dimensions.xy);
    // uv.y = 1.0 - uv.y;

    let ndv = uv.x;
    let rough = uv.y;

    let v = vec3<f32>(sqrt(1.0 - ndv * ndv), 0.0, ndv);
    var res = vec2<f32>(0.0);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    for (var i: u32 = 0; i < sample_count; i++) {
        let x_i: vec2<f32> = hammersley(i);
        let h: vec3<f32> = importance_sample_ggx(x_i, n, rough);
        let l: vec3<f32> = normalize(2.0 * dot(v, h) * h - v);

        let ndl: f32 = max(l.z, 0.0);
        let ndh: f32 = max(h.z, 0.0);
        let vdh: f32 = max(dot(v, h), 0.0);

        if (ndl > 0.0) {
            let g: f32 = geometry_smith(n, v, l, rough);
            let g_vis: f32 = (g * vdh) / (ndh * ndv);
            let fc: f32 = pow(1.0 - vdh, 5.0);

            res += vec2<f32>((1.0 - fc) * g_vis, fc * g_vis);
        }
    }
    textureStore(tex_out, in.id.xy, vec4<f32>(res / f32(sample_count), 0.0, 1.0));
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
    return vec2<f32>(f32(i)/f32(sample_count), rad_inverse_vdc(i));
}

fn importance_sample_ggx(x_i: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a: f32 = roughness * roughness;
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

fn geometry_schlick_ggx(ndv: f32, rough: f32) -> f32 {
    let k: f32 = (rough * rough) / 2.0;
    return ndv / (ndv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
    let ndv: f32 = max(dot(n, v), 0.0);
    let ndl: f32 = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndv, rough) * geometry_schlick_ggx(ndl, rough);
}

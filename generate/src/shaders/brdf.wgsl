override sample_count: u32 = 2048;

@group(0) @binding(0) var tex_out: texture_storage_2d<rgba16float, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
};

const PI = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_out);
    let texel_size = 1.0 / vec2<f32>(dimensions.xy);
    var uv: vec2<f32> = (vec2<f32>(in.id.xy) + 0.5) * texel_size;

    let ndv = uv.x;
    let roughness = uv.y;
    
    let res = integrate_brdf(roughness, ndv);
    
    textureStore(tex_out, in.id.xy, vec4<f32>(res, 0.0, 1.0));


    // let ndv = uv.x;
    // let rough = uv.y;

    // let v = vec3<f32>(sqrt(1.0 - ndv * ndv), 0.0, ndv);
    // var res = vec2<f32>(0.0);
    // let n = vec3<f32>(0.0, 0.0, 1.0);

    // for (var i: u32 = 0; i < sample_count; i++) {
    //     let x_i: vec2<f32> = hammersley(i);
    //     let h: vec3<f32> = importance_sample_ggx(x_i, n, rough);
    //     let l: vec3<f32> = normalize(2.0 * dot(v, h) * h - v);

    //     let ndl: f32 = max(l.z, 0.0);
    //     let ndh: f32 = max(h.z, 0.0);
    //     let vdh: f32 = max(dot(v, h), 0.0);

    //     if (ndl > 0.0) {
    //         let v_pdf: f32 = geometry_smith(ndv, ndl, rough) * vdh * ndl / max(ndh;
    //         let g_vis: f32 = (g * vdh) / (ndh * ndv);
    //         let fc: f32 = pow(1.0 - vdh, 5.0);

    //         res += vec2<f32>((1.0 - fc) * g_vis, fc * g_vis);
    //     }
    // }
}

fn integrate_brdf(roughness: f32, ndv: f32) -> vec2<f32> {
    let N = vec3(0.0, 0.0, 1.0);
    let V = vec3(
        sqrt(1.0 - ndv * ndv), // sin
        0.0,
        ndv // cos
    );

    var a = 0.0;
    var b = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        let x_i = hammersley(i);
        
        let H = importance_sample_ggx(x_i, N, roughness);
        let L = 2.0 * dot(V, H) * H - V;

        let ndl = saturate(dot(N, L));
        let ndh = saturate(dot(N, H));
        let vdh = saturate(dot(V, H));

        if ndl > 0.0 {
            let v_pdf = geometry_smith(ndv, ndl, roughness) * vdh * ndl / max(ndh, 0.001);
            let f_c = pow(1.0 - vdh, 5.0);
            
            a += (1.0 - f_c) * v_pdf;
            b += f_c * v_pdf;
        }
    }
    
    return 4.0 * vec2<f32>(a, b) / f32(sample_count);
}

fn geometry_smith(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let a = pow(roughness, 4.0);
    let ggx_v = ndl * sqrt(ndv * ndv * (1.0 - a) + a);
    let ggx_l = ndv * sqrt(ndl * ndl * (1.0 - a) + a);
    return 0.5 / (ggx_v + ggx_l); 
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
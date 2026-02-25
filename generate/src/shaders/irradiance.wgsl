override delta: f32 = 0.025;

@group(0) @binding(0) var tex_in: texture_cube<f32>;
@group(0) @binding(1) var tex_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var sampler_main: sampler;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
};

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions: vec2<u32> = textureDimensions(tex_out);

    let pos: vec3<f32> = world_pos(in.id, vec2<f32>(dimensions));

    let normal: vec3<f32> = normalize(pos);
    var up: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(normal.z) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    var right: vec3<f32> = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    const PI = 3.14159265359;

    var irradiance = vec3<f32>(0.0);
    var samples: i32 = 0;
    for (var p: f32 = 0.0; p < 2.0 * PI; p += delta) {
        for (var t: f32 = 0.0; t < 0.5 * PI; t += delta) { 
            let tangent = vec3<f32>(sin(t) * cos(p), sin(t) * sin(p), cos(t));

            let dir = tangent.x * right + tangent.y * up + tangent.z * normal;
            let sample = textureSampleLevel(tex_in, sampler_main, dir.xyz, 0.0).rgb;

            irradiance += clamp(sample, vec3<f32>(0.0), vec3<f32>(100.0)) * cos(t) * sin(t);
            samples += 1;
        }
    }
    irradiance = PI * irradiance * (1.0 / f32(samples));

    textureStore(tex_out, vec2<i32>(in.id.xy), in.id.z, vec4<f32>(irradiance, 1.0));
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
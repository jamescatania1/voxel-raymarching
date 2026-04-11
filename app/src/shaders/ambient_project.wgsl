@group(0) @binding(0) var tex_ray_radiance: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var tex_ray_direction: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_out_sh_r: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var tex_out_sh_g: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var tex_out_sh_b: texture_storage_2d<rgba16float, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const PI: f32 = 3.14159265359;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = textureDimensions(tex_out_sh_r).xy;
    if any(in.id.xy > dimensions) {
        return;
    }

    let sample_base = (pos * 2 + 1) / 4;
    let quarter_dimensions = textureDimensions(tex_ray_radiance).xy;

    var sh_r = vec4(0.0);
    var sh_g = vec4(0.0);
    var sh_b = vec4(0.0);
    var weight = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_pos = sample_base + vec2(dx, dy);
            if any(sample_pos < vec2(0)) || any(sample_pos >= vec2<i32>(quarter_dimensions)) {
                continue;
            }

            let radiance = textureLoad(tex_ray_radiance, sample_pos).rgb;
            let dir = textureLoad(tex_ray_direction, sample_pos).rgb;

            let basis = sh2_basis(dir);

            // TODO real gaussian weights/anything spatial that makes sense
            let w = (1.0 - 0.5 * f32(abs(dx))) * (1.0 - 0.5 * f32(abs(dy)));

            sh_r += radiance.r * basis * w;
            sh_g += radiance.g * basis * w;
            sh_b += radiance.b * basis * w;
            weight += w;
        }
    }

    if weight > 0.0 {
        weight = PI / weight;
        sh_r *= weight;
        sh_g *= weight;
        sh_b *= weight;
    }
    textureStore(tex_out_sh_r, in.id.xy, sh_r);
    textureStore(tex_out_sh_g, in.id.xy, sh_g);
    textureStore(tex_out_sh_b, in.id.xy, sh_b);
}

fn sh2_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4(
        0.282094791,
        0.488602512 * d.y,
        0.488602512 * d.z,
        0.488602512 * d.x,
    );
}
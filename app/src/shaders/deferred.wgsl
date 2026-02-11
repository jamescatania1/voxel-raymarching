@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_2d<f32>;
@group(0) @binding(2) var tex_normal: texture_2d<f32>;
@group(0) @binding(3) var tex_depth: texture_2d<f32>;
@group(0) @binding(4) var main_sampler: sampler;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const AMBIENT_INTENSITY: f32 = 0.02;
const DIRECTIONAL_INTENSITY: f32 = 1.0;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_normal).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(in.id.xy) + 0.5) * texel_size;

    let albedo_sample = textureSampleLevel(tex_albedo, main_sampler, uv, 0.0);
    let albedo = albedo_sample.rgb;
    let shadow_factor = albedo_sample.a;
    let normal = normalize(textureSampleLevel(tex_normal, main_sampler, uv, 0.0).rgb);
    let depth = textureSampleLevel(tex_depth, main_sampler, uv, 0.0).rgb;

    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

    let diff = max(dot(normal, ws_light_dir), 0.0);

    var color = albedo * (diff * DIRECTIONAL_INTENSITY * (1.0 - shadow_factor) + AMBIENT_INTENSITY);
    // color *= 0.000001;
    // color += normal;

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

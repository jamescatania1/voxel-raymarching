@group(0) @binding(0) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_normal: texture_2d<f32>;
@group(0) @binding(2) var tex_depth: texture_2d<f32>;
@group(0) @binding(3) var main_sampler: sampler;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_normal).xy;

    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = vec2<f32>(in.id.xy) * texel_size;

    let x = uv.x + texel_size.x * 0.5;
    let y = uv.y + texel_size.y * 0.5;


    let dx = texel_size.x * 0.5;
    let dy = texel_size.y * 0.5;

    let a0 = textureSampleLevel(tex_normal, main_sampler, vec2(x, y), 0.0).rgb;
    let d0 = textureSampleLevel(tex_depth, main_sampler, vec2(x, y), 0.0).r;

    let a1 = textureSampleLevel(tex_normal, main_sampler, vec2(x - dx, y + dy), 0.0).rgb;
    let d1 = textureSampleLevel(tex_depth, main_sampler, vec2(x - dx, y + dy), 0.0).r;

    let a2 = textureSampleLevel(tex_normal, main_sampler, vec2(x + dx, y + dy), 0.0).rgb;
    let d2 = textureSampleLevel(tex_depth, main_sampler, vec2(x + dx, y + dy), 0.0).r;

    let a3 = textureSampleLevel(tex_normal, main_sampler, vec2(x - dx, y - dy), 0.0).rgb;
    let d3 = textureSampleLevel(tex_depth, main_sampler, vec2(x - dx, y - dy), 0.0).r;

    let a4 = textureSampleLevel(tex_normal, main_sampler, vec2(x + dx, y - dy), 0.0).rgb;
    let d4 = textureSampleLevel(tex_depth, main_sampler, vec2(x + dx, y - dy), 0.0).r;

    let dd_max = max(max(max(
        abs(d0 - d1),
        abs(d0 - d2)),
        abs(d0 - d3)),
        abs(d0 - d4)
    ) * 1000.0;
    let blur_fac = 1.0 - smoothstep(0.0, 1.0, dd_max);

    var res = vec3<f32>(0.0);
    res = normalize((a0 + blur_fac * (a1 + a2 + a3 + a4)) / (1.0 + blur_fac));

    res *= 0.00001;
    res += normalize(a0);

    // res += 0.2 * a0;
    // res += 0.2 * (a1 + a2 + a3 + a4);
    // res = normalize(res);

    // res *= 0.00001;
    // res += vec3(blur_fac);

    textureStore(out_normal, vec2<i32>(in.id.xy), vec4(res, 1.0));
}

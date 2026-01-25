@group(0) @binding(0) var out_texture: texture_storage_2d<rgba16float, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(out_texture).xy;

    let pixel_pos = in.id.xy;
    let uv = vec2<f32>(pixel_pos) / vec2<f32>(dimensions);

    textureStore(out_texture, vec2<i32>(in.id.xy), vec4<f32>(uv, 0.0, 1.0));
}

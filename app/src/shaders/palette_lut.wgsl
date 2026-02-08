struct Palette {
    data: array<vec4<u32>, 64>,
}
@group(0) @binding(0) var<uniform> palette: Palette;
@group(0) @binding(1) var palette_lut: texture_storage_3d<r32uint, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let color = (vec3<f32>(in.id) + 0.5) / 255.0;

    var distance = 1e10;
    var index = 0u;
    for (var i = 1u; i < 256u; i++) {
        let rgba = palette.data[i >> 2u][i & 3u];
        let c = vec3<f32>(
            f32((rgba >> 24u) & 0xFFu) / 255.0,
            f32((rgba >> 16u) & 0xFFu) / 255.0,
            f32((rgba >> 8u) & 0xFFu) / 255.0,
        );
        let d = distance(color, c);
        if d < distance {
            distance = d;
            index = i;
        }
    }

    textureStore(palette_lut, vec3<i32>(in.id), vec4<u32>(index, 0u, 0u, 0u));
}
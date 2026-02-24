@group(0) @binding(0) var tex_in: texture_2d<f32>;
@group(0) @binding(1) var tex_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var sampler_main: sampler;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
};

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {    
    // Project from the current pixel and face in the output cubemap to find
    // the equivalent sample point in the equirrectangular input texture.

    let rect_dim: vec2<u32> = textureDimensions(tex_in);
    let cube_dim: vec2<u32> = textureDimensions(tex_out);

    const PI = 3.14159265359;
    const face_transforms: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // right
        vec2<f32>(PI, 0.0), // left
        vec2<f32>(0, -PI / 2.0), // top
        vec2<f32>(0, PI / 2.0), // bottom
        vec2<f32>(-PI / 2.0, 0.0), // back
        vec2<f32>(PI / 2.0, 0.0), // front
    );
    let face_tr: vec2<f32> = face_transforms[in.id.z];

    const an: f32 = sin(PI / 4.0);
    const ak: f32 = cos(PI / 4.0);

    var nx: f32 = (f32(in.id.x) / f32(cube_dim.x)) * 2.0 - 1.0;
    var ny: f32 = (f32(in.id.y) / f32(cube_dim.y)) * 2.0 - 1.0;
    nx *= an;
    ny *= an;

    var u: f32 = 0.0;
    var v: f32 = 0.0;
    if (face_tr.y == 0.0) {
        u = atan2(nx, ak);
        v = atan2(ny * cos(u), ak);
        u += face_tr.x;
    }
    else if (face_tr.y > 0.0) {
        let d: f32 = sqrt(nx * nx + ny * ny);
        u = atan2(ny, nx);
        v = PI / 2.0 - atan2(d, ak);
    }
    else {
        let d: f32 = sqrt(nx * nx + ny * ny);
        u = atan2(-ny, nx);
        v = -PI / 2.0 + atan2(d, ak);
    }
    u /= PI;
    v /= PI / 2.0;
    while (v < -1.0) {
        v += 2.0;
        u += 1.0;
    }
    while (v > 1.0) {
        v -= 2.0;
        u += 1.0;
    }
    while (u < -1.0) {
        u += 2.0;
    }
    while (u > 1.0) {
        u -= 2.0;
    }

    var uv = vec2<f32>(u, v) / 2.0 + 0.5;

    var color: vec3<f32> = textureSampleLevel(tex_in, sampler_main, uv, 0.0).rgb;
    // var rgb = color.rgb * pow(2.0, color.a * 256.0 - 128.0);
    // rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(50.0));

    textureStore(tex_out, in.id.xy, in.id.z, vec4<f32>(color, 1.0));
}
@group(0) @binding(0) var tex_out_illum: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;

struct VoxelSceneMetadata {
    size: vec3<u32>,
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
struct Chunk {
    mask: array<u32, 16>,
}
@group(2) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(1) var<uniform> palette: Palette;
@group(2) @binding(2) var<storage, read> chunk_indices: array<u32>;
@group(2) @binding(3) var<storage, read> chunks: array<Chunk>;
@group(2) @binding(4) var tex_noise: texture_3d<f32>;
@group(2) @binding(5) var sampler_noise: sampler;

struct Environment {
	sun_direction: vec3<f32>,
	shadow_bias: f32,
	camera: Camera,
	prev_camera: Camera,
	shadow_spread: f32,
	filter_shadows: u32,
	shadow_filter_radius: f32,
	max_ambient_distance: u32,
    smooth_normal_factor: f32,
}
struct Camera {
	view_proj: mat4x4<f32>,
	inv_view_proj: mat4x4<f32>,
	ws_position: vec3<f32>,
	forward: vec3<f32>,
	near: f32,
	jitter: vec2<f32>,
	far: f32,
	fov: f32,
}
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
    fxaa_enabled: u32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
    inv_normal_transform: mat3x3<f32>,
}
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const DDA_MAX_STEPS: u32 = 300u;
const SKY_COLOR: vec3<f32> = vec3(0.5, 0.9, 1.5);

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
	let dimensions = vec2<i32>(textureDimensions(tex_depth).xy);
	let texel_size = 1.0 / vec2<f32>(dimensions);

	let uv = (vec2<f32>(pos) + 0.5) * texel_size;
	let uv_jittered  = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;
	
    let ray_length = textureLoad(tex_depth, pos).r;
    if ray_length < 0.0 {
        // primary ray missed
        textureStore(tex_out_illum, pos, vec4(1.0));
        return;
    }

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    let packed = textureLoad(tex_normal, pos).r;
    let voxel = unpack_voxel(packed);

    let ls_normal = normalize(model.inv_normal_transform * voxel.ws_normal);

    let ls_hit_normal = normalize(-vec3<f32>(sign(ray.direction)) * vec3<f32>(voxel.hit_mask));
    // let ws_hit_normal = normalize(model.normal_transform * ls_hit_normal);

    let ls_pos = ray.ls_origin + ray.direction * ray_length;

    let noise = blue_noise(in.id.xy);

    var ambient = 1.0;
    // var ambient = trace_ambient(pos, noise, ls_pos, ls_hit_normal);
    var shadow = trace_shadow(pos, noise, ls_pos, ls_normal);

    textureStore(tex_out_illum, pos, vec4(shadow, ambient, 0.0, 1.0));
}

fn trace_ambient(pos: vec2<i32>, noise: vec3<f32>, ls_pos: vec3<f32>, ls_normal: vec3<f32>) -> f32 {
    var ray: SparseRay;
    ray.origin = ls_pos + environment.shadow_bias * ls_normal;

    let dir = rand_hemisphere_direction(noise.xy, 1.0);
    ray.direction = align_direction(dir, ls_normal);

    const MAX_DISTANCE_OCCLUSION: f32 = 0.0025;
    // const MAX_DISTANCE_OCCLUSION: f32 = 1.0;
    var res = raymarch_sparse(ray, environment.max_ambient_distance);
    if res.hit {
        // return 0.0;
        res.distance = min(f32(environment.max_ambient_distance), res.distance) * MAX_DISTANCE_OCCLUSION;
    }
    // return 1.0;
    return res.distance / f32(environment.max_ambient_distance);
    // ray.direction = normalize(dir);
    // ray.direction = normalize(noise * 2.0 - 1.0);
}

fn trace_shadow(pos: vec2<i32>, noise: vec3<f32>, ls_pos: vec3<f32>, ls_normal: vec3<f32>) -> f32 {
    let light_dir = normalize(model.inv_normal_transform * environment.sun_direction);
    var dir = rand_hemisphere_direction(noise.xy, environment.shadow_spread);
    dir = align_direction(dir, light_dir);
    
    var ray: SparseRay;
    ray.origin = ls_pos + environment.shadow_bias * ls_normal;
    ray.direction = dir;

    let occluded = raymarch_shadow(ray);
    // let occluded = false;
    return select(1.0, 0.0, occluded);
}

// noise from https://github.com/electronicarts/fastnoise/blob/main/FastNoiseDesign.md
fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    const FRACT_PHI: f32 = 0.61803398875;
    const FRACT_SQRT_2: f32 = 0.41421356237;
    const OFFSET: vec2<f32> = vec2<f32>(FRACT_PHI, FRACT_SQRT_2);

    let frame_offset_seed = (frame.frame_id >> 5u) & 0xffu;
    let frame_offset = vec2<u32>(OFFSET * 128.0 * f32(frame_offset_seed));

    let id = pos + frame_offset;
    let sample_pos = vec3<u32>(
        id.x & 0x7fu,
        id.y & 0x7fu,
        frame.frame_id & 0x1fu,
    );
    let noise = textureLoad(tex_noise, sample_pos, 0).rgb;
    return noise;
}

fn rand_hemisphere_direction(noise: vec2<f32>, spread: f32) -> vec3<f32> {
    let xy = (noise * 2.0 - 1.0) * spread;
    let z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
    return vec3(xy, z);
}

struct SparseRay {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct SparseRaymarchResult {
    hit: bool,
    distance: f32,
}

fn raymarch_sparse(ray: SparseRay, max_distance: u32) -> SparseRaymarchResult {
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    if all(step == vec3(0)) {
        return SparseRaymarchResult(false, f32(max_distance));
    }

    for (var i = 0u; i < (max_distance >> 3u) && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {
        let chunk_pos_index = u32(pos.z) * scene.size.x * scene.size.y + u32(pos.y) * scene.size.x + u32(pos.x);
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            for (var j = 0u; all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)); j++) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
                    let t_total = t_entry * 8.0 + t_brick_entry;
                    return SparseRaymarchResult(true, min(t_total, f32(max_distance)));
                }

                prev_ray_length = brick_ray_length;

                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * step;
            }
        }

        prev_ray_length = ray_length;

        mask = step_mask(ray_length);
        ray_length += vec3<f32>(mask) * ray_delta;
        pos += vec3<i32>(mask) * step;
    }
    return SparseRaymarchResult(false, f32(max_distance));
}

fn raymarch_shadow(ray: SparseRay) -> bool {
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    if all(step == vec3(0)) {
        return false;
    }

    for (var i = 0u; i < 256u && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {

        let chunk_pos_index = u32(pos.z) * scene.size.x * scene.size.y + u32(pos.y) * scene.size.x + u32(pos.x);
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // if i > 6u {
            //     return true;
            // }
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            for (var j = 0u; j < 100u && all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)); j++) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    return true;
                }

                prev_ray_length = brick_ray_length;

                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * step;
            }
        }

        prev_ray_length = ray_length;

        mask = step_mask(ray_length);
        ray_length += vec3<f32>(mask) * ray_delta;
        pos += vec3<i32>(mask) * step;
    }
    return false;
}

fn step_mask(ray_length: vec3<f32>) -> vec3<bool> {
    var res = vec3(false);

    res.x = ray_length.x < ray_length.y && ray_length.x < ray_length.z;
    res.y = !res.x && ray_length.y < ray_length.z;
    res.z = !res.x && !res.y;

    return res;
}

struct Ray {
    ls_origin: vec3<f32>,
    direction: vec3<f32>,
};

fn primary_ray(uv: vec2<f32>) -> Ray {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = environment.camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = environment.camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    var ray: Ray;
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
}

// aligns dir to n's tangent space
fn align_direction(dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var tangent = vec3<f32>(0.0);
    var bitangent = vec3<f32>(0.0);
    if (n.z < 0.0) {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        tangent = vec3(1.0 - n.x * n.x * a, -b, n.x);
        bitangent = vec3(b, n.y * n.y * a - 1.0, -n.y);
    }
    else{
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        tangent = vec3(1.0 - n.x * n.x * a, b, -n.x);
        bitangent = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    return normalize(tangent * dir.x + bitangent * dir.y + n * dir.z);
}

struct Voxel {
    ws_normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    hit_mask: vec3<bool>,
}
fn unpack_voxel(packed: u32) -> Voxel {
    var res: Voxel;
    res.ws_normal = decode_normal_octahedral(packed >> 11u);
    res.metallic = f32((packed >> 10u) & 1u);
    res.roughness = f32((packed >> 6u) & 15u) / 16.0;
    res.hit_mask = decode_hit_mask((packed >> 3u) & 7u);
    return res;
}

fn decode_hit_mask(packed: u32) -> vec3<bool> {
    let mask = vec3<u32>(
        (packed >> 2u) & 1u,
        (packed >> 1u) & 1u,
        packed & 1u,
    );
    return vec3<bool>(mask);
}

/// decodes world space normal from lower 21 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn decode_normal_octahedral(packed: u32) -> vec3<f32> {
	let x = f32((packed >> 11u) & 0x3ffu) / 1023.0;
	let y = f32((packed >> 1u) & 0x3ffu) / 1023.0;
	let sgn = f32(packed & 1u) * 2.0 - 1.0;
	var res = vec3<f32>(0.);
	res.x = x - y;
	res.y = x + y - 1.0;
	res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
	return normalize(res);
}
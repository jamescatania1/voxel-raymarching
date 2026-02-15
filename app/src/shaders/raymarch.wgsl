@group(0) @binding(0) var out_albedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var out_velocity: texture_storage_2d<rgba16float, write>;

struct VoxelSceneMetadata {
    size: vec3<u32>,
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
struct Chunk {
    mask: array<u32, 16>,
}
@group(1) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(1) @binding(1) var<uniform> palette: Palette;
@group(1) @binding(2) var<storage, read> chunk_indices: array<u32>;
@group(1) @binding(3) var<storage, read> chunks: array<Chunk>;
@group(1) @binding(4) var brickmap: texture_storage_3d<r32uint, read>;

struct Environment {
    sun_direction: vec3<f32>,
    shadow_bias: f32,
    camera: Camera,
    prev_camera: Camera,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
}
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
}
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
    fxaa_enabled: u32,
}
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

// @group(3) @binding(3) var<

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const DDA_MAX_STEPS: u32 = 300u;
const SKY_COLOR: vec3<f32> = vec3(0.5, 0.9, 1.5);

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = textureDimensions(out_albedo).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let ray = start_ray(in.id.xy);
    let res = raymarch(ray);

    if !res.hit {
        textureStore(out_albedo, pos, vec4(SKY_COLOR, 0.0));
        textureStore(out_normal, pos, vec4(0.0, 0.0, 0.0, 1.0));
        textureStore(out_depth, pos, vec4(0.0, 0.0, 0.0, 1.0));
        textureStore(out_velocity, pos, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let albedo = palette_color(res.palette_index);
    let normal = normalize(model.normal_transform * res.surface_normal);

        var shadow_ray_dir = normalize(environment.sun_direction);
        let shadow_ray_origin = res.local_pos + environment.shadow_bias * res.hit_normal;
        let shadow_factor = 0.0;
        // let in_shadow = raymarch_shadow(Ray(shadow_ray_origin, shadow_ray_dir, 0.0, true));
        // let in_shadow = false;

        let world_pos = model.transform * vec4<f32>(res.local_pos, 1.0);

        var depth = dot(world_pos.xyz - environment.camera.ws_position, environment.camera.forward);
        depth = (depth - environment.camera.near) / (environment.camera.far - environment.camera.near);


        let clip_pos = environment.camera.view_proj * world_pos;
        let ndc = clip_pos.xy / clip_pos.w;
        let uv = ndc * vec2(0.5, -0.5) + 0.5;

        let prev_clip_pos = environment.prev_camera.view_proj * world_pos;
        let prev_ndc = prev_clip_pos.xy / prev_clip_pos.w;
        let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

        let velocity = uv - prev_uv;

    textureStore(out_albedo, vec2<i32>(in.id.xy), vec4(albedo, shadow_factor));
    textureStore(out_normal, vec2<i32>(in.id.xy), vec4(normal, 1.0));
    textureStore(out_depth, vec2<i32>(in.id.xy), vec4(depth, 0.0, 0.0, 1.0));
    textureStore(out_velocity, vec2<i32>(in.id.xy), vec4(velocity, 0.0, 1.0));
}

struct Ray {
    ls_origin: vec3<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_start: f32,
    in_bounds: bool,
}

struct RaymarchResult {
    hit: bool,
    palette_index: u32,
    surface_normal: vec3<f32>,
    hit_normal: vec3<f32>,
    local_pos: vec3<f32>,
}

fn raymarch_shadow(ray: Ray) -> bool {
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    for (var i = 0u; i < DDA_MAX_STEPS && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {

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

            while all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)) {
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

fn raymarch(ray: Ray) -> RaymarchResult {
    if !ray.in_bounds {
        return RaymarchResult();
    }

    let size_chunks = vec3<i32>(scene.size);
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let ray_step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3(0.0);

    if any(pos >= size_chunks) || any(pos < vec3(0)) {
        return RaymarchResult();
    }

    for (var i = 0u; i < DDA_MAX_STEPS; i++) {
        let chunk_pos_index = pos.z * size_chunks.x * size_chunks.y + pos.y * size_chunks.x + pos.x;
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            var mask = step_mask(prev_ray_length);

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            while all(brick_pos >= vec3(0)) && all(brick_pos < vec3(8)) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    let brick_index = i32(chunk_index - 1u);
                    let base_index = vec3<i32>(
                        (brick_index % size_chunks.x) << 3u,
                        ((brick_index / size_chunks.x) % size_chunks.y) << 3u,
                        (brick_index / (size_chunks.x * size_chunks.y)) << 3u
                    );

                    let packed = textureLoad(brickmap, vec3<i32>(base_index) + brick_pos).r;

                    let palette_index = packed & 0x3ffu;
                    let normal_packed = packed >> 11u;

                    // this is the smooth per-voxel normal
                    let surface_normal = decode_normal_octahedral(normal_packed);

                    // this is the flat normal from the sign of the ray entry
                    let hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));

                    // t_total is the total t-value traveled from the camera to the hit voxel
                    // ray.t_start refers to how far we had to project forward to get into the volume
                    let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
                    let t_total = ray.t_start + t_entry * 8.0 + t_brick_entry;
                    let local_pos = ray.ls_origin + dir * t_total;

                    return RaymarchResult(true, palette_index, surface_normal, hit_normal, local_pos, );
                }

                prev_ray_length = brick_ray_length;

                // for some reason the branchless approach is faster here,
                // just some weird register optimization with naga,
                // likely doesn't happen on the outer loop since the scene size is non-constant,
                // worth further investigation as it's non-negligable at least on my machine
                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * ray_step;
            }
        }

        prev_ray_length = ray_length;

        // simple DDA traversal http://cse.yorku.ca/~amana/research/grid.pdf
        // trying clean "branchless" versions ate up ALU cycles on my nvidia card
        // simple is fast
        if ray_length.x < ray_length.y {
            if ray_length.x < ray_length.z {
                pos.x += ray_step.x;
                if pos.x < 0 || pos.x >= size_chunks.x {
                    break;
                }
                ray_length.x += ray_delta.x;
            } else {
                pos.z += ray_step.z;
                if pos.z < 0 || pos.z >= size_chunks.z {
                    break;
                }
                ray_length.z += ray_delta.z;
            }
        } else {
            if ray_length.y < ray_length.z {
                pos.y += ray_step.y;
                if pos.y < 0 || pos.y >= size_chunks.y {
                    break;
                }
                ray_length.y += ray_delta.y;
            } else {
                pos.z += ray_step.z;
                if pos.z < 0 || pos.z >= size_chunks.z {
                    break;
                }
                ray_length.z += ray_delta.z;
            }
        }
    }

    return RaymarchResult();
}

fn step_mask(ray_length: vec3<f32>) -> vec3<bool> {
    var res = vec3(false);

    res.x = ray_length.x < ray_length.y && ray_length.x < ray_length.z;
    res.y = !res.x && ray_length.y < ray_length.z;
    res.z = !res.x && !res.y;

    return res;
}

fn start_ray(pos: vec2<u32>) -> Ray {
    let camera = environment.camera;
    let dimensions = textureDimensions(out_albedo).xy;

    var pixel_pos = vec2<f32>(pos);
    if frame.taa_enabled == 0u {
        pixel_pos += 0.5;
    } else {
        pixel_pos += environment.jitter;
    }
    let uv = pixel_pos / vec2<f32>(dimensions);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = (model.inv_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let bd_max = (model.inv_transform * vec4(vec3<f32>(scene.size * 8u), 1.0)).xyz;

    // let inv_dir = 1.0 / safe_vec3(ls_direction);
    let inv_dir = safe_inverse(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    let t_start = max(0.0, t_near + 1e-7);

    var ray: Ray;
    ray.ls_origin = ls_origin;
    ray.origin = ls_origin + t_start * ls_direction;
    ray.direction = ls_direction;
    ray.t_start = t_start;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    return ray;
}

fn safe_inverse(v: vec3<f32>) -> vec3<f32> {
    return vec3(
        select(1.0 / v.x, 1e10, v.x == 0.0),
        select(1.0 / v.y, 1e10, v.y == 0.0),
        select(1.0 / v.z, 1e10, v.z == 0.0),
    );
}

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
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

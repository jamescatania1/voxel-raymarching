struct SceneData {
    size: vec3<u32>,
}
struct Chunk {
    brick_index: u32,
    mask: array<u32, 16>,
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
@group(0) @binding(0) var out_albedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> scene: SceneData;
@group(0) @binding(4) var<storage, read> chunk_indices: array<u32>;
@group(0) @binding(5) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(6) var voxels: texture_storage_3d<r32uint, read>;
@group(0) @binding(7) var<uniform> palette: Palette;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
}
struct Environment {
    sun_direction: vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var<uniform> environment: Environment;

struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
}
@group(2) @binding(0) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const DDA_MAX_STEPS: u32 = 1000u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {

    let ray = start_ray(in.id.xy);

    let res = raymarch(ray);

    var albedo = vec3(0.0);
    var normal = vec3(0.0);
    var depth = 0.0;
    var shadow_factor = 0.0;

    if res.hit {
        albedo = palette_color(res.palette_index);
        normal = normalize(model.normal_transform * res.normal);

        depth = res.t_total * dot(ray.direction, camera.forward);
        depth = (depth - camera.near) / (camera.far - camera.near);

        if res.in_shadow {
            shadow_factor = 1.0;
        }
    }

    textureStore(out_albedo, vec2<i32>(in.id.xy), vec4(albedo, shadow_factor));
    textureStore(out_normal, vec2<i32>(in.id.xy), vec4(normal, 1.0));
    textureStore(out_depth, vec2<i32>(in.id.xy), vec4(depth, 0.0, 0.0, 1.0));
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_start: f32,
    in_bounds: bool,
}

struct RaymarchResult {
    hit: bool,
    palette_index: u32,
    normal: vec3<f32>,
    t_total: f32,
    in_shadow: bool,
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
                    // let voxel = (bricks[chunk.brick_index - 1u].data[voxel_index >> 2u] >> ((u32(voxel_index) & 3u) << 3u)) & 0xFFu;
                    // let packed = bricks[chunk.brick_index - 1u].data[voxel_index];

                    var size_bricks = textureDimensions(voxels);
                    size_bricks.x = (size_bricks.x + 7u) >> 3u;
                    size_bricks.y = (size_bricks.y + 7u) >> 3u;
                    size_bricks.z = (size_bricks.z + 7u) >> 3u;
                    let brick_index = chunk.brick_index - 1u;
                    let base_index = vec3<u32>(
                        (brick_index % size_bricks.x) << 3u,
                        ((brick_index / size_bricks.x) % size_bricks.y) << 3u,
                        (brick_index / (size_bricks.x * size_bricks.y)) << 3u
                    );

                    let packed = textureLoad(voxels, vec3<i32>(base_index) + brick_pos).r;

                    let palette_index = packed & 0x3ffu;
                    let normal_packed = packed >> 11u;

                    let normal = decode_normal_octahedral(normal_packed);

                    let hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));

                    // t_total is the total t-value traveled from the camera to the hit voxel
                    // ray.t_start refers to how far we had to project forward to get into the volume
                    let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
                    let t_total = ray.t_start + t_entry * 8.0 + t_brick_entry;
                    let hit_pos = ray.origin + dir * t_total;

                    let shadow_ray_dir = (model.transform * vec4(normalize(environment.sun_direction), 0.0)).xyz;
                    let shadow_ray_origin = hit_pos;

                    let in_shadow = raymarch_shadow(Ray(shadow_ray_origin, shadow_ray_dir, 0.0, true));
                    // let in_shadow = false;

                    return RaymarchResult(true, palette_index, normal, t_total, in_shadow);
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
    let dimensions = textureDimensions(out_albedo).xy;
    let uv = vec2<f32>(pos) / vec2<f32>(dimensions);

    let forward = normalize(camera.forward);
    let right = normalize(cross(vec3<f32>(0.0, 0.0, -1.0), forward));
    let up = normalize(cross(forward, right));

    let vp_size = 2.0 * vec2<f32>(
        camera.near * tan(0.5 * camera.fov),
        camera.near * tan(0.5 * camera.fov) * f32(dimensions.y) / f32(dimensions.x),
    );
    let vp_origin = (camera.ws_position + forward * camera.near) - (0.5 * vp_size.x * right) - (0.5 * vp_size.y * up);
    let ws_origin = vp_origin + (uv.x * vp_size.x * right) + (uv.y * vp_size.y * up);
    let ws_direction = normalize(ws_origin - camera.ws_position);

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

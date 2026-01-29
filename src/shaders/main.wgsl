struct SceneData {
    _size: vec3<u32>,
    size: vec3<u32>,
    palette: array<vec4<u32>, 64>
}
struct Chunk {
    brick_index: u32,
    mask: array<u32, 16>,
}
struct Brick {
    data: array<u32, 128>,
}
@group(0) @binding(0) var out_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> scene: SceneData;
@group(0) @binding(2) var<storage, read> chunk_indices: array<u32>;
@group(0) @binding(3) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(4) var<storage, read> bricks: array<Brick>;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: Camera;

struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
}
@group(2) @binding(0) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const EPSILON: f32 = 0.0001;
const DDA_MAX_STEPS: u32 = 1000u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(out_texture).xy;

    let uv = vec2<f32>(in.id.xy) / vec2<f32>(dimensions);

    let ray = start_ray(uv);

    let res = raymarch(ray);

    var color = vec3(0.0);

    if res.material_id > 0u {
        // let ws_normal = normalize(model.normal_transform *  res.normal);
        let ws_normal = normalize(model.normal_transform * res.normal);

        let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

        let diff = max(dot(ws_normal, ws_light_dir), 0.0);

        let albedo = palette_color(res.material_id).xyz;

        color = albedo * (diff + 0.2);
        // color *= 0.000001;
        // color += res.normal;
    }

    textureStore(out_texture, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

fn palette_color(index: u32) -> vec4<f32> {
    let rgba = scene.palette[index >> 2u][index & 3u];
    return vec4<f32>(
        f32((rgba >> 24u) & 0xFFu) / 255.0,
        f32((rgba >> 16u) & 0xFFu) / 255.0,
        f32((rgba >> 8u) & 0xFFu) / 255.0,
        f32(rgba & 0xFFu) / 255.0
    );
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    in_bounds: bool,
}

struct RaymarchResult {
    material_id: u32,
    normal: vec3<f32>,
}

fn raymarch(ray: Ray) -> RaymarchResult {
    if !ray.in_bounds {
        return RaymarchResult();
    }

    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(
        select(-1, 1, dir.x > 0.0),
        select(-1, 1, dir.y > 0.0),
        select(-1, 1, dir.z > 0.0),
    );
    let ray_delta = abs(vec3(1.0) / (dir + EPSILON));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    for (var i = 0u; i < DDA_MAX_STEPS && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {

        let chunk_pos_index = u32(pos.x) * scene.size.y * scene.size.z + u32(pos.y) * scene.size.z + u32(pos.z);
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // now do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            let entrance_pos = origin + dir * (min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z) - EPSILON);
            let brick_origin = clamp((entrance_pos - vec3<f32>(pos)) * 8.0, vec3(EPSILON), vec3(8.0 - EPSILON));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (vec3<f32>(brick_pos) - brick_origin) + (sign(dir) * 0.5) + 0.5);


            while all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)) {
                let voxel_index = (brick_pos.x << 6u) | (brick_pos.y << 3u) | brick_pos.z;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    let voxel = (bricks[chunk.brick_index - 1u].data[voxel_index >> 2u] >> ((u32(voxel_index) & 3u) << 3u)) & 0xFFu;

                    let normal = vec3<f32>(mask) * -vec3<f32>(step);

                    return RaymarchResult(voxel, normal);
                }

                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * step;
            }
        }

        prev_ray_length = ray_length;

        mask = step_mask(ray_length);
        ray_length += vec3<f32>(mask) * ray_delta;
        pos += vec3<i32>(mask) * step;
        // normal = vec3<f32>(mask) * -vec3<f32>(step);
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

fn start_ray(uv: vec2<f32>) -> Ray {
    let hs_far = vec2<f32>(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let cs_far = vec4<f32>(hs_far, 1.0, 1.0);

    let ws_far_sc = camera.inv_view_proj * cs_far;
    let ws_far = ws_far_sc.xyz / ws_far_sc.w;

    let ws_origin = camera.ws_position;
    let ws_direction = ws_far - camera.ws_position;

    // let ls_origin = ws_origin;
    // let ls_direction = normalize(ws_direction);

    let ls_origin = (model.inv_transform * vec4<f32>(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4<f32>(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = vec3<f32>(0.0);
    let bd_max = vec3<f32>(scene.size * 8u);

    let inv_dir = 1.0 / safe_vec3(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    var ray: Ray;
    ray.origin = ls_origin + (max(0.0, t_near) + 1e-3) * ls_direction;
    ray.direction = ls_direction;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    return ray;
}

fn safe_vec3(v: vec3<f32>) -> vec3<f32> {
    return sign(v) * max(vec3(EPSILON), abs(v));
}
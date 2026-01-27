struct SceneData {
    size: vec3<u32>,
    depth: u32,
    palette: array<vec4<u32>, 64>
}
@group(0) @binding(0) var out_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> scene: SceneData;
@group(0) @binding(2) var<storage, read> voxels: array<u32>;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: Camera;

struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
}
@group(2) @binding(0) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const EPSILON: f32 = 1.0 / 65536.0;
const DDA_MAX_STEPS: u32 = 1000u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(out_texture).xy;
    let uv = vec2<f32>(in.id.xy) / vec2<f32>(dimensions);

    let size = 4u << (scene.depth << 1u);

    let ray = start_ray(uv, size);
    let color = raymarch(ray);

    // color = mix(color * 0.1, vec4(1.0), min(1.0, f32(steps_taken) / 600.0));
    textureStore(out_texture, vec2<i32>(in.id.xy), color);
}

fn raymarch(ray: Ray) -> vec4<f32> {
    if !ray.in_bounds {
        return vec4(0.0);
    }

    let size = 4u << (scene.depth << 1u);

    let step = sign(ray.direction);
    let delta = abs(vec3(1.0) / (ray.direction + EPSILON));

    var pos = floor(ray.origin);
    var side_distance = (sign(ray.direction) * (pos - ray.origin) * (sign(ray.direction) * 0.5) + 0.5) * delta;
    var normal = vec3(0.0);
    
    // var steps_taken = 0;

    for (var i = 0u; i < DDA_MAX_STEPS; i++) {
        let res = voxel(vec3<i32>(floor(pos / f32(size))));
        if (res >> 1u) != 0u {
            return palette_color(1u);
        }

        if side_distance.x < side_distance.y {
            if side_distance.x < side_distance.z {
                pos.x += step.x;
                if pos.x < 0 || pos.x > size {
                    break;
                }
                side_distance.x += delta.x;
            } else {
                pos.z += step.z;
                if pos.z < 0 || pos.z > size {
                    break;
                }
            }
        }
        // normal = step(side_distance, min(side_distance.yxy, side_distance.zzx));
        // side_distance = fma(normal, delta, side_distance);
        // pos = fma(normal, step, pos);
    }

    return vec4(0.0);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    in_bounds: bool,
}

fn start_ray(uv: vec2<f32>, size: u32) -> Ray {
    let hs_far = vec2<f32>(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let cs_far = vec4<f32>(hs_far, 1.0, 1.0);

    let ws_far_sc = camera.inv_view_proj * cs_far;
    let ws_far = ws_far_sc.xyz / ws_far_sc.w;

    let ws_origin = camera.ws_position;
    let ws_direction = ws_far - camera.ws_position;

    let ls_origin = (model.inv_transform * vec4<f32>(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4<f32>(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = vec3<f32>(0.0);
    let bd_max = vec3<f32>(f32(size));

    let inv_dir = 1.0 / safe_vec3(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    var ray: Ray;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    ray.origin = ls_origin + (max(0.0, t_near) + 1e-4) * ls_direction;
    ray.direction = ls_direction;
    return ray;
}

fn voxel(p: vec3<i32>) -> u32 {
    return voxels[(u32(p.x) << 4u) | (u32(p.y) << 2u) | u32(p.z)];
}

fn is_leaf(v: u32) -> bool {
    return (v & 1u) == 1u;
}

fn is_internal(v: u32) -> bool {
    return (v & 1u) == 0u;
}

fn sign_11(p: vec3<f32>) -> vec3<i32> {
    return vec3(
        select(1, -1, p.x < 0.0),
        select(1, -1, p.y < 0.0),
        select(1, -1, p.z < 0.0),
    );
}

fn safe_vec3(v: vec3<f32>) -> vec3<f32> {
    return sign(v) * max(vec3(EPSILON), abs(v));
}

/// Palette color lookup
fn palette_color(index: u32) -> vec4<f32> {
    let rgba = scene.palette[index >> 2u][index & 3u];
    return vec4<f32>(
        f32((rgba >> 24u) & 0xFFu) / 255.0,
        f32((rgba >> 16u) & 0xFFu) / 255.0,
        f32((rgba >> 8u) & 0xFFu) / 255.0,
        f32(rgba & 0xFFu) / 255.0
    );
}

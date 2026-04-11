@group(0) @binding(0) var tex_probe_rays: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var tex_probe_irradiance: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var tex_depth: texture_storage_2d<rg16float, read_write>;

struct Environment {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    shadow_bias: f32,
    skybox_rotation: vec2<f32>,
    camera: Camera,
    prev_camera: Camera,
    shadow_spread: f32,
    filter_shadows: u32,
    shadow_filter_radius: f32,
    max_ambient_distance: u32,
    smooth_normal_factor: f32,
    roughness_multiplier: f32,
    indirect_sky_intensity: f32,
    debug_view: u32,
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
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
    inv_normal_transform: mat3x3<f32>,
}
@group(1) @binding(0) var<uniform> environment: Environment;
@group(1) @binding(1) var<uniform> frame: FrameMetadata;
@group(1) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) thread_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

const ACC_ALPHA: f32 = 0.003;

struct ProbeRay {
    radiance: vec3<f32>,
    depth: f32,
    direction: vec3<f32>,
}
var<workgroup> rays: array<ProbeRay, 128>;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let probe_id = in.workgroup_id.x + in.workgroup_id.y * in.num_workgroups.x + in.workgroup_id.z * in.num_workgroups.x * in.num_workgroups.y;

    if in.local_index < 128 {
        let in_tile_base = vec2(
            (probe_id & 127u) << 4,
            (probe_id >> 7u) << 3,
        );
        let in_tile_offset = vec2(
            in.local_index & 15,
            in.local_index >> 4,
        );
        let in_ray_pos = in_tile_base + in_tile_offset;
        let in_raw = textureLoad(tex_probe_rays, in_ray_pos);

        var in_ray: ProbeRay;
        in_ray.radiance = in_raw.rgb;
        in_ray.depth = in_raw.w;
        in_ray.direction = generate_sample_direction(f32(in.local_index), 128.0);

        rays[in.local_index] = in_ray;
    }

    workgroupBarrier();

    // update irradiance
    if in.local_index < 36 {
        let tile_pos = vec2(
            in.local_index % 6,
            in.local_index / 6,
        );
        let uv = (vec2<f32>(tile_pos) + 0.5) / 6.0;
        let dir = decode_octahedral(uv * 2.0 - 1.0);

        var irradiance = vec3(0.0);
        var weight = 0.0;

        for (var i = 0u; i < 128; i++) {
            let ray = rays[i];

            let w = dot(dir, ray.direction);
            if w > 0.0 {
                irradiance += ray.radiance * w;
                weight += w;
            }
        }

        if weight > 1e-6 {
            irradiance /= weight;
        }

        let atlas_pos = probe_atlas_offset_irradiance(probe_id) + tile_pos + 1;

        let prev_irradiance = textureLoad(tex_probe_irradiance, atlas_pos).rgb;
        let res = mix(prev_irradiance, irradiance, ACC_ALPHA);
        textureStore(tex_probe_irradiance, atlas_pos, vec4(res, 1.0));
    }

    // update depth
    if in.local_index >= 36 && in.local_index < 232 {
        let local_index = in.local_index - 36;
        let tile_pos = vec2(
            local_index % 14,
            local_index / 14,
        );
        let uv = (vec2<f32>(tile_pos) + 0.5) / 14.0;
        let dir = decode_octahedral(uv * 2.0 - 1.0);

        var moments = vec2(0.0);
        var weight = 0.0;

        for (var i = 0u; i < 128; i++) {
            let ray = rays[i];
            let w = dot(dir, ray.direction);
            if w > 0.0 && ray.depth >= 0.0 {
                moments += vec2(
                    ray.depth * w,
                    ray.depth * ray.depth * w,
                );
                weight += w;
            }
        }

        if weight > 1e-6 {
            moments /= weight;
        }

        let atlas_pos = probe_atlas_offset_depth(probe_id) + tile_pos + 1;

        var prev_moments: vec2<f32>;
        if frame.frame_id == 0 {
            prev_moments = vec2(20.0, 400.0);
        } else {
            prev_moments = textureLoad(tex_depth, atlas_pos).rg;
        }
        let res = mix(prev_moments, moments, ACC_ALPHA);

        textureStore(tex_depth, atlas_pos, vec4(res, 0.0, 1.0));
    }
}

fn generate_sample_direction(i: f32, n: f32) -> vec3<f32> {
    // TODO rotate per frame
    const GOLDEN: f32 = 1.61803398875;
    const PI: f32 = 3.14159265359;

    let phi = 2.0 * PI * fract(i * (GOLDEN - 1.0));
    let cos_t = 1.0 - (2.0 * i + 1.0) * (1.0 / n);
    let sin_t = sqrt(saturate(1.0 - cos_t * cos_t));

    // return vec3(
    //     cos(phi) * sin_t,
    //     sin(phi) * sin_t,
    //     cos_t,
    // );
    var dir = vec3(
        cos(phi) * sin_t,
        sin(phi) * sin_t,
        cos_t,
    );

    let angle = f32(reverseBits(frame.frame_id)) * 2.3283064365386963e-10 * 6.28318530718;
    let ca = cos(angle);
    let sa = sin(angle);
    return vec3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
}

fn encode_octahedral(direction: vec3<f32>) -> vec2<f32> {
    let l1 = dot(abs(direction), vec3(1.0));
    var uv = direction.xy * (1.0 / l1);
    if direction.z < 0.0 {
        uv = (1.0 - abs(uv.yx)) * select(vec2(-1.0), vec2(1.0), uv.xy >= vec2(0.0));
    }
    return uv;
}

fn decode_octahedral(uv: vec2<f32>) -> vec3<f32> {
    var v = vec3(uv, 1.0 - abs(uv.x) - abs(uv.y));
    if v.z < 0.0 {
        v = vec3((1.0 - abs(v.yx)) * select(vec2(-1.0), vec2(1.0), uv.xy >= vec2(0.0)), v.z);
    }
    return normalize(v);
}

// atlas always has width 2048
// each row contains 256 tiles since each is 8x8 incl. padding
fn probe_atlas_offset_irradiance(probe_id: u32) -> vec2<u32> {
    return vec2(
        (probe_id & 255) << 3,
        (probe_id >> 8u) << 3,
    );
}

// atlas always has width 2048
// each row contains 128 tiles since each is 16x16 incl. padding
fn probe_atlas_offset_depth(probe_id: u32) -> vec2<u32> {
    return vec2(
        (probe_id & 127) << 4,
        (probe_id >> 7u) << 4,
    );
}
override min_roughness: f32 = 0.001;
override dielectric_specular: f32 = 0.04;

struct VoxelLighting {
    irradiance: vec3<f32>,
    shadow: f32,
    ao: f32,
    history_length: u32,
}

@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_voxel_id: texture_storage_2d<r32uint, read>;
@group(0) @binding(4) var tex_specular: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;

@group(2) @binding(0) var sampler_linear: sampler;
@group(2) @binding(1) var sampler_noise: sampler;
@group(2) @binding(2) var tex_skybox: texture_cube<f32>;
@group(2) @binding(3) var tex_irradiance: texture_cube<f32>;
@group(2) @binding(4) var tex_prefilter: texture_cube<f32>;
@group(2) @binding(5) var tex_brdf_lut: texture_2d<f32>;
@group(2) @binding(6) var<storage, read> voxel_lighting: array<array<u32,3>>;

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

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_albedo).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    let uv_jittered = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    let depth = textureLoad(tex_depth, pos).r;
    if depth < 0.0 {
        var sky = textureSampleLevel(tex_skybox, sampler_linear, ray.ws_direction.xzy, 0.0).rgb;
        textureStore(out_color, vec2<i32>(in.id.xy), vec4(sky, 1.0));
        return;
    }

    let voxel_id = textureLoad(tex_voxel_id, pos).r;

    let lighting = unpack_voxel_lighting(voxel_lighting[voxel_id]);

    let velocity = textureLoad(tex_velocity, pos).rg;
    let packed = textureLoad(tex_normal, pos).r;
    let voxel = unpack_voxel(packed);

    let albedo_sample = textureLoad(tex_albedo, pos);
    let albedo = albedo_sample.rgb;
    let specular = textureLoad(tex_specular, pos).rgb;

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    var surface: PbrInput;
    surface.uv = uv;
    surface.ws_pos = ws_pos;
    surface.ws_normal = voxel.ws_normal;
    surface.albedo = albedo;
    surface.metallic = voxel.metallic;
    surface.roughness = max(voxel.roughness, min_roughness);
    surface.shadow = lighting.shadow;
    surface.irradiance = lighting.irradiance;
    surface.ao = lighting.ao;
    surface.specular = specular;
    var color = pbr(surface);

    if environment.debug_view != 0u {
        color *= 0.00001;
        switch environment.debug_view {
            case 1u: {
                color += surface.albedo;
            }
            case 2u {
                color += depth * 0.005;
            }
            case 3u {
                color += voxel.ls_hit_normal;
            }
            case 4u {
                color += voxel.ws_normal;
            }
            case 5u {
                color += vec3(surface.roughness);
            }
            case 6u {
                color += vec3(surface.metallic);
            }
            case 7u {
                color += vec3(lighting.shadow);
            }
            case 8u {
                // color += surface.irradiance;
                color += vec3(lighting.ao);
            }
            case 9u {
                // color += vec3(surface.specular);
                color += specc;
            }
            case 10u {
                color += vec3(abs(velocity), 0.0);
            }
            case 11u {
                color += textureSampleLevel(tex_skybox, sampler_linear, ray.ws_direction.xzy, 0.0).rgb;
            }
            case 12u {
                color += textureSampleLevel(tex_irradiance, sampler_linear, ray.ws_direction.xzy, 0.0).rgb;
            }
            case 13u {
                let t = cos(f32(frame.frame_id) / 300.0) * 0.5 + 0.5;
                color += textureSampleLevel(tex_prefilter, sampler_linear, ray.ws_direction.xzy, t * 5.0).rgb;
            }
            default {}
        }
    }

    // color *= 0.000001;
    // color += vec3(shadow);

    // if illumination.b > 0.5 {
    //     color = color + vec3(0.2, 0.0, 0.0);
    // }

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

var<private> specc: vec3<f32>;

struct PbrInput {
    uv: vec2<f32>,
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    specular: vec3<f32>,
    shadow: f32,
    irradiance: vec3<f32>,
    ao: f32,
}

fn pbr(in: PbrInput) -> vec3<f32> {
    let N = in.ws_normal;
    let V = normalize(environment.camera.ws_position - in.ws_pos);
    let L = normalize(environment.sun_direction);
    let H = normalize(V + L);
    let R = reflect(-V, N);

    const SUN_COLOR: vec3<f32> = vec3<f32>(0.97, 0.855, 0.775) * 3.0;

    var direct: vec3<f32>;
    {
        let f_0 = mix(vec3(0.04), in.albedo, in.metallic);
        let k_s = fresnel(f_0, H, V);
        let k_d = (1.0 - k_s) * (1.0 - in.metallic);

        let ndf = normal_distribution(in.roughness, N, H);
        let geom = geom_smith(in.roughness, N, V, L);

        let ndv = max(dot(N, V), 0.0);
        let ndl = max(dot(N, L), 0.0);

        let diffuse = k_d * in.albedo / PI;

        let specular = ndf * geom * k_s / max(4.0 * ndv * ndl, 0.000001);

        let brdf = diffuse + specular;
        direct = brdf * SUN_COLOR * in.shadow * ndl;
    }
    var indirect: vec3<f32>;
    {
        let f_0 = mix(vec3(0.04), in.albedo, in.metallic);
        let k_s = fresnel_roughness(f_0, N, V, in.roughness);
        let k_d = (1.0 - k_s) * (1.0 - in.metallic);

        let ndv = clamp(dot(N, V), 0.0001, 1.0);
        let vdh = saturate(dot(V, H));

        let sky_prefilter = textureSampleLevel(tex_prefilter, sampler_linear, R.xzy, in.roughness * 5.0).rgb * environment.indirect_sky_intensity;
        let sky_brdf = textureSampleLevel(tex_brdf_lut, sampler_linear, vec2(ndv, in.roughness), 0.0).rg;

        let diffuse = k_d * in.irradiance * in.albedo / PI;

        let smoothing = exp2(-16.0 * in.roughness - 1.0);
        let specular_occlusion = saturate(pow(ndv + in.ao, smoothing) - 1.0 + in.ao);

        let ibl_occluded = specular_occlusion * sky_prefilter;
        specc = ibl_occluded;

        // let specular = ibl_occluded * (f_0 * sky_brdf.x + sky_brdf.y);
        let specular = in.specular * (f_0 * sky_brdf.x + sky_brdf.y);

        indirect = diffuse + specular;
    }
    let res = direct + indirect;
    return res;
}

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz
fn normal_distribution(r: f32, N: vec3<f32>, H: vec3<f32>) -> f32 {
    let ndh = max(dot(N, H), 0.0);
    let d = ndh * ndh * (r * r - 1.0) + 1.0;

    return pow(r, 2.0) / max(PI * d * d, 0.000001);
}

// Smith model
fn geom_smith(r: f32, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> f32 {
    return geom_ggx(r, N, V) * geom_ggx(r, N, L);
}

// Schlick/Beckman geometry shadowing
fn geom_ggx(r: f32, N: vec3<f32>, x: vec3<f32>) -> f32 {
    let ndx = max(dot(N, x), 0.0);

    let k = r / 2.0;
    let den = ndx * (1.0 - k) + k;

    return ndx / max(den, 0.000001);
}

// Fresnel/Schick approximation
fn fresnel(f0: vec3<f32>, H: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    let ndv = max(dot(H, V), 0.0);
    return f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
}

fn fresnel_roughness(f0: vec3<f32>, N: vec3<f32>, V: vec3<f32>, roughness: f32) -> vec3<f32> {
    let ndv = max(dot(N, V), 0.01);
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - ndv, 5.0);
}

struct Ray {
    ls_origin: vec3<f32>,
    ws_direction: vec3<f32>,
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
    ray.ws_direction = ws_direction;
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
}

struct Voxel {
    ws_normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ls_hit_normal: vec3<f32>,
}
fn unpack_voxel(packed: u32) -> Voxel {
    var res: Voxel;
    res.ws_normal = decode_normal_octahedral(packed >> 11u);
    res.metallic = f32((packed >> 10u) & 1u);
    res.roughness = f32((packed >> 6u) & 15u) / 16.0;

    let hit_mask = decode_hit_mask((packed >> 3u) & 7u);
    let ray_dir_sign = vec3<f32>(decode_hit_mask(packed & 7u)) * 2.0 - 1.0;
    res.ls_hit_normal = normalize(-ray_dir_sign * vec3<f32>(hit_mask));

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

fn unpack_voxel_lighting(packed: array<u32, 3>) -> VoxelLighting {
    let irr_rg = unpack2x16float(packed[0]);
    let irr_b_shadow = unpack2x16float(packed[1]);

    var res: VoxelLighting;
    res.irradiance = vec3(irr_rg, irr_b_shadow.r);
    res.shadow = irr_b_shadow.y;
    res.ao = f32(packed[2] >> 16u) / 65535.0;
    res.history_length = packed[2] & 0xFFFFu;
    return res;
}
@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_velocity: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(1) var tex_depth: texture_storage_2d<r32float, read>;
@group(1) @binding(2) var tex_illumination: texture_2d<f32>;

@group(2) @binding(0) var sampler_linear: sampler;
@group(2) @binding(1) var sampler_noise: sampler;
@group(2) @binding(2) var tex_noise: texture_2d<f32>;

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

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_albedo).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
	let uv_jittered  = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));

    let packed = textureLoad(tex_normal, pos).r;
    let voxel = unpack_voxel(packed);

    let noise = blue_noise(in.id.xy);

    let albedo_sample = textureLoad(tex_albedo, pos);
    let albedo = albedo_sample.rgb;
    // let illumination = gather_illumination(pos, noise);
    let illumination = textureLoad(tex_illumination, pos, 0);
    let shadow = illumination.r;
    let radiance = illumination.g;

    let depth = textureLoad(tex_depth, pos).r;
    let velocity = textureLoad(tex_velocity, pos).rg;

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    var surface: PbrInput;
    surface.ws_pos = ws_pos;
    surface.ws_normal = voxel.ws_normal;
    surface.albedo = albedo;
    surface.metallic = voxel.metallic;
    surface.roughness = pow(voxel.roughness, 4.0);
    surface.shadow = shadow;
    surface.ao = radiance;
    var color = pbr(surface);

    // color *= 0.000001;
    // color += vec3(shadow);

    // if illumination.b > 0.5 {
    //     color = color + vec3(0.2, 0.0, 0.0);
    // }

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

struct PbrInput {
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    shadow: f32,
    ao: f32,
}

fn pbr(in: PbrInput) -> vec3<f32> {
    let N = in.ws_normal;
    let V = normalize(environment.camera.ws_position - in.ws_pos);
    let L = normalize(environment.sun_direction);
    let H = normalize(V + L);

    const AMBIENT: vec3<f32> = vec3<f32>(1.0) * 0.005;
    const SUN_COLOR: vec3<f32> = vec3<f32>(0.97, 0.855, 0.775) * 0.8;

    var direct: vec3<f32>;
    {
        let diffuse = in.albedo / PI;

        let f_0 = mix(vec3(0.04), in.albedo, in.metallic);
        let k_s = fresnel(f_0, H, V);
        let k_d = (1.0 - k_s) * (1.0 - in.metallic);

        let ndf = normal_distribution(in.roughness, N, H);
        let geom = geom_smith(in.roughness, N, V, L);

        let ndv = max(dot(N, V), 0.0);
        let ndl = max(dot(N, L), 0.0);
        let specular = ndf * geom * k_s / max(4.0 * ndv * ndl, 0.000001);

        let brdf = k_d * diffuse + specular;
        direct = brdf * SUN_COLOR * in.shadow * ndl;
    }
    var indirect: vec3<f32>;
    {
        indirect = in.albedo * AMBIENT * in.ao;
    }
    let res = direct + indirect;
    return res;
}

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz
fn normal_distribution(r: f32, N: vec3<f32>, H: vec3<f32>) -> f32 {
    let num = pow(r, 2.0);

    let ndh = max(dot(N, H), 0.0);
    let d = ndh * ndh * (r * r - 1.0) + 1.0;
    let den = PI * d * d;

    return num / max(den, 0.000001);
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


fn gather_illumination(pos: vec2<i32>, noise: vec3<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(tex_illumination).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;

    if environment.filter_shadows == 0u {
        return textureLoad(tex_illumination, pos, 0).rgb;
    } else {
        // let lighting = textureLoad(tex_illumination, pos, 0).rgb;
        // let shadow = filter_spatial(uv, texel_size, noise).value.r;
        // return vec3(shadow, lighting.gb);
        return filter_spatial(uv, texel_size, noise).value;
    }
}

const FILTER_KERNEL: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    vec2<f32>(-0.326212, -0.405805),
    vec2<f32>(-0.840144, -0.073580),
    vec2<f32>(-0.695914,  0.457137),
    vec2<f32>(-0.203345,  0.620716),
    vec2<f32>( 0.962340, -0.194983),
    vec2<f32>( 0.473434, -0.480026),
    vec2<f32>( 0.519456,  0.767022),
    vec2<f32>( 0.185461, -0.893124),
    vec2<f32>( 0.507431,  0.064425),
    vec2<f32>( 0.896420,  0.412458),
    vec2<f32>(-0.321940, -0.932615),
    vec2<f32>(-0.791559, -0.597705)
);

struct FilterResult {
    value: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>,
}

fn filter_spatial(uv: vec2<f32>, texel_size: vec2<f32>, noise: vec3<f32>) -> FilterResult {
    var weight = 0.0;
    var cur = vec3(0.0);
    var min_val = vec3(1.0);
    var max_val = vec3(0.0);

    let radius = environment.shadow_filter_radius * texel_size;
    let t = noise.r * 6.2831853;
    let s_t = sin(t);
    let c_t = cos(t);
    let rotation = mat2x2<f32>(c_t, s_t, -s_t, c_t);

    for (var i = 0u; i < 12u; i++) {
        let offset = rotation * FILTER_KERNEL[i];
        let sample_uv = uv + offset * radius;

        let val = textureSampleLevel(tex_illumination, sampler_linear, sample_uv, 0).rgb;
        min_val = min(min_val, val);
        max_val = max(max_val, val);
        cur += val;
        weight += 1.0;
    }
    cur /= weight;

    var res: FilterResult;
    res.value = cur;
    res.min = min_val;
    res.max = max_val;
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

fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    let noise_pos = vec2<u32>(pos.x & 0x7fu, pos.y & 0x7fu);
    let noise = textureLoad(tex_noise, vec2<i32>(noise_pos), 0).rgb;
    return noise;
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

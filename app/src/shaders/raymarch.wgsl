@group(0) @binding(0) var tex_out_albedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_out_velocity: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var tex_out_normal: texture_storage_2d<r32uint, write>;
@group(1) @binding(1) var tex_out_depth: texture_storage_2d<r32float, write>;

struct VoxelSceneMetadata {
	bounding_size: u32,
	index_levels: u32,
	index_chunk_count: u32,
}
struct Palette {
	data: array<vec4<f32>, 1024>,
}
struct IndexChunk {
	child_index: u32,
	mask: array<u32, 2>,
}
@group(2) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(2) @binding(1) var<uniform> palette: Palette;
@group(2) @binding(2) var<storage, read> index_chunks: array<IndexChunk>;
@group(2) @binding(3) var<storage, read> leaf_chunks: array<u32>;
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
	@builtin(local_invocation_index) local_index: u32,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

const DDA_MAX_STEPS: u32 = 300u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
	let pos = vec2<i32>(in.id.xy);
	let res = trace_scene(pos, in.local_index);

	textureStore(tex_out_albedo, pos, vec4(res.albedo, 1.0));
	textureStore(tex_out_normal, pos, vec4(res.normal, 0, 0, 0));
	textureStore(tex_out_depth, pos, vec4(res.depth, 0.0, 0.0, 1.0));
	textureStore(tex_out_velocity, pos, vec4(res.velocity, 0.0, 1.0));
}

struct SceneResult {
	albedo: vec3<f32>,
	normal: u32,
	depth: f32,
	velocity: vec2<f32>,
}

fn trace_scene(pos: vec2<i32>, local_index: u32) -> SceneResult {
	let dimensions = vec2<i32>(textureDimensions(tex_out_albedo).xy);
	let texel_size = 1.0 / vec2<f32>(dimensions);

	let uv = (vec2<f32>(pos) + 0.5) * texel_size;
	let uv_jittered  = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

	// let ray = raymarch_basic(start_ray(select(uv_jittered, uv, frame.taa_enabled == 0u)));
	let ray = raymarch(start_ray(select(uv_jittered, uv, frame.taa_enabled == 0u)), local_index);

	if !ray.hit {
		var res: SceneResult;
		res.albedo = vec3(1.0, 0.0, 0.0);
		res.normal = 0u;
		res.depth = -1.0;
		res.velocity = vec2(0.0);
		return res;
	}

	let ws_pos_h = model.transform * vec4<f32>(ray.local_pos, 1.0);
	let ws_pos = ws_pos_h.xyz;
	let depth = ray.depth;

    let cs_pos = environment.camera.view_proj * ws_pos_h;
    let ndc = cs_pos.xyz / cs_pos.w;
    let cur_uv = ndc.xy * vec2(0.5, -0.5) + 0.5;

    let prev_cs_pos = environment.prev_camera.view_proj * ws_pos_h;
    let prev_ndc = prev_cs_pos.xy / prev_cs_pos.w;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

    let velocity = cur_uv - prev_uv;

	let albedo = palette_color(ray.voxel.palette_index);

    let ls_normal = align_per_voxel_normal(ray.hit_normal, ray.voxel.normal);
	// let ls_normal = ray.hit_normal;
    let ws_normal = normalize(model.normal_transform * ls_normal);

	let packed = repack_voxel(ws_normal,ray.voxel.metallic , ray.voxel.roughness , ray.hit_mask);
	// let packed = repack_voxel(ws_normal,1.0, 0.04, ray.hit_mask);

	var res: SceneResult;
	res.albedo = albedo;
	res.normal = packed;
	res.depth = depth;
	res.velocity = velocity;
	return res;
}

fn blue_noise(pos: vec2<i32>) -> vec3<f32> {
	const FRACT_PHI: f32 = 0.61803398875;
	const FRACT_SQRT_2: f32 = 0.41421356237;
	const OFFSET: vec2<f32> = vec2<f32>(FRACT_PHI, FRACT_SQRT_2);

	let frame_offset_seed = (frame.frame_id >> 5u) & 0xffu;
	let frame_offset = vec2<u32>(OFFSET * 128.0 * f32(frame_offset_seed));

	let id = vec2<u32>(pos) + frame_offset;
	let sample_pos = vec3<u32>(
		id.x & 0x7fu,
		id.y & 0x7fu,
		frame.frame_id & 0x1fu,
	);
	let noise = textureLoad(tex_noise, sample_pos, 0).rgb;
	return noise;
}

fn rand_hemisphere_direction(noise: vec2<f32>) -> vec3<f32> {
	let xy = noise * 2.0 - 1.0;
	let z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
	return vec3(xy, z);
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
	voxel: Voxel,
	hit_normal: vec3<f32>,
	local_pos: vec3<f32>,
	depth: f32,
	hit_mask: vec3<bool>
}

fn raymarch(ray: Ray, local_index: u32) -> RaymarchResult {
	if !ray.in_bounds {
		return RaymarchResult();
	}

	let scale = 1.0 / f32(scene.bounding_size);
	let origin = ray.ls_origin * scale + (ray.t_start * ray.direction) * scale + 1.0;
	
	var dir = ray.direction;
	let inv_dir = sign(dir) / max(vec3(1e-7), abs(dir));

	var pos = clamp(origin, vec3(1.0), vec3(1.9999999));

	// let stack = &shared_stack[local_index];
	var scale_exp = 21u;

	var ci = 0u;
	var chunk = index_chunks[ci];

	var side_distance: vec3<f32>;

	for (var i = 0u; i < 256; i++) {
		var child_offset = chunk_offset(pos, scale_exp);

		while (chunk_contains_child(chunk.mask, child_offset) && !chunk_is_leaf(chunk.child_index) && scale_exp >= 2u) {
			stack[local_index][scale_exp >> 1u] = ci;
			ci = (chunk.child_index >> 1u) + mask_packed_offset(chunk.mask, child_offset);
			chunk = index_chunks[ci];

			scale_exp -= 2u;
			child_offset = chunk_offset(pos, scale_exp);
		}
		if chunk_contains_child(chunk.mask, child_offset) && chunk_is_leaf(chunk.child_index) {
			let leaf_index = mask_packed_offset(chunk.mask, child_offset);
			
			let packed = leaf_chunks[(chunk.child_index >> 1u) + leaf_index];
			
			let t_max = min(min(side_distance.x, side_distance.y), side_distance.z);
			let t_total = ray.t_start + f32(scene.bounding_size) * t_max;
			
			let mask = vec3(t_max) >= side_distance;

			var res: RaymarchResult;
			res.hit = true;
			res.voxel = unpack_voxel(packed);
			res.hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));
			res.depth = t_total;
			res.local_pos = ray.ls_origin + dir * t_total;
			res.hit_mask = mask; 
			return res;
		}

		var adv_scale_exp = scale_exp;

		let snapped_idx = child_offset & 0x2Au;
		if ((chunk.mask[snapped_idx >> 5u] >> (snapped_idx & 31u)) & 0x00330033u) == 0u {
			adv_scale_exp++;
		}

		pos = floor_scale(pos, adv_scale_exp);
		let prev_pos = pos;

		let scale = bitcast<f32>((adv_scale_exp + 104) << 23);
		side_distance = (step(vec3<f32>(0.0), dir) * scale + (pos - origin)) * inv_dir;

		let t_max = min(min(side_distance.x, side_distance.y), side_distance.z);

		// emulate copysign(scale, dir)
		let scale_mag = bitcast<u32>(scale) & 0x7FFFFFFFu;
		let dir_signs = bitcast<vec3<u32>>(dir) & vec3<u32>(0x80000000u);
		let copysign_scale = bitcast<vec3<f32>>(dir_signs | vec3<u32>(scale_mag));

		let tmax_mask = vec3<f32>(t_max) == side_distance;
		let siblPos0 = select(pos, pos + copysign_scale, tmax_mask);

		let bounds_offset = vec3<i32>(i32((1u << adv_scale_exp) - 1u));
		let siblPos1 = bitcast<vec3<f32>>(bitcast<vec3<i32>>(siblPos0) + bounds_offset);

		pos = clamp(origin + (dir * t_max), siblPos0, siblPos1);

		let diffPos = bitcast<vec3<u32>>(pos) ^ bitcast<vec3<u32>>(prev_pos);
		let combined_diff = diffPos.x | diffPos.y | diffPos.z;

		var diffExp: u32 = firstLeadingBit(combined_diff);

		if (diffExp & 1u) == 0u {
			diffExp--;
		}

		// ascend
		if diffExp > scale_exp {
			if diffExp > 21 {
				break;
			}
			
			scale_exp = u32(diffExp);
			ci = stack[local_index][scale_exp >> 1u];
			chunk = index_chunks[ci];
		}
	}

	return RaymarchResult();
}

fn mirrored_pos(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
	var mirrored: vec3<f32> = bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) ^ vec3(0x7FFFFFu));
	
	if any(pos < vec3<f32>(1.0)) || any(pos >= vec3<f32>(2.0)) {
		mirrored = 3.0 - pos;
	}
	return select(pos, mirrored, dir > vec3(0.0));
}

fn mirrored_pos_unchecked(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
	let mirrored: vec3<f32> = bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) ^ vec3(0x7FFFFFu));
	return select(pos, mirrored, dir > vec3(0.0));
}

/// computes floor(pos / scale) * scale
fn floor_scale(pos: vec3<f32>, scale_exp: u32) -> vec3<f32> {
	let mask = ~0u << scale_exp;
	return bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) & vec3(mask));
}

fn chunk_offset(pos: vec3<f32>, scale_exp: u32) -> u32 {
    let chunk_pos = (bitcast<vec3<u32>>(pos) >> vec3(scale_exp)) & vec3(3u);
    return (chunk_pos.y << 4u) | (chunk_pos.z << 2u) | chunk_pos.x;
}

fn chunk_contains_child(mask: array<u32, 2>, offset: u32) -> bool {
	let half_mask = select(mask[0], mask[1], offset >= 32u);
	return (half_mask & (1u << (offset & 31u))) != 0u;
}

/// given mask and index i, gets packed offset based on count of 1s in mask for 0 <= j < i 
fn mask_packed_offset(mask: array<u32, 2>, i: u32) -> u32 {
    if i < 32u {
        return countOneBits(mask[0] & ~(0xffffffffu << i));
    } else {
        return countOneBits(mask[0]) + countOneBits(mask[1] & ~(0xffffffffu << (i - 32u)));
    }
}

fn chunk_is_leaf(child_index: u32) -> bool {
	return (child_index & 1u) == 1u;
}

// fn raymarch(ray: Ray) -> RaymarchResult {
// 	if !ray.in_bounds {
// 		return RaymarchResult();
// 	}

// 	let size_chunks = vec3<i32>(scene.size);
// 	let origin = ray.origin / 8.0;
// 	let dir = ray.direction;

// 	let ray_step = vec3<i32>(sign(dir));
// 	let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

// 	var pos = vec3<i32>(floor(origin));
// 	var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
// 	var prev_ray_length = vec3(0.0);

// 	if any(pos >= size_chunks) || any(pos < vec3(0)) {
// 		return RaymarchResult();
// 	}

// 	for (var i = 0u; i < DDA_MAX_STEPS; i++) {
// 		let chunk_pos_index = pos.z * size_chunks.x * size_chunks.y + pos.y * size_chunks.x + pos.x;
// 		let chunk_index = chunk_indices[chunk_pos_index];

// 		if chunk_index != 0u {
// 			// now we do dda within the brick
// 			var chunk = chunks[chunk_index - 1u];

// 			var mask = step_mask(prev_ray_length);

// 			let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
// 			let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

// 			var brick_pos = vec3<i32>(floor(brick_origin));
// 			var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

// 			prev_ray_length = vec3<f32>(0.0);

// 			while all(brick_pos >= vec3(0)) && all(brick_pos < vec3(8)) {
// 				let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
// 				if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
// 					let brick_index = i32(chunk_index - 1u);
// 					let base_index = vec3<i32>(
// 						(brick_index % size_chunks.x) << 3u,
// 						((brick_index / size_chunks.x) % size_chunks.y) << 3u,
// 						(brick_index / (size_chunks.x * size_chunks.y)) << 3u
// 					);
// 					let packed = textureLoad(brickmap, vec3<i32>(base_index) + brick_pos).r;

// 					// t_total is the total t-value traveled from the camera to the hit voxel
// 					// ray.t_start refers to how far we had to project forward to get into the volume
// 					let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
// 					let t_total = ray.t_start + t_entry * 8.0 + t_brick_entry;
// 					let local_pos = ray.ls_origin + dir * t_total;

// 					var res: RaymarchResult;
// 					res.hit = true;
// 					res.voxel = unpack_voxel(packed);
// 					res.hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));
// 					res.local_pos = local_pos;
// 					res.depth = t_total;
// 					res.hit_mask = mask;
// 					return res;
// 				}

// 				prev_ray_length = brick_ray_length;

// 				// for some reason the branchless approach is faster here,
// 				// just some weird register optimization with naga,
// 				// likely doesn't happen on the outer loop since the scene size is non-constant,
// 				// worth further investigation as it's non-negligable at least on my machine
// 				mask = step_mask(brick_ray_length);
// 				brick_ray_length += vec3<f32>(mask) * ray_delta;
// 				brick_pos += vec3<i32>(mask) * ray_step;
// 			}
// 		}

// 		prev_ray_length = ray_length;

// 		// simple DDA traversal http://cse.yorku.ca/~amana/research/grid.pdf
// 		// trying clean "branchless" versions ate up ALU cycles on my nvidia card
// 		// simple is fast
// 		if ray_length.x < ray_length.y {
// 			if ray_length.x < ray_length.z {
// 				pos.x += ray_step.x;
// 				if pos.x < 0 || pos.x >= size_chunks.x {
// 					break;
// 				}
// 				ray_length.x += ray_delta.x;
// 			} else {
// 				pos.z += ray_step.z;
// 				if pos.z < 0 || pos.z >= size_chunks.z {
// 					break;
// 				}
// 				ray_length.z += ray_delta.z;
// 			}
// 		} else {
// 			if ray_length.y < ray_length.z {
// 				pos.y += ray_step.y;
// 				if pos.y < 0 || pos.y >= size_chunks.y {
// 					break;
// 				}
// 				ray_length.y += ray_delta.y;
// 			} else {
// 				pos.z += ray_step.z;
// 				if pos.z < 0 || pos.z >= size_chunks.z {
// 					break;
// 				}
// 				ray_length.z += ray_delta.z;
// 			}
// 		}
// 	}

// 	return RaymarchResult();
// }

fn step_mask(ray_length: vec3<f32>) -> vec3<bool> {
	var res = vec3(false);

	res.x = ray_length.x < ray_length.y && ray_length.x < ray_length.z;
	res.y = !res.x && ray_length.y < ray_length.z;
	res.z = !res.x && !res.y;

	return res;
}

fn start_ray(uv: vec2<f32>) -> Ray {
	let ndc = vec2<f32>(uv.x, 1.0 - uv.y) * 2.0 - 1.0;

	let ts_near = environment.camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
	let ws_near = ts_near.xyz / ts_near.w;

	let ts_far = environment.camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
	let ws_far = ts_far.xyz / ts_far.w;

	let ws_direction = normalize(ws_far - ws_near);
	let ws_origin = ws_near;

	let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
	let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

	// aabb simple test and project on the scene volume
	let bd_min = (model.inv_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
	let bd_max = (model.inv_transform * vec4(vec3<f32>(scene.bounding_size), 1.0)).xyz;

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
	ray.ls_origin = ls_origin;
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


// clamps per-voxel normal to cone aligned with the hit normal
// https://www.desmos.com/3d/cnbvln5rz6
fn align_per_voxel_normal(n_hit: vec3<f32>, n_surface: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - environment.smooth_normal_factor * 2.0, -0.999, 0.999);

    let d = dot(n_hit, n_surface);
    if d > t {
        return n_surface;
    }

    return t * n_hit + sqrt((1 - t * t) / (1 - d * d)) * (n_surface - d * n_hit);
}

/// encode hit mask into lower 3 bits of u32
/// this encodes the hit normal, as we recover the ray direction from depth
fn encode_hit_mask(mask: vec3<bool>) -> u32 {
    return (u32(mask.x) << 2u) | (u32(mask.y) << 1u) | u32(mask.z);
}

fn decode_hit_mask(packed: u32) -> vec3<bool> {
    let mask = vec3<u32>(
        (packed >> 2u) & 1u,
        (packed >> 1u) & 1u,
        packed & 1u,
    );
    return vec3<bool>(mask);
}

fn repack_voxel(ws_normal: vec3<f32>, metallic: f32, roughness: f32, hit_mask: vec3<bool>) -> u32 {
    let n = encode_normal_octahedral(ws_normal);
    let m = select(1u, 0u, metallic > 0.5);
    let r = u32(roughness * 15.0 + 0.5) & 15u;
	let hm = encode_hit_mask(hit_mask) & 7u;
    return (n << 11u) | (m << 10u) | (r << 6u) | (hm << 3u);
}

struct Voxel {
	normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
	palette_index: u32,
}
fn unpack_voxel(packed: u32) -> Voxel {
	var res: Voxel;
    res.normal = decode_normal_octahedral(packed >> 15u);
    res.metallic = f32((packed >> 14u) & 1u);
    res.roughness = f32((packed >> 10u) & 0xfu) / 15.0;
	res.palette_index = packed & 0x3ffu;
    return res;
}

/// decodes world space normal from lower 17 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn decode_normal_octahedral(packed: u32) -> vec3<f32> {
	let x = f32((packed >> 9u) & 0xffu) / 255.0;
	let y = f32((packed >> 1u) & 0xffu) / 255.0;
	let sgn = f32(packed & 1u) * 2.0 - 1.0;
	var res = vec3<f32>(0.);
	res.x = x - y;
	res.y = x + y - 1.0;
	res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
	return normalize(res);
}

/// encodes world space normal in lower 21 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn encode_normal_octahedral(normal: vec3<f32>) -> u32 {
    var n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var nrm = vec2<f32>(0.);
    nrm.y = n.y * 0.5 + 0.5;
    nrm.x = n.x * 0.5 + nrm.y;
    nrm.y = n.x * -0.5 + nrm.y;
    let sgn = select(0u, 1u, n.z >= 0.0);
    let res = (u32(nrm.x * 1023.0 + 0.5) << 11u) | (u32(nrm.y * 1023.0 + 0.5) << 1u) | sgn;
    return res;
}

struct Scene {
    base: vec3<f32>, // before scaling by voxel scale
    size: vec3<f32>, // before scaling by voxel scale
    scale: f32, // number of voxels per unit
}
struct Material {
    base_albedo: vec4<f32>,
    base_metallic: f32,
    base_roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
    metallic_roughness_index: i32,
    double_sided: u32,
}
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var out_voxels: texture_storage_3d<rg32uint, write>;

// all the gltf's scene textures
// has to be a separate bind group per wgpu
// binding_array might fuck with your LSP since it's non-standard w.r.t. wgsl spec
@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;

struct Primitive {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    material_id: u32,
    index_count: u32,
}
@group(2) @binding(0) var<uniform> primitive: Primitive;
@group(2) @binding(1) var<storage, read> indices: array<u32>;
@group(2) @binding(2) var<storage, read> positions: array<f32>;
@group(2) @binding(3) var<storage, read> normals: array<f32>;
@group(2) @binding(4) var<storage, read> tangents: array<f32>;
@group(2) @binding(5) var<storage, read> uvs: array<f32>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(64, 1, 1)
fn compute_main(in: ComputeIn) {
    if in.id.x * 3u >= primitive.index_count {
        return;
    }

    let index_base = in.id.x * 3u;
    let i0 = indices[index_base + 0u];
    let i1 = indices[index_base + 1u];
    let i2 = indices[index_base + 2u];

    let l_p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
    let l_p1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]);
    let l_p2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]);

    // triangle ws vertex positions
    let v0 = ((primitive.matrix * vec4(l_p0, 1.0)).xyz - scene.base) * scene.scale;
    let v1 = ((primitive.matrix * vec4(l_p1, 1.0)).xyz - scene.base) * scene.scale;
    let v2 = ((primitive.matrix * vec4(l_p2, 1.0)).xyz - scene.base) * scene.scale;

    // triangle ws vertex normals
    let normal_0 = normalize(primitive.normal_matrix * vec3(normals[i0 * 3u], normals[i0 * 3u + 1u], normals[i0 * 3u + 2u]));
    let normal_1 = normalize(primitive.normal_matrix * vec3(normals[i1 * 3u], normals[i1 * 3u + 1u], normals[i1 * 3u + 2u]));
    let normal_2 = normalize(primitive.normal_matrix * vec3(normals[i2 * 3u], normals[i2 * 3u + 1u], normals[i2 * 3u + 2u]));

    // triangle vertex tangents
    let model_rot_scale = mat3x3<f32>(
        primitive.matrix[0].xyz,
        primitive.matrix[1].xyz,
        primitive.matrix[2].xyz
    );
    let tangent_0 = vec4(normalize(model_rot_scale * vec3(tangents[i0 * 4u], tangents[i0 * 4u + 1u], tangents[i0 * 4u + 2u])), tangents[i0 * 4u + 3u]);
    let tangent_1 = vec4(normalize(model_rot_scale * vec3(tangents[i1 * 4u], tangents[i1 * 4u + 1u], tangents[i1 * 4u + 2u])), tangents[i1 * 4u + 3u]);
    let tangent_2 = vec4(normalize(model_rot_scale * vec3(tangents[i2 * 4u], tangents[i2 * 4u + 1u], tangents[i2 * 4u + 2u])), tangents[i2 * 4u + 3u]);

    var min_bd = min(min(v0, v1), v2);
    var max_bd = max(max(v0, v1), v2);

    min_bd = max(floor(min_bd), vec3(0.0));
    max_bd = min(ceil(max_bd), scene.size * scene.scale - 1.0);

    let min_bd_i = vec3<u32>(min_bd);
    let max_bd_i = vec3<u32>(max_bd);

    // TODO this is super slow, might try to subdivide since
    // some threads stall the pipeline with the big triangles
    // maybe new wgpu mesh shaders can help
    // just a generation step though

    for (var x = min_bd_i.x; x < max_bd_i.x; x++) {
        for (var y = min_bd_i.y; y < max_bd_i.y; y++) {
            for (var z = min_bd_i.z; z < max_bd_i.z; z++) {
                let center = vec3(f32(x), f32(y), f32(z)) + 0.5;

                if !test_voxel(center, v0, v1, v2) {
                    continue;
                }
                
                let material = materials[primitive.material_id];
                if material.albedo_index < 0 {
                    continue;
                }

                let point = project_onto_plane(center, v0, v1, v2);
                let weights = tri_weights(point, v0, v1, v2);

                let uv_0 = vec2(uvs[i0 * 2u], uvs[i0 * 2u + 1u]);
                let uv_1 = vec2(uvs[i1 * 2u], uvs[i1 * 2u + 1u]);
                let uv_2 = vec2(uvs[i2 * 2u], uvs[i2 * 2u + 1u]);
                let uv = weights.x * uv_0 + weights.y * uv_1 + weights.z * uv_2;

                let albedo = textureSampleLevel(textures[material.albedo_index], tex_sampler, uv, 0.0);
                let rgba = vec4<u32>(albedo * 255.0 + 0.5);
                let albedo_packed = ((rgba.r & 0xffu) << 24u) | ((rgba.g & 0xffu) << 16u) | ((rgba.b & 0xffu) << 8u) | 0xffu;

                var ws_normal = normalize(weights.x * normal_0 + weights.y * normal_1 + weights.z * normal_2);

                let ws_tangent = normalize((weights.x * tangent_0 + weights.y * tangent_1 + weights.z * tangent_2).xyz);
                let ws_bitangent = cross(ws_normal, ws_tangent) * tangent_0.w;
                let tbn = mat3x3<f32>(
                    ws_tangent,
                    ws_bitangent,
                    ws_normal,
                );

                let tangent_normal = textureSampleLevel(textures[material.normal_index], tex_sampler, uv, 0.0).rgb * 2.0 - 1.0;
                ws_normal = normalize(tbn * tangent_normal);
                
                var metallic = material.base_metallic;
                var roughness = material.base_roughness;
                if material.metallic_roughness_index >= 0 {
                    let mr = textureSampleLevel(textures[material.metallic_roughness_index], tex_sampler, uv, 0.0);
                    roughness *= mr.g;
                    metallic *= mr.b;
                }

                let packed = pack_voxel(ws_normal, metallic, roughness);

                textureStore(out_voxels, vec3<i32>(vec3(x, y, z)), vec4<u32>(albedo_packed, packed, primitive.material_id, 0u));
            }
        }
    }
}

const EXTENT: f32 = 0.5;

fn test_voxel(center: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> bool {
    let v0 = p0 - center;
    let v1 = p1 - center;
    let v2 = p2 - center;

    if any(min(min(v0, v1), v2) > vec3(EXTENT)) || any(max(max(v0, v1), v2) < vec3(-EXTENT)) {
        return false;
    }

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let normal = cross(e0, e1);
    let d = dot(normal, v0);
    let r = 0.5 * (abs(normal.x) + abs(normal.y) + abs(normal.z));
    if abs(d) > r {
        return false;
    }

    let axes = array<vec3<f32>, 9>(
        vec3(0.0, e0.z, -e0.y),
        vec3(0.0, e1.z, -e1.y),
        vec3(0.0, e2.z, -e2.y),
        vec3(-e0.z, 0.0, e0.x),
        vec3(-e1.z, 0.0, e1.x),
        vec3(-e2.z, 0.0, e2.x),
        vec3(e0.y, -e0.x, 0.0),
        vec3(e1.y, -e1.x, 0.0),
        vec3(e2.y, -e2.x, 0.0),
    );

    for (var i = 0; i < 9; i++) {
        let axis = axes[i];
        let p0 = dot(v0, axis);
        let p1 = dot(v1, axis);
        let p2 = dot(v2, axis);
        let r = 0.5 * (abs(axis.x) + abs(axis.y) + abs(axis.z));
        if max(max(p0, p1), p2) < -r || min(min(p0, p1), p2) > r {
            return false;
        }
    }
    return true;
}

fn tri_weights(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return vec3(u, v, w);
}

fn project_onto_plane(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let normal = normalize(cross(b - a, c - a));
    let dist = dot(normal, p - a);
    return p - dist * normal;
}

fn hash(n: u32) -> u32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

fn rand_vec3(seed: u32) -> vec3<f32> {
    let h1 = hash(seed);
    let h2 = hash(h1);
    let h3 = hash(h2);

    return vec3<f32>(
        f32(h1) / 4294967295.0,
        f32(h2) / 4294967295.0,
        f32(h3) / 4294967295.0
    );
}

fn pack_voxel(normal: vec3<f32>, metallic: f32, roughness: f32) -> u32 {
    let n = encode_normal_octahedral(normal);
    let m = select(1u, 0u, metallic > 0.5);
    let r = u32(roughness * 15.0 + 0.5);
    return (n << 15u) | (m << 14u) | (r << 10u);
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

/// encodes world space normal in lower 17 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn encode_normal_octahedral(normal: vec3<f32>) -> u32 {
    var n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var nrm = vec2<f32>(0.);
    nrm.y = n.y * 0.5 + 0.5;
    nrm.x = n.x * 0.5 + nrm.y;
    nrm.y = n.x * -0.5 + nrm.y;
    let sgn = select(0u, 1u, n.z >= 0.0);
    let res = (u32(nrm.x * 255.0 + 0.5) << 9u) | (u32(nrm.y * 255.0 + 0.5) << 1u) | sgn;
    return res;
}

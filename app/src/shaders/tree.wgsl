struct Chunk {
    // brick_index: u32,
    mask: array<u32, 16>,
}
@group(0) @binding(0) var<storage, read_write> chunk_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> chunks: array<Chunk>;
@group(0) @binding(2) var voxels: texture_storage_3d<r32uint, write>;

struct Allocator {
    chunk_count: atomic<u32>, // the total number of chunks allocated
    voxel_count: atomic<u32>, // total number of voxels in the scene
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var raw_voxels: texture_storage_3d<rg32uint, read>;
@group(1) @binding(2) var<uniform> palette: Palette;

struct ComputeIn {
    @builtin(workgroup_id) chunk_pos: vec3<u32>,
    @builtin(num_workgroups) size_chunks: vec3<u32>,
    @builtin(local_invocation_index) brick_index: u32,
    @builtin(local_invocation_id) brick_pos: vec3<u32>,
    @builtin(global_invocation_id) voxel_pos: vec3<u32>,
}

struct BrickGroup {
    is_empty: bool,
    base_pos: vec3<u32>,
    data: array<u32, 512>,
}
var<workgroup> brick: BrickGroup;

@compute @workgroup_size(8, 8, 8)
fn compute_main(in: ComputeIn) {

    let raw = textureLoad(raw_voxels, vec3<i32>(in.voxel_pos)).rg;
    let albedo_packed = raw.r;
    let normal_packed = raw.g;

    let albedo_srgb = vec3<f32>(
        f32(albedo_packed >> 24u) / 255.0,
        f32((albedo_packed >> 16u) & 0xffu) / 255.0,
        f32((albedo_packed >> 8u) & 0xffu) / 255.0,
    );
    let albedo_linear = srgb_to_linear(albedo_srgb);
    let albedo_oklab = linear_rgb_to_oklab(albedo_linear);

    var palette_index = 0u;
    if albedo_packed > 0u {
        var min_distance = 1e10;
        for (var i = 1u; i < 1024u; i++) {
            let palette_rgb = palette.data[i].rgb;
            let palette_oklab = linear_rgb_to_oklab(palette_rgb);

            let d = distance(palette_oklab, albedo_oklab);
            if d < min_distance {
                palette_index = i;
                min_distance = d;
            }
        }
    }

    let packed = (normal_packed << 11u) | palette_index;

    brick.data[in.brick_index] = packed;

    workgroupBarrier();

    if in.brick_index == 0u {
        // build mask here
        var brick_voxel_count = 0u;
        var mask = array<u32, 16>();
        for (var i = 0u; i < 512u; i++) {
            if brick.data[i] != 0u {
                brick_voxel_count += 1u;
                mask[i >> 5u] |= 1u << (i & 31u);
            }
        }
        if brick_voxel_count == 0u {
            brick.is_empty = true;
        } else {
            let chunk_index = atomicAdd(&alloc.chunk_count, 1u);
            // let brick_index = atomicAdd(&alloc.brick_count, 1u);
            atomicAdd(&alloc.voxel_count, brick_voxel_count);

            // pointer to the chunk
            chunk_indices[in.chunk_pos.z * in.size_chunks.y * in.size_chunks.x + in.chunk_pos.y * in.size_chunks.x + in.chunk_pos.x] = chunk_index + 1u;

            // build chunk
            // chunks[chunk_index].brick_index = brick_index + 1u;
            chunks[chunk_index].mask = mask;

            var size_bricks = textureDimensions(voxels);
                size_bricks.x = (size_bricks.x + 7u) >> 3u;
                size_bricks.y = (size_bricks.y + 7u) >> 3u;
                size_bricks.z = (size_bricks.z + 7u) >> 3u;
            brick.base_pos = vec3<u32>(
                (chunk_index % size_bricks.x) << 3u,
                ((chunk_index / size_bricks.x) % size_bricks.y) << 3u,
                (chunk_index / (size_bricks.x * size_bricks.y)) << 3u
            );
            brick.is_empty = false;
        }
    }

    workgroupBarrier();

    if brick.is_empty {
        return;
    }
    textureStore(voxels, vec3<i32>(brick.base_pos + in.brick_pos), vec4<u32>(packed, 0u, 0u, 0u));
}

fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    return pow(srgb, vec3(2.2));
}

fn linear_rgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
    const im1: mat3x3<f32> = mat3x3<f32>(0.4121656120, 0.2118591070, 0.0883097947,
                              0.5362752080, 0.6807189584, 0.2818474174,
                              0.0514575653, 0.1074065790, 0.6302613616);
    const im2: mat3x3<f32> = mat3x3<f32>(0.2104542553, 1.9779984951, 0.0259040371,
                              0.7936177850, -2.4285922050, 0.7827717662,
                              -0.0040720468, 0.4505937099, -0.8086757660);
    let lms = im1 * rgb;
    return im2 * (sign(lms) * pow(abs(lms), vec3(1.0/3.0)));
}

struct IndexChunk {
	child_index: u32,
	child_mask: array<u32, 2>,
}
@group(0) @binding(0) var<storage, read_write> index_chunks: array<IndexChunk>;
@group(0) @binding(1) var<storage, read_write> leaf_chunks: array<u32>;
@group(0) @binding(2) var<storage, read> child_index_map: array<IndexChunk>;
@group(0) @binding(3) var<storage, read_write> parent_index_map: array<IndexChunk>;

struct Allocator {
    index_chunk_count: atomic<u32>, // the total number of index chunks allocated
    voxel_count: atomic<u32>, // total number of voxels in the scene
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var raw_voxels: texture_storage_3d<rg32uint, read>;
@group(1) @binding(2) var<uniform> palette: Palette;

struct ComputeIn {
    @builtin(global_invocation_id) pos: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_pos: vec3<u32>,
    @builtin(local_invocation_id) local_pos: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

struct Shared {
    mask: array<atomic<u32>, 2>,
    is_empty: bool,
    chunk_index: u32,
}
var<workgroup> group: Shared;

@compute @workgroup_size(4, 4, 4)
fn compute_leaf(in: ComputeIn) {
    let local_index = mask_pos_index(in.local_index);

    let raw = textureLoad(raw_voxels, vec3<i32>(in.pos)).rgb;
    let albedo_packed = raw.r;
    let normal_metallic_roughness = raw.g;
    let material_id = raw.b;

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

        atomicOr(&group.mask[local_index >> 5u], 1u << (local_index & 31u));
    }

    let packed = normal_metallic_roughness | palette_index;

    workgroupBarrier();

    if local_index == 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );
        let voxels_count = countOneBits(mask[0]) + countOneBits(mask[1]);
        
        group.is_empty = voxels_count == 0u;
        if !group.is_empty {
            group.chunk_index = atomicAdd(&alloc.voxel_count, voxels_count);
        }
        
        // used by next pass
        let parent_index = in.workgroup_pos.x
        + in.workgroup_pos.z * in.num_workgroups.x
        + in.workgroup_pos.y * in.num_workgroups.x * in.num_workgroups.z;

        parent_index_map[parent_index].child_index = (group.chunk_index << 1u) | 1u; // lowest bit is 1, as it's a leaf chunk
        parent_index_map[parent_index].child_mask = mask;
    }

    workgroupBarrier();

    if !group.is_empty && albedo_packed > 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );

        let offset = mask_packed_offset(mask, local_index);
        leaf_chunks[group.chunk_index + offset] = packed;
    }
}

/// given mask and index i, gets packed offset based on count of 1s in mask for 0 <= j < i 
fn mask_packed_offset(mask: array<u32, 2>, i: u32) -> u32 {
    if i < 32u {
        return countOneBits(mask[0] & ~(0xffffffffu << i));
    } else {
        return countOneBits(mask[0]) + countOneBits(mask[1] & ~(0xffffffffu << (i - 32u)));
    }
}

/// given local_invocation_index (x + 4y + 16z), get xzy-order index for tree64 ordering
fn mask_pos_index(local_index: u32) -> u32 {
    return ((local_index & 12u) << 2u) | ((local_index & 48u) >> 2u) | (local_index & 3u);
}

@compute @workgroup_size(4, 4, 4)
fn compute_index(in: ComputeIn) {
    let local_index = mask_pos_index(in.local_index);

    let child_offset = in.pos.x
    + in.pos.z * (in.num_workgroups.x << 2u)
    + in.pos.y * (in.num_workgroups.x << 2u) * (in.num_workgroups.z << 2u);

    let chunk = child_index_map[child_offset];

    let child_is_empty = chunk.child_mask[0] == 0u && chunk.child_mask[1] == 0u;

    if !child_is_empty {
        atomicOr(&group.mask[local_index >> 5u], 1u << (local_index & 31u));
    }

    workgroupBarrier();

    if in.local_index == 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );
        let chunk_count = countOneBits(mask[0]) + countOneBits(mask[1]);

        group.is_empty = chunk_count == 0u;
        if !group.is_empty {
            group.chunk_index = atomicAdd(&alloc.index_chunk_count, chunk_count) + 1u; // offset is present to reserve the root chunk for index 0
        }

        if all(in.num_workgroups == vec3(1u)) {
            // this is the last pass, also assign the root chunk
            index_chunks[0].child_index = group.chunk_index << 1u;
            index_chunks[0].child_mask = mask;
            atomicAdd(&alloc.index_chunk_count, 1u);
        } else {
            // used by next pass
            let parent_index = in.workgroup_pos.x
            + in.workgroup_pos.z * in.num_workgroups.x
            + in.workgroup_pos.y * in.num_workgroups.x * in.num_workgroups.z;

            parent_index_map[parent_index].child_index = (group.chunk_index << 1u); // lowest bit is 0, as it's not a leaf chunk
            parent_index_map[parent_index].child_mask = mask;
        }
    }

    workgroupBarrier();

    if !group.is_empty && !child_is_empty {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );

        let offset = mask_packed_offset(mask, local_index);

        index_chunks[group.chunk_index + offset] = chunk;
    }
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

struct IndexChunk {
    child_index: u32,
    child_mask: array<u32, 2>,
}
@group(0) @binding(0) var<storage, read_write> index_chunks: array<IndexChunk>;
@group(0) @binding(1) var<storage, read_write> leaf_chunks: array<u32>;
@group(0) @binding(2) var<storage, read_write> index_leaf_positions: array<vec2<u32>>;

/// these get ping-ponged between passes
/// stores the next index chunk at its location
/// this will all be different when there's dynamic allocation
@group(0) @binding(3) var child_index_map: texture_storage_3d<rgba32uint, read>;
@group(0) @binding(4) var parent_index_map: texture_storage_3d<rgba32uint, write>;

struct Allocator {
    index_chunk_count: atomic<u32>, // the total number of index chunks allocated
    index_leaf_count: atomic<u32>, // total number of index chunks pointing to leaf level, i.e., size of index_leaf_positions
    voxel_count: atomic<u32>, // total number of voxels in the scene
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var raw_chunk_indices: texture_storage_3d<r32uint, read>;
@group(1) @binding(2) var raw_voxels: texture_storage_3d<r32uint, read>;
@group(1) @binding(3) var<uniform> palette: Palette;

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

const RAW_CHUNK_SIZE: u32 = 64u;

@compute @workgroup_size(4, 4, 4)
fn compute_leaf(in: ComputeIn) {
    let chunk = build_leaf_chunk(in.pos << vec3(2));
    let res = gather_next_index_chunk(in, chunk, true);

    /// also, we store the leaf index chunk's packed position
    /// this gets used for voxel pos lookups in the per-voxel lighting pass
    if !res.is_empty {
        index_leaf_positions[res.chunk_index] = pack_voxel_pos(in.pos << vec3(2));
    }
}

@compute @workgroup_size(4, 4, 4)
fn compute_index(in: ComputeIn) {
    let chunk = get_child_chunk(in.pos);
    gather_next_index_chunk(in, chunk, false);
}

struct GatherResult {
    is_empty: bool,
    chunk_index: u32,
}

/// Assembles the shared mask for all threads in the group,
/// and allocates/stores the packed chunk
/// also, caches the next index chunk in parent_index_map
fn gather_next_index_chunk(in: ComputeIn, chunk: IndexChunk, increment_leaf_positions_ctr: bool) -> GatherResult {
    let local_index = mask_pos_index(in.local_index);

    let child_is_empty = chunk.child_mask[0] == 0u && chunk.child_mask[1] == 0u;
    if !child_is_empty {
        atomicOr(&group.mask[local_index >> 5u], 1u << (local_index & 31u));
    }

    workgroupBarrier();

    if local_index == 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );
        let chunk_count = countOneBits(mask[0]) + countOneBits(mask[1]);

        group.is_empty = chunk_count == 0u;
        if !group.is_empty {
            group.chunk_index = atomicAdd(&alloc.index_chunk_count, chunk_count) + 1u; // offset is present to reserve the root chunk for index 0
            if increment_leaf_positions_ctr {
                atomicAdd(&alloc.index_leaf_count, chunk_count);
            }
        }

        if all(in.num_workgroups == vec3(1u)) {
            // this is the last pass, also assign the root chunk
            index_chunks[0].child_index = group.chunk_index << 1u;
            index_chunks[0].child_mask = mask;
            atomicAdd(&alloc.index_chunk_count, 1u);
        } else {
            // used in the next pass
            // we store the chunk_index | leaf flag in the index
            let index_map_val = vec4<u32>(
                (group.chunk_index << 1u) | 0u,
                mask[0],
                mask[1],
                0u,
            );
            textureStore(parent_index_map, in.workgroup_pos, index_map_val);
        }
    }

    workgroupBarrier();

    if !group.is_empty && !child_is_empty {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );

        let offset = mask_packed_offset(mask, local_index);

        let ci = group.chunk_index + offset;
        index_chunks[ci] = chunk;

        var res: GatherResult;
        res.is_empty = false;
        res.chunk_index = ci;
        return res;
    }

    var res: GatherResult;
    res.is_empty = true;
    return res;
}

/// This gets run for every workgroup thread in the leaf chunk invocation
/// allocates, builds, and stores an entire leaf chunk of up to 64 voxels
/// returns the index chunk pointing to the created leaf
///
/// I originally had each thread work one one voxel, rather than an entire leaf chunk,
/// but then I would have to increase the index_map sizes by a factor of 64,
/// which is ~17gb for a 4096x4096 bounding volume. This step should be sufficiently
/// fast with a proper palette LUT.
fn build_leaf_chunk(base_pos: vec3<u32>) -> IndexChunk {
    var voxels: array<u32, 64>;
    var chunk: IndexChunk;

    let raw_chunk_bds = textureDimensions(raw_voxels).xyz / RAW_CHUNK_SIZE;
    let raw_chunk_pos = base_pos / RAW_CHUNK_SIZE;

    var raw_ci = textureLoad(raw_chunk_indices, raw_chunk_pos).r;
    if (raw_ci & 1u) == 0u {
        chunk.child_index = 1u;
        return chunk;
    }
    raw_ci >>= 1u;

    let raw_chunk_offset = vec3<u32>(
        raw_ci % raw_chunk_bds.x,
        (raw_ci / raw_chunk_bds.x) % raw_chunk_bds.y,
        raw_ci / (raw_chunk_bds.x * raw_chunk_bds.y)
    ) * RAW_CHUNK_SIZE;

    let block_raw_offset = base_pos % vec3<u32>(RAW_CHUNK_SIZE);

    for (var i = 0u; i < 64u; i++) {
        let local_pos = vec3<u32>(
            i & 3u,
            (i >> 2u) & 3u,
            (i >> 4u) & 3u,
        );
        let raw_pos = raw_chunk_offset + block_raw_offset + local_pos;
        let voxel = textureLoad(raw_voxels, raw_pos).r;

        let local_index = mask_pos_index(i);
        if voxel != 0u {
            chunk.child_mask[local_index >> 5u] |= 1u << (local_index & 31u);
            voxels[i] = voxel;
        }
    }

    let voxels_count = countOneBits(chunk.child_mask[0]) + countOneBits(chunk.child_mask[1]);

    if voxels_count > 0u {
        chunk.child_index = atomicAdd(&alloc.voxel_count, voxels_count);

        for (var i = 0u; i < 64u; i++) {
            let local_index = mask_pos_index(i);

            if chunk_contains_child(chunk.child_mask, local_index) {
                let offset = mask_packed_offset(chunk.child_mask, local_index);
                leaf_chunks[chunk.child_index + offset] = voxels[i];
            }
        }
    }

    // set the leaf flag to 1
    chunk.child_index = (chunk.child_index << 1u) | 1u;

    return chunk;
}

/// This gets run for every workgroup thread in the index chunk invocation
/// gets the child chunk cached in child_index_map that was written to
/// by the preceding workgroup.
fn get_child_chunk(pos: vec3<u32>) -> IndexChunk {
    let child = textureLoad(child_index_map, pos);

    var chunk: IndexChunk;
    chunk.child_index = child.r;
    chunk.child_mask = array<u32, 2>(child.g, child.b);
    return chunk;
}

/// ------------------------------
/// --------- tree utils ---------

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

fn chunk_contains_child(mask: array<u32, 2>, offset: u32) -> bool {
    let half_mask = select(mask[0], mask[1], offset >= 32u);
    return (half_mask & (1u << (offset & 31u))) != 0u;
}

fn pack_voxel_pos(pos: vec3<u32>) -> vec2<u32> {
    return vec2<u32>(
        ((pos.z & 0xFFFFFu) << 10u) | ((pos.y >> 10u) & 0x3FFu),
        ((pos.y & 0x3FFu) << 20u) | (pos.x & 0xFFFFFu),
    );
}

/// ---------------------------------
/// --------- palette utils ---------

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
    return im2 * (sign(lms) * pow(abs(lms), vec3(1.0 / 3.0)));
}
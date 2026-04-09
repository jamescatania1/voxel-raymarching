struct VisibilityInfo {
    voxel_count: u32,
    chunk_count: u32,
    failed_to_add: u32,
}
struct IndirectArgs {
    workgroups: array<u32, 3>,
}
@group(0) @binding(0) var<storage, read> visibility: VisibilityInfo;
@group(0) @binding(1) var<storage, read_write> args: IndirectArgs;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(1, 1, 1)
fn compute_voxels() {
    args.workgroups[0] = min(65535u, (visibility.voxel_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);
    args.workgroups[1] = 1u;
    args.workgroups[2] = 1u;
}

@compute @workgroup_size(1, 1, 1)
fn compute_chunks() {
    args.workgroups[0] = min(65535u, (visibility.chunk_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);
    args.workgroups[1] = 1u;
    args.workgroups[2] = 1u;
}

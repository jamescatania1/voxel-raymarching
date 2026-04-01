@group(0) @binding(0) var tex_probe_rays: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var tex_probe_irradiance: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var tex_probe_depth: texture_storage_2d<rg16float, read_write>;

struct ComputeIn {
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

@compute @workgroup_size(64, 1, 1)
fn compute_main(in: ComputeIn) {
    let probe_id = in.workgroup_id.x + in.workgroup_id.y * in.num_workgroups.x + in.workgroup_id.z * in.num_workgroups.x * in.num_workgroups.y;

    if in.local_index < 28u {
        let b = BORDER_IRRADIANCE[in.local_index];
        let base = probe_atlas_offset_irradiance(probe_id);
        textureStore(tex_probe_irradiance, base + b.xy, textureLoad(tex_probe_irradiance, base + b.zw));
    }

    if in.local_index < 60u {
        let b = BORDER_DEPTH[in.local_index];
        let base = probe_atlas_offset_depth(probe_id);
        textureStore(tex_probe_depth, base + b.xy, textureLoad(tex_probe_depth, base + b.zw));
    }
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

const BORDER_IRRADIANCE: array<vec4<u32>, 28> = array(
    vec4(0u, 0u, 6u, 6u),
    vec4(1u, 0u, 6u, 1u),
    vec4(2u, 0u, 5u, 1u),
    vec4(3u, 0u, 4u, 1u),
    vec4(4u, 0u, 3u, 1u),
    vec4(5u, 0u, 2u, 1u),
    vec4(6u, 0u, 1u, 1u),
    vec4(7u, 0u, 1u, 6u),
    vec4(0u, 1u, 1u, 6u),
    vec4(7u, 1u, 6u, 6u),
    vec4(0u, 2u, 1u, 5u),
    vec4(7u, 2u, 6u, 5u),
    vec4(0u, 3u, 1u, 4u),
    vec4(7u, 3u, 6u, 4u),
    vec4(0u, 4u, 1u, 3u),
    vec4(7u, 4u, 6u, 3u),
    vec4(0u, 5u, 1u, 2u),
    vec4(7u, 5u, 6u, 2u),
    vec4(0u, 6u, 1u, 1u),
    vec4(7u, 6u, 6u, 1u),
    vec4(0u, 7u, 6u, 1u),
    vec4(1u, 7u, 6u, 6u),
    vec4(2u, 7u, 5u, 6u),
    vec4(3u, 7u, 4u, 6u),
    vec4(4u, 7u, 3u, 6u),
    vec4(5u, 7u, 2u, 6u),
    vec4(6u, 7u, 1u, 6u),
    vec4(7u, 7u, 1u, 1u),
);

const BORDER_DEPTH: array<vec4<u32>, 60> = array(
    vec4(0u, 0u, 14u, 14u),
    vec4(1u, 0u, 14u, 1u),
    vec4(2u, 0u, 13u, 1u),
    vec4(3u, 0u, 12u, 1u),
    vec4(4u, 0u, 11u, 1u),
    vec4(5u, 0u, 10u, 1u),
    vec4(6u, 0u, 9u, 1u),
    vec4(7u, 0u, 8u, 1u),
    vec4(8u, 0u, 7u, 1u),
    vec4(9u, 0u, 6u, 1u),
    vec4(10u, 0u, 5u, 1u),
    vec4(11u, 0u, 4u, 1u),
    vec4(12u, 0u, 3u, 1u),
    vec4(13u, 0u, 2u, 1u),
    vec4(14u, 0u, 1u, 1u),
    vec4(15u, 0u, 1u, 14u),
    vec4(0u, 1u, 1u, 14u),
    vec4(15u, 1u, 14u, 14u),
    vec4(0u, 2u, 1u, 13u),
    vec4(15u, 2u, 14u, 13u),
    vec4(0u, 3u, 1u, 12u),
    vec4(15u, 3u, 14u, 12u),
    vec4(0u, 4u, 1u, 11u),
    vec4(15u, 4u, 14u, 11u),
    vec4(0u, 5u, 1u, 10u),
    vec4(15u, 5u, 14u, 10u),
    vec4(0u, 6u, 1u, 9u),
    vec4(15u, 6u, 14u, 9u),
    vec4(0u, 7u, 1u, 8u),
    vec4(15u, 7u, 14u, 8u),
    vec4(0u, 8u, 1u, 7u),
    vec4(15u, 8u, 14u, 7u),
    vec4(0u, 9u, 1u, 6u),
    vec4(15u, 9u, 14u, 6u),
    vec4(0u, 10u, 1u, 5u),
    vec4(15u, 10u, 14u, 5u),
    vec4(0u, 11u, 1u, 4u),
    vec4(15u, 11u, 14u, 4u),
    vec4(0u, 12u, 1u, 3u),
    vec4(15u, 12u, 14u, 3u),
    vec4(0u, 13u, 1u, 2u),
    vec4(15u, 13u, 14u, 2u),
    vec4(0u, 14u, 1u, 1u),
    vec4(15u, 14u, 14u, 1u),
    vec4(0u, 15u, 14u, 1u),
    vec4(1u, 15u, 14u, 14u),
    vec4(2u, 15u, 13u, 14u),
    vec4(3u, 15u, 12u, 14u),
    vec4(4u, 15u, 11u, 14u),
    vec4(5u, 15u, 10u, 14u),
    vec4(6u, 15u, 9u, 14u),
    vec4(7u, 15u, 8u, 14u),
    vec4(8u, 15u, 7u, 14u),
    vec4(9u, 15u, 6u, 14u),
    vec4(10u, 15u, 5u, 14u),
    vec4(11u, 15u, 4u, 14u),
    vec4(12u, 15u, 3u, 14u),
    vec4(13u, 15u, 2u, 14u),
    vec4(14u, 15u, 1u, 14u),
    vec4(15u, 15u, 1u, 1u),
);
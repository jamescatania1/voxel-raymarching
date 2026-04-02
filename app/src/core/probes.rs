use std::collections::VecDeque;
use utils::tree::{Tree, chunk_offset, offset_pos};

#[derive(Debug)]
pub struct ProbeGrid {
    pub size: glam::UVec3,
    pub scale: u32,
    pub probes: Vec<Probe>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Probe {
    pub position: glam::Vec3,
    pub _pad: u32,
}

impl ProbeGrid {
    pub fn new(size: glam::UVec3, scale: u32, tree: &Tree) -> Self {
        let size = size.map(|x| x.div_ceil(scale));

        let mut probes = Vec::with_capacity(size.element_product() as usize);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    probes.push(Probe {
                        position: (glam::uvec3(x, y, z).as_vec3() + 0.5) * scale as f32,
                        ..Default::default()
                    });
                }
            }
        }

        let mut queue = VecDeque::new();
        queue.push_back((0, tree.size, glam::UVec3::ZERO));

        while let Some((cur, chunk_size, chunk_pos)) = queue.pop_front() {
            if scale == chunk_size {
                // non-empty chunk of the target size here
                // we iterate level order BFS through this chunk like in the outer loop
                // until we find the largest, center-most empty child to place our probe in
                let probe_pos = chunk_pos / scale;
                let probe_index =
                    probe_pos.x + probe_pos.y * size.x + probe_pos.z * size.x * size.y;

                let mut queue = VecDeque::new();
                queue.push_back((cur, chunk_size, chunk_pos));

                while let Some((cur, chunk_size, chunk_pos)) = queue.pop_front() {
                    let chunk = &tree.chunks[cur as usize];
                    let next_size = chunk_size / 4;
                    let mut found = false;
                    for (i, offset) in CHILD_OFFSETS_WINDING {
                        if !chunk.contains_child(i) {
                            probes[probe_index as usize].position =
                                (chunk_pos + offset * next_size).as_vec3() + 0.5 * next_size as f32;
                            found = true;
                            break;
                        }
                        if !chunk.is_leaf() {
                            let child_index = chunk.get_child(i);
                            queue.push_back((
                                child_index,
                                next_size,
                                chunk_pos + offset * next_size,
                            ));
                        }
                    }
                    if found {
                        break;
                    }
                }
            }
            let chunk = &tree.chunks[cur as usize];
            if scale >= chunk_size || chunk.is_leaf() {
                continue;
            }
            let next_size = chunk_size / 4;
            for i in 0..64 {
                if chunk.contains_child(i) {
                    let offset = offset_pos(i);
                    let child_index = chunk.get_child(i);
                    queue.push_back((child_index, next_size, chunk_pos + offset * next_size));
                }
            }
        }

        dbg!(&probes);

        Self {
            size,
            scale,
            probes,
        }
    }
}

const CHILD_OFFSETS_WINDING: [(u32, glam::UVec3); 64] = child_offsets_winding();

const fn child_offsets_winding() -> [(u32, glam::UVec3); 64] {
    const OFFSETS_WINDING: [glam::UVec3; 64] = [
        // inner 2x2x2
        glam::uvec3(1, 1, 1),
        glam::uvec3(2, 1, 1),
        glam::uvec3(1, 2, 1),
        glam::uvec3(2, 2, 1),
        glam::uvec3(1, 1, 2),
        glam::uvec3(2, 1, 2),
        glam::uvec3(1, 2, 2),
        glam::uvec3(2, 2, 2),
        // outer face inner 2x2s
        glam::uvec3(1, 1, 0),
        glam::uvec3(2, 1, 0),
        glam::uvec3(1, 2, 0),
        glam::uvec3(2, 2, 0),
        glam::uvec3(1, 1, 3),
        glam::uvec3(2, 1, 3),
        glam::uvec3(1, 2, 3),
        glam::uvec3(2, 2, 3),
        glam::uvec3(1, 0, 1),
        glam::uvec3(2, 0, 1),
        glam::uvec3(1, 0, 2),
        glam::uvec3(2, 0, 2),
        glam::uvec3(1, 3, 1),
        glam::uvec3(2, 3, 1),
        glam::uvec3(1, 3, 2),
        glam::uvec3(2, 3, 2),
        glam::uvec3(0, 1, 1),
        glam::uvec3(0, 2, 1),
        glam::uvec3(0, 1, 2),
        glam::uvec3(0, 2, 2),
        glam::uvec3(3, 1, 1),
        glam::uvec3(3, 2, 1),
        glam::uvec3(3, 1, 2),
        glam::uvec3(3, 2, 2),
        // non-corner edges
        glam::uvec3(1, 0, 0),
        glam::uvec3(2, 0, 0),
        glam::uvec3(1, 3, 0),
        glam::uvec3(2, 3, 0),
        glam::uvec3(1, 0, 3),
        glam::uvec3(2, 0, 3),
        glam::uvec3(1, 3, 3),
        glam::uvec3(2, 3, 3),
        glam::uvec3(0, 1, 0),
        glam::uvec3(0, 2, 0),
        glam::uvec3(3, 1, 0),
        glam::uvec3(3, 2, 0),
        glam::uvec3(0, 1, 3),
        glam::uvec3(0, 2, 3),
        glam::uvec3(3, 1, 3),
        glam::uvec3(3, 2, 3),
        glam::uvec3(0, 0, 1),
        glam::uvec3(0, 0, 2),
        glam::uvec3(3, 0, 1),
        glam::uvec3(3, 0, 2),
        glam::uvec3(0, 3, 1),
        glam::uvec3(0, 3, 2),
        glam::uvec3(3, 3, 1),
        glam::uvec3(3, 3, 2),
        // corners
        glam::uvec3(0, 0, 0),
        glam::uvec3(3, 0, 0),
        glam::uvec3(0, 3, 0),
        glam::uvec3(3, 3, 0),
        glam::uvec3(0, 0, 3),
        glam::uvec3(3, 0, 3),
        glam::uvec3(0, 3, 3),
        glam::uvec3(3, 3, 3),
    ];
    let mut res = [(0, glam::UVec3::ZERO); 64];
    let mut i = 0;
    while i < 64 {
        res[i] = (chunk_offset(OFFSETS_WINDING[i]), OFFSETS_WINDING[i]);
        i += 1;
    }
    res
}

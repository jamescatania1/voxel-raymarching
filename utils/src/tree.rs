#[derive(Debug, Clone)]
pub struct Tree {
    pub size: u32,
    pub chunks: Vec<IndexChunk>,
}

impl Tree {
    pub fn new(size: u32, data: &[u8]) -> Self {
        Self {
            size,
            chunks: bytemuck::cast_slice(data).to_vec(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndexChunk {
    child_index: u32,
    mask: [u32; 2],
}

impl IndexChunk {
    pub fn contains_child(&self, offset: u32) -> bool {
        let half_mask = if offset >= 32 {
            self.mask[1]
        } else {
            self.mask[0]
        };
        return (half_mask & (1 << (offset & 31))) != 0;
    }
    pub fn is_leaf(&self) -> bool {
        (self.child_index & 1) == 1
    }
    pub fn mask_packed_offset(&self, i: u32) -> u32 {
        if i < 32 {
            (self.mask[0] & !(0xffffffff << i)).count_ones()
        } else {
            self.mask[0].count_ones() + (self.mask[1] & !(0xffffffff << (i - 32))).count_ones()
        }
    }
    pub fn get_child(&self, offset: u32) -> u32 {
        (self.child_index >> 1) + self.mask_packed_offset(offset)
    }
}

pub const fn chunk_offset(pos: glam::UVec3) -> u32 {
    pos.y << 4 | pos.z << 2 | pos.x
}

pub const fn offset_pos(offset: u32) -> glam::UVec3 {
    glam::uvec3(offset & 3, offset >> 4, (offset >> 2) & 3)
}

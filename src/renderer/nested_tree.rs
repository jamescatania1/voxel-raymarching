pub struct TwoTree {
    pub size: glam::UVec3,
    pub chunk_indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
    pub bricks: Vec<Brick>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Chunk {
    brick_index: u32,
    mask: [u32; 16],
}
impl Default for Chunk {
    fn default() -> Self {
        Self {
            brick_index: 0,
            mask: [0; 16],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Brick {
    data: [u8; 512],
}
impl Default for Brick {
    fn default() -> Self {
        Self { data: [0; 512] }
    }
}

fn chunk_index(size: glam::UVec3, pos: glam::UVec3) -> u32 {
    (pos.x >> 3) * size.y * size.z + (pos.y >> 3) * size.z + (pos.z >> 3)
}
fn voxel_index(pos: glam::UVec3) -> u32 {
    (pos.x & 7) << 6 | (pos.y & 7) << 3 | (pos.z & 7)
}

impl TwoTree {
    // pub const fn bitmask_lut() -> [u64; 64 * 8] {
    //     // for ray_dir in 0..8 {
    //     //     let dir = glam::
    //     // }
    // }
    //

    pub fn new(size: glam::UVec3) -> Self {
        let size = size.map(|x| (x + 7) >> 3);
        Self {
            size,
            chunk_indices: vec![0; size.element_product() as usize],
            chunks: Vec::new(),
            bricks: Vec::new(),
        }
    }

    pub fn get(&self, pos: glam::UVec3) -> u8 {
        let ci = self.chunk_indices[chunk_index(self.size, pos) as usize];
        if ci == 0 {
            return 0;
        }
        let chunk = &self.chunks[(ci - 1) as usize];

        let voxel_index = voxel_index(pos);

        if chunk.mask[(voxel_index >> 5) as usize] & (1 << (voxel_index & 31)) == 0 {
            return 0;
        }
        self.bricks[chunk.brick_index as usize - 1].data[voxel_index as usize]
    }

    pub fn insert(&mut self, pos: glam::UVec3, value: u8) {
        let pos_index = chunk_index(self.size, pos) as usize;
        let mut ci = self.chunk_indices[pos_index];
        if ci == 0 {
            if value == 0 {
                return;
            }
            ci = self.chunks.len() as u32 + 1;
            self.chunk_indices[pos_index] = ci;
            self.chunks.push(Chunk::default());
        }
        let chunk = &mut self.chunks[ci as usize - 1];
        let voxel_index = voxel_index(pos);

        if value == 0 {
            chunk.mask[(voxel_index >> 5) as usize] &= !(1 << (voxel_index & 31));
        } else {
            chunk.mask[(voxel_index >> 5) as usize] |= 1 << (voxel_index & 31);
        }

        if chunk.brick_index == 0 {
            // allocate new brick

            if value == 0 {
                return;
            }
            chunk.brick_index = self.bricks.len() as u32 + 1;
            self.bricks.push(Brick::default());
        }
        self.bricks[chunk.brick_index as usize - 1].data[voxel_index as usize] = value;
    }

    pub fn from_scene(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();
        let mut _self = Self::new(scene.size.as_uvec3());

        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_uvec3();
                _self.insert(pos, palette_index);
            }
        }

        println!("built tree in {:#?}", timer.elapsed());
        println!(
            "tree length: {} MB",
            (_self.chunks.len() * 68 + _self.bricks.len() * 512) as f64 / 1000000.0
        );
        println!(
            "- chunks total length: {} MB",
            (_self.chunks.len() * 68) as f64 / 1000000.0
        );
        println!(
            "- bricks total length: {} MB",
            (_self.bricks.len() * 512) as f64 / 1000000.0
        );
        println!(
            "original length: {} mb",
            (scene.size.element_product()) as f64 / 1000000.0
        );
        _self
    }

    pub fn d_from_scene(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut _self = Self::new(scene.size.as_uvec3());
        let mut voxels = vec![0; scene.size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;
                voxels[index] = palette_index;
            }
        }

        for x in 0..scene.size.x {
            for y in 0..scene.size.y {
                for z in 0..scene.size.z {
                    let pos = glam::ivec3(x, y, z).as_uvec3();
                    let index = pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                    let voxel = voxels[index];
                    _self.insert(pos, voxel);
                }
            }
        }
        // let _t = std::time::Instant::now();
        // _self.merge_optimize();
        // println!("merge optmization took {:#?}", _t.elapsed());

        {
            let mut errors = 0;
            for x in 0..scene.size.x {
                for y in 0..scene.size.y {
                    for z in 0..scene.size.z {
                        let pos = glam::ivec3(x, y, z).as_uvec3();
                        let index =
                            pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                        let voxel = voxels[index];
                        // assert_eq!(voxel, tree.get(pos));
                        let tval = _self.get(pos);
                        if tval != voxel {
                            println!(
                                "pos {} failed. actual: {}, tree: {}",
                                pos,
                                voxel,
                                _self.get(pos)
                            );
                            errors += 1;
                        }
                    }
                }
            }
            println!("finished with {} errors", errors);
        }

        println!("built tree in {:#?}", timer.elapsed());
        println!(
            "tree length: {} MB",
            (_self.chunks.len() * 68 + _self.bricks.len() * 512) as f64 / 1000000.0
        );
        println!(
            "- chunks total length: {} MB",
            (_self.chunks.len() * 68) as f64 / 1000000.0
        );
        println!(
            "- bricks total length: {} MB",
            (_self.bricks.len() * 512) as f64 / 1000000.0
        );
        println!(
            "original length: {} mb",
            (scene.size.element_product()) as f64 / 1000000.0
        );
        _self
    }
}

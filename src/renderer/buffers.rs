use crate::renderer::tree::Tree;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneDataBuffer {
    size: [u32; 3],
    _pad: u32,
    palette: [u32; 256],
}

impl SceneDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
        let mut palette = [0; 256];
        for (i, mat) in scene.palette.iter().enumerate() {
            let rgba = mat.rgba;
            palette[i] = (rgba[0] as u32) << 24
                | (rgba[1] as u32) << 16
                | (rgba[2] as u32) << 8
                | rgba[3] as u32;
        }
        Self {
            size: [
                scene.size.x as u32,
                scene.size.y as u32,
                scene.size.z as u32,
            ],
            _pad: 0,
            palette,
        }
    }
}

pub struct VoxelDataBuffer(pub Vec<u32>);

impl VoxelDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut voxels = vec![0; ((scene.size.element_product() as usize) + 3) >> 2];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;
                voxels[index >> 2] |= (palette_index as u32) << 8 * (index & 3);
            }
        }

        println!("voxel data load took {:#?}", timer.elapsed());
        Self(voxels)
    }

    pub fn build_tree(scene: &crate::vox::Scene) {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut voxels = vec![0; scene.size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;
                voxels[index] = palette_index;
            }
        }

        let mut tree = Tree::new(scene.size.as_uvec3());

        for x in 0..scene.size.x {
            for y in 0..scene.size.y {
                for z in 0..scene.size.z {
                    let pos = glam::ivec3(x, y, z).as_uvec3();
                    let index = pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                    let voxel = voxels[index];
                    tree.insert(pos, voxel);
                }
            }
        }
        let mut errors = 0;
        for x in 0..scene.size.x {
            for y in 0..scene.size.y {
                for z in 0..scene.size.z {
                    let pos = glam::ivec3(x, y, z).as_uvec3();
                    let index = pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                    let voxel = voxels[index];
                    // assert_eq!(voxel, tree.get(pos));
                    let tval = tree.get(pos);
                    if tval != voxel {
                        println!(
                            "pos {} failed. actual: {}, tree: {}",
                            pos,
                            voxel,
                            tree.get(pos)
                        );
                        errors += 1;
                    }
                }
            }
        }
        println!("finished with {} errors", errors);

        println!("built tree in {:#?}", timer.elapsed());
    }
}

// type TreeKey = u32;

// enum Node {
//     Leaf(u8),
//     Node(Box<[TreeKey; 64]>),
// }

// struct Tree {
//     size: u32,
//     nodes: Vec<Node>,
// }
// impl Tree {
//     fn new(size: glam::UVec3) -> Self {
//         Self {
//             size: size.max_element().next_power_of_two(),
//             nodes: vec![Node::Leaf(0)],
//         }
//     }
//     fn insert(&mut self, pos: glam::UVec3, value: u8) {
//         let mut cur: TreeKey = 0;
//         let mut shift = self.size.ilog2() as i32 - 2;

//         while shift > 0 {
//             let [x, y, z] = [
//                 (pos.x >> shift) & 3,
//                 (pos.y >> shift) & 3,
//                 (pos.z >> shift) & 3,
//             ];
//             let index = x << 4 | y << 2 | z;
//             let node = &self.nodes[cur as usize];
//             match node {
//                 Node::Leaf(v) => {
//                     if *v == value {
//                         // already the same value
//                         return;
//                     }
//                     let mut new_nodes = vec![];
//                     let mut parent = Box::new([0; 64]);
//                     for i in 0..64 {
//                         let child = Node::Leaf(*v);
//                         parent[i as usize] = self.nodes.len() as u32 + i;
//                         new_nodes.push(child);
//                     }
//                     self.nodes[cur as usize] = Node::Node(parent);
//                     cur = self.nodes.len() as u32 + index;
//                     self.nodes.append(&mut new_nodes);
//                 }
//                 Node::Node(node) => {
//                     cur = node[(index) as usize];
//                 }
//             };
//             shift -= 2;
//         }

//         let [x, y, z] = if shift == 0 {
//             [pos.x & 3, pos.y & 3, pos.z & 3]
//         } else {
//             [pos.x & 1, pos.y & 1, pos.z & 1]
//         };
//         let index = x << 4 | y << 2 | z;
//         if let Node::Leaf(v) = self.nodes[cur as usize] {
//             // avoid rewriting the same leaf node
//             if v == value {
//                 return;
//             }
//         }
//     }
// }

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraDataBuffer {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub ws_position: [f32; 3],
    pub _pad: f32,
}

impl CameraDataBuffer {
    pub fn update(&mut self, camera: &crate::engine::Camera) {
        self.view_proj = camera.view_proj.as_mat4().to_cols_array_2d();
        self.inv_view_proj = camera.inv_view_proj.as_mat4().to_cols_array_2d();
        self.ws_position = camera.position.as_vec3().to_array();
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],
}

impl ModelDataBuffer {
    pub fn update(&mut self, model: &crate::engine::Model) {
        self.transform = model.transform.to_cols_array_2d();
        self.inv_transform = model.inv_transform.to_cols_array_2d();
    }
}

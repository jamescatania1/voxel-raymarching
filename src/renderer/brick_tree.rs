pub struct BrickTree {
    size: glam::UVec3,
    bricks: glam::Uvec3,
    shift: glam::UVec3,
    index: Box<[u32; 512]>,
    voxels: Vec<[u8; 512]>,
}

// type NodeId = u32;

// #[derive(Clone, Copy)]
// enum Node {
//     Leaf(u8),
//     /// Here, `Internal(NodeId)` stores the index of the first child node
//     Internal(NodeId),
// }

impl BrickTree {
    pub fn new(size: glam::UVec3) -> Self {
        let size = size.map(|x| x.next_power_of_two());
        let bricks = size.map(|x| x >> 3);
        let shift = size.map(|x| x.ilog2() - 3);

        Self {
            size,
            shift,
            bricks,
            index: Box::new([0; 512]),
            voxels: Vec::new(),
        }
    }

    pub fn from_scene(scene: &crate::vox::Scene) -> Self {
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
        // println!("tree length: {} bytes", _self.nodes.len() * 4);
        // println!("original length: {} bytes", scene.size.element_product());
        _self
    }

    pub fn get(&self, pos: glam::UVec3) -> u8 {
        let i = self

        let mut depth = self.depth as i32 - 1;
        let mut cur = 0;

        loop {
            let [x, y, z] = [
                (px >> depth * 2) & 3,
                (py >> depth * 2) & 3,
                (pz >> depth * 2) & 3,
            ];
            let i = x << 4 | y << 2 | z;
            // (xyz, i) are pos and index at this depth

            let node = self.nodes[(cur + i) as usize];
            if node.is_leaf() {
                return node.leaf_value();
            }

            cur = node.first_child();
            depth -= 1;
        }
    }

    pub fn insert(&mut self, position: glam::UVec3, value: u8) {
        let [px, py, pz] = [position.x, position.y, position.z];

        let mut depth = self.depth as i32 - 1;
        let mut cur = 0;

        loop {
            let [x, y, z] = [
                (px >> depth * 2) & 3,
                (py >> depth * 2) & 3,
                (pz >> depth * 2) & 3,
            ];
            let i = x << 4 | y << 2 | z;
            // (xyz, i) are pos and index at this depth

            let node = self.nodes[(cur + i) as usize];

            if node.is_leaf() {
                let x = node.leaf_value();
                if x == value {
                    // already there
                    return;
                }
                if depth == 0 {
                    // rewrite existing
                    self.nodes[(cur + i) as usize] = (value as u32).leaf();
                    return;
                }
                let first_child = self.nodes.len() as u32;
                self.nodes.extend([node; 64]);
                self.nodes[(cur + i) as usize] = first_child.internal();
                cur = first_child;
            } else {
                cur = node.first_child();
            }
            depth -= 1;
        }
    }

    fn merge_optimize(&mut self) {
        let mut cur = self.nodes.len() as u32 - 64;

        let mut merged = 0;

        loop {
            if cur == 0 {
                break;
            }
            let mut equal = true;
            let val = self.nodes[cur as usize];
            if !val.is_leaf() {
                cur -= 64;
                continue;
            }
            for j in 1..32 {
                if val != self.nodes[j + cur as usize] {
                    equal = false;
                    break;
                }
            }
            if !equal {
                cur -= 64;
                continue;
            }
            merged += 1;
            let candidate = val.internal();
            let new = val.leaf();
            for j in 0..cur {
                if self.nodes[j as usize] == candidate {
                    self.nodes[j as usize] = new;
                }
            }
            cur -= 64;
        }
        println!("merged {} cubes", merged);
    }
}

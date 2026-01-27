pub struct Tree {
    depth: u32,
    nodes: Vec<Node>,
}

type NodeId = u32;

#[derive(Clone, Copy)]
enum Node {
    Leaf(u8),
    /// Here, `Internal(NodeId)` stores the index of the first child node
    Internal(NodeId),
}

impl Tree {
    pub fn new(size: glam::UVec3) -> Self {
        let depth = size.max_element().next_power_of_two().ilog2().div_ceil(2);
        Self {
            depth,
            nodes: vec![Node::Leaf(0)],
        }
    }

    pub fn get(&self, position: glam::UVec3) -> u8 {
        let [px, py, pz] = [position.x, position.y, position.z];

        let mut depth = self.depth as i32 - 1;
        let mut cur = 0;

        loop {
            match self.nodes[cur as usize] {
                Node::Leaf(x) => return x,
                Node::Internal(first_child) => {
                    let [x, y, z] = [
                        (px >> depth * 2) & 3,
                        (py >> depth * 2) & 3,
                        (pz >> depth * 2) & 3,
                    ];
                    let i = x << 4 | y << 2 | z;
                    // (xyz, i) are pos and index at this depth

                    cur = first_child + i;
                    depth -= 1;
                }
            }
        }
    }

    pub fn insert(&mut self, position: glam::UVec3, value: u8) {
        let [px, py, pz] = [position.x, position.y, position.z];

        let mut depth = self.depth as i32 - 1;
        let mut cur = 0;

        loop {
            match self.nodes[cur as usize] {
                Node::Leaf(x) => {
                    if x == value {
                        return;
                    }
                    if depth < 0 {
                        self.nodes[cur as usize] = Node::Leaf(value);
                        return;
                    } else {
                        let first_child = self.nodes.len() as u32;
                        self.nodes.extend([Node::Leaf(x); 64]);
                        self.nodes[cur as usize] = Node::Internal(first_child);
                    }
                }
                Node::Internal(first_child) => {
                    let [x, y, z] = [
                        (px >> depth * 2) & 3,
                        (py >> depth * 2) & 3,
                        (pz >> depth * 2) & 3,
                    ];
                    let i = x << 4 | y << 2 | z;
                    // (xyz, i) are pos and index at this depth

                    cur = first_child + i;
                    depth -= 1;
                }
            }
        }
    }
}

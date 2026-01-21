use wgpu::{Buffer, util::DeviceExt};

pub trait IntoMesh {
    fn mesh(&mut self, device: &wgpu::Device) -> Mesh;
}

/// Mesh object that is to be used with the `base.wgsl` shader.
#[derive(Debug)]
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16], index_count: u32) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 3],
}
pub const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

#[inline]
fn vertex(position: [f32; 3], color: [f32; 3]) -> Vertex {
    Vertex {
        position: [position[0], position[1], position[2], 1.0],
        color,
    }
}

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
const PINK: [f32; 3] = [1.0, 0.0, 1.0];

/// `Mesh` variant of a 3D cube
#[derive(Debug)]
pub struct Cube {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

impl Cube {
    pub fn new() -> Self {
        let vertices = vec![
            // top
            vertex([-1.0, -1.0, 1.0], RED),
            vertex([1.0, -1.0, 1.0], RED),
            vertex([1.0, 1.0, 1.0], RED),
            vertex([-1.0, 1.0, 1.0], RED),
            // bottom
            vertex([-1.0, 1.0, -1.0], GREEN),
            vertex([1.0, 1.0, -1.0], GREEN),
            vertex([1.0, -1.0, -1.0], GREEN),
            vertex([-1.0, -1.0, -1.0], GREEN),
            // right
            vertex([1.0, -1.0, -1.0], BLUE),
            vertex([1.0, 1.0, -1.0], BLUE),
            vertex([1.0, 1.0, 1.0], BLUE),
            vertex([1.0, -1.0, 1.0], BLUE),
            // left
            vertex([-1.0, -1.0, 1.0], YELLOW),
            vertex([-1.0, 1.0, 1.0], YELLOW),
            vertex([-1.0, 1.0, -1.0], YELLOW),
            vertex([-1.0, -1.0, -1.0], YELLOW),
            // front
            vertex([1.0, 1.0, -1.0], CYAN),
            vertex([-1.0, 1.0, -1.0], CYAN),
            vertex([-1.0, 1.0, 1.0], CYAN),
            vertex([1.0, 1.0, 1.0], CYAN),
            // back
            vertex([1.0, -1.0, 1.0], PINK),
            vertex([-1.0, -1.0, 1.0], PINK),
            vertex([-1.0, -1.0, -1.0], PINK),
            vertex([1.0, -1.0, -1.0], PINK),
        ];

        let indices: Vec<u16> = vec![
            0, 1, 2, 2, 3, 0, // top
            4, 5, 6, 6, 7, 4, // bottom
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // front
            20, 21, 22, 22, 23, 20, // back
        ];

        Self { vertices, indices }
    }
}

impl IntoMesh for Cube {
    fn mesh(&mut self, device: &wgpu::Device) -> Mesh {
        Mesh::new(
            device,
            &self.vertices,
            &self.indices,
            self.indices.len() as u32,
        )
    }
}

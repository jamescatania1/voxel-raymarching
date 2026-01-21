use bytemuck::{Pod, Zeroable};
use std::mem;
use wgpu::{Buffer, Device, util::DeviceExt};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 3],
}

pub const VERTEX_SIZE: u64 = mem::size_of::<Vertex>() as u64;

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
const PINK: [f32; 3] = [1.0, 0.0, 1.0];

fn vertex(position: [f32; 3], color: [f32; 3]) -> Vertex {
    Vertex {
        position: [position[0], position[1], position[2], 1.0],
        color,
    }
}

#[derive(Debug)]
pub struct Cube {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl Cube {
    pub fn new(device: &Device) -> Self {
        let (vertices, indices) = {
            let vertices = vec![
                // top (0, 0, 1)
                vertex([-1.0, -1.0, 1.0], RED),
                vertex([1.0, -1.0, 1.0], RED),
                vertex([1.0, 1.0, 1.0], RED),
                vertex([-1.0, 1.0, 1.0], RED),
                // bottom (0, 0, -1)
                vertex([-1.0, 1.0, -1.0], GREEN),
                vertex([1.0, 1.0, -1.0], GREEN),
                vertex([1.0, -1.0, -1.0], GREEN),
                vertex([-1.0, -1.0, -1.0], GREEN),
                // right (1, 0, 0)
                vertex([1.0, -1.0, -1.0], BLUE),
                vertex([1.0, 1.0, -1.0], BLUE),
                vertex([1.0, 1.0, 1.0], BLUE),
                vertex([1.0, -1.0, 1.0], BLUE),
                // left (-1, 0, 0)
                vertex([-1.0, -1.0, 1.0], YELLOW),
                vertex([-1.0, 1.0, 1.0], YELLOW),
                vertex([-1.0, 1.0, -1.0], YELLOW),
                vertex([-1.0, -1.0, -1.0], YELLOW),
                // front (0, 1, 0)
                vertex([1.0, 1.0, -1.0], CYAN),
                vertex([-1.0, 1.0, -1.0], CYAN),
                vertex([-1.0, 1.0, 1.0], CYAN),
                vertex([1.0, 1.0, 1.0], CYAN),
                // back (0, -1, 0)
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
                20, 22, 21, 22, 23, 20, // back
            ];

            (vertices, indices)
        };

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
            index_count: indices.len() as u32,
        }
    }
}

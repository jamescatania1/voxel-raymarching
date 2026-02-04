use std::{
    collections::{HashMap, VecDeque},
    io::{BufReader, Cursor},
    sync::Arc,
};

use anyhow::Context;
use glam::Mat4;
use models::Gltf;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{RendererCtx, SizedWindow, engine::Engine, renderer::buffers::CameraDataBuffer};

#[derive(Debug)]
pub struct ModelLoader {
    gltf: Gltf,
    nodes: Vec<Node>,
    meshes: Vec<Mesh>,
    pipeline: wgpu::RenderPipeline,
    bg_camera: wgpu::BindGroup,
    bg_model: wgpu::BindGroup,
    buffer_camera: wgpu::Buffer,
    buffer_model: wgpu::Buffer,
    view_depth: wgpu::TextureView,
}

#[derive(Debug)]
struct Node {
    mesh_id: usize,
    transform: Mat4,
}

#[derive(Debug)]
struct Mesh {
    primitives: Vec<Primitive>,
}

#[derive(Debug)]
struct Primitive {
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: u32,
    vertex_buffer: wgpu::Buffer,
    normal_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub matrix: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 3],
    pub _pad: [f32; 36],
}
impl Default for ModelDataBuffer {
    fn default() -> Self {
        Self {
            matrix: Default::default(),
            normal_matrix: Default::default(),
            _pad: [0.0; 36],
        }
    }
}

impl ModelLoader {
    pub fn new(
        window: Arc<Window>,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        engine: &Engine,
    ) -> anyhow::Result<Self> {
        let src = std::include_bytes!("../../assets/sponza.glb");
        let mut src = BufReader::new(Cursor::new(src));

        let gltf = Gltf::parse(&mut src)?;

        let (meshes, nodes) = {
            let mut meshes = Vec::new();
            let mut nodes = Vec::new();

            let scene = gltf
                .meta
                .scenes
                .get(gltf.meta.scene.context("no default scene")? as usize)
                .context("unable to find default scene")?;

            let mut mesh_id_map = HashMap::new();
            let mut visit_queue = VecDeque::new();
            for node in &scene.nodes {
                visit_queue.push_back((*node, models::GLTF_Y_UP_TO_Z_UP));
            }

            // visit breadth first over the scene, flattening the matrix transform
            while let Some((node_id, parent_matrix)) = visit_queue.pop_front() {
                let node = gltf
                    .meta
                    .nodes
                    .get(node_id as usize)
                    .context(format!("unable to find node with id {}", node_id))?;
                let transform = parent_matrix * node.transform.matrix;

                // add children to visit
                for child_id in &node.children {
                    visit_queue.push_back((*child_id, transform));
                }

                let Some(gltf_mesh_id) = node.mesh else {
                    continue;
                };
                // current node has a mesh attached
                // our final node list is flat and only has ones with meshes
                if let Some(mesh_id) = mesh_id_map.get(&gltf_mesh_id) {
                    nodes.push(Node {
                        mesh_id: *mesh_id,
                        transform,
                    });
                } else {
                    nodes.push(Node {
                        mesh_id: meshes.len(),
                        transform,
                    });

                    // now, create and push a new mesh
                    let gltf_mesh = gltf
                        .meta
                        .meshes
                        .get(gltf_mesh_id as usize)
                        .context("invalid mesh id")?;
                    let primitives = gltf_mesh
                        .primitives
                        .iter()
                        .filter_map(|p| {
                            // only pay attention to triangle list primitives for now
                            if p.mode != models::schema::DrawMode::Triangles {
                                return None;
                            }

                            // p.indices is None if there's no index buffer and it's direct vertex list
                            // ignore that case for now
                            let indices_acc_index = p.indices?;

                            // should also get texcoord attributes here
                            let &vpos_acc_index = p.attributes.get("POSITION")?;
                            let &vnormal_acc_index = p.attributes.get("NORMAL")?;

                            let indices_acc =
                                gltf.meta.accessors.get(indices_acc_index as usize)?;
                            let indices_view_index = indices_acc.buffer_view?;
                            let index_format = match indices_acc.component_type {
                                models::schema::ComponentType::UnsignedShort => {
                                    wgpu::IndexFormat::Uint16
                                }
                                models::schema::ComponentType::UnsignedInt => {
                                    wgpu::IndexFormat::Uint32
                                }
                                _ => {
                                    return None;
                                }
                            };
                            let index_count = indices_acc.count;

                            let vpos_acc = gltf.meta.accessors.get(vpos_acc_index as usize)?;
                            if vpos_acc.component_type != models::schema::ComponentType::Float {
                                return None;
                            }
                            let vpos_view_index = vpos_acc.buffer_view?;

                            let vnormal_acc =
                                gltf.meta.accessors.get(vnormal_acc_index as usize)?;
                            let vnormal_view_index = vnormal_acc.buffer_view?;

                            let index_buffer = {
                                let view =
                                    gltf.meta.buffer_views.get(indices_view_index as usize)?;
                                let buffer = gltf.meta.buffers.get(view.buffer as usize)?;
                                if buffer.uri.is_some() {
                                    // buffer is external, ignore
                                    return None;
                                }

                                let cmp_len = match index_format {
                                    wgpu::IndexFormat::Uint16 => 2,
                                    wgpu::IndexFormat::Uint32 => 4,
                                };
                                let start = (view.byte_offset
                                    + indices_acc.byte_offset.unwrap_or(0))
                                    as usize;
                                let end = start + (indices_acc.count * cmp_len) as usize;

                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(
                                        &[
                                            Some("indices".into()),
                                            gltf_mesh.name.as_deref(),
                                            indices_acc.name.as_deref(),
                                            view.name.as_deref(),
                                        ]
                                        .into_iter()
                                        .flatten()
                                        .collect::<Vec<&str>>()
                                        .join("_"),
                                    ),
                                    contents: &gltf.bin[start..end],
                                    usage: wgpu::BufferUsages::INDEX,
                                })
                            };

                            let vertex_buffer = {
                                let view = gltf.meta.buffer_views.get(vpos_view_index as usize)?;
                                let buffer = gltf.meta.buffers.get(view.buffer as usize)?;
                                if buffer.uri.is_some() {
                                    // buffer is external, ignore
                                    return None;
                                }

                                let cmp_len = 12;
                                let start =
                                    (view.byte_offset + vpos_acc.byte_offset.unwrap_or(0)) as usize;
                                let end = start + (vpos_acc.count * cmp_len) as usize;

                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(
                                        &[
                                            Some("vertices".into()),
                                            gltf_mesh.name.as_deref(),
                                            vpos_acc.name.as_deref(),
                                            view.name.as_deref(),
                                        ]
                                        .into_iter()
                                        .flatten()
                                        .collect::<Vec<&str>>()
                                        .join("_"),
                                    ),
                                    contents: &gltf.bin[start..end],
                                    usage: wgpu::BufferUsages::VERTEX,
                                })
                            };

                            let normal_buffer = {
                                let view =
                                    gltf.meta.buffer_views.get(vnormal_view_index as usize)?;
                                let buffer = gltf.meta.buffers.get(view.buffer as usize)?;
                                if buffer.uri.is_some() {
                                    // buffer is external, ignore
                                    return None;
                                }

                                let cmp_len = 12;
                                let start = (view.byte_offset
                                    + vnormal_acc.byte_offset.unwrap_or(0))
                                    as usize;
                                let end = start + (vnormal_acc.count * cmp_len) as usize;

                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(
                                        &[
                                            Some("normals".into()),
                                            gltf_mesh.name.as_deref(),
                                            vnormal_acc.name.as_deref(),
                                            view.name.as_deref(),
                                        ]
                                        .into_iter()
                                        .flatten()
                                        .collect::<Vec<&str>>()
                                        .join("_"),
                                    ),
                                    contents: &gltf.bin[start..end],
                                    usage: wgpu::BufferUsages::VERTEX,
                                })
                            };

                            Some(Primitive {
                                index_buffer,
                                index_format,
                                index_count,
                                vertex_buffer,
                                normal_buffer,
                            })
                        })
                        .collect::<Vec<Primitive>>();
                    mesh_id_map.insert(gltf_mesh_id, meshes.len());
                    meshes.push(Mesh { primitives });
                }
            }

            (meshes, nodes)
        };

        let bg_layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model bg layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(256),
                },
                count: None,
            }],
        });
        let bg_layout_camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bg layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraDataBuffer>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test pipeline layout"),
            bind_group_layouts: &[&bg_layout_camera, &bg_layout_model],
            push_constant_ranges: &[],
        });

        let pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("model test"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/model_test.wgsl").into(),
                ),
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 1,
                            }],
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };

        let buffer_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera data uniform buffer"),
            contents: bytemuck::cast_slice(&[CameraDataBuffer::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_model = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera data uniform buffer"),
            contents: bytemuck::cast_slice(
                &nodes
                    .iter()
                    .map(|n| ModelDataBuffer {
                        matrix: n.transform.to_cols_array_2d(),
                        normal_matrix: glam::Mat3::from_mat4(n.transform.inverse())
                            .transpose()
                            .to_cols_array_2d()
                            .map(|v| [v[0], v[1], v[2], 0.0]),
                        _pad: [0.0; 36],
                    })
                    .collect::<Vec<ModelDataBuffer>>(),
            ),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bg_layout_camera,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_camera.as_entire_binding(),
            }],
        });
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model bind group"),
            layout: &bg_layout_model,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_model.as_entire_binding(),
            }],
        });

        let view_depth = Self::create_screen_resource(&window, device);

        Ok(Self {
            gltf,
            meshes,
            nodes,
            pipeline,
            bg_camera,
            bg_model,
            buffer_camera,
            buffer_model,
            view_depth,
        })
    }

    pub fn frame<'a>(&mut self, ctx: &'a mut RendererCtx) {
        let mut camera_data = CameraDataBuffer::default();
        camera_data.update(&ctx.engine.camera);
        ctx.queue
            .write_buffer(&self.buffer_camera, 0, bytemuck::cast_slice(&[camera_data]));

        let surface_texture = ctx.surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(ctx.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("post fx"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.view_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg_camera, &[]);

            for (i, node) in self.nodes.iter().enumerate() {
                let Some(mesh) = self.meshes.get(node.mesh_id) else {
                    continue;
                };
                pass.set_bind_group(1, &self.bg_model, &[256 * i as u32]);
                for primitive in &mesh.primitives {
                    pass.set_index_buffer(primitive.index_buffer.slice(..), primitive.index_format);
                    pass.set_vertex_buffer(0, primitive.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, primitive.normal_buffer.slice(..));
                    pass.draw_indexed(0..primitive.index_count, 0, 0..1);
                }
            }
        }
        ctx.queue.submit([encoder.finish()]);

        ctx.window.pre_present_notify();
        surface_texture.present();
    }

    pub fn handle_resize(&mut self, window: &Window, device: &wgpu::Device) {
        self.view_depth = Self::create_screen_resource(window, device);
    }
}

impl ModelLoader {
    fn create_screen_resource(window: &Window, device: &wgpu::Device) -> wgpu::TextureView {
        let size = window.size();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

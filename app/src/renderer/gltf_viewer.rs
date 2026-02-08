use std::{
    io::{BufReader, Cursor},
    sync::Arc,
};

use models::{Gltf, Scene};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{
    SizedWindow,
    engine::Engine,
    renderer::{RendererCtx, buffers::CameraDataBuffer},
};

#[derive(Debug)]
pub struct ModelViewer {
    scene: Scene,
    pipeline: wgpu::RenderPipeline,
    bg_camera: wgpu::BindGroup,
    bg_model: wgpu::BindGroup,
    bg_scene_textures: wgpu::BindGroup,
    mesh_buffers: Vec<MeshBuffers>,
    buffer_camera: wgpu::Buffer,
    view_depth: wgpu::TextureView,
}

impl ModelViewer {
    pub fn new(
        window: Arc<Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        engine: &Engine,
    ) -> anyhow::Result<Self> {
        let src = std::include_bytes!("../../assets/sponza.glb");
        let mut src = BufReader::new(Cursor::new(src));

        let gltf = Gltf::parse(&mut src)?;
        let scene = Scene::from_gltf(&gltf)?;

        let views: Vec<wgpu::TextureView> = scene
            .textures
            .iter()
            .map(|tex| {
                let (width, height) = tex.dimensions();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    ..Default::default()
                });

                view
            })
            .collect();

        let mesh_buffers: Vec<MeshBuffers> = scene
            .meshes
            .iter()
            .map(|mesh| {
                let primitive_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive data"),
                        contents: bytemuck::cast_slice(
                            &mesh
                                .primitives
                                .iter()
                                .map(|primitive| PrimitiveData {
                                    min: primitive.min.to_array(),
                                    _pad: 0.0,
                                    max: primitive.max.to_array(),
                                    material_id: primitive.material_id,
                                })
                                .collect::<Vec<PrimitiveData>>(),
                        ),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let primitives = mesh
                    .primitives
                    .iter()
                    .map(|p| {
                        let index_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("index buffer"),
                                contents: &gltf.bin[p.indices.start..p.indices.end],
                                usage: wgpu::BufferUsages::INDEX,
                            });

                        let position_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex position buffer"),
                                contents: &gltf.bin[p.positions.start..p.positions.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let normal_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex normals buffer"),
                                contents: &gltf.bin[p.normals.start..p.normals.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let tangent_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex tangent buffer"),
                                contents: &gltf.bin[p.tangents.start..p.tangents.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let uv_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("tex coord buffer"),
                                contents: &gltf.bin[p.uv.start..p.uv.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        PrimitiveBuffers {
                            index_buffer,
                            position_buffer,
                            normal_buffer,
                            tangent_buffer,
                            uv_buffer,
                            index_format: match p.indices.component_type {
                                models::schema::ComponentType::UnsignedShort => {
                                    wgpu::IndexFormat::Uint16
                                }
                                _ => wgpu::IndexFormat::Uint32,
                            },
                            index_count: p.indices.count,
                        }
                    })
                    .collect();

                MeshBuffers {
                    primitive_buffer,
                    primitives,
                }
            })
            .collect();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

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
        let bg_layout_scene_textures =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene textures bg layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: std::num::NonZeroU32::new(views.len() as u32),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test pipeline layout"),
            bind_group_layouts: &[
                &bg_layout_camera,
                &bg_layout_model,
                &bg_layout_scene_textures,
            ],
            immediate_size: 0,
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
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 2,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 16,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 3,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 28,
                                shader_location: 4,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 5,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 16,
                                shader_location: 6,
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
                multiview_mask: None,
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
                &scene
                    .nodes
                    .iter()
                    .map(|n| ModelData {
                        matrix: n.transform.to_cols_array_2d(),
                        normal_matrix: glam::Mat3::from_mat4(n.transform.inverse())
                            .transpose()
                            .to_cols_array_2d()
                            .map(|v| [v[0], v[1], v[2], 0.0]),
                        _pad: [0.0; 36],
                    })
                    .collect::<Vec<ModelData>>(),
            ),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(
                &scene
                    .materials
                    .iter()
                    .map(|mat| MaterialData {
                        base_albedo: mat.base_albedo.to_array(),
                        metallic: mat.metallic,
                        roughness: mat.roughness,
                        normal_scale: mat.normal_scale,
                        albedo_index: mat.albedo_index,
                        normal_index: mat.normal_index,
                        _pad: [0.0, 0.0, 0.0],
                    })
                    .collect::<Vec<MaterialData>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let bg_scene_textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene textures bind group"),
            layout: &bg_layout_scene_textures,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(
                        &views.iter().collect::<Vec<&wgpu::TextureView>>(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_materials.as_entire_binding(),
                },
            ],
        });

        let view_depth = Self::create_screen_resource(&window, device);

        Ok(Self {
            scene,
            mesh_buffers,
            pipeline,
            bg_camera,
            bg_model,
            bg_scene_textures,
            buffer_camera,
            view_depth,
        })
    }

    pub fn frame<'a>(&mut self, ctx: &'a mut RendererCtx) {
        let mut camera_data = CameraDataBuffer::default();
        camera_data.update(&ctx.engine.camera);
        ctx.queue
            .write_buffer(&self.buffer_camera, 0, bytemuck::cast_slice(&[camera_data]));

        let surface_texture = ctx.surface.get_current_texture().unwrap();
        let surface_view = surface_texture
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
                    view: &surface_view,
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
                multiview_mask: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg_camera, &[]);
            pass.set_bind_group(2, &self.bg_scene_textures, &[]);

            for (i, node) in self.scene.nodes.iter().enumerate() {
                let Some(mesh_buffers) = self.mesh_buffers.get(node.mesh_id) else {
                    continue;
                };
                pass.set_bind_group(1, &self.bg_model, &[256 * i as u32]);
                pass.set_vertex_buffer(4, mesh_buffers.primitive_buffer.slice(..));
                pass.set_vertex_buffer(5, mesh_buffers.primitive_buffer.slice(..));
                pass.set_vertex_buffer(6, mesh_buffers.primitive_buffer.slice(..));
                for (j, primitive) in mesh_buffers.primitives.iter().enumerate() {
                    pass.set_index_buffer(primitive.index_buffer.slice(..), primitive.index_format);
                    pass.set_vertex_buffer(0, primitive.position_buffer.slice(..));
                    pass.set_vertex_buffer(1, primitive.normal_buffer.slice(..));
                    pass.set_vertex_buffer(2, primitive.uv_buffer.slice(..));
                    pass.set_vertex_buffer(3, primitive.tangent_buffer.slice(..));
                    pass.draw_indexed(0..primitive.index_count, 0, (j as u32)..(j as u32 + 1));
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

#[derive(Debug)]
struct MeshBuffers {
    primitive_buffer: wgpu::Buffer,
    primitives: Vec<PrimitiveBuffers>,
}
#[derive(Debug)]
struct PrimitiveBuffers {
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: u32,
    position_buffer: wgpu::Buffer,
    normal_buffer: wgpu::Buffer,
    tangent_buffer: wgpu::Buffer,
    uv_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelData {
    matrix: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 3],
    _pad: [f32; 36],
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrimitiveData {
    min: [f32; 3],
    _pad: f32,
    max: [f32; 3],
    material_id: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData {
    base_albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
    _pad: [f32; 3],
}

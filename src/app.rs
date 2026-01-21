use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, UVec2, Vec3, uvec2, vec3};
use wgpu::{
    BindGroup, Buffer, Device, Queue, RenderPipeline, Sampler, Surface, Texture, TextureFormat,
    TextureView, util::DeviceExt,
};
use winit::window::Window;

use crate::mesh::{Cube, IntoMesh, Mesh};

#[derive(Debug)]
pub struct App {
    pub window: Arc<Window>,
    device: Device,
    queue: Queue,
    size: UVec2,
    surface: Surface<'static>,
    format: TextureFormat,
    depth: DepthTexture,
    pipeline: RenderPipeline,
    camera_bind_group: BindGroup,
    camera_uniform: Buffer,
    model_bind_group: BindGroup,
    model_uniform: Buffer,
    model: Model,
    cube: Mesh,
}

impl App {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = uvec2(size.width, size.height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let surface = instance.create_surface(window.clone()).unwrap();
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities.formats[0];

        let cube = Cube::new().mesh(&device);
        let model = Model::new();

        let depth = DepthTexture::new(&device, size);

        let (camera_uniform, camera_bind_group, camera_bind_group_layout) = {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });

            let uniform_buffer = {
                let uniform = CameraUniform::new(size);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("camera uniform buffer"),
                    contents: bytemuck::cast_slice(&[uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            (uniform_buffer, bind_group, layout)
        };

        let (model_uniform, model_bind_group, model_bind_group_layout) = {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("model bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });

            let uniform_buffer = {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("model uniform buffer"),
                    contents: bytemuck::cast_slice(&[model.uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            (uniform_buffer, bind_group, layout)
        };

        let pipeline = {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main pipeline layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("main shader"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("base.wgsl").into()),
            });

            let vertex_buffers = [wgpu::VertexBufferLayout {
                array_stride: crate::mesh::VERTEX_SIZE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 4 * 4,
                        shader_location: 1,
                    },
                ],
            }];

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
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
                    format: DepthTexture::FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            render_pipeline
        };

        let _self = Self {
            window,
            device,
            queue,
            size,
            surface,
            format,
            depth,
            pipeline,
            cube,
            model,
            camera_bind_group,
            camera_uniform,
            model_bind_group,
            model_uniform,
        };

        _self.configure_surface();

        return _self;
    }

    pub fn render(&mut self) {
        // update model
        {
            self.model.rotation += 0.005 * Vec3::ONE;
            self.model.update();
            self.queue.write_buffer(
                &self.model_uniform,
                0,
                bytemuck::cast_slice(&[self.model.uniform]),
            );
        }

        let surface_texture = self.surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // main pass
        {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("main"),
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
                    view: &self.depth.view,
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

            pass.push_debug_group("prepare data for draw");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.model_bind_group, &[]);
            pass.set_index_buffer(self.cube.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.cube.vertex_buffer.slice(..));
            pass.pop_debug_group();

            pass.insert_debug_marker("draw cube");
            pass.draw_indexed(0..self.cube.index_count, 0, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();
    }

    pub fn on_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.size = uvec2(size.width, size.height);
        self.configure_surface();

        self.depth = DepthTexture::new(&self.device, self.size);

        // update camera uniform
        let uniform = CameraUniform::new(self.size);
        self.queue
            .write_buffer(&self.camera_uniform, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn configure_surface(&self) {
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                view_formats: vec![self.format.add_srgb_suffix()],
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                width: self.size.x,
                height: self.size.y,
                desired_maximum_frame_latency: 2,
                present_mode: wgpu::PresentMode::AutoVsync,
            },
        );
    }
}

#[derive(Debug)]
struct DepthTexture {
    view: TextureView,
    _texture: Texture,
    _sampler: Sampler,
}
impl DepthTexture {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn new(device: &Device, size: UVec2) -> Self {
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
            format: DepthTexture::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            view,
            _texture: texture,
            _sampler: sampler,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new(size: UVec2) -> Self {
        const EYE: Vec3 = vec3(1.5, -5.0, 3.0);
        const CENTER: Vec3 = Vec3::ZERO;
        const UP: Vec3 = Vec3::Z;

        let view = Mat4::look_at_rh(EYE, CENTER, UP);
        let projection = Mat4::perspective_rh(45.0, size.x as f32 / size.y as f32, 0.01, 100.0);
        let view_proj_matrix = projection * view;

        Self {
            view_proj_matrix: view_proj_matrix.to_cols_array_2d(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ModelUniform {
    matrix: [[f32; 4]; 4],
}

#[derive(Debug, Clone)]
struct Model {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    matrix: Mat4,
    uniform: ModelUniform,
}

impl Model {
    fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            matrix: Mat4::IDENTITY,
            uniform: ModelUniform {
                matrix: Mat4::IDENTITY.to_cols_array_2d(),
            },
        }
    }

    fn update(&mut self) {
        self.matrix = Mat4::from_scale_rotation_translation(
            self.scale,
            Quat::from_euler(
                EulerRot::XYZ,
                self.rotation.x,
                self.rotation.y,
                self.rotation.z,
            ),
            self.position,
        );
        self.uniform.matrix = self.matrix.to_cols_array_2d();
    }
}

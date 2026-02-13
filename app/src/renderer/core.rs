use std::{
    f32,
    io::{BufReader, Cursor},
    sync::Arc,
    time::Duration,
};

use wgpu::util::DeviceExt;

use crate::{
    SizedWindow,
    engine::Engine,
    renderer::{
        buffers::{self, CameraDataBuffer, EnvironmentDataBuffer, ModelDataBuffer},
        loader::Voxelizer,
        quad::Quad,
    },
    ui::{Ui, UiCtx},
};

pub struct RendererCtx<'a> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface: &'a wgpu::Surface<'static>,
    pub format: &'a wgpu::TextureFormat,
    pub engine: &'a Engine,
    pub ui: &'a mut Ui,
}

pub struct Renderer {
    pipeline_layouts: PipelineLayouts,
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
    bind_groups: BindGroups,
    textures: Textures,
    samplers: Samplers,
    buffers: Buffers,
    timing: Option<RenderTimer>,
    quad: Quad,
    sun_direction: glam::Vec3,
}

struct PipelineLayouts {
    raymarch: wgpu::PipelineLayout,
    deferred: wgpu::PipelineLayout,
    fx: wgpu::PipelineLayout,
}

struct Pipelines {
    raymarch: wgpu::ComputePipeline,
    deferred: wgpu::ComputePipeline,
    fx: wgpu::RenderPipeline,
}

struct BindGroupLayouts {
    raymarch_gbuffer: wgpu::BindGroupLayout,
    raymarch_per_frame: wgpu::BindGroupLayout,
    raymarch_voxels: wgpu::BindGroupLayout,
    deferred_gbuffer: wgpu::BindGroupLayout,
    fx_gbuffer: wgpu::BindGroupLayout,
}

struct BindGroups {
    raymarch_gbuffer: Option<wgpu::BindGroup>,
    raymarch_per_frame: wgpu::BindGroup,
    raymarch_voxels: wgpu::BindGroup,
    deferred_gbuffer: Option<wgpu::BindGroup>,
    fx_gbuffer: Option<wgpu::BindGroup>,
}

struct Textures {
    gbuffer_albedo: Option<wgpu::Texture>,
    gbuffer_normal: Option<wgpu::Texture>,
    gbuffer_depth: Option<wgpu::Texture>,
    gbuffer_out_color: Option<wgpu::Texture>,
    voxel_brickmap: wgpu::Texture,
}

struct Samplers {
    linear: wgpu::Sampler,
}

struct Buffers {
    voxel_scene_metadata: wgpu::Buffer,
    voxel_palette: wgpu::Buffer,
    voxel_chunk_indices: wgpu::Buffer,
    voxel_chunks: wgpu::Buffer,
    camera: wgpu::Buffer,
    environment: wgpu::Buffer,
    model: wgpu::Buffer,
}

// struct Uniforms {
//     scene: Buffer,
//     voxel_chunk_index: Buffer,
//     voxel_count: u32,
//     allocated_chunks: u32,
//     allocated_brick_slices: u32,
//     size_chunks: glam::UVec3,
//     voxel_chunks: Buffer,
//     voxels: Texture,
//     voxel_palette: Buffer,
//     camera: Buffer,
//     camera_data: CameraDataBuffer,
//     environment: Buffer,
//     model: Buffer,
//     model_data: ModelDataBuffer,
// }

impl Renderer {
    pub fn new(
        window: Arc<winit::window::Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        engine: &Engine,
        ui: &mut Ui,
    ) -> Self {
        let bg_layouts = BindGroupLayouts {
            raymarch_gbuffer: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_gbuffer"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }),
            raymarch_voxels: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_voxels"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            }),
            raymarch_per_frame: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_per_frame"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
            deferred_gbuffer: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred_gbuffer"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
            fx_gbuffer: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fx_gbuffer"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
        };

        let pipeline_layouts = PipelineLayouts {
            raymarch: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raymarch"),
                bind_group_layouts: &[
                    &bg_layouts.raymarch_gbuffer,
                    &bg_layouts.raymarch_voxels,
                    &bg_layouts.raymarch_per_frame,
                ],
                immediate_size: 0,
            }),
            deferred: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("deferred"),
                bind_group_layouts: &[&bg_layouts.deferred_gbuffer],
                immediate_size: 0,
            }),
            fx: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fx"),
                bind_group_layouts: &[&bg_layouts.fx_gbuffer],
                immediate_size: 0,
            }),
        };

        struct Shaders {
            raymarch: wgpu::ShaderModule,
            deferred: wgpu::ShaderModule,
            fx: wgpu::ShaderModule,
        }
        let shaders = Shaders {
            raymarch: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raymarch"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/raymarch.wgsl").into(),
                ),
            }),
            deferred: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("deferred"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/deferred.wgsl").into(),
                ),
            }),
            fx: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post processing shader"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/fx.wgsl").into()),
            }),
        };

        let pipelines = Pipelines {
            raymarch: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("raymarch"),
                layout: Some(&pipeline_layouts.raymarch),
                module: &shaders.raymarch,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            deferred: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("deferred"),
                layout: Some(&pipeline_layouts.deferred),
                module: &shaders.deferred,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            fx: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post fx pipeline"),
                layout: Some(&pipeline_layouts.fx),
                vertex: wgpu::VertexState {
                    module: &shaders.fx,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: crate::renderer::quad::VERTEX_SIZE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.fx,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                cache: None,
                multiview_mask: None,
            }),
        };

        let scene = {
            // let src = std::include_bytes!("../../assets/san_miguel.glb");
            // let src = std::include_bytes!("../../assets/bistro.glb");
            let src = std::include_bytes!("../../assets/sponza.glb");
            let mut src = BufReader::new(Cursor::new(src));
            Voxelizer::load_gltf(&mut src, device, queue).unwrap()
        };

        let textures = Textures {
            gbuffer_albedo: None,
            gbuffer_normal: None,
            gbuffer_depth: None,
            gbuffer_out_color: None,
            voxel_brickmap: scene.tex_brickmap,
        };

        let samplers = Samplers {
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
        };

        let buffers = Buffers {
            voxel_scene_metadata: {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct BufferVoxelSceneMetadata {
                    size_chunks: glam::UVec3,
                    _pad: u32,
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel_scene_metadata"),
                    contents: bytemuck::cast_slice(&[BufferVoxelSceneMetadata {
                        size_chunks: scene.size_chunks,
                        _pad: 0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            },
            voxel_palette: scene.buffer_palette,
            voxel_chunk_indices: scene.buffer_chunk_indices,
            voxel_chunks: scene.buffer_chunks,
            camera: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera"),
                size: std::mem::size_of::<buffers::CameraDataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            environment: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("environment"),
                size: std::mem::size_of::<buffers::EnvironmentDataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            model: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("model"),
                size: std::mem::size_of::<buffers::ModelDataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let bind_groups = BindGroups {
            raymarch_gbuffer: None,
            raymarch_voxels: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_voxels"),
                layout: &bg_layouts.raymarch_voxels,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.voxel_scene_metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.voxel_palette.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.voxel_chunk_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffers.voxel_chunks,
                            offset: 0,
                            size: std::num::NonZeroU64::new((scene.allocated_chunks * 64) as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &textures
                                .voxel_brickmap
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("voxel_brickmap"),
                                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                ],
            }),
            raymarch_per_frame: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_per_frame"),
                layout: &bg_layouts.raymarch_per_frame,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.environment.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.model.as_entire_binding(),
                    },
                ],
            }),
            deferred_gbuffer: None,
            fx_gbuffer: None,
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device));

        let quad = Quad::new(device);

        ui.state.sun_azimuth = -2.5;
        ui.state.sun_altitude = 1.3;
        ui.state.shadow_bias = 0.001;

        let mut _self = Self {
            pipeline_layouts,
            pipelines,
            bg_layouts,
            bind_groups,
            textures,
            samplers,
            buffers,
            timing,
            quad,
            sun_direction: Default::default(),
        };

        _self.update_screen_resources(&window, device);

        return _self;
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.update_screen_resources(window, device);
    }

    fn update_screen_resources(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        let size = window.size();

        let tex_albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("albedo texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        });
        let view_albedo = tex_albedo.create_view(&wgpu::TextureViewDescriptor {
            label: Some("albedo texture view"),
            ..Default::default()
        });
        self.textures.gbuffer_albedo = Some(tex_albedo);

        let tex_normal = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        });
        let view_normal = tex_normal.create_view(&wgpu::TextureViewDescriptor {
            label: Some("normal texture storage view"),
            ..Default::default()
        });
        self.textures.gbuffer_normal = Some(tex_normal);

        let tex_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            format: wgpu::TextureFormat::R32Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        });
        let view_depth = tex_depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("depth texture view"),
            ..Default::default()
        });
        self.textures.gbuffer_depth = Some(tex_depth);

        let tex_out_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deferred output texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        });
        let view_out_color = tex_out_color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("deferred output texture view"),
            ..Default::default()
        });
        self.textures.gbuffer_out_color = Some(tex_out_color);

        self.bind_groups.raymarch_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_gbuffer"),
                layout: &self.bg_layouts.raymarch_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_depth),
                    },
                ],
            }));

        self.bind_groups.deferred_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_gbuffer"),
                layout: &self.bg_layouts.deferred_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_out_color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                    },
                ],
            }));

        self.bind_groups.fx_gbuffer = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post fx bind group"),
            layout: &self.bg_layouts.fx_gbuffer,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_out_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                },
            ],
        }));
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut RendererCtx) {
        let size = ctx.window.size();

        // update uniform buffers
        {
            let mut camera_data = CameraDataBuffer::default();
            camera_data.update(&ctx.engine.camera);
            ctx.queue.write_buffer(
                &self.buffers.camera,
                0,
                bytemuck::cast_slice(&[camera_data]),
            );

            self.sun_direction = glam::vec3(
                ctx.ui.state.sun_altitude.cos() * ctx.ui.state.sun_azimuth.cos(),
                ctx.ui.state.sun_altitude.cos() * ctx.ui.state.sun_azimuth.sin(),
                ctx.ui.state.sun_altitude.sin(),
            )
            .normalize();
            ctx.queue.write_buffer(
                &self.buffers.environment,
                0,
                bytemuck::cast_slice(&[EnvironmentDataBuffer {
                    sun_direction: self.sun_direction,
                    shadow_bias: ctx.ui.state.shadow_bias,
                }]),
            );

            let mut model_data = ModelDataBuffer::default();
            model_data.update(&ctx.engine.model);
            ctx.queue
                .write_buffer(&self.buffers.model, 0, bytemuck::cast_slice(&[model_data]));
        }

        let surface_texture = ctx.surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(ctx.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        // raymarch pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("raymarch"),
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.raymarch);
            pass.set_bind_group(0, &self.bind_groups.raymarch_gbuffer, &[]);
            pass.set_bind_group(1, &self.bind_groups.raymarch_voxels, &[]);
            pass.set_bind_group(2, &self.bind_groups.raymarch_per_frame, &[]);

            pass.insert_debug_marker("raymarch");
            pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
        }

        // deferred pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("deferred"),
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.deferred);
            pass.set_bind_group(0, &self.bind_groups.deferred_gbuffer, &[]);

            pass.insert_debug_marker("deferred");
            pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
        }

        // post fx pass
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
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::RenderPassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }
                }),
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.fx);
            pass.set_bind_group(0, &self.bind_groups.fx_gbuffer, &[]);
            pass.set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));

            pass.insert_debug_marker("fx");
            pass.draw_indexed(0..self.quad.index_count, 0, 0..1);
        }

        if let Some(timing) = &self.timing {
            encoder.resolve_query_set(
                &timing.query_set,
                0..(2 * RenderTimer::QUERY_PASS_COUNT),
                &timing.resolve_buffer,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &timing.resolve_buffer,
                0,
                &timing.result_buffer,
                0,
                timing.result_buffer.size(),
            );
        }

        // ctx.ui.state.voxel_count = self..voxel_count;
        // ctx.ui.state.scene_size = self.uniforms.voxels.dimension();
        ctx.ui.frame(&mut UiCtx {
            window: ctx.window,
            device: ctx.device,
            queue: ctx.queue,
            texture_view: &texture_view,
            encoder: &mut encoder,
        });

        ctx.queue.submit([encoder.finish()]);

        ctx.window.pre_present_notify();
        surface_texture.present();

        // if let Some(timing) = &self.timing {
        //     timing
        //         .result_buffer
        //         .slice(..)
        //         .map_async(wgpu::MapMode::Read, |_| ());

        //     ctx.device
        //         .poll(wgpu::PollType::Wait {
        //             submission_index: None,
        //             timeout: Some(Duration::from_secs(5)),
        //         })
        //         .unwrap();

        //     let view = timing.result_buffer.get_mapped_range(..);
        //     let timestamps: &[u64] = bytemuck::cast_slice(&*view);

        //     let time_raymarch = Duration::from_nanos(timestamps[1] - timestamps[0]);
        //     let time_deferred = Duration::from_nanos(timestamps[3] - timestamps[2]);
        //     let time_post_fx = Duration::from_nanos(timestamps[5] - timestamps[4]);

        //     ctx.ui.state.pass_avg = vec![
        //         ("Raymarch".into(), time_raymarch),
        //         ("Deferred".into(), time_deferred),
        //         ("Post FX".into(), time_post_fx),
        //     ];

        //     drop(view);
        //     timing.result_buffer.unmap();
        // }
    }
}

struct RenderTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    result_buffer: Arc<wgpu::Buffer>,
}
impl RenderTimer {
    const QUERY_PASS_COUNT: u32 = 3;

    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp query set"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::QUERY_PASS_COUNT * 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query resolve buffer"),
            size: Self::QUERY_PASS_COUNT as u64 * 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query result buffer"),
            size: Self::QUERY_PASS_COUNT as u64 * 2 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            result_buffer: Arc::new(result_buffer),
        }
    }
}

use std::{sync::Arc, time::Duration};

use wgpu::{ShaderStages, util::DeviceExt};

use crate::{
    SizedWindow,
    config::Config,
    define_shaders,
    engine::Engine,
    lightmap::LIGHTMAPS,
    models::MODELS,
    renderer::{
        buffers::{self, CameraDataBuffer, EnvironmentDataBuffer, ModelDataBuffer},
        noise,
        quad::Quad,
        timing::{RenderTimer, TimedEncoder},
        utils::{
            DeviceSwapExt, PassSwapExt, SwapchainBindGroup, SwapchainBindGroupDescriptor,
            SwapchainBindGroupEntry, SwapchainBindingResource, SwapchainTexture,
        },
    },
    ui::{Ui, UiCtx},
    utils::{
        layout::{DeviceUtils, *},
        pipeline::PipelineUtils,
    },
};

const ATROUS_PASS_COUNT: usize = 3;

define_shaders! {
    raymarch "../shaders/raymarch.wgsl",
    shadow "../shaders/shadow.wgsl",
    ambient "../shaders/ambient.wgsl",
    specular "../shaders/specular.wgsl",
    lighting_resolve "../shaders/lighting_resolve.wgsl",
    atrous "../shaders/atrous.wgsl",
    deferred "../shaders/deferred.wgsl",
    taa "../shaders/taa.wgsl",
    fx "../shaders/fx.wgsl",
}

pub struct RendererCtx<'a> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface: &'a wgpu::Surface<'static>,
    pub format: &'a wgpu::TextureFormat,
    pub engine: &'a mut Engine,
    pub ui: &'a mut Ui,
    pub config: &'a mut Config,
}

pub struct Renderer {
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
    bind_groups: BindGroups,
    textures: Textures,
    samplers: Samplers,
    buffers: Buffers,
    timing: Option<RenderTimer>,
    quad: Quad,
    frame_id: u32,
    size: glam::UVec2,
    size_shadow: glam::UVec2,
    render_scale: f32,
    sun_direction: glam::Vec3,
    prev_camera: Option<CameraDataBuffer>,
}

struct Pipelines {
    raymarch: wgpu::ComputePipeline,
    // shadow: wgpu::ComputePipeline,
    // ambient: wgpu::ComputePipeline,
    // specular: wgpu::ComputePipeline,
    lighting_resolve: wgpu::ComputePipeline,
    // atrous: wgpu::ComputePipeline,
    deferred: wgpu::ComputePipeline,
    taa: wgpu::ComputePipeline,
    fx: wgpu::RenderPipeline,
}

struct BindGroupLayouts {
    per_frame_shared: wgpu::BindGroupLayout,
    raymarch_gbuffer: wgpu::BindGroupLayout,
    raymarch_swap: wgpu::BindGroupLayout,
    raymarch_static: wgpu::BindGroupLayout,
    shadow_gbuffer: wgpu::BindGroupLayout,
    ambient_gbuffer: wgpu::BindGroupLayout,
    ambient_swap: wgpu::BindGroupLayout,
    ambient_static: wgpu::BindGroupLayout,
    specular_gbuffer: wgpu::BindGroupLayout,
    lighting_resolve_gbuffer: wgpu::BindGroupLayout,
    lighting_resolve_swap: wgpu::BindGroupLayout,
    atrous_per_pass: wgpu::BindGroupLayout,
    atrous_swap: wgpu::BindGroupLayout,
    deferred_gbuffer: wgpu::BindGroupLayout,
    deferred_swap: wgpu::BindGroupLayout,
    deferred_static: wgpu::BindGroupLayout,
    taa_input: wgpu::BindGroupLayout,
    taa_output: wgpu::BindGroupLayout,
    fx_input: wgpu::BindGroupLayout,
}

struct BindGroups {
    per_frame_shared: wgpu::BindGroup,
    raymarch_gbuffer: Option<wgpu::BindGroup>,
    raymarch_swap: Option<SwapchainBindGroup>,
    raymarch_static: wgpu::BindGroup,
    shadow_gbuffer: Option<wgpu::BindGroup>,
    ambient_gbuffer: Option<wgpu::BindGroup>,
    ambient_swap: Option<SwapchainBindGroup>,
    // ambient_static: wgpu::BindGroup,
    specular_gbuffer: Option<wgpu::BindGroup>,
    lighting_resolve_gbuffer: Option<wgpu::BindGroup>,
    lighting_resolve_swap: Option<SwapchainBindGroup>,
    atrous_per_pass_primary: Option<SwapchainBindGroup>,
    atrous_per_pass_secondary: Option<SwapchainBindGroup>,
    atrous_per_pass: Option<[wgpu::BindGroup; ATROUS_PASS_COUNT - 2]>,
    atrous_swap: Option<SwapchainBindGroup>,
    deferred_gbuffer: Option<wgpu::BindGroup>,
    deferred_swap: Option<SwapchainBindGroup>,
    deferred_static: wgpu::BindGroup,
    taa_input: Option<wgpu::BindGroup>,
    taa_swap: Option<SwapchainBindGroup>,
    fx_input_swap: Option<SwapchainBindGroup>,
}

struct Textures {
    gbuffer_albedo: Option<wgpu::Texture>,
    gbuffer_normal: Option<SwapchainTexture>,
    gbuffer_depth: Option<SwapchainTexture>,
    gbuffer_velocity: Option<wgpu::Texture>,
    gbuffer_shadow: Option<wgpu::Texture>,
    gbuffer_illumination: Option<wgpu::Texture>,
    gbuffer_acc_illumination: Option<SwapchainTexture>,
    gbuffer_acc_illumination_moments: Option<SwapchainTexture>,
    gbuffer_acc_illumination_history_len: Option<SwapchainTexture>,
    gbuffer_filter_result: Option<SwapchainTexture>,
    gbuffer_specular: Option<wgpu::Texture>,
    deferred_output: Option<wgpu::Texture>,
    out_color: Option<SwapchainTexture>,
    noise_cos_hemisphere_gauss: wgpu::Texture,
    noise_uniform_gauss: wgpu::Texture,
}

struct Samplers {
    linear: wgpu::Sampler,
    nearest_repeat: wgpu::Sampler,
}

struct Buffers {
    voxel_scene_metadata: wgpu::Buffer,
    voxel_palette: wgpu::Buffer,
    voxel_index_chunks: wgpu::Buffer,
    voxel_leaf_chunks: wgpu::Buffer,
    frame_metadata: wgpu::Buffer,
    environment: wgpu::Buffer,
    model: wgpu::Buffer,
}

impl Renderer {
    pub fn new(
        window: Arc<winit::window::Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        config: &mut Config,
        ui: &mut Ui,
    ) -> Self {
        let bg_layouts = BindGroupLayouts {
            per_frame_shared: device.layout(
                "per_frame_shared",
                ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                (uniform_buffer(), uniform_buffer(), uniform_buffer()),
            ),
            raymarch_gbuffer: device.layout(
                "raymarch_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                ),
            ),
            raymarch_swap: device.layout(
                "raymarch_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().write_only(),
                    storage_texture().r32float().dimension_2d().write_only(),
                ),
            ),
            raymarch_static: device.layout(
                "raymarch_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                ),
            ),
            shadow_gbuffer: device.layout(
                "shadow_gbuffer",
                ShaderStages::COMPUTE,
                storage_texture().r32uint().dimension_2d().write_only(),
            ),
            ambient_gbuffer: device.layout(
                "ambient_gbuffer",
                ShaderStages::COMPUTE,
                storage_texture().rgba16float().dimension_2d().write_only(),
            ),
            ambient_swap: device.layout(
                "ambient_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                ),
            ),
            ambient_static: device.layout(
                "ambient_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                ),
            ),
            specular_gbuffer: device.layout(
                "specular_gbuffer",
                ShaderStages::COMPUTE,
                storage_texture().rgba16float().dimension_2d().write_only(),
            ),
            lighting_resolve_gbuffer: device.layout(
                "lighting_resolve_gbuffer",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                ),
            ),
            lighting_resolve_swap: device.layout(
                "lighting_resolve_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().write_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                ),
            ),
            atrous_per_pass: device.layout(
                "atrous_per_pass",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                ),
            ),
            atrous_swap: device.layout(
                "atrous_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                ),
            ),
            deferred_gbuffer: device.layout(
                "deferred_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                ),
            ),
            deferred_swap: device.layout(
                "deferred_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                    texture().float().dimension_2d(),
                    texture().float().dimension_2d(),
                ),
            ),
            deferred_static: device.layout(
                "deferred_static",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    sampler().non_filtering(),
                    texture().unfilterable_float().dimension_2d(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_2d(),
                ),
            ),
            taa_input: device.layout(
                "taa_input",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    texture().float().dimension_2d(),
                ),
            ),
            taa_output: device.layout(
                "taa_output",
                ShaderStages::COMPUTE,
                (
                    texture().float().dimension_2d(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    texture().float().dimension_2d(),
                ),
            ),
            fx_input: device.layout(
                "fx_input",
                ShaderStages::FRAGMENT,
                (texture().float().dimension_2d(), sampler().filtering()),
            ),
        };

        let shaders = device.create_shaders();

        let pipelines = Pipelines {
            raymarch: device
                .compute_pipeline("raymarch", &shaders.raymarch)
                .layout(&[
                    &bg_layouts.raymarch_gbuffer,
                    &bg_layouts.raymarch_swap,
                    &bg_layouts.raymarch_static,
                    &bg_layouts.per_frame_shared,
                ]),
            // shadow: device.compute_pipeline("shadow", &shaders.shadow).layout(&[
            //     &bg_layouts.shadow_gbuffer,
            //     &bg_layouts.ambient_swap,
            //     &bg_layouts.ambient_static,
            //     &bg_layouts.per_frame_shared,
            // ]),
            // ambient: device
            //     .compute_pipeline("ambient", &shaders.ambient)
            //     .layout(&[
            //         &bg_layouts.ambient_gbuffer,
            //         &bg_layouts.ambient_swap,
            //         &bg_layouts.ambient_static,
            //         &bg_layouts.per_frame_shared,
            //     ]),
            // specular: device
            //     .compute_pipeline("specular", &shaders.specular)
            //     .layout(&[
            //         &bg_layouts.specular_gbuffer,
            //         &bg_layouts.ambient_swap,
            //         &bg_layouts.raymarch_static,
            //         &bg_layouts.per_frame_shared,
            //     ]),
            lighting_resolve: device
                .compute_pipeline("lighting_resolve", &shaders.lighting_resolve)
                .layout(&[
                    &bg_layouts.lighting_resolve_gbuffer,
                    &bg_layouts.lighting_resolve_swap,
                    &bg_layouts.per_frame_shared,
                ]),
            // atrous: device.compute_pipeline("atrous", &shaders.atrous).layout(&[
            //     &bg_layouts.atrous_per_pass,
            //     &bg_layouts.atrous_swap,
            //     &bg_layouts.per_frame_shared,
            // ]),
            deferred: device
                .compute_pipeline("deferred", &shaders.deferred)
                .layout(&[
                    &bg_layouts.deferred_gbuffer,
                    &bg_layouts.deferred_swap,
                    &bg_layouts.deferred_static,
                    &bg_layouts.per_frame_shared,
                ]),
            taa: device.compute_pipeline("taa", &shaders.taa).layout(&[
                &bg_layouts.per_frame_shared,
                &bg_layouts.taa_input,
                &bg_layouts.taa_output,
            ]),
            fx: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post fx pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("fx"),
                        bind_group_layouts: &[&bg_layouts.fx_input, &bg_layouts.per_frame_shared],
                        immediate_size: 0,
                    }),
                ),
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

        // see build script
        let scene = MODELS.sponza.load(device, queue).unwrap();
        let ibl = LIGHTMAPS.partly_cloudy.load(device, queue).unwrap();

        let textures = Textures {
            gbuffer_albedo: None,
            gbuffer_normal: None,
            gbuffer_depth: None,
            gbuffer_velocity: None,
            gbuffer_shadow: None,
            gbuffer_illumination: None,
            gbuffer_acc_illumination: None,
            gbuffer_acc_illumination_moments: None,
            gbuffer_acc_illumination_history_len: None,
            gbuffer_filter_result: None,
            gbuffer_specular: None,
            deferred_output: None,
            out_color: None,
            noise_cos_hemisphere_gauss: noise::noise_cos_hemisphere_gauss(device, queue).unwrap(),
            noise_uniform_gauss: noise::noise_uniform_gauss(device, queue).unwrap(),
        };

        let samplers = Samplers {
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            nearest_repeat: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("nearest_repeat"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
        };

        let buffers = Buffers {
            voxel_scene_metadata: {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct BufferVoxelSceneMetadata {
                    bounding_size: u32,
                    index_levels: u32,
                    index_chunk_count: u32,
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel_scene_metadata"),
                    contents: bytemuck::cast_slice(&[BufferVoxelSceneMetadata {
                        bounding_size: scene.meta.bounding_size,
                        index_levels: scene.meta.index_levels,
                        index_chunk_count: scene.meta.allocated_index_chunks,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            },
            voxel_palette: scene.data.buffer_palette,
            voxel_index_chunks: scene.data.buffer_index_chunks,
            voxel_leaf_chunks: scene.data.buffer_leaf_chunks,

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
            frame_metadata: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame_metadata"),
                size: 12,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let bind_groups = BindGroups {
            per_frame_shared: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("per_frame_shared"),
                layout: &bg_layouts.per_frame_shared,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.environment.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.frame_metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.model.as_entire_binding(),
                    },
                ],
            }),
            raymarch_gbuffer: None,
            raymarch_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_static"),
                layout: &bg_layouts.raymarch_static,
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
                        resource: buffers.voxel_index_chunks.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.voxel_leaf_chunks.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &textures.noise_cos_hemisphere_gauss.create_view(
                                &wgpu::TextureViewDescriptor {
                                    ..Default::default()
                                },
                            ),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                ],
            }),
            raymarch_swap: None,
            shadow_gbuffer: None,
            ambient_gbuffer: None,
            ambient_swap: None,
            // ambient_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
            //     label: Some("ambient_static"),
            //     layout: &bg_layouts.ambient_static,
            //     entries: &[
            //         wgpu::BindGroupEntry {
            //             binding: 0,
            //             resource: buffers.voxel_scene_metadata.as_entire_binding(),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 1,
            //             resource: buffers.voxel_palette.as_entire_binding(),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 2,
            //             resource: buffers.index_chunks.as_entire_binding(),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 3,
            //             resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            //                 buffer: &buffers.voxel_chunks,
            //                 offset: 0,
            //                 size: std::num::NonZeroU64::new(
            //                     (scene.meta.allocated_chunks * 64) as u64,
            //                 ),
            //             }),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 4,
            //             resource: wgpu::BindingResource::TextureView(
            //                 &textures.noise_cos_hemisphere_gauss.create_view(
            //                     &wgpu::TextureViewDescriptor {
            //                         ..Default::default()
            //                     },
            //                 ),
            //             ),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 5,
            //             resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
            //         },
            //     ],
            // }),
            specular_gbuffer: None,
            lighting_resolve_gbuffer: None,
            lighting_resolve_swap: None,
            atrous_per_pass_primary: None,
            atrous_per_pass_secondary: None,
            atrous_per_pass: None,
            atrous_swap: None,
            deferred_gbuffer: None,
            deferred_swap: None,
            deferred_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_static"),
                layout: &bg_layouts.deferred_static,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &textures.noise_uniform_gauss.create_view(
                                &wgpu::TextureViewDescriptor {
                                    ..Default::default()
                                },
                            ),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&ibl.cubemap.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("skybox"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&ibl.irradiance.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("irradiance"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&ibl.prefilter.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("prefilter"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&ibl.brdf.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("brdf"),
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                ],
            }),
            taa_input: None,
            taa_swap: None,
            fx_input_swap: None,
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device, 10));

        let quad = Quad::new(device);

        config.sun_azimuth = -2.5;
        config.sun_altitude = 1.3;
        ui.debug.voxel_count = scene.meta.voxel_count;
        ui.debug.scene_size = glam::IVec3::splat(scene.meta.bounding_size as i32);

        let mut _self = Self {
            pipelines,
            bg_layouts,
            bind_groups,
            textures,
            samplers,
            buffers,
            timing,
            quad,
            frame_id: 0,
            render_scale: 1.0,
            size: Default::default(),
            size_shadow: Default::default(),
            sun_direction: Default::default(),
            prev_camera: None,
        };

        _self.update_screen_resources(&window, device);

        return _self;
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.update_screen_resources(window, device);
    }

    fn update_screen_resources(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.size = window
            .size()
            .map(|x| ((x as f32) * self.render_scale).ceil() as u32);
        let size = wgpu::Extent3d {
            width: self.size.x,
            height: self.size.y,
            depth_or_array_layers: 1,
        };
        self.size_shadow = glam::uvec2(self.size.x.div_ceil(8), self.size.y.div_ceil(4));
        let size_shadow = wgpu::Extent3d {
            width: self.size_shadow.x,
            height: self.size_shadow.y,
            depth_or_array_layers: 1,
        };

        self.textures.gbuffer_albedo = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_albedo"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_albedo = self.textures.gbuffer_albedo.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_albedo"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_normal = Some(device.create_texture_swap(&wgpu::TextureDescriptor {
            label: Some("gbuffer_normal"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::R32Uint,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_normal = self.textures.gbuffer_normal.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_normal"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_depth = Some(device.create_texture_swap(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::R32Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_depth = self.textures.gbuffer_depth.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_depth"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_velocity = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_velocity"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_velocity = self
            .textures
            .gbuffer_velocity
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_velocity"),
                ..Default::default()
            });

        self.textures.gbuffer_shadow = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_shadow"),
            size: size_shadow,
            sample_count: 1,
            format: wgpu::TextureFormat::R32Uint,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_shadow = self.textures.gbuffer_shadow.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_shadow"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_illumination =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_illumination"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_illumination = self
            .textures
            .gbuffer_illumination
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_illumination"),
                ..Default::default()
            });

        self.textures.gbuffer_acc_illumination =
            Some(device.create_texture_swap(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_illumination"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_acc_illumination = self
            .textures
            .gbuffer_acc_illumination
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination"),
                ..Default::default()
            });

        self.textures.gbuffer_acc_illumination_moments =
            Some(device.create_texture_swap(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_illumination_moments"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_acc_illumination_moments = self
            .textures
            .gbuffer_acc_illumination_moments
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_moments"),
                ..Default::default()
            });

        self.textures.gbuffer_acc_illumination_history_len =
            Some(device.create_texture_swap(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_illumination_history_len"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::R32Uint,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_acc_illumination_history_len = self
            .textures
            .gbuffer_acc_illumination_history_len
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_history_len"),
                ..Default::default()
            });

        self.textures.gbuffer_filter_result =
            Some(device.create_texture_swap(&wgpu::TextureDescriptor {
                label: Some("gbuffer_filter_result"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_filter_result = self
            .textures
            .gbuffer_filter_result
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.textures.gbuffer_specular = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_specular"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_specular = self
            .textures
            .gbuffer_specular
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_specular"),
                ..Default::default()
            });

        self.textures.deferred_output = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deferred_output"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_deferred_output = self.textures.deferred_output.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("deferred_output"),
                ..Default::default()
            },
        );

        self.textures.out_color = Some(device.create_texture_swap(&wgpu::TextureDescriptor {
            label: Some("out_color_a"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_out_color =
            self.textures
                .out_color
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("out_color"),
                    ..Default::default()
                });

        self.bind_groups.raymarch_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_gbuffer"),
                layout: &self.bg_layouts.raymarch_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                ],
            }));

        self.bind_groups.raymarch_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("raymarch_swap"),
                layout: &self.bg_layouts.raymarch_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_depth.both(),
                    },
                ],
            },
        ));

        self.bind_groups.ambient_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ambient_gbuffer"),
                layout: &self.bg_layouts.ambient_gbuffer,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_illumination),
                }],
            }));

        self.bind_groups.shadow_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_gbuffer"),
                layout: &self.bg_layouts.shadow_gbuffer,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_shadow),
                }],
            }));

        self.bind_groups.ambient_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("ambient_swap"),
                layout: &self.bg_layouts.ambient_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_depth.both(),
                    },
                ],
            },
        ));

        self.bind_groups.specular_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("specular_gbuffer"),
                layout: &self.bg_layouts.specular_gbuffer,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_specular),
                }],
            }));

        self.bind_groups.lighting_resolve_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_resolve_gbuffer"),
                layout: &self.bg_layouts.lighting_resolve_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_shadow),
                    },
                ],
            }));

        self.bind_groups.lighting_resolve_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("lighting_resolve_swap"),
                layout: &self.bg_layouts.lighting_resolve_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        // resource: SwapchainBindingResource::Single(
                        //     wgpu::BindingResource::TextureView(&view_gbuffer_filter_result.a),
                        // ),
                        resource: view_gbuffer_acc_illumination.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_acc_illumination.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: view_gbuffer_acc_illumination_moments.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 3,
                        resource: view_gbuffer_acc_illumination_moments.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 4,
                        resource: view_acc_illumination_history_len.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 5,
                        resource: view_acc_illumination_history_len.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 6,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 7,
                        resource: view_gbuffer_normal.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 8,
                        resource: view_gbuffer_depth.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 9,
                        resource: view_gbuffer_depth.both_reversed(),
                    },
                ],
            },
        ));

        self.bind_groups.atrous_per_pass_primary = Some(
            device.create_bind_group_swap(&SwapchainBindGroupDescriptor {
                label: Some("atrous_per_pass_primary"),
                layout: &self.bg_layouts.atrous_per_pass,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: SwapchainBindingResource::Single(
                            device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("atrous_sample_count"),
                                    contents: bytemuck::cast_slice(&[1u32]),
                                    usage: wgpu::BufferUsages::COPY_DST
                                        | wgpu::BufferUsages::UNIFORM,
                                })
                                .as_entire_binding(),
                        ),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_acc_illumination.both(),
                        // resource: SwapchainBindingResource::Single(
                        //     wgpu::BindingResource::TextureView(&view_gbuffer_acc_illumination.a),
                        // ),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        // resource: view_gbuffer_acc_illumination.both(),
                        resource: SwapchainBindingResource::Single(
                            wgpu::BindingResource::TextureView(&view_gbuffer_filter_result.a),
                        ),
                    },
                ],
            }),
        );

        self.bind_groups.atrous_per_pass_secondary = Some(
            device.create_bind_group_swap(&SwapchainBindGroupDescriptor {
                label: Some("atrous_per_pass_secondary"),
                layout: &self.bg_layouts.atrous_per_pass,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: SwapchainBindingResource::Single(
                            device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("atrous_sample_count"),
                                    contents: bytemuck::cast_slice(&[2u32]),
                                    usage: wgpu::BufferUsages::COPY_DST
                                        | wgpu::BufferUsages::UNIFORM,
                                })
                                .as_entire_binding(),
                        ),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: SwapchainBindingResource::Single(
                            wgpu::BindingResource::TextureView(&view_gbuffer_filter_result.a),
                        ),
                        // resource: SwapchainBindingResource::Single(
                        //     wgpu::BindingResource::TextureView(&view_gbuffer_acc_illumination.a),
                        // ),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: view_gbuffer_acc_illumination.both(),
                    },
                ],
            }),
        );

        self.bind_groups.atrous_per_pass = Some(std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atrous_per_pass"),
                layout: &self.bg_layouts.atrous_per_pass,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("atrous_sample_count"),
                                contents: bytemuck::cast_slice(&[1u32 << ((i + 2) as u32)]),
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                            })
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(match i & 1 {
                            0 => &view_gbuffer_filter_result.b,
                            _ => &view_gbuffer_filter_result.a,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(match i & 1 {
                            0 => &view_gbuffer_filter_result.a,
                            _ => &view_gbuffer_filter_result.b,
                        }),
                    },
                ],
            })
        }));

        self.bind_groups.atrous_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("atrous_swap"),
                layout: &self.bg_layouts.atrous_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_acc_illumination_history_len.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: view_gbuffer_depth.both(),
                    },
                ],
            },
        ));

        self.bind_groups.deferred_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_gbuffer"),
                layout: &self.bg_layouts.deferred_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_deferred_output),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                ],
            }));

        self.bind_groups.deferred_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("deferred_swap"),
                layout: &self.bg_layouts.deferred_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_depth.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        // resource: SwapchainBindingResource::Single(
                        //     wgpu::BindingResource::TextureView(&view_gbuffer_shadow),
                        // ),
                        resource: view_gbuffer_acc_illumination.both(),
                        // resource: SwapchainBindingResource::Single(
                        //     wgpu::BindingResource::TextureView(match ATROUS_PASS_COUNT & 1 {
                        //         0 => &view_gbuffer_filter_result.a,
                        //         _ => &view_gbuffer_filter_result.b,
                        //     }),
                        // ),
                    },
                    SwapchainBindGroupEntry {
                        binding: 3,
                        resource: SwapchainBindingResource::Single(
                            wgpu::BindingResource::TextureView(&view_gbuffer_specular),
                        ),
                    },
                ],
            },
        ));

        self.bind_groups.taa_input = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa_input"),
            layout: &self.bg_layouts.taa_input,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_deferred_output),
                },
            ],
        }));

        self.bind_groups.taa_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("taa_output_swap"),
                layout: &self.bg_layouts.taa_output,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_out_color.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_out_color.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: view_gbuffer_depth.both(),
                    },
                ],
            },
        ));

        self.bind_groups.fx_input_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("fx_input_swap"),
                layout: &self.bg_layouts.fx_input,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_out_color.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: SwapchainBindingResource::Single(wgpu::BindingResource::Sampler(
                            &self.samplers.linear,
                        )),
                    },
                ],
            },
        ));
    }

    pub fn fixed_update<'a>(&mut self, ctx: &'a mut RendererCtx) {
        #[derive(Debug)]
        struct DebugStats {
            fps: f64,
            frame: Duration,
        }
        if (ctx.config.print_debug_info) {
            let stats = DebugStats {
                fps: 1.0 / ctx.ui.debug.frame_avg.as_secs_f64(),
                frame: ctx.ui.debug.frame_avg,
            };
            eprintln!("{:?}", stats);
        }
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut RendererCtx) {
        if ctx.config.render_scale != self.render_scale {
            self.render_scale = ctx.config.render_scale;
            self.update_screen_resources(ctx.window, ctx.device);
        }
        ctx.ui.debug.render_resolution = self.size;

        // update uniform buffers
        {
            let jitter = noise::HALTON_16[(self.frame_id as usize) & 0xf].as_dvec2();

            ctx.engine.camera.size = self.size;
            let camera = CameraDataBuffer::from_camera(&ctx.engine.camera, jitter);
            let prev_camera = match self.prev_camera {
                Some(c) => c,
                None => camera,
            };
            self.prev_camera = Some(camera);
            self.sun_direction = glam::vec3(
                ctx.config.sun_altitude.cos() * ctx.config.sun_azimuth.cos(),
                ctx.config.sun_altitude.cos() * ctx.config.sun_azimuth.sin(),
                ctx.config.sun_altitude.sin(),
            )
            .normalize();
            ctx.ui.debug.sun_direction = self.sun_direction;
            ctx.queue.write_buffer(
                &self.buffers.environment,
                0,
                bytemuck::cast_slice(&[EnvironmentDataBuffer {
                    sun_direction: self.sun_direction,
                    shadow_bias: ctx.config.shadow_bias,
                    camera,
                    prev_camera,
                    shadow_spread: ctx.config.shadow_spread,
                    filter_shadows: ctx.config.filter_shadows as u32,
                    shadow_filter_radius: ctx.config.shadow_filter_radius,
                    max_ambient_distance: ctx.config.ambient_ray_max_distance,
                    voxel_normal_factor: ctx.config.voxel_normal_factor,
                    debug_view: ctx.config.view as u32,
                    indirect_sky_intensity: ctx.config.indirect_sky_intensity,
                    pad: 0.0,
                }]),
            );

            let mut model_data = ModelDataBuffer::default();
            model_data.update(&ctx.engine.model);
            ctx.queue
                .write_buffer(&self.buffers.model, 0, bytemuck::cast_slice(&[model_data]));

            ctx.queue.write_buffer(
                &self.buffers.frame_metadata,
                0,
                bytemuck::cast_slice(&[
                    self.frame_id,
                    ctx.config.taa as u32,
                    ctx.config.fxaa as u32,
                ]),
            );
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
            let mut pass = encoder.begin_compute_pass_timed("Raymarch", &mut self.timing);

            pass.set_pipeline(&self.pipelines.raymarch);
            pass.set_bind_group(0, &self.bind_groups.raymarch_gbuffer, &[]);
            pass.set_bind_group_swap(1, &self.bind_groups.raymarch_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.raymarch_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("raymarch");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // shadow pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Shadow", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.shadow);
        //     pass.set_bind_group(0, &self.bind_groups.shadow_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.ambient_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.ambient_static, &[]);
        //     pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("shadow");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(4), 1);
        // }

        // // ambient pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Ambient", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.ambient);
        //     pass.set_bind_group(0, &self.bind_groups.ambient_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.ambient_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.ambient_static, &[]);
        //     pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("ambient");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        // }

        // specular pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Specular", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.specular);
        //     pass.set_bind_group(0, &self.bind_groups.specular_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.ambient_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.raymarch_static, &[]);
        //     pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("specular");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        // }

        // lighting resolve pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Lighting Resolve", &mut self.timing);

            pass.set_pipeline(&self.pipelines.lighting_resolve);
            pass.set_bind_group(0, &self.bind_groups.lighting_resolve_gbuffer, &[]);
            pass.set_bind_group_swap(
                1,
                &self.bind_groups.lighting_resolve_swap,
                &[],
                self.frame_id,
            );
            pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("lighting_resolve");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // denoise pass
        // for i in 0..ATROUS_PASS_COUNT {
        //     let label = format!("Denoise_{}", i);
        //     let mut pass = encoder.begin_compute_pass_timed(&label, &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.atrous);
        //     if i < 2 {
        //         pass.set_bind_group_swap(
        //             0,
        //             match i {
        //                 0 => &self.bind_groups.atrous_per_pass_primary,
        //                 _ => &self.bind_groups.atrous_per_pass_secondary,
        //             },
        //             &[],
        //             self.frame_id,
        //         );
        //     } else {
        //         pass.set_bind_group(
        //             0,
        //             self.bind_groups
        //                 .atrous_per_pass
        //                 .as_ref()
        //                 .map(|bg| &bg[i - 2]),
        //             &[],
        //         );
        //     }
        //     pass.set_bind_group_swap(1, &self.bind_groups.atrous_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("atrous");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        // }

        // deferred pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Deferred", &mut self.timing);

            pass.set_pipeline(&self.pipelines.deferred);
            pass.set_bind_group(0, &self.bind_groups.deferred_gbuffer, &[]);
            pass.set_bind_group_swap(1, &self.bind_groups.deferred_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.deferred_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("deferred");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // taa pass
        {
            let mut pass = encoder.begin_compute_pass_timed("TAA", &mut self.timing);

            pass.set_pipeline(&self.pipelines.taa);
            pass.set_bind_group(0, &self.bind_groups.per_frame_shared, &[]);
            pass.set_bind_group(1, &self.bind_groups.taa_input, &[]);
            pass.set_bind_group_swap(2, &self.bind_groups.taa_swap, &[], self.frame_id);

            pass.insert_debug_marker("taa");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
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
                ..Default::default()
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.fx);
            pass.set_bind_group_swap(0, &self.bind_groups.fx_input_swap, &[], self.frame_id);
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);
            pass.set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));

            pass.insert_debug_marker("fx");
            pass.draw_indexed(0..self.quad.index_count, 0, 0..1);
        }

        if let Some(timing) = &self.timing {
            timing.resolve(&mut encoder);
        }

        ctx.ui.frame(&mut UiCtx {
            window: ctx.window,
            device: ctx.device,
            queue: ctx.queue,
            texture_view: &texture_view,
            encoder: &mut encoder,
            config: ctx.config,
        });

        ctx.queue.submit([encoder.finish()]);

        ctx.window.pre_present_notify();
        surface_texture.present();

        if let Some(timing) = &mut self.timing
            && let Some(results) = timing.gather_results()
        {
            ctx.ui.debug.pass_avg = results;
        }

        self.frame_id = self.frame_id.wrapping_add(1);
    }
}

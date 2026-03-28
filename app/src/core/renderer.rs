use std::{sync::Arc, time::Duration};

use ddsfile::Dds;
use wgpu::{ShaderStages, util::DeviceExt};

use crate::{
    SizedWindow,
    config::Config,
    core::{
        buffers::{
            self, CameraDataBuffer, EnvironmentDataBuffer, FrameMetadataBuffer, ModelDataBuffer,
            PostFxSettingsBuffer, VoxelMapInfoBuffer,
        },
        engine::Engine,
        noise,
        quad::Quad,
        timing::{RenderTimer, TimedEncoder},
    },
    ui::{Ui, UiCtx},
};
use utils::{
    define_shaders,
    layout::{DeviceUtils, *},
    pipeline::PipelineUtils,
    textures::{
        DeviceSwapExt, PassSwapExt, SwapchainBindGroup, SwapchainBindGroupDescriptor,
        SwapchainBindGroupEntry, SwapchainBindingResource, SwapchainTexture,
    },
};

define_shaders! {
    raymarch "../shaders/raymarch.wgsl",
    visible_indirect_args "../shaders/visible_indirect_args.wgsl",
    shadow "../shaders/shadow.wgsl",
    ambient "../shaders/ambient.wgsl",
    specular "../shaders/specular.wgsl",
    lighting_resolve "../shaders/lighting_resolve.wgsl",
    specular_spatial "../shaders/specular_spatial.wgsl",
    specular_resolve "../shaders/specular_resolve.wgsl",
    resolve "../shaders/resolve.wgsl",
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
    render_scale: f32,
    sun_direction: glam::Vec3,
    prev_camera: Option<CameraDataBuffer>,
}

struct Pipelines {
    raymarch: wgpu::ComputePipeline,
    visible_indirect_args: wgpu::ComputePipeline,
    shadow: wgpu::ComputePipeline,
    ambient: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    specular: wgpu::ComputePipeline,
    specular_spatial: wgpu::ComputePipeline,
    specular_resolve: wgpu::ComputePipeline,
    deferred: wgpu::ComputePipeline,
    taa: wgpu::ComputePipeline,
    fx: wgpu::RenderPipeline,
}

struct BindGroupLayouts {
    per_frame_shared: wgpu::BindGroupLayout,
    raymarch_gbuffer: wgpu::BindGroupLayout,
    raymarch_swap: wgpu::BindGroupLayout,
    raymarch_static: wgpu::BindGroupLayout,
    visible_indirect_args: wgpu::BindGroupLayout,
    shadow_static: wgpu::BindGroupLayout,
    ambient_static: wgpu::BindGroupLayout,
    resolve: wgpu::BindGroupLayout,
    specular_gbuffer: wgpu::BindGroupLayout,
    specular_swap: wgpu::BindGroupLayout,
    specular_static: wgpu::BindGroupLayout,
    spec_spatial_gbuffer: wgpu::BindGroupLayout,
    spec_resolve_gbuffer: wgpu::BindGroupLayout,
    spec_resolve_swap: wgpu::BindGroupLayout,
    deferred_gbuffer: wgpu::BindGroupLayout,
    deferred_swap: wgpu::BindGroupLayout,
    deferred_static: wgpu::BindGroupLayout,
    taa_input: wgpu::BindGroupLayout,
    taa_output: wgpu::BindGroupLayout,
    fx_settings: wgpu::BindGroupLayout,
    fx_input: wgpu::BindGroupLayout,
}

struct BindGroups {
    per_frame_shared: wgpu::BindGroup,
    raymarch_gbuffer: Option<wgpu::BindGroup>,
    raymarch_swap: Option<SwapchainBindGroup>,
    raymarch_static: wgpu::BindGroup,
    visible_indirect_args: wgpu::BindGroup,
    shadow_static: wgpu::BindGroup,
    ambient_static: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
    specular_gbuffer: Option<wgpu::BindGroup>,
    specular_swap: Option<SwapchainBindGroup>,
    specular_static: wgpu::BindGroup,
    spec_spatial_gbuffer: Option<wgpu::BindGroup>,
    spec_resolve_gbuffer: Option<wgpu::BindGroup>,
    spec_resolve_swap: Option<SwapchainBindGroup>,
    deferred_gbuffer: Option<wgpu::BindGroup>,
    deferred_swap: Option<SwapchainBindGroup>,
    deferred_static: wgpu::BindGroup,
    taa_input: Option<wgpu::BindGroup>,
    taa_swap: Option<SwapchainBindGroup>,
    fx_settings: wgpu::BindGroup,
    fx_input_swap: Option<SwapchainBindGroup>,
}

struct Textures {
    gbuffer_albedo: Option<wgpu::Texture>,
    gbuffer_voxel_id: Option<wgpu::Texture>,
    gbuffer_normal: Option<SwapchainTexture>,
    gbuffer_depth: Option<SwapchainTexture>,
    gbuffer_velocity: Option<wgpu::Texture>,
    // gbuffer_specular_velocity: Option<wgpu::Texture>,
    gbuffer_specular: Option<wgpu::Texture>,
    gbuffer_specular_dir_pdf: Option<wgpu::Texture>,
    gbuffer_specular_spatial: Option<wgpu::Texture>,
    gbuffer_acc_specular: Option<SwapchainTexture>,
    deferred_output: Option<wgpu::Texture>,
    out_color: Option<SwapchainTexture>,
    noise_vector3_uniform: wgpu::Texture,
    tonemap_mcmapface_lut: wgpu::Texture,
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
    voxel_map_info: wgpu::Buffer,
    voxel_map: wgpu::Buffer,
    visible_voxels: wgpu::Buffer,
    visible_indirect_args: wgpu::Buffer,
    chunk_lighting: wgpu::Buffer,
    cur_voxel_lighting: wgpu::Buffer,
    acc_voxel_lighting: wgpu::Buffer,
    frame_metadata: wgpu::Buffer,
    fx_settings: wgpu::Buffer,
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
                    storage_texture().r32uint().dimension_2d().write_only(),
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
                    storage_buffer().read_write(),
                    storage_buffer().read_write(),
                    storage_buffer().read_write(),
                ),
            ),
            visible_indirect_args: device.layout(
                "visible_indirect_args",
                ShaderStages::COMPUTE,
                (storage_buffer().read_only(), storage_buffer().read_write()),
            ),
            shadow_static: device.layout(
                "shadow_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
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
                    sampler().filtering(),
                    texture().float().dimension_cube(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                    storage_buffer().read_only(),
                ),
            ),
            resolve: device.layout(
                "resolve_swap",
                ShaderStages::COMPUTE,
                (
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                ),
            ),
            specular_gbuffer: device.layout(
                "specular_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    // storage_texture().rgba16float().dimension_2d().write_only(),
                ),
            ),
            specular_swap: device.layout(
                "specular_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                ),
            ),
            specular_static: device.layout(
                "specular_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                    sampler().filtering(),
                    texture().float().dimension_cube(),
                    storage_buffer().read_only(),
                ),
            ),
            spec_spatial_gbuffer: device.layout(
                "specular_spatial_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                ),
            ),
            spec_resolve_gbuffer: device.layout(
                "specular_resolve_gbuffer",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                ),
            ),
            spec_resolve_swap: device.layout(
                "specular_resolve_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
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
                    storage_texture().r32uint().dimension_2d().read_only(),
                ),
            ),
            deferred_swap: device.layout(
                "deferred_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                ),
            ),
            deferred_static: device.layout(
                "deferred_static",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    sampler().non_filtering(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_cube(),
                    texture().float().dimension_2d(),
                    storage_buffer().read_only(),
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
            fx_settings: device.layout("fx_settings", ShaderStages::FRAGMENT, uniform_buffer()),
            fx_input: device.layout(
                "fx_input",
                ShaderStages::FRAGMENT,
                (
                    texture().float().dimension_2d(),
                    sampler().filtering(),
                    texture().float().dimension_3d(),
                ),
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
            visible_indirect_args: device
                .compute_pipeline("visible_indirect_args", &shaders.visible_indirect_args)
                .layout(&[&bg_layouts.visible_indirect_args]),
            shadow: device
                .compute_pipeline("shadow", &shaders.shadow)
                .layout(&[&bg_layouts.shadow_static, &bg_layouts.per_frame_shared]),
            ambient: device
                .compute_pipeline("ambient", &shaders.ambient)
                .layout(&[&bg_layouts.ambient_static, &bg_layouts.per_frame_shared]),
            resolve: device
                .compute_pipeline("resolve", &shaders.resolve)
                .layout(&[&bg_layouts.resolve, &bg_layouts.per_frame_shared]),
            specular: device
                .compute_pipeline("specular", &shaders.specular)
                .layout(&[
                    &bg_layouts.specular_gbuffer,
                    &bg_layouts.specular_swap,
                    &bg_layouts.specular_static,
                    &bg_layouts.per_frame_shared,
                ]),
            specular_spatial: device
                .compute_pipeline("specular_spatial", &shaders.specular_spatial)
                .layout(&[
                    &bg_layouts.spec_spatial_gbuffer,
                    &bg_layouts.specular_swap,
                    &bg_layouts.per_frame_shared,
                ]),
            specular_resolve: device
                .compute_pipeline("specular_resolve", &shaders.specular_resolve)
                .layout(&[
                    &bg_layouts.spec_resolve_gbuffer,
                    &bg_layouts.spec_resolve_swap,
                    &bg_layouts.per_frame_shared,
                ]),
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
                        bind_group_layouts: &[
                            &bg_layouts.fx_input,
                            &bg_layouts.per_frame_shared,
                            &bg_layouts.fx_settings,
                        ],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &shaders.fx,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: crate::core::quad::VERTEX_SIZE,
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
        let scene = config.init_scene.load(device, queue).unwrap();
        let skybox = config.init_skybox.load(device, queue).unwrap();

        let textures = Textures {
            gbuffer_albedo: None,
            gbuffer_voxel_id: None,
            gbuffer_normal: None,
            gbuffer_depth: None,
            gbuffer_velocity: None,
            // gbuffer_specular_velocity: None,
            gbuffer_specular: None,
            gbuffer_specular_dir_pdf: None,
            gbuffer_specular_spatial: None,
            gbuffer_acc_specular: None,
            deferred_output: None,
            out_color: None,
            noise_vector3_uniform: noise::noise_vector3_uniform_binomial3x3_exp_product(
                device, queue,
            )
            .unwrap(),
            tonemap_mcmapface_lut: load_tonemap_lut(device, queue).unwrap(),
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

        let leaf_chunks_size = scene.data.buffer_leaf_chunks.size();

        let voxel_queue_len = 1920u64 * 1080;

        // let visiblity_mask_size = leaf_chunks_size.div_ceil(32).next_multiple_of(4);
        // let voxel_map_size = leaf_chunks_size; // crude af should be dynamic and smaller
        // let voxel_lighting_size = voxel_map_size.div_ceil(2).next_multiple_of(4);

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
            // voxel_index_leaf_positions: scene.data.buffer_index_leaf_positions,
            voxel_map_info: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel_map_info"),
                size: std::mem::size_of::<VoxelMapInfoBuffer>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            voxel_map: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel_map"),
                size: voxel_queue_len * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visible_voxels: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visible_voxels"),
                size: voxel_queue_len * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visible_indirect_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visible_indirect_args"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            chunk_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk_lighting"),
                size: scene.meta.allocated_index_chunks as u64 * 4 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cur_voxel_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel_lighting"),
                size: voxel_queue_len * 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            acc_voxel_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("acc_voxel_lighting"),
                size: leaf_chunks_size * 3,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            frame_metadata: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame_metadata"),
                size: std::mem::size_of::<buffers::FrameMetadataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            fx_settings: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx_settings"),
                size: std::mem::size_of::<buffers::PostFxSettingsBuffer>() as u64,
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
                            &textures
                                .noise_vector3_uniform
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buffers.voxel_map_info.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: buffers.voxel_map.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: buffers.visible_voxels.as_entire_binding(),
                    },
                ],
            }),
            raymarch_swap: None,
            visible_indirect_args: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("visible_indirect_args"),
                layout: &bg_layouts.visible_indirect_args,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.voxel_map_info.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.visible_indirect_args.as_entire_binding(),
                    },
                ],
            }),
            shadow_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_static"),
                layout: &bg_layouts.shadow_static,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.voxel_scene_metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.voxel_index_chunks.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &textures
                                .noise_vector3_uniform
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.visible_voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.cur_voxel_lighting.as_entire_binding(),
                    },
                ],
            }),
            ambient_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ambient_static"),
                layout: &bg_layouts.ambient_static,
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
                            &textures
                                .noise_vector3_uniform
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            &skybox
                                .downsampled
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("skybox_downsampled"),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: buffers.visible_voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: buffers.cur_voxel_lighting.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: buffers.acc_voxel_lighting.as_entire_binding(),
                    },
                ],
            }),
            resolve: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resolve_swap"),
                layout: &bg_layouts.resolve,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.visible_voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.cur_voxel_lighting.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.acc_voxel_lighting.as_entire_binding(),
                    },
                ],
            }),
            specular_gbuffer: None,
            specular_swap: None,
            specular_static: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("specular_static"),
                layout: &bg_layouts.specular_static,
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
                            &textures
                                .noise_vector3_uniform
                                .create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            &skybox
                                .downsampled
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("skybox_downsampled"),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: buffers.acc_voxel_lighting.as_entire_binding(),
                    },
                ],
            }),
            spec_spatial_gbuffer: None,
            spec_resolve_gbuffer: None,
            spec_resolve_swap: None,
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
                        resource: wgpu::BindingResource::TextureView(&skybox.cubemap.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("skybox"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &skybox
                                .downsampled
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("irradiance"),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &skybox.prefilter.create_view(&wgpu::TextureViewDescriptor {
                                label: Some("prefilter"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&skybox.brdf.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("brdf"),
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buffers.acc_voxel_lighting.as_entire_binding(),
                    },
                ],
            }),
            taa_input: None,
            taa_swap: None,
            fx_settings: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_settings"),
                layout: &bg_layouts.fx_settings,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.fx_settings.as_entire_binding(),
                }],
            }),
            fx_input_swap: None,
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device, 10));

        let quad = Quad::new(device);

        config.voxel_scale = scene.meta.voxels_per_unit;
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
        let quarter_size = wgpu::Extent3d {
            width: self.size.x.div_ceil(2),
            height: self.size.y.div_ceil(2),
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
        let view_gbuffer_albedo = self
            .textures
            .gbuffer_albedo
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.textures.gbuffer_voxel_id = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_voxel_id"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::R32Uint,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_voxel_id = self
            .textures
            .gbuffer_voxel_id
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

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
        let view_gbuffer_normal = self
            .textures
            .gbuffer_normal
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

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
        let view_gbuffer_depth = self
            .textures
            .gbuffer_depth
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

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
            .create_view(&Default::default());

        self.textures.gbuffer_specular = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_specular"),
            size: quarter_size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_specular = self
            .textures
            .gbuffer_specular
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        // self.textures.gbuffer_specular_velocity =
        //     Some(device.create_texture(&wgpu::TextureDescriptor {
        //         label: Some("gbuffer_specular_velocity"),
        //         size: quarter_size,
        //         sample_count: 1,
        //         format: wgpu::TextureFormat::Rgba16Float,
        //         dimension: wgpu::TextureDimension::D2,
        //         usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        //         mip_level_count: 1,
        //         view_formats: &[],
        //     }));
        // let view_gbuffer_specular_velocity = self
        //     .textures
        //     .gbuffer_specular_velocity
        //     .as_ref()
        //     .unwrap()
        //     .create_view(&Default::default());

        self.textures.gbuffer_specular_dir_pdf =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_specular_direction_pdf"),
                size: quarter_size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_specular_dir_pdf = self
            .textures
            .gbuffer_specular_dir_pdf
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.textures.gbuffer_specular_spatial =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_specular_spatial"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_specular_spatial = self
            .textures
            .gbuffer_specular_spatial
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.textures.gbuffer_acc_specular =
            Some(device.create_texture_swap(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_specular"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_acc_specular = self
            .textures
            .gbuffer_acc_specular
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_voxel_id),
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

        self.bind_groups.specular_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("specular_gbuffer"),
                layout: &self.bg_layouts.specular_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_specular),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_specular_dir_pdf,
                        ),
                    },
                    // wgpu::BindGroupEntry {
                    //     binding: 1,
                    //     resource: wgpu::BindingResource::TextureView(
                    //         &view_gbuffer_specular_velocity,
                    //     ),
                    // },
                ],
            }));

        self.bind_groups.specular_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("specular_swap"),
                layout: &self.bg_layouts.specular_swap,
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

        self.bind_groups.spec_spatial_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("specular_spatial_gbuffer"),
                layout: &self.bg_layouts.spec_spatial_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_specular),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_specular_dir_pdf,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_specular_spatial,
                        ),
                    },
                ],
            }));

        self.bind_groups.spec_resolve_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("specular_resolve_gbuffer"),
                layout: &self.bg_layouts.spec_resolve_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_specular_spatial,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            // &view_gbuffer_specular_velocity,
                            &view_gbuffer_specular_dir_pdf,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                ],
            }));

        self.bind_groups.spec_resolve_swap = Some(device.create_bind_group_swap(
            &SwapchainBindGroupDescriptor {
                label: Some("specular_resolve_swap"),
                layout: &self.bg_layouts.spec_resolve_swap,
                entries: &[
                    SwapchainBindGroupEntry {
                        binding: 0,
                        resource: view_gbuffer_acc_specular.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 1,
                        resource: view_gbuffer_acc_specular.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: view_gbuffer_normal.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 3,
                        resource: view_gbuffer_normal.both_reversed(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 4,
                        resource: view_gbuffer_depth.both(),
                    },
                    SwapchainBindGroupEntry {
                        binding: 5,
                        resource: view_gbuffer_depth.both_reversed(),
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_voxel_id),
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
                        //     wgpu::BindingResource::TextureView(&view_gbuffer_specular_spatial),
                        // ),
                        resource: view_gbuffer_acc_specular.both(),
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

        self.bind_groups.fx_input_swap = Some(
            device.create_bind_group_swap(&SwapchainBindGroupDescriptor {
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
                    SwapchainBindGroupEntry {
                        binding: 2,
                        resource: SwapchainBindingResource::Single(
                            wgpu::BindingResource::TextureView(
                                &self
                                    .textures
                                    .tonemap_mcmapface_lut
                                    .create_view(&Default::default()),
                            ),
                        ),
                    },
                ],
            }),
        );
    }

    pub fn fixed_update(&mut self, ctx: RendererCtx<'_>) {
        let RendererCtx {
            device,
            queue,
            ui,
            config,
            ..
        } = ctx;

        #[derive(Debug)]
        struct DebugStats {
            fps: f64,
            frame: Duration,
        }
        if config.print_debug_info {
            let stats = DebugStats {
                fps: 1.0 / ui.debug.frame_avg.as_secs_f64(),
                frame: ui.debug.frame_avg,
            };
            eprintln!("{:?}", stats);
        }

        let buffer_results = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.buffers.voxel_map_info.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.buffers.voxel_map_info,
            0,
            &buffer_results,
            0,
            buffer_results.size(),
        );
        queue.submit([encoder.finish()]);
        let buffer_results = buffer_results.slice(..);
        buffer_results.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer_results.get_mapped_range();
        let data: &[VoxelMapInfoBuffer] = bytemuck::cast_slice(&data);
        let data = data[0];

        dbg!(&data);
    }

    pub fn frame(&mut self, _delta_time: &Duration, ctx: RendererCtx<'_>) {
        let RendererCtx {
            window,
            device,
            queue,
            surface,
            format,
            engine,
            ui,
            config,
        } = ctx;

        if config.render_scale != self.render_scale {
            self.render_scale = config.render_scale;
            self.update_screen_resources(window, device);
        }
        ui.debug.render_resolution = self.size;

        // update uniform buffers
        {
            let jitter = noise::HALTON_16[(self.frame_id as usize) & 0xf].as_dvec2();

            engine.camera.size = self.size;
            let camera = CameraDataBuffer::from_camera(&engine.camera, jitter);
            let prev_camera = match self.prev_camera {
                Some(c) => c,
                None => camera,
            };
            self.prev_camera = Some(camera);
            self.sun_direction = glam::vec3(
                config.sun_altitude.cos() * config.sun_azimuth.cos(),
                config.sun_altitude.cos() * config.sun_azimuth.sin(),
                config.sun_altitude.sin(),
            )
            .normalize();
            ui.debug.sun_direction = self.sun_direction;
            queue.write_buffer(
                &self.buffers.environment,
                0,
                bytemuck::cast_slice(&[EnvironmentDataBuffer {
                    sun_direction: self.sun_direction,
                    sun_intensity: config.sun_intensity,
                    sun_color: config.sun_color,
                    shadow_bias: config.shadow_bias,
                    skybox_rotation_cos_sin: glam::vec2(
                        config.skybox_rotation.cos(),
                        config.skybox_rotation.sin(),
                    ),
                    _pad_0: [0.0; 2],
                    camera,
                    prev_camera,
                    shadow_spread: config.shadow_spread,
                    filter_shadows: config.filter_shadows as u32,
                    shadow_filter_radius: config.shadow_filter_radius,
                    max_ambient_distance: config.ambient_ray_max_distance,
                    voxel_normal_factor: config.voxel_normal_factor,
                    debug_view: config.view as u32,
                    indirect_sky_intensity: config.indirect_sky_intensity,
                    _pad_1: 0.0,
                }]),
            );

            let mut model_data = ModelDataBuffer::default();
            model_data.update(&engine.model);
            queue.write_buffer(&self.buffers.model, 0, bytemuck::cast_slice(&[model_data]));

            queue.write_buffer(
                &self.buffers.frame_metadata,
                0,
                bytemuck::cast_slice(&[FrameMetadataBuffer {
                    frame_id: self.frame_id,
                    taa_enabled: config.taa as u32,
                }]),
            );
            queue.write_buffer(
                &self.buffers.fx_settings,
                0,
                bytemuck::cast_slice(&[PostFxSettingsBuffer {
                    fxaa_enabled: config.fxaa as u32,
                    exposure: config.exposure as f32,
                    tonemapping: config.tonemapping as u32,
                }]),
            );
        }

        let surface_texture = surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = device.create_command_encoder(&Default::default());

        encoder.clear_buffer(&self.buffers.voxel_map_info, 0, None);
        encoder.clear_buffer(&self.buffers.voxel_map, 0, None);
        encoder.clear_buffer(&self.buffers.cur_voxel_lighting, 0, None);

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

        // copy over indirect args for the per-voxel dispatches
        {
            let mut pass =
                encoder.begin_compute_pass_timed("Prepare Visibility Args", &mut self.timing);

            pass.set_pipeline(&self.pipelines.visible_indirect_args);
            pass.set_bind_group(0, &self.bind_groups.visible_indirect_args, &[]);

            pass.insert_debug_marker("visible indirect args");
            pass.dispatch_workgroups(1, 1, 1);
        }

        // shadow pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Shadow", &mut self.timing);

            pass.set_pipeline(&self.pipelines.shadow);
            pass.set_bind_group(0, &self.bind_groups.shadow_static, &[]);
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("shadow");
            pass.dispatch_workgroups_indirect(&self.buffers.visible_indirect_args, 0);
        }

        // ambient pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Ambient", &mut self.timing);

            pass.set_pipeline(&self.pipelines.ambient);
            pass.set_bind_group(0, &self.bind_groups.ambient_static, &[]);
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("ambient");
            pass.dispatch_workgroups_indirect(&self.buffers.visible_indirect_args, 0);
        }

        // resolve pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Resolve", &mut self.timing);

            pass.set_pipeline(&self.pipelines.resolve);
            pass.set_bind_group(0, Some(&self.bind_groups.resolve), &[]);
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("resolve");
            pass.dispatch_workgroups_indirect(&self.buffers.visible_indirect_args, 0);
        }

        // specular pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Specular", &mut self.timing);

            pass.set_pipeline(&self.pipelines.specular);
            pass.set_bind_group(0, &self.bind_groups.specular_gbuffer, &[]);
            pass.set_bind_group_swap(1, &self.bind_groups.specular_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.specular_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            let size_quarter = self.size.map(|x| x.div_ceil(2));

            pass.insert_debug_marker("specular");
            pass.dispatch_workgroups(size_quarter.x.div_ceil(8), size_quarter.y.div_ceil(8), 1);
        }

        // specular spatial filter pass
        {
            let mut pass =
                encoder.begin_compute_pass_timed("Specular Spatial Filter", &mut self.timing);

            pass.set_pipeline(&self.pipelines.specular_spatial);
            pass.set_bind_group(0, &self.bind_groups.spec_spatial_gbuffer, &[]);
            pass.set_bind_group_swap(1, &self.bind_groups.specular_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("specular spatial");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // // specular resolve pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Specular Resolve", &mut self.timing);

            pass.set_pipeline(&self.pipelines.specular_resolve);
            pass.set_bind_group(0, &self.bind_groups.spec_resolve_gbuffer, &[]);
            pass.set_bind_group_swap(1, &self.bind_groups.spec_resolve_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("specular resolve");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

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
            pass.set_bind_group(2, &self.bind_groups.fx_settings, &[]);

            pass.set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));

            pass.insert_debug_marker("fx");
            pass.draw_indexed(0..self.quad.index_count, 0, 0..1);
        }

        if let Some(timing) = &self.timing {
            timing.resolve(&mut encoder);
        }

        ui.frame(&mut UiCtx {
            window: window,
            device: device,
            queue: queue,
            texture_view: &texture_view,
            encoder: &mut encoder,
            config: config,
        });

        queue.submit([encoder.finish()]);

        window.pre_present_notify();
        surface_texture.present();

        if let Some(timing) = &mut self.timing
            && let Some(results) = timing.gather_results()
        {
            ui.debug.pass_avg = results;
        }

        self.frame_id = self.frame_id.wrapping_add(1);
    }
}

/// Load the Tony McMapface LUT
///
/// shout out the fucking goat
/// https://github.com/h3r2tic/tony-mc-mapface
fn load_tonemap_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<wgpu::Texture> {
    let src = std::include_bytes!("../../assets/tonemap.dds");
    let img = Dds::read(src.as_slice())?;
    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tonemap_tony_mcmapface"),
        size: wgpu::Extent3d {
            width: img.get_width(),
            height: img.get_height(),
            depth_or_array_layers: img.get_depth(),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgb9e5Ufloat,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &res,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img.data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.get_width()),
            rows_per_image: Some(img.get_height()),
        },
        wgpu::Extent3d {
            width: img.get_width(),
            height: img.get_height(),
            depth_or_array_layers: img.get_depth(),
        },
    );
    Ok(res)
}

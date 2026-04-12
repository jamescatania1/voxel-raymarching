use std::{sync::Arc, time::Duration};

use ddsfile::Dds;
use image::buffer;
use wgpu::{ShaderStages, Texture, TextureUsages, util::DeviceExt};

use crate::{
    SizedWindow,
    config::Config,
    core::{
        buffers::{
            self, CameraDataBuffer, EnvironmentDataBuffer, FrameMetadataBuffer, ModelDataBuffer,
            PostFxSettingsBuffer, VisibilityInfoBuffer,
        },
        engine::Engine,
        noise,
        probes::ProbeGrid,
        quad::Quad,
        timing::{RenderTimer, TimedEncoder},
    },
    ui::{Ui, UiCtx},
};
use utils::{
    define_shaders,
    layout::{DeviceUtils, *},
    pipeline::PipelineUtils,
    texture::{TextureDescriptorExt, TextureExt, TextureViewExt, texture},
    textures::{PassSwapExt, SwapchainBindGroup, SwapchainBindingResource, SwapchainTexture},
};

define_shaders! {
    probe_trace "../shaders/probe_trace.wgsl",
    probe_update "../shaders/probe_update.wgsl",
    probe_border "../shaders/probe_border.wgsl",
    probe_visualize "../shaders/probe_visualize.wgsl",
    raymarch "../shaders/raymarch.wgsl",
    ambient_trace "../shaders/ambient_trace.wgsl",
    ambient_project "../shaders/ambient_project.wgsl",
    ambient_reproject "../shaders/ambient_reproject.wgsl",
    visible_indirect_args "../shaders/visible_indirect_args.wgsl",
    shadow "../shaders/shadow.wgsl",
    ambient "../shaders/ambient.wgsl",
    resolve "../shaders/resolve.wgsl",
    chunk_resolve "../shaders/chunk_resolve.wgsl",
    specular "../shaders/specular.wgsl",
    specular_spatial "../shaders/specular_spatial.wgsl",
    specular_resolve "../shaders/specular_resolve.wgsl",
    deferred "../shaders/deferred.wgsl",
    taa "../shaders/taa.wgsl",
    fx "../shaders/fx.wgsl",
    shadow_occlusion "../shaders/shadow_occlusion.wgsl",
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
    screen_bind_groups: Option<ScreenBindGroups>,
    samplers: Samplers,
    buffers: Buffers,
    textures: Textures,
    screen_textures: Option<ScreenTextures>,
    timing: Option<RenderTimer>,
    quad: Quad,
    frame_id: u32,
    size: glam::UVec2,
    render_scale: f32,
    sun_direction: glam::Vec3,
    prev_camera: Option<CameraDataBuffer>,
    scene_size: glam::UVec3,
    probe_size: glam::UVec3,
    shadow_update_requested: bool,
}

struct Pipelines {
    shadow_occlusion: wgpu::ComputePipeline,
    probe_trace: wgpu::ComputePipeline,
    probe_update: wgpu::ComputePipeline,
    probe_border: wgpu::ComputePipeline,
    raymarch: wgpu::ComputePipeline,
    ambient_trace: wgpu::ComputePipeline,
    ambient_project: wgpu::ComputePipeline,
    ambient_reproject: wgpu::ComputePipeline,
    voxel_indirect_args: wgpu::ComputePipeline,
    chunk_indirect_args: wgpu::ComputePipeline,
    shadow: wgpu::ComputePipeline,
    ambient: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    chunk_resolve: wgpu::ComputePipeline,
    specular: wgpu::ComputePipeline,
    specular_spatial: wgpu::ComputePipeline,
    specular_resolve: wgpu::ComputePipeline,
    deferred: wgpu::ComputePipeline,
    taa: wgpu::ComputePipeline,
    fx: wgpu::RenderPipeline,
    probe_visualize: wgpu::RenderPipeline,
}

struct BindGroupLayouts {
    per_frame_shared: wgpu::BindGroupLayout,
    shadow_occlusion_static: wgpu::BindGroupLayout,
    probe_trace_static: wgpu::BindGroupLayout,
    probe_update_static: wgpu::BindGroupLayout,
    probe_visualize_swap: wgpu::BindGroupLayout,
    raymarch_gbuffer: wgpu::BindGroupLayout,
    raymarch_swap: wgpu::BindGroupLayout,
    raymarch_static: wgpu::BindGroupLayout,
    visible_indirect_args: wgpu::BindGroupLayout,
    shadow_static: wgpu::BindGroupLayout,
    ambient_static: wgpu::BindGroupLayout,
    resolve: wgpu::BindGroupLayout,
    chunk_resolve_static: wgpu::BindGroupLayout,
    ambient_trace_gbuffer: wgpu::BindGroupLayout,
    ambient_trace_static: wgpu::BindGroupLayout,
    ambient_project_gbuffer: wgpu::BindGroupLayout,
    ambient_reproject_gbuffer: wgpu::BindGroupLayout,
    ambient_reproject_swap: wgpu::BindGroupLayout,
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
    shadow_occlusion_static: wgpu::BindGroup,
    probe_trace_static: wgpu::BindGroup,
    probe_update_static: wgpu::BindGroup,
    raymarch_static: wgpu::BindGroup,
    voxel_indirect_args: wgpu::BindGroup,
    chunk_indirect_args: wgpu::BindGroup,
    shadow_static: wgpu::BindGroup,
    ambient_static: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
    chunk_resolve_static: wgpu::BindGroup,
    ambient_trace_static: wgpu::BindGroup,
    specular_static: wgpu::BindGroup,
    deferred_static: wgpu::BindGroup,
    fx_settings: wgpu::BindGroup,
}
struct ScreenBindGroups {
    probe_visualize_swap: SwapchainBindGroup,
    raymarch_gbuffer: wgpu::BindGroup,
    raymarch_swap: SwapchainBindGroup,
    ambient_trace_gbuffer: wgpu::BindGroup,
    ambient_project_gbuffer: wgpu::BindGroup,
    ambient_reproject_gbuffer: wgpu::BindGroup,
    ambient_reproject_swap: SwapchainBindGroup,
    specular_gbuffer: wgpu::BindGroup,
    specular_swap: SwapchainBindGroup,
    spec_spatial_gbuffer: wgpu::BindGroup,
    spec_resolve_gbuffer: wgpu::BindGroup,
    spec_resolve_swap: SwapchainBindGroup,
    deferred_gbuffer: wgpu::BindGroup,
    deferred_swap: SwapchainBindGroup,
    taa_input: wgpu::BindGroup,
    taa_swap: SwapchainBindGroup,
    fx_input_swap: SwapchainBindGroup,
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
    visibility_info: wgpu::Buffer,
    voxel_map: wgpu::Buffer,
    voxel_shadow_mask: wgpu::Buffer,
    chunk_visibility_mask: wgpu::Buffer,
    visible_voxels: wgpu::Buffer,
    visible_chunks: wgpu::Buffer,
    voxel_indirect_args: wgpu::Buffer,
    chunk_indirect_args: wgpu::Buffer,
    probes: wgpu::Buffer,
    cur_chunk_lighting: wgpu::Buffer,
    acc_chunk_lighting: wgpu::Buffer,
    cur_voxel_lighting: wgpu::Buffer,
    acc_voxel_lighting: wgpu::Buffer,
    frame_metadata: wgpu::Buffer,
    fx_settings: wgpu::Buffer,
    environment: wgpu::Buffer,
    model: wgpu::Buffer,
}

struct Textures {
    probe_irradiance: wgpu::Texture,
    probe_depth: wgpu::Texture,
    probe_ray_results: wgpu::Texture,
    noise_coshemi: wgpu::Texture,
    tonemap_mcmapface_lut: wgpu::Texture,
}
struct ScreenTextures {
    albedo: Texture,
    voxel_id: Texture,
    normal: SwapchainTexture,
    depth: SwapchainTexture,
    velocity: Texture,
    gi_ray_radiance: Texture,
    gi_ray_direction: Texture,
    gi_sh_sample_rgb: [Texture; 3],
    gi_sh_rgb: [SwapchainTexture; 3],
    gi_history: SwapchainTexture,
    specular: Texture,
    specular_direction: Texture,
    specular_spatial: Texture,
    acc_specular: SwapchainTexture,
    deferred_output: Texture,
    out_color: SwapchainTexture,
}

impl ScreenTextures {
    fn create(device: &wgpu::Device, screen_size: glam::UVec2) -> Self {
        use wgpu::TextureUsages;

        let size = screen_size.extend(1);
        let half_size = size.map(|x| x.div_ceil(2));
        let quarter_size = size.map(|x| x.div_ceil(4));

        Self {
            albedo: texture("albedo")
                .rgba16float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create(device),
            voxel_id: texture("voxel_id")
                .r32uint()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create(device),
            normal: texture("normal")
                .r32uint()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create_swap(device),
            depth: texture("depth")
                .r32float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create_swap(device),
            velocity: texture("velocity")
                .rgba16float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create(device),
            gi_ray_radiance: texture("gi_ray_radiance")
                .rgba16float()
                .size(quarter_size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            gi_ray_direction: texture("gi_ray_direction")
                .rgba16float()
                .size(quarter_size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            gi_sh_sample_rgb: ["gi_sh_sample_r", "gi_sh_sample_g", "gi_sh_sample_b"].map(|l| {
                texture(l)
                    .rgba16float()
                    .size(half_size)
                    .d2()
                    .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                    .create(device)
            }),
            gi_sh_rgb: ["gi_sh_r", "gi_sh_g", "gi_sh_b"].map(|l| {
                texture(l)
                    .rgba16float()
                    .size(half_size)
                    .d2()
                    .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                    .create_swap(device)
            }),
            gi_history: texture("gi_history")
                .r32uint()
                .size(half_size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create_swap(device),
            specular: texture("specular")
                .rgba16float()
                .size(half_size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            specular_direction: texture("specular_direction")
                .rgba16float()
                .size(half_size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            specular_spatial: texture("specular_spatial")
                .rgba16float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            acc_specular: texture("acc_specular")
                .rgba16float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create_swap(device),
            deferred_output: texture("deferred_output")
                .rgba16float()
                .size(size)
                .d2()
                .usage(
                    TextureUsages::STORAGE_BINDING
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT,
                )
                .create(device),
            out_color: texture("out_color")
                .rgba16float()
                .size(size)
                .d2()
                .usage(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
                .create_swap(device),
        }
    }
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
                ShaderStages::all(),
                (uniform_buffer(), uniform_buffer(), uniform_buffer()),
            ),
            probe_trace_static: device.layout(
                "probe_trace_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    sampler().filtering(),
                    sampled_texture().float().dimension_cube(),
                    sampled_texture().float().dimension_2d(),
                    sampled_texture().float().dimension_2d(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_buffer().read_only(),
                ),
            ),
            probe_update_static: device.layout(
                "probe_update_static",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_write(),
                    storage_texture().rg16float().dimension_2d().read_write(),
                ),
            ),
            probe_visualize_swap: device.layout(
                "probe_visualize_static",
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer(),
                    sampled_texture().float().dimension_2d(),
                    storage_buffer().read_only(),
                ),
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
                    sampled_texture().unfilterable_float().dimension_3d(),
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
                    sampled_texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                    storage_buffer().read_write(),
                    storage_buffer().read_write(),
                    storage_buffer().read_write(),
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
                    storage_buffer().read_only(),
                    sampled_texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                    sampler().filtering(),
                    sampled_texture().float().dimension_cube(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                    sampled_texture().float().dimension_2d(),
                    sampled_texture().float().dimension_2d(),
                    storage_buffer().read_only(),
                ),
            ),
            chunk_resolve_static: device.layout(
                "chunk_resolve_static",
                ShaderStages::COMPUTE,
                (
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                ),
            ),
            resolve: device.layout(
                "resolve_swap",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                ),
            ),
            ambient_trace_gbuffer: device.layout(
                "ambient_trace_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                ),
            ),
            ambient_trace_static: device.layout(
                "ambient_trace_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    sampled_texture().unfilterable_float().dimension_3d(),
                    sampler().filtering(),
                    sampled_texture().float().dimension_cube(),
                    storage_buffer().read_only(),
                ),
            ),
            ambient_project_gbuffer: device.layout(
                "ambient_project_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                ),
            ),
            ambient_reproject_gbuffer: device.layout(
                "ambient_reproject_gbuffer",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                ),
            ),
            ambient_reproject_swap: device.layout(
                "ambient_reproject_swap",
                ShaderStages::COMPUTE,
                (
                    storage_texture().rgba16float().dimension_2d().read_write(),
                    storage_texture().rgba16float().dimension_2d().read_write(),
                    storage_texture().rgba16float().dimension_2d().read_write(),
                    storage_texture().r32uint().dimension_2d().write_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                    storage_texture().r32float().dimension_2d().read_only(),
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
                    sampled_texture().unfilterable_float().dimension_3d(),
                    sampler().non_filtering(),
                    sampler().filtering(),
                    sampled_texture().float().dimension_cube(),
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
                    sampled_texture().float().dimension_2d(),
                    sampled_texture().float().dimension_2d(),
                    sampled_texture().float().dimension_2d(),
                    storage_texture().r32uint().dimension_2d().read_only(),
                ),
            ),
            deferred_static: device.layout(
                "deferred_static",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    sampler().non_filtering(),
                    sampled_texture().float().dimension_cube(),
                    sampled_texture().float().dimension_cube(),
                    sampled_texture().float().dimension_cube(),
                    sampled_texture().float().dimension_2d(),
                    storage_buffer().read_only(),
                    storage_buffer().read_only(),
                    uniform_buffer(),
                    sampled_texture().float().dimension_2d(),
                    sampled_texture().float().dimension_2d(),
                    storage_buffer().read_only(),
                ),
            ),
            taa_input: device.layout(
                "taa_input",
                ShaderStages::COMPUTE,
                (
                    sampler().filtering(),
                    storage_texture().rgba16float().dimension_2d().read_only(),
                    sampled_texture().float().dimension_2d(),
                ),
            ),
            taa_output: device.layout(
                "taa_output",
                ShaderStages::COMPUTE,
                (
                    sampled_texture().float().dimension_2d(),
                    storage_texture().rgba16float().dimension_2d().write_only(),
                    sampled_texture().float().dimension_2d(),
                ),
            ),
            fx_settings: device.layout("fx_settings", ShaderStages::FRAGMENT, uniform_buffer()),
            fx_input: device.layout(
                "fx_input",
                ShaderStages::FRAGMENT,
                (
                    sampled_texture().float().dimension_2d(),
                    sampler().filtering(),
                    sampled_texture().float().dimension_3d(),
                ),
            ),
            shadow_occlusion_static: device.layout(
                "shadow_occlusion_static",
                ShaderStages::COMPUTE,
                (
                    uniform_buffer(),
                    storage_buffer().read_only(),
                    storage_buffer().read_write(),
                ),
            ),
        };

        let shaders = device.create_shaders();

        let pipelines = Pipelines {
            shadow_occlusion: device
                .compute_pipeline("shadow_occlusion", &shaders.shadow_occlusion)
                .layout(&[
                    &bg_layouts.shadow_occlusion_static,
                    &bg_layouts.per_frame_shared,
                ]),
            probe_trace: device
                .compute_pipeline("probe_trace", &shaders.probe_trace)
                .layout(&[&bg_layouts.probe_trace_static, &bg_layouts.per_frame_shared]),
            probe_update: device
                .compute_pipeline("probe_update", &shaders.probe_update)
                .layout(&[
                    &bg_layouts.probe_update_static,
                    &bg_layouts.per_frame_shared,
                ]),
            probe_border: device
                .compute_pipeline("probe_border", &shaders.probe_border)
                .layout(&[&bg_layouts.probe_update_static]),
            raymarch: device
                .compute_pipeline("raymarch", &shaders.raymarch)
                .layout(&[
                    &bg_layouts.raymarch_gbuffer,
                    &bg_layouts.raymarch_swap,
                    &bg_layouts.raymarch_static,
                    &bg_layouts.per_frame_shared,
                ]),
            voxel_indirect_args: device
                .compute_pipeline("voxel_indirect_args", &shaders.visible_indirect_args)
                .entry_point("compute_voxels")
                .layout(&[&bg_layouts.visible_indirect_args]),
            chunk_indirect_args: device
                .compute_pipeline("chunk_indirect_args", &shaders.visible_indirect_args)
                .entry_point("compute_chunks")
                .layout(&[&bg_layouts.visible_indirect_args]),
            shadow: device
                .compute_pipeline("shadow", &shaders.shadow)
                .layout(&[&bg_layouts.shadow_static, &bg_layouts.per_frame_shared]),
            ambient: device
                .compute_pipeline("ambient", &shaders.ambient)
                .layout(&[&bg_layouts.ambient_static, &bg_layouts.per_frame_shared]),
            chunk_resolve: device
                .compute_pipeline("chunk_resolve", &shaders.chunk_resolve)
                .layout(&[
                    &bg_layouts.chunk_resolve_static,
                    &bg_layouts.per_frame_shared,
                ]),
            resolve: device
                .compute_pipeline("resolve", &shaders.resolve)
                .layout(&[&bg_layouts.resolve, &bg_layouts.per_frame_shared]),
            ambient_trace: device
                .compute_pipeline("ambient_trace", &shaders.ambient_trace)
                .layout(&[
                    &bg_layouts.ambient_trace_gbuffer,
                    &bg_layouts.specular_swap,
                    &bg_layouts.ambient_trace_static,
                    &bg_layouts.per_frame_shared,
                ]),
            ambient_project: device
                .compute_pipeline("ambient_project", &shaders.ambient_project)
                .layout(&[&bg_layouts.ambient_project_gbuffer]),
            ambient_reproject: device
                .compute_pipeline("ambient_reproject", &shaders.ambient_reproject)
                .layout(&[
                    &bg_layouts.ambient_reproject_gbuffer,
                    &bg_layouts.ambient_reproject_swap,
                    &bg_layouts.per_frame_shared,
                ]),
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
            probe_visualize: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("probe_visualize"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("probe_visualize"),
                        bind_group_layouts: &[
                            &bg_layouts.probe_visualize_swap,
                            &bg_layouts.per_frame_shared,
                        ],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &shaders.probe_visualize,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.probe_visualize,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                cache: None,
                multiview_mask: None,
            }),
        };

        // see build script
        let (scene, tree) = config.init_scene.load(device, queue).unwrap();
        let skybox = config.init_skybox.load(device, queue).unwrap();

        let probe_size = scene
            .meta
            .size
            .map(|x: u32| x.div_ceil(config.irradiance_probe_scale));
        let probe_count = probe_size.element_product();

        let probes = ProbeGrid::new(scene.meta.size, config.irradiance_probe_scale, &tree);

        let textures = Textures {
            probe_irradiance: texture("probe_irradiance")
                .rgba16float()
                .size(glam::uvec3(2048, probe_count.div_ceil(256) * 8, 1))
                .d2()
                .usage(TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING)
                .create(device),
            probe_depth: texture("probe_depth")
                .rg16float()
                .size(glam::uvec3(2048, probe_count.div_ceil(128) * 16, 1))
                .d2()
                .usage(TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING)
                .create(device),
            probe_ray_results: texture("probe_ray_results")
                .rgba16float()
                .size(glam::uvec3(2048, probe_count.div_ceil(128) * 8, 1))
                .d2()
                .usage(TextureUsages::STORAGE_BINDING)
                .create(device),
            noise_coshemi: noise::noise_stbn_coshemi(device, queue).unwrap(),
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

        let buffers = Buffers {
            voxel_scene_metadata: {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct BufferVoxelSceneMetadata {
                    size: glam::UVec3,
                    bounding_size: u32,
                    probe_size: glam::UVec3,
                    probe_scale: f32,
                    index_levels: u32,
                    index_chunk_count: u32,
                    _pad: [u32; 2],
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel_scene_metadata"),
                    contents: bytemuck::cast_slice(&[BufferVoxelSceneMetadata {
                        size: scene.meta.size,
                        bounding_size: scene.meta.bounding_size,
                        probe_size,
                        probe_scale: config.irradiance_probe_scale as f32,
                        index_levels: scene.meta.index_levels,
                        index_chunk_count: scene.meta.allocated_index_chunks,
                        _pad: [0; 2],
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            },
            voxel_palette: scene.data.buffer_palette,
            voxel_index_chunks: scene.data.buffer_index_chunks,
            voxel_leaf_chunks: scene.data.buffer_leaf_chunks,
            voxel_shadow_mask: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel_shadow_mask"),
                size: leaf_chunks_size.div_ceil(32).next_multiple_of(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visibility_info: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visible_info"),
                size: std::mem::size_of::<VisibilityInfoBuffer>() as u64,
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
            chunk_visibility_mask: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk_visibility_mask"),
                size: (scene.meta.allocated_index_chunks as u64).div_ceil(32) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visible_voxels: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visible_voxels"),
                size: voxel_queue_len * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visible_chunks: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visible_chunks"),
                size: (scene.meta.allocated_index_chunks as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            voxel_indirect_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel_indirect_args"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            chunk_indirect_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk_indirect_args"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            probes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("probes"),
                contents: bytemuck::cast_slice(&probes.probes),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            cur_chunk_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cur_chunk_lighting"),
                size: (scene.meta.allocated_index_chunks as u64) * 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            acc_chunk_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("acc_chunk_lighting"),
                size: (scene.meta.allocated_index_chunks as u64) * 12,
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

        dbg!(
            &probe_count,
            &textures.probe_irradiance.size(),
            &textures.probe_depth.size()
        );

        let view_skybox_downsampled =
            skybox
                .downsampled
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("skybox_downsampled"),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
        let view_skybox = skybox.cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("skybox"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let view_skybox_prefilter = skybox.prefilter.create_view(&wgpu::TextureViewDescriptor {
            label: Some("skybox_prefilter"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });

        let bind_groups = BindGroups {
            per_frame_shared: device.bind_group(
                "per_frame_shared",
                &bg_layouts.per_frame_shared,
                [
                    buffers.environment.as_binding(),
                    buffers.frame_metadata.as_binding(),
                    buffers.model.as_binding(),
                ],
            ),
            probe_trace_static: device.bind_group(
                "probe_trace_static",
                &bg_layouts.probe_trace_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_palette.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_leaf_chunks.as_binding(),
                    buffers.voxel_shadow_mask.as_binding(),
                    samplers.linear.as_binding(),
                    view_skybox_downsampled.as_binding(),
                    textures.probe_irradiance.view().as_binding(),
                    textures.probe_depth.view().as_binding(),
                    textures.probe_ray_results.view().as_binding(),
                    buffers.probes.as_binding(),
                ],
            ),
            probe_update_static: device.bind_group(
                "probe_update_static",
                &bg_layouts.probe_update_static,
                [
                    textures.probe_ray_results.view().as_binding(),
                    textures.probe_irradiance.view().as_binding(),
                    textures.probe_depth.view().as_binding(),
                ],
            ),
            raymarch_static: device.bind_group(
                "raymarch_static",
                &bg_layouts.raymarch_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_palette.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_leaf_chunks.as_binding(),
                    textures.noise_coshemi.view().as_binding(),
                    samplers.nearest_repeat.as_binding(),
                    buffers.visibility_info.as_binding(),
                    buffers.voxel_map.as_binding(),
                    buffers.visible_voxels.as_binding(),
                ],
            ),
            voxel_indirect_args: device.bind_group(
                "voxel_indirect_args",
                &bg_layouts.visible_indirect_args,
                [
                    buffers.visibility_info.as_binding(),
                    buffers.voxel_indirect_args.as_binding(),
                ],
            ),
            chunk_indirect_args: device.bind_group(
                "chunk_indirect_args",
                &bg_layouts.visible_indirect_args,
                [
                    buffers.visibility_info.as_binding(),
                    buffers.chunk_indirect_args.as_binding(),
                ],
            ),
            shadow_static: device.bind_group(
                "shadow_static",
                &bg_layouts.shadow_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    textures.noise_coshemi.view().as_binding(),
                    samplers.nearest_repeat.as_binding(),
                    buffers.visible_voxels.as_binding(),
                    buffers.cur_voxel_lighting.as_binding(),
                    buffers.visibility_info.as_binding(),
                    buffers.chunk_visibility_mask.as_binding(),
                    buffers.visible_chunks.as_binding(),
                    buffers.cur_chunk_lighting.as_binding(),
                ],
            ),
            ambient_static: device.bind_group(
                "ambient_static",
                &bg_layouts.ambient_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_palette.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_leaf_chunks.as_binding(),
                    buffers.voxel_shadow_mask.as_binding(),
                    textures.noise_coshemi.view().as_binding(),
                    samplers.nearest_repeat.as_binding(),
                    samplers.linear.as_binding(),
                    view_skybox_downsampled.as_binding(),
                    buffers.visible_voxels.as_binding(),
                    buffers.cur_voxel_lighting.as_binding(),
                    buffers.acc_voxel_lighting.as_binding(),
                    buffers.cur_chunk_lighting.as_binding(),
                    textures.probe_irradiance.view().as_binding(),
                    textures.probe_depth.view().as_binding(),
                    buffers.probes.as_binding(),
                ],
            ),
            chunk_resolve_static: device.bind_group(
                "chunk_resolve_static",
                &bg_layouts.chunk_resolve_static,
                [
                    buffers.visible_chunks.as_binding(),
                    buffers.cur_chunk_lighting.as_binding(),
                    buffers.acc_chunk_lighting.as_binding(),
                ],
            ),
            resolve: device.bind_group(
                "resolve_static",
                &bg_layouts.resolve,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_map.as_binding(),
                    buffers.visible_voxels.as_binding(),
                    buffers.cur_voxel_lighting.as_binding(),
                    buffers.acc_voxel_lighting.as_binding(),
                    buffers.cur_chunk_lighting.as_binding(),
                    buffers.acc_chunk_lighting.as_binding(),
                ],
            ),
            ambient_trace_static: device.bind_group(
                "ambient_trace_static",
                &bg_layouts.ambient_trace_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_palette.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_leaf_chunks.as_binding(),
                    textures.noise_coshemi.view().as_binding(),
                    samplers.linear.as_binding(),
                    view_skybox_downsampled.as_binding(),
                    buffers.voxel_shadow_mask.as_binding(),
                ],
            ),
            specular_static: device.bind_group(
                "specular_static",
                &bg_layouts.specular_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_palette.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_leaf_chunks.as_binding(),
                    textures.noise_coshemi.view().as_binding(),
                    samplers.nearest_repeat.as_binding(),
                    samplers.linear.as_binding(),
                    view_skybox_downsampled.as_binding(),
                    buffers.acc_voxel_lighting.as_binding(),
                ],
            ),
            deferred_static: device.bind_group(
                "deferred_static",
                &bg_layouts.deferred_static,
                [
                    samplers.linear.as_binding(),
                    samplers.nearest_repeat.as_binding(),
                    view_skybox.as_binding(),
                    view_skybox_downsampled.as_binding(),
                    view_skybox_prefilter.as_binding(),
                    skybox.brdf.view().as_binding(),
                    buffers.acc_voxel_lighting.as_binding(),
                    buffers.voxel_shadow_mask.as_binding(),
                    buffers.voxel_scene_metadata.as_binding(),
                    textures.probe_irradiance.view().as_binding(),
                    textures.probe_depth.view().as_binding(),
                    buffers.probes.as_binding(),
                ],
            ),
            fx_settings: device.bind_group(
                "fx_settings",
                &bg_layouts.fx_settings,
                [buffers.fx_settings.as_binding()],
            ),
            shadow_occlusion_static: device.bind_group(
                "shadow_occlusion_static",
                &bg_layouts.shadow_occlusion_static,
                [
                    buffers.voxel_scene_metadata.as_binding(),
                    buffers.voxel_index_chunks.as_binding(),
                    buffers.voxel_shadow_mask.as_binding(),
                ],
            ),
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device, 16));

        let quad = Quad::new(device);
        dbg!(&scene.meta);
        config.voxel_scale = scene.meta.voxels_per_unit;
        ui.debug.voxel_count = scene.meta.voxel_count;
        ui.debug.scene_size = glam::IVec3::splat(scene.meta.bounding_size as i32);

        let mut _self = Self {
            pipelines,
            bg_layouts,
            bind_groups,
            screen_bind_groups: None,
            samplers,
            buffers,
            textures,
            screen_textures: None,
            timing,
            quad,
            frame_id: 0,
            render_scale: 1.0,
            size: Default::default(),
            sun_direction: Default::default(),
            prev_camera: None,
            scene_size: scene.meta.size,
            probe_size,
            shadow_update_requested: true,
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

        let gbuffer = ScreenTextures::create(device, self.size);

        let albedo = gbuffer.albedo.view();
        let voxel_id = gbuffer.voxel_id.view();
        let normal = gbuffer.normal.view();
        let depth = gbuffer.depth.view();
        let velocity = gbuffer.velocity.view();
        let gi_ray_radiance = gbuffer.gi_ray_radiance.view();
        let gi_ray_direction = gbuffer.gi_ray_direction.view();
        let gi_sh_sample_r = gbuffer.gi_sh_sample_rgb[0].view();
        let gi_sh_sample_g = gbuffer.gi_sh_sample_rgb[1].view();
        let gi_sh_sample_b = gbuffer.gi_sh_sample_rgb[2].view();
        let gi_sh_r = gbuffer.gi_sh_rgb[0].view();
        let gi_sh_g = gbuffer.gi_sh_rgb[1].view();
        let gi_sh_b = gbuffer.gi_sh_rgb[2].view();
        let gi_history = gbuffer.gi_history.view();
        let specular = gbuffer.specular.view();
        let specular_direction = gbuffer.specular_direction.view();
        let specular_spatial = gbuffer.specular_spatial.view();
        let acc_specular = gbuffer.acc_specular.view();
        let deferred_output = gbuffer.deferred_output.view();
        let out_color = gbuffer.out_color.view();

        self.screen_textures = Some(gbuffer);

        let layouts = &self.bg_layouts;

        self.screen_bind_groups = Some(ScreenBindGroups {
            raymarch_gbuffer: device.bind_group(
                "raymarch_gbuffer",
                &layouts.raymarch_gbuffer,
                [
                    albedo.as_binding(),
                    velocity.as_binding(),
                    voxel_id.as_binding(),
                ],
            ),
            raymarch_swap: device.bind_group_swap(
                "raymarch_swap",
                &layouts.raymarch_swap,
                [normal.both(), depth.both()],
            ),
            ambient_trace_gbuffer: device.bind_group(
                "ambient_trace_gbuffer",
                &layouts.ambient_trace_gbuffer,
                [gi_ray_radiance.as_binding(), gi_ray_direction.as_binding()],
            ),
            ambient_project_gbuffer: device.bind_group(
                "ambient_project_gbuffer",
                &layouts.ambient_project_gbuffer,
                [
                    gi_ray_radiance.as_binding(),
                    gi_ray_direction.as_binding(),
                    gi_sh_sample_r.as_binding(),
                    gi_sh_sample_g.as_binding(),
                    gi_sh_sample_b.as_binding(),
                ],
            ),
            ambient_reproject_gbuffer: device.bind_group(
                "ambient_reproject_gbuffer",
                &layouts.ambient_reproject_gbuffer,
                [
                    gi_sh_sample_r.as_binding(),
                    gi_sh_sample_g.as_binding(),
                    gi_sh_sample_b.as_binding(),
                    velocity.as_binding(),
                ],
            ),
            ambient_reproject_swap: device.bind_group_swap(
                "ambient_reproject_swap",
                &layouts.ambient_reproject_swap,
                [
                    gi_sh_r.both(),
                    gi_sh_g.both(),
                    gi_sh_b.both(),
                    gi_history.both(),
                    gi_sh_r.both_reversed(),
                    gi_sh_g.both_reversed(),
                    gi_sh_b.both_reversed(),
                    gi_history.both_reversed(),
                    normal.both(),
                    depth.both(),
                    normal.both_reversed(),
                    depth.both_reversed(),
                ],
            ),
            specular_gbuffer: device.bind_group(
                "specular_gbuffer",
                &layouts.specular_gbuffer,
                [specular.as_binding(), specular_direction.as_binding()],
            ),
            specular_swap: device.bind_group_swap(
                "specular_swap",
                &layouts.specular_swap,
                [normal.both(), depth.both()],
            ),
            spec_spatial_gbuffer: device.bind_group(
                "specular_spatial_gbuffer",
                &layouts.spec_spatial_gbuffer,
                [
                    specular.as_binding(),
                    specular_direction.as_binding(),
                    velocity.as_binding(),
                ],
            ),
            spec_resolve_gbuffer: device.bind_group(
                "specular_resolve_gbuffer",
                &layouts.spec_resolve_gbuffer,
                [
                    self.samplers.linear.as_binding(),
                    specular_spatial.as_binding(),
                    specular_direction.as_binding(),
                    velocity.as_binding(),
                ],
            ),
            spec_resolve_swap: device.bind_group_swap(
                "specular_resolve_swap",
                &layouts.spec_resolve_swap,
                [
                    acc_specular.both(),
                    acc_specular.both_reversed(),
                    normal.both(),
                    normal.both_reversed(),
                    depth.both(),
                    depth.both_reversed(),
                ],
            ),
            deferred_gbuffer: device.bind_group(
                "deferred_gbuffer",
                &layouts.deferred_gbuffer,
                [
                    deferred_output.as_binding(),
                    albedo.as_binding(),
                    velocity.as_binding(),
                    voxel_id.as_binding(),
                ],
            ),
            deferred_swap: device.bind_group_swap(
                "deferred_swap",
                &layouts.deferred_swap,
                [
                    normal.both(),
                    depth.both(),
                    acc_specular.both(),
                    gi_sh_r.both(),
                    gi_sh_g.both(),
                    gi_sh_b.both(),
                    gi_history.both(),
                ],
            ),
            taa_input: device.bind_group(
                "taa_input",
                &layouts.taa_input,
                [
                    self.samplers.linear.as_binding(),
                    velocity.as_binding(),
                    deferred_output.as_binding(),
                ],
            ),
            taa_swap: device.bind_group_swap(
                "taa_swap",
                &layouts.taa_output,
                [out_color.both_reversed(), out_color.both(), depth.both()],
            ),
            fx_input_swap: device.bind_group_swap(
                "fx_input_swap",
                &layouts.fx_input,
                [
                    out_color.both(),
                    SwapchainBindingResource::Single(self.samplers.linear.as_binding()),
                    SwapchainBindingResource::Single(
                        self.textures.tonemap_mcmapface_lut.view().as_binding(),
                    ),
                ],
            ),
            probe_visualize_swap: device.bind_group_swap(
                "probe_visualize_swap",
                &layouts.probe_visualize_swap,
                [
                    SwapchainBindingResource::Single(
                        self.buffers.voxel_scene_metadata.as_binding(),
                    ),
                    depth.both(),
                    SwapchainBindingResource::Single(self.buffers.probes.as_binding()),
                ],
            ),
        });
    }

    pub fn fixed_update(&mut self, ctx: RendererCtx<'_>) {
        let RendererCtx {
            device,
            queue,
            ui,
            config,
            ..
        } = ctx;

        if !config.print_debug_info {
            return;
        }

        #[derive(Debug)]
        #[allow(unused)]
        struct DebugStats {
            fps: f64,
            frame: Duration,
        }

        let stats = DebugStats {
            fps: 1.0 / ui.debug.frame_avg.as_secs_f64(),
            frame: ui.debug.frame_avg,
        };
        eprintln!("{:?}", stats);

        let buffer_results = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.buffers.visibility_info.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.buffers.visibility_info,
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
        let data: &[VisibilityInfoBuffer] = bytemuck::cast_slice(&data);
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
            if self.sun_direction.distance_squared(ui.debug.sun_direction) > 0.001 {
                self.shadow_update_requested = true;
                println!("update shadow occlusion");
            }
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
                    per_voxel_secondary: config.per_voxel_secondary as u32,
                    shadow_filter_radius: config.shadow_filter_radius,
                    max_ambient_distance: config.ambient_ray_max_distance,
                    voxel_normal_factor: config.voxel_normal_factor,
                    roughness_multiplier: config.roughness_multiplier,
                    indirect_sky_intensity: config.indirect_sky_intensity,
                    debug_view: config.view as u32,
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

        encoder.clear_buffer(&self.buffers.visibility_info, 0, None);
        encoder.clear_buffer(&self.buffers.voxel_map, 0, None);
        encoder.clear_buffer(&self.buffers.chunk_visibility_mask, 0, None);
        encoder.clear_buffer(&self.buffers.cur_chunk_lighting, 0, None);

        // shadow occlusion pass
        if self.shadow_update_requested {
            self.shadow_update_requested = false;

            // encoder.clear_buffer(&self.buffers.acc_voxel_lighting, 0, None);
            encoder.clear_buffer(&self.buffers.voxel_shadow_mask, 0, None);
            // let mut pass = encoder.begin_compute_pass_timed("Shadow Occlusion", &mut self.timing);
            let mut pass = encoder.begin_compute_pass(&Default::default());

            pass.set_pipeline(&self.pipelines.shadow_occlusion);
            pass.set_bind_group(0, &self.bind_groups.shadow_occlusion_static, &[]);
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("shadow occlusion");

            let group_size = self.scene_size.map(|x| x.div_ceil(8).max(2));
            let group_count = 4
                * (group_size.x * group_size.y
                    + group_size.x * group_size.z
                    + group_size.y * group_size.z);
            let wg_x = group_count.min(65535);
            let wg_y = group_count.div_ceil(wg_x);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // // probe trace pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Probe Trace", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.probe_trace);
        //     pass.set_bind_group(0, &self.bind_groups.probe_trace_static, &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("probe trace");

        //     pass.dispatch_workgroups(self.probe_size.x, self.probe_size.y, self.probe_size.z);
        // }

        // // probe update pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Probe Update", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.probe_update);
        //     pass.set_bind_group(0, &self.bind_groups.probe_update_static, &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("probe update");

        //     pass.dispatch_workgroups(self.probe_size.x, self.probe_size.y, self.probe_size.z);
        // }

        // // probe border update pass
        // {
        //     let mut pass =
        //         encoder.begin_compute_pass_timed("Probe Border Update", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.probe_border);
        //     pass.set_bind_group(0, &self.bind_groups.probe_update_static, &[]);

        //     pass.insert_debug_marker("probe border");

        //     pass.dispatch_workgroups(self.probe_size.x, self.probe_size.y, self.probe_size.z);
        // }
        let Some(screen_bind_groups) = &self.screen_bind_groups else {
            return;
        };

        // raymarch pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Raymarch", &mut self.timing);

            pass.set_pipeline(&self.pipelines.raymarch);
            pass.set_bind_group(0, &screen_bind_groups.raymarch_gbuffer, &[]);
            pass.set_bind_group_swap(1, &screen_bind_groups.raymarch_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.raymarch_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("raymarch");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // ambient trace pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Ambient Trace", &mut self.timing);

            pass.set_pipeline(&self.pipelines.ambient_trace);
            pass.set_bind_group(0, &screen_bind_groups.ambient_trace_gbuffer, &[]);
            pass.set_bind_group_swap(1, &screen_bind_groups.specular_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.ambient_trace_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            let size_quarter = self.size.map(|x| x.div_ceil(4));

            pass.insert_debug_marker("ambient trace");
            pass.dispatch_workgroups(size_quarter.x.div_ceil(8), size_quarter.y.div_ceil(8), 1);
        }

        // ambient project pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Ambient Project", &mut self.timing);

            pass.set_pipeline(&self.pipelines.ambient_project);
            pass.set_bind_group(0, &screen_bind_groups.ambient_project_gbuffer, &[]);

            let size_half = self.size.map(|x| x.div_ceil(2));

            pass.insert_debug_marker("ambient project");
            pass.dispatch_workgroups(size_half.x.div_ceil(8), size_half.y.div_ceil(8), 1);
        }

        // ambient reproject pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Ambient Reproject", &mut self.timing);

            pass.set_pipeline(&self.pipelines.ambient_reproject);
            pass.set_bind_group(0, &screen_bind_groups.ambient_reproject_gbuffer, &[]);
            pass.set_bind_group_swap(
                1,
                &screen_bind_groups.ambient_reproject_swap,
                &[],
                self.frame_id,
            );
            pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

            let size_half = self.size.map(|x| x.div_ceil(2));

            pass.insert_debug_marker("ambient reproject");
            pass.dispatch_workgroups(size_half.x.div_ceil(8), size_half.y.div_ceil(8), 1);
        }

        // // copy over indirect args for the per-voxel dispatches
        // {
        //     let mut pass = encoder.begin_compute_pass(&Default::default());

        //     pass.set_pipeline(&self.pipelines.voxel_indirect_args);
        //     pass.set_bind_group(0, &self.bind_groups.voxel_indirect_args, &[]);

        //     pass.insert_debug_marker("voxel indirect args");
        //     pass.dispatch_workgroups(1, 1, 1);
        // }

        // // shadow pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Shadow", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.shadow);
        //     pass.set_bind_group(0, &self.bind_groups.shadow_static, &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("shadow");
        //     pass.dispatch_workgroups_indirect(&self.buffers.voxel_indirect_args, 0);
        // }

        // // copy over indirect args for the per-chunk dispatches
        // {
        //     let mut pass = encoder.begin_compute_pass(&Default::default());

        //     pass.set_pipeline(&self.pipelines.chunk_indirect_args);
        //     pass.set_bind_group(0, &self.bind_groups.chunk_indirect_args, &[]);

        //     pass.insert_debug_marker("chunk indirect args");
        //     pass.dispatch_workgroups(1, 1, 1);
        // }

        // // ambient pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Ambient", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.ambient);
        //     pass.set_bind_group(0, &self.bind_groups.ambient_static, &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("ambient");
        //     pass.dispatch_workgroups_indirect(&self.buffers.voxel_indirect_args, 0);
        // }

        // // chunk resolve pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Chunk Resolve", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.chunk_resolve);
        //     pass.set_bind_group(0, &self.bind_groups.chunk_resolve_static, &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("chunk resolve");
        //     pass.dispatch_workgroups_indirect(&self.buffers.chunk_indirect_args, 0);
        // }

        // // resolve pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Resolve", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.resolve);
        //     pass.set_bind_group(0, Some(&self.bind_groups.resolve), &[]);
        //     pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("resolve");
        //     pass.dispatch_workgroups_indirect(&self.buffers.voxel_indirect_args, 0);
        // }

        // // specular pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Specular", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.specular);
        //     pass.set_bind_group(0, &self.bind_groups.specular_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.specular_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.specular_static, &[]);
        //     pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

        //     let size_quarter = self.size.map(|x| x.div_ceil(2));

        //     pass.insert_debug_marker("specular");
        //     pass.dispatch_workgroups(size_quarter.x.div_ceil(8), size_quarter.y.div_ceil(8), 1);
        // }

        // // specular spatial filter pass
        // {
        //     let mut pass =
        //         encoder.begin_compute_pass_timed("Specular Spatial Filter", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.specular_spatial);
        //     pass.set_bind_group(0, &self.bind_groups.spec_spatial_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.specular_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("specular spatial");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        // }

        // // specular resolve pass
        // {
        //     let mut pass = encoder.begin_compute_pass_timed("Specular Resolve", &mut self.timing);

        //     pass.set_pipeline(&self.pipelines.specular_resolve);
        //     pass.set_bind_group(0, &self.bind_groups.spec_resolve_gbuffer, &[]);
        //     pass.set_bind_group_swap(1, &self.bind_groups.spec_resolve_swap, &[], self.frame_id);
        //     pass.set_bind_group(2, &self.bind_groups.per_frame_shared, &[]);

        //     pass.insert_debug_marker("specular resolve");
        //     pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        // }

        // deferred pass
        {
            let mut pass = encoder.begin_compute_pass_timed("Deferred", &mut self.timing);

            pass.set_pipeline(&self.pipelines.deferred);
            pass.set_bind_group(0, &screen_bind_groups.deferred_gbuffer, &[]);
            pass.set_bind_group_swap(1, &screen_bind_groups.deferred_swap, &[], self.frame_id);
            pass.set_bind_group(2, &self.bind_groups.deferred_static, &[]);
            pass.set_bind_group(3, &self.bind_groups.per_frame_shared, &[]);

            pass.insert_debug_marker("deferred");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // probe visualize pass
        if config.display_probes {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("probe_visualize"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self
                        .screen_textures
                        .as_ref()
                        .unwrap()
                        .deferred_output
                        .view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.probe_visualize);
            pass.set_bind_group_swap(
                0,
                &screen_bind_groups.probe_visualize_swap,
                &[],
                self.frame_id,
            );
            pass.set_bind_group(1, &self.bind_groups.per_frame_shared, &[]);

            pass.draw(0..6, 0..(self.probe_size.element_product()));
        }

        // taa pass
        {
            let mut pass = encoder.begin_compute_pass_timed("TAA", &mut self.timing);

            pass.set_pipeline(&self.pipelines.taa);
            pass.set_bind_group(0, &self.bind_groups.per_frame_shared, &[]);
            pass.set_bind_group(1, &screen_bind_groups.taa_input, &[]);
            pass.set_bind_group_swap(2, &screen_bind_groups.taa_swap, &[], self.frame_id);

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
            pass.set_bind_group_swap(0, &screen_bind_groups.fx_input_swap, &[], self.frame_id);
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

use anyhow::Result;
use image::EncodableLayout;
use std::{
    array,
    io::{BufRead, Seek},
};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct LightingResult {
    pub cubemap: wgpu::Texture,
    pub irradiance: wgpu::Texture,
    pub prefilter: wgpu::Texture,
    pub brdf: wgpu::Texture,
}

struct BindGroupLayouts {
    cubemap: wgpu::BindGroupLayout,
    irradiance: wgpu::BindGroupLayout,
    prefilter: wgpu::BindGroupLayout,
    brdf: wgpu::BindGroupLayout,
}

struct BindGroups {
    cubemap: wgpu::BindGroup,
    irradiance: wgpu::BindGroup,
    prefilter: [wgpu::BindGroup; PREFILTER_LEVELS as usize],
    brdf: wgpu::BindGroup,
}

struct Pipelines {
    cubemap: wgpu::ComputePipeline,
    irradiance: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
    brdf: wgpu::ComputePipeline,
}

struct Resources {
    sampler: wgpu::Sampler,
    tex_hdr: wgpu::Texture,
    tex_cubemap: wgpu::Texture,
    tex_irradiance: wgpu::Texture,
    tex_prefilter: wgpu::Texture,
    tex_brdf_lut: wgpu::Texture,
}

const SKYBOX_RESOLUTION: u32 = 2048;
const SKYBOX_MAX_COMPONENT: f32 = 50.0;
const IRRADIANCE_RESOLUTION: u32 = 128;
const IRRADIANCE_DELTA: f64 = 0.01;
const PREFILTER_RESOLUTION: u32 = 256;
const PREFILTER_LEVELS: u32 = 5;
const PREFILTER_SAMPLES: u32 = 4096;
const BRDF_LUT_RESOLUTION: u32 = 1024;
const BRDF_LUT_SAMPLES: u32 = 2048;

pub fn generate_lighting<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<LightingResult> {
    let tex_hdr = load_hdr(src, device, queue)?;

    let bg_layouts = BindGroupLayouts {
        cubemap: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cubemap"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }),
        irradiance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("irradiance"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }),
        prefilter: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefilter"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        brdf: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brdf"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        }),
    };

    let pipelines = Pipelines {
        cubemap: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cubemap"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cubemap"),
                    bind_group_layouts: &[&bg_layouts.cubemap],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cubemap"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/cubemap.wgsl").into()),
            }),
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        }),
        irradiance: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("irradiance"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("irradiance"),
                    bind_group_layouts: &[&bg_layouts.irradiance],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("irradiance"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("shaders/irradiance.wgsl").into(),
                ),
            }),
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("delta", IRRADIANCE_DELTA)],
                ..Default::default()
            },
            cache: None,
        }),
        prefilter: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefilter"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("prefilter"),
                    bind_group_layouts: &[&bg_layouts.prefilter],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prefilter"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("shaders/prefilter.wgsl").into(),
                ),
            }),
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        }),
        brdf: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("brdf"),
                    bind_group_layouts: &[&bg_layouts.brdf],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("brdf"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/brdf.wgsl").into()),
            }),
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("sample_count", BRDF_LUT_SAMPLES as f64)],
                ..Default::default()
            },
            cache: None,
        }),
    };

    let resources = Resources {
        sampler: device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
        tex_hdr: tex_hdr,
        tex_cubemap: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cubemap"),
            size: wgpu::Extent3d {
                width: SKYBOX_RESOLUTION,
                height: SKYBOX_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_irradiance: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance"),
            size: wgpu::Extent3d {
                width: IRRADIANCE_RESOLUTION,
                height: IRRADIANCE_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_prefilter: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefilter"),
            size: wgpu::Extent3d {
                width: PREFILTER_RESOLUTION,
                height: PREFILTER_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: PREFILTER_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_brdf_lut: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_RESOLUTION,
                height: BRDF_LUT_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
    };

    let bind_groups = BindGroups {
        cubemap: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cubemap"),
            layout: &bg_layouts.cubemap,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_hdr.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_cubemap.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        }),
        irradiance: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("irradiance"),
            layout: &bg_layouts.irradiance,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &resources
                            .tex_cubemap
                            .create_view(&wgpu::TextureViewDescriptor {
                                label: Some("skybox"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            }),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_irradiance.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        }),
        prefilter: array::from_fn(|i| {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PrefilterData {
                roughness: f32,
                sample_count: u32,
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefilter"),
                contents: bytemuck::cast_slice(&[PrefilterData {
                    roughness: (i as f32) / (PREFILTER_LEVELS as f32 - 1.0),
                    sample_count: PREFILTER_SAMPLES,
                }]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("prefilter_{}", i)),
                layout: &bg_layouts.prefilter,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &resources
                                .tex_cubemap
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("skybox"),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &resources
                                .tex_prefilter
                                .create_view(&wgpu::TextureViewDescriptor {
                                    base_mip_level: i as u32,
                                    mip_level_count: Some(1),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            })
        }),
        brdf: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brdf"),
            layout: &bg_layouts.brdf,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &resources.tex_brdf_lut.create_view(&Default::default()),
                ),
            }],
        }),
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cubemap"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cubemap"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.cubemap);
        pass.set_bind_group(0, &bind_groups.cubemap, &[]);
        pass.insert_debug_marker("cubemap");
        pass.dispatch_workgroups(
            SKYBOX_RESOLUTION.div_ceil(8),
            SKYBOX_RESOLUTION.div_ceil(8),
            6,
        );
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("irradiance"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.irradiance);
        pass.set_bind_group(0, &bind_groups.irradiance, &[]);
        pass.insert_debug_marker("irradiance");
        pass.dispatch_workgroups(
            IRRADIANCE_RESOLUTION.div_ceil(8),
            IRRADIANCE_RESOLUTION.div_ceil(8),
            6,
        );
    }

    for i in 0..PREFILTER_LEVELS {
        let mip_resolution = PREFILTER_RESOLUTION >> i;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("prefilter_{}", i)),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.prefilter);
        pass.set_bind_group(0, &bind_groups.prefilter[i as usize], &[]);
        pass.insert_debug_marker("prefilter");
        pass.dispatch_workgroups(mip_resolution.div_ceil(8), mip_resolution.div_ceil(8), 6);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brdf"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.brdf);
        pass.set_bind_group(0, &bind_groups.brdf, &[]);
        pass.insert_debug_marker("brdf");
        pass.dispatch_workgroups(
            BRDF_LUT_RESOLUTION.div_ceil(8),
            BRDF_LUT_RESOLUTION.div_ceil(8),
            1,
        );
    }

    queue.submit([encoder.finish()]);

    Ok(LightingResult {
        cubemap: resources.tex_cubemap,
        irradiance: resources.tex_irradiance,
        prefilter: resources.tex_prefilter,
        brdf: resources.tex_brdf_lut,
    })
}

fn load_hdr<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
    let img = image::load(src, image::ImageFormat::Hdr)?;
    let mut src = img.into_rgba32f();
    src.pixels_mut().for_each(|pixel| {
        pixel.0[0] = pixel.0[0].min(SKYBOX_MAX_COMPONENT);
        pixel.0[1] = pixel.0[1].min(SKYBOX_MAX_COMPONENT);
        pixel.0[2] = pixel.0[2].min(SKYBOX_MAX_COMPONENT);
    });

    let tex_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr"),
        size: wgpu::Extent3d {
            width: src.width(),
            height: src.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_hdr,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &src.as_bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16 * src.width()),
            rows_per_image: Some(src.height()),
        },
        wgpu::Extent3d {
            width: src.width(),
            height: src.height(),
            depth_or_array_layers: 1,
        },
    );

    Ok(tex_hdr)
}

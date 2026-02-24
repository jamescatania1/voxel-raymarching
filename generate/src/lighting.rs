use anyhow::Result;
use image::EncodableLayout;
use std::io::{BufRead, Seek};

struct BindGroupLayouts {
    cubemap: wgpu::BindGroupLayout,
}

struct BindGroups {
    cubemap: wgpu::BindGroup,
}

struct Pipelines {
    cubemap: wgpu::ComputePipeline,
}

struct Resources {
    tex_hdr: wgpu::Texture,
    tex_cubemap: wgpu::Texture,
}

const SKYBOX_RESOLUTION: u32 = 2048;

pub fn generate_lighting<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
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
    };

    let tex_hdr = load_hdr(src, device, queue)?;
    let tex_cubemap = device.create_texture(&wgpu::TextureDescriptor {
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
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_groups = BindGroups {
        cubemap: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cubemap"),
            layout: &bg_layouts.cubemap,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &tex_hdr.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &tex_cubemap.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
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

    queue.submit([encoder.finish()]);

    Ok(tex_cubemap)
}

fn load_hdr<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
    let src = image::load(src, image::ImageFormat::Hdr)?.into_rgba32f();
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

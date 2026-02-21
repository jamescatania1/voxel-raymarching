use std::{
    io::{BufRead, Read, Seek},
    time::Duration,
};

use crate::gltf::{self, Gltf, Scene};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

const VOXELS_PER_UNIT: u32 = 25;

pub fn voxelize<R: BufRead + Seek>(
    src: &mut R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: Option<String>,
) -> Result<VoxelModel> {
    let gltf = Gltf::parse(src)?;
    let scene = Scene::from_gltf(&gltf)?;
    let mut voxelizer = Voxelizer::new(device, scene.textures.len() as u32);
    voxelizer.load_gltf(device, queue, gltf, scene, name)
}

pub fn load_voxel_header<R: Read>(data: &mut R) -> Result<VoxelMetadata> {
    let mut buf = [0; 4];
    data.read_exact(&mut buf)?;
    let header_length = bytemuck::cast_slice::<u8, u32>(&buf)[0] as usize;
    let mut buf = vec![0; header_length as usize];
    data.read_exact(&mut buf)?;
    let header: VoxelFileHeader = serde_json::from_slice(&buf)?;
    Ok(header.meta)
}

pub struct VoxelModel {
    pub meta: VoxelMetadata,
    pub data: VoxelBufferData,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoxelMetadata {
    pub name: String,
    pub size_voxels: glam::UVec3,
    pub size_chunks: glam::UVec3,
    pub voxel_count: u32,
    pub allocated_chunks: u32,
    pub allocated_brick_slices: u32,
}

impl VoxelModel {
    pub fn serialize(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
        let buffer_chunk_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_chunk_indices"),
            size: self.data.buffer_chunk_indices.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_chunks"),
            size: self.data.buffer_chunks.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_palette = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_palette"),
            size: self.data.buffer_palette.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tex_brickmap_size = glam::uvec3(
            self.data.tex_brickmap.width(),
            self.data.tex_brickmap.height(),
            self.data.tex_brickmap.depth_or_array_layers(),
        );
        let tex_brickmap_bytes_per_row_padded = (tex_brickmap_size.x * 4 + 255) & !255;
        let tex_brickmap_bytes_padded = (tex_brickmap_bytes_per_row_padded as u64)
            * (tex_brickmap_size.y as u64)
            * (tex_brickmap_size.z as u64);

        let tex_brickmap = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_brickmap"),
            size: tex_brickmap_bytes_padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_chunk_indices,
            0,
            &buffer_chunk_indices,
            0,
            buffer_chunk_indices.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_chunks,
            0,
            &buffer_chunks,
            0,
            buffer_chunks.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_palette,
            0,
            &buffer_palette,
            0,
            buffer_palette.size(),
        );
        encoder.copy_texture_to_buffer(
            self.data.tex_brickmap.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &tex_brickmap,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_brickmap_bytes_per_row_padded),
                    rows_per_image: Some(tex_brickmap_size.y),
                },
            },
            self.data.tex_brickmap.size(),
        );
        queue.submit([encoder.finish()]);

        let buffer_chunk_indices = buffer_chunk_indices.slice(..);
        buffer_chunk_indices.map_async(wgpu::MapMode::Read, |_| {});

        let buffer_chunks = buffer_chunks.slice(..);
        buffer_chunks.map_async(wgpu::MapMode::Read, |_| {});

        let buffer_palette = buffer_palette.slice(..);
        buffer_palette.map_async(wgpu::MapMode::Read, |_| {});

        let tex_brickmap = tex_brickmap.slice(..);
        tex_brickmap.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let mut body = Vec::new();

        let data_chunk_indices = buffer_chunk_indices.get_mapped_range();
        let buffer_chunk_indices = BufferView {
            start: body.len(),
            end: body.len() + data_chunk_indices.len(),
        };
        body.extend_from_slice(&data_chunk_indices);

        let data_chunks = buffer_chunks.get_mapped_range();
        let buffer_chunks = BufferView {
            start: body.len(),
            end: body.len() + data_chunks.len(),
        };
        body.extend_from_slice(&data_chunks);

        let data_palette = buffer_palette.get_mapped_range();
        let buffer_palette = BufferView {
            start: body.len(),
            end: body.len() + data_palette.len(),
        };
        body.extend_from_slice(&data_palette);

        let data_brickmap = tex_brickmap.get_mapped_range();
        let tex_brickmap = BufferView {
            start: body.len(),
            end: body.len() + data_brickmap.len(),
        };
        body.extend_from_slice(&data_brickmap);

        let header = serde_json::to_vec(&VoxelFileHeader {
            meta: self.meta.clone(),
            file: VoxelFileInfo {
                buffer_chunk_indices,
                buffer_chunks,
                buffer_palette,
                tex_brickmap,
                tex_brickmap_size,
                tex_brickmap_bytes_per_row_padded,
            },
        })?;
        let mut data: Vec<u8> = bytemuck::cast_slice(&[header.len() as u32]).to_vec();
        data.extend_from_slice(&header);
        data.extend_from_slice(&body);

        Ok(data)
    }

    pub fn deserialize(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            bail!("Invalid model")
        }
        let header_length = bytemuck::cast_slice::<u8, u32>(&data[0..4])[0] as usize;

        let header: VoxelFileHeader = serde_json::from_slice(&data[4..(4 + header_length)])?;
        let buf = &data[4 + header_length..];

        let buffer_chunk_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_chunk_indices", header.meta.name)),
            contents: &buf
                [header.file.buffer_chunk_indices.start..header.file.buffer_chunk_indices.end],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_chunks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_chunks", header.meta.name)),
            contents: &buf[header.file.buffer_chunks.start..header.file.buffer_chunks.end],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_palette", header.meta.name)),
            contents: &buf[header.file.buffer_palette.start..header.file.buffer_palette.end],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tex_brickmap_size = wgpu::Extent3d {
            width: header.file.tex_brickmap_size.x,
            height: header.file.tex_brickmap_size.y,
            depth_or_array_layers: header.file.tex_brickmap_size.z,
        };
        let tex_brickmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{}_tex_brickmap", header.meta.name)),
            size: tex_brickmap_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex_brickmap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &buf[header.file.tex_brickmap.start..header.file.tex_brickmap.end],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(header.file.tex_brickmap_bytes_per_row_padded),
                rows_per_image: Some(header.file.tex_brickmap_size.y),
            },
            tex_brickmap_size,
        );

        Ok(Self {
            meta: header.meta,
            data: VoxelBufferData {
                buffer_palette,
                buffer_chunk_indices,
                buffer_chunks,
                tex_brickmap,
            },
        })
    }
}

pub struct VoxelBufferData {
    pub buffer_palette: wgpu::Buffer,
    pub buffer_chunk_indices: wgpu::Buffer,
    pub buffer_chunks: wgpu::Buffer,
    pub tex_brickmap: wgpu::Texture,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct VoxelFileHeader {
    meta: VoxelMetadata,
    file: VoxelFileInfo,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct VoxelFileInfo {
    buffer_chunk_indices: BufferView,
    buffer_chunks: BufferView,
    buffer_palette: BufferView,
    tex_brickmap: BufferView,
    tex_brickmap_size: glam::UVec3,
    tex_brickmap_bytes_per_row_padded: u32,
}
#[derive(Serialize, Deserialize, Debug)]
struct BufferView {
    start: usize,
    end: usize,
}

pub struct Voxelizer {
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
}
struct Pipelines {
    voxelize: wgpu::ComputePipeline,
    // palette: wgpu::ComputePipeline,
    tree: wgpu::ComputePipeline,
}
struct BindGroupLayouts {
    voxelize_shared: wgpu::BindGroupLayout,
    voxelize_scene_textures: wgpu::BindGroupLayout,
    voxelize_per_primitive: wgpu::BindGroupLayout,
    palette: wgpu::BindGroupLayout,
    tree_data: wgpu::BindGroupLayout,
    tree_alloc_texture: wgpu::BindGroupLayout,
}

impl Voxelizer {
    pub fn new(device: &wgpu::Device, scene_tex_count: u32) -> Self {
        let bg_layouts = BindGroupLayouts {
            voxelize_shared: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shared voxelizer bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rg32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            }),
            // have to separate this from the shared layout due to wgpu restriction
            voxelize_scene_textures: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("scene textures bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: std::num::NonZeroU32::new(scene_tex_count),
                    }],
                },
            ),
            voxelize_per_primitive: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("per-primitive voxelizer bind group layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ),
            palette: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("palette data layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
            tree_data: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel brickmap data layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            }),
            tree_alloc_texture: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel brickmap input alloc and texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rg32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
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
        };
        let pipelines = Pipelines {
            voxelize: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxelize pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("voxelizer pipeline layout"),
                        bind_group_layouts: &[
                            &bg_layouts.voxelize_shared,
                            &bg_layouts.voxelize_scene_textures,
                            &bg_layouts.voxelize_per_primitive,
                        ],
                        immediate_size: 0,
                    }),
                ),
                module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxelize"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("shaders/voxelize.wgsl").into(),
                    ),
                }),
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            tree: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("voxel packing pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("voxel brickmap packing"),
                        bind_group_layouts: &[
                            &bg_layouts.tree_data,
                            &bg_layouts.tree_alloc_texture,
                        ],
                        immediate_size: 0,
                    }),
                ),
                module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxel brickmap packing"),
                    source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/tree.wgsl").into()),
                }),
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        };

        Self {
            pipelines,
            bg_layouts,
        }
    }

    pub fn load_gltf(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gltf: Gltf,
        scene: Scene,
        name: Option<String>,
    ) -> Result<VoxelModel> {
        let voxelizer = Self::new(device, scene.textures.len() as u32);

        let name = name.unwrap_or(String::from("model"));

        let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_ivec3();
        let size_voxels = size_unscaled.as_uvec3() * VOXELS_PER_UNIT;
        let size_chunks = (size_voxels + 7) / 8;

        // texture of raw 3d voxels
        let voxels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel texture"),
            size: wgpu::Extent3d {
                width: size_voxels.x,
                height: size_voxels.y,
                depth_or_array_layers: size_voxels.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        let palette = voxelizer.build_palette(device, queue, &mut encoder);
        voxelizer.voxelize(device, queue, &mut encoder, &voxels, &scene, &gltf);
        let brickmap = voxelizer.build_tree(&palette, device, &mut encoder, &voxels, size_chunks);

        queue.submit([encoder.finish()]);

        let alloc_results = {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct AllocatorResults {
                chunk_count: u32,
                voxel_count: u32,
            }
            brickmap
                .buffer_alloc_results
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(20)),
                })
                .unwrap();
            let view = brickmap.buffer_alloc_results.get_mapped_range(..);
            bytemuck::cast_slice::<_, AllocatorResults>(&*view)[0]
        };

        let voxel_count = alloc_results.voxel_count;
        let allocated_chunks = alloc_results.chunk_count;
        let allocated_brick_slices = allocated_chunks.div_ceil(size_chunks.x * size_chunks.y);

        // these ops rely on reading back the atomic counter data, hence the new encoder
        let mut encoder = device.create_command_encoder(&Default::default());

        // shrink the brickmap texture to have the least z layers possible for raymarching
        // in the future, once I'm adding edits this probably can't happen, i'll have to make the allocator
        // request different sectors of brickmap texture as needed
        let voxels_cpct = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel texture"),
            size: wgpu::Extent3d {
                width: size_chunks.x << 3,
                height: size_chunks.y << 3,
                depth_or_array_layers: allocated_brick_slices << 3,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &brickmap.tex_brickmap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfoBase {
                texture: &voxels_cpct,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: size_chunks.x << 3,
                height: size_chunks.y << 3,
                depth_or_array_layers: allocated_brick_slices << 3,
            },
        );
        queue.submit([encoder.finish()]);

        Ok(VoxelModel {
            meta: VoxelMetadata {
                name,
                size_voxels,
                size_chunks,
                voxel_count,
                allocated_chunks,
                allocated_brick_slices,
            },
            data: VoxelBufferData {
                buffer_palette: palette.buffer_palette,
                buffer_chunk_indices: brickmap.buffer_chunk_indices,
                buffer_chunks: brickmap.buffer_chunks,
                tex_brickmap: voxels_cpct,
            },
        })
    }
}

struct BrickmapData {
    buffer_chunk_indices: wgpu::Buffer,
    buffer_chunks: wgpu::Buffer,
    tex_brickmap: wgpu::Texture,
    buffer_alloc_results: wgpu::Buffer,
}
struct PaletteData {
    buffer_palette: wgpu::Buffer,
}

impl Voxelizer {
    fn voxelize(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        brickmap: &wgpu::Texture,
        scene: &Scene,
        gltf: &Gltf,
    ) {
        let base_unscaled = scene.min.floor().as_ivec3();
        let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_ivec3();

        let bg_voxelize_shared = {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct SceneBufferEntry {
                base: glam::Vec4,
                size: glam::Vec3,
                scale: f32,
            }
            // binding 0
            let buffer_scene = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material data storage buffer"),
                contents: bytemuck::cast_slice(&[SceneBufferEntry {
                    base: base_unscaled.as_vec3().extend(0.0),
                    size: size_unscaled.as_vec3(),
                    scale: VOXELS_PER_UNIT as f32,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct MaterialBufferEntry {
                base_albedo: glam::Vec4,
                metallic: f32,
                roughness: f32,
                normal_scale: f32,
                albedo_index: i32,
                normal_index: i32,
                double_sided: u32,
                _pad: [f32; 2],
            }
            // binding 1
            let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material data storage buffer"),
                contents: bytemuck::cast_slice(
                    &scene
                        .materials
                        .iter()
                        .map(|mat| MaterialBufferEntry {
                            base_albedo: mat.base_albedo,
                            metallic: mat.metallic,
                            roughness: mat.roughness,
                            normal_scale: mat.normal_scale,
                            albedo_index: mat.albedo_index,
                            normal_index: mat.normal_index,
                            double_sided: mat.double_sided as u32,
                            _pad: [0.0; 2],
                        })
                        .collect::<Vec<MaterialBufferEntry>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            // binding 2
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shared voxelizer bind group"),
                layout: &self.bg_layouts.voxelize_shared,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer_scene.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer_materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&brickmap.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("voxelized result texture output view"),
                                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                                ..Default::default()
                            },
                        )),
                    },
                ],
            })
        };

        let bg_voxelize_scene_textures = {
            let texture_views = scene
                .textures
                .iter()
                .map(|tex| {
                    let (width, height) = tex.data.dimensions();
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
                        format: match tex.encoding {
                            gltf::scene::TextureEncoding::Linear => wgpu::TextureFormat::Rgba8Unorm,
                            gltf::scene::TextureEncoding::Srgb => wgpu::TextureFormat::Rgba8Unorm,
                        },
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
                        &tex.data,
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
                .collect::<Vec<wgpu::TextureView>>();

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene textures bind group"),
                layout: &self.bg_layouts.voxelize_scene_textures,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(
                        &texture_views.iter().collect::<Vec<&wgpu::TextureView>>(),
                    ),
                }],
            })
        };

        struct PrimitiveGroup {
            bg: wgpu::BindGroup,
            index_count: u32,
        }
        let mut bg_voxelize_per_primitive = Vec::new();
        for node in &scene.nodes {
            let Some(mesh) = scene.meshes.get(node.mesh_id) else {
                continue;
            };
            for primitive in &mesh.primitives {
                // each (object, primitive) pair in the scene gets its own bind group
                // inefficient but idc its just a generation step for now

                #[repr(C)]
                #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct PrimitiveBufferEntry {
                    matrix: glam::Mat4,
                    normal_matrix: [[f32; 4]; 3],
                    material_id: u32,
                    index_count: u32,
                    _pad: [f32; 2],
                }
                // binding 0
                let buffer_primitive =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive data uniform buffer"),
                        contents: bytemuck::cast_slice(&[PrimitiveBufferEntry {
                            matrix: node.transform,
                            normal_matrix: glam::Mat3::from_mat4(node.transform.inverse())
                                .transpose()
                                .to_cols_array_2d()
                                .map(|v| [v[0], v[1], v[2], 0.0]),
                            material_id: primitive.material_id,
                            index_count: primitive.indices.count,
                            _pad: [0.0; 2],
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                // binding 1
                let indices_u32 =
                    bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end])
                        .iter()
                        .map(|idx: &u16| *idx as u32)
                        .collect::<Vec<u32>>();
                let buffer_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive indices data"),
                    contents: match primitive.indices.component_type {
                        gltf::schema::ComponentType::UnsignedShort => {
                            &bytemuck::cast_slice(&indices_u32)
                        }
                        _ => &gltf.bin[primitive.indices.start..primitive.indices.end],
                    },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                // binding 2
                let buffer_positions =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex position data"),
                        contents: &gltf.bin[primitive.positions.start..primitive.positions.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                // binding 3
                let buffer_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex normal data"),
                    contents: &gltf.bin[primitive.normals.start..primitive.normals.end],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                // binding 4
                let buffer_tangents =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex tangent data"),
                        contents: &gltf.bin[primitive.tangents.start..primitive.tangents.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                // binding 5
                let buffer_uv = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex uv data"),
                    contents: &gltf.bin[primitive.uv.start..primitive.uv.end],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

                bg_voxelize_per_primitive.push(PrimitiveGroup {
                    bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("per primitive voxelize bind group"),
                        layout: &self.bg_layouts.voxelize_per_primitive,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer_primitive.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buffer_indices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buffer_positions.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buffer_normals.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buffer_tangents.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: buffer_uv.as_entire_binding(),
                            },
                        ],
                    }),
                    index_count: primitive.indices.count,
                });
            }
        }
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("voxelization pass"),
                timestamp_writes: None,
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.voxelize);
            pass.set_bind_group(0, Some(&bg_voxelize_shared), &[]);
            pass.set_bind_group(1, Some(&bg_voxelize_scene_textures), &[]);

            for primitive in &bg_voxelize_per_primitive {
                pass.set_bind_group(2, Some(&primitive.bg), &[]);

                let tris = primitive.index_count / 3;
                pass.dispatch_workgroups(tris.div_ceil(64), 1, 1);
            }
        }
    }

    fn build_palette(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> PaletteData {
        // create the palette bind group
        // palette is just hand-generated for now, should do some cool generation in a previous compute stage in the future
        // that's based on the actual albedo distribution in the gltf
        let mut palette = vec![glam::Vec4::ZERO];
        for r in 0..9u32 {
            for g in 0..9u32 {
                for b in 0..9u32 {
                    let rgba = (glam::vec3(r as f32, g as f32, b as f32) / 8.0).extend(1.0);
                    palette.push(rgba);
                }
            }
        }
        for i in 0..69u32 {
            let v = (i * 255 / 68) & 0xff;
            let rgb = (glam::uvec3(v, v, v).as_vec3() / 255.0).extend(1.0);
            palette.push(rgb);
        }
        while palette.len() < 1024 {
            palette.push(glam::Vec4::ZERO);
        }

        let buffer_palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette colors"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // let tex_palette_lut = device.create_texture(&wgpu::TextureDescriptor {
        //     label: Some("palette LUT"),
        //     size: wgpu::Extent3d {
        //         width: 32,
        //         height: 32,
        //         depth_or_array_layers: 32,
        //     },
        //     mip_level_count: 1,
        //     sample_count: 1,
        //     dimension: wgpu::TextureDimension::D3,
        //     format: wgpu::TextureFormat::R32Uint,
        //     usage: wgpu::TextureUsages::STORAGE_BINDING,
        //     view_formats: &[],
        // });

        // let bg_palette = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: Some("palette bin group"),
        //     layout: &self.bg_layouts.palette,
        //     entries: &[
        //         wgpu::BindGroupEntry {
        //             binding: 0,
        //             resource: buffer_palette.as_entire_binding(),
        //         },
        //         // wgpu::BindGroupEntry {
        //         //     binding: 1,
        //         //     resource: wgpu::BindingResource::TextureView(&tex_palette_lut.create_view(
        //         //         &wgpu::TextureViewDescriptor {
        //         //             label: Some("palette LUT view"),
        //         //             usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
        //         //             ..Default::default()
        //         //         },
        //         //     )),
        //         // },
        //     ],
        // });
        // {
        //     let descriptor = wgpu::ComputePassDescriptor {
        //         label: Some("palette LUT generation pass"),
        //         timestamp_writes: None,
        //     };
        //     let mut pass = encoder.begin_compute_pass(&descriptor);

        //     pass.set_pipeline(&self.pipelines.palette);
        //     pass.set_bind_group(0, Some(&bg_palette), &[]);

        //     pass.dispatch_workgroups(8, 8, 8);
        // }

        PaletteData {
            buffer_palette,
            // tex_palette_lut,
        }
    }

    fn build_tree(
        &self,
        palette: &PaletteData,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        voxels_raw: &wgpu::Texture,
        size_chunks: glam::UVec3,
    ) -> BrickmapData {
        // create tree data bind group
        // this contains the resulting storage buffers/texture that make up the voxel brickmap
        let buffer_chunk_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk indices"),
            size: (size_chunks.element_product() as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk data"),
            size: (size_chunks.element_product() as u64) * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tex_brickmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel texture"),
            size: wgpu::Extent3d {
                width: size_chunks.x << 3,
                height: size_chunks.y << 3,
                depth_or_array_layers: size_chunks.z << 3,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bg_tree_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel brickmap data"),
            layout: &self.bg_layouts.tree_data,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_chunk_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_chunks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tex_brickmap.create_view(
                        &wgpu::TextureViewDescriptor {
                            label: Some("brickmap write view"),
                            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        // create the brickmap texture & allocator atomic buffer bind group
        // contains both the raw brickmap texture, and atomic counters for building the brickmap
        let buffer_alloc_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brickmap allocator data"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // i read from the alloc metadata buffer by copying it here
        let buffer_alloc_results = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brickmap allocator result buffer"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bg_tree_alloc_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brickmap builder data"),
            layout: &self.bg_layouts.tree_alloc_texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_alloc_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&voxels_raw.create_view(
                        &wgpu::TextureViewDescriptor {
                            label: Some("raw voxels view"),
                            dimension: Some(wgpu::TextureViewDimension::D3),
                            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                            ..Default::default()
                        },
                    )),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: palette.buffer_palette.as_entire_binding(),
                },
            ],
        });

        // now execute
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("voxel brickmap packing"),
                timestamp_writes: None,
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.tree);
            pass.set_bind_group(0, Some(&bg_tree_data), &[]);
            pass.set_bind_group(1, Some(&bg_tree_alloc_texture), &[]);

            pass.dispatch_workgroups(size_chunks.x, size_chunks.y, size_chunks.z);
        }

        // we read from the results buffer
        encoder.copy_buffer_to_buffer(
            &buffer_alloc_data,
            0,
            &buffer_alloc_results,
            0,
            buffer_alloc_data.size(),
        );

        BrickmapData {
            buffer_chunk_indices,
            buffer_chunks,
            tex_brickmap,
            buffer_alloc_results: buffer_alloc_results,
        }
    }
}

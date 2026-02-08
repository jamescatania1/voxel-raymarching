use std::{
    io::{BufRead, BufReader, Cursor, Seek},
    sync::Arc,
};

use anyhow::Result;
use models::{Gltf, Scene};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{RendererCtx, SizedWindow, engine::Engine, renderer::buffers::CameraDataBuffer};

const VOXELS_PER_UNIT: u32 = 15;

pub struct VoxelizeResult {
    pub texture: wgpu::Texture,
    pub scene: Scene,
    pub size: glam::IVec3,
    pub brickmap: BrickmapData,
    pub chunk_count: glam::UVec3,
}

pub struct Voxelizer {
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
    // size: glam::UVec3,
    // size_chunks: glam::UVec3,
}
struct Pipelines {
    voxelize: wgpu::ComputePipeline,
    // palette_lut: wgpu::ComputePipeline,
    tree: wgpu::ComputePipeline,
}
struct BindGroupLayouts {
    voxelize_shared: wgpu::BindGroupLayout,
    voxelize_scene_textures: wgpu::BindGroupLayout,
    voxelize_per_primitive: wgpu::BindGroupLayout,
    tree_data: wgpu::BindGroupLayout,
    tree_alloc_texture: wgpu::BindGroupLayout,
}
// struct BindGroups {
//     voxelize_shared: wgpu::BindGroup,
//     voxelize_scene_textures: wgpu::BindGroup,
//     voxelize_per_primitive: Vec<PrimitiveGroup>,
//     tree_data: wgpu::BindGroup,
//     tree_alloc_texture: wgpu::BindGroup,
// }

impl Voxelizer {
    pub fn new(device: &wgpu::Device, scene_tex_count: u32) -> Self {
        // let src = std::include_bytes!("../../assets/sponza.glb");
        // let mut src = BufReader::new(Cursor::new(src));

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
                            format: wgpu::TextureFormat::Rgba8Unorm,
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D3,
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
                        std::include_str!("../shaders/voxelize.wgsl").into(),
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
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("../shaders/tree.wgsl").into(),
                    ),
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

    pub fn load_gltf<R: BufRead + Seek>(
        &mut self,
        src: &mut R,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<VoxelizeResult> {
        let gltf = Gltf::parse(src)?;
        let scene = Scene::from_gltf(&gltf)?;

        let base_unscaled = scene.min.floor().as_ivec3();
        let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_ivec3();
        let size_scaled = size_unscaled.as_uvec3() * VOXELS_PER_UNIT;
        let size_chunks = (size_scaled + 7) / 8;

        // result 3d voxel texture
        let voxels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel texture"),
            size: wgpu::Extent3d {
                width: size_scaled.x,
                height: size_scaled.y,
                depth_or_array_layers: size_scaled.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let mut state = VoxelizerState {
            gltf,
            scene,
            base_unscaled,
            size_unscaled,
            size_scaled,
            size_chunks,
            voxels,
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        self.voxelize(&mut state, device, queue, &mut encoder);
        let brickmap = self.build_tree(&mut state, device, queue, &mut encoder);

        queue.submit([encoder.finish()]);

        Ok(VoxelizeResult { texture: state.voxels, scene, size: state., brickmap, chunk_count: () })
    }
}

struct VoxelizerState {
    gltf: Gltf,
    scene: Scene,
    base_unscaled: glam::IVec3,
    size_unscaled: glam::IVec3,
    size_scaled: glam::UVec3,
    size_chunks: glam::UVec3,
    voxels: wgpu::Texture,
}

pub struct BrickmapData {
    pub chunk_index: wgpu::Buffer,
    pub chunks: wgpu::Buffer,
    pub bricks: wgpu::Buffer,
}

impl Voxelizer {
    fn voxelize(
        &self,
        state: &VoxelizerState,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
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
                    base: state.base_unscaled.as_vec3().extend(0.0),
                    size: state.size_unscaled.as_vec3(),
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
                _pad: [f32; 3],
            }
            // binding 1
            let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material data storage buffer"),
                contents: bytemuck::cast_slice(
                    &state
                        .scene
                        .materials
                        .iter()
                        .map(|mat| MaterialBufferEntry {
                            base_albedo: mat.base_albedo,
                            metallic: mat.metallic,
                            roughness: mat.roughness,
                            normal_scale: mat.normal_scale,
                            albedo_index: mat.albedo_index,
                            normal_index: mat.normal_index,
                            _pad: [0.0; 3],
                        })
                        .collect::<Vec<MaterialBufferEntry>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            // binding 2
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
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
                        resource: wgpu::BindingResource::TextureView(&state.voxels.create_view(
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
            let texture_views = state
                .scene
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
        for node in &state.scene.nodes {
            let Some(mesh) = state.scene.meshes.get(node.mesh_id) else {
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
                let indices_u32 = bytemuck::cast_slice(
                    &state.gltf.bin[primitive.indices.start..primitive.indices.end],
                )
                .iter()
                .map(|idx: &u16| *idx as u32)
                .collect::<Vec<u32>>();
                let buffer_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive indices data"),
                    contents: match primitive.indices.component_type {
                        models::schema::ComponentType::UnsignedShort => {
                            &bytemuck::cast_slice(&indices_u32)
                        }
                        _ => &state.gltf.bin[primitive.indices.start..primitive.indices.end],
                    },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                // binding 2
                let buffer_positions =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex position data"),
                        contents: &state.gltf.bin
                            [primitive.positions.start..primitive.positions.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                // binding 3
                let buffer_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex normal data"),
                    contents: &state.gltf.bin[primitive.normals.start..primitive.normals.end],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                // binding 4
                let buffer_tangents =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex tangent data"),
                        contents: &state.gltf.bin[primitive.tangents.start..primitive.tangents.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                // binding 5
                let buffer_uv = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex uv data"),
                    contents: &state.gltf.bin[primitive.uv.start..primitive.uv.end],
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

    fn build_tree(
        &self,
        state: &VoxelizerState,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> BrickmapData {
        // create tree data bind group
        // this contains the resulting storage buffers that make up the voxel brickmap
        let buffer_chunk_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk indices"),
            size: (state.size_chunks.element_product() as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buffer_chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk data"),
            size: (state.size_chunks.element_product() as u64) * 68,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buffer_bricks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick data"),
            size: (state.size_chunks.element_product() as u64) * 2048,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
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
                    resource: buffer_bricks.as_entire_binding(),
                },
            ],
        });

        // create the brickmap texture & allocator atomic buffer bind group
        // contains both the raw brickmap texture, and atomic counters for building the brickmap
        let buffer_alloc_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brickmap allocator data"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let view_voxels = state.voxels.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raw voxels view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            ..Default::default()
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
                    resource: wgpu::BindingResource::TextureView(&view_voxels),
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

            pass.dispatch_workgroups(
                state.size_chunks.x,
                state.size_chunks.y,
                state.size_chunks.z,
            );
        }

        BrickmapData {
            chunk_index: buffer_chunk_indices,
            chunks: buffer_chunks,
            bricks: buffer_bricks,
        }
    }
}

// pub fn voxelize(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<VoxelizeResult> {
//     let voxels = pack_voxels(device, queue, chunk_count, &texture);

//     Ok(VoxelizeResult {
//         scene,
//         texture,
//         size,
//         chunk_count,
//         voxels,
//     })
// }

// pub struct PackResult {}

// fn pack_voxels(
//     device: &wgpu::Device,
//     queue: &wgpu::Queue,
//     chunk_count: glam::UVec3,
//     voxels: &wgpu::Texture,
// ) -> PackResult {
//     let bg_layout_packed_data = ;
//     let bg_layout_alloc_texture =

//     let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//         label: Some("voxel brickmap packing"),
//         bind_group_layouts: &[&bg_layout_packed_data, &bg_layout_alloc_texture],
//         immediate_size: 0,
//     });
//     let pipeline = {
//         let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
//             label: Some("voxel brickmap packing"),
//             source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/tree.wgsl").into()),
//         });
//         device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//             label: Some("voxel packing pipeline"),
//             layout: Some(&pipeline_layout),
//             module: &shader,
//             entry_point: Some("compute_main"),
//             compilation_options: Default::default(),
//             cache: None,
//         })
//     };

// }

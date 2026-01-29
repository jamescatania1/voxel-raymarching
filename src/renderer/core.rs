use std::{f32, sync::Arc, time::Duration};

use glam::UVec2;
use wgpu::{
    BindGroup, Buffer, ComputePipeline, Device, RenderPipeline, Sampler, Texture, TextureView,
    util::DeviceExt,
};
use winit::window::Window;

use crate::{
    SizedWindow,
    engine::Engine,
    renderer::{
        buffers::{CameraDataBuffer, ModelDataBuffer, SceneDataBuffer},
        nested_tree::TwoTree,
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
    // depth: DepthTexture,
    // pipeline: RenderPipeline,
    // camera_bind_group: BindGroup,
    // camera_uniform: Buffer,
    // model_bind_group: BindGroup,
    // model_uniform: Buffer,
    // pipeline_raymarch: ComputePipeline,
    // raymarch_bind_group: BindGroup,
    pipelines: Pipelines,
    textures: Textures,
    uniforms: Uniforms,
    bind_groups: BindGroups,
    timing: Option<RenderTimer>,
    quad: Quad,
}

struct Uniforms {
    scene: Buffer,
    voxel_chunk_index: Buffer,
    voxel_chunks: Buffer,
    voxel_bricks: Buffer,
    camera: Buffer,
    camera_data: CameraDataBuffer,
    model: Buffer,
    model_data: ModelDataBuffer,
}

struct Pipelines {
    raymarch: ComputePipeline,
    post_fx: RenderPipeline,
}

struct BindGroups {
    raymarch: Option<BindGroup>,
    post_fx: Option<BindGroup>,
    camera: BindGroup,
    model: BindGroup,
}

struct Textures {
    color: Option<Texture>,
}

impl Renderer {
    pub fn new(
        window: Arc<Window>,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        engine: &Engine,
    ) -> Self {
        let textures = Textures { color: None };

        let uniforms = {
            // let voxels_data = VoxelDataBuffer::new(&engine.scene);
            // let voxels_tree = PTree::from_scene(&engine.scene);
            // let voxels = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //     label: Some("voxel data storage buffer"),
            //     // contents: bytemuck::cast_slice(&voxels_tree.nodes),
            //     contents: bytemuck::cast_slice(&voxels_data.0),
            //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            // });

            // let voxels = BrickMap::from_scene(&engine.scene);
            // let voxel_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //     label: Some("voxel index storage buffer"),
            //     contents: bytemuck::cast_slice(&voxels.index),
            //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            // });
            // let voxel_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //     label: Some("voxel data storage buffer"),
            //     contents: bytemuck::cast_slice(&voxels.data),
            //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            // });

            let voxels = TwoTree::from_scene(&engine.scene);
            let voxel_chunk_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel chunk indices storage buffer"),
                contents: bytemuck::cast_slice(&voxels.chunk_indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let voxel_chunks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel chunks storage buffer"),
                contents: bytemuck::cast_slice(&voxels.chunks),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let voxel_bricks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel bricks storage buffer"),
                contents: bytemuck::cast_slice(&voxels.bricks),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let scene_data = SceneDataBuffer::new(&engine.scene, voxels.size);
            let scene = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene data uniform buffer"),
                contents: bytemuck::cast_slice(&[scene_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let camera_data = CameraDataBuffer::default();
            let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera data uniform buffer"),
                contents: bytemuck::cast_slice(&[camera_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let model_data = ModelDataBuffer::default();
            let model = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model data uniform buffer"),
                contents: bytemuck::cast_slice(&[model_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            Uniforms {
                scene,
                voxel_chunk_index,
                voxel_chunks,
                voxel_bricks,
                camera,
                camera_data,
                model,
                model_data,
            }
        };

        let pipelines = {
            let raymarch = {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("raymarch shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("../shaders/main.wgsl").into(),
                    ),
                });

                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("raymarch pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("compute_main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

            let post_fx = {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("main shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("../shaders/fx.wgsl").into(),
                    ),
                });

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("post fx pipeline"),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
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
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(format.into())],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview: None,
                    cache: None,
                })
            };

            Pipelines { raymarch, post_fx }
        };

        let bind_groups = {
            let camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera bind group"),
                layout: &pipelines.raymarch.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.camera.as_entire_binding(),
                }],
            });

            let model = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model bind group"),
                layout: &pipelines.raymarch.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.model.as_entire_binding(),
                }],
            });

            BindGroups {
                raymarch: None,
                post_fx: None,
                camera,
                model,
            }
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device));

        let quad = Quad::new(device);

        let mut _self = Self {
            pipelines,
            bind_groups,
            textures,
            uniforms,
            timing,
            quad,
        };

        _self.update_screen_resources(&window, device);

        return _self;
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.update_screen_resources(window, device);
    }

    fn update_screen_resources(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        let size = window.size();

        let tex_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("out texture"),
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
        let view_color = tex_color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raymarch out texture storage view"),
            ..Default::default()
        });
        let sampler_color = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("color sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        self.textures.color = Some(tex_color);

        let bg_raymarch = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raymarch bind group"),
            layout: &self.pipelines.raymarch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniforms.scene.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniforms.voxel_chunk_index.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniforms.voxel_chunks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniforms.voxel_bricks.as_entire_binding(),
                },
            ],
        });
        self.bind_groups.raymarch = Some(bg_raymarch);

        let bg_post_fx = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post fx bind group"),
            layout: &self.pipelines.post_fx.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_color),
                },
            ],
        });
        self.bind_groups.post_fx = Some(bg_post_fx);
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut RendererCtx) {
        let size = ctx.window.size();

        // update uniform buffers
        {
            self.uniforms.camera_data.update(&ctx.engine.camera);
            ctx.queue.write_buffer(
                &self.uniforms.camera,
                0,
                bytemuck::cast_slice(&[self.uniforms.camera_data]),
            );
            self.uniforms.model_data.update(&ctx.engine.model);
            ctx.queue.write_buffer(
                &self.uniforms.model,
                0,
                bytemuck::cast_slice(&[self.uniforms.model_data]),
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

            pass.push_debug_group("prepare raymarch data");
            pass.set_pipeline(&self.pipelines.raymarch);
            pass.set_bind_group(0, &self.bind_groups.raymarch, &[]);
            pass.set_bind_group(1, &self.bind_groups.camera, &[]);
            pass.set_bind_group(2, &self.bind_groups.model, &[]);
            pass.pop_debug_group();

            pass.insert_debug_marker("raymarch");
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
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
                depth_stencil_attachment: None,
                occlusion_query_set: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.push_debug_group("prepare post fx data");
            pass.set_pipeline(&self.pipelines.post_fx);
            pass.set_bind_group(0, &self.bind_groups.post_fx, &[]);
            pass.set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));
            pass.pop_debug_group();

            pass.insert_debug_marker("post fx");
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

        if let Some(timing) = &self.timing {
            timing
                .result_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());

            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .unwrap();

            let view = timing.result_buffer.get_mapped_range(..);
            let timestamps: &[u64] = bytemuck::cast_slice(&*view);

            let time_raymarch = Duration::from_nanos(timestamps[1] - timestamps[0]);
            let time_post_fx = Duration::from_nanos(timestamps[3] - timestamps[2]);

            ctx.ui.state.pass_avg = vec![
                ("Raymarch".into(), time_raymarch),
                ("Post FX".into(), time_post_fx),
            ];

            drop(view);
            timing.result_buffer.unmap();
        }
    }
}

struct RenderTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    result_buffer: Arc<wgpu::Buffer>,
}
impl RenderTimer {
    const QUERY_PASS_COUNT: u32 = 2;

    fn new(device: &Device) -> Self {
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

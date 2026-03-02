mod config;
mod engine;
mod renderer;
mod ui;
mod utils;

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

use std::{sync::Arc, time::Instant};

use pollster::block_on;
use winit::{
    application::ApplicationHandler, event::DeviceEvent, event::WindowEvent, window::Window,
};

use crate::config::Config;
use crate::renderer::gltf_viewer::ModelViewer;
use crate::{
    engine::{Engine, EngineCtx},
    renderer::{Renderer, RendererCtx},
    ui::Ui,
};

struct Program {
    state: Option<State>,
}

const DEBUG_MODELS: bool = false;

impl ApplicationHandler for Program {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let cfg = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
            .with_title("wgpu demo");

        let window = Arc::new(event_loop.create_window(cfg).unwrap());
        self.state = Some(block_on(State::new(window.clone())).unwrap());

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        state.handle_input(event_loop, &event);
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();

        let Some(max_fps) = state.config.max_fps else {
            return;
        };

        let now = Instant::now();
        let prev_time = state.prev_time.unwrap_or(now);
        let elapsed = now - prev_time;
        let targ_frame_time = std::time::Duration::from_secs_f64(1.0 / (max_fps as f64));

        if elapsed >= targ_frame_time {
            state.window.request_redraw();
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                now + targ_frame_time - elapsed,
            ));
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        state.handle_device_input(event_loop, &event);
    }
}

struct State {
    window: Arc<winit::window::Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    config: Config,
    engine: Engine,
    renderer: Renderer,
    ui: Ui,
    prev_time: Option<Instant>,
    prev_time_fixed: Option<Instant>,
    debug_viewer: Option<ModelViewer>,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let mut features = wgpu::Features::default();
        features |= wgpu::Features::FLOAT32_FILTERABLE;
        features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
        features |= wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 128;
        // limits.max_binding_array_elements_per_shader_stage = 406;
        limits.max_binding_array_elements_per_shader_stage = 128;
        limits.max_storage_textures_per_shader_stage = 10;
        limits.max_compute_invocations_per_workgroup = 512;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .unwrap();

        let (surface, format) = {
            let surface = instance.create_surface(Arc::clone(&window)).unwrap();
            let capabilities = surface.get_capabilities(&adapter);
            let format = capabilities.formats[0];
            (surface, format)
        };

        let mut config = Config::default();

        let engine = Engine::new(&window);

        let mut ui = Ui::new(&window, &device, format);

        let renderer = Renderer::new(
            Arc::clone(&window),
            &device,
            &queue,
            format,
            &mut config,
            &mut ui,
        );

        let debug_viewer = if DEBUG_MODELS {
            Some(ModelViewer::new(
                Arc::clone(&window),
                &device,
                &queue,
                format,
                &engine,
            )?)
        } else {
            None
        };

        let mut _self = Self {
            window,
            device,
            queue,
            surface,
            format,
            config,
            engine,
            renderer,
            ui,
            debug_viewer,
            prev_time: None,
            prev_time_fixed: None,
        };

        _self.configure_surface();

        Ok(_self)
    }

    fn frame(&mut self) {
        let time = Instant::now();
        let delta_time = time - *self.prev_time.get_or_insert_with(|| time.clone());

        match self.prev_time_fixed {
            Some(prev_time_fixed) => {
                if (time - prev_time_fixed).as_secs_f64() >= 1.0 {
                    self.renderer.fixed_update(&mut RendererCtx {
                        window: &self.window,
                        device: &self.device,
                        queue: &self.queue,
                        surface: &self.surface,
                        format: &self.format,
                        engine: &mut self.engine,
                        ui: &mut self.ui,
                        config: &mut self.config,
                    });
                    self.prev_time_fixed = Some(time);
                }
            }
            None => {
                self.prev_time_fixed = Some(time);
            }
        }

        self.engine.frame(
            &delta_time,
            &mut EngineCtx {
                window: &self.window,
                ui: &mut self.ui,
            },
        );

        if let Some(viewer) = &mut self.debug_viewer {
            viewer.frame(&mut RendererCtx {
                window: &self.window,
                device: &self.device,
                queue: &self.queue,
                surface: &self.surface,
                format: &self.format,
                engine: &mut self.engine,
                ui: &mut self.ui,
                config: &mut self.config,
            });
        } else {
            self.renderer.frame(
                &delta_time,
                &mut RendererCtx {
                    window: &self.window,
                    device: &self.device,
                    queue: &self.queue,
                    surface: &self.surface,
                    format: &self.format,
                    engine: &mut self.engine,
                    ui: &mut self.ui,
                    config: &mut self.config,
                },
            );
        }

        self.engine.input.frame();

        self.prev_time = Some(time);

        if self.config.max_fps.is_none() {
            self.window.request_redraw();
        }
    }

    fn handle_resize(&mut self) {
        self.configure_surface();

        self.engine.handle_resize(&self.window);
        self.renderer.handle_resize(&self.window, &self.device);
        if let Some(viewer) = &mut self.debug_viewer {
            viewer.handle_resize(&self.window, &self.device);
        }
    }

    fn handle_input(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::WindowEvent,
    ) {
        let ui_resp = self.ui.handle_input(&self.window, event);
        if ui_resp.consumed {
            return;
        }

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.frame();
            }
            winit::event::WindowEvent::Resized(_) => {
                self.handle_resize();
            }
            winit::event::WindowEvent::MouseInput { state, button, .. } => {
                self.engine.input.handle_mouse(state, button);
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.engine
                    .input
                    .handle_keyboard(&self.window, event_loop, event);
            }
            _ => {}
        }
    }

    fn handle_device_input(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.ui.handle_mouse_motion(delta.clone());
            }
            _ => {}
        }

        self.engine
            .input
            .handle_device_input(&self.window, event_loop, event);
    }
}

impl State {
    fn configure_surface(&mut self) {
        let size = self.window.size();

        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                view_formats: vec![self.format.add_srgb_suffix()],
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                width: size.x,
                height: size.y,
                desired_maximum_frame_latency: 2,
                present_mode: wgpu::PresentMode::AutoNoVsync,
            },
        );
    }
}

pub trait SizedWindow {
    fn size(&self) -> glam::UVec2;
}
impl SizedWindow for Window {
    fn size(&self) -> glam::UVec2 {
        let size = self.inner_size();
        glam::uvec2(size.width, size.height)
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop.run_app(&mut Program { state: None }).unwrap();
}

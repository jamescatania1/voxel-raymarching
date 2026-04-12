use std::{sync::Arc, time::Instant};

use crate::{
    SizedWindow,
    config::Config,
    core::{Engine, EngineCtx, Input, Renderer, RendererCtx},
    ui::Ui,
};
use anyhow::Result;
use winit::window::Window;

pub struct App {
    pub window: Arc<winit::window::Window>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub format: wgpu::TextureFormat,
    pub config: Config,
    pub input: Input,
    pub engine: Engine,
    pub renderer: Renderer,
    pub ui: Ui,
    pub prev_time: Option<Instant>,
    pub prev_time_fixed: Option<Instant>,
}

impl App {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            ..Default::default()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();

        let mut features = wgpu::Features::default();
        features |= wgpu::Features::FLOAT32_FILTERABLE;
        features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 512 * 1024 * 1024;
        limits.max_storage_textures_per_shader_stage = 12;

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

        let input = Input::new();

        let engine = Engine::new(&window, &config);

        let mut ui = Ui::new(&window, &device, format);

        let renderer = Renderer::new(
            Arc::clone(&window),
            &device,
            &queue,
            format,
            &mut config,
            &mut ui,
        );

        let mut _self = Self {
            window,
            device,
            queue,
            surface,
            format,
            config,
            input,
            engine,
            renderer,
            ui,
            prev_time: None,
            prev_time_fixed: None,
        };

        _self.configure_surface();

        Ok(_self)
    }

    pub fn frame(&mut self) {
        let time = Instant::now();
        let delta_time = time - *self.prev_time.get_or_insert_with(|| time.clone());

        match self.prev_time_fixed {
            Some(prev_time_fixed) => {
                if (time - prev_time_fixed).as_secs_f64() >= 1.0 {
                    self.renderer.fixed_update(RendererCtx {
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
            EngineCtx {
                window: &self.window,
                config: &self.config,
                input: &self.input,
                ui: &mut self.ui,
            },
        );

        self.renderer.frame(
            &delta_time,
            RendererCtx {
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

        self.input.frame();

        self.prev_time = Some(time);

        if self.config.max_fps.is_none() {
            self.window.request_redraw();
        }
    }

    pub fn handle_resize(&mut self) {
        self.configure_surface();

        self.engine.handle_resize(&self.window);
        self.renderer.handle_resize(&self.window, &self.device);
    }

    pub fn handle_input(
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
                self.input.handle_mouse(state, button);
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.input.handle_keyboard(&self.window, event_loop, event);
            }
            _ => {}
        }
    }

    pub fn handle_device_input(
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

        self.input
            .handle_device_input(&self.window, event_loop, event);
    }

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

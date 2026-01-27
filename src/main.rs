mod engine;
mod renderer;
mod ui;
mod vox;

use std::{sync::Arc, time::Instant};

use pollster::block_on;
use winit::{
    application::ApplicationHandler, event::DeviceEvent, event::WindowEvent, window::Window,
};

use crate::{
    engine::{Engine, EngineCtx},
    renderer::{Renderer, RendererCtx},
    ui::Ui,
};

struct Program {
    state: Option<State>,
}

impl ApplicationHandler for Program {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let cfg = Window::default_attributes().with_title("wgpu demo");

        let window = event_loop.create_window(cfg).unwrap();
        let window = Arc::new(window);

        let state = block_on(State::new(window.clone()));
        self.state = Some(state);

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
    engine: Engine,
    renderer: Renderer,
    ui: Ui,
    prev_time: Option<Instant>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let mut features = wgpu::Features::default();
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
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

        let engine = Engine::new(&window);

        let renderer = Renderer::new(Arc::clone(&window), &device, format, &engine);

        let ui = Ui::new(&window, &device, format);

        let mut _self = Self {
            window,
            device,
            queue,
            surface,
            format,
            engine,
            renderer,
            ui,
            prev_time: None,
        };

        _self.configure_surface();

        return _self;
    }

    fn frame(&mut self) {
        let time = Instant::now();
        let delta_time = time - *self.prev_time.get_or_insert_with(|| time.clone());

        self.engine.frame(
            &delta_time,
            &mut EngineCtx {
                window: &self.window,
                ui: &mut self.ui,
            },
        );

        self.renderer.frame(
            &delta_time,
            &mut RendererCtx {
                window: &self.window,
                device: &self.device,
                queue: &self.queue,
                surface: &self.surface,
                format: &self.format,
                engine: &self.engine,
                ui: &mut self.ui,
            },
        );

        self.engine.input.frame();

        self.prev_time = Some(time);
        self.window.request_redraw();
    }

    fn handle_resize(&mut self) {
        self.configure_surface();

        self.engine.handle_resize(&self.window);
        self.renderer.handle_resize(&self.window, &self.device);
    }

    fn handle_input(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::WindowEvent,
    ) {
        self.ui.handle_input(&self.window, event);
        self.engine.handle_input(&self.window, event_loop, event);

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
            _ => {}
        }
    }

    fn handle_device_input(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::DeviceEvent,
    ) {
        self.engine
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

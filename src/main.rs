mod app;
mod cube;

use std::sync::Arc;

use pollster::block_on;
use winit::{application::ApplicationHandler, event::WindowEvent, window::Window};

use crate::app::App;

#[derive(Debug)]
struct Program {
    app: Option<App>,
}

impl Program {
    fn new() -> Self {
        Self { app: None }
    }
}

impl ApplicationHandler for Program {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();
        let window = Arc::new(window);
        let app = block_on(App::new(window.clone()));
        self.app = Some(app);

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let app = self.app.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                app.render();
                app.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                app.on_resize(size);
            }
            _ => (),
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop.run_app(&mut Program::new()).unwrap();
}

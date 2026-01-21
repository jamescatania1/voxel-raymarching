mod app;
mod camera;
mod mesh;
mod model;

use std::{sync::Arc, time::Instant};

use pollster::block_on;
use winit::{application::ApplicationHandler, event::WindowEvent, window::Window};

use crate::app::App;

#[derive(Debug)]
struct Program {
    app: Option<App>,
    prev_time: Option<Instant>,
}

impl Program {
    fn new() -> Self {
        Self {
            app: None,
            prev_time: None,
        }
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
                let time = Instant::now();
                let delta_time = time - *self.prev_time.get_or_insert_with(|| time.clone());
                app.render(&delta_time);
                self.prev_time = Some(time);
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

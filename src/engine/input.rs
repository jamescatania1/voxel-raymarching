use std::collections::HashSet;

use glam::{DVec2, Vec2};
use winit::{
    event::{DeviceEvent, ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Debug, Default)]
pub struct Input {
    pub mouse: Mouse,
    pub scroll: f64,
    pressed_keys: HashSet<KeyCode>,
    released_keys: HashSet<KeyCode>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(mb) = match button {
                    winit::event::MouseButton::Left => Some(&mut self.mouse.left),
                    winit::event::MouseButton::Right => Some(&mut self.mouse.right),
                    _ => None,
                } {
                    if *state == ElementState::Pressed {
                        mb.clicked = true;
                        mb.down = true;
                    } else {
                        mb.released = true;
                        mb.down = false;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.repeat {
                    return;
                }

                if let winit::keyboard::Key::Character(key) = &event.logical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if key == "f" {
                                if window.fullscreen().is_some() {
                                    window.set_fullscreen(None);
                                } else {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(
                                            event_loop.primary_monitor(),
                                        ),
                                    ));
                                }
                            }
                        }
                        ElementState::Released => {}
                    }
                    if key == "f" {}
                }

                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed_keys.insert(code);
                        }
                        ElementState::Released => {
                            self.pressed_keys.remove(&code);
                        }
                    }
                }
            }

            _ => (),
        }
    }

    pub fn handle_device_input(
        &mut self,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::Motion { axis: 0, value } => {
                self.mouse.delta.x += *value;
            }
            DeviceEvent::Motion { axis: 1, value } => {
                self.mouse.delta.y += *value;
            }
            _ => {}
        }
    }

    pub fn key_down(&self, code: KeyCode) -> bool {
        self.pressed_keys.contains(&code)
    }

    // Called at the end of each frame
    pub fn frame(&mut self) {
        self.mouse.left.clicked = false;
        self.mouse.left.released = false;
        self.mouse.right.clicked = false;
        self.mouse.right.released = false;
        self.mouse.delta *= 0.0;
        self.mouse.scroll_delta = 0.0;

        // let scroll_px = window.scroll_y().unwrap_or(0.0);
        // let height_px = window
        //     .inner_height()
        //     .ok()
        //     .and_then(|h| h.as_f64())
        //     .unwrap_or(f64::MAX);
        // state.scroll = scroll_px / height_px;

        // state.resized = false;
    }
}

// impl InputState {
//     fn on_mouse_down(&mut self, e: &MouseEvent) {
//         match e.button() {
//             0 => {
//                 self.mouse.left.clicked = true;
//                 self.mouse.left.down = true;
//             }
//             2 => {
//                 self.mouse.right.clicked = true;
//                 self.mouse.right.down = true;
//             }
//             _ => (),
//         }
//     }
//     fn on_mouse_up(&mut self, e: &MouseEvent) {
//         match e.button() {
//             0 => {
//                 self.mouse.left.released = true;
//                 self.mouse.left.down = false;
//             }
//             2 => {
//                 self.mouse.right.released = true;
//                 self.mouse.right.down = false;
//             }
//             _ => (),
//         }
//     }
//     fn on_mouse_move(&mut self, e: &MouseEvent, canvas: &HtmlCanvasElement) {
//         let rect = canvas.get_bounding_client_rect();
//         self.mouse.position.x =
//             ((e.client_x() as f64 - rect.left()) / (rect.right() - rect.left())) * 2.0 - 1.0;
//         self.mouse.position.x =
//             ((e.client_y() as f64 - rect.top()) / (rect.bottom() - rect.top())) * 2.0 - 1.0;

//         let scale = u32::max(canvas.width(), canvas.height()) as f64;
//         self.mouse.delta.x += e.movement_x() as f64 / scale;
//         self.mouse.delta.y += e.movement_y() as f64 / scale;
//     }
//     fn on_click(&mut self, e: &MouseEvent) {
//         let _ = e;
//     }
//     fn on_scroll(&mut self, e: &WheelEvent) {
//         self.mouse.scroll_delta = e.delta_y() as f64;
//     }
//     fn on_key_down(&mut self, e: &KeyboardEvent) {
//         if let Some(key) = self.key(&e.key()) {
//             if !key.down {
//                 key.pressed = true;
//                 key.down = true;
//             }
//         }
//     }
//     fn on_key_up(&mut self, e: &KeyboardEvent) {
//         if let Some(key) = self.key(&e.key()) {
//             key.down = false;
//             key.released = true;
//         }
//     }
//     fn on_blur(&mut self) {
//         for key in self.keys() {
//             key.down = false;
//             key.released = true;
//         }
//     }
//     fn on_resize(&mut self) {
//         self.resized = true;
//     }
// }

#[derive(Debug, Default)]
pub struct Key {
    /// Whether the key was pressed down this specific frame
    pub pressed: bool,
    /// Whether the key was released this specific frame
    pub released: bool,
    /// Whether the key is currently held down
    pub down: bool,
}

#[derive(Debug, Default)]
pub struct Mouse {
    pub delta: DVec2,
    pub scroll_delta: f64,
    pub left: MouseButton,
    pub right: MouseButton,
}

#[derive(Debug, Default)]
pub struct MouseButton {
    /// Whether the button was clicked this specific frame
    pub clicked: bool,
    /// Whether the button was released this specific frame
    pub released: bool,
    /// Whether the button is currently held down
    pub down: bool,
}

use std::time::Duration;

use winit::keyboard::KeyCode;

use crate::{
    SizedWindow,
    engine::{camera::Camera, input::Input, model::Model},
    ui::Ui,
};

pub struct Engine {
    pub input: Input,
    pub camera: Camera,
    pub model: Model,
    cursor_locked: bool,
}

pub struct EngineCtx<'a> {
    pub window: &'a winit::window::Window,
    pub ui: &'a mut Ui,
}

const FRAME_AVG_DECAY_ALPHA: f64 = 0.02;

impl Engine {
    pub fn new(window: &winit::window::Window) -> Self {
        let input = Input::new();

        let camera = Camera::new(window.size());

        let model = Model::new();

        Self {
            input,
            camera,
            model,
            cursor_locked: false,
        }
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window) {
        self.camera.size = window.size();
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut EngineCtx) {
        if !self.cursor_locked && self.input.mouse.left.clicked {
            self.cursor_locked = true;

            ctx.window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| {
                    ctx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                })
                .unwrap();
            ctx.window.set_cursor_visible(false);
        } else if self.cursor_locked && self.input.key_down(KeyCode::Space) {
            self.cursor_locked = false;

            ctx.window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            ctx.window.set_cursor_visible(true);
        }

        self.camera
            .update(delta_time, &self.input, self.cursor_locked);

        // self.model.rotation += delta_time.as_secs_f64() as f32 * glam::Vec3::ONE;
        // self.model.position = -0.5 * glam::DVec3::ONE;
        // self.model.position = glam::dvec3(-20.0, -30.0, -10.0) * 16.0;
        // self.model.scale = glam::DVec3::ONE / 16.0;
        self.model.scale = glam::DVec3::ONE / 16.0;
        // self.model.scale = glam::Vec3::ONE / self.scene.size.max_element() as f32;
        self.model.update();

        ctx.ui.debug.frame_avg = ctx.ui.debug.frame_avg.mul_f64(1.0 - FRAME_AVG_DECAY_ALPHA)
            + delta_time.mul_f64(FRAME_AVG_DECAY_ALPHA);

        // ctx.ui.state.voxel_count = self.scene.voxel_count;
        // ctx.ui.state.scene_size = self.scene.size;
        ctx.ui.debug.camera_pos = self.camera.position;
        ctx.ui.debug.camera_rotation = self.camera.rotation;
        ctx.ui.debug.camera_forward = self.camera.forward;
        ctx.ui.debug.camera_near = self.camera.near;
        ctx.ui.debug.camera_far = self.camera.far;
    }
}

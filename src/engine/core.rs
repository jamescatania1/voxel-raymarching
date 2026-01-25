use std::time::Duration;

use winit::event::WindowEvent;

use crate::{
    SizedWindow,
    engine::{camera::Camera, input::Input, model::Model},
    ui::Ui,
    vox::Scene,
};

pub struct Engine {
    pub input: Input,
    pub camera: Camera,
    pub scene: Scene,
    pub model: Model,
}

pub struct EngineCtx<'a> {
    pub ui: &'a mut Ui,
}

const FRAME_AVG_DECAY_ALPHA: f64 = 0.02;

impl Engine {
    pub fn new(window: &winit::window::Window) -> Self {
        let input = Input::new();

        let camera = Camera::new(window.size());

        let scene = {
            let src = std::include_bytes!("../../assets/monu2.vox");
            Scene::load(src).unwrap()
        };

        let model = Model::new();

        Self {
            input,
            camera,
            scene,
            model,
        }
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.on_key_event(event, window, event_loop);
            }
            _ => (),
        }
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window) {
        self.camera.size = window.size();
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut EngineCtx) {
        self.camera.update(delta_time, &self.input);

        // self.model.rotation += delta_time.as_secs_f64() as f32 * glam::Vec3::ONE;
        self.model.position = -0.5 * glam::Vec3::ONE;
        self.model.scale = glam::Vec3::ONE / self.scene.size.max_element() as f32;
        self.model.update();

        ctx.ui.frame_avg = ctx.ui.frame_avg.mul_f64(1.0 - FRAME_AVG_DECAY_ALPHA)
            + delta_time.mul_f64(FRAME_AVG_DECAY_ALPHA);

        ctx.ui.voxel_count = self.scene.voxel_count;
        ctx.ui.scene_size = self.scene.size;
        ctx.ui.camera_pos = self.camera.position;
        ctx.ui.camera_forward = self.camera.forward;
        ctx.ui.camera_near = self.camera.near;
        ctx.ui.camera_far = self.camera.far;
    }
}

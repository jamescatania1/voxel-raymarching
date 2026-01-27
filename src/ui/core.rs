use std::{f64::consts::PI, time::Duration};

use crate::{SizedWindow, ui::renderer::UIRenderer};

pub struct Ui {
    pub state: UiState,
    renderer: UIRenderer,
}

pub struct UiCtx<'a, 'b> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub texture_view: &'b wgpu::TextureView,
    pub encoder: &'b mut wgpu::CommandEncoder,
}

#[derive(Debug, Default)]
pub struct UiState {
    pub frame_avg: Duration,
    pub pass_avg: Vec<(String, Duration)>,
    pub voxel_count: u32,
    pub scene_size: glam::IVec3,
    pub camera_pos: glam::DVec3,
    pub camera_rotation: glam::DVec3,
    pub camera_forward: glam::DVec3,
    pub camera_near: f64,
    pub camera_far: f64,
}

impl Ui {
    pub fn new(
        window: &winit::window::Window,
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            renderer: UIRenderer::new(device, out_format, window),
            state: UiState::default(),
        }
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) {
        self.renderer.handle_input(window, event);
    }

    pub fn frame<'a>(&mut self, ctx: &mut UiCtx) {
        self.renderer.begin_frame(ctx.window);

        let state = &self.state;

        egui::Window::new("Debug")
            .default_open(true)
            .resizable(true)
            .vscroll(true)
            .default_size([320.0, 180.0])
            .show(self.renderer.context(), |ui| {
                ui.label(format!(
                    "Display: {}x{}",
                    ctx.window.size().x,
                    ctx.window.size().y
                ));
                ui.label(format!("FPS: {:.2}", 1.0 / state.frame_avg.as_secs_f64()));
                ui.label(format!("Frame: {:.2?}", state.frame_avg));

                for (pass, duration) in &state.pass_avg {
                    ui.label(format!("{}: {:.2?}", pass, duration));
                }

                ui.separator();

                ui.label(format!("Scene Size: {}", state.scene_size));
                ui.label(format!("Voxels: {}", state.voxel_count));

                ui.separator();

                ui.label(format!("Camera Position: {:.2}", state.camera_pos));
                ui.label(format!(
                    "Camera Rotation: {:.2}",
                    (state.camera_rotation * 180.0 / PI) % 360.0,
                ));
                ui.label(format!("View Forward: {:.2}", state.camera_forward));
                ui.label(format!("Camera Near: {:.2}", state.camera_near));
                ui.label(format!("Camera Far: {:.2}", state.camera_far));
            });

        self.renderer.render(
            ctx.device,
            ctx.queue,
            ctx.encoder,
            ctx.window,
            ctx.texture_view,
            egui_wgpu::ScreenDescriptor {
                size_in_pixels: ctx.window.size().to_array(),
                pixels_per_point: ctx.window.scale_factor() as f32 * 1.0,
            },
        );
    }
}

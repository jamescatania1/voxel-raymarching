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
    pub sun_direction: glam::Vec3,
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

        egui::Window::new("Debug")
            .default_open(true)
            .resizable(true)
            .vscroll(true)
            .default_size([320.0, 480.0])
            .show(self.renderer.context(), |ui| {
                ui.label(format!(
                    "Display: {}x{}",
                    ctx.window.size().x,
                    ctx.window.size().y
                ));
                ui.label(format!(
                    "FPS: {:.2}",
                    1.0 / self.state.frame_avg.as_secs_f64()
                ));
                ui.label(format!("Frame: {:.2?}", self.state.frame_avg));

                for (pass, duration) in &self.state.pass_avg {
                    ui.label(format!("{}: {:.2?}", pass, duration));
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0; // Tight spacing

                    // X Axis (Red)
                    ui.label(egui::RichText::new("X").color(egui::Color32::from_rgb(200, 60, 60)));
                    ui.add(egui::DragValue::new(&mut self.state.sun_direction.x).speed(0.1));

                    ui.add_space(10.0); // Space between axes

                    // Y Axis (Green)
                    ui.label(egui::RichText::new("Y").color(egui::Color32::from_rgb(60, 200, 60)));
                    ui.add(egui::DragValue::new(&mut self.state.sun_direction.y).speed(0.1));

                    ui.add_space(10.0);

                    // Z Axis (Blue)
                    ui.label(egui::RichText::new("Z").color(egui::Color32::from_rgb(60, 60, 200)));
                    ui.add(egui::DragValue::new(&mut self.state.sun_direction.z).speed(0.1));
                });
                // let mut x = 0.2;
                // ui.add(egui::Slider::new(&mut x, -1.0..=1.0).text("x"));

                ui.separator();

                ui.label(format!("Scene Size: {}", self.state.scene_size));
                ui.label(format!("Voxels: {}", self.state.voxel_count));

                ui.separator();

                ui.label(format!("Camera Position: {:.2}", self.state.camera_pos));
                ui.label(format!(
                    "Camera Rotation: {:.2}",
                    (self.state.camera_rotation * 180.0 / PI) % 360.0,
                ));
                ui.label(format!("View Forward: {:.2}", self.state.camera_forward));
                ui.label(format!("Camera Near: {:.2}", self.state.camera_near));
                ui.label(format!("Camera Far: {:.2}", self.state.camera_far));
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

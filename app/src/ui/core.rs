use std::{f64::consts::PI, time::Duration};

use crate::{SizedWindow, models, ui::renderer::UIRenderer};

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
    pub render_resolution: glam::UVec2,
    pub render_scale: f32,
    pub voxel_count: u32,
    pub scene_size: glam::IVec3,
    pub camera_pos: glam::DVec3,
    pub camera_rotation: glam::DVec3,
    pub camera_forward: glam::DVec3,
    pub camera_near: f64,
    pub camera_far: f64,
    pub sun_altitude: f32,
    pub sun_azimuth: f32,
    pub sun_direction: glam::Vec3,
    pub shadow_bias: f32,
    pub shadow_spread: f32,
    pub filter_shadows: bool,
    pub shadow_filter_radius: f32,
    pub voxel_normal_factor: f32,
    pub ambient_ray_max_distance: u32,
    pub fxaa: bool,
    pub taa: bool,
    pub limit_fps: bool,
    pub max_fps: u32,
}

impl Ui {
    pub fn new(
        window: &winit::window::Window,
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            renderer: UIRenderer::new(device, out_format, window),
            state: UiState {
                render_scale: 0.5,
                shadow_bias: 0.0005,
                shadow_spread: 0.05,
                filter_shadows: true,
                shadow_filter_radius: 7.0,
                ambient_ray_max_distance: 10,
                voxel_normal_factor: 0.5,
                taa: false,
                fxaa: false,
                limit_fps: false,
                max_fps: 300,
                ..Default::default()
            },
        }
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.renderer.handle_window_event(window, event)
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        self.renderer.handle_mouse_motion(delta);
    }

    pub fn frame<'a>(&mut self, ctx: &mut UiCtx) {
        self.renderer.begin_frame(ctx.window);

        egui::Window::new("Debug")
            .default_open(true)
            .resizable(true)
            .vscroll(true)
            .default_width(240.0)
            .default_height(480.0)
            .show(self.renderer.context(), |ui| {
                egui::Grid::new("grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Stats").strong());
                        ui.end_row();

                        ui.label("Display");
                        ui.label(format!("{}x{}", ctx.window.size().x, ctx.window.size().y));
                        ui.end_row();

                        ui.label("Resolution");
                        ui.label(format!(
                            "{}x{}",
                            self.state.render_resolution.x, self.state.render_resolution.y
                        ));
                        ui.end_row();

                        ui.label("FPS");
                        ui.label(format!("{:.2}", 1.0 / self.state.frame_avg.as_secs_f64()));
                        ui.end_row();

                        ui.label("Frame");
                        ui.label(format!("{:.2?}", self.state.frame_avg));
                        ui.end_row();

                        for (pass, duration) in &self.state.pass_avg {
                            ui.label(pass);
                            ui.label(format!("{:.2?}", duration));
                            ui.end_row();
                        }

                        ui.end_row();
                        if ui.button("Reload Scene").clicked() {
                            let src = std::include_bytes!("../../assets/models/sponza.glb");
                            let mut src = std::io::BufReader::new(std::io::Cursor::new(src));
                            loader::voxelize(&mut src, ctx.device, ctx.queue, None).unwrap();
                        }
                        ui.end_row();

                        ui.end_row();
                        ui.label("Cap FPS");
                        ui.checkbox(&mut self.state.limit_fps, "");
                        ui.end_row();

                        ui.label("Max FPS");
                        ui.add(
                            egui::Slider::new(&mut self.state.max_fps, 10..=999).logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Scale");
                        ui.add(
                            egui::Slider::new(&mut self.state.render_scale, 0.125..=2.0)
                                .step_by(0.125)
                                .suffix(" x"),
                        );
                        ui.end_row();

                        ui.label("FXAA");
                        ui.checkbox(&mut self.state.fxaa, "");
                        ui.end_row();

                        ui.label("TAA");
                        ui.checkbox(&mut self.state.taa, "");
                        ui.end_row();

                        ui.end_row();
                        ui.label(egui::RichText::new("Scene").strong());
                        ui.end_row();

                        ui.label("Bounds");
                        ui.label(format!("{}", self.state.scene_size));
                        ui.end_row();

                        ui.label("Voxels");
                        ui.label(format!("{}", self.state.voxel_count));
                        ui.end_row();

                        ui.end_row();
                        ui.label(egui::RichText::new("Lighting").strong());
                        ui.end_row();

                        ui.label("Azimuth");
                        ui.add(
                            egui::Slider::new(
                                &mut self.state.sun_azimuth,
                                -std::f32::consts::PI..=std::f32::consts::PI,
                            )
                            .suffix(" rad"),
                        );
                        ui.end_row();

                        ui.label("Altitutde");
                        ui.add(
                            egui::Slider::new(
                                &mut self.state.sun_altitude,
                                -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                            )
                            .suffix(" rad"),
                        );
                        ui.end_row();

                        ui.label("Per Voxel Normals");
                        ui.add(egui::Slider::new(
                            &mut self.state.voxel_normal_factor,
                            0.0..=1.0,
                        ));
                        ui.end_row();

                        ui.label("Ambient Max Distance");
                        ui.add(egui::Slider::new(
                            &mut self.state.ambient_ray_max_distance,
                            0..=2000,
                        ));
                        ui.end_row();

                        ui.label("Shadow Bias");
                        ui.add(egui::Slider::new(&mut self.state.shadow_bias, 0.0..=0.025));
                        ui.end_row();

                        ui.label("Shadow Spread");
                        ui.add(
                            egui::Slider::new(&mut self.state.shadow_spread, 0.0..=1.0)
                                .logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Filter Shadows");
                        ui.checkbox(&mut self.state.filter_shadows, "");
                        ui.end_row();

                        ui.label("Shadow Filter Radius");
                        ui.add(egui::Slider::new(
                            &mut self.state.shadow_filter_radius,
                            0.0..=20.0,
                        ));
                        ui.end_row();

                        ui.label("Sun Direction");
                        ui.label(format!("{:.2}", self.state.sun_direction));
                        ui.end_row();

                        ui.end_row();
                        ui.label(egui::RichText::new("Camera").strong());
                        ui.end_row();

                        ui.label("Position");
                        ui.label(format!("{:.2}", self.state.camera_pos));
                        ui.end_row();

                        ui.label("Rotation");
                        ui.label(format!(
                            "{:.2}",
                            (self.state.camera_rotation * 180.0 / PI) % 360.0,
                        ));
                        ui.end_row();

                        ui.label("Forward");
                        ui.label(format!("{:.2}", self.state.camera_forward));
                        ui.end_row();

                        ui.label("Near");
                        ui.label(format!("{:.2}", self.state.camera_near));
                        ui.end_row();

                        ui.label("Far");
                        ui.label(format!("{:.2}", self.state.camera_far));
                        ui.end_row();
                    });
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

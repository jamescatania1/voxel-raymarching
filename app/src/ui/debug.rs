use std::time::Duration;

use crate::{
    config::{Config, DEBUG_VIEWS, DebugView, TONEMAPPING_ALGORITHMS, TonemappingAlgorithm},
    ui::UiCtx,
};

#[derive(Debug, Default)]
pub struct DebugWindow {
    pub frame_avg: Duration,
    pub pass_avg: Vec<(String, Duration)>,
    pub render_resolution: glam::UVec2,
    pub voxel_count: u32,
    pub scene_size: glam::IVec3,
    pub camera_pos: glam::DVec3,
    pub camera_rotation: glam::DVec3,
    pub camera_forward: glam::DVec3,
    pub camera_near: f64,
    pub camera_far: f64,
    pub sun_direction: glam::Vec3,
    view: (&'static str, DebugView),
    tonemapping: (&'static str, TonemappingAlgorithm),
    limit_fps: bool,
    max_fps: u32,
}

impl DebugWindow {
    pub fn new() -> Self {
        let cfg = Config::default();
        Self {
            view: DEBUG_VIEWS[cfg.view as usize],
            tonemapping: TONEMAPPING_ALGORITHMS[cfg.tonemapping as usize],
            ..Default::default()
        }
    }
}

fn grid<R>(ui: &mut egui::Ui, label: &str, add_contents: impl FnOnce(&mut egui::Ui) -> R) {
    ui.spacing_mut().item_spacing.y = ui.style().spacing.item_spacing.y;
    egui::CollapsingHeader::new(egui::RichText::new(label).size(14.0).strong())
        .default_open(true)
        .show(ui, |ui| {
            ui.style_mut().override_font_id = Some(egui::FontId::monospace(11.0));
            egui::Grid::new(label)
                .num_columns(2)
                .min_col_width(180.0)
                .max_col_width(180.0)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, add_contents);
        });
}

impl DebugWindow {
    pub fn ui(&mut self, ctx: &mut UiCtx, ui: &mut egui::Ui) {
        let config = &mut ctx.config;

        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.spacing_mut().item_spacing.y = 10.0;

            grid(ui, "Stats", |ui| {
                ui.label("Resolution");
                ui.label(format!(
                    "{}x{}",
                    self.render_resolution.x, self.render_resolution.y
                ));
                ui.end_row();

                ui.label("FPS");
                ui.label(format!("{:.2}", 1.0 / self.frame_avg.as_secs_f64()));
                ui.end_row();

                ui.label("Frame Time");
                ui.label(format!("{:.2?}", self.frame_avg));

                let gpu_total: Duration = self.pass_avg.iter().map(|(_, t)| t).sum();
                ui.end_row();
                ui.label(format!("    Total",));
                ui.label(format!("{:.2?}", gpu_total));

                for (pass, duration) in &self.pass_avg {
                    ui.end_row();
                    ui.label(format!("    {}", pass));
                    ui.label(format!("{:.2?}", duration));
                }
            });

            grid(ui, "Scene", |ui| {
                ui.label("Bounds");
                ui.label(format!("{}", self.scene_size));
                ui.end_row();

                ui.label("Voxels");
                ui.label(format!("{}", self.voxel_count));
                ui.end_row();

                ui.label("Sun Direction");
                ui.label(format!("{:.2}", self.sun_direction));
            });

            grid(ui, "Camera", |ui| {
                ui.label("Position");
                ui.label(format!("{:.2}", self.camera_pos));
                ui.end_row();

                ui.label("Forward");
                ui.label(format!("{:.2}", self.camera_forward));
                ui.end_row();

                ui.label("Near/Far");
                ui.label(format!("{:.2} / {:.2}", self.camera_near, self.camera_far));
            });

            grid(ui, "Settings", |ui| {
                ui.label("Cap FPS");
                ui.checkbox(&mut self.limit_fps, "");
                ui.end_row();

                ui.label("Max FPS");
                ui.add(egui::Slider::new(&mut self.max_fps, 10..=999).logarithmic(true));
                ui.end_row();

                ui.label("Scale");
                ui.add(
                    egui::Slider::new(&mut config.render_scale, 0.125..=2.0)
                        .step_by(0.125)
                        .suffix(" x"),
                );
                ui.end_row();

                ui.label("Print Debug Info");
                ui.checkbox(&mut config.print_debug_info, "");
                ui.end_row();

                ui.label("Display Probes");
                ui.checkbox(&mut config.display_probes, "");
                ui.end_row();

                ui.label("View");
                egui::ComboBox::from_label("")
                    .selected_text(self.view.0)
                    .show_ui(ui, |ui| {
                        for view in DEBUG_VIEWS {
                            ui.selectable_value(&mut self.view, *view, view.0);
                        }
                    });
            });

            grid(ui, "Post Processing", |ui| {
                ui.label("FXAA");
                ui.checkbox(&mut config.fxaa, "");
                ui.end_row();

                ui.label("TAA");
                ui.checkbox(&mut config.taa, "");
                ui.end_row();

                ui.label("Exposure");
                ui.add(egui::Slider::new(&mut config.exposure, 0.0..=10.0));
                ui.end_row();

                ui.label("Tonemapping");
                egui::ComboBox::from_label("")
                    .selected_text(self.tonemapping.0)
                    .show_ui(ui, |ui| {
                        for alg in TONEMAPPING_ALGORITHMS {
                            ui.selectable_value(&mut self.tonemapping, *alg, alg.0);
                        }
                    });
            });

            grid(ui, "Lighting", |ui| {
                ui.label("Sun Azimuth");
                ui.add(
                    egui::Slider::new(
                        &mut config.sun_azimuth,
                        -std::f32::consts::PI..=std::f32::consts::PI,
                    )
                    .suffix(" rad"),
                );
                ui.end_row();

                ui.label("Sun Altitutde");
                ui.add(
                    egui::Slider::new(
                        &mut config.sun_altitude,
                        -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                    )
                    .suffix(" rad"),
                );
                ui.end_row();

                ui.label("Sky Azimuth");
                ui.add(
                    egui::Slider::new(
                        &mut config.skybox_rotation,
                        -std::f32::consts::PI..=std::f32::consts::PI,
                    )
                    .suffix(" rad"),
                );
                ui.end_row();

                ui.label("Per Voxel Normals");
                ui.add(egui::Slider::new(
                    &mut config.voxel_normal_factor,
                    0.0..=1.0,
                ));
                ui.end_row();

                ui.label("Roughness Multiplier");
                ui.add(egui::Slider::new(
                    &mut config.roughness_multiplier,
                    0.0..=1.0,
                ));
                ui.end_row();

                ui.label("Indirect Sky Intensity");
                ui.add(egui::Slider::new(
                    &mut config.indirect_sky_intensity,
                    0.0..=10.0,
                ));
                ui.end_row();

                ui.label("Ambient Max Distance");
                ui.add(egui::Slider::new(
                    &mut config.ambient_ray_max_distance,
                    0..=2000,
                ));
                ui.end_row();

                ui.label("Shadow Bias");
                ui.add(egui::Slider::new(&mut config.shadow_bias, 0.0..=10.0));
                ui.end_row();

                ui.label("Shadow Spread");
                ui.add(egui::Slider::new(&mut config.shadow_spread, 0.0..=1.0).logarithmic(true));
                ui.end_row();

                ui.label("Per Voxel Secondary GI Bounces");
                ui.checkbox(&mut config.per_voxel_secondary, "");
                ui.end_row();

                ui.label("Ambient Filter Scale");
                ui.add(egui::Slider::new(
                    &mut config.ambient_filter_scale,
                    0.0..=100.0,
                ));
            });
        });

        ctx.config.view = self.view.1;
        ctx.config.tonemapping = self.tonemapping.1;

        if ctx.config.max_fps.is_some() != self.limit_fps
            || ctx.config.max_fps.is_some_and(|m| m != self.max_fps)
        {
            ctx.config.max_fps = match self.limit_fps {
                false => None,
                true => Some(self.max_fps),
            }
        }
    }
}

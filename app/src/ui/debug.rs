use std::{f64::consts::PI, time::Duration};

use crate::{config::DebugView, ui::UiCtx};

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
    limit_fps: bool,
    max_fps: u32,
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

                ui.label("FXAA");
                ui.checkbox(&mut config.fxaa, "");
                ui.end_row();

                ui.label("TAA");
                ui.checkbox(&mut config.taa, "");
                ui.end_row();

                ui.label("View");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", config.view))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut config.view, DebugView::Composite, "Composite");
                        ui.selectable_value(&mut config.view, DebugView::Depth, "Depth");
                        ui.selectable_value(&mut config.view, DebugView::Albedo, "Albedo");
                        ui.selectable_value(&mut config.view, DebugView::HitNormal, "Hit normal");
                        ui.selectable_value(
                            &mut config.view,
                            DebugView::SurfaceNormal,
                            "Surface normal",
                        );
                        ui.selectable_value(&mut config.view, DebugView::Ambient, "Ambient");
                        ui.selectable_value(&mut config.view, DebugView::Shadow, "Shadow");
                        ui.selectable_value(&mut config.view, DebugView::Velocity, "Velocity");
                        ui.selectable_value(&mut config.view, DebugView::SkyAlbedo, "Sky Albedo");
                        ui.selectable_value(
                            &mut config.view,
                            DebugView::SkyIrradiance,
                            "Sky Irradiance",
                        );
                        ui.selectable_value(
                            &mut config.view,
                            DebugView::SkyPrefiler,
                            "Sky Prefilter",
                        );
                    });
                ui.end_row();

                if ui.button("Reload Scene").clicked() {
                    let src = std::include_bytes!("../../assets/models/sponza.glb");
                    let mut src = std::io::BufReader::new(std::io::Cursor::new(src));
                    generate::voxelize(&mut src, ctx.device, ctx.queue, None).unwrap();
                }
            });

            grid(ui, "Lighting", |ui| {
                ui.label("Azimuth");
                ui.add(
                    egui::Slider::new(
                        &mut config.sun_azimuth,
                        -std::f32::consts::PI..=std::f32::consts::PI,
                    )
                    .suffix(" rad"),
                );
                ui.end_row();

                ui.label("Altitutde");
                ui.add(
                    egui::Slider::new(
                        &mut config.sun_altitude,
                        -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
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

                ui.label("Ambient Max Distance");
                ui.add(egui::Slider::new(
                    &mut config.ambient_ray_max_distance,
                    0..=2000,
                ));
                ui.end_row();

                ui.label("Shadow Bias");
                ui.add(egui::Slider::new(&mut config.shadow_bias, 0.0..=0.025));
                ui.end_row();

                ui.label("Shadow Spread");
                ui.add(egui::Slider::new(&mut config.shadow_spread, 0.0..=1.0).logarithmic(true));
                ui.end_row();

                ui.label("Filter Shadows");
                ui.checkbox(&mut config.filter_shadows, "");
                ui.end_row();

                ui.label("Shadow Filter Radius");
                ui.add(egui::Slider::new(
                    &mut config.shadow_filter_radius,
                    0.0..=20.0,
                ));
            });
        });

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

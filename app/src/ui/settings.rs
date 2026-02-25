use crate::{config::DebugView, ui::UiCtx};

#[derive(Debug, Default)]
pub struct SettingsView {
    limit_fps: bool,
    max_fps: u32,
}

impl SettingsView {
    pub fn ui(&mut self, ctx: &mut UiCtx, ui: &mut egui::Ui) {
        let config = &mut ctx.config;

        egui::Grid::new("grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Settings").strong());
                ui.end_row();

                if ui.button("Reload Scene").clicked() {
                    let src = std::include_bytes!("../../assets/models/sponza.glb");
                    let mut src = std::io::BufReader::new(std::io::Cursor::new(src));
                    generate::voxelize(&mut src, ctx.device, ctx.queue, None).unwrap();
                }
                ui.end_row();

                ui.end_row();
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
                    });
                ui.end_row();

                ui.end_row();
                ui.label(egui::RichText::new("Lighting").strong());
                ui.end_row();

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
                ui.end_row();
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

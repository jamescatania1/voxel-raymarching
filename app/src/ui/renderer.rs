use std::sync::Arc;

use crate::{
    SizedWindow,
    config::Config,
    ui::{debug::DebugWindow, settings::SettingsView},
};
use egui::epaint::{
    CornerRadiusF32,
    text::{FontInsert, InsertFontFamily},
};
use egui_wgpu::wgpu::StoreOp;
use egui_wgpu::{Renderer, RendererOptions, wgpu};
use egui_winit::State;

pub struct Ui {
    pub debug: DebugWindow,
    pub settings: SettingsView,
    renderer: Renderer,
    window_state: State,
    frame_started: bool,
}

pub struct UiCtx<'a, 'b> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub texture_view: &'b wgpu::TextureView,
    pub encoder: &'b mut wgpu::CommandEncoder,
    pub config: &'a mut Config,
}

impl Ui {
    pub fn new(
        window: &winit::window::Window,
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = egui::Context::default();

        ctx.global_style_mut(|style| {
            style.visuals.window_corner_radius = egui::CornerRadius::ZERO;
            style.visuals.window_shadow = egui::Shadow::NONE;
        });

        ctx.add_font(FontInsert::new(
            "figtree",
            egui::FontData::from_static(std::include_bytes!("../../assets/figtree.ttf")),
            vec![InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Highest,
            }],
        ));

        let window_state = egui_winit::State::new(
            ctx,
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024),
        );
        let renderer = Renderer::new(device, out_format, RendererOptions::default());

        Self {
            window_state,
            renderer,
            frame_started: false,
            debug: Default::default(),
            settings: Default::default(),
        }
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.window_state.on_window_event(window, event)
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        self.window_state.on_mouse_motion(delta);
    }

    pub fn frame<'a>(&mut self, ctx: &mut UiCtx) {
        let raw_input = self.window_state.take_egui_input(ctx.window);
        self.egui().begin_pass(raw_input);
        self.frame_started = true;

        egui::Window::new("Debug")
            .default_open(true)
            .resizable(true)
            .vscroll(true)
            .default_pos(egui::pos2(0.0, 0.0))
            .default_height(640.0)
            .show(self.window_state.egui_ctx(), |ui| {
                self.debug.ui(ctx, ui);
            });

        if !self.frame_started {
            return;
        }

        let descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: ctx.window.size().to_array(),
            pixels_per_point: ctx.window.scale_factor() as f32 * 1.0,
        };

        self.egui()
            .set_pixels_per_point(descriptor.pixels_per_point);

        let full_output = self.egui().end_pass();
        self.window_state
            .handle_platform_output(ctx.window, full_output.platform_output);

        let tris = self
            .egui()
            .tessellate(full_output.shapes, self.egui().pixels_per_point());
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(ctx.device, ctx.queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(ctx.device, ctx.queue, ctx.encoder, &tris, &descriptor);

        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.texture_view,
                resolve_target: None,
                ops: egui_wgpu::wgpu::Operations {
                    load: egui_wgpu::wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(&mut pass.forget_lifetime(), &tris, &descriptor);

        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }
        self.frame_started = false;
    }

    fn egui(&self) -> &egui::Context {
        self.window_state.egui_ctx()
    }
}

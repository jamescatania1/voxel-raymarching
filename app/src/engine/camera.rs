use std::{
    f64::{EPSILON, consts::PI},
    time::Duration,
};

use winit::keyboard::KeyCode;

use crate::engine::input::Input;

#[derive(Debug, Default)]
pub struct Camera {
    pub position: glam::DVec3,
    pub velocity: glam::DVec3,
    pub forward: glam::DVec3,
    pub rotation: glam::DVec3,
    pub fov: f64,
    pub near: f64,
    pub far: f64,
    pub size: glam::UVec2,
    pub view_proj: glam::DMat4,
    pub inv_view_proj: glam::DMat4,
}

impl Camera {
    const UP: glam::DVec3 = glam::DVec3::Z;

    pub fn new(size: glam::UVec2) -> Self {
        let mut _self = Self {
            // position: glam::dvec3(8.0, 14.0, 3.0),
            position: glam::dvec3(9.0, 14.0, 5.0),
            velocity: glam::DVec3::ZERO,
            rotation: glam::dvec3(0.0, 0.0, 0.0),
            fov: 90.0 * (PI / 180.0),
            near: 0.01,
            far: 100.0,
            size,
            view_proj: glam::DMat4::IDENTITY,
            inv_view_proj: glam::DMat4::IDENTITY,
            ..Default::default()
        };

        return _self;
    }

    pub fn update(&mut self, delta_time: &Duration, input: &Input, cursor_locked: bool) {
        let mut in_vec = glam::ivec2(
            input.key_down(KeyCode::KeyD) as i32 - input.key_down(KeyCode::KeyA) as i32,
            input.key_down(KeyCode::KeyW) as i32 - input.key_down(KeyCode::KeyS) as i32,
        )
        .as_dvec2();
        in_vec /= in_vec.length().max(1.0);
        // if in_vec.y.abs() < 0.001 {
        //     in_vec.y = 0.1;
        // }

        if cursor_locked {
            const MOUSE_SENSITIVITY: f64 = 0.001;
            self.rotation.z -= input.mouse.delta.x * MOUSE_SENSITIVITY;
            self.rotation.x -= input.mouse.delta.y * MOUSE_SENSITIVITY;
            self.rotation.x = self
                .rotation
                .x
                .clamp(-PI / 2.0 + EPSILON, PI / 2.0 - EPSILON);
        }
        self.forward = glam::dvec3(
            self.rotation.z.cos() * self.rotation.x.cos(),
            self.rotation.z.sin() * self.rotation.x.cos(),
            self.rotation.x.sin(),
        )
        .normalize();
        let right = -Self::UP.cross(self.forward);

        let targ_velocity = in_vec.x * right + in_vec.y * self.forward;

        let delta_ms = (delta_time.as_secs_f64() * 1000.0).clamp(0.1, 1000.0);
        self.velocity = self.velocity.lerp(targ_velocity, delta_ms * 0.01);

        self.position += self.velocity * (delta_ms * 0.005);

        let view = glam::DMat4::look_at_rh(self.position, self.position + self.forward, Self::UP);
        let proj = glam::DMat4::perspective_rh(
            self.fov,
            self.size.x as f64 / self.size.y.max(1) as f64,
            self.near,
            self.far,
        );
        self.view_proj = proj * view;
        self.inv_view_proj = self.view_proj.inverse();
    }
}

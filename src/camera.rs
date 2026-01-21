use glam::UVec2;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj_matrix: [[f32; 4]; 4],
}

#[derive(Debug)]
pub struct Camera {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub uniform: CameraUniform,
}

impl Camera {
    const UP: glam::Vec3 = glam::Vec3::Z;

    pub fn new(size: UVec2) -> Self {
        let mut _self = Self {
            position: glam::Vec3::NEG_Y * 5.0,
            forward: glam::Vec3::Y,
            fov: 45.0,
            near: 0.01,
            far: 100.0,
            uniform: CameraUniform::default(),
        };

        _self.update(size);

        return _self;
    }

    pub fn update(&mut self, size: UVec2) {
        let view = glam::Mat4::look_at_rh(self.position, self.position + self.forward, Self::UP);
        let proj = glam::Mat4::perspective_rh(
            self.fov,
            size.x as f32 / size.y.max(1) as f32,
            self.near,
            self.far,
        );
        let view_proj = proj * view;

        self.uniform.view_proj_matrix = view_proj.to_cols_array_2d();
    }
}

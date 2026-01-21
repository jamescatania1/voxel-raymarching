#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniform {
    pub matrix: [[f32; 4]; 4],
}

/// A 3D model instance with a world transform
#[derive(Debug, Clone)]
pub struct Model {
    pub position: glam::Vec3,
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
    pub uniform: ModelUniform,
}

impl Model {
    pub fn new() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: glam::Vec3::ONE,
            uniform: ModelUniform {
                matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
        }
    }

    /// Updates the model's uniform world matrix
    pub fn update(&mut self) {
        let matrix = glam::Mat4::from_scale_rotation_translation(
            self.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotation.x,
                self.rotation.y,
                self.rotation.z,
            ),
            self.position,
        );
        self.uniform.matrix = matrix.to_cols_array_2d();
    }
}

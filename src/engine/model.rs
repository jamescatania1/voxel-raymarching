/// A 3D model instance with a world transform
#[derive(Debug, Clone)]
pub struct Model {
    pub position: glam::Vec3,
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
    pub transform: glam::Mat4,
    pub inv_transform: glam::Mat4,
}

impl Model {
    pub fn new() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Vec3::ZERO,
            scale: glam::Vec3::ONE * 0.05,
            transform: glam::Mat4::IDENTITY,
            inv_transform: glam::Mat4::IDENTITY,
        }
    }

    /// Updates the model's uniform world matrix
    pub fn update(&mut self) {
        self.transform = glam::Mat4::from_scale_rotation_translation(
            self.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotation.x,
                self.rotation.y,
                self.rotation.z,
            ),
            self.position,
        );
        self.inv_transform = self.transform.inverse();
    }
}

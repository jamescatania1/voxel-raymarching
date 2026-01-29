/// A 3D model instance with a world transform
#[derive(Debug, Clone)]
pub struct Model {
    pub position: glam::DVec3,
    pub rotation: glam::DVec3,
    pub scale: glam::DVec3,
    pub transform: glam::DMat4,
    pub inv_transform: glam::DMat4,
}

impl Model {
    pub fn new() -> Self {
        Self {
            position: glam::DVec3::ZERO,
            rotation: glam::DVec3::ZERO,
            scale: glam::DVec3::ONE * 0.05,
            transform: glam::DMat4::IDENTITY,
            inv_transform: glam::DMat4::IDENTITY,
        }
    }

    /// Updates the model's uniform world matrix
    pub fn update(&mut self) {
        self.transform = glam::DMat4::from_scale_rotation_translation(
            self.scale,
            glam::DQuat::from_euler(
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

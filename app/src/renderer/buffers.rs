#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvironmentDataBuffer {
    pub sun_direction: glam::Vec3,
    pub shadow_bias: f32,
    pub camera: CameraDataBuffer,
    pub prev_camera: CameraDataBuffer,
    pub shadow_spread: f32,
    pub filter_shadows: u32,
    pub shadow_filter_radius: f32,
    pub max_ambient_distance: u32,
    pub voxel_normal_factor: f32,
    pub indirect_sky_intensity: f32,
    pub debug_view: u32,
    pub pad: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelMapInfoBuffer {
    pub visible_count: u32,
    pub failed_to_insert_count: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraDataBuffer {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub ws_position: [f32; 3],
    pub _pad_0: f32,
    pub forward: [f32; 3],
    pub near: f32,
    pub jitter: [f32; 2],
    pub far: f32,
    pub fov: f32,
}

impl CameraDataBuffer {
    pub fn from_camera(camera: &crate::engine::Camera, jitter: glam::DVec2) -> Self {
        Self {
            view_proj: camera.view_proj.as_mat4().to_cols_array_2d(),
            inv_view_proj: camera.inv_view_proj.as_mat4().to_cols_array_2d(),
            ws_position: camera.position.as_vec3().to_array(),
            forward: camera.forward.as_vec3().to_array(),
            near: camera.near as f32,
            jitter: jitter.as_vec2().to_array(),
            far: camera.far as f32,
            fov: camera.fov as f32,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],
    pub normal_transform: [[f32; 4]; 3],
    pub inv_normal_transform: [[f32; 4]; 3],
}

impl ModelDataBuffer {
    pub fn update(&mut self, model: &crate::engine::Model) {
        self.transform = model.transform.as_mat4().to_cols_array_2d();
        self.inv_transform = model.inv_transform.as_mat4().to_cols_array_2d();
        self.normal_transform = glam::DMat3::from_mat4(model.inv_transform)
            .transpose()
            .as_mat3()
            .to_cols_array_2d()
            .map(|v| [v[0], v[1], v[2], 0.0]);
        self.inv_normal_transform = glam::DMat3::from_mat4(model.transform)
            .transpose()
            .as_mat3()
            .to_cols_array_2d()
            .map(|v| [v[0], v[1], v[2], 0.0]);
    }
}

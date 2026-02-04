#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneDataBuffer {
    size: [u32; 3],
    _pad_2: u32,
    index_bounds: [u32; 3],
    _pad_1: u32,
    palette: [u32; 256],
}

impl SceneDataBuffer {
    pub fn new(scene: &vox::Scene, index_bounds: glam::UVec3) -> Self {
        let mut palette = [0; 256];
        for (i, mat) in scene.palette.iter().enumerate() {
            let rgba = mat.rgba;
            palette[i] = (rgba[0] as u32) << 24
                | (rgba[1] as u32) << 16
                | (rgba[2] as u32) << 8
                | rgba[3] as u32;
        }
        Self {
            size: [
                scene.size.x as u32,
                scene.size.y as u32,
                scene.size.z as u32,
            ],
            _pad_1: 0,
            index_bounds: index_bounds.to_array(),
            _pad_2: 0,
            palette,
        }
    }
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
    pub far: f32,
    pub fov: f32,
    pub _pad_1: f32,
    pub _pad_2: f32,
}

impl CameraDataBuffer {
    pub fn update(&mut self, camera: &crate::engine::Camera) {
        self.view_proj = camera.view_proj.as_mat4().to_cols_array_2d();
        self.inv_view_proj = camera.inv_view_proj.as_mat4().to_cols_array_2d();
        self.ws_position = camera.position.as_vec3().to_array();
        self.forward = camera.forward.as_vec3().to_array();
        self.near = camera.near as f32;
        self.far = camera.far as f32;
        self.fov = camera.fov as f32;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],
    pub normal_transform: [[f32; 4]; 3],
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
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneDataBuffer {
    size: [u32; 3],
    _pad: u32,
    palette: [u32; 256],
}

impl SceneDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
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
            _pad: 0,
            palette,
        }
    }
}

pub struct VoxelDataBuffer(pub Vec<u8>);

impl VoxelDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut voxels = vec![0; scene.size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;

                voxels[index] = palette_index;
            }
        }

        println!("voxel data load took {:#?}", timer.elapsed());
        Self(voxels)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraDataBuffer {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub ws_position: [f32; 3],
    pub _pad: f32,
}

impl CameraDataBuffer {
    pub fn update(&mut self, camera: &crate::engine::Camera) {
        self.view_proj = camera.view_proj.to_cols_array_2d();
        self.inv_view_proj = camera.inv_view_proj.to_cols_array_2d();
        self.ws_position = camera.position.to_array();
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],
}

impl ModelDataBuffer {
    pub fn update(&mut self, model: &crate::engine::Model) {
        self.transform = model.transform.to_cols_array_2d();
        self.inv_transform = model.inv_transform.to_cols_array_2d();
    }
}

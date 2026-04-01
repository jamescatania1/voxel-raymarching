pub struct ProbeGrid {
    pub size: glam::UVec3,
    pub scale: u32,
    pub probes: Vec<Probe>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Probe {
    pub position: glam::UVec3,
    pub _pad: u32,
}

impl ProbeGrid {
    pub fn new(size: glam::UVec3, scale: u32) -> Self {
        let size = size.map(|x| x.div_ceil(scale));

        let mut probes = Vec::with_capacity(size.element_product() as usize);
        for z in 0..size.x {
            for y in 0..size.y {
                for x in 0..size.x {
                    probes.push(Probe {
                        position: glam::uvec3(x, y, z) + scale / 2,
                        ..Default::default()
                    });
                }
            }
        }

        Self {
            size,
            scale,
            probes,
        }
    }
}

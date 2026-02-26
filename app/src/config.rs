#[derive(Debug)]
pub struct Config {
    pub render_scale: f32,
    pub sun_altitude: f32,
    pub sun_azimuth: f32,
    pub shadow_bias: f32,
    pub shadow_spread: f32,
    pub filter_shadows: bool,
    pub shadow_filter_radius: f32,
    pub voxel_normal_factor: f32,
    pub ambient_ray_max_distance: u32,
    pub view: DebugView,
    pub fxaa: bool,
    pub taa: bool,
    pub max_fps: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            render_scale: 0.5,
            sun_azimuth: -2.5,
            sun_altitude: 1.3,
            shadow_bias: 0.0005,
            shadow_spread: 0.05,
            filter_shadows: true,
            shadow_filter_radius: 7.0,
            voxel_normal_factor: 0.5,
            ambient_ray_max_distance: 10,
            view: DebugView::Composite,
            fxaa: false,
            taa: true,
            max_fps: None,
        }
    }
}

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub enum DebugView {
    #[default]
    Composite = 0,
    Albedo = 1,
    Depth = 2,
    HitNormal = 3,
    SurfaceNormal = 4,
    Shadow = 5,
    Ambient = 6,
    Velocity = 7,
    SkyAlbedo = 8,
    SkyIrradiance = 9,
    SkyPrefiler = 10,
}

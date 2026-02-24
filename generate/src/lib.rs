pub mod gltf;
mod lighting;
mod voxelize;

pub use lighting::*;
pub use voxelize::*;

pub const MODEL_FILE_EXT: &'static str = "voxel";

#![allow(unused)]

use crate::layout::ext::{LayoutBindingType, LayoutEntry};

/// A texture binding.
///
/// Example WGSL syntax:
/// ```rust,ignore
/// @group(0) @binding(0)
/// var t: texture_2d<f32>;
/// ```
///
/// Example GLSL syntax:
/// ```cpp,ignore
/// layout(binding = 0)
/// uniform texture2D t;
/// ```
///
/// Corresponds to [WebGPU `GPUTextureBindingLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturebindinglayout).
pub fn sampled_texture() -> TextureBase {
    TextureBase {}
}

pub struct TextureBase {}

impl TextureBase {
    /// Sampling returns floats.
    ///
    /// Able to be filtered with either a filtering, or non-filtering sampler.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<f32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    pub fn float(self) -> TextureSampleType {
        TextureSampleType {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
        }
    }
    /// Sampling returns unfilterable floats.
    ///
    /// Only able to be filtered with a non-filtering sampler.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<f32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    pub fn unfilterable_float(self) -> TextureSampleType {
        TextureSampleType {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
        }
    }
    /// Sampling does the depth reference comparison.
    ///
    /// This is also compatible with a non-filtering sampler.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_depth_2d;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2DShadow t;
    /// ```
    pub fn depth(self) -> TextureSampleType {
        TextureSampleType {
            sample_type: wgpu::TextureSampleType::Depth,
        }
    }
    /// Sampling returns signed integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<i32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform itexture2D t;
    /// ```
    pub fn signed_int(self) -> TextureSampleType {
        TextureSampleType {
            sample_type: wgpu::TextureSampleType::Sint,
        }
    }
    /// Sampling returns unsigned integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<u32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform utexture2D t;
    /// ```
    pub fn unsigned_int(self) -> TextureSampleType {
        TextureSampleType {
            sample_type: wgpu::TextureSampleType::Uint,
        }
    }
}

pub struct TextureSampleType {
    sample_type: wgpu::TextureSampleType,
}

impl TextureSampleType {
    /// A one dimensional texture. `texture_1d` in WGSL and `texture1D` in GLSL.
    pub fn dimension_1d(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::D1,
                multisampled: false,
            },
            count: None,
        }
    }
    /// A two dimensional texture. `texture_2d` in WGSL and `texture2D` in GLSL.
    pub fn dimension_2d(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }
    /// A two dimensional array texture. `texture_2d_array` in WGSL and `texture2DArray` in GLSL.
    pub fn dimension_2d_array(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }
    }
    /// A three dimensional texture. `texture_3d` in WGSL and `texture3D` in GLSL.
    pub fn dimension_3d(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }
    }
    /// A cubemap texture. `texture_cube` in WGSL and `textureCube` in GLSL.
    pub fn dimension_cube(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
            count: None,
        }
    }
    /// A cubemap array texture. `texture_cube_array` in WGSL and `textureCubeArray` in GLSL.
    pub fn dimension_cube_array(self) -> LayoutEntry<Texture> {
        LayoutEntry {
            visibility: wgpu::ShaderStages::NONE,
            ty: Texture {
                sample_type: self.sample_type,
                view_dimension: wgpu::TextureViewDimension::CubeArray,
                multisampled: false,
            },
            count: None,
        }
    }
}

pub struct Texture {
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
}

impl LayoutBindingType for Texture {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            sample_type: self.sample_type,
            view_dimension: self.view_dimension,
            multisampled: self.multisampled,
        }
    }
}

impl LayoutEntry<Texture> {
    /// This texture is multisampled, i.e., has a sample count greater than 1.
    /// This texture must be declared as `texture_multisampled_2d` or
    /// `texture_depth_multisampled_2d` in the shader, and read using `textureLoad`.
    pub fn multisampled(mut self, value: bool) -> Self {
        self.ty.multisampled = value;
        self
    }
}

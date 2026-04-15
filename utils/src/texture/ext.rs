use wgpu::{BindingResource, Extent3d, Texture, TextureFormat, TextureUsages, TextureView};

use crate::textures::{DeviceSwapExt, SwapchainTexture, SwapchainTextureView};

pub trait TextureDescriptorExt {
    fn create(self, device: &wgpu::Device) -> Texture;
    fn create_swap(self, device: &wgpu::Device) -> SwapchainTexture;
}
impl TextureDescriptorExt for wgpu::TextureDescriptor<'static> {
    fn create(self, device: &wgpu::Device) -> Texture {
        device.create_texture(&self)
    }

    fn create_swap(self, device: &wgpu::Device) -> SwapchainTexture {
        device.create_texture_swap(&self)
    }
}

pub trait TextureExt<T> {
    fn view(&self) -> T;
}
impl TextureExt<TextureView> for Texture {
    fn view(&self) -> TextureView {
        self.create_view(&Default::default())
    }
}
impl TextureExt<SwapchainTextureView> for SwapchainTexture {
    fn view(&self) -> SwapchainTextureView {
        self.create_view(&Default::default())
    }
}

pub trait TextureViewExt {
    fn as_binding(&self) -> BindingResource;
}
impl TextureViewExt for TextureView {
    fn as_binding(&self) -> BindingResource {
        BindingResource::TextureView(self)
    }
}

/// Describes a [`Texture`](../wgpu/struct.Texture.html).
///
/// Corresponds to [WebGPU `GPUTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturedescriptor).
pub fn texture(label: &'static str) -> TextureDescriptorBase {
    TextureDescriptorBase { label }
}

pub struct TextureDescriptorBase {
    label: &'static str,
}

impl TextureDescriptorBase {
    // Normal 8 bit formats
    /// Red channel only. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    pub fn r8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R8Unorm,
        }
    }
    /// Red channel only. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    pub fn r8snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R8Snorm,
        }
    }
    /// Red channel only. 8 bit integer per channel. Unsigned in shader.
    pub fn r8uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R8Uint,
        }
    }
    /// Red channel only. 8 bit integer per channel. Signed in shader.
    pub fn r8sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R8Sint,
        }
    }

    // Normal 16 bit formats
    /// Red channel only. 16 bit integer per channel. Unsigned in shader.
    pub fn r16uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R16Uint,
        }
    }
    /// Red channel only. 16 bit integer per channel. Signed in shader.
    pub fn r16sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R16Sint,
        }
    }
    /// Red channel only. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn r16unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R16Unorm,
        }
    }
    /// Red channel only. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn r16snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R16Snorm,
        }
    }
    /// Red channel only. 16 bit float per channel. Float in shader.
    pub fn r16float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R16Float,
        }
    }
    /// Red and green channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    pub fn rg8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg8Unorm,
        }
    }
    /// Red and green channels. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    pub fn rg8snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg8Snorm,
        }
    }
    /// Red and green channels. 8 bit integer per channel. Unsigned in shader.
    pub fn rg8uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg8Uint,
        }
    }
    /// Red and green channels. 8 bit integer per channel. Signed in shader.
    pub fn rg8sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg8Sint,
        }
    }

    // Normal 32 bit formats
    /// Red channel only. 32 bit integer per channel. Unsigned in shader.
    pub fn r32uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R32Uint,
        }
    }
    /// Red channel only. 32 bit integer per channel. Signed in shader.
    pub fn r32sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R32Sint,
        }
    }
    /// Red channel only. 32 bit float per channel. Float in shader.
    pub fn r32float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R32Float,
        }
    }
    /// Red and green channels. 16 bit integer per channel. Unsigned in shader.
    pub fn rg16uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg16Uint,
        }
    }
    /// Red and green channels. 16 bit integer per channel. Signed in shader.
    pub fn rg16sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg16Sint,
        }
    }
    /// Red and green channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn rg16unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg16Unorm,
        }
    }
    /// Red and green channels. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn rg16snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg16Snorm,
        }
    }
    /// Red and green channels. 16 bit float per channel. Float in shader.
    pub fn rg16float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg16Float,
        }
    }
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    pub fn rgba8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba8Unorm,
        }
    }
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    pub fn rgba8unormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba8UnormSrgb,
        }
    }
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    pub fn rgba8snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba8Snorm,
        }
    }
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Unsigned in shader.
    pub fn rgba8uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba8Uint,
        }
    }
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Signed in shader.
    pub fn rgba8sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba8Sint,
        }
    }
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    pub fn bgra8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bgra8Unorm,
        }
    }
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    pub fn bgra8unormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bgra8UnormSrgb,
        }
    }

    // Packed 32 bit formats
    /// Packed unsigned float with 9 bits mantisa for each RGB component, then a common 5 bits exponent
    pub fn rgb9e5ufloat(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgb9e5Ufloat,
        }
    }
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. Unsigned in shader.
    pub fn rgb10a2uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgb10a2Uint,
        }
    }
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. [0, 1023] ([0, 3] for alpha) converted to/from float [0, 1] in shader.
    pub fn rgb10a2unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgb10a2Unorm,
        }
    }
    /// Red, green, and blue channels. 11 bit float with no sign bit for RG channels. 10 bit float with no sign bit for blue channel. Float in shader.
    pub fn rg11b10ufloat(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg11b10Ufloat,
        }
    }

    // Normal 64 bit formats
    /// Red channel only. 64 bit integer per channel. Unsigned in shader.
    ///
    /// [`Features::TEXTURE_INT64_ATOMIC`] must be enabled to use this texture format.
    pub fn r64uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::R64Uint,
        }
    }
    /// Red and green channels. 32 bit integer per channel. Unsigned in shader.
    pub fn rg32uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg32Uint,
        }
    }
    /// Red and green channels. 32 bit integer per channel. Signed in shader.
    pub fn rg32sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg32Sint,
        }
    }
    /// Red and green channels. 32 bit float per channel. Float in shader.
    pub fn rg32float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rg32Float,
        }
    }
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Unsigned in shader.
    pub fn rgba16uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba16Uint,
        }
    }
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Signed in shader.
    pub fn rgba16sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba16Sint,
        }
    }
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn rgba16unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba16Unorm,
        }
    }
    /// Red, green, blue, and alpha. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    pub fn rgba16snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba16Snorm,
        }
    }
    /// Red, green, blue, and alpha channels. 16 bit float per channel. Float in shader.
    pub fn rgba16float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba16Float,
        }
    }

    // Normal 128 bit formats
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Unsigned in shader.
    pub fn rgba32uint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba32Uint,
        }
    }
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Signed in shader.
    pub fn rgba32sint(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba32Sint,
        }
    }
    /// Red, green, blue, and alpha channels. 32 bit float per channel. Float in shader.
    pub fn rgba32float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Rgba32Float,
        }
    }

    // Depth and stencil formats
    /// Stencil format with 8 bit integer stencil.
    pub fn stencil8(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Stencil8,
        }
    }
    /// Special depth format with 16 bit integer depth.
    pub fn depth16unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Depth16Unorm,
        }
    }
    /// Special depth format with at least 24 bit integer depth.
    pub fn depth24plus(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Depth24Plus,
        }
    }
    /// Special depth/stencil format with at least 24 bit integer depth and 8 bits integer stencil.
    pub fn depth24plusstencil8(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Depth24PlusStencil8,
        }
    }
    /// Special depth format with 32 bit floating point depth.
    pub fn depth32float(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Depth32Float,
        }
    }
    /// Special depth/stencil format with 32 bit floating point depth and 8 bits integer stencil.
    ///
    /// [`Features::DEPTH32FLOAT_STENCIL8`] must be enabled to use this texture format.
    pub fn depth32floatstencil8(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Depth32FloatStencil8,
        }
    }

    /// YUV 4:2:0 chroma subsampled format.
    ///
    /// Contains two planes:
    /// - 0: Single 8 bit channel luminance.
    /// - 1: Dual 8 bit channel chrominance at half width and half height.
    ///
    /// Valid view formats for luminance are [`TextureFormat::R8Unorm`].
    ///
    /// Valid view formats for chrominance are [`TextureFormat::Rg8Unorm`].
    ///
    /// Width and height must be even.
    ///
    /// [`Features::TEXTURE_FORMAT_NV12`] must be enabled to use this texture format.
    pub fn nv12(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::NV12,
        }
    }

    /// YUV 4:2:0 chroma subsampled format.
    ///
    /// Contains two planes:
    /// - 0: Single 16 bit channel luminance, of which only the high 10 bits
    ///   are used.
    /// - 1: Dual 16 bit channel chrominance at half width and half height, of
    ///   which only the high 10 bits are used.
    ///
    /// Valid view formats for luminance are [`TextureFormat::R16Unorm`].
    ///
    /// Valid view formats for chrominance are [`TextureFormat::Rg16Unorm`].
    ///
    /// Width and height must be even.
    ///
    /// [`Features::TEXTURE_FORMAT_P010`] must be enabled to use this texture format.
    pub fn p010(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::P010,
        }
    }

    // Compressed textures usable with `TEXTURE_COMPRESSION_BC` feature. `TEXTURE_COMPRESSION_SLICED_3D` is required to use with 3D textures.
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// [0, 63] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc1rgbaunorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc1RgbaUnorm,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// Srgb-color [0, 63] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc1rgbaunormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc1RgbaUnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// [0, 63] ([0, 15] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc2rgbaunorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc2RgbaUnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc2rgbaunormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc2RgbaUnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// [0, 63] ([0, 255] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc3rgbaunorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc3RgbaUnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc3rgbaunormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc3RgbaUnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc4runorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc4RUnorm,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc4rsnorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc4RSnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc5rgunorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc5RgUnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc5rgsnorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc5RgSnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit unsigned float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc6hrgbufloat(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc6hRgbUfloat,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit signed float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc6hrgbfloat(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc6hRgbFloat,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc7rgbaunorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc7RgbaUnorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    pub fn bc7rgbaunormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Bc7RgbaUnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgb8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgb8Unorm,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgb8unormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgb8UnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// [0, 255] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgb8a1unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgb8A1Unorm,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// Srgb-color [0, 255] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgb8a1unormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgb8A1UnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgba8unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgba8Unorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn etc2rgba8unormsrgb(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Etc2Rgba8UnormSrgb,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn eacr11unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::EacR11Unorm,
        }
    }
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn eacr11snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::EacR11Snorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn eacrg11unorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::EacRg11Unorm,
        }
    }
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    pub fn eacrg11snorm(self) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::EacRg11Snorm,
        }
    }
    /// block compressed texture. 16 bytes per block.
    ///
    /// Features [`TEXTURE_COMPRESSION_ASTC`] or [`TEXTURE_COMPRESSION_ASTC_HDR`]
    /// must be enabled to use this texture format.
    ///
    /// [`TEXTURE_COMPRESSION_ASTC`]: Features::TEXTURE_COMPRESSION_ASTC
    /// [`TEXTURE_COMPRESSION_ASTC_HDR`]: Features::TEXTURE_COMPRESSION_ASTC_HDR
    ///
    /// ## Parameters
    ///
    /// `block` - compressed block dimensions
    ///
    /// `channel` - ASTC RGBA channel
    pub fn astc(
        self,
        block: wgpu::AstcBlock,
        channel: wgpu::AstcChannel,
    ) -> TextureDescriptorFormat {
        TextureDescriptorFormat {
            label: self.label,
            format: TextureFormat::Astc { block, channel },
        }
    }
}

pub struct TextureDescriptorFormat {
    label: &'static str,
    format: TextureFormat,
}

impl TextureDescriptorFormat {
    /// Size of the texture. All components must be greater than zero. For a
    /// regular 1D/2D texture, the unused sizes will be 1. For 2DArray textures,
    /// Z is the number of 2D textures in that array.
    pub fn size(self, size: glam::UVec3) -> TextureDescriptorSized {
        TextureDescriptorSized {
            label: self.label,
            format: self.format,
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: size.z,
            },
        }
    }
}

pub struct TextureDescriptorSized {
    label: &'static str,
    format: TextureFormat,
    size: Extent3d,
}

impl TextureDescriptorSized {
    /// Specifies a texture that has one dimension, width. "1d" textures cannot have mipmaps, be multisampled, use compressed or depth/stencil formats, or be used as a render target.
    pub fn d1(self) -> TextureDescriptorDimension {
        TextureDescriptorDimension {
            label: self.label,
            format: self.format,
            size: self.size,
            dimension: wgpu::TextureDimension::D1,
        }
    }
    /// Specifies a texture that has a width and height, and may have layers.
    pub fn d2(self) -> TextureDescriptorDimension {
        TextureDescriptorDimension {
            label: self.label,
            format: self.format,
            size: self.size,
            dimension: wgpu::TextureDimension::D2,
        }
    }
    /// Specifies a texture that has a width, height, and depth. "3d" textures cannot be multisampled, and their format must support 3d textures (all plain color formats and some packed/compressed formats).
    pub fn d3(self) -> TextureDescriptorDimension {
        TextureDescriptorDimension {
            label: self.label,
            format: self.format,
            size: self.size,
            dimension: wgpu::TextureDimension::D3,
        }
    }
}

pub struct TextureDescriptorDimension {
    label: &'static str,
    format: TextureFormat,
    size: Extent3d,
    dimension: wgpu::TextureDimension,
}

impl TextureDescriptorDimension {
    /// Allowed usages of the texture. If used in other ways, the operation will panic.
    pub fn usage(self, usage: TextureUsages) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: Some(self.label),
            size: self.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: self.dimension,
            format: self.format,
            usage,
            view_formats: &[],
        }
    }
}

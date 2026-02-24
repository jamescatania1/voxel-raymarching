use std::{
    collections::HashMap,
    io::{BufRead, Seek},
};

use anyhow::{Result, ensure};
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Transforms glTF coordinate space (+y, right handed) to (+z, right handed)
pub const GLTF_Y_UP_TO_Z_UP: glam::Mat4 = glam::mat4(
    glam::vec4(1.0, 0.0, 0.0, 0.0),
    glam::vec4(0.0, 0.0, 1.0, 0.0),
    glam::vec4(0.0, -1.0, 0.0, 0.0),
    glam::vec4(0.0, 0.0, 0.0, 1.0),
);

#[derive(Debug)]
pub struct Header {
    pub magic: u32,
    pub version: u32,
    pub length: u32,
}

#[derive(Debug)]
pub struct Gltf {
    pub header: Header,
    pub meta: GltfJson,
    pub bin: Vec<u8>,
}

impl Gltf {
    pub fn parse<R: BufRead + Seek>(src: &mut R) -> Result<Self> {
        let header = Header {
            magic: src.read_u32::<LittleEndian>()?,
            version: src.read_u32::<LittleEndian>()?,
            length: src.read_u32::<LittleEndian>()?,
        };
        ensure!(header.magic == 0x46546C67, "mismatched magic number");
        ensure!(header.version == 2, "only version 2.0 GLTF is supported");

        // json chunk
        let meta = {
            let length = src.read_u32::<LittleEndian>()?;
            let ty = src.read_u32::<LittleEndian>()?;
            ensure!(ty == 0x4E4F534A, "expected JSON chunk");
            let mut buf = vec![0; length as usize];
            src.read_exact(&mut buf)?;
            src.seek_relative((length - length & !3) as i64)?;

            // println!("{:#?}", serde_json::from_slice::<serde_json::Value>(&buf)?);
            serde_json::from_slice::<GltfJson>(&buf)?
        };

        let bin = {
            let length = src.read_u32::<LittleEndian>()?;
            let ty = src.read_u32::<LittleEndian>()?;
            ensure!(ty == 0x004E4942, "expected BIN chunk");
            let mut buf = vec![0; length as usize];
            src.read_exact(&mut buf)?;
            src.seek_relative((length - length & !3) as i64)?;

            buf
        };

        Ok(Self { header, bin, meta })
    }
}

/// The root object for a glTF asset.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GltfJson {
    /// Metadata about the glTF asset.
    pub asset: Asset,

    /// The index of the default scene.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene: Option<u32>,

    /// An array of scenes.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scenes: Vec<Scene>,

    /// An array of nodes.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<Node>,

    /// An array of meshes.  A mesh is a set of primitives to be rendered.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<Mesh>,

    /// An array of materials.  A material defines the appearance of a primitive.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<Material>,

    /// An array of images.  An image defines data used to create a texture.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,

    /// An array of samplers.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub samplers: Vec<Sampler>,

    /// An array of textures.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<Texture>,

    /// An array of accessors.  An accessor is a typed view into a bufferView.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accessors: Vec<Accessor>,

    /// An array of bufferViews.  A bufferView is a view into a buffer generally representing a subset of the buffer.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buffer_views: Vec<BufferView>,

    /// An array of buffers.  A buffer points to binary geometry, animation, or skins.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<Buffer>,
}

/// Metadata about the glTF asset.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    /// The glTF version in the form of `<major>.<minor>` that this asset targets.
    pub version: String,

    /// A copyright message suitable for display to credit the content creator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copyright: Option<String>,

    /// Tool that generated this glTF model.  Useful for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,

    /// The minimum glTF version in the form of `<major>.<minor>` that this asset targets. This property **MUST NOT** be greater than the asset version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_version: Option<String>,
}

/// A typed view into a buffer view that contains raw binary data.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Accessor {
    /// The index of the buffer view. When undefined, the accessor **MUST** be initialized with zeros; `sparse` property or extensions **MAY** override zeros with actual values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_view: Option<u32>,

    /// The offset relative to the start of the buffer view in bytes.  This **MUST** be a multiple of the size of the component datatype. This property **MUST NOT** be defined when `bufferView` is undefined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_offset: Option<u32>,

    /// The datatype of the accessor's components.
    pub component_type: ComponentType,

    /// Specifies whether integer data values are normalized (`true`) to [0, 1] (for unsigned types) or to [-1, 1] (for signed types) when they are accessed. This property **MUST NOT** be set to `true` for accessors with `FLOAT` or `UNSIGNED_INT` component type.
    #[serde(default)]
    pub normalized: bool,

    /// The number of elements referenced by this accessor, not to be confused with the number of bytes or number of components.
    pub count: u32,

    /// Specifies if the accessor's elements are scalars, vectors, or matrices.
    #[serde(rename = "type")]
    pub ty: AccessorType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Maximum value of each component in this accessor.  Array elements **MUST** be treated as having the same data type as accessor's `componentType`. Both `min` and `max` arrays have the same length.  The length is determined by the value of the `type` property; it can be 1, 2, 3, 4, 9, or 16.\n\n`normalized` property has no effect on array values: they always correspond to the actual values stored in the buffer. When the accessor is sparse, this property **MUST** contain maximum values of accessor data with sparse substitution applied.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<Vec<GltfNumber>>,

    /// Minimum value of each component in this accessor.  Array elements **MUST** be treated as having the same data type as accessor's `componentType`. Both `min` and `max` arrays have the same length.  The length is determined by the value of the `type` property; it can be 1, 2, 3, 4, 9, or 16.\n\n`normalized` property has no effect on array values: they always correspond to the actual values stored in the buffer. When the accessor is sparse, this property **MUST** contain minimum values of accessor data with sparse substitution applied.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<Vec<GltfNumber>>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GltfNumber {
    Integer(i64),
    Float(f64),
}

/// The datatype of the accessor's components.  UNSIGNED_INT type **MUST NOT** be used for any accessor that is not referenced by `mesh.primitive.indices`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ComponentType {
    Byte,
    UnsignedByte,
    Short,
    UnsignedShort,
    UnsignedInt,
    Float,
    Other(u32),
}
impl Serialize for ComponentType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::Byte => 5120,
            Self::UnsignedByte => 5121,
            Self::Short => 5122,
            Self::UnsignedShort => 5123,
            Self::UnsignedInt => 5125,
            Self::Float => 5126,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for ComponentType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(match value {
            5120 => Self::Byte,
            5121 => Self::UnsignedByte,
            5122 => Self::Short,
            5123 => Self::UnsignedShort,
            5125 => Self::UnsignedInt,
            5126 => Self::Float,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum AccessorType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
    Other(String),
}
impl Serialize for AccessorType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            AccessorType::Scalar => "SCALAR",
            AccessorType::Vec2 => "VEC2",
            AccessorType::Vec3 => "VEC3",
            AccessorType::Vec4 => "VEC4",
            AccessorType::Mat2 => "MAT2",
            AccessorType::Mat3 => "MAT3",
            AccessorType::Mat4 => "MAT4",
            AccessorType::Other(v) => v,
        })
    }
}
impl<'de> Deserialize<'de> for AccessorType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "SCALAR" => AccessorType::Scalar,
            "VEC2" => AccessorType::Vec2,
            "VEC3" => AccessorType::Vec3,
            "VEC4" => AccessorType::Vec4,
            "MAT2" => AccessorType::Mat2,
            "MAT3" => AccessorType::Mat3,
            "MAT4" => AccessorType::Mat4,
            other => AccessorType::Other(other.into()),
        })
    }
}

/// The root nodes of a scene.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Scene {
    /// The indices of each root node.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A set of primitives to be rendered.  Its global transform is defined by a node that references it.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mesh {
    /// An array of primitives, each defining geometry to be rendered.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub primitives: Vec<Primitive>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Primitive {
    /// The index of the material to apply to this primitive when rendering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<u32>,

    /// A plain JSON object, where each key corresponds to a mesh attribute semantic and each value is the index of the accessor containing attribute's data.
    #[serde(default)]
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, u32>,

    /// The index of the accessor that contains the vertex indices.  When this is undefined, the primitive defines non-indexed geometry.  When defined, the accessor **MUST** have `SCALAR` type and an unsigned integer component type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indices: Option<u32>,

    #[serde(default)]
    pub mode: DrawMode,
}

#[derive(Debug, PartialEq)]
pub enum DrawMode {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Other(u32),
}
impl Default for DrawMode {
    fn default() -> Self {
        Self::Triangles
    }
}
impl Serialize for DrawMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::Points => 0,
            Self::Lines => 1,
            Self::LineLoop => 2,
            Self::LineStrip => 3,
            Self::Triangles => 4,
            Self::TriangleStrip => 5,
            Self::TriangleFan => 6,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for DrawMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(match value {
            0 => Self::Points,
            1 => Self::Lines,
            2 => Self::LineLoop,
            3 => Self::LineStrip,
            4 => Self::Triangles,
            5 => Self::TriangleStrip,
            6 => Self::TriangleFan,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Transform {
    /// The node's transform matrix
    pub matrix: Mat4,
    /// The node's translation along the x, y, and z axes.
    pub translation: Vec3,
    /// The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
    pub rotation: Quat,
    /// The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
    pub scale: Vec3,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "RawNode", into = "RawNode")]
/// A node in the node hierarchy.
pub struct Node {
    pub name: Option<String>,
    /// The index of the mesh in this node.
    pub mesh: Option<u32>,
    /// The indices of this node's children.
    pub children: Vec<u32>,
    /// The local transform of this node
    pub transform: Transform,
}

#[derive(Debug, Serialize, Deserialize)]
struct RawNode {
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh: Option<u32>,

    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    matrix: Option<Mat4>,

    #[serde(skip_serializing_if = "Option::is_none")]
    translation: Option<Vec3>,

    #[serde(skip_serializing_if = "Option::is_none")]
    rotation: Option<Quat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    scale: Option<Vec3>,

    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl From<RawNode> for Node {
    fn from(value: RawNode) -> Self {
        let mut transform = Transform::default();
        if let Some(matrix) = value.matrix {
            transform.matrix = matrix;
            let trf = matrix.to_scale_rotation_translation();
            transform.scale = trf.0;
            transform.rotation = trf.1;
            transform.translation = trf.2;
        } else {
            if let Some(translation) = value.translation {
                transform.translation = translation;
            }
            if let Some(rotation) = value.rotation {
                transform.rotation = rotation;
            }
            if let Some(scale) = value.scale {
                transform.scale = scale;
            }
            transform.matrix = Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation,
            );
        }
        Self {
            mesh: value.mesh,
            name: value.name,
            children: value.children,
            transform,
        }
    }
}

impl From<Node> for RawNode {
    fn from(value: Node) -> Self {
        Self {
            mesh: value.mesh,
            name: value.name,
            children: value.children,
            matrix: Some(Mat4::from_scale_rotation_translation(
                value.transform.scale,
                value.transform.rotation,
                value.transform.translation,
            )),
            translation: None,
            rotation: None,
            scale: None,
        }
    }
}

/// A view into a buffer generally representing a subset of the buffer.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Buffer {
    /// The URI (or IRI) of the buffer.  Relative paths are relative to the current glTF asset.  Instead of referencing an external file, this field **MAY** contain a `data:`-URI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    /// The length of the buffer in bytes.
    pub byte_length: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A view into a buffer generally representing a subset of the buffer.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct BufferView {
    /// The index of the buffer.
    pub buffer: u32,

    /// The offset into the buffer in bytes.
    #[serde(default)]
    pub byte_offset: u32,

    /// The length of the bufferView in bytes.
    pub byte_length: u32,

    /// The stride, in bytes, between vertex attributes.  When this is not defined, data is tightly packed. When two or more accessors use the same buffer view, this field **MUST** be defined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_stride: Option<u32>,

    /// The hint representing the intended GPU buffer type to use with this buffer view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<BufferType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum BufferType {
    ArrayBuffer,
    ElementArrayBuffer,
    Other(u32),
}
impl From<u32> for BufferType {
    fn from(value: u32) -> Self {
        match value {
            34962 => Self::ArrayBuffer,
            34963 => Self::ElementArrayBuffer,
            _ => Self::Other(value),
        }
    }
}
impl Serialize for BufferType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::ArrayBuffer => 34962,
            Self::ElementArrayBuffer => 34963,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for BufferType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(u32::deserialize(deserializer)?.into())
    }
}

/// The material appearance of a primitive.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Material {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// A set of parameter values that are used to define the metallic-roughness material model from Physically Based Rendering (PBR) methodology. When undefined, all the default values of `pbrMetallicRoughness` **MUST** apply.
    pub pbr_metallic_roughness: Option<PbrMetallicRoughness>,

    /// The tangent space normal texture. The texture encodes RGB components with linear transfer function. Each texel represents the XYZ components of a normal vector in tangent space. The normal vectors use the convention +X is right and +Y is up. +Z points toward the viewer. If a fourth component (A) is present, it **MUST** be ignored. When undefined, the material does not have a tangent space normal texture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_texture: Option<NormalTextureInfo>,

    /// The occlusion texture. The occlusion values are linearly sampled from the R channel. Higher values indicate areas that receive full indirect lighting and lower values indicate no indirect lighting. If other channels are present (GBA), they **MUST** be ignored for occlusion calculations. When undefined, the material does not have an occlusion texture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occlusion_texture: Option<OcclusionTextureInfo>,

    /// The emissive texture. It controls the color and intensity of the light being emitted by the material. This texture contains RGB components encoded with the sRGB transfer function. If a fourth component (A) is present, it **MUST** be ignored. When undefined, the texture **MUST** be sampled as having `1.0` in RGB components.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive_texture: Option<TextureInfo>,

    /// The factors for the emissive color of the material. This value defines linear multipliers for the sampled texels of the emissive texture.
    #[serde(default = "default_vec3_zero")]
    pub emissive_factor: glam::Vec3,

    /// The material's alpha rendering mode enumeration specifying the interpretation of the alpha value of the base color.
    #[serde(default)]
    pub alpha_mode: Option<AlphaMode>,

    /// Specifies the cutoff threshold when in `MASK` alpha mode. If the alpha value is greater than or equal to this value then it is rendered as fully opaque, otherwise, it is rendered as fully transparent. A value greater than `1.0` will render the entire material as fully transparent. This value **MUST** be ignored for other alpha modes. When `alphaMode` is not defined, this value **MUST NOT** be defined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f32>,

    /// Specifies whether the material is double sided. When this value is false, back-face culling is enabled. When this value is true, back-face culling is disabled and double-sided lighting is enabled. The back-face **MUST** have its normals reversed before the lighting equation is evaluated.
    #[serde(default)]
    pub double_sided: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A set of parameter values that are used to define the metallic-roughness material model from Physically-Based Rendering (PBR) methodology.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PbrMetallicRoughness {
    /// The factors for the base color of the material. This value defines linear multipliers for the sampled texels of the base color texture.
    #[serde(default = "default_vec4_one")]
    pub base_color_factor: glam::Vec4,

    /// The factor for the metalness of the material. This value defines a linear multiplier for the sampled metalness values of the metallic-roughness texture.    #[serde(default = "default_scalar_factor")]
    #[serde(default = "default_scalar_one")]
    pub metallic_factor: f32,

    /// The factor for the roughness of the material. This value defines a linear multiplier for the sampled roughness values of the metallic-roughness texture.    #[serde(default = "default_scalar_factor")]
    #[serde(default = "default_scalar_one")]
    pub roughness_factor: f32,

    /// The base color texture. The first three components (RGB) **MUST** be encoded with the sRGB transfer function. They specify the base color of the material. If the fourth component (A) is present, it represents the linear alpha coverage of the material. Otherwise, the alpha coverage is equal to `1.0`. The `material.alphaMode` property specifies how alpha is interpreted. The stored texels **MUST NOT** be premultiplied. When undefined, the texture **MUST** be sampled as having `1.0` in all components.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_color_texture: Option<TextureInfo>,

    /// The metallic-roughness texture. The metalness values are sampled from the B channel. The roughness values are sampled from the G channel. These values **MUST** be encoded with a linear transfer function. If other channels are present (R or A), they **MUST** be ignored for metallic-roughness calculations. When undefined, the texture **MUST** be sampled as having `1.0` in G and B components.    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metallic_roughness_texture: Option<TextureInfo>,
}

#[derive(Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
    Other(String),
}
impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}
impl Serialize for AlphaMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            Self::Opaque => "OPAQUE",
            Self::Mask => "MASK",
            Self::Blend => "BLEND",
            Self::Other(v) => v,
        })
    }
}
impl<'de> Deserialize<'de> for AlphaMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "OPAQUE" => Self::Opaque,
            "MASK" => Self::Mask,
            "BLEND" => Self::Blend,
            _ => Self::Other(value),
        })
    }
}

/// A texture and its sampler.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Texture {
    /// The index of the sampler used by this texture. When undefined, a sampler with repeat wrapping and auto filtering **SHOULD** be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampler: Option<u32>,

    /// The index of the image used by this texture. When undefined, an extension or other mechanism **SHOULD** supply an alternate texture source, otherwise behavior is undefined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Reference to a texture.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TextureInfo {
    /// The index of the texture.
    pub index: u32,

    /// The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    /// This integer value is used to construct a string in the format `TEXCOORD_<set index>` which is a reference to a key in `mesh.primitives.attributes` (e.g. a value of `0` corresponds to `TEXCOORD_0`). A mesh primitive **MUST** have the corresponding texture coordinate attributes for the material to be applicable to it.
    #[serde(default)]
    pub tex_coord: u32,
}

/// Material Normal Texture Info
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct NormalTextureInfo {
    /// The index of the texture.
    pub index: u32,

    /// The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    /// This integer value is used to construct a string in the format `TEXCOORD_<set index>` which is a reference to a key in `mesh.primitives.attributes` (e.g. a value of `0` corresponds to `TEXCOORD_0`). A mesh primitive **MUST** have the corresponding texture coordinate attributes for the material to be applicable to it.
    #[serde(default)]
    pub tex_coord: u32,

    /// The scalar parameter applied to each normal vector of the texture. This value scales the normal vector in X and Y directions using the formula: `scaledNormal =  normalize((<sampled normal texture value> * 2.0 - 1.0) * vec3(<normal scale>, <normal scale>, 1.0))`.
    #[serde(default = "default_scalar_one")]
    pub scale: f32,
}

/// Material Occlusion Texture Info
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct OcclusionTextureInfo {
    /// The index of the texture.
    pub index: u32,

    /// The set index of texture's TEXCOORD attribute used for texture coordinate mapping.
    /// This integer value is used to construct a string in the format `TEXCOORD_<set index>` which is a reference to a key in `mesh.primitives.attributes` (e.g. a value of `0` corresponds to `TEXCOORD_0`). A mesh primitive **MUST** have the corresponding texture coordinate attributes for the material to be applicable to it.
    #[serde(default)]
    pub tex_coord: u32,

    /// A scalar parameter controlling the amount of occlusion applied. A value of `0.0` means no occlusion. A value of `1.0` means full occlusion. This value affects the final occlusion value as: `1.0 + strength * (<sampled occlusion texture value> - 1.0)`.
    #[serde(default = "default_scalar_one")]
    pub strength: f32,
}

/// Texture sampler properties for filtering and wrapping modes.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Sampler {
    /// Magnification filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mag_filter: Option<MagFilter>,

    /// Minification filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_filter: Option<MinFilter>,

    /// S (U) wrapping mode.
    #[serde(default)]
    pub wrap_s: WrapMode,

    /// T (V) wrapping mode.
    #[serde(default)]
    pub wrap_t: WrapMode,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum MagFilter {
    Nearest,
    Linear,
    Other(u32),
}
impl Default for MagFilter {
    fn default() -> Self {
        Self::Linear
    }
}
impl Serialize for MagFilter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::Nearest => 9728,
            Self::Linear => 9729,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for MagFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(match value {
            9728 => Self::Nearest,
            9729 => Self::Linear,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum MinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
    Other(u32),
}
impl Default for MinFilter {
    fn default() -> Self {
        Self::Linear
    }
}
impl Serialize for MinFilter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::Nearest => 9728,
            Self::Linear => 9729,
            Self::NearestMipmapNearest => 9984,
            Self::LinearMipmapNearest => 9985,
            Self::NearestMipmapLinear => 9986,
            Self::LinearMipmapLinear => 9987,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for MinFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(match value {
            9728 => Self::Nearest,
            9729 => Self::Linear,
            9984 => Self::NearestMipmapNearest,
            9985 => Self::LinearMipmapNearest,
            9986 => Self::NearestMipmapLinear,
            9987 => Self::LinearMipmapLinear,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum WrapMode {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
    Other(u32),
}
impl Default for WrapMode {
    fn default() -> Self {
        Self::Repeat
    }
}
impl Serialize for WrapMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(match self {
            Self::ClampToEdge => 33071,
            Self::MirroredRepeat => 33648,
            Self::Repeat => 10497,
            Self::Other(v) => *v,
        })
    }
}
impl<'de> Deserialize<'de> for WrapMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(match value {
            33071 => Self::ClampToEdge,
            33648 => Self::MirroredRepeat,
            10497 => Self::Repeat,
            _ => Self::Other(value),
        })
    }
}

/// Image data used to create a texture. Image **MAY** be referenced by an URI (or IRI) or a buffer view index.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Image {
    /// The URI (or IRI) of the image.  Relative paths are relative to the current glTF asset.  Instead of referencing an external file, this field **MAY** contain a `data:`-URI. This field **MUST NOT** be defined when `bufferView` is defined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    /// The image's media type. This field **MUST** be defined when `bufferView` is defined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<MimeType>,

    /// The index of the bufferView that contains the image. This field **MUST NOT** be defined when `uri` is defined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_view: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum MimeType {
    Jpeg,
    Png,
    Other(String),
}
impl Serialize for MimeType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Other(v) => v,
        })
    }
}
impl<'de> Deserialize<'de> for MimeType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "image/jpeg" => Self::Jpeg,
            "image/png" => Self::Png,
            _ => Self::Other(value),
        })
    }
}

fn default_vec4_one() -> glam::Vec4 {
    glam::Vec4::ONE
}
fn default_vec3_zero() -> glam::Vec3 {
    glam::Vec3::ZERO
}
fn default_scalar_one() -> f32 {
    1.0
}

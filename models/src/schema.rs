use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

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

    /// An array of accessors.  An accessor is a typed view into a bufferView.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accessors: Vec<Accessor>,

    /// An array of meshes.  A mesh is a set of primitives to be rendered.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<Mesh>,

    /// An array of buffers.  A buffer points to binary geometry, animation, or skins.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<Buffer>,

    /// An array of bufferViews.  A bufferView is a view into a buffer generally representing a subset of the buffer.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buffer_views: Vec<BufferView>,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub max: Vec<GltfNumber>,

    /// Minimum value of each component in this accessor.  Array elements **MUST** be treated as having the same data type as accessor's `componentType`. Both `min` and `max` arrays have the same length.  The length is determined by the value of the `type` property; it can be 1, 2, 3, 4, 9, or 16.\n\n`normalized` property has no effect on array values: they always correspond to the actual values stored in the buffer. When the accessor is sparse, this property **MUST** contain minimum values of accessor data with sparse substitution applied.
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub min: Vec<GltfNumber>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GltfNumber {
    Integer(i64),
    Float(f64),
}

/// The datatype of the accessor's components.  UNSIGNED_INT type **MUST NOT** be used for any accessor that is not referenced by `mesh.primitive.indices`.
#[derive(Debug, PartialEq)]
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
            "Vec2" => AccessorType::Vec2,
            "Vec3" => AccessorType::Vec3,
            "Vec4" => AccessorType::Vec4,
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

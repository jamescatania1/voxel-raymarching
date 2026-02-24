use std::collections::{HashMap, VecDeque};

use crate::gltf::schema;
use anyhow::{Context, Result, bail, ensure};

/// Parsed scene data
#[derive(Debug)]
pub struct Scene {
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

#[derive(Debug)]
pub struct Material {
    pub base_albedo: glam::Vec4,
    pub base_metallic: f32,
    pub base_roughness: f32,
    pub normal_scale: f32,
    pub albedo_index: i32,
    pub normal_index: i32,
    pub metallic_roughness_index: i32,
    pub double_sided: bool,
}

#[derive(Debug)]
pub struct Texture {
    pub data: image::RgbaImage,
    pub encoding: TextureEncoding,
}

#[derive(Debug)]
pub enum TextureEncoding {
    Linear,
    Srgb,
}

#[derive(Debug)]
pub struct Node {
    pub mesh_id: usize,
    pub transform: glam::Mat4,
}

#[derive(Debug)]
pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

#[derive(Debug)]
pub struct Primitive {
    pub indices: PrimitiveBufferDescriptor,
    pub positions: PrimitiveBufferDescriptor,
    pub normals: PrimitiveBufferDescriptor,
    pub tangents: PrimitiveBufferDescriptor,
    pub uv: PrimitiveBufferDescriptor,
    pub material_id: u32,
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

#[derive(Debug)]
pub struct PrimitiveBufferDescriptor {
    pub component_type: schema::ComponentType,
    pub count: u32,
    pub start: usize,
    pub end: usize,
}

impl Scene {
    pub fn from_gltf(gltf: &schema::Gltf) -> Result<Self> {
        let mut textures = Vec::new();
        let mut img_index_map = HashMap::new();
        for (i, img) in gltf.meta.images.iter().enumerate() {
            let label = format!("img_{}_{}", i, img.name.as_deref().unwrap_or(""));

            match image::RgbaImage::from_gltf(&gltf, img, &label) {
                Ok(texture) => {
                    img_index_map.insert(i as u32, textures.len());
                    textures.push(Texture {
                        data: texture,
                        encoding: TextureEncoding::Linear,
                    });
                }
                Err(err) => {
                    eprintln!("error loading image {} - {}", &label, err);
                }
            }
        }

        let materials = (&gltf.meta.materials)
            .iter()
            .map(|m| Material::from_gltf(&gltf, &img_index_map, m))
            .collect::<Vec<Material>>();
        for material in &materials {
            if material.albedo_index >= 0 {
                if let Some(texture) = textures.get_mut(material.albedo_index as usize) {
                    texture.encoding = TextureEncoding::Srgb;
                }
            }
        }

        let mut meshes = Vec::new();
        let mut nodes = Vec::new();
        let mut min = glam::Vec3::MAX;
        let mut max = glam::Vec3::MIN;
        let scene = gltf
            .meta
            .scenes
            .get(gltf.meta.scene.context("no default scene")? as usize)
            .context("unable to find default scene")?;

        let mut visit_queue = VecDeque::new();
        for node in &scene.nodes {
            visit_queue.push_back((*node, schema::GLTF_Y_UP_TO_Z_UP));
        }

        let mut mesh_id_map = HashMap::new();
        // visit breadth first over the scene, flattening the matrix transform
        while let Some((node_id, parent_matrix)) = visit_queue.pop_front() {
            let node = gltf
                .meta
                .nodes
                .get(node_id as usize)
                .context(format!("unable to find node with id {}", node_id))?;
            let transform = parent_matrix * node.transform.matrix;

            // add children to visit
            for child_id in &node.children {
                visit_queue.push_back((*child_id, transform));
            }

            let Some(gltf_mesh_id) = node.mesh else {
                continue;
            };
            // current node has a mesh attached
            // our final node list is flat and only has ones with meshes
            let mesh_id = match mesh_id_map.get(&gltf_mesh_id) {
                Some(id) => anyhow::Ok(*id),
                None => {
                    // now, create and push a new mesh
                    let gltf_mesh = gltf
                        .meta
                        .meshes
                        .get(gltf_mesh_id as usize)
                        .context("invalid mesh id")?;
                    let primitives = gltf_mesh
                        .primitives
                        .iter()
                        .filter_map(|p| {
                            Primitive::from_gltf(&gltf, p)
                                .inspect_err(|err| eprintln!("error loading primitive: {:#?}", err))
                                .ok()
                        })
                        .collect::<Vec<Primitive>>();

                    mesh_id_map.insert(gltf_mesh_id, meshes.len());
                    meshes.push(Mesh { primitives });
                    Ok(meshes.len() - 1)
                }
            }?;
            let mesh = &meshes[mesh_id];
            for primitive in &mesh.primitives {
                let size = primitive.max - primitive.min;
                const CORNERS: [glam::Vec3; 8] = [
                    glam::vec3(0.0, 0.0, 0.0),
                    glam::vec3(0.0, 0.0, 1.0),
                    glam::vec3(0.0, 1.0, 0.0),
                    glam::vec3(0.0, 1.0, 1.0),
                    glam::vec3(1.0, 0.0, 0.0),
                    glam::vec3(1.0, 0.0, 1.0),
                    glam::vec3(1.0, 1.0, 0.0),
                    glam::vec3(1.0, 1.0, 1.0),
                ];
                for corner in CORNERS {
                    let pos = transform.transform_point3(primitive.min + size * corner);
                    min = pos.min(min);
                    max = pos.max(max);
                }
            }

            nodes.push(Node { mesh_id, transform });
        }

        Ok(Self {
            nodes,
            meshes,
            textures,
            materials,
            min,
            max,
        })
    }
}

trait TextureExt {
    fn from_gltf(gltf: &schema::Gltf, img: &schema::Image, label: &str)
    -> Result<image::RgbaImage>;
}
impl TextureExt for image::RgbaImage {
    fn from_gltf(gltf: &schema::Gltf, img: &schema::Image, label: &str) -> Result<Self> {
        let buf_view_index = img
            .buffer_view
            .context("no buffer view for image. TODO: load image uri's")?;
        let buf_view = gltf
            .meta
            .buffer_views
            .get(buf_view_index as usize)
            .context("unable to find buffer view")?;

        let src = &gltf.bin[(buf_view.byte_offset as usize)
            ..(buf_view.byte_offset as usize + buf_view.byte_length as usize)];

        let loaded = match img.mime_type {
            Some(schema::MimeType::Jpeg) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Jpeg)
            }
            Some(schema::MimeType::Png) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Png)
            }
            _ => image::load_from_memory(src),
        }?;

        Ok(loaded.to_rgba8())
    }
}

impl Material {
    fn from_gltf(
        gltf: &schema::Gltf,
        img_index_map: &HashMap<u32, usize>,
        m: &schema::Material,
    ) -> Self {
        let albedo_index = m
            .pbr_metallic_roughness
            .as_ref()
            .and_then(|pbr| (&pbr.base_color_texture).as_ref())
            .and_then(|info| gltf.meta.textures.get(info.index as usize))
            .and_then(|tex| tex.source)
            .and_then(|img_index| img_index_map.get(&img_index).map(|i| *i as i32))
            .unwrap_or(-1);
        let normal_index = (&m.normal_texture)
            .as_ref()
            .and_then(|info| gltf.meta.textures.get(info.index as usize))
            .and_then(|tex| tex.source)
            .and_then(|img_index| img_index_map.get(&img_index).map(|i| *i as i32))
            .unwrap_or(-1);
        let metallic_roughness_index = m
            .pbr_metallic_roughness
            .as_ref()
            .and_then(|pbr| (&pbr.metallic_roughness_texture).as_ref())
            .and_then(|info| gltf.meta.textures.get(info.index as usize))
            .and_then(|tex| tex.source)
            .and_then(|img_index| img_index_map.get(&img_index).map(|i| *i as i32))
            .unwrap_or(-1);

        let (base_albedo, base_roughness, base_metallic) = (&m.pbr_metallic_roughness)
            .as_ref()
            .map(|pbr| {
                (
                    pbr.base_color_factor,
                    pbr.roughness_factor,
                    pbr.metallic_factor,
                )
            })
            .unwrap_or((glam::Vec4::ZERO, 0.5, 0.0));

        let normal_scale = (&m.normal_texture).as_ref().map(|n| n.scale).unwrap_or(0.0);
        let double_sided = m.double_sided;

        Self {
            base_albedo,
            base_roughness,
            base_metallic,
            normal_scale,
            albedo_index,
            normal_index,
            metallic_roughness_index,
            double_sided,
        }
    }
}

impl Primitive {
    fn from_gltf(gltf: &schema::Gltf, p: &schema::Primitive) -> Result<Self> {
        ensure!(
            p.mode == schema::DrawMode::Triangles,
            "only DrawMode::Triangles is supported. TODO: add others"
        );
        let material_id = p.material.context("primitive has no material ID")?;

        // walks gltf and finds accessor, buffer view, buffer
        let get_buf_descriptor =
            |acc_index: u32,
             cmp_type: Option<schema::ComponentType>,
             acc_type: Option<schema::AccessorType>| {
                let accessor = gltf
                    .meta
                    .accessors
                    .get(acc_index as usize)
                    .context("accessor not found")?;
                let view = accessor
                    .buffer_view
                    .context("no buffer view specified")
                    .and_then(|index| {
                        gltf.meta
                            .buffer_views
                            .get(index as usize)
                            .context("no buffer view found")
                    })?;
                let buffer = gltf
                    .meta
                    .buffers
                    .get(view.buffer as usize)
                    .context("no buffer source found")?;
                ensure!(
                    buffer.uri.is_none(),
                    "buffer has external source. TODO: support this"
                );
                ensure!(
                    cmp_type
                        .as_ref()
                        .is_none_or(|ty| *ty == accessor.component_type),
                    format!(
                        "component type mismatch. expected {:?}, received {:?}",
                        cmp_type.unwrap(),
                        accessor.component_type
                    )
                );
                ensure!(
                    acc_type.as_ref().is_none_or(|ty| *ty == accessor.ty),
                    format!(
                        "accessor type mismatch. expected {:?}, received {:?}",
                        acc_type.unwrap(),
                        accessor.ty,
                    )
                );
                let mut component_length = match accessor.component_type {
                    schema::ComponentType::Byte | schema::ComponentType::UnsignedByte => 1,
                    schema::ComponentType::Short | schema::ComponentType::UnsignedShort => 2,
                    schema::ComponentType::UnsignedInt | schema::ComponentType::Float => 4,
                    schema::ComponentType::Other(_) => bail!("invalid component type"),
                };
                component_length *= match accessor.ty {
                    schema::AccessorType::Scalar => 1,
                    schema::AccessorType::Vec2 => 2,
                    schema::AccessorType::Vec3 => 3,
                    schema::AccessorType::Vec4 => 4,
                    schema::AccessorType::Mat2 => 4,
                    schema::AccessorType::Mat3 => 9,
                    schema::AccessorType::Mat4 => 16,
                    schema::AccessorType::Other(_) => {
                        bail!("invalid accessor element type")
                    }
                };
                let start = (view.byte_offset + accessor.byte_offset.unwrap_or(0)) as usize;
                let end = start + (component_length * accessor.count) as usize;
                if start >= buffer.byte_length as usize || end >= buffer.byte_length as usize {
                    bail!("accessor view extends beyond buffer's bounds");
                }
                Ok((
                    PrimitiveBufferDescriptor {
                        component_type: accessor.component_type.clone(),
                        count: accessor.count,
                        start,
                        end,
                    },
                    [accessor.min.clone(), accessor.max.clone()],
                ))
            };

        let (indices, _) = p
            .indices
            .context("no index buffer. TODO: add support for direct vertex lists")
            .and_then(|i| {
                get_buf_descriptor(i, None, Some(schema::AccessorType::Scalar))
                    .context("index buffer")
            })?;
        match indices.component_type {
            schema::ComponentType::UnsignedShort | schema::ComponentType::UnsignedInt => (),
            _ => {
                bail!("index format must be be either u16 or u32");
            }
        };

        let (positions, [min, max]) = p
            .attributes
            .get("POSITION")
            .context("no attribute")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(schema::ComponentType::Float),
                    Some(schema::AccessorType::Vec3),
                )
            })
            .context("vertex position buffer")?;

        let (normals, _) = p
            .attributes
            .get("NORMAL")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(schema::ComponentType::Float),
                    Some(schema::AccessorType::Vec3),
                )
            })
            .context("vertex normal buffer")?;

        let (tangents, _) = p
            .attributes
            .get("TANGENT")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(schema::ComponentType::Float),
                    Some(schema::AccessorType::Vec4),
                )
            })
            .context("vertex tangent buffer")?;

        let (uv, _) = p
            .attributes
            .get("TEXCOORD_0")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(schema::ComponentType::Float),
                    Some(schema::AccessorType::Vec2),
                )
            })
            .context("vertex coordinate buffer")?;

        let min = min.context("position attribute is missing minimum bounds")?;
        let max = max.context("position attribute is missing maximum bounds")?;
        let [min, max] = [min, max].map(|bd| {
            glam::Vec3::from_slice(
                &bd[..3]
                    .iter()
                    .map(|x| match x {
                        schema::GltfNumber::Float(x) => *x as f32,
                        _ => 0.0,
                    })
                    .collect::<Vec<f32>>(),
            )
        });

        Ok(Primitive {
            indices,
            positions,
            normals,
            tangents,
            uv,
            material_id,
            min,
            max,
        })
    }
}

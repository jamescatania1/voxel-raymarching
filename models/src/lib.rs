pub mod schema;

use std::{
    fmt::Debug,
    io::{BufRead, Seek},
};

use anyhow::{Result, ensure};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::schema::GltfJson;

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

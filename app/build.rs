use anyhow::{Context, Result};
use generate::{
    LIGHTMAP_FILE_EXT, LightmapHeader, MODEL_FILE_EXT, VoxelMetadata, load_lightmap_header,
    load_voxel_header,
};
use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    // println!("cargo::rerun-if-changed=assets/generated");

    let out_dir = env::var_os("OUT_DIR").context("OUT_DIR not set")?;
    let out_dir = Path::new(&out_dir);

    let mut models = Vec::new();
    for path in find_generated_assets(MODEL_FILE_EXT)? {
        let file = File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let mut decoder = flate2::read::ZlibDecoder::new(reader);
        let meta = load_voxel_header(&mut decoder)?;
        let path = Path::new("app")
            .join(path)
            .to_str()
            .map(|p| String::from(p))
            .context("error getting path")?;

        models.push((meta, path));
    }

    let mut lightmaps = Vec::new();
    for path in find_generated_assets(LIGHTMAP_FILE_EXT)? {
        let file = File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let mut decoder = flate2::read::ZlibDecoder::new(reader);
        let meta = load_lightmap_header(&mut decoder)?;
        let path = Path::new("app")
            .join(path)
            .to_str()
            .map(|p| String::from(p))
            .context("error getting path")?;

        lightmaps.push((meta, path));
    }

    let models = generate_model_defs(&models);
    let lightmaps = generate_lightmap_defs(&lightmaps);

    let res = quote::quote! {
        #models
        #lightmaps
    };
    let ast = syn::parse2(res).unwrap();
    let formatted = prettyplease::unparse(&ast);

    let dest_path = out_dir.join("assets.rs");
    fs::write(&dest_path, formatted)?;

    Ok(())
}

fn find_generated_assets(ext: &str) -> Result<Vec<PathBuf>> {
    let mut res = Vec::new();
    for entry in fs::read_dir("assets/generated")? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .is_none_or(|e| !e.eq_ignore_ascii_case(ext))
        {
            continue;
        }
        res.push(path);
    }
    Ok(res)
}

fn generate_model_defs(sources: &[(VoxelMetadata, String)]) -> proc_macro2::TokenStream {
    let idents = sources
        .iter()
        .map(|src| quote::format_ident!("{}", &src.0.name));

    let meta_defs = sources.iter().map(|src| {
        let name_ident = quote::format_ident!("{}", &src.0.name);
        let name = &src.0.name;
        let path = &src.1;
        quote::quote! {
            #name_ident: ModelEntry {
                name: #name,
                path: #path,
            }
        }
    });

    quote::quote! {
        mod models {
            use wgpu::util::DeviceExt;
            use generate::VoxelModel;
            use utils::tree::Tree;
            use std::io::Read;

            /// Atlas of all the available models in the runtime.
            ///
            /// `model.load(device, queue)` on any of the members loads the binary file into memory and populates GPU resources for rendering.
            pub const MODELS: ModelAtlas = ModelAtlas {
                #(#meta_defs,)*
            };

            #[derive(Debug)]
            pub struct ModelAtlas {
                #(pub #idents: ModelEntry,)*
            }

            /// Listing of a model available to the runtime.
            ///
            /// Running `self.load(device, queue)` will load the associated `.bin` file and popuate the GPU.
            #[derive(Debug)]
            pub struct ModelEntry {
                pub name: &'static str,
                pub path: &'static str,
            }

            impl ModelEntry {
                pub fn load(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<(VoxelModel, Tree)> {
                    let timer = std::time::Instant::now();

                    let file = std::fs::File::open(&self.path)?;
                    let reader = std::io::BufReader::new(file);
                    let mut decoder = flate2::bufread::ZlibDecoder::new(reader);
                    let mut buf = Vec::new();
                    decoder.read_to_end(&mut buf)?;

                    println!("file read: {:#?}", timer.elapsed());
                    let timer = std::time::Instant::now();

                    let model = VoxelModel::deserialize(device, queue, &buf)?;

                    println!("data deserialize: {:#?}", timer.elapsed());

                    Ok(model)
                }
            }
        }
    }
}

fn generate_lightmap_defs(sources: &[(LightmapHeader, String)]) -> proc_macro2::TokenStream {
    let idents = sources
        .iter()
        .map(|src| quote::format_ident!("{}", &src.0.name));

    let meta_defs = sources.iter().map(|src| {
        let name_ident = quote::format_ident!("{}", &src.0.name);
        let name = &src.0.name;
        let path = &src.1;
        quote::quote! {
            #name_ident: LightmapEntry {
                name: #name,
                path: #path,
            }
        }
    });

    quote::quote! {
        mod lightmap {
            use wgpu::util::DeviceExt;
            use generate::LightmapResult;
            use std::io::Read;

            /// Atlas of all the available models in the runtime.
            ///
            /// `model.load(device, queue)` on any of the members loads the binary file into memory and populates GPU resources for rendering.
            pub const LIGHTMAPS: LightmapAtlas = LightmapAtlas {
                #(#meta_defs,)*
            };

            #[derive(Debug)]
            pub struct LightmapAtlas {
                #(pub #idents: LightmapEntry,)*
            }

            /// Listing of a lightmap available to the runtime.
            ///
            /// Running `self.load(device, queue)` will load the associated `.bin` file and popuate the GPU.
            #[derive(Debug)]
            pub struct LightmapEntry {
                pub name: &'static str,
                pub path: &'static str,
            }

            impl LightmapEntry {
                pub fn load(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<LightmapResult> {
                    let timer = std::time::Instant::now();

                    let file = std::fs::File::open(&self.path)?;
                    let reader = std::io::BufReader::new(file);
                    let mut decoder = flate2::bufread::ZlibDecoder::new(reader);
                    let mut buf = Vec::new();
                    decoder.read_to_end(&mut buf)?;

                    println!("lightmap file read: {:#?}", timer.elapsed());
                    let timer = std::time::Instant::now();

                    let model = LightmapResult::deserialize(device, queue, &buf)?;

                    println!("lightmap deserialize: {:#?}", timer.elapsed());

                    Ok(model)
                }
            }
        }
    }
}

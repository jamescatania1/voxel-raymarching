use anyhow::{Context, Result};
use generate::{MODEL_FILE_EXT, VoxelMetadata, load_voxel_header};
use std::{
    env,
    fs::{self, File},
    path::Path,
};

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=assets/generated");

    let out_dir = env::var_os("OUT_DIR").context("OUT_DIR not set")?;
    let out_dir = Path::new(&out_dir);

    let mut models = Vec::new();
    for entry in fs::read_dir("assets/generated")? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .is_none_or(|e| !e.eq_ignore_ascii_case(MODEL_FILE_EXT))
        {
            continue;
        }
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

    generate_model_defs(&models, out_dir)?;

    Ok(())
}

fn generate_model_defs(sources: &[(VoxelMetadata, String)], out_dir: &Path) -> Result<()> {
    let model_idents = sources
        .iter()
        .map(|src| quote::format_ident!("{}", &src.0.name));

    let model_meta_defs = sources.iter().map(|src| {
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

    let res = quote::quote! {
        mod models {
            use wgpu::util::DeviceExt;
            use generate::VoxelModel;
            use std::io::Read;

            /// Atlas of all the available models in the runtime.
            ///
            /// `model.load(device, queue)` on any of the members loads the binary file into memory and populates GPU resources for rendering.
            pub const MODELS: ModelAtlas = ModelAtlas {
                #(#model_meta_defs,)*
            };

            #[derive(Debug)]
            pub struct ModelAtlas {
                #(pub #model_idents: ModelEntry,)*
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
                pub fn load(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<VoxelModel> {
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
    };

    let ast = syn::parse2(res).unwrap();
    let formatted = prettyplease::unparse(&ast);

    let dest_path = out_dir.join("assets.rs");
    fs::write(&dest_path, formatted)?;

    Ok(())
}

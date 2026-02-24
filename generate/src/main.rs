use anyhow::{Context, Result};
use flate2::write::ZlibEncoder;
use generate::{MODEL_FILE_EXT, voxelize};
use std::{
    fs,
    io::{self, Seek, Write},
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    let sources = walk_asset_sources(Path::new("app/assets"), "hdr")
        .context("error while retrieving skybox sources")?;
    create_lighting(&sources, Path::new("app/assets/generated"))
        .context("error generating lighting")?;

    // let sources = walk_glb_sources(Path::new("app/asses/models"), "glb")
    //     .context("error while retrieving model sources")?;
    // voxelize_models(&sources, Path::new("app/assets/generated"))
    //     .context("error voxelizing models")?;

    Ok(())
}

#[derive(Debug)]
struct AssetSource {
    name: String,
    path: PathBuf,
}

fn walk_asset_sources(path: &Path, ext: &str) -> Result<Vec<AssetSource>> {
    let mut sources = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut name_fallback_counter = 0;
    for entry in fs::read_dir(path)? {
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
        let name = path
            .file_stem()
            .and_then(|stem| {
                let mut name = stem.to_string_lossy().to_string();
                name = name.to_ascii_lowercase();
                name = name.replace(|c: char| !c.is_alphanumeric(), "_");
                while name.contains("__") {
                    name = name.replace("__", "_");
                }
                if name.chars().next().map_or(false, |c| c.is_numeric()) {
                    name.insert(0, '_');
                }
                if names.contains(&name) {
                    None
                } else {
                    Some(name)
                }
            })
            .unwrap_or_else(|| {
                loop {
                    let name = format!("model_{}", name_fallback_counter);
                    name_fallback_counter += 1;
                    if !names.contains(&name) {
                        names.insert(name.clone());
                        return name;
                    }
                }
            });

        sources.push(AssetSource { name, path });
    }

    Ok(sources)
}

fn create_lighting(sources: &[AssetSource], out_dir: &Path) -> Result<()> {
    todo!()
}

fn voxelize_models(sources: &[AssetSource], out_dir: &Path) -> Result<()> {
    let (device, queue) = {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

        let mut features = wgpu::Features::default();
        features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
        features |= wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 260;
        limits.max_buffer_size = 2 * 1024 * 1024 * 1024;
        limits.max_binding_array_elements_per_shader_stage = 260;
        limits.max_storage_textures_per_shader_stage = 6;
        limits.max_compute_invocations_per_workgroup = 512;

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))
    }?;

    for src in sources {
        let glb = fs::File::open(&src.path)?;
        let mut reader = io::BufReader::new(&glb);
        let data = voxelize(&mut reader, &device, &queue, Some(src.name.clone()))?;
        let data = data.serialize(&device, &queue)?;

        let path = out_dir.join(format!("{}.{}", &src.name, MODEL_FILE_EXT));
        let file = fs::File::create(&path)?;
        let mut enc = ZlibEncoder::new(file, flate2::Compression::best());
        enc.write_all(&data)?;
        enc.finish()?;
    }

    Ok(())
}

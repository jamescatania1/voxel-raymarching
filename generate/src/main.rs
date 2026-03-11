use anyhow::{Context, Result};
use clap::Parser;
use flate2::write::ZlibEncoder;
use generate::{
    LIGHTMAP_FILE_EXT, MAX_STORAGE_BUFFER_BINDING_SIZE, MODEL_FILE_EXT, generate_lighting, voxelize,
};
use std::{
    fs,
    io::{self, Seek, Write},
    path::{Path, PathBuf},
};

/// Generate engine assets
#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Generate specific models from `app/assets/models`
    #[arg(short, long)]
    models: bool,

    /// Generate specific models from `app/assets/lightmaps`
    #[arg(short, long)]
    lightmaps: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let generate_all = !args.models && !args.lightmaps;

    let (device, queue) = init_device().context("failed to initialize GPU context")?;

    // {
    //     let sources = walk_asset_sources(Path::new("app/assets/models"), "glb")
    //         .context("error while retrieving model sources")?;

    //     eprintln!("Generating models");
    //     for src in &sources {
    //         eprintln!("-    {} {:?}", src.name, src.path);
    //     }

    //     for src in sources {
    //         let glb = fs::File::open(&src.path)?;
    //         let mut reader = io::BufReader::new(&glb);
    //         let data = generate::models_t64::voxelize(
    //             &mut reader,
    //             &device,
    //             &queue,
    //             Some(src.name.clone()),
    //         )?;
    //         // let data = data.serialize(&device, &queue)?;

    //         // let path = out_dir.join(format!("{}.{}", &src.name, MODEL_FILE_EXT));
    //         // let file = fs::File::create(&path)?;
    //         // let mut enc = ZlibEncoder::new(file, flate2::Compression::best());
    //         // enc.write_all(&data)?;
    //         // let mut res = enc.finish()?;
    //         // let length = res
    //         //     .stream_position()
    //         //     .map(|len| len as f64 / (1024.0 * 1024.0))?;

    //         // eprintln!("Completed {} ({:.2} MB)", src.name, length);
    //     }
    //     // voxelize_models(&device, &queue, &sources, Path::new("app/assets/generated"))
    //     //     .context("error voxelizing models")?;
    // }
    // return Ok(());

    if generate_all || args.lightmaps {
        let sources = walk_asset_sources(Path::new("app/assets/lightmaps"), "hdr")
            .context("error while retrieving lightmap sources")?;

        eprintln!("Generating lightmaps");
        for src in &sources {
            eprintln!("-    {} {:?}", src.name, src.path);
        }

        create_lighting(&device, &queue, &sources, Path::new("app/assets/generated"))
            .context("error generating lighting")?;
    }

    if generate_all || args.models {
        let sources = walk_asset_sources(Path::new("app/assets/models"), "glb")
            .context("error while retrieving model sources")?;

        eprintln!("Generating models");
        for src in &sources {
            eprintln!("-    {} {:?}", src.name, src.path);
        }

        voxelize_models(&device, &queue, &sources, Path::new("app/assets/generated"))
            .context("error voxelizing models")?;
    }

    Ok(())
}

fn init_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

    let mut features = wgpu::Features::default();
    features |= wgpu::Features::FLOAT32_FILTERABLE;
    features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
    features |= wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

    let mut limits = wgpu::Limits::default();
    limits.max_sampled_textures_per_shader_stage = 260;
    limits.max_buffer_size = 2 * 1024 * 1024 * 1024;
    limits.max_binding_array_elements_per_shader_stage = 260;
    limits.max_storage_textures_per_shader_stage = 6;
    limits.max_compute_invocations_per_workgroup = 512;
    limits.max_storage_buffer_binding_size = MAX_STORAGE_BUFFER_BINDING_SIZE;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: limits,
        ..Default::default()
    }))?;

    Ok((device, queue))
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

fn create_lighting(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sources: &[AssetSource],
    out_dir: &Path,
) -> Result<()> {
    for src in sources {
        let hdr = fs::File::open(&src.path)?;
        let mut reader = io::BufReader::new(&hdr);
        let lightmap = generate_lighting(&mut reader, device, queue)?;

        let path = out_dir.join(format!("{}.{}", &src.name, LIGHTMAP_FILE_EXT));

        let hdr = fs::File::open(&src.path)?;
        let mut reader = io::BufReader::new(&hdr);
        let data = lightmap.serialize(&src.name, &mut reader, device, queue)?;

        let file = fs::File::create(&path)?;
        let mut enc = ZlibEncoder::new(file, flate2::Compression::best());
        enc.write_all(&data)?;
        let mut res = enc.finish()?;
        let length = res
            .stream_position()
            .map(|len| len as f64 / (1024.0 * 1024.0))?;

        eprintln!("Completed {} ({:.2} MB)", src.name, length);
    }
    Ok(())
}

fn voxelize_models(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sources: &[AssetSource],
    out_dir: &Path,
) -> Result<()> {
    for src in sources {
        let glb = fs::File::open(&src.path)?;
        let mut reader = io::BufReader::new(&glb);
        let data = voxelize(&mut reader, &device, &queue, Some(src.name.clone()))?;
        let data = data.serialize(&device, &queue)?;

        let path = out_dir.join(format!("{}.{}", &src.name, MODEL_FILE_EXT));
        let file = fs::File::create(&path)?;
        let mut enc = ZlibEncoder::new(file, flate2::Compression::best());
        enc.write_all(&data)?;
        let mut res = enc.finish()?;
        let length = res
            .stream_position()
            .map(|len| len as f64 / (1024.0 * 1024.0))?;

        eprintln!("Completed {} ({:.2} MB)", src.name, length);
    }

    Ok(())
}

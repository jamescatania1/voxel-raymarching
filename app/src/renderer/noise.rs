use anyhow::Result;

pub const HALTON_16: [glam::Vec2; 16] = [
    glam::vec2(0.500000, 0.333333),
    glam::vec2(0.250000, 0.666667),
    glam::vec2(0.750000, 0.111111),
    glam::vec2(0.125000, 0.444444),
    glam::vec2(0.625000, 0.777778),
    glam::vec2(0.375000, 0.222222),
    glam::vec2(0.875000, 0.555556),
    glam::vec2(0.062500, 0.888889),
    glam::vec2(0.562500, 0.037037),
    glam::vec2(0.312500, 0.370370),
    glam::vec2(0.812500, 0.703704),
    glam::vec2(0.187500, 0.148148),
    glam::vec2(0.687500, 0.481481),
    glam::vec2(0.437500, 0.814815),
    glam::vec2(0.937500, 0.259259),
    glam::vec2(0.031250, 0.592593),
];

pub fn noise_uniform_gauss(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    let src = std::include_bytes!("../../assets/noise/vector3_uniform_gauss1_0.png");
    let img = image::load_from_memory_with_format(src, image::ImageFormat::Png)?.to_rgba8();
    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("noise_uniform_gauss"),
        size: wgpu::Extent3d {
            width: img.width(),
            height: img.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &res,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.width()),
            rows_per_image: Some(img.height()),
        },
        wgpu::Extent3d {
            width: img.width(),
            height: img.height(),
            depth_or_array_layers: 1,
        },
    );
    Ok(res)
}

pub fn noise_sphere_gauss(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    macro_rules! include_images {
        ($($n:expr),*) => {
            [
                $(std::include_bytes!(concat!("../../assets/noise/sphere_uniform_gauss1_0_exp0101_separate05_", $n, ".png"))),*
            ]
        }
    }
    static IMAGE_SRC: [&[u8]; 32] = include_images!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31
    );
    let images = IMAGE_SRC
        .iter()
        .map(|src| {
            image::load_from_memory_with_format(src, image::ImageFormat::Png)
                .map(|img| img.to_rgba8())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blue_noise"),
        size: wgpu::Extent3d {
            width: images[0].width(),
            height: images[0].height(),
            depth_or_array_layers: 32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, img) in images.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &res,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: i as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width()),
                rows_per_image: Some(img.height()),
            },
            wgpu::Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(res)
}

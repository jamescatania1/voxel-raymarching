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

pub const HALTON_64: [glam::Vec2; 64] = [
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
    glam::vec2(0.531250, 0.925926),
    glam::vec2(0.281250, 0.074074),
    glam::vec2(0.781250, 0.407407),
    glam::vec2(0.156250, 0.740741),
    glam::vec2(0.656250, 0.185185),
    glam::vec2(0.406250, 0.518519),
    glam::vec2(0.906250, 0.851852),
    glam::vec2(0.093750, 0.296296),
    glam::vec2(0.593750, 0.629630),
    glam::vec2(0.343750, 0.962963),
    glam::vec2(0.843750, 0.012346),
    glam::vec2(0.218750, 0.345679),
    glam::vec2(0.718750, 0.679012),
    glam::vec2(0.468750, 0.123457),
    glam::vec2(0.968750, 0.456790),
    glam::vec2(0.015625, 0.790123),
    glam::vec2(0.515625, 0.234568),
    glam::vec2(0.265625, 0.567901),
    glam::vec2(0.765625, 0.901235),
    glam::vec2(0.140625, 0.049383),
    glam::vec2(0.640625, 0.382716),
    glam::vec2(0.390625, 0.716049),
    glam::vec2(0.890625, 0.160494),
    glam::vec2(0.078125, 0.493827),
    glam::vec2(0.578125, 0.827160),
    glam::vec2(0.328125, 0.271605),
    glam::vec2(0.828125, 0.604938),
    glam::vec2(0.203125, 0.938272),
    glam::vec2(0.703125, 0.086420),
    glam::vec2(0.453125, 0.419753),
    glam::vec2(0.953125, 0.753086),
    glam::vec2(0.046875, 0.197531),
    glam::vec2(0.546875, 0.530864),
    glam::vec2(0.296875, 0.864198),
    glam::vec2(0.796875, 0.308642),
    glam::vec2(0.171875, 0.641975),
    glam::vec2(0.671875, 0.975309),
    glam::vec2(0.421875, 0.024691),
    glam::vec2(0.921875, 0.358025),
    glam::vec2(0.109375, 0.691358),
    glam::vec2(0.609375, 0.135802),
    glam::vec2(0.359375, 0.469136),
    glam::vec2(0.859375, 0.802469),
    glam::vec2(0.234375, 0.246914),
    glam::vec2(0.734375, 0.580247),
    glam::vec2(0.484375, 0.913580),
    glam::vec2(0.984375, 0.061728),
    glam::vec2(0.007812, 0.395062),
];

#[allow(unused)]
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

#[allow(unused)]
pub fn noise_stbn_coshemi(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    macro_rules! include_images {
        ($($n:expr),*) => {
            [
                $(std::include_bytes!(concat!("../../assets/noise/stbn_unitvec3_cosine_2Dx1D_128x128x64_", $n, ".png"))),*
            ]
        }
    }
    static IMAGE_SRC: [&[u8]; 64] = include_images!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
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
            depth_or_array_layers: 64,
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

#[allow(unused)]
pub fn noise_stbn_scalar(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    macro_rules! include_images {
        ($($n:expr),*) => {
            [
                $(std::include_bytes!(concat!("../../assets/noise/stbn_scalar_2Dx1Dx1D_128x128x64x1_", $n, ".png"))),*
            ]
        }
    }
    static IMAGE_SRC: [&[u8]; 64] = include_images!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    );
    let images = IMAGE_SRC
        .iter()
        .map(|src| {
            image::load_from_memory_with_format(src, image::ImageFormat::Png)
                .map(|img| img.to_luma8())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("noise_stbn_scalar"),
        size: wgpu::Extent3d {
            width: images[0].width(),
            height: images[0].height(),
            depth_or_array_layers: 64,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
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
                bytes_per_row: Some(img.width()),
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

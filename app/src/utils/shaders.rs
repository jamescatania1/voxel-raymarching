#[macro_export]
macro_rules! define_shaders {
    {$($name:ident $src:literal,)* } => {
        struct Shaders {
            $($name: wgpu::ShaderModule),*
        }

        trait ShadersExt {
            fn create_shaders(&self) -> Shaders;
        }

        impl ShadersExt for wgpu::Device {
            fn create_shaders(&self) -> Shaders {
                Shaders {
                    $($name: self.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some($src),
                        source: wgpu::ShaderSource::Wgsl(
                            std::include_str!($src).into(),
                        ),
                    })),*
                }
            }
        }
    };
}

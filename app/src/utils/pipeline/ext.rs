pub trait PipelineUtils {
    fn compute_pipeline<'a>(
        &'a self,
        label: &'a str,
        module: &'a wgpu::ShaderModule,
    ) -> ComputePipelineBase<'a>;
}
impl PipelineUtils for wgpu::Device {
    fn compute_pipeline<'a>(
        &'a self,
        label: &'a str,
        module: &'a wgpu::ShaderModule,
    ) -> ComputePipelineBase<'a> {
        ComputePipelineBase {
            device: self,
            label,
            module,
            cache: None,
            immediate_size: 0,
            compilation_options: Default::default(),
        }
    }
}

pub struct ComputePipelineBase<'a> {
    pub device: &'a wgpu::Device,
    pub label: &'a str,
    pub module: &'a wgpu::ShaderModule,
    pub cache: Option<&'a wgpu::PipelineCache>,
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
    pub immediate_size: u32,
}

impl<'a> ComputePipelineBase<'a> {
    pub fn immediate_size(mut self, value: u32) -> Self {
        self.immediate_size = value;
        self
    }

    pub fn compilation_options(mut self, options: wgpu::PipelineCompilationOptions<'a>) -> Self {
        self.compilation_options = options;
        self
    }

    pub fn cache(mut self, cache: &'a wgpu::PipelineCache) -> Self {
        self.cache = Some(cache);
        self
    }
}

impl ComputePipelineBase<'_> {
    pub fn auto(self) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(self.label),
                layout: None,
                module: self.module,
                entry_point: None,
                compilation_options: self.compilation_options,
                cache: self.cache,
            })
    }

    pub fn layout<'a>(self, entries: &'a [&'a wgpu::BindGroupLayout]) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(self.label),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some(self.label),
                        bind_group_layouts: entries,
                        immediate_size: self.immediate_size,
                    },
                )),
                module: self.module,
                entry_point: None,
                compilation_options: self.compilation_options,
                cache: self.cache,
            })
    }
}

#![allow(unused)]

use std::num::NonZero;

use wgpu::{BindGroupEntry, BindingResource, ShaderStages};

use crate::textures::{
    DeviceSwapExt, SwapchainBindGroup, SwapchainBindGroupDescriptor, SwapchainBindGroupEntry,
    SwapchainBindingResource,
};

pub trait DeviceUtils {
    fn layout(
        &self,
        label: &str,
        visibility: ShaderStages,
        entries: impl IntoEntries,
    ) -> wgpu::BindGroupLayout;
    fn bind_group<const N: usize>(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: [BindingResource; N],
    ) -> wgpu::BindGroup;
    fn bind_group_swap<const N: usize>(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: [SwapchainBindingResource; N],
    ) -> SwapchainBindGroup;
}
impl DeviceUtils for wgpu::Device {
    fn layout(
        &self,
        label: &str,
        visibility: ShaderStages,
        entries: impl IntoEntries,
    ) -> wgpu::BindGroupLayout {
        self.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries.into_entries(visibility),
        })
    }

    fn bind_group<const N: usize>(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: [BindingResource; N],
    ) -> wgpu::BindGroup {
        let mut entries = entries.map(|e| BindGroupEntry {
            binding: 0,
            resource: e,
        });
        for i in 0..N {
            entries[i].binding = i as u32;
        }
        self.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    fn bind_group_swap<const N: usize>(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: [SwapchainBindingResource; N],
    ) -> SwapchainBindGroup {
        let mut entries = entries.map(|e| SwapchainBindGroupEntry {
            binding: 0,
            resource: e,
        });
        for i in 0..N {
            entries[i].binding = i as u32;
        }
        self.create_bind_group_swap(&SwapchainBindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
}

pub trait BindingResourceExt {
    fn as_binding(&self) -> wgpu::BindingResource<'_>;
}
impl BindingResourceExt for wgpu::Sampler {
    fn as_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Sampler(self)
    }
}
impl BindingResourceExt for wgpu::Buffer {
    fn as_binding(&self) -> wgpu::BindingResource<'_> {
        self.as_entire_binding()
    }
}

pub trait LayoutBindingType {
    fn into_base(&self) -> wgpu::BindingType;
}

pub struct LayoutEntry<T: LayoutBindingType> {
    pub visibility: wgpu::ShaderStages,
    pub ty: T,
    pub count: Option<NonZero<u32>>,
}
impl<T: LayoutBindingType> LayoutEntry<T> {
    /// Which shader stages can see this binding.
    ///
    /// Overwrites the default set in `layout`.
    pub fn visibility(mut self, visibility: wgpu::ShaderStages) -> Self {
        self.visibility = visibility;
        self
    }
    /// If the binding is an array of multiple resources. Corresponds to `binding_array<T>` in the shader.
    ///
    /// When this is specified the following validation applies:
    /// - Size must be of value 1 or greater.
    /// - When `ty == BindingType::Texture`, [`Features::TEXTURE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Sampler`, [`Features::TEXTURE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Buffer`, [`Features::BUFFER_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Buffer` and `ty.ty == BufferBindingType::Storage`, [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::StorageTexture`, [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] must be supported.
    /// - When any binding in the group is an array, no `BindingType::Buffer` in the group may have `has_dynamic_offset == true`
    /// - When any binding in the group is an array, no `BindingType::Buffer` in the group may have `ty.ty == BufferBindingType::Uniform`.
    ///
    pub fn count(mut self, count: NonZero<u32>) -> Self {
        self.count = Some(count);
        self
    }
}

pub trait IntoEntries {
    fn into_entries(self, base_visibility: ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry>;
}
impl<T: LayoutBindingType> IntoEntries for LayoutEntry<T> {
    fn into_entries(self, base_visibility: ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: base_visibility | self.visibility,
            ty: self.ty.into_base(),
            count: self.count,
        }]
    }
}
macro_rules! impl_into_entries {
    ($($name:ident),* ) => {
        #[allow(non_snake_case)]
        impl<$($name: LayoutBindingType),*> IntoEntries for ($(LayoutEntry<$name>,)*) {
            fn into_entries(self, base_visibility: ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
                let ($($name,)*) = self;
                let mut _i = 0;
                vec![$(
                    wgpu::BindGroupLayoutEntry {
                        binding: {
                            let cur = _i;
                            _i += 1;
                            cur
                        },
                        visibility: base_visibility | $name.visibility,
                        ty: $name.ty.into_base(),
                        count: $name.count,
                    }
                ),*]
            }
        }
    }
}
impl_into_entries!(T1);
impl_into_entries!(T1, T2);
impl_into_entries!(T1, T2, T3);
impl_into_entries!(T1, T2, T3, T4);
impl_into_entries!(T1, T2, T3, T4, T5);
impl_into_entries!(T1, T2, T3, T4, T5, T6);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
impl_into_entries!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15
);
impl_into_entries!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
);

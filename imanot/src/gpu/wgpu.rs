//! WebGPU, Vulkan, Metal, DX12 and WebGL2 through `egui-wgpu`.

use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
};

use egui_wgpu::wgpu;

use super::{
    GpuBackend,
    mask::{HEADER_SIZE, INSTANCE_SIZE, MaskFrame, UNIFORM_SIZE, VERTICES_PER_INSTANCE},
    sealed,
};

/// eframe's `wgpu` renderer. Needs [`Wgpu::register`] once.
pub struct Wgpu;

impl sealed::Sealed for Wgpu {}

impl Wgpu {
    /// Prepares the renderer of `render_state` to draw imanot's content.
    /// Call it once, when eframe hands out the render state (the
    /// `CreationContext` of the app).
    ///
    /// A paint callback must match egui's render pass, which egui-wgpu
    /// doesn't tell its callbacks: pass the options eframe was configured
    /// with (multisampling and depth buffer).
    pub fn register(render_state: &egui_wgpu::RenderState, options: &egui_wgpu::RendererOptions) {
        let pipeline = MaskPipeline::new(&render_state.device, render_state.target_format, options);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(pipeline);
    }
}

impl GpuBackend for Wgpu {
    type MaskBuffer = WgpuMasks;

    fn mask_callback(buffer: Arc<WgpuMasks>, frame: MaskFrame) -> egui::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(frame.rect, MaskCallback { buffer, frame })
    }
}

/// Buffers of one [`State`](crate::State)'s masks, created by the first paint.
#[derive(Default)]
pub struct WgpuMasks(Mutex<Option<MaskBuffers>>);

struct MaskBuffers {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Possibly larger than the instances, see [`create_instance_buffer`]
    instances: wgpu::Buffer,
    /// [`MaskFrame::revision`] of the uploaded instances and layers
    uploaded: Option<u64>,
    /// Start of `uniforms`, which only changes with pan, zoom, hover,
    /// settings and the number of layers
    uploaded_header: Option<[u8; HEADER_SIZE]>,
}

impl MaskBuffers {
    fn new(device: &wgpu::Device, pipeline: &MaskPipeline, frame: &MaskFrame) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imanot_mask_uniforms"),
            size: UNIFORM_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("imanot_mask_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        Self {
            uniforms,
            bind_group,
            instances: create_instance_buffer(device, frame.instances().len()),
            uploaded: None,
            uploaded_header: None,
        }
    }

    fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &MaskFrame,
        header: [u8; HEADER_SIZE],
    ) {
        if self.uploaded_header != Some(header) {
            queue.write_buffer(&self.uniforms, 0, &header);
            self.uploaded_header = Some(header);
        }
        if self.uploaded == Some(frame.revision) {
            return;
        }
        let instances = frame.instances();
        let (size, len) = (self.instances.size(), instances.len() as u64);
        // Too small, or wasting GPU memory after the masks shrank
        if size < len || size > len + len / 2 {
            self.instances = create_instance_buffer(device, instances.len());
        }
        queue.write_buffer(&self.instances, 0, instances);
        queue.write_buffer(&self.uniforms, HEADER_SIZE as u64, frame.layers());
        self.uploaded = Some(frame.revision);
    }
}

fn create_instance_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imanot_mask_instances"),
        // A little room to grow as masks are edited, not to recreate the
        // buffer for every stroke
        size: (len + len / 8).max(INSTANCE_SIZE).next_multiple_of(4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct MaskCallback {
    buffer: Arc<WgpuMasks>,
    frame: MaskFrame,
}

impl egui_wgpu::CallbackTrait for MaskCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(pipeline) = resources.get::<MaskPipeline>() else {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::error!("Masks can't be drawn: call imanot::gpu::Wgpu::register first");
            }
            return Vec::new();
        };
        // The viewport `paint` gets, see `egui_wgpu::Renderer::render`
        let viewport = egui::PaintCallbackInfo {
            viewport: self.frame.rect,
            clip_rect: self.frame.rect,
            pixels_per_point: screen.pixels_per_point,
            screen_size_px: screen.size_in_pixels,
        }
        .viewport_in_pixels();
        let Some(header) = self.frame.header(&viewport, screen.pixels_per_point) else {
            return Vec::new();
        };
        let mut buffers = self.buffer.0.lock().unwrap_or_else(PoisonError::into_inner);
        let buffers =
            buffers.get_or_insert_with(|| MaskBuffers::new(device, pipeline, &self.frame));
        buffers.update(device, queue, &self.frame, header);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let buffers = self.buffer.0.lock().unwrap_or_else(PoisonError::into_inner);
        let (Some(pipeline), Some(buffers)) = (resources.get::<MaskPipeline>(), buffers.as_ref())
        else {
            return;
        };
        // Uploaded by `prepare`, unless it had nothing to draw
        if buffers.uploaded != Some(self.frame.revision) {
            return;
        }
        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        render_pass.set_vertex_buffer(0, buffers.instances.slice(..));
        render_pass.draw(0..VERTICES_PER_INSTANCE, 0..self.frame.instance_count());
    }
}

/// Shared by all states drawn by one renderer, see [`Wgpu::register`].
struct MaskPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MaskPipeline {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        options: &egui_wgpu::RendererOptions,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imanot_mask"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/mask.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imanot_mask_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE as u64),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imanot_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // Same render pass as egui's own pipeline, so the same attachments
        let depth_stencil = options
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("imanot_mask_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: INSTANCE_SIZE as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    // The fields of `Instance`: `x`, then `y`
                    attributes: &wgpu::vertex_attr_array![0 => Uint16x4, 1 => Uint16x2],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: options.msaa_samples.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                // Like egui's own shapes
                entry_point: Some(if target_format.is_srgb() {
                    "fs_main_linear_framebuffer"
                } else {
                    "fs_main_gamma_framebuffer"
                }),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Premultiplied alpha, like egui
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

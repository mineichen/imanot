//! OpenGL (desktop) and WebGL2 through `egui_glow`.

use std::{
    cell::RefCell,
    sync::{Arc, Mutex, MutexGuard, PoisonError, Weak},
};

use egui_glow::{
    ShaderVersion,
    glow::{self, HasContext as _},
};

use super::{
    GpuBackend,
    mask::{HEADER_SIZE, INSTANCE_SIZE, MaskFrame, UNIFORM_SIZE, VERTICES_PER_INSTANCE},
    sealed,
};

const MASK_VERTEX: &str = include_str!("shader/mask.vert");
const MASK_FRAGMENT: &str = include_str!("shader/mask.frag");
/// Binding point of the `MaskUniforms` block
const MASK_UNIFORMS_BINDING: u32 = 0;

/// eframe's `glow` renderer: OpenGL 3.3 on desktop, WebGL2 on the web.
/// Needs no setup.
pub struct Glow;

impl sealed::Sealed for Glow {}

impl GpuBackend for Glow {
    type MaskBuffer = GlowMasks;

    fn mask_callback(buffer: Arc<GlowMasks>, frame: MaskFrame) -> egui::PaintCallback {
        egui::PaintCallback {
            rect: frame.rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                buffer.paint(painter.gl(), &info, &frame);
            })),
        }
    }
}

/// GL objects of one [`State`](crate::State)'s masks, created by its first
/// paint: its buffers and the vertex array reading them. The program is shared
/// by all states on a context, see [`ContextId::with_program`].
///
/// A drop has no GL context to delete them with, so it hands them over to
/// [`ORPHANS`]: the next mask paint on their context deletes them.
#[derive(Default)]
pub struct GlowMasks(Mutex<Objects>);

#[derive(Default)]
enum Objects {
    #[default]
    NotCreated,
    Created(MaskObjects),
    /// Logged once, the masks are not drawn
    Failed,
}

/// GL objects of dropped [`GlowMasks`], waiting for a paint on their context.
static ORPHANS: Mutex<Vec<MaskObjects>> = Mutex::new(Vec::new());

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Drop for GlowMasks {
    fn drop(&mut self) {
        let objects = self.0.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Objects::Created(objects) = std::mem::take(objects) {
            lock(&ORPHANS).push(objects);
        }
    }
}

impl GlowMasks {
    fn paint(&self, gl: &Arc<glow::Context>, info: &egui::PaintCallbackInfo, frame: &MaskFrame) {
        let (context, program) = ContextId::with_program(gl);
        delete_orphans(gl, context);
        let (Some(program), Some(header)) = (
            program,
            frame.header(&info.viewport_in_pixels(), info.pixels_per_point),
        ) else {
            return;
        };
        let mut objects = lock(&self.0);
        if let Objects::Created(created) = &*objects
            && created.context != context
        {
            // Made on a context eframe replaced since: their names mean
            // nothing (or something else) here
            if let Objects::Created(stale) = std::mem::take(&mut *objects) {
                lock(&ORPHANS).push(stale);
            }
        }
        if matches!(*objects, Objects::NotCreated) {
            *objects = match MaskObjects::new(gl, context) {
                Ok(created) => Objects::Created(created),
                Err(e) => {
                    log::error!("Masks can't be drawn: {e}");
                    Objects::Failed
                }
            };
        }
        if let Objects::Created(objects) = &mut *objects {
            objects.draw(gl, program, frame, &header);
        }
    }
}

/// Deletes the orphans made on `context` and forgets those of dropped
/// contexts, which took their objects along.
fn delete_orphans(gl: &glow::Context, context: ContextId) {
    let mut orphans = lock(&ORPHANS);
    if orphans.is_empty() {
        return;
    }
    orphans.retain(|orphan| {
        if orphan.context == context {
            // SAFETY: Made on this context, which is current while it paints
            unsafe { orphan.delete(gl) };
            false
        } else {
            !orphan.context.is_dropped()
        }
    });
}

/// A GL context, by the address of its `Arc`. That address can't be reused by
/// a later context, as [`CONTEXTS`] keeps the allocation alive.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ContextId(usize);

thread_local! {
    /// The contexts masks were painted on (eframe has one per process). Only
    /// the allocation is kept, the context itself is dropped as usual.
    static CONTEXTS: RefCell<Vec<KnownContext>> = const { RefCell::new(Vec::new()) };
}

struct KnownContext {
    context: Weak<glow::Context>,
    /// Shared by all masks on the context and deleted with it. `None` if it
    /// can't be linked (logged once): the masks are not drawn then.
    program: Option<glow::Program>,
}

impl ContextId {
    /// The id of `gl` and its program, linked by the first call.
    fn with_program(gl: &Arc<glow::Context>) -> (Self, Option<glow::Program>) {
        let id = Self(Arc::as_ptr(gl) as usize);
        let program = CONTEXTS.with_borrow_mut(|contexts| {
            if let Some(known) = contexts.iter().find(|known| id.is(&known.context)) {
                return known.program;
            }
            let program = link_program(gl)
                .inspect_err(|e| log::error!("Masks can't be drawn: {e}"))
                .ok();
            contexts.push(KnownContext {
                context: Arc::downgrade(gl),
                program,
            });
            program
        });
        (id, program)
    }

    fn is(self, context: &Weak<glow::Context>) -> bool {
        context.as_ptr() as usize == self.0
    }

    /// Unknown contexts (used by another thread) count as alive.
    fn is_dropped(self) -> bool {
        CONTEXTS.with_borrow(|contexts| {
            contexts
                .iter()
                .any(|known| self.is(&known.context) && known.context.strong_count() == 0)
        })
    }
}

struct MaskObjects {
    /// The context the objects were made on
    context: ContextId,
    vertex_array: glow::VertexArray,
    instances: glow::Buffer,
    uniforms: glow::Buffer,
    /// [`MaskFrame::revision`] of the uploaded instances and layers
    uploaded: Option<u64>,
    /// Start of `uniforms`, which only changes with pan, zoom, hover,
    /// settings and the number of layers
    uploaded_header: Option<[u8; HEADER_SIZE]>,
}

impl MaskObjects {
    fn new(gl: &glow::Context, context: ContextId) -> Result<Self, String> {
        // SAFETY: GL calls on the painter's context, which is current while it paints
        unsafe {
            let vertex_array = gl.create_vertex_array()?;
            let instances = gl.create_buffer()?;
            let uniforms = gl.create_buffer()?;

            gl.bind_vertex_array(Some(vertex_array));
            // The fields of `Instance`, one per instance: `x`, then `y`.
            // Locations: see `link_program`
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(instances));
            let stride = INSTANCE_SIZE as i32;
            gl.vertex_attrib_pointer_i32(0, 4, glow::UNSIGNED_SHORT, stride, 0);
            gl.vertex_attrib_pointer_i32(1, 2, glow::UNSIGNED_SHORT, stride, 8);
            for location in [0, 1] {
                gl.enable_vertex_attrib_array(location);
                gl.vertex_attrib_divisor(location, 1);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            // Allocated once, `draw` only overwrites it
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(uniforms));
            gl.buffer_data_size(
                glow::UNIFORM_BUFFER,
                UNIFORM_SIZE as i32,
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);

            Ok(Self {
                context,
                vertex_array,
                instances,
                uniforms,
                uploaded: None,
                uploaded_header: None,
            })
        }
    }

    /// # Safety
    /// `gl` is the context the objects were made on, and current.
    unsafe fn delete(&self, gl: &glow::Context) {
        // SAFETY: See the function docs
        unsafe {
            gl.delete_vertex_array(self.vertex_array);
            gl.delete_buffer(self.instances);
            gl.delete_buffer(self.uniforms);
        }
    }

    fn draw(
        &mut self,
        gl: &glow::Context,
        program: glow::Program,
        frame: &MaskFrame,
        header: &[u8; HEADER_SIZE],
    ) {
        // SAFETY: GL calls on the context the objects were created with, see `paint`
        unsafe {
            if self.uploaded != Some(frame.revision) {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.instances));
                // Sized to the instances, no GPU memory to spare
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, frame.instances(), glow::STATIC_DRAW);
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.uniforms));
                gl.buffer_sub_data_u8_slice(
                    glow::UNIFORM_BUFFER,
                    HEADER_SIZE as i32,
                    frame.layers(),
                );
                self.uploaded = Some(frame.revision);
            }
            if self.uploaded_header.as_ref() != Some(header) {
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.uniforms));
                gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, header);
                self.uploaded_header = Some(*header);
            }
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                MASK_UNIFORMS_BINDING,
                Some(self.uniforms),
            );

            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(self.vertex_array));
            // Premultiplied alpha, like egui
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func_separate(
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE_MINUS_DST_ALPHA,
                glow::ONE,
            );
            gl.draw_arrays_instanced(
                glow::TRIANGLE_STRIP,
                0,
                VERTICES_PER_INSTANCE as i32,
                frame.instance_count() as i32,
            );
            // egui restores its own state, but doesn't know about these
            gl.bind_vertex_array(None);
            gl.bind_buffer_base(glow::UNIFORM_BUFFER, MASK_UNIFORMS_BINDING, None);
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        }
    }
}

fn link_program(gl: &glow::Context) -> Result<glow::Program, String> {
    let version = ShaderVersion::get(gl);
    // Integer attributes, `gl_VertexID` and uniform blocks
    if !matches!(version, ShaderVersion::Gl140 | ShaderVersion::Es300) {
        return Err(format!(
            "{version:?} is not supported, needs GLSL 1.40 or ES 3.00"
        ));
    }
    // Highp is mandatory for both stages in ES 3.00
    let precision = if version.is_embedded() {
        "precision highp float;\nprecision highp int;\n"
    } else {
        ""
    };
    let header = format!("{}{precision}", version.version_declaration());
    // SAFETY: GL calls on the painter's context, which is current while it paints
    unsafe {
        let program = gl.create_program()?;
        let mut shaders = Vec::new();
        let mut compiled = Ok(());
        for (kind, source) in [
            (glow::VERTEX_SHADER, MASK_VERTEX),
            (glow::FRAGMENT_SHADER, MASK_FRAGMENT),
        ] {
            let shader = gl.create_shader(kind)?;
            shaders.push(shader);
            gl.shader_source(shader, &format!("{header}{source}"));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                compiled = Err(format!("Mask shader: {}", gl.get_shader_info_log(shader)));
                break;
            }
            gl.attach_shader(program, shader);
        }
        if compiled.is_ok() {
            // Explicit, `layout(location)` would need GLSL 3.30 on desktop
            for (location, name) in [(0, "a_runs_x"), (1, "a_runs_y")] {
                gl.bind_attrib_location(program, location, name);
            }
            gl.link_program(program);
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        if let Err(e) = compiled {
            gl.delete_program(program);
            return Err(e);
        }
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(format!("Mask program: {log}"));
        }
        let Some(block) = gl.get_uniform_block_index(program, "MaskUniforms") else {
            gl.delete_program(program);
            return Err("Mask program has no MaskUniforms".to_owned());
        };
        gl.uniform_block_binding(program, block, MASK_UNIFORMS_BINDING);
        Ok(program)
    }
}

module Mantle

# The device vocabulary, shared with Lava. It lives in `KernelInterface`, which
# exports nothing on purpose, so the names are listed.
#
# It went there while Lava could NOT be a dependency here — Mantle weak-depended
# on it, so an edge back would have closed a cycle. That is no longer why: since
# 2026-08-27 Mantle depends on Lava outright. It stays in KI because a Metal
# backend has to name `MatrixShape` and `DeviceCaps` too, and making it import a
# SPIR-V compiler for them would be absurd.
#
# `DeviceCaps` and the matrix types were each written twice, here and in Lava,
# and bridged by a positional copy in `MantleLavaExt`. Both copies are deleted:
# there is one type, and `caps` fills it in.
# ColorTypes is a DEPENDENCY of Mantle, declared in Project.toml, and
# `runtime/format.jl` documents `RGBA{N0f8}` and `BGRA{N0f8}` as the portable
# way to name a pixel format. It was never `using`-ed, and three separate
# comments concluded from that it was absent — which cost `mtlformat` a
# structural match on `nameof(T)` (so any foreign type called `RGBA` was
# accepted) and cost the Metal backend the portable `Window(backend, w, h)`
# that is in the vocabulary and that Lava answers.
using ColorTypes: RGBA, BGRA, Colorant
export RGBA, BGRA

using KernelInterface: MatrixUse, MatrixA, MatrixB, Accumulator,
    MatrixScope, SubgroupScope, WorkgroupScope, MatrixShape, DeviceCaps
# The one device intrinsic core reaches for. `gemv.jl`'s inner loop reduces
# across the subgroup, and it used to call Lava's `subgroup_add` — the single
# thing keeping a compiler dependency in code that is otherwise portable. KI
# owns the generic now and each backend maps it to its own instruction.
using KernelInterface: sub_group_reduce_add
# The cooperative-matrix type and the nine operations a backend lowers. These
# were Lava's, and naming a SPIR-V compiler's type was the one thing blocking
# GEMM's coopmat half from living in core — see
# `test/vulkan/test_array_algorithm_portability.jl`.
using KernelInterface: CoopMatrix, AcceleratedMatrix, WorkgroupMatrix,
    matrixuse, matrixscope, coopmat_load, coopmat_store, coopmat_muladd,
    coopmat_zero, coopmat_undef, coopmat_convert, coopmat_length,
    coopmat_getcomp, coopmat_setcomp
# Primitive topology: KI's, because a compiler emits execution modes from it and
# every backend creates a pipeline from it. Re-exported below, same as `caps`.
# The device-side shader vocabulary, re-exported so a downstream package writes
# `using Mantle` and names no backend and no compiler. Declared in
# KernelInterface because both Lava and Metal can reach it and neither can
# reach the other's runtime — see the header of `KernelInterface/src/graphics.jl`.
using KernelInterface: vertex_index, instance_index, frag_coord, frag_coord_x,
    frag_coord_y, frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy,
    set_point_size!, sample_texture_2d, emit_vertex!, end_primitive!,
    primitive_id_in, clip_y
export vertex_index, instance_index, frag_coord, frag_coord_x, frag_coord_y,
    frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy, set_point_size!,
    sample_texture_2d, emit_vertex!, end_primitive!, primitive_id_in, clip_y
# The mesh pipeline's half of the same vocabulary. `emit!`/`endprimitive!` are
# what a geometry body calls, and the emitter it is handed decides whether that
# reaches a native geometry stage or a mesh stage — which is why the body needs
# no backend name and no second version. See `KernelInterface/src/mesh.jl`.
using KernelInterface: MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter,
    MeshEmitter, emit!, endprimitive!, set_mesh_vertex!, set_mesh_triangle!,
    set_mesh_line!, set_mesh_point!, set_mesh_outputs!, set_mesh_groups!,
    mesh_thread_index, mesh_group_index
# `Flat` and the interface-list accessors. `GeometryConfig` was Lava's, which
# meant a portable pipeline description could not hold one without depending on
# a SPIR-V compiler; it is beside `MeshConfig` now, for the reason both are
# there — a compiler reads every field.
using KernelInterface: Flat, isflat, unflat, flatnames, smoothnames, valuetypes,
    GeometryConfig, inputvertices
export MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter, MeshEmitter,
    emit!, endprimitive!, set_mesh_vertex!, set_mesh_triangle!, set_mesh_line!,
    set_mesh_point!, set_mesh_outputs!, set_mesh_groups!, mesh_thread_index,
    mesh_group_index
export Flat, isflat, unflat, GeometryConfig
using KernelInterface: rt_launch_id_x, rt_hit_object_trace_ray, rt_reorder_thread,
    rt_hit_object_execute_shader, rt_ignore_intersection, rt_primitive_id,
    rt_instance_id, rt_instance_custom_index, rt_ray_tmax, rt_hit_bary_u,
    rt_hit_bary_v
export rt_launch_id_x, rt_hit_object_trace_ray, rt_reorder_thread,
    rt_hit_object_execute_shader, rt_ignore_intersection, rt_primitive_id,
    rt_instance_id, rt_instance_custom_index, rt_ray_tmax, rt_hit_bary_u,
    rt_hit_bary_v

using KernelInterface: primitivevertices,
    Topology, TriangleList, TriangleStrip, LineList,
    LineStrip, PointList, PatchList, LineListAdjacency, LineStripAdjacency
# `import`, not `using … :` — these get `Mantle.Device` methods below, and
# `caps` in particular becomes one function with a `Device` method here and a
# `KI.Backend` method in each backend.
import KernelInterface: supports, bestshape, caps, matrix_shapes, wggranularity

# Vulkan and Lava are NOT here. They are `[weakdeps]`, and `src/vulkan/` is
# loaded by `ext/MantleVulkanExt.jl` when both are present — which is what lets
# `using Mantle` work on a machine with no Vulkan loader, since `VulkanCore`'s
# `__init__` calls `error()` rather than degrading when it cannot `dlopen` one.
#
# The backend markers are `VulkanAPI`/`MetalAPI` rather than `Vulkan`/`Metal` for
# a related reason: a marker named after its package would shadow the package in
# whichever module loads both.
#
# What is left below is what CORE needs. Several are thinner than they look —
# the backend is the only caller of some — but they load anywhere, so they are
# not what stood between this package and a driverless machine.
import Serialization
import PrecompileTools
# `@setup_workload` is PrecompileTools', re-exported. It used to reach callers
# through the Vulkan backend's export list, which an extension cannot have —
# and Hikari says `Mantle.@setup_workload`, so it belongs on the parent. The
# device-taking `@compile_workload` is a different macro and stays with the
# backend that needs a device to freeze kernels for.
using PrecompileTools: @setup_workload
export @setup_workload, @compile_workload
using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix
using UnsafeAtomics
using AcceleratedKernels
using SPIRV_Tools_jll
using LinearAlgebra
using StaticArrays
using GeometryBasics
using Raycore: Ray
# …and the module itself, which `using Raycore: Ray` does NOT bind. `HWTLAS` and
# `AdaptedAccel` name `Raycore.AbstractAccel` / `Raycore.AbstractAdaptedAccel`
# as supertypes, so a qualified path has to resolve.
import Raycore

include("memory/interval.jl")
include("memory/model.jl")
include("memory/bound.jl")
include("memory/placement.jl")
include("memory/pool.jl")
include("memory/array.jl")
include("memory/csv.jl")

include("sync/usage.jl")      # ResourceKind, which backend.jl dispatches on
include("sync/backend.jl")
include("sync/transition.jl")
include("runtime/backends.jl")
include("runtime/format.jl")
include("runtime/api.jl")
include("runtime/dispatch.jl")
include("phases.jl")
# The graph itself: one set of data structures, shared by every backend.
include("graph/types.jl")
# Before `lifetime.jl`: a `SubmitChannel` holds the outstanding list this declares.
include("graph/submission.jl")
# After it: `SubmitChannel` holds the `Outstanding` list that file declares, and
# `oneshot!` goes through `submitted!`. Phases 2.2 and 2.3 of
# docs/mantle-owns-it.md — the hold list and the recording pool, in core.
include("graph/lifetime.jl")
include("graph/queue.jl")
include("memory/resources.jl")   # needs Resource (api.jl) and blocksize (phases.jl)

# ── Portable array algorithms ─────────────────────────────────────────────────
#
# Written against KernelAbstractions and `KernelInterface.caps`, so any backend
# gets them. They were in `src/vulkan/array/` and the only thing keeping them
# there was a capability query routed through a `VkContext`; see
# `test/vulkan/test_array_algorithm_portability.jl` for what is left to move and
# what is deliberately staying.
include("array/gemv.jl")
include("array/fft.jl")

# ── Geometry: shapes, transforms and the collision pipeline ───────────────────
#
# These were under `src/vulkan/`, and nothing about them was Vulkan's: GJK and
# EPA are arithmetic over a support function, the geometry types are
# descriptions of what to build an acceleration structure FROM, and the
# narrow-phase kernels are KernelAbstractions. Checked rather than assumed —
# none of these files names a Vulkan or a Lava binding in code, only in prose.
#
# What stayed behind is the part that genuinely is one API's: the instance
# RECORD, whose bytes are `VkAccelerationStructureInstanceKHR`, as against the
# transform it carries, which every backend lays out the same way.
# Fixed-function pipeline state, taken over from Lava — see the file for which
# pieces came here, which went to KernelInterface, and why.
include("graphics/state.jl")
include("graphics/resources.jl")   # needs RenderTarget (state.jl) and Window (runtime/api.jl)
include("graphics/stages.jl")      # needs Topology and the KI configs above
include("graphics/pipeline.jl")    # needs the state vocabulary above
include("graphics/mesh.jl")        # needs the same state vocabulary
# The geometry-to-mesh translation. After both pipelines: it reads one and
# builds the other.
include("graphics/lowering.jl")
include("graphics/commands.jl")
# Recording a pass by hand, over the same three primitives the graph uses. The
# imperative verb family this replaces existed only on Vulkan, which is why
# RayMakie's overlay could not composite on Metal at all.
include("graphics/record.jl")
include("graphics/builtins.jl")

include("geometry/transform.jl")
include("geometry/types.jl")
include("geometry/convex_shape.jl")   # ConvexShape and `support`, which GJK/EPA are written against
include("geometry/gjk.jl")
include("geometry/epa.jl")            # needs GJK's simplex helpers
include("geometry/narrow_phase.jl")   # needs both

# ── The graph's backend interface ─────────────────────────────────────────────
#
# Mantle owns the graph; this is the whole of what a backend answers for it.
include("graph/backend.jl")

# ── Ray tracing ───────────────────────────────────────────────────────────────
include("raytracing/pipeline.jl")
# The instances a top-level structure holds: the order, the handles, the
# reindex on delete. Portable; a backend supplies only the batch type.
include("raytracing/batches.jl")
include("raytracing/accel.jl")
include("raytracing/api.jl")

# Building and running the graph. Last of the core includes: it names `Buffer`
# (memory/resources.jl), `DrawIndirectCommand` (graphics/pipeline.jl) and the
# hooks in `graph/backend.jl`, so everything it touches has to exist first.
include("graph/build.jl")
# Running a compiled graph on a KernelAbstractions backend — shared by the host
# and Metal backends; Vulkan records commands instead and overrides it.
include("graph/kalaunch.jl")

# ── backends ──────────────────────────────────────────────────────────────────
#
# Everything above this line is what a backend implements against, and it must
# load with no driver and no compiler present — `test_pool.jl` drives the whole
# allocator with neither.
#
# `src/vulkan/` is loaded by `ext/MantleVulkanExt.jl` when both Lava and Vulkan
# are, and `src/metal/` will be loaded the same way by `ext/MantleMetalExt.jl`.
# The host backend needs no weak dependency, so it is not an extension: KA is a
# hard dependency of core, and an extension triggered on a hard dependency
# always fires. It is included at the BOTTOM of this file, after every
# declaration it adds a method to.

export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is deliberately not exported: it is a Span predicate nothing outside
# this package calls, and GeometryBasics exports the same name.
export segments, maxload, hmax, fragmentation
export place, LowestFit, BestFit
export Pool, Block, Region, acquire!, release!, trim!, reserved, retire!, reclaim!, fence, passed, waitfor, waitfor!
export Arena, reserve!, tenant!, untenant!, sharing, remap!, headroom, largestfree, remappable
export DeviceArray, giveup!, blocksize
export upload!, download, deviceview, bufferusage, devicecopy!, Persistent, Unified, Readback
export rawalloc, rawfree, constraintof, compatible, maxalloc, mergeconstraints
export readproblem

export Backend, VulkanAPI, MetalAPI, WebGPUAPI, HostAPI
export Usage, ResourceKind, BufferKind, ImageKind, AccelKind
export Access, ReadOnly, WriteOnly, ReadWrite, NoAccess, Src, Dst
export Vertices, Indices, Indirect, Predicated, Uniform, Sampled, Present, Undefined
export CopySrc, CopyDst, TraceRead, TraceBuild, Storage, ColorAttachment, Depth, Unordered, Traced
export reads, writes, discards, kindof, aliasable, evictable, unordered, inner
export Transition, ResourceState, transition!, transitions, needs_transition, barrierhazards
export stages, access, layout

# Fixed-function pipeline state and primitive topology. These were Lava's — a
# SPIR-V compiler exporting `AlphaBlend` and `CullBack` — and nothing in it
# dispatched on them. `Topology` is re-exported from KernelInterface rather than
# defined, since the compiler needs the same names.
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied
export CullFace, NoCull, CullBack, CullFront
export DepthMode, DepthLess, DepthLessEq, DepthGreater, DepthAlways, DepthOff
export RenderTarget

# The resources a pipeline is built from. Abstract here, concrete in whichever
# backend is loaded — see `graphics/resources.jl` for why each is one or the
# other, and which three turned out portable outright.
export Texture, Texture1D, Texture2D, Sampler, SampledTexture, TextureBindings
export Framebuffer, WindowTarget, OffscreenTarget, CompiledGraphicsPipeline
export HWTLAS, AccelBuildContext, ExternalImage
# A submission channel and what a submission holds — 2.2 and 2.3. `hold!` is what
# `pin!` meant, with the lifetime owned by core instead of by a backend.
export SubmitChannel, channelof, deviceof, hold!, oneshot!, acquire!, Submission
export Stamp, stampof, retire!, reclaim!, drain!, handover!
export allocate_batch_queue!, release_batch_queue!, submit!, waitidle
export supports_graphics, supports_geometry_stage, supports_tessellation, supports_batch_queue, use_bindings!, supports_rt_pipeline
export batchqueue
export defaultbackend, availablebackends, eachbackend, register_backend!, register_kernel_recorder!
export devicearray
export bind_textures

# Rasterization commands. These were `begin_pass!` and friends.
export begin_pass!, end_pass!, draw_in_pass!, draw_indexed_in_pass!,
       draw_indirect_in_pass!, set_viewport!, reset_device!

# The graph's backend interface. `isdepth` is a definition, not a hook — see
# `graph/backend.jl` for why it and `aspect` swapped places.
export isdepth, target_extent, checkextents, refit!,
       collect!
export blit!, present_frame!, acquire_next_image!, transition_image!
export InstanceBatches, register!, batchof, ninstances
export readback_framebuffer, readback_window, readback_target

# Pipeline descriptions and the indirect draw record.
# The stages a pipeline is made of. Each declares only what it PRODUCES: its
# inputs are the previous stage's outputs, so no interface is written twice.
export ShaderStage, VertexShader, FragmentShader, GeometryShader, MeshShader,
    ObjectShader
export stagefunction, stageoutputs, stageinputs, stageconfig, flatoutputs,
    smoothoutputs, outputtype
# What the fragment stage reads, which depends on which stages a pipeline has.
export lastgeometrystage, fragmentinputs, fragmentinputtype
# Recording a pass by hand. Shaped like the graph's `render!`/`draw!` on purpose.
export pass!, viewport!, bindings!, PassRecorder
export setviewport!, colorimage, depthimage, currentimage, blittarget, todevice
export lower_geometry_to_mesh
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export DrawIndirectCommand
# The mesh pipeline. Described here, run by a backend that answers
# `supports_mesh_pipeline`; see `graphics/mesh.jl`.
export MeshPipeline, meshconfig, objectconfig
export supports_mesh_pipeline
# The shader builtins were declared here and are now KernelInterface's, imported
# and re-exported above: phase 1.4 deleted the file, see docs/mantle-owns-it.md.
export RayTracingPipeline, AdaptedAccel

# Hardware ray tracing.
export build_accel!, refit_tlas!, set_anyhit_pipeline!
export trace_rays!, trace_rays_indirect!
# The DECLARATION verb, beside `dispatch!` rather than beside the recording ones
# above: `trace!` says a pass traces, the others record a trace that was already
# decided on.
export trace!, Trace
export trace_closest_hits!, trace_closest_hits_indirect!
export trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export primitivevertices
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList,
       PatchList, LineListAdjacency, LineStripAdjacency

# Geometry, from `src/geometry/`. These were exported by the Vulkan backend
# because that is where the files happened to sit; none of them names anything a
# driver owns, and `support` in particular is the single most-used name Hikari
# and RayMakie take from this package.
export Mat3x4f, identity_transform, quat_to_rot3x3, build_4x3, build_4x3_pervec
export AABB, AABBsGeometry, TrianglesGeometry, GeometryType
export ConvexShape, UnitCube, support
export GJKResult, gjk, EPAResult, epa
export ContactRecord, NO_CONTACT, narrow_phase_kernel, narrow_phase_contacts_kernel

export pixelbytes
export LoadOp, Clear, Keep, Discard
export Device, Resource, Graph, Plan, Transient, Window, backend, screenshot
export DeviceInfo, devices, defaultdevice!
export DeviceCaps, caps
export MatrixShape, MatrixUse, MatrixA, MatrixB, Accumulator
export CoopMatrix, AcceleratedMatrix, WorkgroupMatrix, matrixuse, matrixscope
export MatrixScope, SubgroupScope, WorkgroupScope, supports, bestshape
# `copy!` is deliberately not exported: the name exists in Base, and exporting it
# would make the bare name ambiguous in any module that does `using Mantle`.
export Buffer, GPURef, Surface, Attribute, draw!, dispatch!, render!, compute!
# Its own verb because the two APIs differ at allocation; see `memory/resources.jl`.
export indexbuffer
# `repeat!` and its vocabulary: a loop recorded once, whose trip count the device
# decides. `Predicate` is exported because a kernel writing predicates by hand
# names the type; `supportspredicate` because a caller may want to pick between
# `repeat!` and a host loop rather than be thrown at.
export repeat!, Predicate, supportspredicate
export Dispatch, DeviceRange, countresource, indirectcount!, passof, graphof, touch!
export newpass, handle, dispatches
export IdTable, resourceid, byid, checklive
export Phase, Dag, Schedule, Liveness, Place, Aliasing, Barriers, Pipelines
export Policy, Compact, Overlap
export PHASES, compile!
# `update!` is deliberately not exported: ComputePipeline exports it, and that is
# the one a Makie-shaped caller means — `update!(plot; positions = …)` sets an
# attribute. Mantle's writes a buffer now, which is a different verb with the same
# spelling, so it stays `Mantle.update!` and the bare name belongs to Makie's.
export run!, npipelines, capacity, use, peakbytes, naivebytes, storage, free!
export record!, recorded
export timings, PassTiming, NSAMPLES


"""Whether this plan's commands have been written and are what `run!` submits.
False by default, which is the honest answer for a backend that never builds
one."""
recorded(plan) = false

# `rebind!` is gone. It re-read every `Ref` a recorded plan was given and wrote
# the current value into the plan's argument memory, once per (entry, `Ref`)
# pair, so that a recorded plan meant the same thing as an interpreted one. A
# value that changes is a [`GPURef`](@ref) now: the commands hold its address,
# one `Update` writes it, and there is nothing per-run for a backend to offer.

function update! end
function use end

"""
    peakbytes(plan) -> Int
    naivebytes(plan) -> Int

Bytes a plan actually reserved for transients — the maximum cross section — and
what allocating every transient separately would have cost. The pair is the whole
report on whether aliasing bought anything.

Both read a field, and both did so identically in every backend, so the method is
here and a `Plan` supplies the fields. `Plan` is the right place for it rather
than a per-backend accessor: the numbers come out of `Analysis`, which is core's.
"""
peakbytes(pl::Plan) = pl.peak
@doc (@doc peakbytes) naivebytes(pl::Plan) = pl.naive
"""
    storage(x)

The backend object behind a resource: what actually gets bound, copied or
launched with. Anything that is already one is itself.

The identity fallback lives HERE and not in an extension. Both extensions had
their own copy, and with both loaded the second overwrote the first — which is an
error during precompilation, so SAM2Runner simply failed to build. One
implementation, in core, is the same rule that moved `Buffer` and the phases; a
one-line method is not an exception to it.
"""
storage(x) = x

"""
    capacity(dev) -> Int

How many bytes of device memory a transient arena may use.

`typemax(Int)` by default, for the same reason [`maxalloc`](@ref) has one: a
backend that will not say is not blocked, it is merely unbounded. A backend that
CAN say should — it is what turns "the driver returned VK_ERROR_OUT_OF_
DEVICE_MEMORY somewhere later" into a message naming the buffers that did not
fit.
"""
capacity(dev) = typemax(Int)

# `__init__` is `MantleVulkanExt`'s: both halves of it — the pipeline builder
# thread and the `atexit` device-lost hook — reach into the Vulkan context, and
# there is nothing for them to do in a session with no backend loaded.


# The host backend, last: every method in it is a method on something declared
# above, and it says so — the file qualifies all 83 of them as `Mantle.x`,
# because it was `ext/MantleHostExt.jl` until the KernelAbstractions weakdep
# turned out to be unworkable. Left qualified rather than rewritten: it reads as
# the backend-facing surface it is, and it stays trivially re-extractable.
include("host/host.jl")

end

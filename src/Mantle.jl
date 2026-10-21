module Mantle

# Shared device/shader vocabulary lives in KernelInterface; native lowering
# belongs to the selected backend. Color formats dispatch on ColorTypes types.
using ColorTypes: RGBA, BGRA, Colorant
export RGBA, BGRA

using KernelInterface: MatrixUse, MatrixA, MatrixB, Accumulator,
    MatrixScope, SubgroupScope, WorkgroupScope, MatrixShape, DeviceCaps
# Portable subgroup reduction; each backend supplies the intrinsic.
using KernelInterface: sub_group_reduce_add
# Portable cooperative matrices, lowered by each backend.
using KernelInterface: CoopMatrix, AcceleratedMatrix, WorkgroupMatrix,
    matrixuse, matrixscope, coopmat_load, coopmat_store, coopmat_muladd,
    coopmat_mul, coopmat_add,
    coopmat_zero, coopmat_undef, coopmat_convert, coopmat_length,
    coopmat_getcomp, coopmat_setcomp
# Portable tensor addressing for workgroup matrices.
using KernelInterface: TensorLayout, TensorView, TENSOR_CLAMP_UNDEFINED,
    TENSOR_CLAMP_CONSTANT, TENSOR_CLAMP_TO_EDGE, tensor_layout, tensor_setdim,
    tensor_setstride, tensor_setclampvalue, tensor_slice, tensor_view, tensor_load,
    tensor_store
# Portable shader builtins, re-exported for user shader bodies.
using KernelInterface: vertex_index, instance_index, frag_coord, frag_coord_x,
    frag_coord_y, frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy,
    set_point_size!, sample_texture_2d, emit_vertex!, end_primitive!,
    primitive_id_in, clip_y, discard
export vertex_index, instance_index, frag_coord, frag_coord_x, frag_coord_y,
    frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy, set_point_size!,
    sample_texture_2d, emit_vertex!, end_primitive!, primitive_id_in, clip_y,
    discard
# Explicit vertex index for stages without a vertex_index builtin.
using KernelInterface: VertexIndex, wantsvertexindex
export VertexIndex
# Emitter dispatch selects native geometry or mesh lowering for the same body.
using KernelInterface: MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter,
    MeshEmitter, emit!, endprimitive!, set_mesh_vertex!, set_mesh_triangle!,
    set_mesh_line!, set_mesh_point!, set_mesh_primitive_data!, set_mesh_outputs!,
    set_mesh_groups!, mesh_thread_index, mesh_group_index
# Primitive counts used by geometry-to-mesh lowering.
using KernelInterface: nvertices, nprimitives
# Interpolation and stage-interface descriptions shared with compilers.
using KernelInterface: Flat, isflat, unflat, flatnames, smoothnames, valuetypes,
    GeometryConfig, inputvertices
export MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter, MeshEmitter,
    emit!, endprimitive!, set_mesh_vertex!, set_mesh_triangle!, set_mesh_line!,
    set_mesh_point!, set_mesh_primitive_data!, set_mesh_outputs!, set_mesh_groups!,
    mesh_thread_index, mesh_group_index
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
# Primitive assembly used by draw planning and mesh shaders.
using KernelInterface: primitivestride, primitivecount, firstinputvertex
# Extend the shared generics with Device methods.
import KernelInterface: supports, bestshape, caps, matrix_shapes, wggranularity

# Qualified interface namespace for backend method definitions.
import KernelInterface
const KI = KernelInterface


# Platform backend includes follow core definitions at the bottom of this file.
import Serialization
import Mmap
import PrecompileTools
# Re-export workload macros for consumer packages.
using PrecompileTools: @setup_workload, @compile_workload
export @setup_workload, @compile_workload
using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
# Extend GPUArrays.unsafe_free!; a separate local generic breaks array finalizers.
import GPUArrays: unsafe_free!
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
# Common names must be defined before either platform backend is included.
const KA = KernelAbstractions
using Base: @propagate_inbounds
using ColorTypes.FixedPointNumbers: FixedPoint, N0f8
using ColorTypes: FixedPointNumbers
using Raycore: Ray
# Qualified Raycore names are also used by acceleration implementations.
import Raycore

# Dispatch policy is explicit; Mantle adds no task-local policy state.
include("memory/interval.jl")
include("memory/model.jl")
include("memory/bound.jl")
include("memory/placement.jl")
include("memory/pool.jl")
include("memory/csv.jl")

include("sync/usage.jl")      # ResourceKind, which backend.jl dispatches on
include("sync/backend.jl")
include("sync/transition.jl")
include("runtime/backends.jl")
include("runtime/cache.jl")
include("runtime/backendhooks.jl")  # declared here, answered in vulkan/ or metal/
include("runtime/format.jl")
include("runtime/api.jl")
include("memory/array.jl")    # names `Device` and `backend`, both from api.jl
# Device lifecycle hooks; trydevicearray needs the array API above.
include("runtime/lifecycle.jl")
include("runtime/dispatch.jl")
include("phases.jl")
# The graph itself: one set of data structures, shared by every backend.
include("graph/types.jl")
include("graph/access.jl")   # what a kernel does to its arguments, read off the kernel
# Before `lifetime.jl`: a `SubmitChannel` holds the outstanding list this declares.
include("graph/submission.jl")
# Submission tracking precedes the channel lifetime owner.
include("graph/lifetime.jl")
include("graph/queue.jl")
include("memory/resources.jl")   # needs Resource (api.jl) and blocksize (phases.jl)

# Portable array algorithms and launch selection.
include("array/launch.jl")   # what a library routine IS, before it is submitted
include("array/indexing.jl") # portable index decomposition and launch geometry
include("array/gemv.jl")
include("array/gemm.jl")
include("array/gemm_tensile.jl") # rocBLAS's RDNA3 kernel, the default for aligned fp16 shapes
include("array/gemm_cm2.jl")   # workgroup-scope matrices through tensor addressing
include("array/fft.jl")

# Graphics descriptions and portable geometry/collision algorithms.
include("graphics/state.jl")
include("graphics/resources.jl")   # needs RenderTarget (state.jl) and Window (runtime/api.jl)
include("graphics/stages.jl")      # needs Topology and the KI configs above
include("graphics/pipeline.jl")    # needs the state vocabulary above
include("graphics/mesh.jl")        # needs the same state vocabulary
# Lowering reads both pipeline descriptions.
include("graphics/lowering.jl")
include("graphics/commands.jl")
# Portable manual pass recording uses the graph backend primitives.
include("graphics/record.jl")
include("graphics/builtins.jl")

include("geometry/transform.jl")
include("geometry/types.jl")
include("geometry/convex_shape.jl")   # ConvexShape and `support`, which GJK/EPA are written against
include("geometry/gjk.jl")
include("geometry/epa.jl")            # needs GJK's simplex helpers
include("geometry/narrow_phase.jl")   # needs both

# Graph backend protocol and shared argument packing.
include("graph/backend.jl")
# Shared packing and argument traversal for both recording backends.
include("graph/packing.jl")

# ── Ray tracing ───────────────────────────────────────────────────────────────
include("raytracing/pipeline.jl")
# Shared instance ordering, registration and mutation.
include("raytracing/batches.jl")
include("raytracing/accel.jl")
include("raytracing/geometry.jl")
include("raytracing/instances.jl")
include("raytracing/mutation.jl")
include("raytracing/api.jl")

# Graph compilation requires resources, graphics commands and backend hooks above.
include("graph/build.jl")
# Shared host/Metal launch execution; Vulkan overrides it with native recording.
include("graph/kalaunch.jl")

# Public core API. Platform implementations follow the declarations below.
export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is deliberately not exported: it is a Span predicate nothing outside
# this package calls, and GeometryBasics exports the same name.
export segments, maxload, hmax, fragmentation
export place, LowestFit, BestFit
export Pool, Block, Region, acquire!, release!, trim!, reserved, retire!, reclaim!, fence, passed, waitfor, waitfor!
export Arena, reserve!, tenant!, untenant!, sharing, remap!, headroom, arenaroom, largestfree, remappable
export DeviceArray, giveup!, blocksize
export upload!, download, deviceview, bufferusage, devicecopy!, Persistent, Unified, Readback
export rawalloc, rawfree, constraintof, compatible, maxalloc, mergeconstraints
export readproblem

export Backend, VulkanAPI, MetalAPI, WebGPUAPI, ROCmAPI
export Usage, ResourceKind, BufferKind, ImageKind, AccelKind
export Access, ReadOnly, WriteOnly, ReadWrite, NoAccess, Src, Dst
export Vertices, Indices, Indirect, Predicated, Uniform, Sampled, Present, Undefined
export CopySrc, CopyDst, TraceRead, TraceBuild, Storage, ColorAttachment, Depth, Unordered, Traced
export reads, writes, discards, kindof, aliasable, evictable, unordered, inner
export Transition, ResourceState, transition!, transitions, needs_transition, barrierhazards
export stages, access, layout

# Portable fixed-function state; topology is shared with KernelInterface.
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied, NoWrite, PerAttachment
export CullFace, NoCull, CullBack, CullFront
export DepthMode, DepthLess, DepthLessEq, DepthLessEqReadOnly, DepthGreater, DepthAlways, DepthOff
export RenderTarget

# Portable resource descriptions and native storage interfaces.
export Texture, Texture1D, Texture2D, Sampler, SampledTexture, TextureBindings
export upload_texture_data!, mipmapped
export argidentity
export Framebuffer, WindowTarget, OffscreenTarget, CompiledGraphicsPipeline
export HWTLAS, AccelBuildContext, ExternalImage
# Core submission ownership and retirement.
export SubmitChannel, channelof, deviceof, hold!, oneshot!, acquire!, Submission
export ownthread, WrongThread
export Stamp, stampof, retire!, reclaim!, drain!, handover!
export allocate_batch_queue!, release_batch_queue!, submit!, waitidle
export supports_graphics, supports_geometry_stage, supports_tessellation, supports_batch_queue, use_bindings!, supports_rt_pipeline
export supports_procedural_traversal, supports_hwtlas, supports_float64, supports_int64_atomics
export batchqueue
export defaultbackend, availablebackends, eachbackend, register_backend!
export devicearray
export bind_textures

# Rasterization commands. These were `begin_pass!` and friends.
export begin_pass!, end_pass!, draw_in_pass!, draw_indexed_in_pass!,
       draw_indirect_in_pass!, set_viewport!, reset_device!

# Graph backend operations.
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
    smoothoutputs, outputtype, ntextures
# What the fragment stage reads, which depends on which stages a pipeline has.
export lastgeometrystage, fragmentinputs, fragmentinputtype
# Recording a pass by hand. Shaped like the graph's `render!`/`draw!` on purpose.
export pass!, viewport!, bindings!, PassRecorder
export setviewport!, colorimage, depthimage, currentimage, blittarget, todevice
# Geometry-to-mesh lowering and its compiled description.
export lower_geometry_to_mesh, GeometryAsMesh, GeometryPipeline
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export DrawIndirectCommand
# The mesh pipeline. Described here, run by a backend that answers
# `supports_mesh_pipeline`; see `graphics/mesh.jl`.
export MeshPipeline, meshconfig, objectconfig
export supports_mesh_pipeline
# The shader builtins are KernelInterface's, imported and re-exported above.
export RayTracingPipeline, AdaptedAccel
# The procedural-geometry protocol: a scene implements these to put geometry
# that is not triangles into the traversal. See `raytracing/accel.jl`.
export procedural_miss, procedural_candidate, procedural_commit, procedural_bary
export candidate_primitive_index, candidate_object_ray, commit_intersection!

# Hardware ray tracing.
export build_accel!, build_blas_aabb, refit_tlas!, set_anyhit_pipeline!
export trace_rays!, trace_rays_indirect!
# Declare tracing passes; trace_rays! records already planned native work.
export trace!, Trace
export trace_closest_hits!, trace_closest_hits_indirect!
export trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export primitivevertices
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList,
       PatchList, LineListAdjacency, LineStripAdjacency

# Portable geometry, transforms and collision queries.
export Mat3x4f, identity_transform, quat_to_rot3x3, build_4x3, build_4x3_pervec
export AABB, AABBsGeometry, TrianglesGeometry, GeometryType
export ConvexShape, UnitCube, support
export GJKResult, gjk, EPAResult, epa
export ContactRecord, NO_CONTACT, narrow_phase_kernel, narrow_phase_contacts_kernel

export pixelbytes
export LoadOp, Clear, Keep, Discard
export Device, Resource, Graph, Plan, Transient, Window, backend, screenshot
export DeviceInfo, devices, defaultdevice!
export DebugConfig, debugenv, validating, validationmessages!, submissions, retired,
    trydevicearray
export DeviceCaps, caps
export MatrixShape, MatrixUse, MatrixA, MatrixB, Accumulator
export CoopMatrix, AcceleratedMatrix, WorkgroupMatrix, matrixuse, matrixscope
export coopmat_load, coopmat_store, coopmat_muladd, coopmat_mul, coopmat_add,
    coopmat_zero, coopmat_undef, coopmat_convert, coopmat_length,
    coopmat_getcomp, coopmat_setcomp
export MatrixScope, SubgroupScope, WorkgroupScope, supports, bestshape
# `copy!` is deliberately not exported: the name exists in Base, and exporting it
# would make the bare name ambiguous in any module that does `using Mantle`.
export Buffer, GPURef, Surface, Attribute, draw!, dispatch!, render!, runonce!
export DrawBinding, rebind!
export ResourceView, viewof
# Its own verb because the two APIs differ at allocation; see `memory/resources.jl`.
export indexbuffer
# Device-decided recorded loops and their predicates.
export repeat!, Predicate, supportspredicate
export kernelcompiles, resetkernelcompiles!
export Dispatch, DeviceRange, countresource, indirectcount!, passof, graphof, touch!
export newpass, handle, dispatches
export IdTable, resourceid, byid, checklive
export Phase, Dag, Schedule, Liveness, Place, Aliasing, Barriers, Pipelines
export Policy, Compact, Overlap
export PHASES, compile!
# Keep update! qualified to avoid a name collision with Makie/ComputePipeline.
export run!, npipelines, capacity, use, peakbytes, naivebytes, storage, free!
export record!, recorded
export timings, PassTiming, NSAMPLES


"""Whether run! submits commands already recorded for this plan. Default false."""
recorded(plan) = false

function update! end
function use end

"""    peakbytes(plan) -> Int
    naivebytes(plan) -> Int

Transient bytes reserved with aliasing, and the sum of separate transient sizes."""
peakbytes(pl::Plan) = pl.peak
@doc (@doc peakbytes) naivebytes(pl::Plan) = pl.naive
"""    storage(x)

Resolve a graph resource to its backend storage; backend objects return themselves."""
storage(x) = x

"""Resolve grouped graph arguments element by element; keep one rule for recording and eager launch."""
storage(x::Tuple) = argmap(storage, x)

"""    argmap(f, args::Tuple) -> Tuple

Unroll mapping at any concrete tuple length. Base.map allocates above 31 elements;
a literal tuple also avoids ntuple closure allocation for large shader arguments."""
@generated function argmap(f::F, args::T) where {F, T<:Tuple}
    # Inference may ask about an abstract tuple; a call never passes one.
    isconcretetype(T) || return :(map(f, args))
    return Expr(:tuple, (:(f(getfield(args, $i))) for i in 1:fieldcount(T))...)
end

"""    basealignment(x) -> Int

Known power-of-two device alignment, or 1 when unknown. Wide-load kernels must
check the operand's alignment; declared resources report placement guarantees."""
basealignment(x) = 1
basealignment(b::Buffer) = basealignment(storage(b))
basealignment(r::BufferRange) = basealignment(r.parent)
basealignment(v::ResourceView{T}) where {T} =
    min(basealignment(v.parent), powalign(v.offset * sizeof(T)))
"""Unplaced transient alignment uses the common backend floor; test_basealignment.jl checks it."""
basealignment(::TransientBuffer) = TRANSIENT_ALIGN_FLOOR
const TRANSIENT_ALIGN_FLOOR = 16

"""The power of two dividing byte offset or address `off`; zero constrains nothing.
Any integer type: device addresses have their top bits set and are no `Int`."""
powalign(off::Integer) = iszero(off) ? typemax(Int) : 1 << min(trailing_zeros(off), 62)

"""    capacity(dev) -> Int

Device-memory budget for transient placement. Default typemax(Int) means unknown;
backends report a bound when available so placement can diagnose excess demand."""
capacity(dev) = typemax(Int)

# Metal is included on Apple; Vulkan is included when a loader is available.
# ROCm is an optional extension. Native methods extend this module's generics.
# Import Vulkan qualified because its resource names overlap Mantle's.
import Vulkan as VK
@static if Sys.isapple()
    using Metal: MtlArray
    using Metal.ObjectiveC: NSArray
    using KernelInterface: DeviceCaps, MatrixShape, MatrixScope, SubgroupScope
    # Extend Raycore's traversal interface.
    import Raycore: closest_hit, any_hit, sync!, world_bound, n_geometries,
        n_instances, update_transforms!, update_transform!, wait_for_gpu!
    include("metal/metal.jl")
end
# Vulkan also supplies an explicit lavapipe CPU device when its loader is present.
@static if !Sys.isapple() || VK.HAS_LOADER
    using Vulkan: unwrap, iserror, unwrap_error
    include("vulkan/vulkan.jl")
end

"""    Device() -> Device

Default platform device: Metal on Apple, Vulkan elsewhere. Explicit API constructors
and device selectors can choose another context without changing that default."""
Device() = @static Sys.isapple() ? Device(MetalAPI()) : Device(VulkanAPI())

"""    devices() -> Vector{DeviceInfo}

Enumerate the default platform API: Metal on Apple, Vulkan elsewhere."""
devices() = @static Sys.isapple() ? devices(MetalAPI()) : devices(VulkanAPI())

"""What selects a device: a case-insensitive substring of its name or driver, its
index in [`devices`](@ref), or a predicate over [`DeviceInfo`](@ref). See
[`selectdevice`](@ref)."""
const DeviceSelector = Union{AbstractString,Integer,Function}

"""    Device(select) -> Device

Create an independent context selected by name substring, index or predicate;
defaultdevice! installs it as the process default. On Apple with a Vulkan loader,
try Metal first and Vulkan if no Metal device matches (including lavapipe)."""
function Device(select::DeviceSelector)
    @static if Sys.isapple() && VK.HAS_LOADER
        any(i -> admits(select, i), devices(MetalAPI())) || return Device(VulkanAPI(); select)
        return Device(MetalAPI(); select)
    elseif Sys.isapple()
        return Device(MetalAPI(); select)
    else
        return Device(VulkanAPI(); select)
    end
end

"""    defaultdevice!(select) -> Device

Create the selected device, install it as process default and return it."""
defaultdevice!(select::DeviceSelector) = defaultdevice!(Device(select))

function __init__()
    @static Sys.isapple() && initbackend!(MetalAPI())
    @static (!Sys.isapple() || VK.HAS_LOADER) && initbackend!(VulkanAPI())
    return nothing
end

end

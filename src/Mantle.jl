module Mantle

# KernelInterface owns the device and shader vocabulary; each backend lowers it.
using ColorTypes: RGBA, BGRA, Colorant
export RGBA, BGRA

using KernelInterface: MatrixUse, MatrixA, MatrixB, Accumulator,
    MatrixScope, SubgroupScope, WorkgroupScope, MatrixShape, DeviceCaps
using KernelInterface: sub_group_reduce_add
using KernelInterface: CoopMatrix, AcceleratedMatrix, WorkgroupMatrix,
    matrixuse, matrixscope, coopmat_load, coopmat_store, coopmat_muladd,
    coopmat_mul, coopmat_add,
    coopmat_zero, coopmat_undef, coopmat_convert, coopmat_length,
    coopmat_getcomp, coopmat_setcomp
using KernelInterface: TensorLayout, TensorView, TENSOR_CLAMP_UNDEFINED,
    TENSOR_CLAMP_CONSTANT, TENSOR_CLAMP_TO_EDGE, tensor_layout, tensor_setdim,
    tensor_setstride, tensor_setclampvalue, tensor_slice, tensor_view, tensor_load,
    tensor_store
using KernelInterface: vertex_index, instance_index, frag_coord, frag_coord_x,
    frag_coord_y, frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy,
    set_point_size!, sample_texture_2d, emit_vertex!, end_primitive!,
    primitive_id_in, clip_y, discard
export vertex_index, instance_index, frag_coord, frag_coord_x, frag_coord_y,
    frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy, set_point_size!,
    sample_texture_2d, emit_vertex!, end_primitive!, primitive_id_in, clip_y,
    discard
using KernelInterface: VertexIndex, wantsvertexindex
export VertexIndex
using KernelInterface: MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter,
    MeshEmitter, emit!, endprimitive!, set_mesh_vertex!, set_mesh_triangle!,
    set_mesh_line!, set_mesh_point!, set_mesh_primitive_data!, set_mesh_outputs!,
    set_mesh_groups!, mesh_thread_index, mesh_group_index
using KernelInterface: nvertices, nprimitives
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
using KernelInterface: primitivestride, primitivecount, firstinputvertex
# Extended with Device methods.
import KernelInterface: supports, bestshape, caps, matrix_shapes, wggranularity

import KernelInterface
const KI = KernelInterface


import Serialization
import Mmap
import PrecompileTools
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
const KA = KernelAbstractions
using Base: @propagate_inbounds
using ColorTypes.FixedPointNumbers: FixedPoint, N0f8
using ColorTypes: FixedPointNumbers
using Raycore: Ray
import Raycore

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
include("runtime/hostwait.jl")   # every host wait on the device; backends call it
include("runtime/cache.jl")
include("runtime/backendhooks.jl")  # declared here, answered in vulkan/ or metal/
include("runtime/format.jl")
include("runtime/api.jl")
include("memory/array.jl")    # names `Device` and `backend`, both from api.jl
# `trydevicearray` needs the array API above.
include("runtime/lifecycle.jl")
include("runtime/dispatch.jl")
include("phases.jl")
include("graph/types.jl")
include("graph/access.jl")
# Before `lifetime.jl`: a `SubmitChannel` holds the outstanding list this declares.
include("graph/submission.jl")
include("graph/lifetime.jl")
include("graph/queue.jl")
include("memory/resources.jl")   # needs Resource (api.jl) and blocksize (phases.jl)

include("array/launch.jl")
include("array/indexing.jl")
include("array/gemv.jl")
include("array/gemm.jl")
include("array/gemm_tensile.jl") # rocBLAS's RDNA3 kernel, default for aligned fp16 shapes
include("array/gemm_cm2.jl")
include("array/fft.jl")

include("graphics/state.jl")
include("graphics/resources.jl")   # needs RenderTarget (state.jl) and Window (runtime/api.jl)
include("graphics/stages.jl")      # needs Topology and the KI configs above
include("graphics/pipeline.jl")    # needs the state vocabulary above
include("graphics/mesh.jl")        # needs the same state vocabulary
# Lowering reads both pipeline descriptions.
include("graphics/lowering.jl")
include("graphics/commands.jl")
include("graphics/record.jl")
include("graphics/builtins.jl")

include("geometry/transform.jl")
include("geometry/types.jl")
include("geometry/convex_shape.jl")   # ConvexShape and `support`, used by GJK/EPA
include("geometry/gjk.jl")
include("geometry/epa.jl")            # needs GJK's simplex helpers
include("geometry/narrow_phase.jl")   # needs both

include("graph/backend.jl")
include("graph/packing.jl")

include("raytracing/pipeline.jl")
include("raytracing/batches.jl")
include("raytracing/accel.jl")
include("raytracing/geometry.jl")
include("raytracing/instances.jl")
include("raytracing/mutation.jl")
include("raytracing/api.jl")

# Needs resources, graphics commands and backend hooks above.
include("graph/build.jl")
# KernelAbstractions plan execution; Vulkan overrides it with native recording.
include("graph/readback.jl")
include("graph/kalaunch.jl")
export ReadbackFuture

export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is not exported: GeometryBasics exports the same name.
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
export reads, writes, discards, kindof, unordered, inner
export Transition, ResourceState, transition!, transitions, needs_transition, barrierhazards
export stages, access, layout

export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied, NoWrite, PerAttachment
export CullFace, NoCull, CullBack, CullFront
export DepthMode, DepthLess, DepthLessEq, DepthLessEqReadOnly, DepthGreater, DepthAlways, DepthOff
export RenderTarget

export Texture, Texture2D, Sampler, SampledTexture, TextureBindings
export upload_texture_data!, mipmapped
export argidentity
export Framebuffer, WindowTarget, OffscreenTarget, CompiledGraphicsPipeline
export HWTLAS, AccelBuildContext, ExternalImage
export SubmitChannel, deviceof, hold!, oneshot!, acquire!, Submission
export ownthread, WrongThread
export Stamp, stampof, retire!, reclaim!, drain!, handover!
export allocate_batch_queue!, release_batch_queue!, submit!, waitidle
export supports_graphics, supports_geometry_stage, supports_tessellation, supports_batch_queue, use_bindings!, supports_rt_pipeline
export supports_procedural_traversal, supports_hwtlas, supports_float64, supports_int64_atomics
export batchqueue
export defaultbackend, availablebackends, eachbackend, register_backend!
export devicearray
export bind_textures

export begin_pass!, end_pass!, draw_in_pass!, draw_indexed_in_pass!,
       draw_indirect_in_pass!, set_viewport!, reset_device!

export isdepth, target_extent, checkextents, refit!,
       collect!
export blit!, present_frame!, acquire_next_image!
export InstanceBatches, register!, batchof, ninstances
export readback_framebuffer, readback_window, readback_target

export ShaderStage, VertexShader, FragmentShader, GeometryShader, MeshShader,
    ObjectShader
export stagefunction, stageoutputs, stageinputs, stageconfig, flatoutputs,
    smoothoutputs, outputtype, ntextures
export lastgeometrystage, fragmentinputs, fragmentinputtype
export pass!, viewport!, bindings!, PassRecorder
export setviewport!, colorimage, depthimage, currentimage, blittarget, todevice
export lower_geometry_to_mesh, GeometryAsMesh, GeometryPipeline
export GraphicsPipeline, Rasterizer
export DrawIndirectCommand
export MeshPipeline, meshconfig, objectconfig
export supports_mesh_pipeline
export RayTracingPipeline, AdaptedAccel, tracescall
# Implemented by a scene to add non-triangle geometry to traversal.
export procedural_miss, procedural_candidate, procedural_commit, procedural_bary
export candidate_primitive_index, candidate_object_ray, commit_intersection!

export build_accel!, build_blas_aabb, refit_tlas!
export trace!, Trace
export primitivevertices
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList,
       PatchList, LineListAdjacency, LineStripAdjacency

export Mat3x4f, identity_transform, quat_to_rot3x3, build_4x3_pervec
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
# `copy!` is not exported: it would be ambiguous with Base's under `using Mantle`.
export Buffer, GPURef, Surface, Attribute, draw!, dispatch!, render!, runonce!
export DrawBinding, rebind!
export ResourceView, viewof
# Its own verb: Vulkan needs the index-buffer usage bit at allocation.
export indexbuffer
export repeat!, Predicate, supportspredicate
export kernelcompiles, resetkernelcompiles!
export Dispatch, DeviceRange, countresource, indirectcount!, passof, graphof, touch!
export newpass, handle, dispatches
export IdTable, resourceid, byid, checklive
export Phase, Dag, Schedule, Liveness, Place, Aliasing, Barriers, Pipelines
export Policy, Compact, Overlap, InOrder
export PHASES, compile!
# `update!` is not exported: Makie/ComputePipeline export the same name.
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

"""Resolve each element of grouped graph arguments, for recording and eager launch alike."""
storage(x::Tuple) = argmap(storage, x)
# Named operands have the same leaves as positional ones. Keep their keys,
# resolving handles before a backend adapts the aggregate into kernel arguments.
storage(x::NamedTuple{Names}) where {Names} = NamedTuple{Names}(argmap(storage, values(x)))

"""    argmap(f, args::Tuple) -> Tuple

`map` over a tuple, unrolled for any concrete length. `Base.map` allocates above
31 elements and an `ntuple` closure allocates for large shader argument tuples."""
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
"""An unplaced transient guarantees only the alignment floor common to all backends."""
basealignment(::TransientBuffer) = TRANSIENT_ALIGN_FLOOR
const TRANSIENT_ALIGN_FLOOR = 16

"""The power of two dividing byte offset or address `off`; zero constrains nothing.
Any integer type: device addresses have their top bits set and are no `Int`."""
powalign(off::Integer) = iszero(off) ? typemax(Int) : 1 << min(trailing_zeros(off), 62)

"""    capacity(dev) -> Int

Device-memory budget for transient placement. Default typemax(Int) means unknown;
backends report a bound when available so placement can diagnose excess demand."""
capacity(dev) = typemax(Int)

# Metal on Apple; Vulkan off Apple, and on Apple when a loader (e.g. lavapipe) is
# present; ROCm is a package extension. Vulkan is imported qualified because its
# resource names overlap Mantle's.
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

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
using KernelInterface: Topology, TriangleList, TriangleStrip, LineList,
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
# Before `queue.jl`: a `BatchQueue` holds the outstanding list this declares.
include("graph/submission.jl")
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
include("graphics/pipeline.jl")    # needs the state vocabulary above
include("graphics/commands.jl")
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
export Arena, reserve!, tenant!, untenant!, sharing, remap!, headroom, largestfree, remappable, takeover!
export DeviceArray, giveup!, blocksize
export upload!, download, deviceview, bufferusage, devicecopy!, Persistent
export rawalloc, rawfree, constraintof, compatible, maxalloc, mergeconstraints
export readproblem

export Backend, VulkanAPI, MetalAPI, WebGPUAPI, HostAPI
export Usage, ResourceKind, BufferKind, ImageKind, AccelKind
export Access, ReadOnly, WriteOnly, ReadWrite, NoAccess, Src, Dst
export Vertices, Indices, Indirect, Predicated, Uniform, Sampled, Present, Undefined
export CopySrc, CopyDst, TraceRead, TraceBuild, Storage, ColorAttachment, Depth, Unordered
export reads, writes, discards, kindof, aliasable, evictable, unordered, inner
export Transition, ResourceState, transition!, transitions, needs_transition
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
export HWTLAS, AccelBuildContext, BatchQueue, ExternalImage
export allocate_batch_queue!, release_batch_queue!, ensure_active_batch!, waitidle
export supports_graphics, supports_batch_queue, use_bindings!, supports_rt_pipeline
export batchqueue
export defaultbackend, availablebackends, register_backend!, register_kernel_recorder!
export devicearray
export bind_textures

# Rasterization commands. These were `begin_pass!` and friends.
export begin_pass!, end_pass!, draw_in_pass!, draw_indexed_in_pass!,
       draw_indirect_in_pass!, set_viewport!, reset_device!

# The graph's backend interface. `isdepth` is a definition, not a hook — see
# `graph/backend.jl` for why it and `aspect` swapped places.
export isdepth, target_extent, checkextents, refit!, record!, record_pass!,
       rename!, inplace!, nextslot!, collect!
export blit!, present_frame!, acquire_next_image!, transition_image!
export readback_framebuffer, readback_window, readback_target

# Pipeline descriptions and the indirect draw record.
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
# The shader builtins. Exported because a shader is written against them and
# nothing else; see `graphics/builtins.jl` for why they are overridden rather
# than defined.
export vertex_index, instance_index, frag_coord, frag_coord_x, frag_coord_y,
       frag_coord_z, frag_coord_w, frag_coord_xy, clip_y
export DrawIndirectCommand
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

export pixelbytes, vkformat
export LoadOp, Clear, Keep, Discard
export Device, Resource, Graph, Plan, Transient, Window, backend, screenshot
export DeviceCaps, caps
export MatrixShape, MatrixUse, MatrixA, MatrixB, Accumulator
export CoopMatrix, AcceleratedMatrix, WorkgroupMatrix, matrixuse, matrixscope
export MatrixScope, SubgroupScope, WorkgroupScope, supports, bestshape
# `copy!` is deliberately not exported: the name exists in Base, and exporting it
# would make the bare name ambiguous in any module that does `using Mantle`.
export Buffer, Scalar, Surface, Attribute, draw!, dispatch!, render!, compute!, Update
# `repeat!` and its vocabulary: a loop recorded once, whose trip count the device
# decides. `Predicate` is exported because a kernel writing predicates by hand
# names the type; `supportspredicate` because a caller may want to pick between
# `repeat!` and a host loop rather than be thrown at.
export repeat!, Predicate, supportspredicate
export Dispatch, DeviceRange, countresource, indirectcount!, passof, graphof, touch!,
       argvalue
export UpdateRef, anypending, applyupdates!, custombody, registerupdate!
export newpass, handle, dispatches
export IdTable, resourceid, byid, checklive
export Phase, Dag, Schedule, Liveness, Place, Aliasing, Barriers, Pipelines
export Policy, Compact, Overlap
export PHASES, compile!
# `update!` is deliberately not exported: ComputePipeline exports it, and that is
# the one a Makie-shaped caller means — `update!(plot; positions = …)` sets an
# attribute. Mantle's writes a buffer now, which is a different verb with the same
# spelling, so it stays `Mantle.update!` and the bare name belongs to Makie's.
export run!, npipelines, capacity, use, peakbytes, naivebytes, storage, custom!, free!
export bake!, baked, rebind!, rebindable
export timings, PassTiming, NSAMPLES

"""
    bake!(plan) -> plan

Record the plan once and re-submit that recording on every later `run!`.

The prize is host time. A plan whose launch sequence is identical every
invocation pays to rebuild it every invocation, and on SAM 2's encoder that is
16.3 ms of recording against ~12 ms of GPU work — recording the step costs more
than running it.

The precondition is that every device address the recording names is the same
next time. A plan already gives that: placement is fixed, the arena is the
device's, and a plan holds references to everything it names — which is the
property raw capture lacks and the reason this lives here rather than on the
capture. What a plan does *not* fix is an input written by reallocating, so an
`Update` that renames is recorded fresh per invocation and submitted ahead of the
replay rather than baked into it.

Opt-in, and it stays opt-in: `run!` on an unbaked plan records as it always did.
That is what lets the same plan be measured both ways in one process, which
matters because a placement bug and a stale-recording bug both present as a
number that moved.

    plan = Plan(g)
    run!(plan)          # records
    bake!(plan)
    run!(plan)          # replays

**This RUNS the plan, once.** The recording is taken by recording, and the
commands reach the queue on the way past — so `bake!` is `run!` plus a capture,
not a capture instead of a run. For a plan that computes a value from its inputs
that is invisible; for one that ACCUMULATES it is a whole extra contribution, and
it lands on whichever invocation happened to build the plan. That shows up as a
first frame that is wrong and every later frame exact, which reads like a bug in
the work rather than in when it was baked. Bake such a plan where an extra run
does not count, or leave it unbaked — a plan of one dispatch has nothing to gain
here anyway.

An argument that moves needs [`rebind!`](@ref): a baked plan does not record, so
the values it packed at `bake!` are the ones it replays until told otherwise.

Not yet for plans with a surface: a swapchain image is a different image every
frame and the recording names one. Headless plans have no such thing.

A backend that does not record per run has nothing to capture, and gets the
default: baking is the plan, unchanged. That is a no-op rather than an error
because `bake!` asks for an outcome — "stop paying to rebuild this" — that such a
backend already has, and making the caller ask which backend it is on is the
branch this library exists to remove.
"""
bake!(plan) = plan

"""Whether `bake!` has been called on this plan and the recording is in use.
False by default, which is the honest answer for a backend that never records
one."""
baked(plan) = false

"""
    rebind!(plan) -> plan

Re-read the arguments a baked plan's work was given, so a value that moved is
replayed as it is now rather than as it was when the recording was captured.

A backend that records nothing per run has to offer this or `bake!` is only safe
for plans whose every argument is constant — and unsafe SILENTLY, since a stale
value produces a plausible result rather than an error. A backend that resolves
arguments at launch (the host one does) needs nothing, and gets the default.

The ordering is the caller's: see the Lava method for what a baked plan's single
argument slot means for a rebind that overlaps a replay.
"""
rebind!(plan) = plan

"""
    rebindable(plan) -> Bool

Whether [`rebind!`](@ref) can deliver its contract for this plan.

It cannot when the plan has a `custom!` pass. Such a pass declares what it
touches but not how, and its body packs its own arguments while it runs — so a
baked plan, which never runs the body again, replays whatever the body packed at
capture, and nothing outside the body knows where those bytes are to rewrite
them. `rebind!` would return having quietly done nothing for that pass.

Ask this before baking anything whose arguments move. The answer is a property of
the graph, not of the values, so it is stable for the life of the plan.

True by default: a backend that resolves arguments at launch rebinds by
construction.
"""
rebindable(plan) = true

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

"""
    custom!(f, graph, name) -> pass

A pass that declares what it touches but not how. `f` receives the pass handle,
calls `use` for every resource the work reads or writes, and returns a zero-arg
callable; that callable runs at record time with the batch open, and may launch
whatever it likes.

`dispatch!` needs one kernel, its arguments and an ndrange. Work that is a *unit*
to the caller but several launches underneath — a fused attention block, an ATen
operator with a host-side branch, a library call — has no way to say so, and
wrapping each launch as its own pass would declare a resource sequence the caller
does not actually have. This is the escape hatch: the graph still derives
lifetimes and barriers from the declaration, and stays out of the body.

The declaration is a promise. Nothing checks that the body touches only what was
declared, and memory it reaches without saying so is memory the placer is free to
alias with something else.
"""
function custom!(f, g::Graph, name::AbstractString)
    p = newpass(g, name, :custom)
    push!(passes(g), p)
    push!(dispatches(p), custombody(f(handle(g, p))))
    return p
end

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

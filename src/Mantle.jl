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
# `import`, not `using … :` — these get `Mantle.Device` methods below, and
# `caps` in particular becomes one function with a `Device` method here and a
# `KI.Backend` method in each backend.
import KernelInterface: supports, bestshape, caps, matrix_shapes, wggranularity

# What the Vulkan backend below needs. These arrived with the runtime that moved
# here from Lava, and they are hard dependencies rather than an extension's
# because the backend is part of this package now — the same reason `caps` and
# `DeviceCaps` stopped being two things.
import Serialization
import PrecompileTools
# BOTH, and both are needed. `using` binds the ~60 names the moved runtime calls
# unqualified — `unwrap`, `cmd_dispatch`, `DeviceMemory`, the `FORMAT_*` and
# `ERROR_*` constants — exactly as it did when that code lived in Lava. `as VK`
# gives the qualified spelling the same code uses for the rest.
#
# This is only possible because the backend markers are `VulkanAPI`/`MetalAPI`
# and not `Vulkan`/`Metal`: a marker named after its package shadows it here.
using Vulkan
import Vulkan as VK
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
import GLFW

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
include("runtime/format.jl")
include("runtime/api.jl")
include("runtime/dispatch.jl")
include("phases.jl")
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

# ── the Vulkan backend ────────────────────────────────────────────────────────
# Lava's runtime, moved here 2026-08-27. See `vulkan/vulkan.jl` for what had to
# be untangled first and why these are included into `Mantle` rather than a
# submodule. Last, because every method in it is a method on something above.
include("vulkan/vulkan.jl")

export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is deliberately not exported: it is a Span predicate nothing outside
# this package calls, and GeometryBasics exports the same name.
export segments, maxload, hmax, fragmentation
export place, LowestFit, BestFit
export Pool, Block, Region, acquire!, release!, trim!, reserved, retire!, reclaim!, fence, passed, waitfor
export Arena, reserve!, tenant!, untenant!, sharing, remap!, headroom, largestfree, remappable, takeover!
export DeviceArray, giveup!, blocksize
export upload!, download, deviceview, bufferusage, devicecopy!, Persistent
export rawalloc, rawfree, constraintof, compatible, maxalloc, mergeconstraints
export readproblem

export Backend, VulkanAPI, MetalAPI, WebGPUAPI, HostAPI
export Usage, ResourceKind, BufferKind, ImageKind, AccelKind
export Access, ReadOnly, WriteOnly, ReadWrite, NoAccess, Src, Dst
export Vertices, Indices, Indirect, Uniform, Sampled, Present, Undefined
export CopySrc, CopyDst, TraceRead, TraceBuild, Storage, ColorAttachment, Depth, Unordered
export reads, writes, discards, kindof, aliasable, evictable, unordered, inner
export Transition, ResourceState, transition!, transitions, needs_transition
export stages, access, layout

export pixelbytes, vkformat
export LoadOp, Clear, Keep, Discard
export Device, Resource, Graph, Plan, Transient, Window, backend, screenshot
export DeviceCaps, caps
export MatrixShape, MatrixUse, MatrixA, MatrixB, Accumulator
export MatrixScope, SubgroupScope, WorkgroupScope, supports, bestshape
# `copy!` is deliberately not exported: the name exists in Base, and exporting it
# would make the bare name ambiguous in any module that does `using Mantle`.
export Buffer, Scalar, Surface, Attribute, draw!, dispatch!, render!, compute!, Update
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

"""
Everything that needs a live device, at load.

Lava had this and kept the compiler half of it; these two are the device half and
came here with `runtime/device.jl` on 2026-08-27. Leaving them behind was not
theoretical: without the `atexit` hook, a `LavaArray` finalizer running during
Julia's shutdown sweep calls `query_timeline` on a semaphore whose device is
already gone, and the process takes a SIGSEGV inside the driver — reproduced
immediately after the move, which is how the omission was found.
"""
function __init__()
    # The pipeline builder thread. Pipelines are created off the main thread so a
    # first launch does not block on the driver's compiler.
    init_pipeline_thread!()

    # Mark the device lost during shutdown so GC finalizers do not call into the
    # Vulkan driver after it has been torn down. `atexit` runs BEFORE Julia's
    # global finalizer sweep, which is the whole point — a finalizer that reaches
    # a dead device segfaults inside the driver, where no Julia `try` can catch
    # it and no stack trace names the buffer that did it.
    atexit() do
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || mark_device_lost!(ctx)
        bind_context!(nothing)
    end
end

end

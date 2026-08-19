module Mantle

# The matrix vocabulary is shared with Lava, which cannot be a dependency here —
# Mantle weak-depends on Lava for `MantleLavaExt`, so an edge back would close a
# cycle. `supports`/`bestshape` are imported by name because `DeviceCaps` methods
# below extend them rather than defining a second pair.
using KernelInterfaces
import KernelInterfaces: supports, bestshape

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

export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is deliberately not exported: it is a Span predicate nothing outside
# this package calls, and GeometryBasics exports the same name.
export segments, maxload, hmax, fragmentation
export place, LowestFit, BestFit
export Pool, Block, Region, acquire!, release!, trim!, reserved
export Arena, reserve!, tenant!, untenant!, sharing, remap!, headroom, largestfree, remappable, takeover!
export DeviceArray, giveup!, blocksize
export upload!, download, deviceview, bufferusage, devicecopy!, Persistent
export rawalloc, rawfree, constraintof, compatible, maxalloc, mergeconstraints
export readproblem

export Backend, Vulkan, Metal, WebGPU, Host
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
export run!, npipelines, capacity, use, peakbytes, storage, custom!, free!
export bake!, baked, rebind!
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

function update! end
function use end
function peakbytes end
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

end

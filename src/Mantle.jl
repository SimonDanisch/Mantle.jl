module Mantle

include("memory/interval.jl")
include("memory/model.jl")
include("memory/bound.jl")
include("memory/placement.jl")
include("memory/csv.jl")

include("sync/usage.jl")      # ResourceKind, which backend.jl dispatches on
include("sync/backend.jl")
include("sync/transition.jl")
include("runtime/format.jl")
include("runtime/api.jl")
include("phases.jl")

export Span, OffsetWindow, Gap, Item, Problem, Placement
# `overlaps` is deliberately not exported: it is a Span predicate nothing outside
# this package calls, and GeometryBasics exports the same name.
export segments, maxload, hmax, fragmentation
export place, LowestFit, BestFit
export readproblem

export Backend, Vulkan, Metal, WebGPU
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
# `copy!` is deliberately not exported: the name exists in Base, and exporting it
# would make the bare name ambiguous in any module that does `using Mantle`.
export Buffer, Scalar, Surface, Attribute, draw!, dispatch!, render!, compute!, Update
export Phase, Dag, Schedule, Liveness, Place, Aliasing, Barriers, Pipelines
export Policy, Compact, Overlap
export PHASES, compile!
# `update!` is deliberately not exported: ComputePipeline exports it, and that is
# the one a Makie-shaped caller means — `update!(plot; positions = …)` sets an
# attribute. Mantle's writes a buffer now, which is a different verb with the same
# spelling, so it stays `Mantle.update!` and the bare name belongs to Makie's.
export run!, npipelines, capacity, use, peakbytes, storage, custom!
export timings, PassTiming, NSAMPLES

function update! end
function use end
function peakbytes end
function storage end
function capacity end

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
function custom! end

end

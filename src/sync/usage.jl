"""
The usage vocabulary.

Everything here is portable. Stages, access masks and image layouts are Vulkan
concepts that Metal and WebGPU do not have, so they live in the backend
extension and are reached by dispatch. The core knows only what a usage *is*,
which of read and write it does, and what kind of resource it applies to.
"""
abstract type Usage end

abstract type ResourceKind end
struct BufferKind <: ResourceKind end
struct ImageKind <: ResourceKind end
struct AccelKind <: ResourceKind end

"""
    Access{RD,WR}

Direction, in type position. `Storage(p, x)` with neither flag declares neither a
read nor a write, which is meaningless, so both are mandatory at the call site.
"""
struct Access{RD,WR} end

const ReadOnly = Access{true,false}
const WriteOnly = Access{false,true}
const ReadWrite = Access{true,true}

"""
Neither read nor written. Not a declaration a pass can make — `Storage(p, x)` with
both flags off is rejected — but the honest answer for the stencil half of a
depth-only image, which has no stencil aspect to access at all.
"""
const NoAccess = Access{false,false}

reads(::Type{Access{RD,WR}}) where {RD,WR} = RD
writes(::Type{Access{RD,WR}}) where {RD,WR} = WR

"""Which side of a barrier an access sits on.

The same usage lowers differently as source and destination: depth writes
complete at late fragment tests but reads begin at early fragment tests, and
`TOP_OF_PIPE` is only meaningful as a destination. RPS carries this as a template
parameter (`rps_vk_runtime_backend.cpp:119`); here it is a dispatch argument, so a
stage that is illegal as a source is one no `::Src` method returns.
"""
struct Src end
struct Dst end

# ── usages with a single legal direction ──────────────────────────────────────
struct Vertices <: Usage end
struct Indices <: Usage end
struct Indirect <: Usage end
# `repeat!`'s per-iteration flags, read by the command processor when it decides
# whether to run a predicated pass. Distinct from `Indirect` because the stage
# and access it lowers to are the conditional-rendering ones, not the draw ones.
struct Predicated <: Usage end
struct Uniform <: Usage end
struct Sampled <: Usage end
struct Present <: Usage end
struct CopySrc <: Usage end
struct CopyDst <: Usage end
struct TraceRead <: Usage end
struct TraceBuild <: Usage end

"""
No contents. What a transient is in before anything writes it, and what one
becomes when another takes over its bytes.

It is a state and never a declaration: `use(p, x; ...)` cannot produce it. Its
value is that "undefined" then travels through the same tracking as every other
state instead of being a special case at the point a barrier is emitted, so a
first write and a write after aliasing lower identically.
"""
struct Undefined <: Usage end

# ── usages that carry a direction ─────────────────────────────────────────────
"""Shader-addressable storage. One name for buffers and images: the resource kind
is a parameter, so `Storage{ImageKind}` and `Storage{BufferKind}` are one concept."""
struct Storage{K<:ResourceKind,A<:Access} <: Usage end

"""
A colour attachment. `Discard` is true when the load op throws the previous
contents away (clear or don't-care), which is what lets the lowering drop the
read access bit. RPS derives the same thing from `DISCARD_DATA_BEFORE`
(`rps_vk_runtime_backend.cpp:143`); our §3 instead scanned the recorded draws for
blending, which is a later and more fragile way to learn it.
"""
struct ColorAttachment{Discard} <: Usage end

"""
A depth-stencil attachment.

Depth and stencil have independent directions, which is why four distinct layouts
exist for them; a single write flag reaches only two of the four. `NoAccess` for
the stencil half says the image has no stencil aspect, which is a fifth case and
not the same as read-only stencil.

`Discard` is the load op throwing the previous contents away, exactly as for
[`ColorAttachment`](@ref): it is what lets the barrier into the attachment come
from UNDEFINED, which a transient depth buffer sharing bytes with something else
needs, and what drops the read bit for a pass that cleared before testing.
"""
struct Depth{DA<:Access,SA<:Access,Discard} <: Usage end

# ── portable properties ───────────────────────────────────────────────────────
kindof(::Type{Vertices}) = BufferKind()
kindof(::Type{Indices}) = BufferKind()
kindof(::Type{Indirect}) = BufferKind()
kindof(::Type{Predicated}) = BufferKind()
kindof(::Type{Uniform}) = BufferKind()
kindof(::Type{Sampled}) = ImageKind()
kindof(::Type{Present}) = ImageKind()
# Copy applies to buffers and to images alike, so it constrains nothing. Resource
# kind is a property of the resource; only usages that genuinely admit one kind
# declare it, and those are what make `Vertices(p, img)` a MethodError.
kindof(::Type{CopySrc}) = nothing
kindof(::Type{CopyDst}) = nothing
kindof(::Type{Undefined}) = nothing
kindof(::Type{TraceRead}) = AccelKind()
kindof(::Type{TraceBuild}) = AccelKind()
kindof(::Type{<:Storage{K}}) where {K} = K()
kindof(::Type{<:ColorAttachment}) = ImageKind()
kindof(::Type{<:Depth}) = ImageKind()

reads(::Type{<:Usage}) = true
reads(::Type{Undefined}) = false
reads(::Type{CopyDst}) = false
reads(::Type{TraceBuild}) = false
reads(::Type{<:Storage{K,A}}) where {K,A} = reads(A)
reads(::Type{ColorAttachment{Discard}}) where {Discard} = !Discard
reads(::Type{Depth{DA,SA,D}}) where {DA,SA,D} = !D && (reads(DA) || reads(SA))

writes(::Type{<:Usage}) = false
writes(::Type{CopyDst}) = true
writes(::Type{TraceBuild}) = true
writes(::Type{<:Storage{K,A}}) where {K,A} = writes(A)
writes(::Type{<:ColorAttachment}) = true
writes(::Type{Depth{DA,SA,D}}) where {DA,SA,D} = writes(DA) || writes(SA)

"""
Does this usage throw away whatever was there before it?

A discarding destination makes the old layout irrelevant, so the barrier into it
can come from UNDEFINED whatever the resource was previously in. RPS derives the
same from `DISCARD_DATA_BEFORE` (`rps_vk_runtime_backend.cpp:143`).
"""
discards(::Type{<:Usage}) = false
discards(::Type{ColorAttachment{D}}) where {D} = D
discards(::Type{Depth{DA,SA,D}}) where {DA,SA,D} = D

"""May this usage's resource share memory with a disjoint one? A video decode
reference picture may not, and neither may anything a running plan reserved."""
aliasable(::Type{<:Usage}) = true
evictable(::Type{<:Usage}) = true

"""
Two accesses that both declare themselves unordered may overlap even when they
touch the same resource, because the caller asserts the order does not change the
result: disjoint elements, or commutative atomics.

Checked on both sides, so forgetting it anywhere gives the barrier back. RPS ANDs
the same bit across the pair (`rps_access_dag_build.hpp:499`).
"""
struct Unordered{U<:Usage} <: Usage end

unordered(::Type{<:Usage}) = false
unordered(::Type{<:Unordered}) = true
inner(::Type{Unordered{U}}) where {U} = U
inner(::Type{U}) where {U<:Usage} = U

"""
A shader access made by a ray-tracing PIPELINE rather than by a compute, vertex
or fragment shader: the same read or write, at a different pipeline stage.

The stage is what a barrier is made of, and a pass that traces does its storage
reads and writes from raygen, hit and miss shaders. Lowered as an ordinary
`Storage` access, its barriers named the compute/vertex/fragment stages, and on
a device where ray tracing is its own stage — NVIDIA; AMD runs it as compute —
that is no dependency at all in either direction: the trace could start before
the pass that filled its queue had finished, and the pass after it could read
the queue it was still writing. Bit-identical images almost always, and one
`VK_ERROR_DEVICE_LOST` on a medium-heavy scene when it was not.

Applied by `compute!` to every shader usage of a pass that holds a [`Trace`](@ref)
(see [`traced`](@ref)), so no call site names it — a `use(p, x; read = true)`
inside a trace pass IS a traced read. Outermost is `Unordered`, when both apply.
"""
struct Traced{U<:Usage} <: Usage end

for f in (:kindof, :reads, :writes, :discards, :aliasable, :evictable)
    @eval $f(::Type{Unordered{U}}) where {U} = $f(U)
    @eval $f(::Type{Traced{U}}) where {U} = $f(U)
end

"""
    traced(U) -> Type

`U` as a ray-tracing pipeline makes it. Shader accesses — storage, uniform,
sampled — become [`Traced`](@ref); what the command processor reads (indirect
counts, predicates), what a copy touches and the acceleration structure itself
are what they were, because a trace pass does not access those from a shader.
"""
traced(::Type{Unordered{U}}) where {U} = Unordered{traced(U)}
traced(::Type{S}) where {S<:Storage} = Traced{S}
traced(::Type{Uniform}) = Traced{Uniform}
traced(::Type{Sampled}) = Traced{Sampled}
traced(::Type{U}) where {U<:Usage} = U

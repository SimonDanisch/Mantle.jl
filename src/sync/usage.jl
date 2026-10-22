"""
The portable usage vocabulary: what a usage is, whether it reads or writes, and
which resource kind it applies to. Stages, access masks and layouts are backend
lowerings reached by dispatch.
"""
abstract type Usage end

abstract type ResourceKind end
struct BufferKind <: ResourceKind end
struct ImageKind <: ResourceKind end
struct AccelKind <: ResourceKind end

"""
    Access{RD,WR}

Read and write direction, in type position.
"""
struct Access{RD,WR} end

const ReadOnly = Access{true,false}
const WriteOnly = Access{false,true}
const ReadWrite = Access{true,true}

"""Neither read nor written: the stencil half of a depth-only image, which has no
stencil aspect."""
const NoAccess = Access{false,false}

reads(::Type{Access{RD,WR}}) where {RD,WR} = RD
writes(::Type{Access{RD,WR}}) where {RD,WR} = WR

"""Which side of a barrier an access sits on. A usage lowers differently as source
and destination (depth writes complete at late fragment tests, reads begin at
early ones), so a stage illegal as a source is one no `::Src` method returns."""
struct Src end
struct Dst end

# Usages with a single legal direction.
struct Vertices <: Usage end
struct Indices <: Usage end
struct Indirect <: Usage end
# `repeat!`'s per-iteration flags, read to decide whether a predicated pass runs.
# Not `Indirect`: it lowers to the conditional-rendering stage and access.
struct Predicated <: Usage end
struct Uniform <: Usage end
struct Sampled <: Usage end
struct Present <: Usage end
struct CopySrc <: Usage end
struct CopyDst <: Usage end
struct TraceRead <: Usage end
struct TraceBuild <: Usage end

"""
No contents: a transient's state before its first write and after another
transient takes over its bytes. Never inferred from a kernel. Tracked like any
state, so a first write and a write after aliasing lower identically.
"""
struct Undefined <: Usage end

# Usages that carry a direction.
"""Shader-addressable storage. One name for buffers and images: the resource kind
is a parameter, so `Storage{ImageKind}` and `Storage{BufferKind}` are one concept."""
struct Storage{K<:ResourceKind,A<:Access} <: Usage end

"""
A colour attachment. `Discard` is true when the load op drops the previous
contents (clear or don't-care), which removes the read access.
"""
struct ColorAttachment{Discard} <: Usage end

"""
A depth-stencil attachment. Depth and stencil have independent directions (four
layouts); `NoAccess` for the stencil half means the image has no stencil aspect,
which differs from read-only stencil. `Discard`, as for [`ColorAttachment`](@ref),
lets the barrier come from UNDEFINED and removes the read access.
"""
struct Depth{DA<:Access,SA<:Access,Discard} <: Usage end

kindof(::Type{Vertices}) = BufferKind()
kindof(::Type{Indices}) = BufferKind()
kindof(::Type{Indirect}) = BufferKind()
kindof(::Type{Predicated}) = BufferKind()
kindof(::Type{Uniform}) = BufferKind()
kindof(::Type{Sampled}) = ImageKind()
kindof(::Type{Present}) = ImageKind()
# Copy applies to buffers and images alike, so it constrains no kind; only usages
# that admit a single kind declare one.
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
Whether this usage discards the previous contents, so the barrier into it can come
from UNDEFINED whatever state the resource is in.
"""
discards(::Type{<:Usage}) = false
discards(::Type{ColorAttachment{D}}) where {D} = D
discards(::Type{Depth{DA,SA,D}}) where {DA,SA,D} = D

"""Whether a resource in this usage may share memory with a disjoint one
(`aliasable`) or be evicted (`evictable`). `true` for every usage defined here."""
aliasable(::Type{<:Usage}) = true
evictable(::Type{<:Usage}) = true

"""
An access the caller asserts is order-independent (disjoint elements, commutative
atomics). Two accesses skip the barrier only when both are `Unordered`.
"""
struct Unordered{U<:Usage} <: Usage end

unordered(::Type{<:Usage}) = false
unordered(::Type{<:Unordered}) = true
inner(::Type{Unordered{U}}) where {U} = U
inner(::Type{U}) where {U<:Usage} = U

"""
A shader access made by a ray-tracing pipeline: the same read or write at the
ray-tracing stage. Lowered as plain `Storage`, its barriers would name the
compute/vertex/fragment stages, which on NVIDIA (ray tracing is its own stage; AMD
runs it as compute) orders nothing and occasionally loses the device.
[`trace!`](@ref) applies it to every shader usage of its pass ([`traced`](@ref));
`Unordered` is outermost when both apply.
"""
struct Traced{U<:Usage} <: Usage end

for f in (:kindof, :reads, :writes, :discards, :aliasable, :evictable)
    @eval $f(::Type{Unordered{U}}) where {U} = $f(U)
    @eval $f(::Type{Traced{U}}) where {U} = $f(U)
end

"""
    traced(U) -> Type

`U` as a ray-tracing pipeline makes it: storage, uniform and sampled accesses
become [`Traced`](@ref). Indirect counts, predicates, copies and the acceleration
structure are not shader accesses and stay as they are.
"""
traced(::Type{Unordered{U}}) where {U} = Unordered{traced(U)}
traced(::Type{S}) where {S<:Storage} = Traced{S}
traced(::Type{Uniform}) = Traced{Uniform}
traced(::Type{Sampled}) = Traced{Sampled}
traced(::Type{U}) where {U<:Usage} = U

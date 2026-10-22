# GPUArrays interface for `LavaArray`: Adapt conversion, broadcast, transfers,
# `resize!`, `norm` and `permutedims!`.

import GPUArrays
import Adapt

"""
    adapt_storage(::LavaAdaptor, a::LavaArray) -> LavaDeviceArray

The `(address, dims)` value of `a`. It retains nothing: callers `holdleaves!`
the original arguments before adapting.
"""
function Adapt.adapt_storage(::LavaAdaptor, a::LavaArray{T,N}) where {T,N}
    return LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end
Adapt.adapt_storage(::LavaAdaptor, x) = x  # Scalars pass through

# Type-based adapt: Array → LavaArray (used by GPUArrays test suite's compare())
Adapt.adapt_storage(::Type{<:LavaArray}, xs::AT) where {AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray, xs)
Adapt.adapt_storage(::Type{<:LavaArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray{T}, xs)
Adapt.adapt_storage(::Type{Array}, xs::LavaArray) = convert(Array, xs)

"""Isbits stand-in for a `Base.RefValue` kernel argument."""
struct LavaRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::LavaRefValue) = r.x

Adapt.adapt_structure(to::LavaAdaptor, r::Base.RefValue) = LavaRefValue(Adapt.adapt(to, r[]))

# A type is not a GPU value: carry it as a type parameter.
struct LavaRefType{T} <: Ref{DataType} end
Base.getindex(::LavaRefType{T}) where {T} = T
Adapt.adapt_structure(::LavaAdaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    LavaRefType{r[]}()

# `T.(args...)`: replace the type with a closure that calls it.
Adapt.adapt_structure(to::LavaAdaptor,
                      bc::Base.Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Base.Broadcast.Broadcasted{Style}((x...) -> T(x...), Adapt.adapt(to, bc.args), bc.axes)

struct LavaArrayStyle{N} <: GPUArrays.AbstractGPUArrayStyle{N} end

LavaArrayStyle{M}(::Val{N}) where {M,N} = LavaArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{<:LavaArray{T,N}}) where {T,N} = LavaArrayStyle{N}()

# Structured matrices (Diagonal, Tridiagonal, ...) over LavaArrays broadcast on the GPU.
Base.Broadcast.BroadcastStyle(::LavaArrayStyle{N}, ::LinearAlgebra.StructuredMatrixStyle{T}) where {N,T} = LavaArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{T}, ::LavaArrayStyle{N}) where {N,T} = LavaArrayStyle{N}()

# View wrappers over a LavaArray take `LavaArrayStyle` at any nesting depth;
# `DefaultArrayStyle` iterates and hits the scalar-indexing guard. Graph
# conversion produces deep stacks, such as a five-deep `PermutedDimsArray`.

"""
    lavabacked(T) -> Bool

Whether an array type bottoms out in a `LavaArray` through any stack of view
wrappers. Recursive, and resolved at compile time for a concrete type.
"""
lavabacked(::Type{<:LavaArray}) = true
lavabacked(::Type{<:PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:SubArray{<:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:Base.ReshapedArray{<:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:LinearAlgebra.Transpose{<:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:LinearAlgebra.Adjoint{<:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type) = false

# Non-Lava wrappers keep Base's `DefaultArrayStyle`.
lavastyle(::Type{W}, ::Val{N}) where {W, N} =
    lavabacked(W) ? LavaArrayStyle{N}() : Base.Broadcast.DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: PermutedDimsArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: SubArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: Base.ReshapedArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, W <: LinearAlgebra.Transpose{T}} =
    lavastyle(W, Val(2))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, W <: LinearAlgebra.Adjoint{T}} =
    lavastyle(W, Val(2))

"""
    lavaroot(x) -> LavaArray

The device array underneath any stack of view wrappers.
"""
lavaroot(x::LavaArray) = x
lavaroot(x::Union{PermutedDimsArray, SubArray, Base.ReshapedArray,
                  LinearAlgebra.Transpose, LinearAlgebra.Adjoint}) = lavaroot(parent(x))

"""
    dense(x) -> LavaArray

Materialise a wrapped device array as a dense `LavaArray` by broadcast.
`Array(x)` and `copyto!` iterate a nested wrapper and hit the scalar-indexing
guard, so download with `Array(dense(x))`. Other arrays pass through unchanged.
"""
dense(x::LavaArray) = x
function dense(x::AbstractArray)
    lavabacked(typeof(x)) || return x
    y = similar(lavaroot(x), eltype(x), size(x))
    y .= x                      # not copyto!, which iterates
    return y
end

const AnyLavaArray{T} = Union{LavaArray{T},
                              SubArray{T, <:Any, <:LavaArray},
                              PermutedDimsArray{T, <:Any, <:Any, <:Any, <:LavaArray},
                              Base.ReshapedArray{T, <:Any, <:LavaArray},
                              LinearAlgebra.Transpose{T, <:LavaArray},
                              LinearAlgebra.Adjoint{T, <:LavaArray}}

# Broadcasts launch over a flat index space: an N-D `ndrange` gets N-D
# workgroups whose consecutive lanes do not touch consecutive memory, about 6x
# slower for `D .= A .+ B` into a 3-D destination. Each thread handles `U`
# elements `WG` apart, so every iteration reads one contiguous run per workgroup
# and keeps more loads in flight; a larger workgroup does not help.

# How thread index `I` addresses the destination and the source: linearly, or as
# the Cartesian index within the extents `sz` (one decomposition when both use it).
struct LinearPair end
struct MixedPair{S}
    sz::S
end
struct CartesianPair{S}
    sz::S
end
@inline indexpair(::LinearPair, I) = (I, I)
@inline indexpair(p::MixedPair, I) = (I, CartesianIndex(cart32(UInt32(I) - UInt32(1), p.sz)))
@inline function indexpair(p::CartesianPair, I)
    J = CartesianIndex(cart32(UInt32(I) - UInt32(1), p.sz))
    return (J, J)
end

function lava_broadcast_flat!(dest, bc, pair, n, ::Val{U}, ::Val{WG}) where {U, WG}
    l = KI.get_local_id().x
    g = KI.get_group_id().x
    base = (g - 1) * (WG * U) + l
    @inbounds for u in 0:(U - 1)
        I = base + u * WG
        if I <= n
            d, s = indexpair(pair, I)
            dest[d] = bc[s]
        end
    end
    return nothing
end

"""
    broadcastlaunch(n; wg, maxunroll, mingroups) -> (workgroupsize, Val(U), ndrange)

Launch geometry for a flat broadcast over `n` elements. `U` starts at
`maxunroll` elements per thread and halves while the grid would have fewer than
`mingroups` workgroups. `ndrange` is rounded up to whole workgroups; the kernels
bound every access by `n`. Each distinct `U` compiles its own kernel.
"""
@inline function broadcastlaunch(n::Int; wg::Int = 256, maxunroll::Int = 8,
                                 mingroups::Int = 192)
    u = maxunroll
    while u > 1 && cld(n, wg * u) < mingroups
        u >>= 1
    end
    return wg, Val(u), wg * cld(n, wg * u)
end

# `dest .= PermutedDimsArray(a, perm)` on the mixed kernel is bound by access
# order, not index arithmetic: decomposing only the row index when
# `perm[1] == 1` gained nothing. A faster permute must change the access
# pattern, such as a tiled transpose.

"""
    flatok(dest, x) -> Bool

Whether every operand can be read with `dest`'s linear index: same size and
`IndexLinear`. `IndexStyle(::Broadcasted)` is too conservative for this case.
"""
flatok(dest, x) = true                                  # scalars, refs, functions
flatok(dest, x::AbstractArray) =
    size(x) == size(dest) && IndexStyle(x) === IndexLinear()
# A tuple is a broadcast container with its own axes; `flat1` keeps its length,
# so flattening would throw `DimensionMismatch`.
flatok(dest, x::Tuple) = false
flatok(dest, x::Broadcast.Extruded) = flatok(dest, x.x)
flatok(dest, x::Broadcast.Broadcasted) = all(a -> flatok(dest, a), x.args)

"""
    flat1(x)

Rebuild a broadcast tree over 1-D reshapes of its leaves, so integer indexing
does not go through `CartesianIndices`. Valid only where `flatok` holds.
"""
flat1(x) = x
flat1(x::AbstractArray) = reshape(x, length(x))
flat1(bc::Broadcast.Broadcasted) = Broadcast.broadcasted(bc.f, map(flat1, bc.args)...)

leafsizes(x) = Any[]
leafsizes(x::AbstractArray) = Any[(size(x), nameof(typeof(x)), IndexStyle(x) === IndexLinear())]
leafsizes(x::PermutedDimsArray{T,N,perm}) where {T,N,perm} = Any[(size(parent(x)), :Permuted, perm)]
leafsizes(x::Broadcast.Extruded) = leafsizes(x.x)
leafsizes(bc::Broadcast.Broadcasted) = reduce(vcat, map(leafsizes, bc.args); init = Any[])

"""
    probe_broadcast!(path, dest, bc)

When `ctx.diag.broadcast_probe` is a `Dict` (default `nothing`), count the
broadcast under `(path, dest size, leaf sizes)`. The paths differ by about 3x
in bandwidth and are chosen by operand shapes.
"""
function probe_broadcast!(path, dest, bc)
    p = vk_context(dest).diag.broadcast_probe
    p === nothing && return
    key = (path, size(dest), Tuple(unique(leafsizes(bc))))
    p[key] = get(p, key, 0) + 1
    return
end

# `fastdiv` is a keyword so tests can compare both index paths.
function GPUArrays._copyto!(dest::AnyLavaArray, bc::Broadcast.Broadcasted;
                            fastdiv::Bool = BROADCAST_FASTDIV_DEFAULT)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    n = length(dest)
    # The array's backend, not `LavaBackend()`: an unpinned backend dispatches
    # on the global `vk_context()`, which may be another GPU.
    backend = KernelAbstractions.get_backend(dest)
    wg, u, nd = broadcastlaunch(n)
    if ndims(dest) > 1 && IndexStyle(dest) === IndexLinear() && flatok(dest, bc)
        probe_broadcast!(:flat, dest, bc)
        flat = Broadcast.instantiate(flat1(bc))
        d1 = reshape(dest, n)
        launchbroadcast!(backend, d1, Broadcast.preprocess(d1, flat), LinearPair(), n, u, wg, nd)
        return dest
    end
    bc = Broadcast.preprocess(dest, bc)
    # One launch per branch, so each sees a concrete pair. Extents are magic
    # numbers built on the host, so `cart32` multiplies instead of dividing.
    if ndims(dest) == 1
        probe_broadcast!(:linear1d, dest, bc)
        launchbroadcast!(backend, dest, bc, LinearPair(), n, u, wg, nd)
    elseif IndexStyle(dest) === IndexLinear()
        probe_broadcast!(:mixed, dest, bc)
        launchbroadcast!(backend, dest, bc, MixedPair(broadcastextents(size(dest); fastdiv)), n, u, wg, nd)
    else
        probe_broadcast!(:cartesian, dest, bc)
        launchbroadcast!(backend, dest, bc, CartesianPair(broadcastextents(size(dest); fastdiv)), n, u, wg, nd)
    end
    return dest
end

@inline launchbroadcast!(backend, dest, bc, pair, n, u, wg, nd) =
    KI.Kernel(backend, lava_broadcast_flat!)(dest, bc, pair, n, u, Val(wg); ndrange = nd, workgroupsize = wg)

# GPUArrays' `repeat` covers `AnyGPUArray`, one wrapper layer deep; two layers
# fall back to Base's scalar-indexing `repeat`. Densify first.
const NestedLavaWrapper = Union{
    Base.ReshapedArray{<:Any, <:Any, <:Union{SubArray{<:Any, <:Any, <:LavaArray},
                                             PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    SubArray{<:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                   PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                                          SubArray{<:Any, <:Any, <:LavaArray}}},
}

function Base.repeat(x::NestedLavaWrapper; inner=nothing, outer=nothing)
    dense = similar(x)
    dense .= x
    repeat(dense; inner, outer)
end

# Broadcast output goes on the device of the first device array in the tree,
# since the style carries no device.
firstdevicearray(x::LavaArray) = x
firstdevicearray(bc::Base.Broadcast.Broadcasted) = firstdevicearray(bc.args)
firstdevicearray(x::Base.Broadcast.Extruded) = firstdevicearray(x.x)
firstdevicearray(::Tuple{}) = nothing
function firstdevicearray(t::Tuple)
    a = firstdevicearray(first(t))
    return a === nothing ? firstdevicearray(Base.tail(t)) : a
end
firstdevicearray(x::SubArray) = firstdevicearray(parent(x))
firstdevicearray(x::Base.ReshapedArray) = firstdevicearray(parent(x))
firstdevicearray(x::PermutedDimsArray) = firstdevicearray(parent(x))
firstdevicearray(x::LinearAlgebra.Transpose) = firstdevicearray(parent(x))
firstdevicearray(x::LinearAlgebra.Adjoint) = firstdevicearray(parent(x))
# Unknown wrappers fall back to the default device instead of throwing:
# GPUArrays' suite broadcasts over ad-hoc wrapper types.
firstdevicearray(::Any) = nothing
function Base.similar(bc::Base.Broadcast.Broadcasted{LavaArrayStyle{N}}, ::Type{T}, dims) where {T,N}
    a = firstdevicearray(bc)
    bq = a === nothing ? vk_context().default_bq : queueof(a)
    # dims can be axes (OneTo) or plain integers
    LavaArray{T}(undef, map(Base.to_dim, dims); bq)
end

# GPUArrays' scalar `getindex`/`setindex!` go through these `copyto!` methods.
# `upload_typed!` and `copy_buffer!` handle mapping, staging and queue
# synchronization.
Base.copyto!(dest::LavaArray{T}, doffs::Integer,
             src::Array{T}, soffs::Integer, n::Integer) where {T} =
    copyhost!(dest,doffs,src,soffs,n)

function copyhost!(dest::LavaArray{T}, doffs::Integer,
                   src::AbstractArray{T}, soffs::Integer, n::Integer) where {T}
    n == 0 && return dest
    doffs,soffs,n = copybounds(dest,doffs,src,soffs,n)
    GC.@preserve dest upload_typed!(dest.buf[],src,soffs,n;offset=dest.offset+(doffs-1)*sizeof(T))
    return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    doffs,soffs,n = copybounds(dest,doffs,src,soffs,n)
    # Recorded on the channel that last named `src.buf[]`, so the host waits on
    # one timeline.
    GC.@preserve dest src copy_buffer!(:download, src.buf[],
                                   Ptr{UInt8}(pointer(dest, doffs)), n * sizeof(T);
                                   offset = src.offset + (Int(soffs) - 1) * sizeof(T))
    return dest
end

Base.copyto!(dest::LavaArray{T},doffs::Integer,
             src::HostReinterpretArray{T},soffs::Integer,n::Integer) where {T} =
    copyhost!(dest,doffs,src,soffs,n)
Base.copyto!(dest::LavaArray{T},src::HostReinterpretArray{T}) where T =
    copyto!(dest,firstindex(dest),src,firstindex(src),length(src))

function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    doffs,soffs,n = copybounds(dest,doffs,src,soffs,n)
    # Cross-device copies go through host memory, synchronously.
    if (src.buf[].ctx::VkContext) !== (dest.buf[].ctx::VkContext)
        staging = Vector{T}(undef, n)
        copyto!(staging, 1, src, soffs, n)
        copyto!(dest, doffs, staging, 1, n)
        return dest
    end
    src_offset = pool_offset(src.buf[]) + src.offset + (Int(soffs) - 1) * sizeof(T)
    dst_offset = pool_offset(dest.buf[]) + dest.offset + (Int(doffs) - 1) * sizeof(T)
    nbytes = n * sizeof(T)
    bq = (dest.buf[].ctx::VkContext).default_bq
    # Hold the arrays, not just their buffers, until the copy completes.
    oneshot!(bq; tag = :copy) do e
        hold!(e, src)
        hold!(e, dest)
        cmd_copy_buffer!(e, src.buf[], dest.buf[], nbytes;
                         src_off=src_offset, dst_off=dst_offset)
    end
    # Return without a host wait; core orders later resource users.
    return dest
end

function Base.fill!(a::LavaArray{T}, val) where T
    length(a) == 0 && return a
    v = convert(T, val)
    # The array's backend, not the global context, which may be another GPU.
    k = KI.Kernel(KernelAbstractions.get_backend(a), fill_kernel!)
    k(a, v; ndrange=length(a))
    return a
end

# Capacity-aware `resize!`, 1-D and N-D (textures). If the new size fits the
# buffer only `dims` changes, and new elements are undefined as for `Vector`.
# Otherwise the common prefix is copied into a new pool buffer and the old one is
# `unsafe_free!`d; core retires it after in-flight use, so callers need no
# `synchronize`. Throws on a freed array.

function Base.resize!(a::LavaArray{T,N}, new_dims::Dims{N}) where {T,N}
    all(>=(0), new_dims) || throw(ArgumentError("new size must be non-negative"))
    a.buf.freed && throw(ArgumentError(
        "resize!(::LavaArray{$T,$N}): the backing DataRef was already freed. " *
        "This array was unsafe_free!'d elsewhere — typically an HW-accel rebuild " *
        "that released state, or a cached slot whose owner was torn down."))
    Tuple(a.dims) == new_dims && return a
    old_len = length(a)
    new_len = prod(new_dims)

    buf = a.buf[]
    # In place only for the buffer's sole owner: a derived array (a view or a
    # reshape) shares its parent's buffer and would grow into the parent.
    if a.buf.rc.count[] == 1 && new_len * sizeof(T) <= buf.size - a.offset
        a.dims = new_dims
        return a
    end

    ctx = buf.ctx::VkContext
    bq = ctx.default_bq
    new_buf = pool_alloc(bq, max(new_len * sizeof(T), 16))
    if old_len > 0 && new_len > 0
        copy_len = min(old_len, new_len) * sizeof(T)
        src_off = pool_offset(buf) + a.offset
        oneshot!(bq; tag = :copy) do e
            cmd_copy_buffer!(e, buf, new_buf, copy_len;
                             src_off=src_off, dst_off=pool_offset(new_buf))
        end
        # The one-shot's submit stamps `buf` as in use, so `unsafe_free!` below
        # defers its destroy with `retire!`; no flush is needed.
    end
    new_ref = GPUArrays.DataRef(new_buf) do b
        vk_free!(b)
    end
    old_ref = a.buf
    a.buf = new_ref
    a.dims = new_dims
    a.offset = 0
    GPUArrays.unsafe_free!(old_ref)
    return a
end

Base.resize!(a::LavaArray{T,N}, new_dims::Vararg{Integer,N}) where {T,N} =
    resize!(a, Dims{N}(Int.(new_dims)))
Base.resize!(a::LavaArray{T,1}, n::Integer) where T = resize!(a, (Int(n),))

function Base.append!(a::LavaArray{T,1}, items::AbstractVector) where T
    n = length(a)
    resize!(a, n + length(items))
    isempty(items) || copyto!(a, n + 1, collect(T, items), 1, length(items))
    return a
end

# GPUArrays' `norm` captures variables in `Core.Box` closures, which fail to
# compile for the GPU; the reductions here capture only isbits values.

import LinearAlgebra

function LinearAlgebra.norm(v::LavaArray{T}, p::Real=2) where T
    RT = typeof(float(LinearAlgebra.norm(zero(T))))
    isempty(v) && return zero(RT)
    # `Base.count`: the bare name is Mantle's resource element `count`.
    p == 0 && return convert(RT, Base.count(!iszero, v))
    p == Inf && return convert(RT, maximum(abs, v))
    p == -Inf && return convert(RT, minimum(abs, v))
    # Other p-norms need transcendental math that is inaccurate on some Vulkan
    # drivers (RDNA3.5), so they run on the host.
    (p != 1 && p != 2) && return convert(RT, LinearAlgebra.norm(Array(v), p))
    # Float16/Float32 and their complex types accumulate in Float64, which cannot overflow.
    if RT === Float32 || RT === Float16
        p == 1 && return RT(sum(x -> Float64(abs(x)), v; init = 0.0))
        return RT(sqrt(sum(x -> Float64(abs(x))^2, v; init = 0.0)))
    end
    p == 1 && return convert(RT, sum(abs, v; init = 0.0))
    maxabs = convert(Float64, maximum(abs, v))
    (iszero(maxabs) || isinf(maxabs)) && return convert(RT, maxabs)
    # Unscaled when the sum of squares can neither overflow nor underflow.
    isfinite(length(v) * maxabs^2) && !iszero(maxabs^2) &&
        return convert(RT, sqrt(sum(abs2, v; init = 0.0)))
    # Rescaled by `maxabs`, dividing by `sqrt(maxabs)` twice: drivers lower `x / m`
    # to `x * (1/m)`, and for `m` near `floatmax` that reciprocal is subnormal and
    # flushed to zero (lavapipe, RADV).
    sm = sqrt(maxabs)
    return convert(RT, maxabs * sqrt(sum(x -> ((abs(x) / sm) / sm)^2, v; init = 0.0)))
end

# Two methods, matching GPUArrays' `NTuple` and general signatures to avoid ambiguity.
"""
    permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm)

`dest[I...] = src[I[perm]...]`, computed as `dest .= PermutedDimsArray(src, perm)`.
Overrides GPUArrays' method. An N-D launch without index divisions is 2.6-5.9x
slower than the flat broadcast: access order dominates, not index arithmetic.
"""
Base.permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::NTuple{N, T}) where {N, T} =
    lavapermutedims!(dest, src, perm)
Base.permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::AbstractVector) =
    lavapermutedims!(dest, src, Tuple(perm))

function lavapermutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::Tuple)
    N = ndims(src)
    length(perm) == N ||
        throw(ArgumentError("permutedims!: perm has $(length(perm)) entries for a $N-d array"))
    isperm(perm) || throw(ArgumentError("permutedims!: $perm is not a permutation"))
    ndims(dest) == N ||
        throw(DimensionMismatch("permutedims!: destination is $(ndims(dest))-d, source $N-d"))
    for d in 1:N
        size(dest, d) == size(src, perm[d]) ||
            throw(DimensionMismatch("permutedims!: destination axis $d is $(size(dest, d)), " *
                                    "source axis $(perm[d]) is $(size(src, perm[d]))"))
    end
    N == 0 && return dest
    # One kernel for both spellings; a faster permute belongs in the broadcast path.
    dest .= PermutedDimsArray(src, perm)
    return dest
end

# Backend-independent launch geometry and integer index decomposition.
#
# These helpers are used by portable kernels in DNNKernels and by Mantle's
# Vulkan array implementations.  Keeping them in the Vulkan extension made the
# same kernels fail at graph construction on Metal, even though every operation
# here is ordinary integer arithmetic.

struct FastDiv32
    d::UInt32
    mp::UInt32
    L::UInt32
end

"""
    FastDiv32(d)

Magic multiplier and shift for exact unsigned division by the positive 32-bit
integer `d`. Kernels use [`fastdiv`](@ref) to replace integer division with a
high multiply and shift.
"""
function FastDiv32(d::Integer)
    du = UInt32(d)
    du == 0 && throw(ArgumentError("FastDiv32: divisor must be positive"))
    L = UInt32(0)
    while L < 0x20 && (UInt32(1) << L) < du
        L += UInt32(1)
    end
    mp = UInt32(((UInt64(1) << 32) * ((UInt64(1) << L) - UInt64(du)) ÷ UInt64(du)) + 1)
    return FastDiv32(du, mp, L)
end

@inline fastdiv(n::UInt32, f::FastDiv32) =
    (UInt32((UInt64(n) * UInt64(f.mp)) >> 32) + n) >> f.L

const BROADCAST_FASTDIV_DEFAULT = true

"""Encode broadcast extents for [`cart32`](@ref), using fast division by default."""
@inline broadcastextents(sz::Dims; fastdiv::Bool = BROADCAST_FASTDIV_DEFAULT) =
    fastdiv ? map(FastDiv32, sz) : map(Int32, sz)

@inline cart32(q::UInt32, sz::Tuple{Int32, Vararg{Int32}}) = cart32(Int32(q), sz)
@inline cart32(::UInt32, ::Tuple{}) = ()
@inline function cart32(q::UInt32, sz::Tuple{FastDiv32, Vararg{FastDiv32}})
    f = first(sz)
    hi = fastdiv(q, f)
    return (Int(q - hi * f.d) + 1, cart32(hi, Base.tail(sz))...)
end
@inline cart32(::Int32, ::Tuple{}) = ()
@inline function cart32(q::Int32, sz::Tuple{Int32, Vararg{Int32}})
    s = first(sz)
    return (Int(q % s) + 1, cart32(q ÷ s, Base.tail(sz))...)
end

"""
    splitidx(idx, ::Val{N}) -> (idx % N, idx ÷ N)

Decompose a flat staging index without emitting integer division. The
power-of-two form uses a mask and shift; other constants use [`FastDiv32`](@ref).
"""
@generated function splitidx(idx::Integer, ::Val{N}) where {N}
    ispow2(N) && return :((Int(idx) & $(N - 1), Int(idx) >> $(trailing_zeros(N))))
    fd = FastDiv32(N)
    quote
        u = UInt32(idx)
        q = Int((UInt32((UInt64(u) * $(UInt64(fd.mp))) >> 32) + u) >> $(fd.L))
        (Int(u) - q * $N, q)
    end
end

@inline groupfill(::Tuple{}, left::Int) = ()
@inline function groupfill(sz::Tuple{Int, Vararg{Int}}, left::Int)
    t = min(first(sz), left)
    return (t, groupfill(Base.tail(sz), max(1, left ÷ t))...)
end

"""
    launchgroup(sz, target = 256) -> Dims

Fill a workgroup from the fastest axis outward while staying within `target`.
This keeps consecutive lanes on consecutive elements for multidimensional
launches.
"""
@inline launchgroup(sz::Dims, target::Int = 256) = groupfill(sz, target)

"""
    staticgroup(sz, target = 256) -> Dims

Like [`launchgroup`](@ref), but gives reachable interior axes at least two
threads so the shape can be embedded in a kernel type without triggering the
dynamic-workgroup fallback.
"""
function staticgroup(sz::Dims{N}, target::Int = 256) where {N}
    N <= 2 && return launchgroup(sz, target)
    w = ones(Int, N)
    for d in 2:(N - 1)
        c = min(2, sz[d])
        c > 1 && c * prod(w) > target && break
        w[d] = c
    end
    w[1] = min(sz[1], max(1, target ÷ prod(w)))
    for d in 1:(N - 1)
        room = target ÷ prod(w)
        room <= 1 && break
        w[d] = min(sz[d], w[d] * room)
    end
    return ntuple(d -> w[d], Val(N))
end

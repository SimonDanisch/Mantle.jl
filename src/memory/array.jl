# Typed pool-region handles and shared size/span validation.

"""Plain-bit byte extent; reject negative dimensions and element-count/byte overflow."""
@inline function checkedbytes(::Type{T}, dims::Dims) where {T}
    isbitstype(T) || throw(ArgumentError("device storage requires an isbits element type, got $T"))
    all(d -> d >= 0,dims) || throw(ArgumentError("dimensions must be nonnegative: $dims"))
    any(iszero,dims) && return 0
    return Base.checked_mul(sizeof(T),foldl(Base.checked_mul,dims;init=1))
end

"""Validate a byte span without overflowing offset + bytes."""
@inline function checkspan(capacity::Integer, offset::Integer, bytes::Integer)
    (0 <= offset <= capacity && 0 <= bytes <= capacity-offset) ||
        throw(ArgumentError("$bytes bytes at offset $offset exceed $capacity-byte storage"))
    return nothing
end

"""DeviceArray{T,N}(region, dims)

A typed, non-owning view of a pool region. The pool owns its memory; retire the
region explicitly after use. storage supplies the backend-native array view."""
struct DeviceArray{T,N}
    region::Region
    dims::NTuple{N,Int}
    function DeviceArray{T,N}(region::Region, dims::NTuple{N,Int}) where {T,N}
        need = checkedbytes(T,dims)
        need <= length(region) || throw(ArgumentError(
            "$(join(dims, "x")) $T needs $need bytes, region has $(length(region))"))
        new{T,N}(region,dims)
    end
end

DeviceArray{T}(region::Region, dims::NTuple{N,Int}) where {T,N} =
    DeviceArray{T,N}(region,dims)
DeviceArray{T}(region::Region, dims::Integer...) where {T} =
    DeviceArray{T}(region, Int.(dims))

Base.eltype(::DeviceArray{T}) where {T} = T
Base.ndims(::DeviceArray{T,N}) where {T,N} = N
Base.size(a::DeviceArray) = a.dims
Base.size(a::DeviceArray, i::Integer) = i <= ndims(a) ? a.dims[i] : 1
Base.length(a::DeviceArray) = prod(a.dims)
Base.sizeof(a::DeviceArray{T}) where {T} = length(a) * sizeof(T)

"""Which block, and where in it. What `materialize!` and the backend need."""
region(a::DeviceArray) = a.region
offset(a::DeviceArray) = offset(a.region)
memoryof(a::DeviceArray) = memoryof(a.region)

Base.show(io::IO, a::DeviceArray{T,N}) where {T,N} =
    print(io, "DeviceArray{", T, ",", N, "}(", join(a.dims, "x"),
          " @ ", offset(a), ")")

"""allocate(pool, dev, kind, T, dims; align) -> DeviceArray

Validate size, apply pool pressure policy and acquire a typed region. Retire
region(a) through the pool; reclaim! releases it after the device finishes."""
function allocate(pool::Pool, dev, kind, ::Type{T}, dims::NTuple{N,Int};
                  align::Int = 256, blocksize::Int = 64 << 20,
                  constraint = nothing) where {T,N}
    bytes = checkedbytes(T,dims)
    # The pool's own policy, on every backend: hand dead capacity back, and past
    # the soft cap collect before growing. See `PoolPolicy`.
    autotrim!(pool, dev)
    let cap = pool.policy.soft_cap
        cap > 0 && reserved(pool) >= cap && autocollect!(() -> reclaim!(pool, dev), pool)
    end
    r = acquire!(pool, dev, kind, nothing, bytes;
                 align, blocksize, constraint)
    return DeviceArray{T}(r, dims)
end
allocate(pool::Pool, dev, kind, ::Type{T}, dims::Integer...; kw...) where {T} =
    allocate(pool, dev, kind, T, Int.(dims); kw...)

# ── Backend-generic device queries ────────────────────────────────────────────

"""Maximum workgroup invocations for the backend of x."""
workgrouplimit(x) = caps(KernelAbstractions.get_backend(x)).workgrouplimit

"""Workgroup-memory budget in bytes for the backend of x."""
sharedbudget(x) = caps(KernelAbstractions.get_backend(x)).sharedbudget

"""Actual subgroup width for x; reductions must use this device fact."""
subgroupwidth(x) = caps(KernelAbstractions.get_backend(x)).subgroup

"""Whether the backend of x supports the cooperative-matrix GEMM shape."""
coopmatgemm(x) = caps(KernelAbstractions.get_backend(x)).coopmat

"""devicearray(backend, data) -> AbstractGPUArray

Allocate and copy data on the chosen backend. Backends may specialize allocation."""
function devicearray(backend, data::AbstractArray)
    a = KernelAbstractions.allocate(backend, eltype(data), size(data))
    copyto!(a, data)
    return a
end

devicearray(d::Device, data::AbstractArray) = devicearray(backend(d), data)

"""devicearray(x, T, dims...; unified) -> AbstractGPUArray

Allocate uninitialized storage on a device/backend; forward allocation keywords."""
devicearray(x, ::Type{T}, dims::Dims; kw...) where {T} =
    KernelAbstractions.allocate(backend(x), T, dims; kw...)

devicearray(x, ::Type{T}, dims::Integer...; kw...) where {T} =
    devicearray(x, T, Dims(dims); kw...)

"""Forward device adaptation to the backend belonging to that device."""
Adapt.adapt_storage(d::Device, x) = Adapt.adapt_storage(backend(d), x)

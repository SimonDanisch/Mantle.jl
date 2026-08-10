# A typed handle on a region of a pool block. Backend-independent, on purpose.
#
# Mantle used to reach for the backend's array type — `LavaArray` for the buffer
# arena — and suballocate inside it. That stacked two allocators AND made the
# ownership question unanswerable from here: whether a `LavaArray` built over an
# existing buffer frees that buffer on GC is Lava's business, and the answer
# decided whether Mantle's memory would be handed back to Lava's pool by the
# finalizer thread.
#
# Owning the type removes the question rather than answering it. Ownership is
# stated here, once, and it is not negotiable per backend:
#
#   the Block owns the memory, the Pool owns the Block, `trim!` frees.
#
# A `DeviceArray` owns NOTHING. Dropping one does not free anything, has no
# finalizer, and cannot race the GC thread — which is the failure mode this
# codebase has already paid for three times.
#
# This is also the direction of travel: Lava's high-level surface moves here over
# time and Lava becomes the mechanism underneath, so the array type belonging to
# Mantle is where it was always going to end up.

"""
    DeviceArray{T,N}(region, dims)

`dims` elements of `T`, living in `region`.

A handle, not an array: it is not `AbstractArray` and does not index, because the
bytes are wherever the backend put them and reading them from the host is a
transfer rather than a `getindex`. What a kernel actually receives is the
backend's business — see [`storage`](@ref).
"""
struct DeviceArray{T,N}
    region::Region
    dims::NTuple{N,Int}
end

function DeviceArray{T}(region::Region, dims::NTuple{N,Int}) where {T,N}
    need = prod(dims) * sizeof(T)
    need <= length(region) ||
        throw(ArgumentError("$(join(dims, "x")) $T needs $need bytes, region has $(length(region))"))
    DeviceArray{T,N}(region, dims)
end
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

"""
    allocate(pool, dev, kind, T, dims; align) -> DeviceArray

Suballocate `dims` elements of `T` out of the pool.

The counterpart to [`acquire!`](@ref) for callers that want a typed handle rather
than raw bytes. Freeing is `release!(region(a))` — explicit, like everything else
here, and never a finalizer.
"""
function allocate(pool::Pool, dev, kind, ::Type{T}, dims::NTuple{N,Int};
                  align::Int = 256, blocksize::Int = 64 << 20) where {T,N}
    r = acquire!(pool, dev, kind, nothing, prod(dims) * sizeof(T); align, blocksize)
    return DeviceArray{T}(r, dims)
end
allocate(pool::Pool, dev, kind, ::Type{T}, dims::Integer...; kw...) where {T} =
    allocate(pool, dev, kind, T, Int.(dims); kw...)

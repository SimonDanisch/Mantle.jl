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
than raw bytes. Handing it back is `retire!(pool, region(a))` — explicit, like
everything else here, and released by `reclaim!` once the device is done rather
than on the spot, so no caller has to know what is in flight.
"""
function allocate(pool::Pool, dev, kind, ::Type{T}, dims::NTuple{N,Int};
                  align::Int = 256, blocksize::Int = 64 << 20) where {T,N}
    r = acquire!(pool, dev, kind, nothing, prod(dims) * sizeof(T); align, blocksize)
    return DeviceArray{T}(r, dims)
end
allocate(pool::Pool, dev, kind, ::Type{T}, dims::Integer...; kw...) where {T} =
    allocate(pool, dev, kind, T, Int.(dims); kw...)

# ── Backend-generic device queries ────────────────────────────────────────────

"""
    workgrouplimit(x) -> Int

The largest workgroup the backend behind `x` will launch, where `x` is a device
array or anything `KernelAbstractions.get_backend` accepts.

Here rather than in the Vulkan backend because the ALGORITHMS that need it are
portable and their coupling to Vulkan was this query and nothing else. `gemv.jl`
and `fft.jl` reached it as `workgroup_limit(vk_context(A))` — array to context to
caps — and a context is the one step in that chain a second backend does not
have. Array to backend to `KernelInterface.caps` is the same three fields with
no Vulkan in the middle, and it is already what `caps(::LavaBackend)` answers.

This is the shape the rest of `src/vulkan/array/` has to take before it can move
out of the backend: `gemm.jl`, `fft.jl`, `gemv.jl` and `gemm_cm2.jl` are ~4000
lines of KernelAbstractions with a single-digit number of Vulkan references each,
and every one of them is a capability query like this. What is NOT ready is the
dispatch: they are written on `::LavaArray`, and widening that to
`::AbstractGPUArray` today would make Mantle pirate `LinearAlgebra.mul!` for
every GPU array package in the session. That widening wants a second backend to
be designed against, which is what step 5's "prove a vendor override works" is
for.
"""
workgrouplimit(x) = caps(KernelAbstractions.get_backend(x)).workgrouplimit

"""
    sharedbudget(x) -> Int

Workgroup-memory bytes a kernel on `x`'s backend may declare.

The companion of [`workgrouplimit`](@ref) and the other half of what a tiled
kernel needs to size itself: `fft.jl`'s `fftgroup` takes both, and between them
they were that file's ENTIRE dependency on Vulkan. It has none now.
"""
sharedbudget(x) = caps(KernelAbstractions.get_backend(x)).sharedbudget

"""
    coopmatgemm(x) -> Bool

Whether a cooperative-matrix GEMM is usable on `x`'s backend.

`DeviceCaps.coopmat` is exactly this — the Vulkan backend fills that field from
`coopmat_gemm_available(ctx)`, which probes the shape table and the subgroup
width — so reading the field is the portable way to ask, and it is cached where
the probe is not.

The PROBE stays in the backend and should: "does this device implement a 16x16x16
Float16 cooperative matrix, at a subgroup width the kernel was tuned for" is
answered out of Vulkan's shape table and `VK_EXT_subgroup_size_control`. What is
portable is the question, not the way it is answered.
"""
coopmatgemm(x) = caps(KernelAbstractions.get_backend(x)).coopmat


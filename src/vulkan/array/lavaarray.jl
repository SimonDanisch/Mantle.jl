# `LavaArray{T,N}`: the GPUArrays array type over a Vulkan buffer.

import GPUArraysCore: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

export LavaArray

"""
    LavaArray{T,N} <: AbstractGPUArray{T,N}

GPU array over a `VkManagedBuffer`; kernels receive its buffer device address.
"""
mutable struct LavaArray{T,N} <: AbstractGPUArray{T,N}
    buf::GPUArrays.DataRef{VkManagedBuffer}
    dims::NTuple{N,Int}
    # Byte offset of the first element in the buffer. Bytes, not elements: a
    # reinterpret of an Int32 view can start mid-Int64 (offset 4).
    offset::Int

    # `DataRef` releases only on `unsafe_free!`, never through GC, so without
    # the finalizer unreferenced arrays leak. `DataRef.freed` and the buffer's
    # atomic state make a second `unsafe_free!` a no-op.
    function LavaArray{T,N}(buf::GPUArrays.DataRef{VkManagedBuffer}, dims::NTuple{N,Int};
                            offset::Integer=0) where {T,N}
        xs = new{T,N}(buf, dims, offset)
        finalizer(unsafe_free!, xs)
        return xs
    end
end

# `bq` defaults to the process default device. Backend code that has a device
# must pass its queue (`test_no_ambient_allocation.jl`).
function LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Int};
                        bq::SubmitChannel{<:VulkanQueue}=Device(VulkanAPI()).default_bq,
                        extra_usage::UInt32=UInt32(0),
                        scratch::Bool=false,
                        unified::Bool=false) where {T,N}
    # Otherwise the selector bytes are silently dropped and values read back wrong.
    Base.isbitsunion(T) &&
        throw(ArgumentError("LavaArray does not support isbits-union element types like $T: their type-selector bytes are not stored"))
    ctx = deviceof(bq)
    nbytes = checkedbytes(T,dims)

    # Acceleration-structure scratch needs `ctx.as_scratch_align`: over-allocate
    # by `align - 1` bytes and offset the array onto the boundary.
    align = bda_alignment_for(ctx, scratch)
    slack = align > 1 ? Int(align) - 1 : 0

    # Extra usage flags, scratch and host-mapped memory need their own
    # VkBuffer/VkDeviceMemory flags, so they bypass the pool.
    use_pool = extra_usage == UInt32(0) && !scratch && !unified
    allocation_bytes = max(Base.checked_add(nbytes,slack),16)
    managed_buf = use_pool ?
        pool_alloc(bq, allocation_bytes) :
        vk_alloc(bq, allocation_bytes; extra_usage, unified)

    byte_offset = 0
    if align > UInt64(1)
        base = managed_buf.address
        aligned = cld(base, align) * align
        byte_offset = Int(aligned - base)
        @assert byte_offset % sizeof(T) == 0  "extra_usage alignment ($(align)) is not a multiple of sizeof($T)"
    end

    ref = GPUArrays.DataRef(managed_buf) do buf
        vk_free!(buf)
    end
    LavaArray{T,N}(ref, dims; offset=byte_offset)
end

LavaArray{T,N}(::UndefInitializer, dims::Int...; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T,N}(::UndefInitializer, dims::Integer...; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)

LavaArray{T,1}() where {T} = LavaArray{T,1}(undef, (0,))

LavaArray{T}(::UndefInitializer, dims::NTuple{N,Int}; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T}(::UndefInitializer, dims::Integer...; kw...) where {T} = LavaArray{T}(undef, Int.(dims); kw...)

# Upload host data; allocation keywords (`bq`, `extra_usage`, `unified`) pass through.
function LavaArray{T,N}(data::AbstractArray{T,N}; kw...) where {T,N}
    arr = LavaArray{T,N}(undef, size(data); kw...)
    GC.@preserve arr upload!(arr, data)
    return arr
end
LavaArray(data::AbstractArray{T,N}; kw...) where {T,N} = LavaArray{T,N}(data; kw...)

LavaArray{T}(data::AbstractArray{S,N}; kw...) where {T,S,N} =
    LavaArray{T,N}(convert(AbstractArray{T}, data); kw...)
LavaArray{T,N}(data::AbstractArray{S,N}; kw...) where {T,S,N} =
    LavaArray{T,N}(convert(AbstractArray{T}, data); kw...)

# UniformScaling constructors resolve an ambiguity with GPUArrays'.
import LinearAlgebra: UniformScaling

# Top level: a closure per call would be a new function, compiled every time.
function identity_kernel!(res, stride, val)
    i = KI.get_global_id().x
    ilin = (stride * (i - 1)) + i
    if ilin <= length(res)
        @inbounds res[ilin] = val
    end
    return nothing
end

function (::Type{LavaArray{T,N}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T,N}
    res = similar(LavaArray{T,N}, dims)
    fill!(res, zero(T))
    isempty(res) && return res
    # The array's device, not the global `Device(VulkanAPI())`, which may be another GPU.
    KI.Kernel(KernelAbstractions.get_backend(res), identity_kernel!)(
        res, size(res, 1), T(s.λ); ndrange=minimum(dims))
    return res
end
(::Type{LavaArray{T}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T} =
    LavaArray{T,2}(s, dims)

GPUArrays.storage(a::LavaArray) = a.buf

function GPUArrays.derive(::Type{T}, a::LavaArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.buf)
    # `offset` arrives in units of T (GPUArrays contract); a.offset is bytes.
    byte_offset = a.offset + offset * sizeof(T)
    LavaArray{T,N}(ref, dims; offset=byte_offset)
end

"""
    GPUArrays.memory_location(a::LavaArray) -> (base, offset)

The buffer's device address and the byte offset of `a`'s first element.
GPUArrays derives `mightalias`, `dataids` and Base's `_parentsmatch` from it,
so arrays alias exactly where their byte ranges overlap. A buffer without a
device address is keyed by its allocation instead.
"""
function GPUArrays.memory_location(a::LavaArray)
    m = a.buf[]
    base = m.address == 0 ? UInt(objectid(a.buf.rc)) : UInt(m.address)
    return (base, a.offset)
end

function Base.copy(a::LavaArray{T,N}) where {T,N}
    b = similar(a)
    copyto!(b, 1, a, 1, length(a))
    return b
end

# Allocate on the array's own device, not the process default, so broadcast,
# `copy` and `map` results stay on it.
queueof(a::LavaArray) = (a.buf[].ctx::LavaDevice).default_bq
Base.similar(a::LavaArray{T,N}) where {T,N} = LavaArray{T,N}(undef, a.dims; bq = queueof(a))
Base.similar(a::LavaArray{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims; bq = queueof(a))
Base.similar(a::LavaArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims; bq = queueof(a))

Base.size(a::LavaArray) = a.dims
Base.length(a::LavaArray) = prod(a.dims)
Base.sizeof(a::LavaArray{T}) where T = length(a) * sizeof(T)
Base.eltype(::LavaArray{T}) where T = T
Base.ndims(::LavaArray{T,N}) where {T,N} = N
Base.IndexStyle(::Type{<:LavaArray}) = IndexLinear()
Base.elsize(::Type{<:LavaArray{T}}) where T = sizeof(T)

# Device address of the first element, view offset included.
function bda_address(a::LavaArray{T}) where T
    a.buf[].address + a.offset
end

# Alignment of the actual address, view offset included.
basealignment(a::LavaArray) = powalign(bda_address(a))

# The generic `hold!(::Closed, obj)` holds the `LavaArray` wrapper, not just its
# buffer: the wrapper's finalizer would otherwise release the buffer
# mid-recording.

# The array's lifetime stamp is its buffer's. Through `a.buf.rc.obj`, not
# `a.buf[]`, which throws after `unsafe_free!`: a hold list can be dropped
# after the array it names was freed.
@inline stampof(a::LavaArray) = stampof(a.buf.rc.obj::VkManagedBuffer)

# Through `buf[]`, so naming a freed array in a command throws.
@inline syncbuf!(owner::Closed, a::LavaArray) = syncbuf!(owner, a.buf[])

# Adapt.jl adaptor converting to device values; it holds nothing, callers
# hold the originals with `holdleaves!`. `batch === nothing` means compiling
# without a recording owner.
struct LavaAdaptor{P<:Union{Nothing,Closed}}
    batch::P
end

"""Upload host `data` into `dst`; lengths must match."""
function upload!(dst::LavaArray{T}, data::AbstractArray{T}) where T
    @assert length(data) == length(dst) "Size mismatch: $(length(data)) vs $(length(dst))"
    GC.@preserve dst upload_typed!(dst.buf[], data; offset=dst.offset)
end

"""Download `src` to a new host `Array`."""
function Base.Array(src::LavaArray{T,N}) where {T,N}
    result = Array{T}(undef, src.dims...)
    download_typed!(vec(result), src.buf[]; offset=src.offset)
    return result
end

Base.collect(a::LavaArray) = Array(a)

# `unsafe_free!` is GPUArrays' generic `unsafe_free!(storage(x))`.

"""
    LavaDeviceArray(a::LavaArray)

The `(pointer, dims)` value a kernel receives for `a`. Qualified as
`Lava.LavaDeviceArray` because extending an imported-by-`using` name is
deprecated in Julia 1.12.
"""
function Lava.LavaDeviceArray(a::LavaArray{T,N}) where {T,N}
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end

# A recorded draw bakes the address; it changes exactly when `resize!` reallocates.
Mantle.argidentity(a::LavaArray) = bda_address(a)

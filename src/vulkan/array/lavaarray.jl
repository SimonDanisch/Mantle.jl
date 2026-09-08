# LavaArray{T,N} — GPU array backed by Vulkan buffer
#
# Full GPUArrays.jl compatible implementation with DataRef, offset, derive.

import GPUArraysCore: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

# Exported, as `LavaBackend` is. It was Lava's export until the runtime moved
# here, and the move left it defined but unexported — so 43 test files that say
# `LavaArray` unqualified stopped resolving, each with an `UndefVarError` at
# whatever line first names it rather than at load. The host array type of a
# backend is part of its surface; the alternative is qualifying it in every one
# of those files to say the same thing.
export LavaArray

"""
    LavaArray{T,N} <: AbstractGPUArray{T,N}

GPU array backed by a Vulkan device-local buffer with BDA (Buffer Device Address).
"""
mutable struct LavaArray{T,N} <: AbstractGPUArray{T,N}
    buf::GPUArrays.DataRef{VkManagedBuffer}
    dims::NTuple{N,Int}
    # Byte offset from the start of the backing VkManagedBuffer to the first
    # element of this view. Stored in bytes (not elements) so that a
    # reinterpret that changes `sizeof(eltype)` doesn't have to round-trip
    # through element counts — e.g. `reinterpret(Int64, view(a::LavaArray{Int32}, 2:7))`
    # creates a LavaArray{Int64} with `offset = 4`, which is a legal byte
    # offset even though it isn't a whole Int64 element.
    offset::Int

    # Register `unsafe_free!` as a GC finalizer so LavaArrays that fall out
    # of scope without an explicit `unsafe_free!` call actually release their
    # GPU memory. `GPUArrays.DataRef` is refcount-based — its `release`
    # callback only fires on explicit `unsafe_free!`, NOT via Julia GC — so
    # WITHOUT this finalizer every scratch LavaArray leaks until the process
    # exits. Matches AMDGPU.ROCArray's pattern (`finalizer(unsafe_free!, xs)`
    # at `AMDGPU/src/array.jl:13,19`). Idempotency is guaranteed by the
    # `DataRef.freed` flag + `VkManagedBuffer.state` atomic CAS, so a second
    # `unsafe_free!` (e.g. explicit user call before GC) is a no-op.
    function LavaArray{T,N}(buf::GPUArrays.DataRef{VkManagedBuffer}, dims::NTuple{N,Int};
                            offset::Integer=0) where {T,N}
        xs = new{T,N}(buf, dims, offset)
        finalizer(unsafe_free!, xs)
        return xs
    end
end

function LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Int};
                        bq::VulkanBatchQueue=vk_context().default_bq,
                        extra_usage::UInt32=UInt32(0),
                        scratch::Bool=false,
                        unified::Bool=false) where {T,N}
    ctx = bq.ctx::VkContext
    nbytes = prod(dims) * sizeof(T)

    # AS-build scratch buffers need a specific BDA alignment
    # (`ctx.as_scratch_align`).  Other buffers default to 1.  We over-allocate
    # by (align - 1) bytes and shift the LavaArray's element offset so
    # `bda_address(arr)` lands on the right boundary.  For T=UInt8 the offset
    # is always valid; otherwise we assert divisibility.
    align = bda_alignment_for(ctx, scratch)
    slack = align > 1 ? Int(align) - 1 : 0

    # Non-default usage (index buffer, AS input/storage, scratch, etc.) or
    # BAR-mapped (unified) memory bypasses the pool since those buffers need
    # specific flags at VkBuffer/DeviceMemory creation time.
    use_pool = extra_usage == UInt32(0) && !scratch && !unified
    managed_buf = use_pool ?
        pool_alloc(bq, max(nbytes + slack, 16)) :
        vk_alloc(bq, max(nbytes + slack, 16); extra_usage, unified)

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

# Varargs constructor for LavaArray{T,N}(undef, d1, d2, ...)
LavaArray{T,N}(::UndefInitializer, dims::Int...; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T,N}(::UndefInitializer, dims::Integer...; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)

"""Allocate a LavaArray with INDEX_BUFFER_BIT for use as Vulkan index buffer."""
alloc_index_buffer(data::AbstractVector{UInt32}) = begin
    arr = LavaArray{UInt32,1}(undef, (length(data),);
        extra_usage=UInt32(VK.BUFFER_USAGE_INDEX_BUFFER_BIT))
    upload!(arr, data)
    return arr
end

# Empty vector constructor (matches Array{T,1}() behavior)
LavaArray{T,1}() where {T} = LavaArray{T,1}(undef, (0,))

LavaArray{T}(::UndefInitializer, dims::NTuple{N,Int}; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T}(::UndefInitializer, dims::Integer...; kw...) where {T} = LavaArray{T}(undef, Int.(dims); kw...)

# Construct from host data.  Forwards alloc kwargs (bq/extra_usage/unified)
# so callers don't need a separate "alloc-then-upload" helper for each usage.
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

# UniformScaling constructor (resolve ambiguity with GPUArrays inner constructor)
import LinearAlgebra: UniformScaling
function (::Type{LavaArray{T,N}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T,N}
    res = similar(LavaArray{T,N}, dims)
    fill!(res, zero(T))
    isempty(res) && return res
    @kernel cpu=false function identity_kernel!(res, stride, val)
        i = @index(Global, Linear)
        ilin = (stride * (i - 1)) + i
        if ilin <= length(res)
            @inbounds res[ilin] = val
        end
    end
    # `KA.get_backend(res)`, NOT `LavaBackend()`. An unpinned backend resolves
    # its queue through `vk_context()`, so on a second device this dispatches on
    # whichever context happens to be global — the work lands on the wrong GPU
    # and the buffer's own device never sees it. `get_backend` derives the
    # context from the array's buffer, which has always carried it.
    kernel = identity_kernel!(KernelAbstractions.get_backend(res))
    kernel(res, size(res, 1), T(s.λ); ndrange=minimum(dims))
    return res
end
(::Type{LavaArray{T}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T} =
    LavaArray{T,2}(s, dims)

# ── GPUArrays interface: storage & derive ──

GPUArrays.storage(a::LavaArray) = a.buf

function GPUArrays.derive(::Type{T}, a::LavaArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.buf)
    # `offset` arrives in units of T (GPUArrays contract); a.offset is bytes.
    byte_offset = a.offset + offset * sizeof(T)
    LavaArray{T,N}(ref, dims; offset=byte_offset)
end

# ── copy ──

function Base.copy(a::LavaArray{T,N}) where {T,N}
    b = similar(a)
    copyto!(b, 1, a, 1, length(a))
    return b
end

# ── similar ──

Base.similar(a::LavaArray{T,N}) where {T,N} = LavaArray{T,N}(undef, a.dims)
Base.similar(a::LavaArray{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims)
Base.similar(a::LavaArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims)

# ── Base interface ──

Base.size(a::LavaArray) = a.dims
Base.length(a::LavaArray) = prod(a.dims)
Base.sizeof(a::LavaArray{T}) where T = length(a) * sizeof(T)
Base.eltype(::LavaArray{T}) where T = T
Base.ndims(::LavaArray{T,N}) where {T,N} = N
Base.IndexStyle(::Type{<:LavaArray}) = IndexLinear()
Base.elsize(::Type{<:LavaArray{T}}) where T = sizeof(T)

# BDA address for kernel argument passing (includes offset, already in bytes)
function bda_address(a::LavaArray{T}) where T
    a.buf[].address + a.offset
end

# pin! the LavaArray wrapper itself, NOT just its VkManagedBuffer. The wrapper
# is what Julia's GC traces: pinning the raw VkManagedBuffer wouldn't keep the
# LavaArray alive, so its GC finalizer could fire mid-batch, call `unsafe_free!`
# on the underlying DataRef, and leave the wavefront using a DEFERRED/DEAD
# buffer — which `sync_access!` rightly asserts against.
# `sync_access!(::LavaArray)` below forwards to the underlying VkManagedBuffer
# so cross-queue last_write tracking still runs on the leaf.
@inline pin!(batch::O, a::LavaArray) where {O<:Closed} = begin
    a in batch.pinned && return
    push!(batch.pinned, a)
    # Two claims, and both are needed:
    #   * the retained DataRef keeps `ref[]` dereferenceable after an explicit
    #     `unsafe_free!(a)` drops the array's own ref (see pinned_refs), and
    #   * the buffer pin stops `vk_free!` from marking the VkManagedBuffer
    #     DEFERRED/DEAD underneath us, which `sync_access!` asserts against.
    # Without the pin the DataRef would stay readable but describe a buffer
    # already queued for destruction.
    ref = copy(a.buf)
    push!(batch.pinned_refs, ref)
    pin_buffer!(ref[])
    return nothing
end

@inline sync_access!(sub::Submission, a::LavaArray) = sync_access!(sub, a.buf[])

# LavaAdaptor: converts LavaArray → LavaDeviceArray (Ptr-wrapping) for GPU
# kernel compilation, and pins every visited LavaArray into the current batch.
# Declared here (ahead of ka_backend.jl which uses it in method signatures);
# the `adapt_storage` / `adapt_structure` methods live in gpuarrays.jl.
# PARAMETERISED on the owner, not `batch::Closed`. An abstract-typed field makes
# the struct non-concrete, so `adaptor.batch` is a dynamic load and every
# `pin_leaves!`/`pack_arg!` reached through it becomes a dynamic call — 835 bytes
# per dispatch, measured by `test_dispatch_allocation.jl`, which exists for
# exactly this class of regression.
#
# `nothing` is an owner too: at COMPILE there is nothing to emit into yet, and
# the adaptor is used for its pure half — `adapt_storage` is a strip, and the
# pinning is a separate walk that never sees this field.
struct LavaAdaptor{P<:Union{Nothing,Closed}}
    batch::P
end

# ── Transfers ──

"""Upload host data to GPU array (must have matching length)."""
function upload!(dst::LavaArray{T}, data::AbstractArray{T}) where T
    @assert length(data) == length(dst) "Size mismatch: $(length(data)) vs $(length(dst))"
    src = vec(collect(data))
    nbytes = length(src) * sizeof(T)
    bytes = Vector{UInt8}(undef, nbytes)
    GC.@preserve src bytes begin
        unsafe_copyto!(Ptr{UInt8}(pointer(bytes)), Ptr{UInt8}(pointer(src)), nbytes)
    end
    GC.@preserve dst upload!(dst.buf[], bytes; offset=dst.offset)
end

# `update!` is deleted — use `Base.resize!(arr, length(new)) + Base.copyto!(arr, new)`.
# `Base.resize!(::LavaArray)` is capacity-aware (see array/gpuarrays.jl); `copyto!` is
# the standard GPUArrays upload path.  That pair is the full "make dst match src"
# operation and composes with Base semantics across CuArray / ROCArray / LavaArray
# without a Lava-specific hook.

"""Download GPU array to host."""
function Base.Array(src::LavaArray{T,N}) where {T,N}
    result = Array{T}(undef, src.dims...)
    download_typed!(vec(result), src.buf[]; offset=src.offset)
    return result
end

"""Convenience: collect downloads to host."""
Base.collect(a::LavaArray) = Array(a)

# ── Memory management ──
#
# No `unsafe_free!(::LavaArray)`: GPUArrays' generic method is
# `unsafe_free!(storage(x))`, and `storage(a::LavaArray)` is `a.buf` above.

# ── Aliasing ──
#
# `dest .= src` where both sides are views asks Base whether the two might
# overlap, and Base answers by comparing the parents: for `DenseArray` parents
# that means `pointer(A) == pointer(B)` (multidimensional.jl `_parentsmatch`).
# `LavaArray <: AbstractGPUArray <: DenseArray` but has no host pointer, so the
# check threw `conversion to pointer not defined` — any `view(a) .= view(b)`
# between device arrays was unreachable, whatever the shapes.
#
# Identity for a LavaArray is its backing buffer plus the byte offset into it,
# which is exactly what a pointer would encode. Answering it directly keeps the
# rest of Base's machinery — once the parents match it compares the actual index
# ranges, so an overlapping copy still gets its temporary and a disjoint one
# still does not.
Base._parentsmatch(A::LavaArray, B::LavaArray) = A.buf === B.buf && A.offset == B.offset

# ── Device-side array ──
#
# `LavaDeviceArray` itself is in `device/devicearray.jl`, with the rest of the
# code a kernel is compiled against: it is a `(pointer, dims)` pair and knows
# nothing about buffers, pools or lifetimes. Only the conversion is here, because
# only the host handle knows its own buffer address.

"""Convert a host `LavaArray` to the form a kernel receives."""
# Qualified. `using Lava: LavaDeviceArray` makes the name visible but does not
# make this an extension of Lava's constructor — 1.12 assumes it is one and
# deprecates the assumption, so it is said outright. `import`ing the name
# instead would work equally well and read as though Mantle owned the type.
function Lava.LavaDeviceArray(a::LavaArray{T,N}) where {T,N}
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end

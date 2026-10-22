# Persistent resources, one implementation for every backend. They differ from
# transients only by lifetime (held until the user drops them) and come from the
# same pool, so weights and frame buffers are visible to one allocator.

"""
    Buffer(dev, data; capacity = length(data)) -> Buffer
    Buffer(dev, T, n) -> Buffer
    Buffer(dev, T, dims) -> Buffer

A device array of rank `N`. For `N == 1`, `store` is always at `capacity` and
`len` is what a draw covers, so a per-frame count change does not reallocate.
For `N > 1` both are `prod(dims)` and `resize!` does not apply.
"""
mutable struct Buffer{T,N} <: Resource
    store::DeviceArray{T,N}
    len::Int
    capacity::Int
    dev::Any
    # Stores made through `setindex!` and not yet landed, `(range, data)` in
    # store order, and the flag a run reads before taking the lock.
    pending::Vector{Tuple{UnitRange{Int},Vector{T}}}
    pendinglock::ReentrantLock
    @atomic dirty::Bool
    # Set once the region is handed back; see [`retireonce!`](@ref).
    @atomic retired::Bool
end

function Buffer{T,N}(store, len::Int, capacity::Int, dev) where {T,N}
    b = Buffer{T,N}(store, len, capacity, dev, Tuple{UnitRange{Int},Vector{T}}[],
                    ReentrantLock(), false, false)
    finalizer(retireonce!, b)
    return b
end

"""
    GPURef(dev, x) -> GPURef{T}

One value on the device at an address that never moves. A dispatch is packed
with the address once, at `record!`; `ref[] = x` writes the value as a command
in the next run's submission, so a change costs one small transfer and never
rewrites argument memory. A value passed directly is fixed for the plan's life.
Its `stride` is 0, so it binds as a vertex attribute shared by every element.
"""
mutable struct GPURef{T} <: Resource
    store::DeviceArray{T,1}
    dev::Any
    # The last stored value and the seqlock a run reads it under; see `setindex!`.
    pending::Base.RefValue{T}
    @atomic seq::UInt64
    @atomic dirty::Bool
    @atomic retired::Bool
end

function GPURef{T}(store, dev) where {T}
    r = GPURef{T}(store, dev, Base.RefValue{T}(), UInt64(0), false, false)
    finalizer(retireonce!, r)
    return r
end

"""
    upload!(dev, dst::DeviceArray, first, data)

Write host `data` into `dst` starting at element `first`. Backend-supplied.
"""
function upload! end

"""
    download(dev, src::DeviceArray) -> Vector

`src`'s contents on the host; the inverse of [`upload!`](@ref).
"""
function download end

"""
    deviceview(dev, a::DeviceArray)

What the backend hands a kernel or library for `a`: a `(ptr, dims)` device
struct for a kernel, or a non-owning host-side array handle for a library or a
copy (the `Block` owns the memory).
"""
function deviceview end

"""
    hostspan(dev, memory) -> (Ptr{UInt8}, nbytes)

Where a block begins in the host address space, and its size, for a backend the
CPU addresses directly (Metal: a `Shared` `MTLBuffer`'s `contents`).
[`hostview`](@ref) and the `host*` transfer verbs build on it. A backend with
unmapped memory (Vulkan) does not answer and implements the transfers itself.
"""
function hostspan end

"""
    hostview(T, base, nbytes, offset, dims) -> Array{T}
    hostview(dev, a::DeviceArray) -> Array{T}

A bounds-checked host `Array` over device memory, without copying; padded bits
types are allowed. The caller must keep the backing region alive while using it.
"""
function hostview(::Type{T}, base::Ptr{UInt8}, nbytes::Integer, offset::Integer,
                  dims::Dims) where {T}
    checkspan(nbytes,offset,checkedbytes(T,dims))
    return unsafe_wrap(Array, Ptr{T}(base + offset), dims; own = false)
end

function hostview(dev, a::DeviceArray{T}) where {T}
    base, nbytes = hostspan(dev, memoryof(a))
    return hostview(T, base, nbytes, offset(a), size(a))
end

"""
    hostupload!(dev, a, first, data) -> a
    hostdownload(dev, a)             -> Vector
    hostdevicecopy!(dev, dst, src, n) -> dst

`upload!`, `download` and `devicecopy!` for a backend the CPU addresses
directly, as `copyto!` over [`hostview`](@ref). The two that read device memory
first call [`awaitwrites`](@ref): shared memory does not mean the writing kernel
has finished, and an unwaited read silently returns stale bytes. `upload!` does
not wait; the graph orders writes before readers.
"""
function hostupload!(dev, a::DeviceArray, first::Integer, data::AbstractVector)
    copyto!(hostview(dev, a), first, data, 1, length(data))
    return a
end

function hostdownload(dev, a::DeviceArray)
    awaitwrites(dev)
    return collect(hostview(dev, a))
end

function hostdevicecopy!(dev, dst::DeviceArray{T}, src::DeviceArray{T},
                         n::Integer) where {T}
    awaitwrites(dev)
    copyto!(hostview(dev, dst), 1, hostview(dev, src), 1, Int(n))
    return dst
end

"""
    bufferusage(dev, T) -> constraint

What memory a persistent buffer of `T` needs, in the form [`compatible`](@ref)
reads. Some element types need more than the ordinary bits: draw commands need
`INDIRECT_BUFFER_BIT`, or the draw fails validation.
"""
function bufferusage end

"""Which arena persistent buffers live in. Separate from the transient arenas so
a long-lived allocation does not fragment the space plans reuse every frame."""
struct Persistent end

# `todevice` so a caller passing a KA backend gets a buffer holding a real
# device; otherwise `pool(::MetalBackend)` fails two calls later.
function Buffer(dev, data::AbstractVector{T}; capacity::Integer = length(data)) where {T}
    d = todevice(dev)
    cap = max(Int(capacity), length(data))
    b = Buffer{T,1}(persistentarray(d, T, (cap,)), length(data), cap, d)
    isempty(data) || upload!(d, b.store, 1, data)
    return b
end

function Buffer(dev, ::Type{T}, dims::Dims{N}) where {T,N}
    d = todevice(dev)
    n = prod(dims)
    return Buffer{T,N}(persistentarray(d, T, dims), n, n, d)
end
Buffer(dev, ::Type{T}, n::Integer) where {T} = Buffer(dev, T, (Int(n),))

"""
    indexbuffer(dev, indices::AbstractVector{UInt32}) -> Buffer

`indices` on the device, usable as an indexed draw's index buffer. The default
suits Metal, which takes any buffer; Vulkan needs
`BUFFER_USAGE_INDEX_BUFFER_BIT` at allocation and overrides this. Not
`extrausage`, which keys on the element type and would give every `UInt32`
buffer the bit.
"""
indexbuffer(dev::Device, indices::AbstractVector{UInt32}) = Buffer(dev, indices)
# Converted to a `Device` before dispatch so the Vulkan override is reached; a
# raw backend would get a buffer without INDEX_BUFFER_BIT.
indexbuffer(b::KernelAbstractions.Backend, indices::AbstractVector{UInt32}) =
    indexbuffer(Device(b), indices)

"`data`'s shape and contents on the device. The `N > 1` counterpart of the
vector constructor; the upload is linear, because a region is."
function Buffer(dev, data::AbstractArray{T,N}) where {T,N}
    d = todevice(dev)
    b = Buffer(d, T, size(data))
    isempty(data) || upload!(d, b.store, 1, vec(collect(data)))
    return b
end

GPURef(dev, x::T) where {T} =
    (d = todevice(dev); s = GPURef{T}(persistentarray(d, T, (1,)), d);
     upload!(d, s.store, 1, [x]); s)

"""
A region of `dims` elements of `T` from the device's persistent arena, with
`bufferusage(dev, T)` as the block constraint. Without it a conditional-rendering
predicate lacks `CONDITIONAL_RENDERING_BIT`, which hangs the GPU on NVIDIA.
"""
persistentarray(dev, ::Type{T}, dims::Dims) where {T} =
    allocate(pool(dev), dev, Persistent(), T, dims;
             align = 256, blocksize = blocksize(dev),
             constraint = bufferusage(dev, T))

"""
A resource's leaf is its region. The generic walker would follow `dev` into the
`VkContext` and `Vulkan.Instance`, whose `destructor` captures the instance
back. A launch must keep alive the memory its commands name; the device
outlives its resources.
"""
@inline holdleaves!(holder, r::Union{Buffer,GPURef}) = holdleaves!(holder, r.store)

Base.length(b::Buffer) = b.len
Base.size(b::Buffer) = size(b.store)
# Per-dimension `size`, `1` past the rank as Base does.
Base.size(b::Buffer, d::Integer) = d <= ndims(b) ? size(b)[d] : 1
Base.ndims(::Buffer{T,N}) where {T,N} = N
Base.fill!(b::Buffer, v) = (fill!(storage(b), v); b)
# `similar` returns a `Buffer`, freed by the same `free!`.
Base.similar(b::Buffer{T,N}) where {T,N} = Buffer(b.dev, T, size(b))
Base.similar(b::Buffer, ::Type{T}) where {T} = Buffer(b.dev, T, size(b))
Base.similar(b::Buffer{T}, dims::Dims) where {T} = Buffer(b.dev, T, dims)
Base.similar(b::Buffer, ::Type{T}, dims::Dims) where {T} = Buffer(b.dev, T, dims)
Base.similar(b::Buffer{T}, dims::Integer...) where {T} = Buffer(b.dev, T, Dims(dims))
Base.similar(b::Buffer, ::Type{T}, dims::Integer...) where {T} = Buffer(b.dev, T, Dims(dims))
# `reshape` and `view` return arrays over the same bytes and own nothing;
# `similar` allocates. This is `Base.view`, unrelated to Mantle's
# `ResourceView` (a pass binding, made by `viewof`).
Base.reshape(b::Buffer, dims::Dims) = reshape(storage(b), dims)
Base.reshape(b::Buffer, dims::Integer...) = reshape(storage(b), dims)
Base.view(b::Buffer, I...) = view(storage(b), I...)
# `Base.strides`, not Mantle's binding `stride` below.
Base.strides(b::Buffer) = strides(storage(b))
Base.copyto!(dst::Buffer, src) = (copyto!(storage(dst), src); dst)
Base.copyto!(dst::AbstractArray, src::Buffer) = copyto!(dst, storage(src))
Base.length(::GPURef) = 1
# `stride` 0 means one value shared by every element, so a `GPURef` and a
# per-element `Buffer` share shader path and pipeline (see `stride` in
# runtime/api.jl). `count` is what a draw covers.
stride(::Buffer) = 1
stride(::GPURef) = 0
count(b::Buffer) = b.len
count(::GPURef) = 1
Base.eltype(::Buffer{T}) where {T} = T
Base.eltype(::GPURef{T}) where {T} = T

capacity(b::Buffer) = b.capacity
storage(b::Buffer) = deviceview(b.dev, b.store)
storage(s::GPURef) = deviceview(s.dev, s.store)
# Only the vector form trims to `len`; an `N > 1` buffer keeps its shape.
Base.Array(b::Buffer{T,1}) where {T} = download(b.dev, b.store)[1:b.len]
Base.Array(b::Buffer) = download(b.dev, b.store)

update!(b::Buffer{T,N}, data::AbstractVector) where {T,N} = update!(b, 1:length(data), data)
function update!(b::Buffer{T,N}, r::AbstractUnitRange, data::AbstractVector) where {T,N}
    last(r) <= b.capacity ||
        throw(ArgumentError("update! writes $(last(r)) elements into a capacity of $(b.capacity)"))
    upload!(b.dev, b.store, first(r), data)
    b.len = max(b.len, last(r))
    return b
end
# Immediate staged write, for a ref outside a graph. Inside one, `ref[] = x`
# lands in the run's submission, ordered against its readers.
update!(s::GPURef{T}, x) where {T} = (upload!(s.dev, s.store, 1, T[x]); s)

"""
    ref[] = x
    buf[range] = data
    buf[:] = data

Queue a store for the next plan that reads the resource; callable from any
thread, never submits. A `GPURef` keeps the latest value. Buffers and transients
keep linear range stores in order (`buf[:]` is `1:length(data)`); a `Vector{T}`
is kept without copying, so pass `copy(data)` if it changes before the run. A
transient must be declared `hostwritten = true`. The run lands stores before
readers in its own submission. `update!` uploads immediately.
"""
function Base.setindex!(r::GPURef{T}, x) where {T}
    v = convert(T, x)
    # Claim the payload: even `seq` -> odd. A concurrent setter spins on the
    # same CAS; a reader sees the odd count and waits the write out.
    while true
        s = @atomic :acquire r.seq
        iseven(s) || continue
        _, ok = @atomicreplace :sequentially_consistent :monotonic r.seq s => s + one(UInt64)
        ok && break
    end
    r.pending[] = v
    @atomic :release r.seq += one(UInt64)
    @atomic :release r.dirty = true
    return r
end

function Base.setindex!(b::Union{Buffer{T},TransientBuffer{T}}, data::AbstractVector,
                        r::AbstractUnitRange) where {T}
    # Transient stores must participate in placement before any bytes are shared.
    if b isa TransientBuffer
        b.hostwritten || throw(ArgumentError(
            "setindex!: this transient was not declared host-written, so the placer " *
            "may have aliased its bytes against something still live at the update " *
            "pass. Build it with `Transient.Buffer(g, T, dims...; hostwritten = true)`."))
    end
    rng = UnitRange{Int}(r)
    length(data) == length(rng) || throw(DimensionMismatch(
        "setindex!: $(length(data)) elements were given for a range of $(length(rng))"))
    first(rng) >= 1 || throw(ArgumentError("setindex!: a range starts at 1 or later, not $(first(rng))"))
    limit = b isa Buffer ? b.capacity : length(b)
    last(rng) <= limit || throw(ArgumentError(
        "setindex!: a store to $(last(rng)) elements into " *
        (b isa Buffer ? "a capacity of $limit" : "a transient of $limit")))
    v = data isa Vector{T} ? data : convert(Vector{T}, data)
    lock(b.pendinglock) do
        push!(b.pending, (rng, v))
        @atomic :release b.dirty = true
    end
    return b
end

Base.setindex!(b::Union{Buffer,TransientBuffer}, data::AbstractVector, ::Colon) =
    setindex!(b, data, 1:length(data))

"""Whether a resource holds a store not yet landed: one flag read."""
isdirty(r::Union{Buffer,GPURef,TransientBuffer}) = @atomic :acquire r.dirty

"""Whether any resource has pending stores. Unrolled to avoid heterogeneous-tuple
boxing on the allocation-free run path."""
@generated function anydirty(rs::T) where {T<:Tuple}
    n = fieldcount(T)
    n == 0 && return :(false)
    expr = :(isdirty(rs[1]))
    for i in 2:n
        expr = :($expr || isdirty(rs[$i]))
    end
    expr
end

"""
    landstores!(f, r)
    landstores!(f, rs::Tuple)

Consume pending stores; a clean resource costs one flag read. A `GPURef` calls
`f(r, ptr::Ptr{T})` under its seqlock, possibly more than once. Buffers and
transients call `f(resource, data, first)` in store order under their lock; if
`f` throws, the list stays dirty for a retry. The tuple form walks a plan's
`hostwritten` resources.
"""
function landstores!(f, r::GPURef{T}) where {T}
    isdirty(r) || return nothing
    # Cleared before the read: a later store sets it again for the next run,
    # whereas clearing after the read would drop a store made in between.
    @atomic :release r.dirty = false
    while true
        s1 = @atomic :acquire r.seq
        isodd(s1) && continue
        GC.@preserve r f(r, Base.unsafe_convert(Ptr{T}, r.pending))
        (@atomic :acquire r.seq) == s1 && return nothing
    end
end

function landstores!(f, b::Union{Buffer,TransientBuffer})
    isdirty(b) || return nothing
    lock(b.pendinglock) do
        lo, hi = typemax(Int), typemin(Int)
        for (i, (rng, data)) in enumerate(b.pending)
            if !isempty(rng)
                # Sorted disjoint updates avoid rescanning their predecessors.
                first(rng) <= hi && last(rng) >= lo && orderstore!(f, b, i)
                lo, hi = min(lo,first(rng)), max(hi,last(rng))
            end
            f(b, data, first(rng))
            # Persistent draws cover the stored prefix; transients keep their shape.
            b isa Buffer && (b.len = max(b.len, last(rng)))
        end
        empty!(b.pending)
        @atomic :release b.dirty = false
    end
    return nothing
end

# Synchronous callbacks already execute in order; command emitters may need a dependency.
orderstore!(f, b, i) = nothing

function overlappingstore(b, i)
    rng = b.pending[i][1]
    isempty(rng) && return false
    for j in 1:i-1
        prev = b.pending[j][1]
        isempty(prev) && continue
        max(first(prev),first(rng)) <= min(last(prev),last(rng)) && return true
    end
    return false
end

# Unrolled for the same reason as `anydirty`: a plan's `hostwritten` is walked
# on the run path, which allocates nothing.
@generated function landstores!(f, rs::T) where {T<:Tuple}
    Expr(:block, (:(landstores!(f, rs[$i])) for i in 1:fieldcount(T))..., :(nothing))
end

"""
    resize!(b::Buffer{T,1}, n) -> b

Grow to `n` elements, keeping the contents; shrinking only lowers `len`. The old
region is retired, not released, because the device copy out of it is recorded
but may not have run. `notify_move!` patches recorded plans with the new address
in their next run, so they survive the resize without re-recording.
"""
function Base.resize!(b::Buffer{T,1}, n::Integer) where {T}
    n = Int(n)
    n <= b.capacity && (b.len = min(b.len, n); return b)
    fresh = persistentarray(b.dev, T, (n,))
    b.len > 0 && devicecopy!(b.dev, fresh, b.store, b.len)
    old = b.store
    b.store, b.capacity = fresh, n
    retire!(pool(b.dev), b.dev, region(old))
    notify_move!(pool(b.dev), deviceaddress(b.dev, region(old)),
                 deviceaddress(b.dev, region(fresh)), length(old.region))
    return b
end

"""
    retireonce!(r::Union{Buffer,GPURef})

Hand the region back at most once, from anywhere including a finalizer. The CAS
keeps a dropped and explicitly freed resource from being retired twice and
handed to two owners. Calls only the finalizer-safe [`retire!`](@ref).
"""
function retireonce!(r::Union{Buffer,GPURef})
    _, won = @atomicreplace r.retired false => true
    won || return nothing
    retire!(pool(r.dev), r.dev, region(r.store))
    return nothing
end

"""
    free!(r::Union{Buffer,GPURef})

Retire a persistent resource's region now instead of at finalization. Optional,
since the finalizer does the same. No precondition: [`reclaim!`](@ref) releases
the region only once the device is done with it. The resource must not be used
afterwards; its region may belong to someone else.
"""
free!(r::Union{Buffer,GPURef}) = retireonce!(r)

"""
    devicecopy!(dev, dst::DeviceArray, src::DeviceArray, n)

Copy `n` elements between regions on the device, as `resize!` does.
Backend-supplied. Not `copy!`, which is the graph copy pass.
"""
function devicecopy! end

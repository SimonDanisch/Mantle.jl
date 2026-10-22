# Persistent resources. One implementation, every backend.
#
# One type and not a concrete one per backend holding that backend's array type:
# nothing in them is backend-specific except where the bytes come from, and once
# that is the device's `Pool` the difference disappears.
#
# The lifetime is what separates these from a transient, not the ownership. A
# transient's region is scoped to a plan and reclaimed with it; a persistent
# resource's is held until the user drops it. Both come from the same pool, which
# is the point: a model's weights and an editor's frame buffers are then visible
# to one allocator instead of two.

"""
    Buffer(dev, data; capacity = length(data)) -> Buffer
    Buffer(dev, T, n) -> Buffer
    Buffer(dev, T, dims) -> Buffer

A device array. `N` is one unless `dims` says otherwise.

`len` and `capacity` are separate so a count that changes every frame never
reallocates: `store` is always at capacity, `len` is what a draw covers. For
`N > 1` they are both `prod(dims)`: an image-shaped buffer is not a list with a
count, and `resize!` and `update!` are `N == 1` for exactly that reason.

`N` exists because a renderer's own buffers are not all vectors — a framebuffer,
an albedo layer and a depth layer are matrices, and flattening them here would
push the width back into every kernel that indexes them.
"""
mutable struct Buffer{T,N} <: Resource
    store::DeviceArray{T,N}
    len::Int
    capacity::Int
    dev::Any
    # Stores made through `setindex!` and not yet landed — `(range, data)`, in
    # the order they were made — and the flag a run reads before touching the
    # list. See `setindex!` below.
    pending::Vector{Tuple{UnitRange{Int},Vector{T}}}
    pendinglock::ReentrantLock
    @atomic dirty::Bool
    # Whether the region has been handed back, so the finalizer and an explicit
    # `free!` cannot both do it. See [`retireonce!`](@ref).
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

One value, on the device, at an address that never moves.

The thing a plan is given when the value has to change between runs. A dispatch
is packed with the ref's ADDRESS — once, at `record!`, like every other argument
— and the value behind it is written by a store, `ref[] = x`, which lands as a
command in the run's own submission. So changing it costs one small transfer however many
dispatches read it, and nothing rewrites the plan's argument memory, which is
what let the argument ring go.

Passing the value directly instead says the opposite, and that is the whole of
the distinction: what a dispatch is given by value is resolved when the plan
records and is that value for the plan's life.

A distinct type rather than a length-1 `Buffer`: "one value shared by every
element" is a claim about meaning rather than a length that happens to be one,
and it is what `stride` reads to bind a vertex attribute at stride zero.

Named for what it is, which is the storage a per-run value lives in, and not
for the attribute binding it can also be used as.
"""
mutable struct GPURef{T} <: Resource
    store::DeviceArray{T,1}
    dev::Any
    # The value of the last store, and the seqlock a run reads it under. See
    # `setindex!` below.
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

Write `data` into `dst` starting at element `first`.

The one genuinely backend-shaped thing a persistent resource needs: getting host
bytes onto the device. Everything else about `Buffer` — length, capacity, resize,
rename — is arithmetic over a region.
"""
function upload! end

"""
    download(dev, src::DeviceArray) -> Vector

`src`'s contents on the host. The inverse of [`upload!`](@ref), and the only
other place bytes cross.
"""
function download end

"""
    deviceview(dev, a::DeviceArray)

What the backend wants handed to a kernel or a library for `a`.

Two shapes exist and both are normal: a kernel takes a plain `(ptr, dims)` device
struct, while a library or a copy wants the backend's host-side array handle —
non-owning, since the `Block` owns the memory. `unsafe_wrap(…, own = false)` is
that hatch on CUDA and Metal; on a backend we control it is a view built by hand.
"""
function deviceview end

"""
    hostspan(dev, memory) -> (Ptr{UInt8}, nbytes)

Where a block of this backend's memory begins in the HOST address space, and how
many bytes there are.

This is what a backend the CPU can address directly answers: Metal from a
`Shared` `MTLBuffer`'s `contents`. Everything over it — [`hostview`](@ref)
and the three transfer verbs below — is then one implementation.

A backend whose device memory is not mapped does not answer, and writes
`upload!`, `download` and `devicecopy!` itself: Vulkan stages through a transfer
queue, which is a different operation and not a longer version of this one.
"""
function hostspan end

"""hostview(T, base, nbytes, offset, dims) -> Array{T}
    hostview(dev, a::DeviceArray) -> Array{T}

Borrow a bounded host span without copying, including padded plain-bit elements.
The caller must keep its backing pool region alive while using the returned Array."""
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
directly: a `copyto!` over [`hostview`](@ref), and on the two that READ device
memory a wait first. A backend forwards its three vocabulary methods here.

The wait is [`awaitwrites`](@ref) and it is not optional. Unified memory means
the CPU and the GPU address the same bytes; it does NOT mean a kernel writing
them has finished. Reading without waiting returns whatever was there before the
launch — silently, and correctly often enough to look fine in a test that
uploads and reads straight back. `upload!` does not wait: the graph orders a
write against the passes that read it.
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

What memory a persistent buffer of `T` needs, as whatever [`compatible`](@ref)
consumes.

Element types can demand more than the ordinary bits — a buffer of draw commands
is read by the command processor, and one without `INDIRECT_BUFFER_BIT` is a
validation error at the draw rather than at the allocation. That is a
backend's vocabulary, so the backend answers.
"""
function bufferusage end

"""Which arena persistent buffers live in. Separate from the transient arenas so
a long-lived allocation does not fragment the space plans reuse every frame."""
struct Persistent end

# `todevice` on the way in, so a caller holding the KA backend it named — which
# is what a downstream package has — gets a buffer whose stored device is a real
# one. Without it the failure surfaced two calls deep as `no method pool(::MetalBackend)`.
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

`indices` on the device, usable as an indexed draw's index buffer.

Its own name and not `Buffer`, because the two APIs genuinely differ: Vulkan
requires `BUFFER_USAGE_INDEX_BUFFER_BIT` at allocation and refuses a buffer that
lacks it, while Metal's `drawIndexedPrimitives` takes any buffer. So the default
here is what Metal needs and the Vulkan backend overrides it.

Not expressed through `extrausage`, which dispatches on the ELEMENT type: an
index happens to be a `UInt32` and so is half the data in a renderer, so giving
every `UInt32` buffer the bit would be granting a capability to answer a question
nobody asked.
"""
indexbuffer(dev::Device, indices::AbstractVector{UInt32}) = Buffer(dev, indices)
# Normalised before dispatch and not inside, because the Vulkan device OVERRIDES
# this: a backend handed in raw would have missed its own method and got the
# generic buffer, which on Vulkan is a buffer without INDEX_BUFFER_BIT.
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

"""A region of `dims` elements of `T`, from the device's pool.

`bufferusage(dev, T)` is threaded through as the block's constraint, and it was
not: `allocate` passed `nothing`, so the hook was declared, documented,
implemented by both backends — and never called. Every persistent `Buffer` got
the ordinary usage bits whatever its element type asked for.

Nothing failed visibly, because the two types that ask for more were only ever
reached by other paths. `Predicate` found it immediately: a conditional-rendering
predicate read from a buffer without `CONDITIONAL_RENDERING_BIT` is undefined,
and the two drivers here disagree about what undefined means — RADV returned the
right answer and NVIDIA hung the GPU, with `vkQueueWaitIdle` never returning.
"""
persistentarray(dev, ::Type{T}, dims::Dims) where {T} =
    allocate(pool(dev), dev, Persistent(), T, dims;
             align = 256, blocksize = blocksize(dev),
             constraint = bufferusage(dev, T))

"""
A resource's leaf is its REGION, not the device it was allocated on.

`holdleaves!`'s generic walker is documented as being for "a plain Julia value
tree", and a `Buffer` is not one: it carries `dev` so it can free itself, and
walking that reaches the `VkContext` and the `Vulkan.Instance`, whose
`destructor` captures the `Instance` back. `DeviceArray`'s own method stops the
other route into the driver; this one stops this route.

Holding the store is also the right answer on the merits. What a launch must
keep alive is the memory its commands name, and that is the region; the device
outlives every resource on it by construction.
"""
@inline holdleaves!(holder, r::Union{Buffer,GPURef}) = holdleaves!(holder, r.store)

Base.length(b::Buffer) = b.len
Base.size(b::Buffer) = size(b.store)
# The per-dimension form, which `Base` gives an `AbstractArray` for free and a
# `Buffer` is not one. `1` past the rank, as Base does, so shape arithmetic that
# indexes a fixed number of axes works on a resource of any rank.
Base.size(b::Buffer, d::Integer) = d <= ndims(b) ? size(b)[d] : 1
Base.ndims(::Buffer{T,N}) where {T,N} = N
# Element-wise verbs forward to the region. A `Buffer` is a container of `len`
# elements of `T` — it answers `size`, `length` and `eltype` — so a caller that
# has one should not have to reach for `storage` to write to it.
Base.fill!(b::Buffer, v) = (fill!(storage(b), v); b)
# `similar` answers a BUFFER, not a bare array: the allocation verb on a
# resource should give back the same kind of thing, owned by the same `free!`.
Base.similar(b::Buffer{T,N}) where {T,N} = Buffer(b.dev, T, size(b))
Base.similar(b::Buffer, ::Type{T}) where {T} = Buffer(b.dev, T, size(b))
Base.similar(b::Buffer{T}, dims::Dims) where {T} = Buffer(b.dev, T, dims)
Base.similar(b::Buffer, ::Type{T}, dims::Dims) where {T} = Buffer(b.dev, T, dims)
# The varargs spellings too, which is how most callers write it.
Base.similar(b::Buffer{T}, dims::Integer...) where {T} = Buffer(b.dev, T, Dims(dims))
Base.similar(b::Buffer, ::Type{T}, dims::Integer...) where {T} = Buffer(b.dev, T, Dims(dims))
# `reshape` and `view` answer an ARRAY while `similar` answers a `Buffer`, and
# the difference is the point: both of the first two are a second reading of the
# SAME bytes and own nothing, while `similar` allocates and must be freed.
#
# `view` here is `Base.view` and has nothing to do with Mantle's `ResourceView`,
# which is a descriptor a PASS binds — a different function with a different
# name, reached only through `viewfor`.
Base.reshape(b::Buffer, dims::Dims) = reshape(storage(b), dims)
Base.reshape(b::Buffer, dims::Integer...) = reshape(storage(b), dims)
Base.view(b::Buffer, I...) = view(storage(b), I...)
# `Base.strides`, which is NOT Mantle's own `stride` below — that one answers
# "elements to advance per index" for a binding and is a different question
# with a confusingly similar name. A dense region's strides are its shape's.
Base.strides(b::Buffer) = strides(storage(b))
Base.copyto!(dst::Buffer, src) = (copyto!(storage(dst), src); dst)
Base.copyto!(dst::AbstractArray, src::Buffer) = copyto!(dst, storage(src))
Base.length(::GPURef) = 1
# `stride` zero means "one value shared by every element", so a `GPURef` and a
# per-element `Buffer` take the same shader path and the same pipeline — see
# `stride`'s docstring in runtime/api.jl. `count` is what a draw covers, taken
# from the binding so it cannot disagree with it.
stride(::Buffer) = 1
stride(::GPURef) = 0
count(b::Buffer) = b.len
count(::GPURef) = 1
Base.eltype(::Buffer{T}) where {T} = T
Base.eltype(::GPURef{T}) where {T} = T

capacity(b::Buffer) = b.capacity
storage(b::Buffer) = deviceview(b.dev, b.store)
storage(s::GPURef) = deviceview(s.dev, s.store)
# `len` is a vector's notion — the prefix a draw covers — so only the vector
# form trims. An `N > 1` buffer comes back with its shape.
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
# The immediate write, for a ref that is not in a graph. Inside one, `ref[] = x`
# is the route: it lands in the run's submission instead of stalling for a
# staged upload, and it is ordered against the dispatches that read it.
update!(s::GPURef{T}, x) where {T} = (upload!(s.dev, s.store, 1, T[x]); s)

"""
    ref[] = x
    buf[range] = data
    buf[:] = data

Pend a store for the next plan that reads the resource. Setters may run on any
thread and never submit commands. GPURef converts one value; the latest wins,
including stores equal to the old value. Buffers/transients retain ordered linear
range stores, converting data to Vector{T} when needed. A Vector{T} is retained
without copying: pass copy(data) if the caller will mutate it before the run.
Transient stores require hostwritten=true at declaration so placement covers them.

The update pass lands stores before readers in the run's own submission;
overlapping ranges preserve store order. buf[:] means 1:length(data).
update! uploads immediately. Array(buf) explicitly downloads device contents.
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

Consume pending stores; a clean resource reads only its dirty flag.
GPURef calls f(r,ptr::Ptr{T}) under its seqlock and may repeat the callback.
Buffers/transients call f(resource,data,first) in store order under their lock;
a throwing callback leaves the pending list dirty for retry. The tuple form
walks a plan's hostwritten resources.
"""
function landstores!(f, r::GPURef{T}) where {T}
    isdirty(r) || return nothing
    # Cleared BEFORE the read. A store that lands after this line sets the flag
    # again and is landed by the next run whichever value this read saw; cleared
    # after the read, a store made between the two would be dropped.
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
Grow to `n` elements, keeping what fits.

The old region goes back AFTER the copy, so growth never trips over its own
source — and it is retired rather than released, because that copy is a DEVICE
copy: it has been recorded, not necessarily run, and handing those bytes to the
next caller on the strength of having recorded a read of them is the same
use-after-free in a different costume.

The store MOVING is announced (`notify_move!`): a recorded plan packed with
the old address is patched, not re-recorded — its next run writes the new
address into the plan's argument memory as a command in that run's own
submission. So a plan survives its buffers resizing.
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
    retireonce!(r::Buffer) / retireonce!(r::GPURef)

Hand the region back, at most once, from anywhere — including a finalizer.

The CAS is the whole of it: a resource that is dropped AND explicitly freed
retires once, and the second caller is a no-op. Without it the same region could
be pushed onto `pending` twice and handed to two owners, which is worse than
leaking it.

`retire!` is the only thing this calls, and it appends under a lock and does
nothing else — no free list, no driver, nothing a GC thread may not touch. The
release itself is [`reclaim!`](@ref), on the owning thread.
"""
function retireonce!(r::Union{Buffer,GPURef})
    _, won = @atomicreplace r.retired false => true
    won || return nothing
    retire!(pool(r.dev), r.dev, region(r.store))
    return nothing
end

"""
    free!(r::Buffer) / free!(r::GPURef)

Give a persistent resource's region back to the pool, now rather than whenever
the GC gets to it.

**Optional.** Every `Buffer` and `GPURef` carries a finalizer that does exactly
this, so a resource that is simply dropped comes back on its own and forgetting
`free!` is not a leak. What calling it buys is timing: the region is retired at
a point you chose, which matters when the next allocation is large and you would
rather not wait for a GC to notice.

**No precondition either way.** The region is RETIRED, not released: it goes back
on a free list only once [`passed`](@ref) says the device is finished with it,
which [`reclaim!`](@ref) checks. So there is no "the GPU must be idle" rule to
get wrong, and no reason to reach for a synchronize first.

Using the resource afterwards is what it always was — the region may already
belong to somebody else.

One method for both because they are one thing, a region and a length, and a
caller that owns a mix should not have to remember which is which.
"""
free!(r::Union{Buffer,GPURef}) = retireonce!(r)

"""
    devicecopy!(dev, dst::DeviceArray, src::DeviceArray, n)

`n` elements from one region to another, device-side. Used by `resize!`, and the
third and last thing a backend has to supply for persistent resources.

Not `copy!`: `Mantle.copy!` already means "a copy as a graph pass", which is a
different verb with the same spelling.
"""
function devicecopy! end

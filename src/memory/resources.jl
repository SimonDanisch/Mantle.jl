# Persistent resources. One implementation, every backend.
#
# These used to be a concrete type per backend — `LavaBuffer`, and whatever a
# second backend would have written next to it — each holding that backend's
# array type. Nothing in them was backend-specific except where the bytes came
# from, and once that is the device's `Pool` the difference disappears.
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
end

Buffer{T,N}(store, len::Int, capacity::Int, dev) where {T,N} =
    Buffer{T,N}(store, len, capacity, dev, Tuple{UnitRange{Int},Vector{T}}[],
                ReentrantLock(), false)

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

This was `Scalar`, and the rename is the content — it was described as an
attribute binding and used as one, while what it actually is is the storage a
per-run value lives in.
"""
mutable struct GPURef{T} <: Resource
    store::DeviceArray{T,1}
    dev::Any
    # The value of the last store, and the seqlock a run reads it under. See
    # `setindex!` below.
    pending::Base.RefValue{T}
    @atomic seq::UInt64
    @atomic dirty::Bool
end

GPURef{T}(store, dev) where {T} =
    GPURef{T}(store, dev, Base.RefValue{T}(), UInt64(0), false)

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
    bufferusage(dev, T) -> constraint

What memory a persistent buffer of `T` needs, as whatever [`compatible`](@ref)
consumes.

Element types can demand more than the ordinary bits — a buffer of draw commands
is read by the command processor, and one without `INDIRECT_BUFFER_BIT` is a
validation error at the draw rather than where it was allocated. That is a
backend's vocabulary, so the backend answers.
"""
function bufferusage end

"""Which arena persistent buffers live in. Separate from the transient arenas so
a long-lived allocation does not fragment the space plans reuse every frame."""
struct Persistent end

function Buffer(dev, data::AbstractVector{T}; capacity::Integer = length(data)) where {T}
    cap = max(Int(capacity), length(data))
    b = Buffer{T,1}(persistentarray(dev, T, (cap,)), length(data), cap, dev)
    isempty(data) || upload!(dev, b.store, 1, data)
    return b
end

function Buffer(dev, ::Type{T}, dims::Dims{N}) where {T,N}
    n = prod(dims)
    return Buffer{T,N}(persistentarray(dev, T, dims), n, n, dev)
end
Buffer(dev, ::Type{T}, n::Integer) where {T} = Buffer(dev, T, (Int(n),))

"`data`'s shape and contents on the device. The `N > 1` counterpart of the
vector constructor; the upload is linear, because a region is."
function Buffer(dev, data::AbstractArray{T,N}) where {T,N}
    b = Buffer(dev, T, size(data))
    isempty(data) || upload!(dev, b.store, 1, vec(collect(data)))
    return b
end

GPURef(dev, x::T) where {T} =
    (s = GPURef{T}(persistentarray(dev, T, (1,)), dev); upload!(dev, s.store, 1, [x]); s)

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

Base.length(b::Buffer) = b.len
Base.size(b::Buffer) = size(b.store)
Base.ndims(::Buffer{T,N}) where {T,N} = N
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

update!(b::Buffer{T,1}, data::AbstractVector) where {T} = update!(b, 1:length(data), data)
function update!(b::Buffer{T,1}, r::AbstractUnitRange, data::AbstractVector) where {T}
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

Store a value into a persistent resource, to be landed by the next plan that
reads it.

A store retains what it is given and marks the resource dirty. It copies
nothing, touches no command buffer and may be made from any thread, which is
what lets an observable's handler or a render loop's caller feed a plan without
holding a queue. The bytes land at the update pass of the next `run!` of any
plan that declares the resource — `use(p, x; read = true)`, an `Attribute`, a
slice — as a command in that run's own submission, ordered ahead of every pass
that reads them by the barriers the graph derives from the `CopyDst` the plan
registered for it (see `hostwritten!`).

A `GPURef` holds one pending value and the last store wins, which is what a
scalar means. A `Buffer` holds a LIST of pending `(range, data)` stores and
lands them in order, so two stores to different ranges before one run are two
writes and the second does not drop the first. `data` is retained, not copied:
a caller who means to mutate it before the next run passes `copy(data)`.
`buf[:] = data` is `buf[1:length(data)] = data`, as `update!` reads it.

Nothing is compared against the last store. A ref the device also writes — a
counter the host resets — must land a store of an equal value, so a caller who
wants to skip unchanged values compares on their side, where they know what
changed. There is no `getindex`: reading a device value is a download, and it
is spelled as one (`Array(buf)`).

`update!` is the other verb and not the same one: it uploads NOW, on the
calling thread, for a resource that is not in a graph.
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

function Base.setindex!(b::Buffer{T,1}, data::AbstractVector, r::AbstractUnitRange) where {T}
    rng = UnitRange{Int}(r)
    length(data) == length(rng) || throw(DimensionMismatch(
        "setindex!: $(length(data)) elements were given for a range of $(length(rng))"))
    first(rng) >= 1 || throw(ArgumentError("setindex!: a range starts at 1 or later, not $(first(rng))"))
    last(rng) <= b.capacity || throw(ArgumentError(
        "setindex!: a store to $(last(rng)) elements into a capacity of $(b.capacity)"))
    v = data isa Vector{T} ? data : convert(Vector{T}, data)
    lock(b.pendinglock) do
        push!(b.pending, (rng, v))
        @atomic :release b.dirty = true
    end
    return b
end

Base.setindex!(b::Buffer{T,1}, data::AbstractVector, ::Colon) where {T} =
    setindex!(b, data, 1:length(data))

"""Whether a resource holds a store not yet landed: one flag read."""
isdirty(r::Union{Buffer,GPURef}) = @atomic :acquire r.dirty

"""Whether any resource in the tuple does — a plan's `hostwritten`. Unrolled at
compile time (`@generated`), so a heterogeneous tuple costs no `Base.tail` box
on a run's allocation-free path; the plain recursion left 48 bytes on Hikari's
five-element sample tuple."""
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

Hand what is waiting on `r` to `f` and clear it; nothing but the flag read when
nothing is.

For a `GPURef`, `f(r, ptr::Ptr{T})` with a pointer to the pending value, read
in place under the seqlock — no copy of the value exists on the way, and `f`
runs again if a store lands mid-read, so it must be safe to repeat. For a
`Buffer`, `f(b, data, from)` once per pending `(range, data)`, in order, under
the store lock. The tuple form walks a plan's `hostwritten`.
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

function landstores!(f, b::Buffer{T}) where {T}
    isdirty(b) || return nothing
    lock(b.pendinglock) do
        for (rng, data) in b.pending
            f(b, data, first(rng))
            # What a draw covers, as `update!` keeps it.
            b.len = max(b.len, last(rng))
        end
        empty!(b.pending)
        @atomic :release b.dirty = false
    end
    return nothing
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

The store MOVING is announced (`resource_moved!`): a recorded plan packed with
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
    retire!(pool(b.dev), region(old))
    resource_moved!(b.dev, old, fresh)
    return b
end

"""A persistent resource's storage moved from `old` to `fresh`; a backend with
recorded plans patches their argument memory (see `notify_move!`). `nothing`
where there are no device addresses to patch."""
resource_moved!(dev, old, fresh) = nothing

"""
    free!(r::Buffer) / free!(r::GPURef)

Give a persistent resource's region back to the pool.

**No precondition.** The region is RETIRED, not released: it goes back on a free
list only once [`passed`](@ref) says the device is finished with it, which
[`reclaim!`](@ref) checks. So there is no "the GPU must be idle" rule to get
wrong, and no reason for a caller to reach for a synchronize first — which is
what every caller used to do, and every one of those was a place to forget.

Still explicit, and still never called for you: skipping it is a leak the pool
can report. Using the resource afterwards is what it always was — the region may
already belong to somebody else.

One method for both because they are one thing, a region and a length, and a
caller that owns a mix should not have to remember which is which.
"""
free!(r::Union{Buffer,GPURef}) = (retire!(pool(r.dev), region(r.store)); nothing)

"""
    devicecopy!(dev, dst::DeviceArray, src::DeviceArray, n)

`n` elements from one region to another, device-side. Used by `resize!`, and the
third and last thing a backend has to supply for persistent resources.

Not `copy!`: `Mantle.copy!` already means "a copy as a graph pass", which is a
different verb with the same spelling.
"""
function devicecopy! end

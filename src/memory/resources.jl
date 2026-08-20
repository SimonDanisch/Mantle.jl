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
end

"""
One value shared by every element.

A distinct type rather than a length-1 buffer: "shared by all" is a claim about
meaning, not a length that happens to be one.
"""
mutable struct Scalar{T} <: Resource
    store::DeviceArray{T,1}
    dev::Any
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

Scalar(dev, x::T) where {T} =
    (s = Scalar{T}(persistentarray(dev, T, (1,)), dev); upload!(dev, s.store, 1, [x]); s)

"A region of `dims` elements of `T`, from the device's pool."
persistentarray(dev, ::Type{T}, dims::Dims) where {T} =
    allocate(pool(dev), dev, Persistent(), T, dims;
             align = 256, blocksize = blocksize(dev))

Base.length(b::Buffer) = b.len
Base.size(b::Buffer) = size(b.store)
Base.ndims(::Buffer{T,N}) where {T,N} = N
Base.length(::Scalar) = 1
# `stride` zero means "one value shared by every element", so a `Scalar` and a
# per-element `Buffer` take the same shader path and the same pipeline — see
# `stride`'s docstring in runtime/api.jl. `count` is what a draw covers, taken
# from the binding so it cannot disagree with it.
stride(::Buffer) = 1
stride(::Scalar) = 0
count(b::Buffer) = b.len
count(::Scalar) = 1
Base.eltype(::Buffer{T}) where {T} = T
Base.eltype(::Scalar{T}) where {T} = T
capacity(b::Buffer) = b.capacity
storage(b::Buffer) = deviceview(b.dev, b.store)
storage(s::Scalar) = deviceview(s.dev, s.store)
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
update!(s::Scalar{T}, x) where {T} = (upload!(s.dev, s.store, 1, T[x]); s)

"""
Grow to `n` elements, keeping what fits.

The old region goes back AFTER the copy, so growth never trips over its own
source — and it is retired rather than released, because that copy is a DEVICE
copy: it has been recorded, not necessarily run, and handing those bytes to the
next caller on the strength of having recorded a read of them is the same
use-after-free in a different costume.
"""
function Base.resize!(b::Buffer{T,1}, n::Integer) where {T}
    n = Int(n)
    n <= b.capacity && (b.len = min(b.len, n); return b)
    fresh = persistentarray(b.dev, T, (n,))
    b.len > 0 && devicecopy!(b.dev, fresh, b.store, b.len)
    old = b.store
    b.store, b.capacity = fresh, n
    retire!(pool(b.dev), region(old))
    return b
end

"""
    free!(r::Buffer) / free!(r::Scalar)

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
free!(r::Union{Buffer,Scalar}) = (retire!(pool(r.dev), region(r.store)); nothing)

"""
    devicecopy!(dev, dst::DeviceArray, src::DeviceArray, n)

`n` elements from one region to another, device-side. Used by `resize!`, and the
third and last thing a backend has to supply for persistent resources.

Not `copy!`: `Mantle.copy!` already means "a copy as a graph pass", which is a
different verb with the same spelling.
"""
function devicecopy! end

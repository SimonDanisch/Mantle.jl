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

A device vector.

`len` and `capacity` are separate so a count that changes every frame never
reallocates: `store` is always at capacity, `len` is what a draw covers.
"""
mutable struct Buffer{T} <: Resource
    store::DeviceArray{T,1}
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
    b = Buffer{T}(persistentarray(dev, T, cap), length(data), cap, dev)
    isempty(data) || upload!(dev, b.store, 1, data)
    return b
end

Buffer(dev, ::Type{T}, n::Integer) where {T} =
    Buffer{T}(persistentarray(dev, T, Int(n)), Int(n), Int(n), dev)

Scalar(dev, x::T) where {T} =
    (s = Scalar{T}(persistentarray(dev, T, 1), dev); upload!(dev, s.store, 1, [x]); s)

"A region for `n` elements of `T`, from the device's pool."
persistentarray(dev, ::Type{T}, n::Integer) where {T} =
    allocate(pool(dev), dev, Persistent(), T, (Int(n),);
             align = 256, blocksize = blocksize(dev))

Base.length(b::Buffer) = b.len
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
Base.Array(b::Buffer) = download(b.dev, b.store)[1:b.len]

update!(b::Buffer, data::AbstractVector) = update!(b, 1:length(data), data)
function update!(b::Buffer{T}, r::AbstractUnitRange, data::AbstractVector) where {T}
    last(r) <= b.capacity ||
        throw(ArgumentError("update! writes $(last(r)) elements into a capacity of $(b.capacity)"))
    upload!(b.dev, b.store, first(r), data)
    b.len = max(b.len, last(r))
    return b
end
update!(s::Scalar{T}, x) where {T} = (upload!(s.dev, s.store, 1, T[x]); s)

"""
Grow to `n` elements, keeping what fits.

The old region goes back to the pool AFTER the copy, so growth never trips over
its own source — and it goes back explicitly, because nothing here is freed by a
finalizer.
"""
function Base.resize!(b::Buffer{T}, n::Integer) where {T}
    n = Int(n)
    n <= b.capacity && (b.len = min(b.len, n); return b)
    fresh = persistentarray(b.dev, T, n)
    b.len > 0 && devicecopy!(b.dev, fresh, b.store, b.len)
    old = b.store
    b.store, b.capacity = fresh, n
    release!(region(old))
    return b
end

"""
    devicecopy!(dev, dst::DeviceArray, src::DeviceArray, n)

`n` elements from one region to another, device-side. Used by `resize!`, and the
third and last thing a backend has to supply for persistent resources.

Not `copy!`: `Mantle.copy!` already means "a copy as a graph pass", which is a
different verb with the same spelling.
"""
function devicecopy! end

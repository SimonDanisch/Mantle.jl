# Moving bytes, and handing a pool region to something that expects an array.
#
# Unified memory makes most of this shorter than the Vulkan equivalent: there is
# no staging buffer, no transfer queue and no barrier between a CPU write and a
# GPU read. What does NOT change is who owns the memory — the pool does, and
# everything here borrows.

"""The bytes of a `Shared` buffer, as a host array. No copy."""
hostbytes(buf::MTL.MTLBuffer) =
    unsafe_wrap(Array, convert(Ptr{UInt8}, MTL.contents(buf)), Int(buf.length))

"""
    hostview(buf, ::Type{T}, offset, dims) -> Array{T}

`dims` elements of `T` over `buf`, starting at byte `offset`. No copy.

A typed `unsafe_wrap`, NOT `reinterpret` over a byte view. `reinterpret` refuses
a struct with padding — "Padding of type X is not compatible with type UInt8" —
and Hikari's `LightBVHNode` is 60 bytes holding 54 of fields, so every scene with
a light BVH failed on this. The host backend's `wrapbytes` avoids it the same
way, for the same reason.

`own = false`: the pool owns the buffer, as everywhere else in Mantle.
"""
function hostview(buf::MTL.MTLBuffer, ::Type{T}, offset::Integer, dims::Dims) where {T}
    need = prod(dims) * sizeof(T)
    offset + need <= Int(buf.length) || throw(ArgumentError(
        "$(join(dims, "x")) $T at offset $offset needs $need bytes, buffer has $(buf.length)"))
    return unsafe_wrap(Array, Ptr{T}(convert(Ptr{UInt8}, MTL.contents(buf)) + offset),
                       dims; own = false)
end

"""
    deviceview(dev, a) -> MtlArray

A **borrowed** `MtlArray` over `a`'s slice of its pool block.

This is the bridge the whole backend exists for: MPSGraphs, Metal's FFT and any
`MtlArray` code work on a Mantle region without knowing it is one, and without
copying. `a` keeps a suballocated view of a block the pool owns.

Borrowed is the load-bearing word. The `DataRef` finalizer frees NOTHING — the
`MTLBuffer` belongs to the pool, and a second owner would free it out from under
every other tenant of the same block. `MtlArray`'s own `offset` field is what
makes the suballocation invisible to the consumer, so nothing downstream has to
learn about regions.
"""
function deviceview(::MetalDevice, a::DeviceArray{T,N}) where {T,N}
    r = region(a)
    buf = memoryof(r)::MTL.MTLBuffer
    off = offset(a)
    ref = GPUArrays.DataRef(_ -> nothing, buf)
    return MtlArray{T,N}(ref, size(a); maxsize = Int(buf.length) - off, offset = off)
end

"""
    upload!(dev, a, first, data)

Write `data` into `a` starting at element `first`.

A plain `copyto!` into the mapped bytes, because the storage is `Shared` and the
CPU and GPU see the same memory. On a discrete part this would be a staging
buffer and a blit; here the absence of one is the point.
"""
function upload!(d::MetalDevice, a::DeviceArray{T}, first::Integer,
                 data::AbstractVector) where {T}
    r = region(a)
    buf = memoryof(r)::MTL.MTLBuffer
    v = hostview(buf, T, offset(a), (length(a),))
    copyto!(v, first, data, 1, length(data))
    return a
end

"""
    download(dev, a) -> Vector

Read `a` back to the host.

**Synchronises, and must.** Unified memory means the CPU and GPU address the
same bytes; it does NOT mean a kernel writing them has finished. Reading without
waiting returns whatever was there before the launch — silently, and correctly
often enough to look fine in a test that uploads and reads straight back.
`Mantle`'s own `Window` docstring states the rule for the frame loop: there is
no `flush` in it, "a readback synchronises itself, because it must".

This is the one place in the transfer path that waits. `upload!` does not: the
graph orders a write against the passes that read it.
"""
function download(d::MetalDevice, a::DeviceArray{T}) where {T}
    Metal.synchronize()
    buf = memoryof(region(a))::MTL.MTLBuffer
    return collect(hostview(buf, T, offset(a), (length(a),)))
end

"""
    devicecopy!(dev, dst, src, n) -> dst

Copy `n` elements device-to-device.

Through the mapped bytes rather than a blit encoder: both regions are `Shared`,
so once outstanding work has landed this is a `memmove`. `resize!` on a `Buffer` is the caller
that matters — it copies the old contents into a fresh region before releasing
the old one, and a blit there would need a submission and a wait.
"""
function devicecopy!(d::MetalDevice, dst::DeviceArray{T}, src::DeviceArray{T},
                     n::Integer) where {T}
    # Same reason `download` waits: this reads `src`'s bytes on the CPU, and a
    # kernel may still be writing them. `resize!` on a `Buffer` is the caller
    # that matters, and it copies before releasing the old region.
    Metal.synchronize()
    sb = memoryof(region(src))::MTL.MTLBuffer
    db = memoryof(region(dst))::MTL.MTLBuffer
    s = hostview(sb, T, offset(src), (length(src),))
    t = hostview(db, T, offset(dst), (length(dst),))
    copyto!(t, 1, s, 1, Int(n))
    return dst
end

"""
    bufferusage(dev, T) -> nothing

What extra a buffer of `T` must be created with.

Nothing, ever. Vulkan needs `BUFFER_USAGE_INDIRECT_BUFFER_BIT` on a buffer a
draw will read its parameters from, and that bit has to be known at creation —
which is why the portable interface has this hook at all. Metal decides what a
buffer is for when it is bound, so there is nothing to declare.
"""
bufferusage(::MetalDevice, ::Type) = nothing

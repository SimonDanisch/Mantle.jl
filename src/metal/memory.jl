# Moving bytes, and handing a pool region to something that expects an array.
#
# Unified memory makes most of this shorter than the Vulkan equivalent: there is
# no staging buffer, no transfer queue and no barrier between a CPU write and a
# GPU read. What does NOT change is who owns the memory — the pool does, and
# everything here borrows.

# Where a `Shared` buffer is in the host address space. Core builds `hostview`
# and the three transfer verbs over this — see `hostspan` in
# `memory/resources.jl`. This backend used to carry its own copy of the checked
# `unsafe_wrap`, character for character the host backend's, down to the reason
# in the comment: `reinterpret` refuses a struct with padding, and Hikari's
# `LightBVHNode` is 60 bytes holding 54 of fields.
Mantle.hostspan(::MetalDevice, buf::MTL.MTLBuffer) =
    (convert(Ptr{UInt8}, MTL.contents(buf)), Int(buf.length))

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
buffer and a blit; here the absence of one is the point, and the copy itself is
core's over `hostspan`.
"""
upload!(d::MetalDevice, a::DeviceArray, first::Integer, data::AbstractVector) =
    Mantle.hostupload!(d, a, first, data)

"""
    download(dev, a) -> Vector

Read `a` back to the host.

**Synchronises, and must** — core's `hostdownload` waits on `awaitwrites` first.
Unified memory means the CPU and GPU address the same bytes; it does NOT mean a
kernel writing them has finished.
"""
download(d::MetalDevice, a::DeviceArray) = Mantle.hostdownload(d, a)

"""
    devicecopy!(dev, dst, src, n) -> dst

Copy `n` elements device-to-device.

Through the mapped bytes rather than a blit encoder: both regions are `Shared`,
so once outstanding work has landed this is a `memmove`. `resize!` on a `Buffer`
is the caller that matters — it copies the old contents into a fresh region
before releasing the old one, and a blit there would need a submission and a
wait. Core's `hostdevicecopy!` does the waiting for the same reason `download`
does.
"""
devicecopy!(d::MetalDevice, dst::DeviceArray{T}, src::DeviceArray{T}, n::Integer) where {T} =
    Mantle.hostdevicecopy!(d, dst, src, n)

"""
    bufferusage(dev, T) -> nothing

What extra a buffer of `T` must be created with.

Nothing, ever. Vulkan needs `BUFFER_USAGE_INDIRECT_BUFFER_BIT` on a buffer a
draw will read its parameters from, and that bit has to be known at creation —
which is why the portable interface has this hook at all. Metal decides what a
buffer is for when it is bound, so there is nothing to declare.
"""
bufferusage(::MetalDevice, ::Type) = nothing

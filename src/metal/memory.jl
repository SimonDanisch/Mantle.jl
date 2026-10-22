# Transfers and array views over pool regions. The pool owns every `MTLBuffer`;
# everything here borrows.

# Host address of a `Shared` buffer. Core builds `hostview` and the transfer verbs
# over it (`hostspan` in `memory/resources.jl`).
Mantle.hostspan(::MetalDevice, buf::MTL.MTLBuffer) =
    (convert(Ptr{UInt8}, MTL.contents(buf)), Int(buf.length))

"""
    deviceview(dev, a) -> MtlArray

A borrowed `MtlArray` over `a`'s slice of its pool block, without copying, so
MPSGraph and other `MtlArray` code can run on a Mantle region. Its `DataRef`
finalizer frees nothing: the pool owns the `MTLBuffer`. The `Managed` is the
buffer's (`managedbuffer`), shared by all its views, so a host read through any
view waits for the queue that last used it.
"""
function deviceview(d::MetalDevice, a::DeviceArray{T,N}) where {T,N}
    buf = memoryof(region(a))::MTL.MTLBuffer
    return deviceslice(d, T, size(a), buf, offset(a))
end

"""
    upload!(dev, a, first, data)

Write `data` into `a` starting at element `first`: a `copyto!` into the mapped
`Shared` bytes (core's `hostupload!`). Does not wait; the graph orders it.
"""
upload!(d::MetalDevice, a::DeviceArray, first::Integer, data::AbstractVector) =
    Mantle.hostupload!(d, a, first, data)

"""
    download(dev, a) -> Vector

Read `a` back to the host. Waits on `awaitwrites` first: unified memory does not
mean a kernel writing the bytes has finished.
"""
download(d::MetalDevice, a::DeviceArray) = Mantle.hostdownload(d, a)

"""
    devicecopy!(dev, dst, src, n) -> dst

Copy `n` elements device-to-device through the mapped bytes, after `awaitwrites`.
`resize!` copies through this, where a blit would need its own submission.
"""
devicecopy!(d::MetalDevice, dst::DeviceArray{T}, src::DeviceArray{T}, n::Integer) where {T} =
    Mantle.hostdevicecopy!(d, dst, src, n)

"""
    bufferusage(dev, T) -> nothing

Always `nothing`: Metal decides a buffer's use at bind time, so no creation flag
such as Vulkan's `INDIRECT_BUFFER_BIT` exists.
"""
bufferusage(::MetalDevice, ::Type) = nothing

# The only part of a move core asks of a backend; which recorded plans hold an
# address in the moved range, and patching them, is core's.
Mantle.deviceaddress(::MetalDevice, buf::MTL.MTLBuffer) = UInt64(buf.gpuAddress)

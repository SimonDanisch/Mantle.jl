# Transient render targets. The graph places them; targets with disjoint
# lifetimes share an offset. The arena is a `MTLHeapTypePlacement` heap, not a
# buffer: a texture over buffer memory is linear and Apple GPUs cannot render
# into it, and an `Automatic` heap picks its own offsets, so deliberate aliasing
# would not happen.

"""
    imageusage(dev, T) -> MTLTextureUsage

Render target and shader read, for colour and depth alike: Metal has no depth
usage bit. Cast back because `|` on two `@cenum` values widens to `UInt64`.
"""
Mantle.imageusage(::MetalDevice, ::Type) =
    MTL.MTLTextureUsage(MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead)

# The placer reads `size`/`alignment`; `materialize!` places `desc` into the heap.
const MetalImageReq = NamedTuple{(:size, :alignment, :desc),
                                 Tuple{Int,Int,MTL.MTLTextureDescriptor}}

# The image is `nothing` until placed. A small union rather than `Any` because
# `target_view` is read while a pass is recorded.
const MetalTransientImage{T} = Mantle.TransientImage{T,MTL.MTLPixelFormat,MTL.MTLTextureUsage,
                                                     Union{Nothing,MTL.MTLTexture},
                                                     MetalImageReq}

"""
    image_descriptor(T, width, height, srgb, usage) -> MTLTextureDescriptor

A `Private` render-target descriptor (no host reads; readback blits). Hazard
tracking must stay `Tracked`: this path turns no graph transition into an
`MTLFence` or `MTLEvent`, and untracked lets the driver overlap a pass with the
one feeding it, so a readback right after a render pass on the same queue
returned the clear colour.
"""
function image_descriptor(::Type{T}, width::Int, height::Int, srgb::Bool,
                          usage::MTL.MTLTextureUsage) where {T}
    desc = MTL.MTLTextureDescriptor(pixelformat(MetalAPI(), T; srgb), width, height, false)
    desc.usage = usage
    desc.storageMode = MTL.MTLStorageModePrivate
    desc.hazardTrackingMode = MTL.MTLHazardTrackingModeTracked
    return desc
end

function Mantle.makeimage(dev::MetalDevice, ::Type{T}, width::Int, height::Int,
                          srgb::Bool, usage::MTL.MTLTextureUsage, source) where {T}
    desc = image_descriptor(T, width, height, srgb, usage)
    sa = MTL.heap_texture_size_and_align(dev.dev, desc)
    req = (; size = Int(sa.size), alignment = Int(sa.align), desc)
    return MetalTransientImage{T}(width, height, desc.pixelFormat, usage, nothing, req,
                                  typemax(Int), 0, nothing, nothing, source)
end

function Mantle.remakeimage!(dev::MetalDevice, t::MetalTransientImage{T}) where {T}
    desc = image_descriptor(T, t.width, t.height, false, t.usage)
    # `t.format` already encodes `srgb`, which the transient does not keep.
    desc.pixelFormat = t.format
    sa = MTL.heap_texture_size_and_align(dev.dev, desc)
    t.req = (; size = Int(sa.size), alignment = Int(sa.align), desc)
    t.image = nothing
    return t
end

"""
    materialize!(dev, t, heap, offset) -> MTLTexture

Place the target in the arena's heap at `offset`. `view` and `image` are the
same `MTLTexture`, so shared render-pass code can bind `target_view`. Throws
`OutOfMemoryError` if the heap cannot hold it.
"""
function Mantle.materialize!(::MetalDevice, t::MetalTransientImage, heap::MTL.MTLHeap, offset::Int)
    t.memory = heap                     # the texture outlives the call; the heap must too
    tex = MTL.MTLTexture(heap, t.req.desc, offset)
    tex === nothing && throw(OutOfMemoryError())
    t.image = tex
    t.view = tex
    return tex
end

"""
    readbackimage(f, dev, tex, T, width, height) -> Matrix{T}

Blit `tex` into a `Shared` buffer behind everything already submitted, since
`getBytes!` refuses `Private` textures; `f(cmd)` adds to the same command buffer.
Waits for completion and returns the texels with tight rows (`width * sizeof(T)`).
"""
function readbackimage(f, dev::MetalDevice, tex, ::Type{T}, width::Int, height::Int) where {T}
    row = width * sizeof(T)
    buf = MTL.MTLBuffer(dev.dev, row * height; storage = Metal.SharedStorage)
    cmd = framebuffer!(dev.dev)
    MTL.MTLBlitCommandEncoder(cmd) do enc
        MTL.append_copy!(enc, buf, 0, row, 0, tex,
                         MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(width, height, 1))
    end
    f(cmd)
    submitwait!(cmd)
    out = Matrix{T}(undef, width, height)
    unsafe_copyto!(pointer(out), convert(Ptr{T}, buf), length(out))
    return out
end

"""
    readback_target(dev, t, T) -> Matrix{T}

Copy a placed render target to the host ([`readbackimage`](@ref)).
"""
function Mantle.readback_target(dev::MetalDevice, t::MetalTransientImage{T}, ::Type) where {T}
    return readbackimage(identity, dev, t.image, T, t.width, t.height)
end

"""
    copy_target!(dev, dst, src; region = nothing)

The graph's `copy!` pass: blit a render target (or `region = (x, y, w, h)` of it)
into a device buffer without a host round trip. Rows are tight, so a reader
indexes `y * width + x`, as Vulkan's copy with a zero `bufferRowLength`.
"""
function Mantle.copy_target!(d::MetalDevice, dst, src::MetalTransientImage{T}; region = nothing) where {T}
    tex = src.image
    buf = metal_buffer(dst)
    buf === nothing &&
        error("a copy pass writes into device memory, and $(typeof(dst)) is not any")
    cmd = framebuffer!(d.dev)
    # Core checked that `dst` holds the pixels (`checkcopy`).
    x, y, w, h = region === nothing ? (0, 0, src.width, src.height) : region
    MTL.MTLBlitCommandEncoder(cmd) do enc
        MTL.append_copy!(enc, buf, byteoffset(dst), w * sizeof(T), 0, tex,
                         MTL.MTLOrigin(x, y, 0), MTL.MTLSize(w, h, 1))
    end
    # One command buffer per copy, committed here (see `framebuffer!`).
    commit!(cmd)
    return nothing
end

"""
    byteoffset(x)

Where `x` starts inside its `MTLBuffer`. A transient is a view over the arena;
writing at zero would overwrite another region.
"""
byteoffset(x::Metal.MtlArray) = pointer(x).offset
byteoffset(::MTL.MTLBuffer) = 0
# A `Commands` names a Mantle resource, not the array behind it.
byteoffset(x) = byteoffset(Mantle.storage(x))

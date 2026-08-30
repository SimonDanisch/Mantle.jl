# Transient render targets on Metal.
#
# The graph places these: two targets whose lifetimes do not overlap are given
# the same offset and share the bytes. Mantle owns that decision — what a
# backend supplies is an image object that can be asked how much room it wants,
# and a way to put one at a given offset.
#
# ── Why the image arena is a heap and the buffer arena is not ─────────────────
#
# Every other arena here suballocates a `MTLBuffer`, and a texture CAN be made
# over buffer memory (`MTLTexture(buffer, desc, offset, bytesPerRow)`). It is a
# LINEAR texture, and an Apple GPU cannot render into one — the tile memory path
# needs the driver's own swizzled layout. So the image arena is a
# `MTLHeapTypePlacement` heap instead, which is the same bargain in Metal's
# vocabulary: one device allocation, explicit offsets, resources at overlapping
# offsets alias.
#
# `Automatic` would have been the easier heap type and is the wrong one: it
# chooses its own offsets, so the placer's answer would be advisory and two
# targets it aliased deliberately might not alias at all.

"""What a render target of this element type is for, in Metal's vocabulary."""
# `|` on two `@cenum` values widens to the underlying `UInt64`, so the result is
# cast back: the usage a transient carries has to keep Metal's type or nothing
# downstream can dispatch on it.
Mantle.imageusage(::MetalDevice, ::Type) =
    MTL.MTLTextureUsage(MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead)
# Depth wants the same two. Metal has no depth-specific usage bit — the
# attachment kind is decided by the pixel format and by which slot of the render
# pass descriptor the texture is bound to, not by a flag at creation.

# Everything Metal needs to make this image again, and what the placer reads.
# `size`/`alignment` are the two fields Mantle looks at; `desc` is Metal's own
# and is what `materialize!` places into the heap.
const MetalImageReq = NamedTuple{(:size, :alignment, :desc),
                                 Tuple{Int,Int,MTL.MTLTextureDescriptor}}

# The texture does not exist until the heap does, so the image field starts
# empty. A small union rather than `Any`: `target_view` is read while a pass is
# recorded.
const MetalTransientImage{T} = Mantle.TransientImage{T,MTL.MTLPixelFormat,MTL.MTLTextureUsage,
                                                     Union{Nothing,MTL.MTLTexture},
                                                     MetalImageReq}

"""
The descriptor for a render target of `T` at this size, plus what it will cost
in a heap.

`storageMode = Private` because a render target lives in GPU memory: it is
written by the rasteriser and read by a later pass, and `Shared` would ask for
CPU-coherent memory nothing reads. Readback goes through `getBytes!`, which is
the only way off an Apple render target anyway.

`hazardTrackingMode = Tracked`, and that is not a default left alone — it was
`Untracked` here, on the reasoning that Mantle's graph had already emitted the
dependency. **It has not.** The graph's `Barriers` phase computes transitions,
and the KernelAbstractions path this backend runs on does not consume them: no
`MTLFence`, no `MTLEvent`, nothing. Untracked told the driver it was free to
overlap a pass with the one that feeds it, and it did — a readback issued right
after a render pass on the SAME queue came back with the texture untouched,
clear colour and all, once per process and then never again.

Committing to one queue orders when command buffers START; it does not stop the
GPU overlapping them when it can see no dependency. Tracking is what makes the
dependency visible. It can go back to untracked the day the backend turns
Mantle's transitions into fences, and not before.
"""
function image_descriptor(::Type{T}, width::Int, height::Int, srgb::Bool,
                          usage::MTL.MTLTextureUsage) where {T}
    desc = MTL.MTLTextureDescriptor(mtlformat(T; srgb), width, height, false)
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

function Mantle.remakeimage!(t::MetalTransientImage{T}) where {T}
    dev = Mantle.Device(MetalAPI())
    desc = image_descriptor(T, t.width, t.height, false, t.usage)
    # `srgb` is not re-derived: `t.format` already carries the answer the first
    # call reached, and asking again from a flag the transient does not keep
    # would be a second place for it to be decided.
    desc.pixelFormat = t.format
    sa = MTL.heap_texture_size_and_align(dev.dev, desc)
    t.req = (; size = Int(sa.size), alignment = Int(sa.align), desc)
    t.image = nothing
    return t
end

"""
Place the target in the arena's heap.

`view` and `image` are the same object on this backend. Vulkan distinguishes
them because a `VkImage` is bound by barriers and a `VkImageView` by draws; a
`MTLTexture` is both, and giving `target_view` an answer is what lets the shared
render-pass code bind it without asking which backend it came from.
"""
function Mantle.materialize!(t::MetalTransientImage, heap::MTL.MTLHeap, offset::Int)
    t.memory = heap                     # the texture outlives the call; the heap must too
    tex = MTL.MTLTexture(heap, t.req.desc, offset)
    tex === nothing && throw(OutOfMemoryError())
    t.image = tex
    t.view = tex
    return tex
end

"""
Copy a placed render target back to the host.

Through a blit into a shared buffer, because the arena is `Private`: a
`getBytes!` on a private texture is not something Metal will do, and making the
arena shared to allow it would put every G-buffer in CPU-coherent memory to
serve a path that runs in tests and screenshots.

The row stride is tight — `width * sizeof(T)` — so the bytes come back as a
`Matrix{T}` with no padding to strip.
"""
function Mantle.readback_target(t::MetalTransientImage{T}) where {T}
    tex = t.image
    tex === nothing &&
        error("this target has not been placed yet; readback needs a compiled plan")
    dev = Mantle.Device(MetalAPI())
    row = t.width * sizeof(T)
    buf = MTL.MTLBuffer(dev.dev, row * t.height; storage = Metal.SharedStorage)
    # Everything the frame recorded has to have run before these bytes mean
    # anything, and the blit itself is the last thing in that command buffer.
    cmd = framebuffer!(dev.dev)
    MTL.MTLBlitCommandEncoder(cmd) do enc
        MTL.append_copy!(enc, buf, 0, row, 0, tex,
                         MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(t.width, t.height, 1))
    end
    submitwait!(dev.dev)
    out = Matrix{T}(undef, t.width, t.height)
    unsafe_copyto!(pointer(out), convert(Ptr{T}, buf), length(out))
    return out
end

"""
Copy a render target into a device buffer, without touching the host.

The body of the graph's `copy!` pass, and the one route off a `Private` texture
that stays on the GPU: a compute pass reads the destination in the same frame,
so `readback_target`'s trip through shared memory would be a stall per
attachment per frame.

`bytesPerRow` is tight, so the buffer is the image in row-major order and the
kernel that reads it indexes `y * width + x` — the same layout the Vulkan side's
`vkCmdCopyImageToBuffer` produces with a zero `bufferRowLength`.
"""
function Mantle.copy_target!(d::MetalDevice, dst, src::MetalTransientImage{T}) where {T}
    tex = src.image
    tex === nothing &&
        error("copying from a target that has not been placed; the plan is not compiled")
    buf = metal_buffer(dst)
    buf === nothing &&
        error("a copy pass writes into device memory, and $(typeof(dst)) is not any")
    cmd = framebuffer!(d.dev)
    MTL.MTLBlitCommandEncoder(cmd) do enc
        MTL.append_copy!(enc, buf, byteoffset(dst), src.width * sizeof(T), 0, tex,
                         MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(src.width, src.height, 1))
    end
    # Left open, like a render pass: four consecutive copies share one command
    # buffer. See `framebuffer!`.
    return nothing
end

"""
Where a destination array starts inside its `MTLBuffer`.

A transient's storage is a VIEW over the arena, so the copy has to land at its
offset rather than at zero — writing to zero would overwrite whichever tenant
was placed first.
"""
byteoffset(x::Metal.MtlArray) = pointer(x).offset
byteoffset(::MTL.MTLBuffer) = 0
# A `Commands` names a Mantle resource, not the array behind it.
byteoffset(x) = byteoffset(Mantle.storage(x))

# Running a Mantle graph on Metal.
#
# Two methods. That is the whole compute path, and it is the measurement this
# refactor was for: the Vulkan backend's equivalent is `array/ka_backend.jl`,
# 1,122 lines, because it implements a KernelAbstractions backend from scratch —
# `LavaBackend <: KA.GPU`, `launch_config`, `mkcontext`, ndrange padding, the
# kernel object, indirect dispatch.
#
# None of that is needed here twice over. Metal.jl already IS a KA backend, so
# the launch machinery is its; and `graph/kalaunch.jl` already IS the plan
# execution, shared with the host backend, so the baking is Mantle's. What is
# genuinely this backend's is how a graph resource becomes a kernel argument,
# and whether barriers have to be emitted.

# ── 2.6: why this backend emits no barriers, measured rather than assumed ────
#
# Two deleted comments gave opposite reasons for the same behaviour, and both
# were wrong. It is not "every pooled allocation is `Shared`" (images are
# `PrivateStorage`), and it is not the driver's hazard tracking standing in for
# something unfinished.
#
# It is COMMIT ORDER. Since phase 1.2 every render and copy pass opens its own
# `MTLCommandBuffer` and commits it before the call returns, all on one queue,
# and Metal runs command buffers on a queue in the order they were committed.
# The order the graph scheduled IS the order they execute, with nothing between
# them to arrange.
#
# Measured 2026-09-08, a chained graph (render A, copy A, render B, copy B) at
# 2048x2048, median of five runs of forty frames:
#
#     Tracked      0.364 ms/frame     4194304 pixels correct
#     Untracked    0.332 ms/frame     4194304 pixels correct
#
# So hazard tracking costs about 9% and buys nothing here — but the heap stays
# `Tracked` and that is deliberate. Untracked is only safe BECAUSE of the one
# buffer per pass; if object reuse (phase 2.3) ever batches two passes into one
# command buffer again, commit order stops separating them and the 9% becomes a
# race. Flipping it is a one-line change to make once that question is settled,
# and this note is what says so.
#
# `needs_transition` answers `false` EXPLICITLY, which is what
# `sync/backend.jl:49-52` asks of a backend whose lowering emits nothing:
# reaching the same place through a generic `true` and an empty default is the
# absence of a decision rather than one.
needs_transition(::MetalAPI, ::Mantle.ResourceKind, before::Type, after::Type) = false

syncbackend(::MetalDevice) = MetalAPI()

# `resolve` is NOT overridden here. Core's default is `storage(x)`, and that is
# already right: a `Buffer`'s storage is `deviceview(dev, store)` — the borrowed
# `MtlArray` over its pool region — and a transient's is what `materialize!`
# below put in `block`. The hook exists; this backend has nothing to say to it.

"""
Give a transient its slice of the pool block.

`block` holds the borrowed `MtlArray` a launch will pass, built once here
rather than per launch — core's `storage(::TransientBuffer)` hands it straight
back.

The slab is an `MTLBuffer`, where the host backend's is a `Vector{UInt8}` and
the Vulkan backend's is a `BufferBlock`. Same hook, three storages.
"""
function materialize!(t::TransientBuffer{T}, buf::MTL.MTLBuffer, off::Int) where {T}
    ref = GPUArrays.DataRef(_ -> nothing, buf)
    t.block = MtlArray{T,1}(ref, (t.n,); maxsize = Int(buf.length) - off, offset = off)
    t.offset = off
    return t
end

"""
16 bytes, not Vulkan's 256 and not the host's cache line.

Metal has no `minStorageBufferOffsetAlignment`: a buffer binding needs only its
element type's natural alignment, and 16 covers every vector type MSL has.
Over-aligning would waste arena space the placer could have packed.
"""
alignment(::MetalDevice, ::TransientBuffer) = 16

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

# The `Barriers` phase is core's and runs here too; what it derives is lowered
# by `passbarriers`, whose default emits nothing. That is right on this device:
# Metal orders work within a command queue by submission, and every pooled
# allocation here is `Shared` on unified memory — so the scheduled order the
# graph computed IS the synchronisation, exactly as on the host backend. A
# discrete GPU reached through Metal would need `passbarriers` to become real;
# an Apple one does not.
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

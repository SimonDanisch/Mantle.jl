# Hardware ray tracing on Metal.
#
# Mantle declares the verbs in `raytracing/api.jl`; these are the Metal answers.
# The shapes line up closely with Vulkan's — build a bottom-level structure over
# geometry, build a top-level one over transformed instances of it, refit when
# things move — which is why the portable API has the shape it does.
#
# Two places where they genuinely differ, and both are ABI rather than concept:
#
#   * **The transform is transposed.** `Mat3x4f` is a row-major 3×4, which is
#     `VkTransformMatrixKHR` byte for byte. `MTLPackedFloat4x3` is four columns
#     of three — the same twelve floats in the other order. Reinterpreting gives
#     a sheared matrix and no error, so [`packedtransform`](@ref) converts.
#   * **An instance names its geometry by INDEX, not address.** Vulkan puts a
#     64-bit device address of the BLAS in the record; Metal puts a `UInt32`
#     index into the descriptor's `instancedAccelerationStructures` array. That
#     is why `VulkanInstanceRecord` stayed in the Vulkan backend rather than
#     becoming a shared type: only the leading 48 bytes are common.

"""
    packedtransform(m::Mat3x4f) -> MTL._MTLPackedFloat4x3

Convert Mantle's row-major 3×4 affine into Metal's four-columns-of-three.

Not a reinterpret. Both are twelve `Float32` and both are 48 bytes, and they are
transposes of each other: the identity is `[1,0,0,0, 0,1,0,0, 0,0,1,0]` in the
first and `[1,0,0, 0,1,0, 0,0,1, 0,0,0]` in the second. Reading one as the other
silently shears every instance, which is why this function exists and why
`test_raytracing_metal.jl` pins the identity.
"""
function packedtransform(m::Mat3x4f)
    # `m` is a column-major 4×3 whose bytes read as a row-major 3×4: element
    # (row, col) of the affine is `m[col, row]`.
    col(j) = MTL._MTLPackedFloat3(reinterpret(NTuple{12,UInt8},
                                              (m[j, 1], m[j, 2], m[j, 3])))
    MTL._MTLPackedFloat4x3((col(1), col(2), col(3), col(4)))
end

"""
    MetalTLAS

A built top-level acceleration structure, and what it took to build it.

The BLASes are held because Metal's instance descriptors reference them by index
into `instancedAccelerationStructures`: the array has to outlive the HWTLAS, and
nothing else owns it. `scratch` is kept only when the structure was built for
refit; a refittable structure may still need zero bytes of it.
"""
mutable struct MetalTLAS{Tri} <: HWTLAS{Tri}
    handle::MTL.MTLAccelerationStructure
    desc::Any                    # MTLInstanceAccelerationStructureDescriptor
    blases::Vector{MTL.MTLAccelerationStructure}  # what instances index into
    instances::MTL.MTLBuffer     # MTLAccelerationStructureInstanceDescriptor[]
    # Kept, not sized-checked: `refitScratchBufferSize` is legitimately 0 for a
    # small structure, so "no scratch" does NOT mean "cannot refit". Whether a
    # refit is allowed is what `usage` asked for, which is this flag.
    scratch::MTL.MTLBuffer
    refittable::Bool
    count::Int
end

"""
    MetalBLAS

A built bottom-level acceleration structure over one geometry.

`desc` is retained because a refit needs the descriptor the build used, and
Metal will not reconstruct it from the structure.
"""
mutable struct MetalBLAS
    handle::MTL.MTLAccelerationStructure
    desc::Any
    scratch::MTL.MTLBuffer
    refittable::Bool
end

"""
Encode `f(encoder)` into a fresh command buffer and wait for it.

Builds are one-off setup rather than per-frame work, so this synchronises rather
than threading them through the graph's timeline. When acceleration-structure
builds move into a pass, this becomes an encoder the graph hands over.
"""
function withaccelencoder(f, d::MetalDevice)
    cb = MTL.MTLCommandBuffer(d.queue)
    enc = MTL.MTLAccelerationStructureCommandEncoder(cb)
    try
        f(enc)
    finally
        close(enc)
    end
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    cb.status == MTL.MTLCommandBufferStatusCompleted ||
        error("Metal acceleration-structure build failed: $(cb.status)")
    return nothing
end

"""
    build_blas(dev, vertices; indices = nothing, refittable = false) -> MetalBLAS

Build a bottom-level structure over triangle geometry.

`vertices` is a device buffer of `Float32` triples. `refittable` asks for
`MTLAccelerationStructureUsageRefit`; without it the descriptor does not permit
a refit and Metal rejects one. Note that a refittable structure may still report
`refitScratchBufferSize == 0` — that means "no scratch needed", not "no refit
allowed", which is why the permission is tracked as a flag.
"""
function build_blas(d::MetalDevice, vertices::MTL.MTLBuffer, ntriangles::Integer;
                    stride::Integer = 12, refittable::Bool = false)
    geo = MTL.MTLAccelerationStructureTriangleGeometryDescriptor()
    geo.vertexBuffer = vertices
    geo.vertexStride = stride
    geo.triangleCount = ntriangles

    desc = MTL.MTLPrimitiveAccelerationStructureDescriptor()
    desc.geometryDescriptors = NSArray([geo])
    refittable && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = MTL.alloc_acceleration_structure(d.dev, sizes.accelerationStructureSize)
    scratch = MTL.MTLBuffer(d.dev, max(sizes.buildScratchBufferSize, 1);
                            storage = Metal.PrivateStorage)
    withaccelencoder(d) do enc
        MTL.build!(enc, accel, desc, scratch)
    end
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalBLAS(accel, desc, keep, refittable)
end

"""
    build_accel!(dev, blases, transforms; refittable = false) -> MetalTLAS

Build a top-level structure over `blases`, one instance per transform.

`transforms` are Mantle [`Mat3x4f`](@ref)s and are converted by
[`packedtransform`](@ref) — see the note at the top of this file about why they
cannot simply be copied.
"""
function build_accel!(d::MetalDevice, blases::Vector{MetalBLAS},
                      transforms::Vector{Mat3x4f}; refittable::Bool = false)
    n = length(transforms)
    n == length(blases) || throw(ArgumentError(
        "build_accel!: $(length(blases)) structures but $n transforms; each " *
        "instance names exactly one"))

    ibuf = MTL.MTLBuffer(d.dev, max(n, 1) * sizeof(MTL.MTLAccelerationStructureInstanceDescriptor);
                         storage = Metal.SharedStorage)
    ptr = convert(Ptr{MTL.MTLAccelerationStructureInstanceDescriptor}, MTL.contents(ibuf))
    for i in 1:n
        unsafe_store!(ptr, MTL.MTLAccelerationStructureInstanceDescriptor(
            packedtransform(transforms[i]),
            MTL.MTLAccelerationStructureInstanceOptionOpaque,
            UInt32(0xFF),          # cull mask: visible to every ray
            UInt32(0),             # intersection function table offset
            UInt32(i - 1)),        # INDEX, not an address — see the header
            i)
    end

    # Concretely typed: `NSArray` takes a `Vector{<:ObjectiveC.Object}`, and a
    # `Vector{Any}` of the same objects is not one.
    handles = MTL.MTLAccelerationStructure[b.handle for b in blases]
    desc = MTL.MTLInstanceAccelerationStructureDescriptor()
    desc.instanceDescriptorBuffer = ibuf
    desc.instanceCount = n
    desc.instancedAccelerationStructures = NSArray(handles)
    refittable && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = MTL.alloc_acceleration_structure(d.dev, sizes.accelerationStructureSize)
    scratch = MTL.MTLBuffer(d.dev, max(sizes.buildScratchBufferSize, 1);
                            storage = Metal.PrivateStorage)
    withaccelencoder(d) do enc
        MTL.build!(enc, accel, desc, scratch)
    end
    # `max(…, 1)`: Metal wants a buffer even when it needs no bytes of it, and
    # a zero-length `MTLBuffer` is rejected.
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalTLAS{Raycore.Triangle{UInt32}}(accel, desc, handles, ibuf,
                                              keep, refittable, n)
end

"""
    refit_tlas!(tlas, transforms)

Update `tlas` in place for instance transforms that changed.

Writes the new transforms into the instance buffer and encodes a refit. Much
cheaper than a rebuild — the point of holding a HWTLAS across frames — but it does
not re-cluster, so instances that move far enough degrade traversal until they
are rebuilt.
"""
function refit_tlas!(d::MetalDevice, tlas::MetalTLAS, transforms::Vector{Mat3x4f})
    length(transforms) == tlas.count || throw(ArgumentError(
        "refit_tlas!: built with $(tlas.count) instances, given $(length(transforms)); " *
        "a refit cannot change the instance count — rebuild instead"))
    tlas.refittable || throw(ArgumentError(
        "refit_tlas!: this structure was not built with `refittable = true`, so " *
        "its descriptor lacks `MTLAccelerationStructureUsageRefit` and Metal will " *
        "reject the refit. Rebuild instead."))

    ptr = convert(Ptr{MTL.MTLAccelerationStructureInstanceDescriptor}, MTL.contents(tlas.instances))
    for i in 1:tlas.count
        old = unsafe_load(ptr, i)
        unsafe_store!(ptr, MTL.MTLAccelerationStructureInstanceDescriptor(
            packedtransform(transforms[i]), old.options, old.mask,
            old.intersectionFunctionTableOffset, old.accelerationStructureIndex), i)
    end
    withaccelencoder(d) do enc
        MTL.refit!(enc, tlas.handle, tlas.desc, tlas.handle, tlas.scratch)
    end
    return tlas
end

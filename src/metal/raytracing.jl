# Metal's acceleration structures, for the native verbs under `raytracing/api.jl`. Two
# ABI differences from Vulkan: `MTLPackedFloat4x3` is the transpose of a row-major
# `Mat3x4f` (reinterpreting silently shears), and an instance names its BLAS by
# `UInt32` index into `instancedAccelerationStructures`, not by address. Callers
# write `Raycore.InstanceRecord`s; core's `instances_kernel!` packs them into
# [`MetalInstanceDescriptor`](@ref)s (`instancerecord` in `hwtlas.jl`).

# Metal cannot infer residency for a structure a kernel names only by its
# `gpuResourceID`, and a non-resident structure makes rays miss silently. Through
# Metal.jl's per-queue residency set rather than `add_allocation!`: the set holds
# strong references, and only allocations Metal.jl tracks leave it when freed
# (`forget_resident!`).
function make_resident!(d::MetalDevice, resources...)
    Metal.install_queue_residency!(cmdqueue(d), d.dev)
    for r in resources
        r === nothing && continue
        Metal.make_persistently_resident!(r, d.dev)
    end
    return nothing
end

"""
    MetalInstanceDescriptor

`MTLAccelerationStructureUserIDInstanceDescriptor` byte for byte, with plain
`Float32` fields so a kernel can write it (core's `instances_kernel!`, through
`instancerecord` in `hwtlas.jl`). `transform` is Metal's packed 4×3, the
transpose of a `Mat3x4f`'s storage.
"""
struct MetalInstanceDescriptor
    transform::NTuple{12,Float32}
    options::UInt32
    mask::UInt32
    intersectionFunctionTableOffset::UInt32
    accelerationStructureIndex::UInt32
    userID::UInt32
end
@assert sizeof(MetalInstanceDescriptor) == sizeof(MTL.MTLAccelerationStructureUserIDInstanceDescriptor)

"""
    MetalTLAS

A built top-level acceleration structure. `blases` keeps the structures the
instance descriptors index alive; nothing else owns them. `scratch` is the refit
scratch, at least 1 byte.
"""
mutable struct MetalTLAS <: AccelStructure
    handle::MTL.MTLAccelerationStructure
    desc::Any                    # MTLInstanceAccelerationStructureDescriptor
    blases::Vector{MTL.MTLAccelerationStructure}  # what instances index into
    # The `MtlVector{MetalInstanceDescriptor}` `desc` names, which it owns; every refit reads it.
    instances::Any
    # `refitScratchBufferSize` can be 0 for a refittable structure, so refit
    # permission is the `refittable` flag, not the scratch size.
    scratch::MTL.MTLBuffer
    refittable::Bool
    count::Int
    freed::Bool
    function MetalTLAS(handle, desc, blases, instances, scratch, refittable, count)
        t = new(handle, desc, blases, instances, scratch, refittable, count, false)
        return finalizer(unsafe_free!, t)
    end
end

"""
    MetalBLAS

A built bottom-level acceleration structure over one geometry. `desc` is kept
because a refit needs the build's descriptor.
"""
mutable struct MetalBLAS <: AccelStructure
    handle::MTL.MTLAccelerationStructure
    desc::Any
    scratch::MTL.MTLBuffer
    refittable::Bool
    # The device arrays `desc` names, which it owns: the vertices and indices, or the
    # boxes. A refit reads the first, which the caller rewrote (`refitinput`).
    inputs::Vector{Any}
    # The device it was built on, which a refit runs on.
    device::MetalDevice
    freed::Bool
    function MetalBLAS(handle, desc, scratch, refittable, inputs, device)
        b = new(handle, desc, scratch, refittable, inputs, device, false)
        return finalizer(unsafe_free!, b)
    end
end

# Free the `MTLBuffer`s a structure owns, which are unmanaged, and its inputs, once: an
# explicit release and the finalizer both come here. `Metal.free` defers past the work
# using a buffer and is finalizer-safe. The structure itself leaves the residency set
# when collected (`newstructure`), as a trace of an older structure may still hold it.
function unsafe_free!(b::MetalBLAS)
    b.freed && return nothing
    b.freed = true
    Metal.free(b.scratch)
    foreach(unsafe_free!, b.inputs)
    return nothing
end

function unsafe_free!(t::MetalTLAS)
    t.freed && return nothing
    t.freed = true
    Metal.free(t.scratch)
    unsafe_free!(t.instances)
    return nothing
end

"""
    newstructure(dev, bytes) -> MTLAccelerationStructure

An empty acceleration structure that leaves the residency set when collected.
The set holds strong references ([`make_resident!`](@ref)), so without this a
structure would never be freed. The finalizer is on the structure, not its
`MetalBLAS`, because a TLAS keeps structures alive without their `MetalBLAS`.
"""
function newstructure(d::MetalDevice, bytes::Integer)
    accel = MTL.alloc_acceleration_structure(d.dev, bytes)
    return finalizer(Metal.forget_resident!, accel)
end

"""
Build `desc` into a new structure with scratch that is freed as soon as the build has run.
"""
function buildstructure(d::MetalDevice, desc, sizes)
    accel = newstructure(d, sizes.accelerationStructureSize)
    scratch = MTL.MTLBuffer(d.dev, max(sizes.buildScratchBufferSize, 1);
                            storage = Metal.PrivateStorage)
    try
        withaccelencoder(d) do enc
            MTL.build!(enc, accel, desc, scratch)
        end
    finally
        Metal.free(scratch)
    end
    return accel
end

"""
Encode `f(encoder)` into a fresh command buffer, wait for it and throw if it
failed. Synchronous because builds are setup, outside the graph's timeline.
"""
function withaccelencoder(f, d::MetalDevice)
    cb = MTL.MTLCommandBuffer(cmdqueue(d))
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

"""Core's build context over a Metal queue. `into` is the queue's device: each build
encodes into a command buffer of its own and waits for it ([`withaccelencoder`](@ref))."""
const MetalAccelBuildContext = AccelBuildContext{<:Metal.BatchedCommandQueue,<:MetalDevice}

# Nothing orders a build's own command buffer after the work open on `bq`, which may
# write its inputs (core's instance records), so that is committed and waited for first.
function Mantle.build_accel!(f, bq::Metal.BatchedCommandQueue)
    flush!(bq)
    return f(AccelBuildContext(bq, deviceof(bq), Any[]))
end

"""
A primitive structure over the geometry `geo` names, keeping `inputs`. `refittable`
sets `MTLAccelerationStructureUsageRefit`, without which Metal rejects a refit.
"""
function newblas(d::MetalDevice, geo, refittable::Bool, inputs::Vector{Any})
    desc = MTL.MTLPrimitiveAccelerationStructureDescriptor()
    # `NSArray`, not a typed `Vector`: `@objcwrapper` types are no Julia subtype
    # hierarchy, so `convert` refuses the concrete descriptor.
    desc.geometryDescriptors = NSArray([geo])
    refittable && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)
    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = buildstructure(d, desc, sizes)
    # `max(…, 1)`: a zero-length `MTLBuffer` is rejected.
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalBLAS(accel, desc, keep, refittable, inputs, d)
end

function buildnative!(ctx::MetalAccelBuildContext, input::TriangleInput, allow_update::Bool)
    geo = MTL.MTLAccelerationStructureTriangleGeometryDescriptor()
    geo.vertexBuffer, geo.vertexBufferOffset = boundbuffer(pointer(input.vertices))
    geo.vertexStride = sizeof(NTuple{3,Float32})
    geo.indexBuffer, geo.indexBufferOffset = boundbuffer(pointer(input.indices))
    geo.indexType = MTL.MTLIndexTypeUInt32
    geo.triangleCount = length(input.indices) ÷ 3
    return newblas(ctx.into, geo, allow_update, Any[input.vertices, input.indices])
end

# `AABB` is two `Point3f`: `MTLAxisAlignedBoundingBox`'s layout, 24 bytes a box.
function buildnative!(ctx::MetalAccelBuildContext, input::BoxInput, allow_update::Bool)
    geo = MTL.MTLAccelerationStructureBoundingBoxGeometryDescriptor()
    geo.boundingBoxBuffer, geo.boundingBoxBufferOffset = boundbuffer(pointer(input.boxes))
    geo.boundingBoxStride = sizeof(AABB)
    geo.boundingBoxCount = length(input.boxes)
    return newblas(ctx.into, geo, allow_update, Any[input.boxes])
end

# The user-ID descriptor: MSL reads `userID` as `user_instance_id`, Vulkan's instance
# custom index. The structure keeps its BLASes' handles, which the descriptors index.
function buildnative!(ctx::MetalAccelBuildContext, input::InstanceInput, allow_update::Bool)
    d = ctx.into
    # Concretely typed: `NSArray` takes a `Vector{<:ObjectiveC.Object}`.
    handles = MTL.MTLAccelerationStructure[b.handle for b in input.blases]
    desc = MTL.MTLInstanceAccelerationStructureDescriptor()
    desc.instanceDescriptorType = MTL.MTLAccelerationStructureInstanceDescriptorTypeUserID
    desc.instanceDescriptorBuffer, desc.instanceDescriptorBufferOffset = boundbuffer(pointer(input.records))
    desc.instanceCount = input.n
    desc.instancedAccelerationStructures = NSArray(handles)
    allow_update && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)
    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = buildstructure(d, desc, sizes)
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    # Non-resident structures are not mapped for the dispatch: every ray misses silently.
    make_resident!(d, accel, handles...)
    return MetalTLAS(accel, desc, handles, input.records, keep, allow_update, input.n)
end

function refitnative!(ctx::MetalAccelBuildContext, as::Union{MetalBLAS,MetalTLAS})
    withaccelencoder(ctx.into) do enc
        MTL.refit!(enc, as.handle, as.desc, as.handle, as.scratch)
    end
    return as
end

refittable(as::Union{MetalBLAS,MetalTLAS}) = as.refittable
refitinput(b::MetalBLAS) = first(b.inputs)
refitinput(t::MetalTLAS) = t.instances
deviceof(b::MetalBLAS) = b.device

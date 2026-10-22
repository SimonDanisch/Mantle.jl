# Metal's acceleration structures, for the verbs in `raytracing/api.jl`. Two ABI
# differences from Vulkan: `MTLPackedFloat4x3` is the transpose of a row-major
# `Mat3x4f` (reinterpreting silently shears), and an instance names its BLAS by
# `UInt32` index into `instancedAccelerationStructures`, not by address. Callers
# write `Raycore.InstanceRecord`s; [`MetalInstanceDescriptor`](@ref) converts.

"""
    packedtransform(m::Mat3x4f) -> MTL._MTLPackedFloat4x3

Convert Mantle's row-major 3×4 affine into Metal's four columns of three. The
two 48-byte layouts are transposes; reinterpreting shears every instance.
"""
function packedtransform(m::Mat3x4f)
    # `m` is a column-major 4×3 whose bytes read as a row-major 3×4: element
    # (row, col) of the affine is `m[col, row]`.
    col(j) = MTL._MTLPackedFloat3(reinterpret(NTuple{12,UInt8},
                                              (m[j, 1], m[j, 2], m[j, 3])))
    MTL._MTLPackedFloat4x3((col(1), col(2), col(3), col(4)))
end

"""
    MetalInstanceDescriptor

`MTLAccelerationStructureUserIDInstanceDescriptor` byte for byte, with plain
`Float32` fields so a kernel can write it (`writeinstances!` in `hwtlas.jl`).
`transform` is Metal's packed 4×3, the transpose of a `Mat3x4f`'s storage.
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
    MetalInstanceDescriptor(r::Raycore.InstanceRecord, index, options, keep)

The descriptor of instance `r` of the geometry at zero-based `index` into
`instancedAccelerationStructures`. `keep` is ANDed into the mask (`0xff` shown,
`0` hidden). The custom index is truncated to 24 bits, Vulkan's width.
"""
@inline MetalInstanceDescriptor(r::Raycore.InstanceRecord, index::UInt32, options::UInt32, keep::UInt32) =
    MetalInstanceDescriptor(Tuple(transpose(r.transform)), options, r.mask & keep, UInt32(0),
                            index, r.id & 0x00ffffff)

"""
    MetalTLAS

A built top-level acceleration structure. `blases` keeps the structures the
instance descriptors index alive; nothing else owns them. `scratch` is the refit
scratch, at least 1 byte.
"""
mutable struct MetalTLAS{Tri} <: HWTLAS{Tri}
    handle::MTL.MTLAccelerationStructure
    desc::Any                    # MTLInstanceAccelerationStructureDescriptor
    blases::Vector{MTL.MTLAccelerationStructure}  # what instances index into
    # The `MtlVector{MetalInstanceDescriptor}` `desc` names; every refit reads it.
    instances::Any
    # `refitScratchBufferSize` can be 0 for a refittable structure, so refit
    # permission is the `refittable` flag, not the scratch size.
    scratch::MTL.MTLBuffer
    refittable::Bool
    count::Int
    function MetalTLAS{Tri}(handle, desc, blases, instances, scratch, refittable, count) where {Tri}
        t = new{Tri}(handle, desc, blases, instances, scratch, refittable, count)
        return finalizer(freebuffers!, t)
    end
end

"""
    MetalBLAS

A built bottom-level acceleration structure over one geometry. `desc` is kept
because a refit needs the build's descriptor.
"""
mutable struct MetalBLAS
    handle::MTL.MTLAccelerationStructure
    desc::Any
    scratch::MTL.MTLBuffer
    refittable::Bool
    # The vertex buffer `desc` names (three `Float32` per vertex), which
    # `Raycore.refit!` overwrites; `nothing` for boxes.
    vertices::Union{Nothing, MTL.MTLBuffer}
    nvertices::Int
    function MetalBLAS(handle, desc, scratch, refittable, vertices, nvertices)
        b = new(handle, desc, scratch, refittable, vertices, nvertices)
        return finalizer(freebuffers!, b)
    end
end
MetalBLAS(handle, desc, scratch, refittable) = MetalBLAS(handle, desc, scratch, refittable, nothing, 0)

"""
Finalizer: free the `MTLBuffer`s a structure owns, which are unmanaged.
`Metal.free` is finalizer-safe. A TLAS's descriptor `MtlArray` frees itself.
"""
function freebuffers!(b::MetalBLAS)
    Metal.free(b.scratch)
    b.vertices === nothing || Metal.free(b.vertices)
    return nothing
end
freebuffers!(t::MetalTLAS) = (Metal.free(t.scratch); nothing)

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

"""
    build_blas(dev, vertices, ntriangles; indices = nothing, nvertices,
               refittable = false) -> MetalBLAS

Build a bottom-level structure over triangles. `vertices` holds `nvertices`
`Float32` triples (`stride` bytes apart); `indices` (`UInt32`, three per
triangle) is optional, otherwise consecutive triples form triangles.
`refittable` sets `MTLAccelerationStructureUsageRefit`, without which Metal
rejects a refit.
"""
function build_blas(d::MetalDevice, vertices::MTL.MTLBuffer, ntriangles::Integer;
                                     stride::Integer = 12, refittable::Bool = false,
                                     indices::Union{Nothing, MTL.MTLBuffer} = nothing,
                                     nvertices::Integer = 3 * ntriangles)
    geo = MTL.MTLAccelerationStructureTriangleGeometryDescriptor()
    geo.vertexBuffer = vertices
    geo.vertexStride = stride
    geo.triangleCount = ntriangles
    if indices !== nothing
        geo.indexBuffer = indices
        geo.indexType = MTL.MTLIndexTypeUInt32
    end

    desc = MTL.MTLPrimitiveAccelerationStructureDescriptor()
    desc.geometryDescriptors = NSArray([geo])
    refittable && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = buildstructure(d, desc, sizes)
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalBLAS(accel, desc, keep, refittable, vertices, Int(nvertices))
end

"""What `build_accel!`'s `do` block receives. Each build waits on its own command
buffer ([`withaccelencoder`](@ref)), so the context is the device plus `preserves`,
host data kept alive until the GPU has read it.
"""
struct MetalAccelBuildContext
    device::MetalDevice
    preserves::Vector{Any}
end

"""
    build_accel!(f, bq::Metal.BatchedCommandQueue)

Run `f` against a build context:

    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
"""
function Mantle.build_accel!(f, bq::Metal.BatchedCommandQueue)
    # `batchqueue` does not carry its device; builds use the default `Device()`.
    ctx = MetalAccelBuildContext(Mantle.Device(), Any[])
    return f(ctx)
end

"""
    build_blas(ctx, vertices, indices; opaque = true, allow_update = false) -> MetalBLAS

The portable triangle build (`raytracing/api.jl`). `vertices` go to a `Shared`
buffer the BLAS owns for refits. `opaque` is ignored: every instance is built
with `MTLAccelerationStructureInstanceOptionOpaque`.
"""
function Mantle.build_blas(ctx::MetalAccelBuildContext, vertices::Vector{NTuple{3,Float32}},
                           indices::Vector{UInt32}; opaque::Bool = true, allow_update::Bool = false)
    d = ctx.device
    length(indices) % 3 == 0 && !isempty(indices) || throw(ArgumentError(
        "build_blas: $(length(indices)) indices are not a whole number of triangles"))
    vbuf, ibuf = GC.@preserve vertices indices (
        MTL.MTLBuffer(d.dev, sizeof(vertices), pointer(vertices); storage = Metal.SharedStorage),
        MTL.MTLBuffer(d.dev, sizeof(indices), pointer(indices); storage = Metal.SharedStorage))
    # The descriptor retains the index buffer, so this reference is freed.
    blas = build_blas(d, vbuf, length(indices) ÷ 3; indices = ibuf, nvertices = length(vertices),
                      refittable = allow_update)
    Metal.free(ibuf)
    return blas
end

"""
    Raycore.refit!(blas::MetalBLAS, vertices) -> blas

Write the moved `vertices` into the structure's vertex buffer and refit in place.
Throws `ArgumentError` without `allow_update` or for a different vertex count.
"""
function Raycore.refit!(blas::MetalBLAS, vertices::AbstractVector)
    blas.refittable && blas.vertices !== nothing || throw(ArgumentError(
        "refit!: this BLAS was not built with `allow_update = true`, so its descriptor " *
        "lacks `MTLAccelerationStructureUsageRefit`; rebuild it instead"))
    length(vertices) == blas.nvertices || throw(ArgumentError(
        "refit!: $(length(vertices)) vertices for a BLAS built over $(blas.nvertices); a " *
        "refit keeps the topology, rebuild for a different mesh"))
    flat = NTuple{3,Float32}[Tuple(Float32.(v)) for v in vertices]
    GC.@preserve flat unsafe_copyto!(convert(Ptr{NTuple{3,Float32}}, MTL.contents(blas.vertices)),
                                     pointer(flat), length(flat))
    withaccelencoder(Device(MetalAPI())) do enc
        MTL.refit!(enc, blas.handle, blas.desc, blas.handle, blas.scratch)
    end
    return blas
end

"""
    build_blas_aabb(ctx, aabbs; opaque = true) -> MetalBLAS

A bottom-level structure over procedural boxes. `opaque` is ignored, as on
Vulkan's ray-query path; it keeps the call identical across backends.
"""
function Mantle.build_blas_aabb(ctx::MetalAccelBuildContext,
                                                 aabbs::Vector{Mantle.AABB}; opaque::Bool = true)
    d = ctx.device
    n = length(aabbs)
    n > 0 || throw(ArgumentError("build_blas_aabb: no boxes to build over"))

    # `(min.xyz, max.xyz)`, stride 24, the layout of Vulkan's `VkAabbPositionsKHR`.
    boxes = Vector{Float32}(undef, 6n)
    @inbounds for (i, a) in enumerate(aabbs)
        o = 6 * (i - 1)
        boxes[o + 1] = a.min[1]; boxes[o + 2] = a.min[2]; boxes[o + 3] = a.min[3]
        boxes[o + 4] = a.max[1]; boxes[o + 5] = a.max[2]; boxes[o + 6] = a.max[3]
    end
    buf = MTL.MTLBuffer(d.dev, sizeof(boxes), pointer(boxes);
                        storage = Metal.SharedStorage)
    push!(ctx.preserves, boxes)

    geo = MTL.MTLAccelerationStructureBoundingBoxGeometryDescriptor()
    geo.boundingBoxBuffer = buf
    geo.boundingBoxStride = 24
    geo.boundingBoxCount  = n

    desc = MTL.MTLPrimitiveAccelerationStructureDescriptor()
    # `NSArray`, not a typed `Vector`: `@objcwrapper` types are no Julia subtype
    # hierarchy, so `convert` refuses the concrete descriptor.
    desc.geometryDescriptors = NSArray([geo])

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = buildstructure(d, desc, sizes)
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    # The geometry descriptor retains the box buffer for the `MetalBLAS`'s
    # lifetime, so this reference is freed.
    blas = MetalBLAS(accel, desc, keep, false)
    Metal.free(buf)
    return blas
end

"""
    build_accel!(dev, blases, descriptors, n; refittable = false) -> MetalTLAS

Build a top-level structure over `n` instances already written into
`descriptors` (a `MtlVector{MetalInstanceDescriptor}`), each indexing `blases`.
The structure keeps `descriptors` for refits.
"""
function build_accel!(d::MetalDevice, blases::Vector{MetalBLAS}, descriptors::MtlArray,
                                       n::Integer; refittable::Bool = false)
    eltype(descriptors) === MetalInstanceDescriptor || throw(ArgumentError(
        "build_accel!: descriptors must be MetalInstanceDescriptor, got $(eltype(descriptors))"))
    n <= length(descriptors) || throw(ArgumentError(
        "build_accel!: $n instances but $(length(descriptors)) descriptors"))
    # Concretely typed: `NSArray` takes a `Vector{<:ObjectiveC.Object}`.
    handles = MTL.MTLAccelerationStructure[b.handle for b in blases]
    buf, off = mtlbuffer(descriptors)
    desc = MTL.MTLInstanceAccelerationStructureDescriptor()
    # The user-ID descriptor: MSL reads `userID` as `user_instance_id`, Vulkan's
    # instance custom index.
    desc.instanceDescriptorType = MTL.MTLAccelerationStructureInstanceDescriptorTypeUserID
    desc.instanceDescriptorBuffer = buf
    desc.instanceDescriptorBufferOffset = off
    desc.instanceCount = n
    desc.instancedAccelerationStructures = NSArray(handles)
    refittable && (desc.usage = MTL.MTLAccelerationStructureUsageRefit)

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = buildstructure(d, desc, sizes)
    # `max(…, 1)`: a zero-length `MTLBuffer` is rejected.
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalTLAS{Raycore.Triangle{UInt32}}(accel, desc, handles, descriptors,
                                              keep, refittable, Int(n))
end

"""
    refit_tlas!(dev, tlas)

Refit `tlas` in place from its rewritten instance descriptors (same instances,
new transforms, ids and masks). A refit does not re-cluster, so instances that
move far degrade traversal until a rebuild.
"""
function refit_tlas!(d::MetalDevice, tlas::MetalTLAS)
    tlas.refittable || throw(ArgumentError(
        "refit_tlas!: this structure was not built with `refittable = true`, so " *
        "its descriptor lacks `MTLAccelerationStructureUsageRefit` and Metal will " *
        "reject the refit. Rebuild instead."))
    withaccelencoder(d) do enc
        MTL.refit!(enc, tlas.handle, tlas.desc, tlas.handle, tlas.scratch)
    end
    return tlas
end

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
#     is why neither driver record is the shared type: callers write
#     `Raycore.InstanceRecord`s, and each backend writes its own from those and
#     from the geometry a batch instances ([`MetalInstanceDescriptor`](@ref)).

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
    MetalInstanceDescriptor

`MTLAccelerationStructureUserIDInstanceDescriptor` byte for byte, with the
transform as twelve plain `Float32`s, so a KERNEL can write one: the instance
descriptors a top-level structure is built and refit from are written on the
device from the batches' `Raycore.InstanceRecord`s (`writeinstances!` in
`hwtlas.jl`). `transform` is Metal's packed 4×3, four columns of three — the
transpose of a `Mat3x4f`'s storage, see [`packedtransform`](@ref).
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

The descriptor of the instance `r` of the geometry at `index` (zero-based, into
the structure's `instancedAccelerationStructures`). `keep` is ANDed into the
mask: `0xff` for a shown batch, `0` for a hidden one. The custom index keeps its
low 24 bits and the mask its low 8, the widths Vulkan has, so a record means
the same on both.
"""
@inline MetalInstanceDescriptor(r::Raycore.InstanceRecord, index::UInt32, options::UInt32, keep::UInt32) =
    MetalInstanceDescriptor(Tuple(transpose(r.transform)), options, r.mask & keep, UInt32(0),
                            index, r.id & 0x00ffffff)

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
    # The descriptors it was built from and every refit reads, written on the
    # device: a `MtlVector{MetalInstanceDescriptor}`, which `desc` names.
    instances::Any
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
    # The vertex buffer `desc` names, three `Float32` per vertex, which a refit
    # writes the moved vertices into (`Raycore.refit!`); `nothing` for boxes.
    vertices::Union{Nothing, MTL.MTLBuffer}
    nvertices::Int
end
MetalBLAS(handle, desc, scratch, refittable) = MetalBLAS(handle, desc, scratch, refittable, nothing, 0)

"""
Encode `f(encoder)` into a fresh command buffer and wait for it.

Builds are one-off setup rather than per-frame work, so this synchronises rather
than threading them through the graph's timeline. When acceleration-structure
builds move into a pass, this becomes an encoder the graph hands over.
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

Build a bottom-level structure over triangle geometry.

`vertices` is a device buffer of `Float32` triples, `nvertices` of them; with
`indices` (a buffer of `UInt32`, three per triangle) they are indexed, without it
every three consecutive vertices are a triangle. `refittable` asks for
`MTLAccelerationStructureUsageRefit`; without it the descriptor does not permit
a refit and Metal rejects one. Note that a refittable structure may still report
`refitScratchBufferSize == 0` — that means "no scratch needed", not "no refit
allowed", which is why the permission is tracked as a flag.
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
    accel = MTL.alloc_acceleration_structure(d.dev, sizes.accelerationStructureSize)
    scratch = MTL.MTLBuffer(d.dev, max(sizes.buildScratchBufferSize, 1);
                            storage = Metal.PrivateStorage)
    withaccelencoder(d) do enc
        MTL.build!(enc, accel, desc, scratch)
    end
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    return MetalBLAS(accel, desc, keep, refittable, vertices, Int(nvertices))
end

# ── Procedural geometry ──────────────────────────────────────────────────────
#
# A BLAS of axis-aligned boxes rather than triangles. What is INSIDE a box is not
# the traversal's business: a ray that enters one is handed to an intersection
# function, which answers whether and where it really hit. `Hikari`'s `FEMMaterial`
# builds one box per curved element and Newton-solves the isoparametric map there,
# which is how the traced half gets an exact silhouette with no triangles at all.

"""What `build_accel!`'s `do` block is handed on this backend.

The Vulkan side passes a context carrying its one-shot encoder, because every build
there goes into one submission on the timeline. This backend's
[`withaccelencoder`](@ref) opens and waits for a command buffer per build, so there
is nothing to thread through but the device — and `preserves` for anything the
caller needs kept alive until the GPU has read it.
"""
struct MetalAccelBuildContext
    device::MetalDevice
    preserves::Vector{Any}
end

"""
    build_accel!(f, bq::Metal.BatchedCommandQueue)

Run `f` against a build context, the portable spelling `Hikari` uses:

    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
"""
function Mantle.build_accel!(f, bq::Metal.BatchedCommandQueue)
    # One device per process on this backend, and `batchqueue` is derived FROM it
    # rather than carrying it, so there is nothing to invert -- `Device()` is the
    # portable spelling for the one that exists.
    ctx = MetalAccelBuildContext(Mantle.Device(), Any[])
    return f(ctx)
end

"""
    build_blas(ctx, vertices, indices; opaque = true, allow_update = false) -> MetalBLAS

The portable triangle build (see `raytracing/api.jl`): `vertices` uploaded to a
shared buffer, which a refit later writes into, and `indices` as an index buffer.
`opaque` is accepted and has nothing to do: every instance is built with
`MTLAccelerationStructureInstanceOptionOpaque`, which makes its geometry opaque.
"""
function Mantle.build_blas(ctx::MetalAccelBuildContext, vertices::Vector{NTuple{3,Float32}},
                           indices::Vector{UInt32}; opaque::Bool = true, allow_update::Bool = false)
    d = ctx.device
    length(indices) % 3 == 0 && !isempty(indices) || throw(ArgumentError(
        "build_blas: $(length(indices)) indices are not a whole number of triangles"))
    vbuf, ibuf = GC.@preserve vertices indices (
        MTL.MTLBuffer(d.dev, sizeof(vertices), pointer(vertices); storage = Metal.SharedStorage),
        MTL.MTLBuffer(d.dev, sizeof(indices), pointer(indices); storage = Metal.SharedStorage))
    # The descriptor keeps both buffers; the BLAS keeps the descriptor.
    return build_blas(d, vbuf, length(indices) ÷ 3; indices = ibuf, nvertices = length(vertices),
                      refittable = allow_update)
end

"""
    Raycore.refit!(blas::MetalBLAS, vertices) -> blas

Write the moved `vertices` (as many as it was built with) into the structure's
vertex buffer and refit it in place. An `ArgumentError` for a structure built
without `allow_update`, or a different vertex count.
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

A bottom-level structure over procedural boxes.

`opaque` is accepted and ignored, as it is on the Vulkan side for a ray query: a box
is never opaque in the sense a triangle is, because there is nothing to intersect
until an intersection function says so. The flag exists so a caller spells the build
the same way on both backends.
"""
function Mantle.build_blas_aabb(ctx::MetalAccelBuildContext,
                                aabbs::Vector{Mantle.AABB}; opaque::Bool = true)
    d = ctx.device
    n = length(aabbs)
    n > 0 || throw(ArgumentError("build_blas_aabb: no boxes to build over"))

    # `(min.xyz, max.xyz)` as six Float32, stride 24 -- the same packing Vulkan's
    # `VkAabbPositionsKHR` uses, so the two backends read one buffer layout.
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
    # `NSArray`, not a typed `Vector`: the `@objcwrapper` types are not a Julia
    # subtype hierarchy, so a `Vector{MTLAccelerationStructureGeometryDescriptor}`
    # cannot hold the concrete descriptor and `convert` refuses it.
    desc.geometryDescriptors = NSArray([geo])

    sizes = MTL.accelerationStructureSizes(d.dev, desc)
    accel = MTL.alloc_acceleration_structure(d.dev, sizes.accelerationStructureSize)
    scratch = MTL.MTLBuffer(d.dev, max(sizes.buildScratchBufferSize, 1);
                            storage = Metal.PrivateStorage)
    withaccelencoder(d) do enc
        MTL.build!(enc, accel, desc, scratch)
    end
    keep = MTL.MTLBuffer(d.dev, max(sizes.refitScratchBufferSize, 1);
                         storage = Metal.PrivateStorage)
    # The box buffer has to outlive the build: the structure references it.
    blas = MetalBLAS(accel, desc, keep, false)
    push!(ctx.preserves, buf)
    return blas
end

"""
    build_accel!(dev, blases, descriptors, n; refittable = false) -> MetalTLAS

Build a top-level structure over `n` instances whose descriptors a kernel has
already written into `descriptors` (a `MtlVector{MetalInstanceDescriptor}`),
each naming its geometry by index into `blases`. The structure keeps
`descriptors`: a refit reads them again.
"""
function build_accel!(d::MetalDevice, blases::Vector{MetalBLAS}, descriptors::MtlArray,
                      n::Integer; refittable::Bool = false)
    eltype(descriptors) === MetalInstanceDescriptor || throw(ArgumentError(
        "build_accel!: descriptors must be MetalInstanceDescriptor, got $(eltype(descriptors))"))
    n <= length(descriptors) || throw(ArgumentError(
        "build_accel!: $n instances but $(length(descriptors)) descriptors"))
    # Concretely typed: `NSArray` takes a `Vector{<:ObjectiveC.Object}`, and a
    # `Vector{Any}` of the same objects is not one.
    handles = MTL.MTLAccelerationStructure[b.handle for b in blases]
    buf, off = _mtlbuffer(descriptors)
    desc = MTL.MTLInstanceAccelerationStructureDescriptor()
    # The USER-ID descriptor, always, rather than the plain one: it is the plain
    # one plus a `userID`, which is what MSL reads back as `user_instance_id` —
    # Vulkan's instance custom index.
    desc.instanceDescriptorType = MTL.MTLAccelerationStructureInstanceDescriptorTypeUserID
    desc.instanceDescriptorBuffer = buf
    desc.instanceDescriptorBufferOffset = off
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
    return MetalTLAS{Raycore.Triangle{UInt32}}(accel, desc, handles, descriptors,
                                              keep, refittable, Int(n))
end

"""
    refit_tlas!(dev, tlas)

Refit `tlas` in place from its instance descriptors, which a kernel has
rewritten since the build: new transforms, ids and masks, the same instances.

Much cheaper than a rebuild — the point of holding a HWTLAS across frames — but it
does not re-cluster, so instances that move far enough degrade traversal until
they are rebuilt.
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

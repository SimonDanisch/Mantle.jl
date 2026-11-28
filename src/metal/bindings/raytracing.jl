# Acceleration structures. Metal.jl's generated `libmtl.jl` declares every
# acceleration-structure class and property (`vertexBuffer`, `triangleCount`,
# `instanceDescriptorBuffer`, ...); these are the calls it does not wrap: making a
# descriptor, sizing and allocating a structure, and encoding a build or a refit.
#
#     desc  = descriptor(MTL.MTLPrimitiveAccelerationStructureDescriptor)
#     desc.geometryDescriptors = ...
#     sizes = acceleration_structure_sizes(dev, desc)
#     accel = acceleration_structure(dev, sizes.accelerationStructureSize)
#     enc   = acceleration_structure_encoder(cmdbuf)
#     build!(enc, accel, desc, scratch)
#     close(enc)

"""
    descriptor(T) -> T

A new descriptor of acceleration-structure class `T`, from the class's `descriptor`
method, the documented way to make one (there is no `alloc`/`init` pair):

- `MTLAccelerationStructureTriangleGeometryDescriptor`: triangles for a bottom-level
  structure; set `vertexBuffer`, `vertexStride`, `triangleCount`, and
  `indexBuffer`/`indexType` if indexed.
- `MTLAccelerationStructureBoundingBoxGeometryDescriptor`: procedural geometry, boxes
  of six `Float32` (`min.xyz, max.xyz`, stride 24) whose contents an intersection
  query answers; set `boundingBoxBuffer`, `boundingBoxStride`, `boundingBoxCount`.
- `MTLPrimitiveAccelerationStructureDescriptor`: a bottom-level structure; set
  `geometryDescriptors`.
- `MTLInstanceAccelerationStructureDescriptor`: a top-level structure; set
  `instanceDescriptorBuffer`, `instanceCount`, `instancedAccelerationStructures`.
"""
function descriptor end

descriptor(::Type{MTL.MTLAccelerationStructureTriangleGeometryDescriptor}) =
    @objc [MTLAccelerationStructureTriangleGeometryDescriptor descriptor]::MTL.MTLAccelerationStructureTriangleGeometryDescriptor
descriptor(::Type{MTL.MTLAccelerationStructureBoundingBoxGeometryDescriptor}) =
    @objc [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor]::MTL.MTLAccelerationStructureBoundingBoxGeometryDescriptor
descriptor(::Type{MTL.MTLPrimitiveAccelerationStructureDescriptor}) =
    @objc [MTLPrimitiveAccelerationStructureDescriptor descriptor]::MTL.MTLPrimitiveAccelerationStructureDescriptor
descriptor(::Type{MTL.MTLInstanceAccelerationStructureDescriptor}) =
    @objc [MTLInstanceAccelerationStructureDescriptor descriptor]::MTL.MTLInstanceAccelerationStructureDescriptor

"""
    acceleration_structure_sizes(mtldevice, descriptor) -> MTLAccelerationStructureSizes

How large a structure `descriptor` needs, and how much scratch to build it with:
`accelerationStructureSize`, `buildScratchBufferSize` (released once the build ran)
and `refitScratchBufferSize`, which a later [`refit!`](@ref) needs.
"""
acceleration_structure_sizes(dev::MTL.MTLDevice, desc::MTL.MTLAccelerationStructureDescriptorLike) =
    @objc [dev::id{MTL.MTLDevice} accelerationStructureSizesWithDescriptor:desc::id{MTL.MTLAccelerationStructureDescriptor}]::MTL.MTLAccelerationStructureSizes

"""
    acceleration_structure(mtldevice, bytes) -> MTLAccelerationStructure

An empty structure of `bytes`, sized by [`acceleration_structure_sizes`](@ref), to build into.
Its contents are the driver's own layout.
"""
acceleration_structure(dev::MTL.MTLDevice, bytes::Integer) =
    @objc [dev::id{MTL.MTLDevice} newAccelerationStructureWithSize:bytes::UInt64]::MTL.MTLAccelerationStructure

"""
    acceleration_structure_encoder(cmdbuf) -> MTLAccelerationStructureCommandEncoder

An encoder for builds and refits, ended with `close`.
"""
acceleration_structure_encoder(cb::MTL.MTLCommandBufferLike) =
    @objc [cb::id{MTL.MTLCommandBuffer} accelerationStructureCommandEncoder]::MTL.MTLAccelerationStructureCommandEncoder

"""
    build!(enc, accel, descriptor, scratch, offset = 0)

Encode a build of `descriptor` into `accel`, with `scratch` as working memory.
"""
build!(enc::MTL.MTLAccelerationStructureCommandEncoder, accel::MTL.MTLAccelerationStructure,
       desc::MTL.MTLAccelerationStructureDescriptorLike, scratch::MTL.MTLBuffer, offset::Integer = 0) =
    @objc [enc::id{MTL.MTLAccelerationStructureCommandEncoder} buildAccelerationStructure:accel::id{MTL.MTLAccelerationStructure}
                                                               descriptor:desc::id{MTL.MTLAccelerationStructureDescriptor}
                                                               scratchBuffer:scratch::id{MTL.MTLBuffer}
                                                               scratchBufferOffset:offset::UInt64]::Nothing

"""
    refit!(enc, src, descriptor, dst, scratch, offset = 0)

Encode a refit of `src` into `dst` (which may be `src`) for geometry that moved. Much
cheaper than a rebuild, and it does not re-cluster, so geometry that moves far
degrades traversal until rebuilt. `descriptor` must allow
`MTLAccelerationStructureUsageRefit`.
"""
refit!(enc::MTL.MTLAccelerationStructureCommandEncoder, src::MTL.MTLAccelerationStructure,
       desc::MTL.MTLAccelerationStructureDescriptorLike, dst::MTL.MTLAccelerationStructure,
       scratch::MTL.MTLBuffer, offset::Integer = 0) =
    @objc [enc::id{MTL.MTLAccelerationStructureCommandEncoder} refitAccelerationStructure:src::id{MTL.MTLAccelerationStructure}
                                                               descriptor:desc::id{MTL.MTLAccelerationStructureDescriptor}
                                                               destination:dst::id{MTL.MTLAccelerationStructure}
                                                               scratchBuffer:scratch::id{MTL.MTLBuffer}
                                                               scratchBufferOffset:offset::UInt64]::Nothing

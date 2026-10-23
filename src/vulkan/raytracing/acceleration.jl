# Native AS construction; allocation holds and retirement belong to core.

"""Native triangle or procedural BLAS. Core retires its handle and owned buffers
after their last submission. `geom`, `count` and `geo_flags` are what it was built
from, so a refit (MODE_UPDATE_KHR) rebuilds it in place; `update_scratch` is that
refit's scratch size, `nothing` unless built with `allow_update`."""
mutable struct LavaBLAS
    accel::VK.AccelerationStructureKHR
    # Core tracks AS last use and destruction through its backing storage.
    storage::LavaArray{UInt8, 1}
    address::UInt64  # AS device address (for HWTLAS instances)
    # Input arrays this BLAS owns: the vertices (as bytes) and indices, or the
    # boxes. Refits rewrite the first in place.
    preserves::Vector{LavaArray}
    geom::GeometryType
    count::UInt32
    geo_flags::UInt32
    update_scratch::Union{Nothing,UInt64}
end

"""Native instance hierarchy retaining referenced BLASes and its instance records
(`preserves`), with the refit state of a [`LavaBLAS`](@ref). Core retires its
descriptor pools, handle and storage after their last use."""
mutable struct LavaTLAS
    accel::VK.AccelerationStructureKHR
    storage::LavaArray{UInt8, 1}
    blases::Vector{LavaBLAS}
    preserves::Vector{LavaArray}
    geom::GeometryType
    count::UInt32
    geo_flags::UInt32
    update_scratch::Union{Nothing,UInt64}
    # Layout-handle → descriptor pool/set. Destroy each pool before its AS.
    desc_sets::Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}
end

# Retire AS handles through the same stamp/recording holds as backing storage.

function unsafe_free!(as::Union{LavaBLAS, LavaTLAS})
    # Explicit release and finalization may both request retirement.
    as.storage.buf.freed && return
    ctx = (as.storage.buf[]::VkManagedBuffer).ctx::LavaDevice
    # A lost device signals nothing ever again, so nothing can be waited for and
    # the handles are dropped as they are.
    device_lost(ctx) && return destroy_now!(as)
    retire!(ctx.default_bq, as)
    return
end

# An AS and its storage are the same bytes, so it uses the storage's stamp.
stampof(as::Union{LavaBLAS, LavaTLAS}) = stampof(as.storage)

function destroy_now!(as::Union{LavaBLAS, LavaTLAS})
    # Retired descriptor pools are destroyed before the AS they reference.
    if as isa LavaTLAS
        for (_, (pool, _)) in as.desc_sets
            try
                pool.destructor()
            catch ex
                # Report destruction failures without throwing from retirement.
                safe_fin_log("Lava LavaTLAS destroy_now!: descriptor pool destructor failed: " * sprint(showerror, ex) * "\n")
            end
        end
        empty!(as.desc_sets)
    end
    as.accel.destructor()
    # Release each owned DataRef after core drops submission/recording holds.
    unsafe_free!(as.storage)
    for p in as.preserves
        p isa LavaArray && unsafe_free!(p)
    end
    empty!(as.preserves)
    as isa LavaTLAS && empty!(as.blases)
end

# A TLAS use stamps its own storage and every referenced BLAS.

function syncbuf!(owner::Closed, t::LavaTLAS)
    syncbuf!(owner, t.storage)
    for b in t.blases
        syncbuf!(owner, b.storage)
    end
    return nothing
end
syncbuf!(owner::Closed, b::LavaBLAS) = syncbuf!(owner, b.storage)

"""AccelBuildContext over a Vulkan submit channel. Core holds and stamps named
resources; `preserves` keeps scratch and temporary host objects until completion."""
const VulkanAccelBuildContext = AccelBuildContext{SubmitChannel{VulkanQueue{LavaDevice},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}}

@inline buildcmd(ctx::VulkanAccelBuildContext)   = ctx.into.cmd
@inline as_device(ctx::VulkanAccelBuildContext)  = vkdevice(ctx.bq)

# Build inputs participate in core's cross-channel waits and retirement holds.
@inline holdbuild!(ctx::VulkanAccelBuildContext,x) = (hold!(ctx.into,x); nothing)
@inline function holdbuild!(ctx::VulkanAccelBuildContext,as::Union{LavaBLAS,LavaTLAS})
    hold!(ctx.into,as)
    for input in as.preserves
        hold!(ctx.into,input)
    end
    return nothing
end

"""Make kernel, transfer and prior AS writes visible to a build/refit."""
function asbuildbarrier!(ctx::VulkanAccelBuildContext)
    memorybarrier!(buildcmd(ctx),deviceof(ctx.bq),
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                   VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK.PIPELINE_STAGE_TRANSFER_BIT,
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                   VK.ACCESS_SHADER_WRITE_BIT | VK.ACCESS_TRANSFER_WRITE_BIT,
                   VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
    return ctx
end

"""The instance records of a TLAS build, by device address."""
struct VulkanInstancesGeometry <: GeometryType
    address::UInt64
end

astype(::VulkanInstancesGeometry) = UInt32(0)   # TOP_LEVEL
astype(::GeometryType) = UInt32(1)              # BOTTOM_LEVEL
buildflags(allow_update::Bool) =
    UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
    (allow_update ? UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) : UInt32(0))

"""Device address of `n` scratch bytes for one build, released after its submission."""
function buildscratch!(ctx::VulkanAccelBuildContext, n)
    scratch = LavaArray{UInt8,1}(undef, (max(Int(n), 16),); ctx.bq,
                                 extra_usage = AS_SCRATCH_USAGE, scratch = true)
    push!(ctx.preserves, scratch)
    holdbuild!(ctx, scratch)
    return bda_address(scratch)
end

"""
    buildstructure!(ctx, geom, count, geo_flags, allow_update) -> (accel, storage, address, update_scratch)

Create the acceleration structure for `count` primitives of `geom` and record its
build after the writes of its inputs.
"""
function buildstructure!(ctx::VulkanAccelBuildContext, geom::GeometryType, count::Integer,
                         geo_flags::UInt32, allow_update::Bool)
    dev = as_device(ctx)
    type, flags = astype(geom), buildflags(allow_update)
    sizes = query_as_build_sizes(dev, geom; as_type = type, build_flags = flags, geo_flags,
                                 max_primitive_count = UInt32(count))
    storage = LavaArray{UInt8,1}(undef, (max(Int(sizes.acceleration_structure_size), 16),);
                                 ctx.bq, extra_usage = AS_STORAGE_USAGE)
    accel = VK.AccelerationStructureKHR(dev, VK.AccelerationStructureCreateInfoKHR(
        storage.buf[].buffer, UInt64(0), sizes.acceleration_structure_size,
        type == 0 ? VK.ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR : VK.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR))
    asbuildbarrier!(ctx)
    build_as_on_gpu(ctx, accel, buildscratch!(ctx, sizes.build_scratch_size), geom;
                    as_type = type, build_flags = flags, geo_flags, primitive_count = UInt32(count))
    address = VK.get_acceleration_structure_device_address_khr(dev,
        VK.AccelerationStructureDeviceAddressInfoKHR(accel))
    return accel, storage, address, allow_update ? UInt64(sizes.update_scratch_size) : nothing
end

"""Refuse a refit of a structure built without `allow_update`, before its inputs change."""
refitcapable(as::Union{LavaBLAS,LavaTLAS}) = as.update_scratch === nothing && throw(ArgumentError(
    "refit: the structure was built with allow_update = false; rebuild it with allow_update = true."))

"""Refit `as` in place (MODE_UPDATE_KHR) from its inputs, which the caller rewrote."""
function refitstructure!(ctx::VulkanAccelBuildContext, as::Union{LavaBLAS,LavaTLAS})
    refitcapable(as)
    holdbuild!(ctx, as)
    asbuildbarrier!(ctx)
    build_as_on_gpu(ctx, as.accel, buildscratch!(ctx, as.update_scratch), as.geom;
                    as_type = astype(as.geom), build_flags = buildflags(true), geo_flags = as.geo_flags,
                    primitive_count = as.count, mode = UInt32(1), src_as = as.accel)
    return as
end

function newblas!(ctx::VulkanAccelBuildContext, geom::GeometryType, count::Integer, opaque::Bool,
                  allow_update::Bool, inputs::Vector{LavaArray})
    geo_flags = opaque ? UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR) : UInt32(0)
    accel, storage, address, update = buildstructure!(ctx, geom, count, geo_flags, allow_update)
    blas = LavaBLAS(accel, storage, address, inputs, geom, UInt32(count), geo_flags, update)
    finalizer(unsafe_free!, blas)
    holdbuild!(ctx, blas)
    return blas
end

"""build_blas(ctx, vertices, indices; opaque, allow_update) -> LavaBLAS

Record triangle geometry inside build_accel!. LavaArray inputs are read in
place and require AS_INPUT_USAGE; Vector inputs are uploaded first. Core tracks
input dependencies and retains resources through the submission.
"""
function build_blas(ctx::VulkanAccelBuildContext, vertices::LavaArray{NTuple{3,Float32},1},
                    indices::LavaArray{UInt32,1}; opaque::Bool=true, allow_update::Bool=false)
    geom = TrianglesGeometry(; vertex_format = UInt32(VK.FORMAT_R32G32B32_SFLOAT),
        vertex_addr = bda_address(vertices), vertex_stride = UInt64(sizeof(NTuple{3,Float32})),
        max_vertex = UInt32(length(vertices) - 1), index_type = UInt32(VK.INDEX_TYPE_UINT32),
        index_addr = bda_address(indices), transform_addr = UInt64(0))
    return newblas!(ctx, geom, length(indices) ÷ 3, opaque, allow_update,
                    LavaArray[reinterpret(UInt8, vertices), indices])
end

build_blas(ctx::VulkanAccelBuildContext, vertices::Vector{NTuple{3,Float32}}, indices::Vector{UInt32};
           kw...) =
    build_blas(ctx, LavaArray(vertices; ctx.bq, extra_usage=AS_INPUT_USAGE),
               LavaArray(indices; ctx.bq, extra_usage=AS_INPUT_USAGE); kw...)

"""refit_blas!(ctx, blas, vertices)

Rewrite the retained vertex buffer and refit in place. Requires allow_update=true
and the original vertex count; rejects misuse with ArgumentError. Rebuild when
geometry changes enough to degrade traversal quality."""
function refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS,
                     vertices::Vector{NTuple{3,Float32}})
    blas.geom isa TrianglesGeometry || throw(ArgumentError("refit_blas!: this BLAS holds no triangles."))
    refitcapable(blas)
    vbytes = first(blas.preserves)
    sizeof(vertices) == length(vbytes) || throw(ArgumentError(
        "refit_blas!: vertex count changed ($(sizeof(vertices)) bytes vs " *
        "$(length(vbytes)) at build). MODE_UPDATE_KHR cannot change " *
        "topology — rebuild the BLAS instead."))
    # Refit reuses the original vertex allocation and address.
    GC.@preserve vbytes upload_typed!(vbytes.buf[], vertices; offset = vbytes.offset)
    return refitstructure!(ctx, blas)
end

# Host-vector refits run on this device’s dispatch channel.
Raycore.refit!(blas::LavaBLAS, vertices::AbstractVector) =
    build_accel!(KA.get_backend(blas.storage).dispatch_bq) do ctx
        refit_blas!(ctx, blas, NTuple{3,Float32}[Tuple(Float32.(v)) for v in vertices])
    end

"""aabbinput(backend::LavaBackend, n) -> LavaArray{AABB,1}

Allocate n AS-readable boxes for kernels to write before build/refit."""
aabbinput(backend::LavaBackend, n::Integer) =
    LavaArray{AABB,1}(undef, (Int(n),); bq = backend.dispatch_bq, extra_usage = AS_INPUT_USAGE)

"""build_blas_aabb(ctx, aabbs; opaque=true, allow_update=false) -> LavaBLAS

Build procedural geometry from AS-readable device boxes; Vector inputs upload
once. Requires ray-query support. The BLAS retains its boxes for refits."""
function build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::LavaArray{AABB,1};
                         opaque::Bool=true, allow_update::Bool=false)
    (deviceof(ctx.bq)).ray_query_available || error(
        "build_blas_aabb: this Vulkan device does not support " *
        "VK_KHR_ray_query. Procedural-AABB BLASes are only useful with " *
        "ray_query in this codebase. Build on a device that supports it.")
    # `AABB` is two `Point3f`: VkAabbPositionsKHR's layout, 24 bytes a box.
    geom = AABBsGeometry(; aabb_addr = bda_address(aabbs), aabb_stride = UInt64(sizeof(AABB)))
    return newblas!(ctx, geom, length(aabbs), opaque, allow_update, LavaArray[aabbs])
end

build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::Vector{AABB}; opaque::Bool=true, allow_update::Bool=false) =
    build_blas_aabb(ctx, LavaArray(aabbs; bq = ctx.bq, extra_usage = AS_INPUT_USAGE); opaque, allow_update)

"""refit_blas!(ctx, blas, aabbs)

Refit procedural geometry built with allow_update=true. Reads the original
AS-readable buffer, or copies equal-length replacement boxes into it on device."""
function refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS, aabbs::LavaArray{AABB,1})
    blas.geom isa AABBsGeometry || throw(ArgumentError(
        "refit_blas!: this BLAS holds no boxes; it is not a refittable procedural BLAS."))
    refitcapable(blas)
    own = first(blas.preserves)
    length(aabbs) == length(own) || throw(ArgumentError(
        "refit_blas!: box count changed ($(length(aabbs)) vs $(length(own)) at build). " *
        "MODE_UPDATE_KHR cannot change the primitive count — rebuild instead."))
    aabbs === own || copyto!(own, aabbs)
    return refitstructure!(ctx, blas)
end

# On the device's dispatch channel, after the kernels that wrote the boxes.
Raycore.refit!(blas::LavaBLAS, aabbs::LavaArray{AABB,1}) =
    build_accel!(KA.get_backend(blas.storage).dispatch_bq) do ctx
        refit_blas!(ctx, blas, aabbs)
    end

"""build_tlas(ctx, instance_buf, n; allow_update=false, blases=LavaBLAS[]) -> LavaTLAS

Build from AS_INPUT_USAGE device records without a CPU readback. Pass their
referenced blases so every trace retains both hierarchy levels. The caller owns
the BLASes; releasing the TLAS only drops its references. The TLAS owns
`instance_buf`, which a refit reuses."""
function build_tlas(ctx::VulkanAccelBuildContext, instance_buf::LavaArray{VulkanInstanceRecord, 1},
                    n::Integer; allow_update::Bool=false, blases::Vector{LavaBLAS}=LavaBLAS[])
    n <= length(instance_buf) || throw(ArgumentError(
        "build_tlas: n = $n exceeds the instance buffer's $(length(instance_buf)) records"))
    geom = VulkanInstancesGeometry(bda_address(instance_buf))
    accel, storage, _, update = buildstructure!(ctx, geom, n, UInt32(0), allow_update)
    tlas = LavaTLAS(accel, storage, unique(blases), LavaArray[instance_buf], geom, UInt32(n),
                    UInt32(0), update, Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}())
    finalizer(unsafe_free!, tlas)
    holdbuild!(ctx,tlas)
    return tlas
end

const VkBRI = VK.VulkanCore.LibVulkan.VkAccelerationStructureBuildRangeInfoKHR

"""refit_tlas!(ctx, tlas, instance_buf, n)

Refit from fresh AS-readable instance records. Requires allow_update=true and
the original instance count. Reuses native AS storage; scratch is submit-scoped."""
function refit_tlas!(ctx::VulkanAccelBuildContext, tlas::LavaTLAS,
                     instance_buf::LavaArray{VulkanInstanceRecord, 1}, n::Integer)
    n <= length(instance_buf) || throw(ArgumentError(
        "refit_tlas!: n = $n exceeds the instance buffer's $(length(instance_buf)) records"))
    tlas.geom, tlas.count = VulkanInstancesGeometry(bda_address(instance_buf)), UInt32(n)
    holdbuild!(ctx, instance_buf)
    return refitstructure!(ctx, tlas)
end

# scratch=true applies the device’s required AS-scratch BDA alignment.
const AS_INPUT_USAGE = UInt32(
    VK.BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK.BUFFER_USAGE_TRANSFER_DST_BIT)

const AS_STORAGE_USAGE = UInt32(
    VK.BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
    VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)

const AS_SCRATCH_USAGE = UInt32(
    VK.BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)

# VulkanCore.jl declares VkDeviceOrHostAddressConstKHR as NTuple{8,UInt8}
# (alignment 1; the C union has 8), which misplaces fields of the AS geometry
# structs (sizeof 88 vs 96), so all AS structs are packed by hand at C offsets.

const C_SIZEOF_AS_GEOMETRY_KHR = 96

# VkStructureType values
const VK_STYPE_BGI = Int32(1000150000)   # BUILD_GEOMETRY_INFO
const VK_STYPE_SIZES = Int32(1000150020) # BUILD_SIZES_INFO (VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
const VK_STYPE_GEO = Int32(1000150006)   # GEOMETRY
const VK_STYPE_TRI = Int32(1000150005)   # GEOMETRY_TRIANGLES_DATA
const VK_STYPE_INST = Int32(1000150004)  # GEOMETRY_INSTANCES_DATA
const VK_STYPE_AABB = Int32(1000150003)  # GEOMETRY_AABBS_DATA

"""Pack VkAccelerationStructureGeometryKHR (96 bytes, correct C layout) into `buf`.

C layout:
  0:  sType (4)  |  4: pad (4)  |  8: pNext (8)
  16: geometryType (4)  |  20: pad (4)
  24: geometry union (64 bytes)
  88: flags (4)  |  92: pad (4)
"""
function pack_geometry!(buf::Vector{UInt8}, offset::Int,
                        geom::TrianglesGeometry; geo_flags::UInt32=UInt32(0))
    p = pointer(buf, offset + 1)
    q = p + 24  # geometry union start
    unsafe_store!(Ptr{Int32}(p), VK_STYPE_GEO)                      # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)                   # pNext @ 8
    unsafe_store!(Ptr{UInt32}(p + 16), UInt32(0))                    # geometryType = TRIANGLES @ 16
    unsafe_store!(Ptr{Int32}(q), VK_STYPE_TRI)                       # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)                   # pNext @ 8
    unsafe_store!(Ptr{UInt32}(q + 16), geom.vertex_format)           # vertexFormat @ 16
    unsafe_store!(Ptr{UInt64}(q + 24), geom.vertex_addr)             # vertexData @ 24
    unsafe_store!(Ptr{UInt64}(q + 32), geom.vertex_stride)           # vertexStride @ 32
    unsafe_store!(Ptr{UInt32}(q + 40), geom.max_vertex)              # maxVertex @ 40
    unsafe_store!(Ptr{UInt32}(q + 44), geom.index_type)              # indexType @ 44
    unsafe_store!(Ptr{UInt64}(q + 48), geom.index_addr)              # indexData @ 48
    unsafe_store!(Ptr{UInt64}(q + 56), geom.transform_addr)          # transformData @ 56
    unsafe_store!(Ptr{UInt32}(p + 88), geo_flags)                    # flags @ 88
    return nothing
end

function pack_geometry!(buf::Vector{UInt8}, offset::Int,
                        geom::AABBsGeometry; geo_flags::UInt32=UInt32(0))
    p = pointer(buf, offset + 1)
    q = p + 24  # geometry union start
    unsafe_store!(Ptr{Int32}(p), VK_STYPE_GEO)                      # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)                   # pNext @ 8
    unsafe_store!(Ptr{UInt32}(p + 16), UInt32(1))                    # geometryType = AABBS @ 16
    unsafe_store!(Ptr{Int32}(q), VK_STYPE_AABB)                      # sType @ q+0
    unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)                   # pNext @ q+8
    unsafe_store!(Ptr{UInt64}(q + 16), geom.aabb_addr)               # data @ q+16
    unsafe_store!(Ptr{UInt64}(q + 24), geom.aabb_stride)             # stride @ q+24
    unsafe_store!(Ptr{UInt32}(p + 88), geo_flags)                    # flags @ 88
    return nothing
end

function pack_geometry!(buf::Vector{UInt8}, offset::Int,
                        geom::VulkanInstancesGeometry; geo_flags::UInt32=UInt32(0))
    p = pointer(buf, offset + 1)
    q = p + 24
    unsafe_store!(Ptr{Int32}(p), VK_STYPE_GEO)
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)
    unsafe_store!(Ptr{UInt32}(p + 16), UInt32(2)) # INSTANCES
    unsafe_store!(Ptr{Int32}(q), VK_STYPE_INST)
    unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)
    unsafe_store!(Ptr{UInt32}(q + 16), UInt32(0)) # arrayOfPointers
    unsafe_store!(Ptr{UInt64}(q + 24), geom.address)
    unsafe_store!(Ptr{UInt32}(p + 88), geo_flags)
    return nothing
end

"""Pack VkAccelerationStructureBuildGeometryInfoKHR (80 bytes, C layout matches Julia)."""
function pack_build_geometry_info!(buf::Vector{UInt8}, offset::Int;
        as_type::UInt32, mode::UInt32=UInt32(0),
        build_flags::UInt32=UInt32(0),
        src_as::Ptr{Nothing}=C_NULL, dst_as::Ptr{Nothing}=C_NULL,
        geometry_count::UInt32=UInt32(1),
        p_geometries::Ptr{Nothing}=C_NULL,
        scratch_addr::UInt64=UInt64(0))
    p = pointer(buf, offset + 1)
    unsafe_store!(Ptr{Int32}(p), VK_STYPE_BGI)              # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)           # pNext @ 8
    unsafe_store!(Ptr{UInt32}(p + 16), as_type)               # type @ 16
    unsafe_store!(Ptr{UInt32}(p + 20), build_flags)           # flags @ 20
    unsafe_store!(Ptr{UInt32}(p + 24), mode)                  # mode @ 24
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 32), src_as)          # srcAccelerationStructure @ 32
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 40), dst_as)          # dstAccelerationStructure @ 40
    unsafe_store!(Ptr{UInt32}(p + 48), geometry_count)        # geometryCount @ 48
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 56), p_geometries)    # pGeometries @ 56
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 64), C_NULL)          # ppGeometries @ 64
    unsafe_store!(Ptr{UInt64}(p + 72), scratch_addr)          # scratchData @ 72
    return nothing
end

"""Query AS build sizes by calling vkGetAccelerationStructureBuildSizesKHR with
hand-packed structs (VulkanCore.jl's layout is wrong, see above)."""
function query_as_build_sizes(dev::VK.Device, geom::GeometryType;
        as_type::UInt32,
        build_flags::UInt32=UInt32(0), max_primitive_count::UInt32=UInt32(0),
        geo_flags::UInt32=UInt32(0))
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0, geom; geo_flags)
    return query_as_build_sizes_impl(dev, geo_buf, as_type, build_flags, max_primitive_count)
end

function query_as_build_sizes_impl(dev::VK.Device, geo_buf::Vector{UInt8},
        as_type::UInt32, build_flags::UInt32, max_primitive_count::UInt32)

    bgi_buf = zeros(UInt8, 80)

    # Output struct: VkAccelerationStructureBuildSizesInfoKHR
    # sType(4) + pad(4) + pNext(8) + accelerationStructureSize(8) + updateScratchSize(8) + buildScratchSize(8) = 40
    sizes_buf = zeros(UInt8, 40)
    unsafe_store!(Ptr{Int32}(pointer(sizes_buf)), VK_STYPE_SIZES)

    fptr = VK.function_pointer(dev, "vkGetAccelerationStructureBuildSizesKHR")

    GC.@preserve geo_buf bgi_buf sizes_buf begin
        geo_ptr = pointer(geo_buf)
        pack_build_geometry_info!(bgi_buf, 0;
            as_type, build_flags,
            geometry_count=UInt32(1),
            p_geometries=Ptr{Nothing}(geo_ptr))

        max_counts = Ref(max_primitive_count)

        GC.@preserve max_counts begin
            ccall(fptr, Cvoid,
                (Ptr{Nothing}, UInt32,     # device, buildType
                 Ptr{Nothing},              # pBuildInfo
                 Ptr{UInt32},               # pMaxPrimitiveCounts
                 Ptr{Nothing}),             # pSizeInfo
                dev.vks,
                UInt32(1),                  # VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR
                pointer(bgi_buf),
                max_counts,
                pointer(sizes_buf))
        end
    end

    as_size = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 16))
    update_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 24))
    build_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 32))

    return (acceleration_structure_size=as_size,
            update_scratch_size=update_scratch,
            build_scratch_size=build_scratch)
end

"""build_accel!(f, bq)

Record builds in one submission on bq, wait for completion and release scratch.
The callback receives a VulkanAccelBuildContext. BLAS addresses are available
while recording so later commands can build a TLAS from them. Resource holds,
last-use stamps and cross-channel waits belong to core's submission engine.
"""
function build_accel!(f, bq::SubmitChannel{<:VulkanQueue})
    # The head barrier covers prior work; named inputs add cross-channel waits.
    preserves = Any[]
    result = Ref{Any}(nothing)
    tok = oneshot!(bq; tag = :accel) do e
        result[] = f(AccelBuildContext(bq, e, preserves))
        # Final barrier: make all AS writes visible to RT shader reads
        memorybarrier!(e.cmd,e.ctx,
                       VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                       VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                       VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK.ACCESS_SHADER_READ_BIT)
    end
    # Completion and hold release use the caller’s channel.
    waitfor!(bq, tok)
    drain!(bq)

    # Retained geometry stays with its AS; completed scratch is released immediately.
    for p in preserves
        if p isa LavaArray
            unsafe_free!(p)
        end
    end
    empty!(preserves)
    return result[]
end

"""Pack geometry and record an AS build with prior-build ordering."""
function build_as_on_gpu(ctx::VulkanAccelBuildContext, accel::VK.AccelerationStructureKHR,
                         scratch_addr::UInt64, geom::GeometryType;
                         as_type::UInt32, build_flags::UInt32=UInt32(0),
                         primitive_count::UInt32=UInt32(0),
                         geo_flags::UInt32=UInt32(0),
                         mode::UInt32=UInt32(0),
                         src_as::Union{Nothing, VK.AccelerationStructureKHR}=nothing)
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0, geom; geo_flags)
    memorybarrier!(buildcmd(ctx),deviceof(ctx.bq),
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                   VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
    record_as_build!(ctx, accel, scratch_addr, geo_buf, as_type, build_flags,
                     primitive_count, mode, src_as)
end

# `mode` is MODE_BUILD_KHR (0) or MODE_UPDATE_KHR (1); an update also needs
# `src_as`, which is the same AS for an in-place refit.
function record_as_build!(ctx::VulkanAccelBuildContext, accel::VK.AccelerationStructureKHR,
                               scratch_addr::UInt64, geo_buf::Vector{UInt8},
                               as_type::UInt32, build_flags::UInt32, primitive_count::UInt32,
                               mode::UInt32=UInt32(0),
                               src_as::Union{Nothing, VK.AccelerationStructureKHR}=nothing)
    cmd = buildcmd(ctx)

    bgi_buf = zeros(UInt8, 80)

    c_range = VkBRI(primitive_count, UInt32(0), UInt32(0), UInt32(0))

    fptr = VK.function_pointer(as_device(ctx), "vkCmdBuildAccelerationStructuresKHR")

    GC.@preserve geo_buf bgi_buf c_range begin
        geo_ptr = pointer(geo_buf)
        dst_ptr = accel.vks

        pack_build_geometry_info!(bgi_buf, 0;
            as_type, mode, build_flags, dst_as=dst_ptr,
            src_as=(src_as === nothing ? C_NULL : src_as.vks),
            geometry_count=UInt32(1),
            p_geometries=Ptr{Nothing}(geo_ptr),
            scratch_addr)

        c_ranges = [c_range]
        pp_ranges = Ref(pointer(c_ranges))

        GC.@preserve c_ranges pp_ranges begin
            ccall(fptr, Cvoid,
                (Ptr{Nothing}, UInt32,
                 Ptr{Nothing},
                 Ptr{Ptr{VkBRI}}),
                cmd.vks, UInt32(1),
                pointer(bgi_buf),
                pp_ranges)
        end
    end

    # Keep packed buffers alive until build_accel!() submits
    push!(ctx.preserves, (geo_buf, bgi_buf, c_range))
end

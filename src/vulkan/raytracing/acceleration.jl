# Native AS construction; allocation holds and retirement belong to core.

"""Native triangle/procedural BLAS. Core retires its handle and owned buffers
after their last submission; allow_update enables fixed-topology refits."""
mutable struct LavaBLAS
    accel::VK.AccelerationStructureKHR
    # Core tracks AS last use and destruction through its backing storage.
    storage::LavaArray{UInt8, 1}
    address::UInt64  # AS device address (for HWTLAS instances)
    # Input arrays retained by this legacy owner; refits reuse their addresses.
    preserves::Vector{LavaArray}

    # Refit support is opt-in; update-enabled builds may trade traversal quality.
    allow_update::Bool
    # Scratch for MODE_UPDATE_KHR, queried at build time. Smaller than build
    # scratch; 0 means this BLAS is not refit-capable.
    update_scratch_size::UInt64
    # Retained triangle geometry for in-place refits; topology stays fixed.
    vertex_arr::Union{Nothing, LavaArray{UInt8, 1}}
    index_addr::UInt64
    n_triangles::UInt32
    max_vertex::UInt32
    geo_flags::UInt32
    # Retained procedural boxes, or nothing for triangle geometry.
    aabbs::Union{Nothing, LavaArray{AABB, 1}}
end

# Static BLAS wrapper for the pooled importer.
LavaBLAS(accel, storage, address, preserves) =
    LavaBLAS(accel, storage, address, preserves,
             false, UInt64(0), nothing, UInt64(0), UInt32(0), UInt32(0), UInt32(0), nothing)

"""Native instance hierarchy retaining referenced BLASes and instance records.
Core retires its descriptor pools, handle and storage after their last use.
allow_update caches scratch requirements for fixed-count refits."""
mutable struct LavaTLAS
    accel::VK.AccelerationStructureKHR
    storage::LavaArray{UInt8, 1}
    blases::Vector{LavaBLAS}
    preserves::Vector{LavaArray}
    # Refit support:
    allow_update::Bool
    update_scratch_size::UInt64
    instance_buf::Union{Nothing, LavaArray}   # pinned across refits when set
    # Layout-handle → descriptor pool/set. Destroy each pool before its AS.
    desc_sets::Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}
end

# Retire AS handles through the same stamp/recording holds as backing storage.

function unsafe_free!(as::Union{LavaBLAS, LavaTLAS})
    # Explicit release and finalization may both request retirement.
    as.storage.buf.freed && return
    ctx = (as.storage.buf[]::VkManagedBuffer).ctx::VkContext
    # A lost device signals nothing ever again, so nothing can be waited for and
    # the handles are dropped as they are.
    device_lost(ctx) && return destroy_now!(as)
    retire!(ctx.default_bq, as)
    return
end

# Where core writes what it knows about this structure's lifetime: the storage
# buffer's stamp, because the AS and its storage are the same bytes.
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

"""VulkanAccelBuildContext

AccelBuildContext over a Vulkan submit channel. Core holds/stamps named
resources; preserves keeps scratch and temporary host objects until completion.
"""
const VulkanAccelBuildContext = AccelBuildContext{SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}}

# Derived accessors so call sites stay readable.
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

"""build_blas(ctx, vertices, indices; opaque, allow_update) -> LavaBLAS

Record triangle geometry inside build_accel!. LavaArray inputs are read in
place and require AS_INPUT_USAGE; Vector inputs are uploaded first. Core tracks
input dependencies and retains resources through the submission.
"""
function build_blas(ctx::VulkanAccelBuildContext,
                    vertices::LavaArray{NTuple{3,Float32},1},
                    indices::LavaArray{UInt32,1};
                    opaque::Bool=true, allow_update::Bool=false)
    return build_blas_from_addresses(ctx, reinterpret(UInt8,vertices), indices,
                                     bda_address(vertices), bda_address(indices),
                                     length(vertices), length(indices);
                                     opaque, allow_update)
end

function build_blas(ctx::VulkanAccelBuildContext, vertices::Vector{NTuple{3,Float32}}, indices::Vector{UInt32};
                    opaque::Bool=true, allow_update::Bool=false)
    bq = ctx.bq
    # Upload vertex/index data to device-local buffers (LavaArrays), then hand
    # the shared implementation their addresses — which is all it needs.
    vertex_arr = reinterpret(UInt8,LavaArray(vertices; bq, extra_usage=AS_INPUT_USAGE))
    index_arr  = LavaArray(indices; bq, extra_usage=AS_INPUT_USAGE)
    return build_blas_from_addresses(ctx, vertex_arr, index_arr,
                                     vertex_arr.buf[].address, index_arr.buf[].address,
                                     length(vertices), length(indices);
                                     opaque, allow_update)
end

function build_blas_from_addresses(ctx::VulkanAccelBuildContext, vertex_arr, index_arr,
                                   vertex_addr::UInt64, index_addr::UInt64,
                                   nvertices::Integer, nindices::Integer;
                                   opaque::Bool=true, allow_update::Bool=false)
    bq = ctx.bq
    dev = as_device(ctx)
    n_triangles = UInt32(nindices ÷ 3)
    max_vertex = UInt32(nvertices - 1)

    vfmt = UInt32(VK.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(VK.INDEX_TYPE_UINT32)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    if allow_update
        build_flags |= UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
    end
    geo_flags = opaque ? UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR) : UInt32(0)

    geom = TrianglesGeometry(; vertex_format=vfmt, vertex_addr, vertex_stride=vstride,
                               max_vertex, index_type=itype, index_addr,
                               transform_addr=UInt64(0))

    sizes = query_as_build_sizes(dev, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, max_primitive_count=n_triangles)

    storage = LavaArray{UInt8,1}(undef, (max(Int(sizes.acceleration_structure_size), 16),);
                                  bq, extra_usage=AS_STORAGE_USAGE)
    accel = VK.AccelerationStructureKHR(dev, VK.AccelerationStructureCreateInfoKHR(
        storage.buf[].buffer, UInt64(0), sizes.acceleration_structure_size,
        VK.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR))

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(sizes.build_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)

    # This owner retains its input arrays alongside storage.
    blas_preserves = LavaArray[vertex_arr, index_arr]
    # Release scratch explicitly after the build submission completes.
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)

    build_as_on_gpu(ctx, accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, primitive_count=n_triangles)

    addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
    if (ctxof(ctx.bq)).diag.alloc_debug
        push!((ctxof(ctx.bq)).diag.alloc_log,
              (kind=:blas_as, addr=as_addr, size=0, pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end
    blas = LavaBLAS(accel, storage, as_addr, blas_preserves,
                    allow_update,
                    allow_update ? UInt64(sizes.update_scratch_size) : UInt64(0),
                    allow_update ? vertex_arr : nothing,
                    index_addr, n_triangles, max_vertex, geo_flags, nothing)
    finalizer(unsafe_free!, blas)
    holdbuild!(ctx,blas)
    return blas
end

"""refit_blas!(ctx, blas, vertices)

Rewrite the retained vertex buffer and refit in place. Requires allow_update=true
and the original vertex count; rejects misuse with ArgumentError. Rebuild when
geometry changes enough to degrade traversal quality."""
function refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS,
                     vertices::Vector{NTuple{3,Float32}})
    blas.allow_update || throw(ArgumentError(
        "refit_blas!: BLAS was built with allow_update=false; cannot refit. " *
        "Rebuild it via build_blas(...; allow_update=true)."))
    blas.update_scratch_size > 0 || throw(ArgumentError(
        "refit_blas!: cached update_scratch_size is 0; the BLAS is not refit-capable."))
    vertex_arr = blas.vertex_arr
    vertex_arr === nothing && throw(ArgumentError(
        "refit_blas!: no vertex buffer retained; the BLAS is not refit-capable."))
    sizeof(vertices) == length(vertex_arr) || throw(ArgumentError(
        "refit_blas!: vertex count changed ($(sizeof(vertices)) bytes vs " *
        "$(length(vertex_arr)) at build). MODE_UPDATE_KHR cannot change " *
        "topology — rebuild the BLAS instead."))

    bq = ctx.bq
    dev = as_device(ctx)

    # Refit reuses the original vertex allocation and address.
    GC.@preserve vertex_arr upload_typed!(vertex_arr.buf[],vertices;offset=vertex_arr.offset)

    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
                  UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(blas.update_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)

    geom = TrianglesGeometry(;
        vertex_format=UInt32(VK.FORMAT_R32G32B32_SFLOAT),
        vertex_addr=bda_address(vertex_arr),
        vertex_stride=UInt64(sizeof(NTuple{3,Float32})),
        max_vertex=blas.max_vertex,
        index_type=UInt32(VK.INDEX_TYPE_UINT32),
        index_addr=blas.index_addr,
        transform_addr=UInt64(0))

    # The vertex rewrite above is a transfer write the AS build has to see, and a
    # previous build/refit of this same AS has to have finished.
    holdbuild!(ctx,blas)
    asbuildbarrier!(ctx)

    build_as_on_gpu(ctx, blas.accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags=blas.geo_flags, primitive_count=blas.n_triangles,
        mode=UInt32(1), src_as=blas.accel)   # MODE_UPDATE_KHR, in place
    return blas
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
once. Requires ray-query support. An updateable BLAS retains its input for refits."""
function build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::LavaArray{AABB,1};
                         opaque::Bool=true, allow_update::Bool=false)
    (ctxof(ctx.bq)).ray_query_available || error(
        "build_blas_aabb: this Vulkan device does not support " *
        "VK_KHR_ray_query. Procedural-AABB BLASes are only useful with " *
        "ray_query in this codebase. Build on a device that supports it.")

    bq  = ctx.bq
    dev = as_device(ctx)
    n   = length(aabbs)

    # `AABB` is two `Point3f`: VkAabbPositionsKHR's layout, 24 bytes a box.
    geom       = AABBsGeometry(; aabb_addr = bda_address(aabbs), aabb_stride = UInt64(sizeof(AABB)))
    geo_flags  = opaque ? UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR) : UInt32(0)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    if allow_update
        build_flags |= UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
    end

    sizes = query_as_build_sizes(dev, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, max_primitive_count=UInt32(n))

    storage = LavaArray{UInt8,1}(undef, (max(Int(sizes.acceleration_structure_size), 16),);
                                  bq, extra_usage=AS_STORAGE_USAGE)
    accel = VK.AccelerationStructureKHR(dev, VK.AccelerationStructureCreateInfoKHR(
        storage.buf[].buffer, UInt64(0), sizes.acceleration_structure_size,
        VK.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR))

    scratch_arr  = LavaArray{UInt8,1}(undef, (max(Int(sizes.build_scratch_size), 16),);
                                       bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)

    # The boxes a kernel wrote before this build must be visible to it.
    asbuildbarrier!(ctx)
    build_as_on_gpu(ctx, accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, primitive_count=UInt32(n))

    addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr   = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
    if (ctxof(ctx.bq)).diag.alloc_debug
        push!((ctxof(ctx.bq)).diag.alloc_log,
              (kind=:blas_aabb_as, addr=as_addr, size=0, pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end
    blas = LavaBLAS(accel, storage, as_addr, LavaArray[aabbs],
                    allow_update,
                    allow_update ? UInt64(sizes.update_scratch_size) : UInt64(0),
                    nothing, UInt64(0), UInt32(0), UInt32(0), geo_flags,
                    allow_update ? aabbs : nothing)
    finalizer(unsafe_free!, blas)
    holdbuild!(ctx,blas)
    return blas
end

build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::Vector{AABB}; opaque::Bool=true, allow_update::Bool=false) =
    build_blas_aabb(ctx, LavaArray(aabbs; bq = ctx.bq, extra_usage = AS_INPUT_USAGE); opaque, allow_update)

"""Make kernel, transfer and prior AS writes visible to a build/refit."""
function asbuildbarrier!(ctx::VulkanAccelBuildContext)
    memorybarrier!(buildcmd(ctx),ctxof(ctx.bq),
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                   VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK.PIPELINE_STAGE_TRANSFER_BIT,
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                   VK.ACCESS_SHADER_WRITE_BIT | VK.ACCESS_TRANSFER_WRITE_BIT,
                   VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                   VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
    return ctx
end

"Serialize builds that reuse one scratch allocation."
@inline function asscratchbarrier!(ctx::VulkanAccelBuildContext)
    access = VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
             VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
    memorybarrier!(buildcmd(ctx),ctxof(ctx.bq),
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,access,access)
    return nothing
end

"""refit_blas!(ctx, blas, aabbs)

Refit procedural geometry built with allow_update=true. Reads the original
AS-readable buffer, or copies equal-length replacement boxes into it on device."""
function refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS, aabbs::LavaArray{AABB,1})
    blas.allow_update || throw(ArgumentError(
        "refit_blas!: BLAS was built with allow_update=false; cannot refit. " *
        "Rebuild it via build_blas_aabb(...; allow_update=true)."))
    own = blas.aabbs
    own === nothing && throw(ArgumentError("refit_blas!: this BLAS holds no boxes; it is not a refittable procedural BLAS."))
    length(aabbs) == length(own) || throw(ArgumentError(
        "refit_blas!: box count changed ($(length(aabbs)) vs $(length(own)) at build). " *
        "MODE_UPDATE_KHR cannot change the primitive count — rebuild instead."))
    aabbs === own || copyto!(own, aabbs)

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(blas.update_scratch_size), 16),);
                                      bq = ctx.bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)
    geom = AABBsGeometry(; aabb_addr = bda_address(own), aabb_stride = UInt64(sizeof(AABB)))
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
                  UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
    holdbuild!(ctx,blas)
    asbuildbarrier!(ctx)
    build_as_on_gpu(ctx, blas.accel, bda_address(scratch_arr), geom;
        as_type=UInt32(1), build_flags,
        geo_flags=blas.geo_flags, primitive_count=UInt32(length(own)),
        mode=UInt32(1), src_as=blas.accel)   # MODE_UPDATE_KHR, in place
    return blas
end

# On the device's dispatch channel, after the kernels that wrote the boxes.
Raycore.refit!(blas::LavaBLAS, aabbs::LavaArray{AABB,1}) =
    build_accel!(KA.get_backend(blas.storage).dispatch_bq) do ctx
        refit_blas!(ctx, blas, aabbs)
    end

"""build_tlas(ctx, blas_list; transforms, custom_indices, masks, allow_update=false)

Pack instance records and delegate to the device-record builder inside build_accel!."""
function build_tlas(ctx::VulkanAccelBuildContext, blas_list::Vector{LavaBLAS};
                    transforms::Union{Nothing, Vector{NTuple{12,Float32}}}=nothing,
                    custom_indices::Union{Nothing, Vector{UInt32}}=nothing,
                    masks::Union{Nothing, Vector{UInt8}}=nothing,
                    allow_update::Bool=false)
    records = [VulkanInstanceRecord(
        transforms === nothing ? identity_transform() : Mat3x4f(transforms[i]),
        blas.address;
        custom_index=custom_indices === nothing ? UInt32(0) : custom_indices[i],
        mask=masks === nothing ? UInt8(0xff) : masks[i])
        for (i,blas) in enumerate(blas_list)]
    instance_buf = LavaArray(records;bq=ctx.bq,extra_usage=AS_INPUT_USAGE)
    return build_tlas(ctx,instance_buf,length(records);allow_update,blases=blas_list)
end

"""build_tlas(ctx, instance_buf, n; allow_update=false, blases=LavaBLAS[]) -> LavaTLAS

Build from AS_INPUT_USAGE device records without a CPU readback. Pass their
referenced blases so every trace retains both hierarchy levels. The caller owns
the BLASes; releasing the TLAS only drops its references. allow_update enables
fixed-count refits using the retained instance buffer."""
function build_tlas(ctx::VulkanAccelBuildContext, instance_buf::LavaArray{VulkanInstanceRecord, 1},
                    n::Integer; allow_update::Bool=false, blases::Vector{LavaBLAS}=LavaBLAS[])
    bq = ctx.bq
    dev = as_device(ctx)
    n_instances = Int(n)
    @assert n_instances <= length(instance_buf) "n=$n_instances exceeds instance buffer length $(length(instance_buf))"

    inst_addr = bda_address(instance_buf)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    if allow_update
        build_flags |= UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
    end

    geom = VulkanInstancesGeometry(inst_addr)
    sizes = query_as_build_sizes(dev, geom; as_type=UInt32(0), build_flags,
                                max_primitive_count=UInt32(n_instances))

    storage = LavaArray{UInt8,1}(undef, (max(Int(sizes.acceleration_structure_size), 16),);
                                  bq, extra_usage=AS_STORAGE_USAGE)
    accel = VK.AccelerationStructureKHR(dev, VK.AccelerationStructureCreateInfoKHR(
        storage.buf[].buffer, UInt64(0), sizes.acceleration_structure_size,
        VK.ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR))

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(sizes.build_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)

    # instance_buf must outlive the HWTLAS -- pin it on the HWTLAS itself for refit too.
    tlas_preserves = LavaArray[instance_buf]
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)

    build_as_on_gpu(ctx, accel, scratch_addr, geom;
        as_type=UInt32(0), build_flags, primitive_count=UInt32(n_instances))

    # Retain both levels for traces; the caller still owns the BLASes.
    tlas = LavaTLAS(accel, storage, unique(blases), tlas_preserves,
                    allow_update, sizes.update_scratch_size, instance_buf,
                    Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}())
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
    tlas.allow_update || error(
        "refit_tlas!: HWTLAS was built with allow_update=false; cannot refit. " *
        "Rebuild the HWTLAS via build_tlas(...; allow_update=true).")
    tlas.update_scratch_size > 0 || error(
        "refit_tlas!: cached update_scratch_size is 0; the HWTLAS is not refit-capable.")

    bq = ctx.bq
    n_instances = Int(n)
    @assert n_instances <= length(instance_buf) "n=$n_instances exceeds instance buffer length $(length(instance_buf))"

    inst_addr = bda_address(instance_buf)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
                  UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)

    # Allocate update scratch (cached size, not full build scratch).
    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(tlas.update_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    push!(ctx.preserves, scratch_arr)
    holdbuild!(ctx,scratch_arr)

    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0, VulkanInstancesGeometry(inst_addr))
    holdbuild!(ctx,tlas)
    holdbuild!(ctx,instance_buf)
    asbuildbarrier!(ctx)
    record_as_build!(ctx,tlas.accel,scratch_addr,geo_buf,UInt32(0),build_flags,
                     UInt32(n_instances),UInt32(1),tlas.accel)
    return tlas
end

# ── Internal helpers ──

"""Allocate a coherent, host-visible AS input pool. Returns (buffer, memory, address)."""
function create_as_input_pool(ctx::VkContext, nbytes::UInt64)
    dev = ctx.device

    buf = VK.Buffer(
        dev, max(nbytes, 16),
        VK.BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )

    mem_reqs = VK.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = find_memory_type(
        ctx, mem_reqs.memory_type_bits,
        VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK.MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )

    alloc_flags = VK.MemoryAllocateFlagsInfo(UInt32(0);
        flags=VK.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = VK.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    throw_if_error(ctx, "vkBindBufferMemory",
        VK.bind_buffer_memory(dev, buf, memory, 0))

    addr = VK.get_buffer_device_address(dev, VK.BufferDeviceAddressInfo(buf))

    # Include host-visible AS input addresses in allocation diagnostics.
    if ctx.diag.alloc_debug
        push!(ctx.diag.alloc_log,
              (kind=:as_input_pool, addr=addr, size=Int(nbytes), pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end

    return buf, memory, addr
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

# VulkanCore.jl alignment bug: VkDeviceOrHostAddressConstKHR is NTuple{8,UInt8} (alignment 1)
# but in C it's a union of uint64_t/void* (alignment 8). This causes misaligned fields in:
#   VkAccelerationStructureGeometryKHR:       Julia sizeof=88 vs C sizeof=96
#   VkAccelerationStructureGeometryDataKHR:   geometry union at offset 20 vs C offset 24
# We construct all AS-related C structs manually with correct field offsets.

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

struct VulkanInstancesGeometry <: GeometryType
    address::UInt64
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

"""Query AS build sizes using correctly-packed C structs.

Calls vkGetAccelerationStructureBuildSizesKHR directly via ccall to work around
VulkanCore.jl alignment bug in VkAccelerationStructureGeometryKHR.
"""
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

    # Read back sizes
    as_size = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 16))
    update_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 24))
    build_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 32))

    return (acceleration_structure_size=as_size,
            update_scratch_size=update_scratch,
            build_scratch_size=build_scratch)
end

# ── AS Build ──

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
    memorybarrier!(buildcmd(ctx),ctxof(ctx.bq),
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

    # Pack build geometry info (80 bytes)
    bgi_buf = zeros(UInt8, 80)

    # Pack range info (16 bytes)
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

# Raycore's BLAS records contain borrowed device views; retain/use their owning arrays.
build_hw_accel_from_tlas(tlas::Raycore.TLAS;kw...) =
    build_hw_accel_from_tlas((instances=tlas.instances,blas_array=tlas.blas_storage);kw...)

"""build_hw_accel_from_tlas(tlas; ctx, bq = ctx.default_bq)

Legacy host-mediated import into pooled native BLAS/TLAS storage. Compatible
sources provide owning blas_array[i].primitives arrays and instance descriptors
with 1-based blas_index, canonical row-major affine transform, ID, mask and flags.
Raycore.TLAS supplies its separate owning blas_storage arrays.

Returns native TLAS, flat CPU primitives, BLAS offsets and per-instance offsets.
Look up a hit through per_instance_tri_offsets[instance_id+1] + primitive_id + 1;
instance_custom_index retains the instance’s custom ID.
"""
function build_hw_accel_from_tlas(tlas;
                                  ctx::VkContext,
                                  bq::SubmitChannel{<:VulkanQueue}=ctx.default_bq)
    instances = to_cpu_vector(tlas.instances)
    blas_array = to_cpu_vector(tlas.blas_array)

    n_blas = length(blas_array)
    n_instances = length(instances)
    dev = ctx.device

    # ── Pass 1: Collect all geometry on CPU ──
    blas_offsets = Vector{UInt32}(undef, n_blas)
    all_primitives = []
    prim_offset = UInt32(0)

    # Per-BLAS vertex/index arrays
    all_vertices = Vector{Vector{NTuple{3,Float32}}}(undef, n_blas)
    all_indices = Vector{Vector{UInt32}}(undef, n_blas)

    for i in 1:n_blas
        primitives = to_cpu_vector(blas_array[i].primitives)
        blas_offsets[i] = prim_offset
        prim_offset += UInt32(length(primitives))
        append!(all_primitives, primitives)
        all_vertices[i] = trianglevertices(primitives)
        all_indices[i] = collect(UInt32,0:length(all_vertices[i])-1)
    end

    # ── Pass 2: Query build sizes and compute pool layout ──
    vfmt = UInt32(VK.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(VK.INDEX_TYPE_UINT32)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    geo_flags = UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR)

    # Layout tracking
    vertex_offsets = Vector{UInt64}(undef, n_blas)  # byte offset into input pool
    index_offsets = Vector{UInt64}(undef, n_blas)
    as_offsets = Vector{UInt64}(undef, n_blas)       # byte offset into AS storage pool
    as_sizes_list = Vector{UInt64}(undef, n_blas)
    max_scratch_size = UInt64(0)

    input_cursor = UInt64(0)  # cursor into input pool (vertex then index, interleaved per BLAS)
    as_cursor = UInt64(0)     # cursor into AS storage pool (256-byte aligned)

    for i in 1:n_blas
        n_tris = UInt32(length(all_indices[i]) ÷ 3)
        max_vertex = UInt32(length(all_vertices[i]) - 1)

        # Query sizes -- addresses don't matter for size queries
        geom_i = TrianglesGeometry(; vertex_format=vfmt, vertex_addr=UInt64(0),
                                     vertex_stride=vstride, max_vertex,
                                     index_type=itype, index_addr=UInt64(0))
        sizes = query_as_build_sizes(dev, geom_i;
            as_type=UInt32(1), build_flags,
            geo_flags, max_primitive_count=n_tris)

        # Input pool layout: vertex data then index data, 16-byte aligned
        vertex_offsets[i] = input_cursor
        vbytes = UInt64(length(all_vertices[i]) * sizeof(NTuple{3,Float32}))
        input_cursor += (vbytes + 15) & ~UInt64(15)  # 16-byte align

        index_offsets[i] = input_cursor
        ibytes = UInt64(length(all_indices[i]) * sizeof(UInt32))
        input_cursor += (ibytes + 15) & ~UInt64(15)  # 16-byte align

        # AS storage pool layout: 256-byte aligned
        as_offsets[i] = as_cursor
        as_sizes_list[i] = sizes.acceleration_structure_size
        as_cursor += (sizes.acceleration_structure_size + 255) & ~UInt64(255)

        max_scratch_size = max(max_scratch_size, sizes.build_scratch_size)
    end

    total_input_bytes = input_cursor
    total_as_bytes = as_cursor

    # ── Pass 3: Allocate pooled buffers ──
    # Input pool: HOST_VISIBLE for direct upload
    input_buf, input_mem, input_base_addr = create_as_input_pool(ctx, max(total_input_bytes, 16))

    # AS storage pool: DEVICE_LOCAL (LavaArray — shared across all pooled BLASes)
    as_pool_arr = LavaArray{UInt8,1}(undef, (max(Int(total_as_bytes), 16),);
                                      bq, extra_usage=AS_STORAGE_USAGE)
    as_pool_buf = as_pool_arr.buf[].buffer

    # Scratch buffer: single allocation, reused for all builds
    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(max_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)

    # ── Pass 4: Upload vertex/index data into input pool ──
    mapped_ptr = throw_if_error(ctx, "vkMapMemory",
        VK.map_memory(dev, input_mem, 0, total_input_bytes))
    for i in 1:n_blas
        # Copy vertex data
        vdata = all_vertices[i]
        vbytes = length(vdata) * sizeof(NTuple{3,Float32})
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + vertex_offsets[i]),
                       Ptr{UInt8}(pointer(vdata)), vbytes)
        # Copy index data
        idata = all_indices[i]
        ibytes = length(idata) * sizeof(UInt32)
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + index_offsets[i]),
                       Ptr{UInt8}(pointer(idata)), ibytes)
    end
    VK.unmap_memory(dev, input_mem)

    # ── Pass 5: Create AS objects and build all BLASes ──
    hw_blas_list = Vector{LavaBLAS}(undef, n_blas)

    # Create AccelerationStructureKHR objects (one per BLAS, all referencing the pool)
    for i in 1:n_blas
        as_ci = VK.AccelerationStructureCreateInfoKHR(
            as_pool_buf, as_offsets[i], as_sizes_list[i],
            VK.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        )
        accel = VK.AccelerationStructureKHR(dev, as_ci)
        addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
        as_addr = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
        if ctx.diag.alloc_debug
            push!(ctx.diag.alloc_log,
                  (kind=:blas_as_pool2, addr=as_addr, size=0, pool=true, mtype=-1, unified=false, usage=UInt32(0)))
        end
        blas = LavaBLAS(accel, as_pool_arr, as_addr, LavaArray[])
        # No finalizer: each pooled BLAS shares `as_pool_arr` as storage.
        # Julia GC keeps `as_pool_arr` alive while any BLAS references it;
        # its VkManagedBuffer finalizer handles timeline-gated destruction.
        hw_blas_list[i] = blas
    end

    # Batch all BLAS + HWTLAS builds into a single GPU submission
    hw_tlas = GC.@preserve input_buf input_mem scratch_arr as_pool_arr begin
        build_accel!(bq) do as_ctx
            holdbuild!(as_ctx,scratch_arr)
            holdbuild!(as_ctx,as_pool_arr)
            for i in 1:n_blas
                n_tris = UInt32(length(all_indices[i]) ÷ 3)
                max_vertex = UInt32(length(all_vertices[i]) - 1)

                geom_i = TrianglesGeometry(; vertex_format=vfmt,
                    vertex_addr=input_base_addr + vertex_offsets[i],
                    vertex_stride=vstride, max_vertex,
                    index_type=itype,
                    index_addr=input_base_addr + index_offsets[i])
                build_as_on_gpu(as_ctx, hw_blas_list[i].accel, scratch_addr, geom_i;
                    as_type=UInt32(1), build_flags,
                    geo_flags, primitive_count=n_tris)
                # Barrier between builds: shared scratch buffer must be drained before reuse.
                i < n_blas && asscratchbarrier!(as_ctx)
            end

            # Instance position indexes triangle offsets; custom_index carries the user ID.
            hw_blas_refs = Vector{LavaBLAS}(undef, n_instances)
            transforms = Vector{NTuple{12,Float32}}(undef, n_instances)
            custom_indices = Vector{UInt32}(undef, n_instances)
            masks = Vector{UInt8}(undef,n_instances)

            for i in 1:n_instances
                inst = instances[i]
                blas_idx = Int(inst.blas_index)
                hw_blas_refs[i] = hw_blas_list[blas_idx]
                transforms[i] = Tuple(inst.transform)
                custom_indices[i] = inst.instance_id
                masks[i] = inst.flags & Raycore.INSTANCE_HIDDEN == 0 ? UInt8(inst.mask & 0xff) : 0x00
            end

            return build_tlas(as_ctx, hw_blas_refs; transforms, custom_indices, masks)
        end
    end

    # build_accel! has waited and drained holds; release temporary inputs/scratch.
    unsafe_free!(scratch_arr)
    finalize(input_buf); finalize(input_mem)

    # Convert all_primitives to typed vector
    if !isempty(all_primitives)
        T = typeof(all_primitives[1])
        typed_prims = T[p for p in all_primitives]
    else
        typed_prims = Any[]
    end

    # gl_InstanceID indexes the BLAS offset in the flat primitive array.
    per_instance_tri_offsets = UInt32[blas_offsets[Int(instances[i].blas_index)] for i in 1:n_instances]

    return (hw_tlas, typed_prims, blas_offsets, per_instance_tri_offsets)
end


"""Download GPU array to CPU Vector, or return as-is if already a CPU collection."""
function to_cpu_vector(x)
    x isa Vector && return x
    x isa Tuple && return collect(x)
    # Only absent Array conversion falls through; propagate transfer/device errors.
    try
        return Array(x)
    catch ex
        ex isa MethodError || rethrow()
        return collect(x)
    end
end

# Acceleration structure build for Lava.jl
#
# BLAS (bottom-level): triangle geometry
# HWTLAS (top-level): instances referencing BLASes
#
# Buffers need ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR for vertex/index data
# and ACCELERATION_STRUCTURE_STORAGE_BIT_KHR for AS backing storage.

"""
    LavaBLAS

Bottom-level acceleration structure wrapping a VkAccelerationStructureKHR.
Destroyed automatically via finalizer when GC'd (unless device was lost).
"""
mutable struct LavaBLAS
    accel::VK.AccelerationStructureKHR
    # AS backing storage. The only "last_write" we care about lives here:
    # when the GPU traces rays, it reads this buffer. Cross-queue sync and
    # timeline-gated destruction fall out of LavaArray's existing machinery.
    storage::LavaArray{UInt8, 1}
    address::UInt64  # AS device address (for HWTLAS instances)
    # Vertex/index buffers the driver may internally retain VAs to.  Kept
    # alive as long as the BLAS is alive.  Each entry is a LavaArray whose
    # own `last_write` tracks in-flight usage, so their VkManagedBuffers
    # are also timeline-gated when the BLAS is finalized.
    preserves::Vector{LavaArray}

    # ── Refit support (MODE_UPDATE_KHR), only populated when built with
    # `allow_update=true`.
    #
    # Opt-in per BLAS and NOT the default, because ALLOW_UPDATE makes the driver
    # build a lower-quality BVH: every static scene would pay traversal
    # performance for a capability it never uses, which would move the pbrt and
    # crown benchmark numbers. A deforming mesh promotes itself by rebuilding
    # once with this set, then refits from then on.
    allow_update::Bool
    # Scratch for MODE_UPDATE_KHR, queried at build time. Smaller than build
    # scratch; 0 means this BLAS is not refit-capable.
    update_scratch_size::UInt64
    # The geometry a refit re-issues. Vertices are rewritten in place, so the
    # buffer and its device address survive and the driver's retained VAs stay
    # valid. Topology cannot change: `MODE_UPDATE_KHR` keeps the tree and only
    # refits bounds, so triangle count and index buffer are fixed at build.
    vertex_arr::Union{Nothing, LavaArray{UInt8, 1}}
    index_addr::UInt64
    n_triangles::UInt32
    max_vertex::UInt32
    geo_flags::UInt32
end

# A BLAS with no refit support — what the AABB and pooled builders produce.
LavaBLAS(accel, storage, address, preserves) =
    LavaBLAS(accel, storage, address, preserves,
             false, UInt64(0), nothing, UInt64(0), UInt32(0), UInt32(0), UInt32(0))

"""
    LavaTLAS

Top-level acceleration structure wrapping a VkAccelerationStructureKHR.
Destroyed automatically via finalizer when GC'd (unless device was lost).

When built with `allow_update=true`, `update_scratch_size` is the scratch
buffer size to allocate for `MODE_UPDATE_KHR` refit calls (queried at build
time). `instance_buf` holds the GPU instance buffer when the HWTLAS was built
from a `LavaArray{VulkanInstanceRecord, 1}` (refit needs to keep this pinned
across calls).
"""
mutable struct LavaTLAS
    accel::VK.AccelerationStructureKHR
    storage::LavaArray{UInt8, 1}
    blases::Vector{LavaBLAS}
    preserves::Vector{LavaArray}
    # Refit support:
    allow_update::Bool
    update_scratch_size::UInt64
    instance_buf::Union{Nothing, LavaArray}   # pinned across refits when set
    # Per-HWTLAS RT descriptor sets, keyed by descriptor-set-layout Vulkan handle
    # (UInt64).  Lazily populated by `get_rt_descriptor_set` and destroyed in
    # `destroy_now!` so each VkAccelerationStructureKHR / descriptor set pair
    # has a lifetime tied to *this* LavaTLAS object — no global cache, no
    # objectid-keyed reuse-after-free.  Each entry holds (DescriptorPool,
    # DescriptorSet) where the pool exclusively owns the set so destroying
    # the pool destroys the set.
    desc_sets::Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}
end

# ── Timeline-aware destruction for AS objects ──────────────────────────────
#
# LavaBLAS/LavaTLAS hold LavaArrays for all their GPU memory.  Each LavaArray's
# finalizer calls `vk_free!`, which defers via the BQ's timeline semaphore if
# the buffer is still in flight.  So we just need to:
#   1. Destroy the Vulkan AccelerationStructureKHR handle (gated on the
#      storage buffer's last write — same GPU memory backs the accel).
#   2. Drop refs to the LavaArrays; their finalizers handle the rest.
# If the storage's last write hasn't been signalled yet, the whole AS object
# is resurrected onto bq.deferred_as_frees and destroyed on the next drain.
#
# Same shape as `vk_free!`, for the same reason: the last-write stamp is the
# owning thread's alone. A finalizer on any other thread hands the AS to the
# default queue's deferred list WITHOUT reading it, and the owning thread's
# drain asks the stamp. It used to read the stamp from the finalizer thread and
# destroy the AS right there when the device was done — the one reader that
# made the stamp an atomic tuple, boxed at every submit.

function unsafe_free!(as::Union{LavaBLAS, LavaTLAS})
    # Idempotent: if `destroy_now!` already released `as.storage`'s DataRef
    # (e.g. via an earlier explicit `unsafe_free!` + finalizer double-tap),
    # there is nothing left to do.  Without this guard, the GC-scheduled
    # finalizer trips `storage.buf[]` → `ArgumentError("Attempt to use a
    # freed reference.")` from GPUArrays.
    as.storage.buf.freed && return
    buf = as.storage.buf[]::VkManagedBuffer
    ctx = buf.ctx::VkContext
    if device_lost(ctx)
        destroy_now!(as)
        return
    end
    bq = ctx.default_bq
    if Threads.threadid() != bq.owning_thread
        lock(bq.deferred_frees_lock) do
            push!(bq.deferred_as_frees, as)   # resurrect + defer, unread
        end
        return
    end
    let wbq = buf.last_write_bq
        if wbq !== nothing
            w = wbq::VulkanBatchQueue
            # query_timeline throws on healthy-device failure; device_lost is
            # checked fresh on the next call.
            if !queue_released(w) && !passed(w, buf.last_write_val)
                lock(w.deferred_frees_lock) do
                    push!(w.deferred_as_frees, as)   # resurrect + defer
                end
                return
            end
        end
    end
    destroy_now!(as)
end

function destroy_now!(as::Union{LavaBLAS, LavaTLAS})
    # HWTLAS-only: destroy any RT descriptor pools we lazily created for this
    # AS.  The pool owns the descriptor set; destroying it invalidates the set.
    # Doing this BEFORE `as.accel.destructor()` keeps the spec-required order
    # "free descriptors that reference an AS, then destroy the AS" obvious;
    # in practice both happen synchronously after the storage timeline is
    # signalled so there is no in-flight dispatch left to read either.
    if as isa LavaTLAS
        for (_, (pool, _)) in as.desc_sets
            try
                pool.destructor()
            catch ex
                # A destructor may not throw, but it can name the fault: the
                # message was fixed text, so a driver error and a bug in this
                # file printed identically.
                safe_fin_log("Lava LavaTLAS destroy_now!: descriptor pool destructor failed: " * sprint(showerror, ex) * "\n")
            end
        end
        empty!(as.desc_sets)
    end
    as.accel.destructor()     # vkDestroyAccelerationStructureKHR — let
                              # Julia's finalizer logger surface any failure
    # Release storage + preserves through DataRef's refcount path.
    # LavaArray has no finalizer after the Phase 3 refactor, so `finalize(p)`
    # would be a no-op.  RT dispatches pin `tlas.storage` / `blas.storage`,
    # so in-flight dispatches still hold the backing VkManagedBuffer alive
    # through the batch's timeline-gated defer path.
    unsafe_free!(as.storage)
    for p in as.preserves
        p isa LavaArray && unsafe_free!(p)
    end
    empty!(as.preserves)
    as isa LavaTLAS && empty!(as.blases)
end


# The AS is done with when its storage is: `lastwritepassed` on the buffer, and
# a storage already released explicitly has nothing left in flight.
function lastwritepassed(as::Union{LavaBLAS, LavaTLAS}, bq::VulkanBatchQueue, current::UInt64)
    as.storage.buf.freed && return true
    return lastwritepassed(as.storage.buf[]::VkManagedBuffer, bq, current)
end

# What a trace reads of each level: the handle, and the storage it lives in.
# The handle is a `VK.AccelerationStructureKHR` (generic pin), the storage a
# `LavaArray` (retained ref plus buffer pin). Core's `pintrace!` walks the
# levels; these say what holding one level means here.
pin!(o::Closed, t::LavaTLAS) = (pin!(o, t.accel); pin!(o, t.storage); nothing)
pin!(o::Closed, b::LavaBLAS) = (pin!(o, b.accel); pin!(o, b.storage); nothing)
blases(t::LavaTLAS) = t.blases

"""
    drain_deferred_as_frees!(bq::VulkanBatchQueue)

Destroy any LavaBLAS/LavaTLAS in `bq.deferred_as_frees` whose storage
buffer's last write has been reached.  Called at flush sync points.
"""
function drain_deferred_as_frees!(bq::VulkanBatchQueue)
    isempty(bq.deferred_as_frees) && return
    ctx = bq.ctx::VkContext
    if device_lost(ctx)
        lock(bq.deferred_frees_lock) do
            empty!(bq.deferred_as_frees)
        end
        return
    end
    current = query_timeline(bq)
    # Hold the SpinLock for the sweep — shares the lock with deferred_frees
    # since finalizer-thread pushes can target either list.
    lock(bq.deferred_frees_lock) do
        i = 1
        while i <= length(bq.deferred_as_frees)
            as = bq.deferred_as_frees[i]
            if lastwritepassed(as, bq, current)
                destroy_now!(as)
                deleteat!(bq.deferred_as_frees, i)
            else
                i += 1
            end
        end
    end
    return nothing
end

"""
    VulkanAccelBuildContext

Explicit context for acceleration structure builds. Owns a command buffer,
fence, and preserves list. All `build_blas`/`build_tlas` calls take this
as a required parameter. Created by `build_accel!()` which manages the lifecycle.
"""
# `AccelBuildContext` is Mantle's — see `graph/queue.jl`'s neighbour in
# `raytracing/accel.jl`. Both its fields are portable now that `BatchQueue` is
# shared, so there was nothing backend-specific left in it.
const VulkanAccelBuildContext = AccelBuildContext{VulkanBatchQueue{VkContext}}

# Derived accessors so call sites stay readable.
@inline buildcmd(ctx::VulkanAccelBuildContext)   = ctx.into.cmd
@inline as_queue(ctx::VulkanAccelBuildContext)   = ctx.bq.queue
@inline as_device(ctx::VulkanAccelBuildContext)  = ctx.bq.device
@inline as_vkctx(ctx::VulkanAccelBuildContext)   = ctx.bq.ctx::VkContext

"""
    build_blas(ctx::VulkanAccelBuildContext, vertices, indices; opaque=true) -> LavaBLAS

Build a bottom-level acceleration structure. Records into `ctx`'s command buffer.
Must be called inside `build_accel!()`.
"""
function build_blas(ctx::VulkanAccelBuildContext, vertices::Vector{NTuple{3,Float32}}, indices::Vector{UInt32};
                    opaque::Bool=true, allow_update::Bool=false)
    bq = ctx.bq
    dev = as_device(ctx)

    # Upload vertex/index data to device-local buffers (LavaArrays).
    vertex_arr = LavaArray(collect(reinterpret(UInt8, vertices)); bq, extra_usage=AS_INPUT_USAGE)
    index_arr  = LavaArray(collect(reinterpret(UInt8, indices));  bq, extra_usage=AS_INPUT_USAGE)
    vertex_addr = vertex_arr.buf[].address
    index_addr  = index_arr.buf[].address

    n_triangles = UInt32(length(indices) ÷ 3)
    max_vertex = UInt32(length(vertices) - 1)

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

    # Build inputs (vertex + index) must outlive the BLAS -- the driver may
    # retain VAs into them.  Kept as LavaArrays so their own last_write
    # tracks in-flight access and their vk_free! is timeline-gated.
    blas_preserves = LavaArray[vertex_arr, index_arr]
    # Scratch is submit-lifetime only -- pushed into ctx.preserves so build_accel!'s
    # fence-wait (or the ctx-scoped drain) keeps it alive until the submit
    # completes, then its own finalizer frees it.
    push!(ctx.preserves, scratch_arr)

    build_as_on_gpu(ctx, accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, primitive_count=n_triangles)

    addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
    if (ctx.bq.ctx::VkContext).diag.alloc_debug
        push!((ctx.bq.ctx::VkContext).diag.alloc_log,
              (kind=:blas_as, addr=as_addr, size=0, pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end
    blas = LavaBLAS(accel, storage, as_addr, blas_preserves,
                    allow_update,
                    allow_update ? UInt64(sizes.update_scratch_size) : UInt64(0),
                    allow_update ? vertex_arr : nothing,
                    index_addr, n_triangles, max_vertex, geo_flags)
    finalizer(unsafe_free!, blas)
    return blas
end

"""
    refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS,
                vertices::Vector{NTuple{3,Float32}})

Update a BLAS in place via `MODE_UPDATE_KHR` after its vertices moved. Reuses
the AS storage and the original vertex buffer, so the device addresses the
driver retained stay valid.

The BLAS must have been built with `allow_update=true`, and `vertices` must have
the same length as at build time: `MODE_UPDATE_KHR` refits the existing tree and
cannot change topology.

Refitting keeps the tree the mesh was built with, so traversal quality decays as
the geometry moves away from that pose. Callers animating a mesh should rebuild
periodically rather than refit forever.

Errors loudly on misuse.
"""
function refit_blas!(ctx::VulkanAccelBuildContext, blas::LavaBLAS,
                     vertices::Vector{NTuple{3,Float32}})
    blas.allow_update || error(
        "refit_blas!: BLAS was built with allow_update=false; cannot refit. " *
        "Rebuild it via build_blas(...; allow_update=true).")
    blas.update_scratch_size > 0 || error(
        "refit_blas!: cached update_scratch_size is 0; the BLAS is not refit-capable.")
    vertex_arr = blas.vertex_arr
    vertex_arr === nothing && error(
        "refit_blas!: no vertex buffer retained; the BLAS is not refit-capable.")
    bytes = collect(reinterpret(UInt8, vertices))
    length(bytes) == length(vertex_arr) || error(
        "refit_blas!: vertex count changed ($(length(bytes)) bytes vs " *
        "$(length(vertex_arr)) at build). MODE_UPDATE_KHR cannot change " *
        "topology — rebuild the BLAS instead.")

    bq = ctx.bq
    dev = as_device(ctx)

    # Rewrite in place. The address must not move: the driver may hold VAs into
    # this buffer from the original build.
    copyto!(vertex_arr, bytes)

    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
                  UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(blas.update_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    push!(ctx.preserves, scratch_arr)

    geom = TrianglesGeometry(;
        vertex_format=UInt32(VK.FORMAT_R32G32B32_SFLOAT),
        vertex_addr=bda_address(vertex_arr),
        vertex_stride=UInt64(sizeof(NTuple{3,Float32})),
        max_vertex=blas.max_vertex,
        index_type=UInt32(VK.INDEX_TYPE_UINT32),
        index_addr=blas.index_addr,
        transform_addr=UInt64(0))

    # Same barrier reasoning as refit_tlas!: the vertex rewrite above is a
    # transfer write that the AS build has to see, and a previous build/refit of
    # this same AS has to have finished.
    cmd = buildcmd(ctx)
    pre_barrier = VK.MemoryBarrier(
        C_NULL,
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        VK.ACCESS_SHADER_WRITE_BIT |
        VK.ACCESS_TRANSFER_WRITE_BIT,
        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    )
    VK.cmd_pipeline_barrier(
        cmd, [pre_barrier], [], [];
        src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                       VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                       VK.PIPELINE_STAGE_TRANSFER_BIT,
        dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    )

    build_as_on_gpu(ctx, blas.accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags=blas.geo_flags, primitive_count=blas.n_triangles,
        mode=UInt32(1), src_as=blas.accel)   # MODE_UPDATE_KHR, in place
    return blas
end

"""
    build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::Vector{AABB}; opaque=true) -> LavaBLAS

Build a procedural-AABB BLAS from a vector of `AABB` records. The resulting
BLAS is intended for use with inline ray queries (rayQuery + AABB candidate
intersection); accordingly, this function errors loudly if the device does
not support `VK_KHR_ray_query`.
"""
function build_blas_aabb(ctx::VulkanAccelBuildContext, aabbs::Vector{AABB}; opaque::Bool=true)
    (ctx.bq.ctx::VkContext).ray_query_available || error(
        "build_blas_aabb: this Vulkan device does not support " *
        "VK_KHR_ray_query. Procedural-AABB BLASes are only useful with " *
        "ray_query in this codebase. Build on a device that supports it.")

    bq  = ctx.bq
    dev = as_device(ctx)

    # Pack into VkAabbPositionsKHR layout (24 bytes per AABB: minX,Y,Z,maxX,Y,Z).
    n     = length(aabbs)
    bytes = Vector{UInt8}(undef, 24 * n)
    GC.@preserve bytes begin
        p = pointer(bytes)
        for (i, a) in enumerate(aabbs)
            base = p + (i - 1) * 24
            unsafe_store!(Ptr{Float32}(base +  0), a.min[1])
            unsafe_store!(Ptr{Float32}(base +  4), a.min[2])
            unsafe_store!(Ptr{Float32}(base +  8), a.min[3])
            unsafe_store!(Ptr{Float32}(base + 12), a.max[1])
            unsafe_store!(Ptr{Float32}(base + 16), a.max[2])
            unsafe_store!(Ptr{Float32}(base + 20), a.max[3])
        end
    end

    aabb_arr  = LavaArray(bytes; bq, extra_usage=AS_INPUT_USAGE)
    aabb_addr = aabb_arr.buf[].address

    geom       = AABBsGeometry(; aabb_addr, aabb_stride=UInt64(24))
    geo_flags  = opaque ? UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR) : UInt32(0)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)

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

    blas_preserves = LavaArray[aabb_arr]

    build_as_on_gpu(ctx, accel, scratch_addr, geom;
        as_type=UInt32(1), build_flags,
        geo_flags, primitive_count=UInt32(n))

    addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr   = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
    if (ctx.bq.ctx::VkContext).diag.alloc_debug
        push!((ctx.bq.ctx::VkContext).diag.alloc_log,
              (kind=:blas_aabb_as, addr=as_addr, size=0, pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end
    blas = LavaBLAS(accel, storage, as_addr, blas_preserves)
    finalizer(unsafe_free!, blas)
    return blas
end

"""
    build_tlas(ctx::VulkanAccelBuildContext, blas_list; transforms=nothing, custom_indices=nothing) -> LavaTLAS

Build a top-level acceleration structure. Records into `ctx`'s command buffer.
Must be called inside `build_accel!()`.
"""
function build_tlas(ctx::VulkanAccelBuildContext, blas_list::Vector{LavaBLAS};
                    transforms::Union{Nothing, Vector{NTuple{12,Float32}}}=nothing,
                    custom_indices::Union{Nothing, Vector{UInt32}}=nothing,
                    masks::Union{Nothing, Vector{UInt8}}=nothing,
                    allow_update::Bool=false)
    bq = ctx.bq
    dev = as_device(ctx)
    n_instances = length(blas_list)

    instance_data = Vector{UInt8}(undef, 64 * n_instances)
    for i in 1:n_instances
        offset = (i - 1) * 64
        pack_as_instance!(instance_data, offset, blas_list[i].address;
            transform=transforms === nothing ? nothing : transforms[i],
            custom_index=custom_indices === nothing ? UInt32(0) : custom_indices[i],
            mask=masks === nothing ? UInt8(0xff) : masks[i],
        )
    end

    inst_arr = LavaArray(instance_data; bq, extra_usage=AS_INPUT_USAGE)
    inst_addr = inst_arr.buf[].address
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    if allow_update
        build_flags |= UInt32(VK.BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
    end

    geo_buf_inst = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf_inst, 0; geometry_type=:instances, instance_addr=inst_addr)
    sizes = query_as_build_sizes_impl(dev, geo_buf_inst, UInt32(0), build_flags, UInt32(n_instances))

    storage = LavaArray{UInt8,1}(undef, (max(Int(sizes.acceleration_structure_size), 16),);
                                  bq, extra_usage=AS_STORAGE_USAGE)
    accel = VK.AccelerationStructureKHR(dev, VK.AccelerationStructureCreateInfoKHR(
        storage.buf[].buffer, UInt64(0), sizes.acceleration_structure_size,
        VK.ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR))

    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(sizes.build_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)

    # Instance buffer must outlive the HWTLAS (driver may retain VAs).
    # Scratch is submit-lifetime — ctx.preserves + fence wait handles it.
    tlas_preserves = LavaArray[inst_arr]
    push!(ctx.preserves, scratch_arr)

    build_as_on_gpu(ctx, accel, scratch_addr;
        as_type=UInt32(0), build_flags,
        geometry_type=:instances,
        instance_addr=inst_addr,
        primitive_count=UInt32(n_instances))

    unique_blas = unique(blas_list)
    tlas = LavaTLAS(accel, storage, unique_blas, tlas_preserves,
                    allow_update, sizes.update_scratch_size, nothing,
                    Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}())
    finalizer(unsafe_free!, tlas)
    return tlas
end

"""
    build_tlas(ctx::VulkanAccelBuildContext, instance_buf::LavaArray{VulkanInstanceRecord, 1},
               n::Integer; allow_update::Bool=false) -> LavaTLAS

Build a HWTLAS from a GPU-resident instance buffer. `instance_buf[1:n]` must
be valid `VulkanInstanceRecord`s (typically written by `write_grain_instances_kernel`).
No CPU-side packing pass -- the buffer's device address is fed to the Vulkan
build directly. When `allow_update=true`, the HWTLAS is buildable for in-place
refit via `refit_tlas!`.

`instance_buf` must be allocated with `extra_usage = AS_INPUT_USAGE` so the
driver can read it as an AS build input; omitting that flag will cause Vulkan
validation errors at build time.

The instance records name their BLASes only by device address, so the caller
passes the ones `instance_buf` references as `blases`: a trace of the returned
TLAS holds every one of them (`pintrace!`) for as long as it runs, and that is
what keeps a BLAS the owner drops from being destroyed under a trace still
walking it. Ownership stays with the caller — releasing the TLAS drops the
references and destroys nothing below it. Left empty, the TLAS pins only itself,
which is right only for a caller that never frees a BLAS while a trace is out.
"""
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

    geo_buf_inst = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf_inst, 0; geometry_type=:instances, instance_addr=inst_addr)
    sizes = query_as_build_sizes_impl(dev, geo_buf_inst, UInt32(0), build_flags, UInt32(n_instances))

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

    build_as_on_gpu(ctx, accel, scratch_addr;
        as_type=UInt32(0), build_flags,
        geometry_type=:instances,
        instance_addr=inst_addr,
        primitive_count=UInt32(n_instances))

    # The BLASes the instance buffer names, so a trace pins them. This was
    # `LavaBLAS[]` with a comment saying the VulkanTLAS pinned them "at the
    # higher level" — it did not: every trace path pinned `tlas.blases`, which
    # was this empty list, so a BLAS `sync!` dropped could be destroyed under a
    # trace still walking it. `test_pintrace.jl` pins the count.
    tlas = LavaTLAS(accel, storage, unique(blases), tlas_preserves,
                    allow_update, sizes.update_scratch_size, instance_buf,
                    Dict{UInt64, Tuple{VK.DescriptorPool, VK.DescriptorSet}}())
    finalizer(unsafe_free!, tlas)
    return tlas
end

const VkBRI = VK.VulkanCore.LibVulkan.VkAccelerationStructureBuildRangeInfoKHR

"""
    refit_tlas!(ctx::VulkanAccelBuildContext, tlas::LavaTLAS,
                instance_buf::LavaArray{VulkanInstanceRecord, 1}, n::Integer)

Update a HWTLAS in place via `MODE_UPDATE_KHR`. Reuses `tlas.accel`'s storage
(no new allocation), uses the cached `tlas.update_scratch_size`, and reads
fresh instance data from `instance_buf[1:n]`.

The HWTLAS must have been built with `allow_update=true`. The instance count
`n` MUST equal the count used at build time -- `MODE_UPDATE_KHR` cannot
change topology.

Errors loudly on misuse.
"""
function refit_tlas!(ctx::VulkanAccelBuildContext, tlas::LavaTLAS,
                     instance_buf::LavaArray{VulkanInstanceRecord, 1}, n::Integer)
    tlas.allow_update || error(
        "refit_tlas!: HWTLAS was built with allow_update=false; cannot refit. " *
        "Rebuild the HWTLAS via build_tlas(...; allow_update=true).")
    tlas.update_scratch_size > 0 || error(
        "refit_tlas!: cached update_scratch_size is 0; the HWTLAS is not refit-capable.")

    bq = ctx.bq
    dev = as_device(ctx)
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

    # Re-pack the geometry buffer (instance addr unchanged but spec requires it
    # in the BuildGeometryInfo for UPDATE too).
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0; geometry_type=:instances, instance_addr=inst_addr)

    bgi_buf = zeros(UInt8, 80)
    c_range = VkBRI(UInt32(n_instances), UInt32(0), UInt32(0), UInt32(0))

    fptr = VK.function_pointer(dev, "vkCmdBuildAccelerationStructuresKHR")
    cmd = buildcmd(ctx)

    # Pre-barrier: cover both prior AS builds AND prior compute/transfer writes
    # to instance_buf. The latter is the per-frame hot path where a compute
    # kernel writes instance records and refit reads them; without this,
    # compute writes are not guaranteed visible to the AS-build read.
    pre_barrier = VK.MemoryBarrier(
        C_NULL,
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        VK.ACCESS_SHADER_WRITE_BIT |
        VK.ACCESS_TRANSFER_WRITE_BIT,
        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    )
    VK.cmd_pipeline_barrier(
        cmd, [pre_barrier], [], [];
        src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                       VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                       VK.PIPELINE_STAGE_TRANSFER_BIT,
        dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    )

    GC.@preserve geo_buf bgi_buf c_range begin
        geo_ptr = pointer(geo_buf)
        # mode=UPDATE, src=dst=tlas.accel for in-place refit.
        pack_build_geometry_info!(bgi_buf, 0;
            as_type=UInt32(0),                           # TOP_LEVEL
            mode=UInt32(1),                               # MODE_UPDATE_KHR
            build_flags,
            src_as=tlas.accel.vks,
            dst_as=tlas.accel.vks,
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

    push!(ctx.preserves, (geo_buf, bgi_buf, c_range))
    return tlas
end

# ── Internal helpers ──

"""Pack a VkAccelerationStructureInstanceKHR into `buf` at byte `offset`.
Layout (64 bytes):
  - [0:47]  transform: VkTransformMatrixKHR (3×4 float32, row-major)
  - [48:50] instanceCustomIndex (24 bits)
  - [51]    mask (8 bits)
  - [52:54] instanceShaderBindingTableRecordOffset (24 bits)
  - [55]    flags (8 bits)
  - [56:63] accelerationStructureReference (uint64)
"""
function pack_as_instance!(buf::Vector{UInt8}, offset::Int, blas_addr::UInt64;
                            transform::Union{Nothing, NTuple{12,Float32}}=nothing,
                            custom_index::UInt32=UInt32(0),
                            mask::UInt8=0xff,
                            sbt_offset::UInt32=UInt32(0),
                            flags::UInt8=0x00)
    # Identity transform if not specified (row-major 3×4)
    if transform === nothing
        transform = (1f0, 0f0, 0f0, 0f0,
                     0f0, 1f0, 0f0, 0f0,
                     0f0, 0f0, 1f0, 0f0)
    end

    # Transform: 12 × Float32 = 48 bytes
    for j in 1:12
        unsafe_store!(Ptr{Float32}(pointer(buf, offset + 1 + (j-1)*4)), transform[j])
    end

    # instanceCustomIndex (24 bits) + mask (8 bits) = 4 bytes at offset 48
    val_48 = (custom_index & 0x00FFFFFF) | (UInt32(mask) << 24)
    unsafe_store!(Ptr{UInt32}(pointer(buf, offset + 49)), val_48)

    # instanceShaderBindingTableRecordOffset (24 bits) + flags (8 bits) = 4 bytes at offset 52
    val_52 = (sbt_offset & 0x00FFFFFF) | (UInt32(flags) << 24)
    unsafe_store!(Ptr{UInt32}(pointer(buf, offset + 53)), val_52)

    # accelerationStructureReference = BLAS device address (8 bytes at offset 56)
    unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 57)), blas_addr)

    return nothing
end

"""Create a HOST_VISIBLE buffer pool for AS input data (vertices/indices).

Unlike a typical LavaArray alloc which uploads specific data, this allocates
an empty mappable buffer of `nbytes` for the caller to fill via map/memcpy/unmap.
Returns (buffer, memory, base_device_address).
"""
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

    # DEBUG: log AS-input host-visible buffer addresses (separate memory heap
    # from device-local pool), to correlate with cross-scene cascade fault
    # addresses in the 0x8000_xxxx_xxxx range.
    if ctx.diag.alloc_debug
        push!(ctx.diag.alloc_log,
              (kind=:as_input_pool, addr=addr, size=Int(nbytes), pool=false, mtype=-1, unified=false, usage=UInt32(0)))
    end

    return buf, memory, addr
end

# Buffer usage bits for AS build paths.  AS-build scratch needs a specific
# BDA alignment (`ctx.as_scratch_align`); pass `scratch=true` to LavaArray
# alongside `extra_usage=AS_SCRATCH_USAGE` to apply it.
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

# Legacy Symbol-dispatch method; used by the `:instances` path in build_tlas.
# The instances path is not being migrated in this task -- left intact.
function pack_geometry!(buf::Vector{UInt8}, offset::Int;
        geometry_type::Symbol,
        vertex_format::UInt32=UInt32(0), vertex_addr::UInt64=UInt64(0),
        vertex_stride::UInt64=UInt64(0), max_vertex::UInt32=UInt32(0),
        index_type::UInt32=UInt32(0), index_addr::UInt64=UInt64(0),
        transform_addr::UInt64=UInt64(0),
        instance_addr::UInt64=UInt64(0),
        geo_flags::UInt32=UInt32(0))
    p = pointer(buf, offset + 1)
    unsafe_store!(Ptr{Int32}(p), VK_STYPE_GEO)           # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)         # pNext @ 8

    q = p + 24  # geometry union start (only used by the :instances branch)
    if geometry_type == :triangles
        geom = TrianglesGeometry(vertex_format, vertex_addr, vertex_stride,
                                 max_vertex, index_type, index_addr, transform_addr)
        pack_geometry!(buf, offset, geom; geo_flags)
        return nothing
    elseif geometry_type == :instances
        unsafe_store!(Ptr{UInt32}(p + 16), UInt32(2))       # geometryType = INSTANCES @ 16
        unsafe_store!(Ptr{Int32}(q), VK_STYPE_INST)                 # sType @ 0
        unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)              # pNext @ 8
        unsafe_store!(Ptr{UInt32}(q + 16), UInt32(0))                # arrayOfPointers @ 16
        unsafe_store!(Ptr{UInt64}(q + 24), instance_addr)            # data @ 24
    else
        error("Unknown geometry type: $geometry_type")
    end
    unsafe_store!(Ptr{UInt32}(p + 88), geo_flags)           # flags @ 88
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

# Legacy keyword-dispatch overload; used by build_tlas (:instances path).
function query_as_build_sizes(dev::VK.Device; as_type::UInt32,
        build_flags::UInt32=UInt32(0), max_primitive_count::UInt32=UInt32(0),
        kwargs...)
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0; kwargs...)
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

"""
    build_accel!(f, bq)

Build acceleration structures in a single GPU submission. The callback
receives an `VulkanAccelBuildContext` that must be passed to `build_blas`/`build_tlas`.

# Example
```julia
blases, tlas = build_accel!(bq) do ctx
    bs = [build_blas(ctx, verts, idxs) for (verts, idxs) in meshes]
    tlas = build_tlas(ctx, bs)
    return (bs, tlas)
end
```

BLAS device addresses are available immediately after `build_blas` returns
(even before the GPU build executes), so `build_tlas` can reference them.
"""
function build_accel!(f, bq::VulkanBatchQueue)
    # The builds go into a one-shot of their own, on the timeline like every
    # other submission — it used to be a dedicated command buffer and a fence
    # beside the queue, submitted by a second route. The one-shot opens with
    # the global barrier, so the vertex, index and instance buffers prior
    # dispatches wrote are visible to the build without the flush that stood
    # here.
    preserves = Any[]
    result = Ref{Any}(nothing)
    tok = oneshot!(bq; tag = :accel) do e
        result[] = f(AccelBuildContext(bq, e, preserves))
        # Final barrier: make all AS writes visible to RT shader reads
        post_barrier = VK.MemoryBarrier(
            C_NULL,
            VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK.ACCESS_SHADER_READ_BIT,
        )
        VK.cmd_pipeline_barrier(
            e.cmd, [post_barrier], [], [];
            src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                           VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        )
    end
    # Callers used to wait on the fence, so the build is complete when this
    # returns: the token is what they wait for now — on `bq`'s timeline, which
    # is the caller's queue and need not be the device's primary one.
    waitfor!(bq, tok)

    # Inputs that must outlive the GPU submit (vertex/index for BLAS, instance
    # buffer for HWTLAS) are owned by LavaBLAS/LavaTLAS via their preserves.
    # Scratch buffers in `preserves` are submit-scoped and have been through
    # the wait above, so eagerly release them here via the DataRef refcount
    # path (LavaArray itself has no finalizer; lifetime is the DataRef's sole
    # responsibility after the Phase 3 refactor). Without this explicit
    # release, each build_accel! leaks ~hundreds of MB of scratch per call —
    # multi-frame renders (dolphin HQ video) hit VRAM OOM.
    for p in preserves
        if p isa LavaArray
            unsafe_free!(p)
        end
    end
    empty!(preserves)
    return result[]
end

"""Record an acceleration structure build into an VulkanAccelBuildContext's command buffer.

Always records into the one-shot `build_accel!()` opened (`buildcmd(ctx)`),
which manages the full lifecycle: open, record builds, submit, wait.
"""
function build_as_on_gpu(ctx::VulkanAccelBuildContext, accel::VK.AccelerationStructureKHR,
                         scratch_addr::UInt64, geom::GeometryType;
                         as_type::UInt32, build_flags::UInt32=UInt32(0),
                         primitive_count::UInt32=UInt32(0),
                         geo_flags::UInt32=UInt32(0),
                         mode::UInt32=UInt32(0),
                         src_as::Union{Nothing, VK.AccelerationStructureKHR}=nothing)
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0, geom; geo_flags)
    build_as_on_gpu_impl(ctx, accel, scratch_addr, geo_buf, as_type, build_flags,
                         primitive_count, mode, src_as)
end

# Legacy keyword-dispatch overload; used by the `:instances` path in build_tlas.
function build_as_on_gpu(ctx::VulkanAccelBuildContext, accel::VK.AccelerationStructureKHR,
                         scratch_addr::UInt64;
                         as_type::UInt32, build_flags::UInt32=UInt32(0),
                         primitive_count::UInt32=UInt32(0),
                         kwargs...)
    geo_buf = zeros(UInt8, C_SIZEOF_AS_GEOMETRY_KHR)
    pack_geometry!(geo_buf, 0; kwargs...)
    build_as_on_gpu_impl(ctx, accel, scratch_addr, geo_buf, as_type, build_flags,
                         primitive_count, UInt32(0), nothing)
end

# `mode` is MODE_BUILD_KHR (0) or MODE_UPDATE_KHR (1); an update also needs
# `src_as`, which is the same AS for an in-place refit.
function build_as_on_gpu_impl(ctx::VulkanAccelBuildContext, accel::VK.AccelerationStructureKHR,
                               scratch_addr::UInt64, geo_buf::Vector{UInt8},
                               as_type::UInt32, build_flags::UInt32, primitive_count::UInt32,
                               mode::UInt32=UInt32(0),
                               src_as::Union{Nothing, VK.AccelerationStructureKHR}=nothing)
    cmd = buildcmd(ctx)

    # packed by caller so typed and legacy dispatch paths share this body

    # Pack build geometry info (80 bytes)
    bgi_buf = zeros(UInt8, 80)

    # Pack range info (16 bytes)
    c_range = VkBRI(primitive_count, UInt32(0), UInt32(0), UInt32(0))

    fptr = VK.function_pointer(as_device(ctx), "vkCmdBuildAccelerationStructuresKHR")

    # Synchronize prior AS builds before this build command.
    pre_barrier = VK.MemoryBarrier(
        C_NULL,
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    )
    VK.cmd_pipeline_barrier(
        cmd, [pre_barrier], [], [];
        src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    )

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

"""
    build_blas_pooled(all_vertices, all_indices) -> Vector{LavaBLAS}

Build multiple BLASes using pooled memory allocation. All vertex/index data
goes into a single HOST_VISIBLE buffer, all AS storage into a single
DEVICE_LOCAL buffer, and one scratch buffer is reused. This reduces thousands
of Vulkan allocations to ~6 regardless of mesh count.

`all_vertices[i]::Vector{NTuple{3,Float32}}` and `all_indices[i]::Vector{UInt32}`
provide per-BLAS geometry. Returns one `LavaBLAS` per input.
"""
function build_blas_pooled(all_vertices::Vector{Vector{NTuple{3,Float32}}},
                           all_indices::Vector{Vector{UInt32}};
                           bq::VulkanBatchQueue)
    n_blas = length(all_vertices)
    n_blas == 0 && return LavaBLAS[]
    @assert length(all_indices) == n_blas

    ctx = bq.ctx::VkContext
    dev = ctx.device

    vfmt = UInt32(VK.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(VK.INDEX_TYPE_UINT32)
    build_flags = UInt32(VK.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    geo_flags = UInt32(VK.GEOMETRY_OPAQUE_BIT_KHR)

    # Pass 1: Query build sizes and compute pool layout
    vertex_offsets = Vector{UInt64}(undef, n_blas)
    index_offsets = Vector{UInt64}(undef, n_blas)
    as_offsets = Vector{UInt64}(undef, n_blas)
    as_sizes_list = Vector{UInt64}(undef, n_blas)
    max_scratch_size = UInt64(0)
    input_cursor = UInt64(0)
    as_cursor = UInt64(0)

    for i in 1:n_blas
        n_tris = UInt32(length(all_indices[i]) ÷ 3)
        max_vertex = UInt32(length(all_vertices[i]) - 1)

        # Addresses are 0 for size queries; only counts and formats matter.
        geom_i = TrianglesGeometry(; vertex_format=vfmt, vertex_addr=UInt64(0),
                                     vertex_stride=vstride, max_vertex,
                                     index_type=itype, index_addr=UInt64(0))
        sizes = query_as_build_sizes(dev, geom_i;
            as_type=UInt32(1), build_flags,
            geo_flags, max_primitive_count=n_tris)

        vertex_offsets[i] = input_cursor
        vbytes = UInt64(length(all_vertices[i]) * sizeof(NTuple{3,Float32}))
        input_cursor += (vbytes + 15) & ~UInt64(15)

        index_offsets[i] = input_cursor
        ibytes = UInt64(length(all_indices[i]) * sizeof(UInt32))
        input_cursor += (ibytes + 15) & ~UInt64(15)

        as_offsets[i] = as_cursor
        as_sizes_list[i] = sizes.acceleration_structure_size
        as_cursor += (sizes.acceleration_structure_size + 255) & ~UInt64(255)

        max_scratch_size = max(max_scratch_size, sizes.build_scratch_size)
    end

    total_input_bytes = input_cursor
    total_as_bytes = as_cursor

    # Pass 2: Allocate pooled buffers (3 allocations total)
    input_buf, input_mem, input_base_addr = create_as_input_pool(ctx, max(total_input_bytes, 16))
    as_pool_arr = LavaArray{UInt8,1}(undef, (max(Int(total_as_bytes), 16),);
                                      bq, extra_usage=AS_STORAGE_USAGE)
    as_pool_buf = as_pool_arr.buf[].buffer
    as_pool_mem = as_pool_arr.buf[].memory
    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(max_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    scratch_buf = scratch_arr.buf[].buffer
    scratch_mem = scratch_arr.buf[].memory

    # Pass 3: Upload all vertex/index data into the input pool
    mapped_ptr = throw_if_error(ctx, "vkMapMemory",
        VK.map_memory(dev, input_mem, 0, total_input_bytes))
    for i in 1:n_blas
        vdata = all_vertices[i]
        vbytes = length(vdata) * sizeof(NTuple{3,Float32})
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + vertex_offsets[i]),
                       Ptr{UInt8}(pointer(vdata)), vbytes)
        idata = all_indices[i]
        ibytes = length(idata) * sizeof(UInt32)
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + index_offsets[i]),
                       Ptr{UInt8}(pointer(idata)), ibytes)
    end
    VK.unmap_memory(dev, input_mem)

    # Pass 4: Create AS objects and build all BLASes
    hw_blas_list = Vector{LavaBLAS}(undef, n_blas)
    for i in 1:n_blas
        as_ci = VK.AccelerationStructureCreateInfoKHR(
            as_pool_buf, as_offsets[i], as_sizes_list[i],
            VK.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        )
        accel = VK.AccelerationStructureKHR(dev, as_ci)
        addr_info = VK.AccelerationStructureDeviceAddressInfoKHR(accel)
        as_addr = VK.get_acceleration_structure_device_address_khr(dev, addr_info)
        if (bq.ctx::VkContext).diag.alloc_debug
            push!((bq.ctx::VkContext).diag.alloc_log,
                  (kind=:blas_as_pool, addr=as_addr, size=0, pool=true, mtype=-1, unified=false, usage=UInt32(0)))
        end
        # All pooled BLASes share `as_pool_arr` as their storage. They
        # reference the same LavaArray so its VkManagedBuffer's last_write
        # tracks in-flight use of any of them collectively.
        blas = LavaBLAS(accel, as_pool_arr, as_addr, LavaArray[])
        # No finalizer here: destroying the AccelerationStructureKHR on each
        # BLAS individually is fine, but the shared storage/pool is only
        # released when *every* LavaBLAS sharing it has retired and dropped
        # its ref. Julia GC handles that via the as_pool_arr reference
        # count on each BLAS.
        hw_blas_list[i] = blas
    end

    GC.@preserve input_buf input_mem scratch_arr as_pool_arr begin
        build_accel!(bq) do as_ctx
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
                if i < n_blas
                    scratch_barrier = VK.MemoryBarrier(
                        C_NULL,
                        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    )
                    VK.cmd_pipeline_barrier(
                        buildcmd(as_ctx), [scratch_barrier], [], [];
                        src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    )
                end
            end
        end
    end

    # Eagerly free temporary buffers to reclaim VRAM immediately.
    # The AS build is complete (build_accel! waits for GPU), so these are no longer needed.
    # Without explicit cleanup, they linger until GC runs, wasting hundreds of MB on
    # large scenes (e.g., crown scene: ~170MB input + ~50MB scratch).
    # Free temporaries eagerly.  `scratch_arr` is a LavaArray — release
    # through the DataRef refcount path (LavaArray has no finalizer after
    # Phase 3).  `input_buf`/`input_mem` are raw Vulkan handles from
    # `create_as_input_pool` — the build_accel! fence wait already drained them,
    # so `finalize(...)` on their `destructor` closures is safe.
    unsafe_free!(scratch_arr)
    finalize(input_buf); finalize(input_mem)

    return hw_blas_list
end

# ── Raycore-compatible bridge functions ──

"""
    build_blas_from_primitives(bq, primitives) -> LavaBLAS

Build a hardware BLAS from an array of triangle primitives.
Each primitive must have a `.vertices` field with 3 vertex positions
(e.g., `SVector{3, Point3f}`). Works with Raycore's `Triangle{T}` type.

Primitives on GPU (LavaArray, etc.) are downloaded to CPU automatically.
"""
function build_blas_from_primitives(bq::VulkanBatchQueue, primitives; opaque::Bool=true)
    cpu_prims = to_cpu_vector(primitives)
    n_tris = length(cpu_prims)

    # Extract vertex positions: 3 vertices per triangle
    vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        verts = cpu_prims[i].vertices
        for j in 1:3
            v = verts[j]
            vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end

    # Sequential indices (each triangle has 3 unique vertices)
    indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        indices[i+1] = UInt32(i)
    end

    return build_accel!(bq) do ctx
        build_blas(ctx, vertices, indices; opaque)
    end
end

"""
    build_hw_accel_from_tlas(tlas; ctx, bq = ctx.default_bq)
        -> (hw_tlas, triangle_data, blas_offsets, per_instance_tri_offsets)

Build hardware acceleration structures from a Raycore-compatible HWTLAS.

Uses pooled memory allocation: all vertex/index data goes into a single
HOST_VISIBLE buffer, all BLAS AS storage into a single DEVICE_LOCAL buffer,
and one scratch buffer is reused across all builds. This reduces ~3000+
individual Vulkan allocations to ~6 regardless of mesh count.

The HWTLAS must have:
- `.blas_array`: indexable collection of BLAS objects, each with `.primitives`
- `.instances`: array of instance descriptors with `.blas_index` (1-based),
  `.transform` (4×4 matrix, local-to-world)

Returns:
- `hw_tlas::LavaTLAS` — Vulkan top-level acceleration structure
- `triangle_data::Vector` — CPU vector of all primitives (caller uploads to GPU)
- `blas_offsets::Vector{UInt32}` — per-BLAS offset into triangle_data (0-based)

After a ray trace, look up the hit triangle:
```
tri = triangle_data[blas_offsets[instance_custom_index + 1] + primitive_id + 1]
```
where `instance_custom_index` = BLAS index (0-based) set by this function.
"""
function build_hw_accel_from_tlas(tlas;
                                  ctx::VkContext,
                                  bq::VulkanBatchQueue=ctx.default_bq)
    instances = to_cpu_vector(tlas.instances)
    blas_array = to_cpu_vector(tlas.blas_array)

    n_blas = length(blas_array)
    n_instances = length(instances)
    dev = ctx.device

    # ── Pass 1: Collect all geometry on CPU ──
    cpu_prims_list = Vector{Any}(undef, n_blas)
    blas_offsets = Vector{UInt32}(undef, n_blas)
    all_primitives = []
    prim_offset = UInt32(0)

    # Per-BLAS vertex/index arrays
    all_vertices = Vector{Vector{NTuple{3,Float32}}}(undef, n_blas)
    all_indices = Vector{Vector{UInt32}}(undef, n_blas)

    for i in 1:n_blas
        cpu_prims_list[i] = to_cpu_vector(blas_array[i].primitives)
        blas_offsets[i] = prim_offset
        prim_offset += UInt32(length(cpu_prims_list[i]))
        append!(all_primitives, cpu_prims_list[i])

        # Extract vertices/indices for this BLAS
        n_tris = length(cpu_prims_list[i])
        verts = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
        idxs = Vector{UInt32}(undef, n_tris * 3)
        for j in 1:n_tris
            vs = cpu_prims_list[i][j].vertices
            for k in 1:3
                v = vs[k]
                verts[(j-1)*3 + k] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
                idxs[(j-1)*3 + k] = UInt32((j-1)*3 + k - 1)
            end
        end
        all_vertices[i] = verts
        all_indices[i] = idxs
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
    as_pool_mem = as_pool_arr.buf[].memory

    # Scratch buffer: single allocation, reused for all builds
    scratch_arr = LavaArray{UInt8,1}(undef, (max(Int(max_scratch_size), 16),);
                                      bq, extra_usage=AS_SCRATCH_USAGE, scratch=true)
    scratch_addr = bda_address(scratch_arr)
    scratch_buf = scratch_arr.buf[].buffer
    scratch_mem = scratch_arr.buf[].memory

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
                if i < n_blas
                    scratch_barrier = VK.MemoryBarrier(
                        C_NULL,
                        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                        VK.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    )
                    VK.cmd_pipeline_barrier(
                        buildcmd(as_ctx), [scratch_barrier], [], [];
                        src_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        dst_stage_mask=VK.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    )
                end
            end

            # Build instance list for HWTLAS.
            #
            # Vulkan reads two per-instance 24-bit fields: `gl_InstanceID`
            # (0-based instance array position, always present) and
            # `gl_InstanceCustomIndexEXT` (user-supplied override).
            #
            # Under our semantics: `custom_indices[i]` = the interface
            # override `inst.instance_id` (forwarded as the 5th closest_hit
            # return value in SW; here handed straight to the shader as
            # gl_InstanceCustomIndexEXT).  Callers look up the per-instance
            # triangle offset via `gl_InstanceID → per_instance_tri_offsets`.
            hw_blas_refs = Vector{LavaBLAS}(undef, n_instances)
            transforms = Vector{NTuple{12,Float32}}(undef, n_instances)
            custom_indices = Vector{UInt32}(undef, n_instances)

            for i in 1:n_instances
                inst = instances[i]
                blas_idx = Int(inst.blas_index)
                hw_blas_refs[i] = hw_blas_list[blas_idx]
                transforms[i] = mat4_to_vk_transform(inst.transform)
                custom_indices[i] = inst.instance_id
            end

            return build_tlas(as_ctx, hw_blas_refs; transforms, custom_indices)
        end
    end

    # Eagerly free temporary buffers to reclaim VRAM immediately.
    # GPU build is complete (build_accel! waits for fences). AS storage pool
    # stays alive via LavaBLAS.buffer references.
    # Free temporaries eagerly.  `scratch_arr` is a LavaArray — release
    # through the DataRef refcount path (LavaArray has no finalizer after
    # Phase 3).  `input_buf`/`input_mem` are raw Vulkan handles from
    # `create_as_input_pool` — the build_accel! fence wait already drained them,
    # so `finalize(...)` on their `destructor` closures is safe.
    unsafe_free!(scratch_arr)
    finalize(input_buf); finalize(input_mem)

    # Convert all_primitives to typed vector
    if !isempty(all_primitives)
        T = typeof(all_primitives[1])
        typed_prims = T[p for p in all_primitives]
    else
        typed_prims = Any[]
    end

    # Per-instance triangle-offset table.  Indexed by `gl_InstanceID`
    # (0-based), each entry is the offset of that instance's BLAS in the
    # flat `typed_prims` array.  Lets the closest-hit callback find its
    # triangle with a single indexed load: `typed_prims[off + prim_id + 1]`.
    per_instance_tri_offsets = UInt32[blas_offsets[Int(instances[i].blas_index)] for i in 1:n_instances]

    return (hw_tlas, typed_prims, blas_offsets, per_instance_tri_offsets)
end


"""Download GPU array to CPU Vector, or return as-is if already a CPU collection."""
function to_cpu_vector(x)
    x isa Vector && return x
    x isa Tuple && return collect(x)
    # Try Array() for GPU arrays (LavaArray, CuArray, ROCArray, etc.).  Only
    # fall through on MethodError — every other failure (DEVICE_LOST during
    # readback, OOM in staging buffer, …) must propagate so the caller sees
    # the real cause.
    try
        return Array(x)
    catch ex
        ex isa MethodError || rethrow()
        return collect(x)
    end
end

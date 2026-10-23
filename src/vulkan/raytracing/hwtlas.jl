# VulkanTLAS, the Vulkan `BatchedTLAS`. Every push! makes a batch owning a
# `LavaArray{Raycore.InstanceRecord, 1}` that kernels may write
# (`Raycore.instance_buffer`); `sync!` rebuilds or refits from the dirty flags and
# writes the `VulkanInstanceRecord`s the driver builds from.

import Raycore
import Adapt
using GeometryBasics
using StaticArrays: SVector, SMatrix
using Base: @propagate_inbounds
import KernelAbstractions
const KA = KernelAbstractions

const Mat4f = SMatrix{4, 4, Float32, 16}

# `unsafe_free!` has no method for `nothing`; callers check optional backings.

const VulkanInstanceBatch{Tri} =
    InstanceBatch{Tri,LavaBLAS,LavaArray{Raycore.InstanceRecord,1}}

"""
    VulkanTLAS{Tri} <: BatchedTLAS{Tri}

The Vulkan hardware TLAS, typed on the shading triangle `Tri` (usually
`Raycore.Triangle{UInt32}`). `push!(hwtlas, mesh, transform)` adds geometry and
`Raycore.sync!(hwtlas)` builds it: a full rebuild after `push!`/`delete!`, a
`MODE_UPDATE_KHR` refit after `update_transforms!`, or after a kernel wrote
`Raycore.instance_buffer` and `Raycore.refit!` was called.

`sync!` owns `hwtlas.static_tlas` and may replace it when a buffer is reallocated,
so read it, or `Adapt.adapt(backend, hwtlas)`, per dispatch; a cached
`AdaptedAccel` goes stale across mutations. `sync!` waits for its own build
(`build_accel!`); old structures go through `unsafe_free!`, which retires them
once their last submission passes, safe because a trace holds every structure it
reads. `Raycore.wait_for_gpu!(hwtlas)`
waits on `hwtlas.bq`.
"""
mutable struct VulkanTLAS{Tri} <: BatchedTLAS{Tri}
    backend::LavaBackend

    # Channel for every AS build and trace of this TLAS.
    bq::SubmitChannel{<:VulkanQueue}

    # Geometry, accumulated by push!.
    blas_list::Vector{LavaBLAS}

    # One batch of N instances of one BLAS per push!; order and handles are core's
    # (`raytracing/batches.jl`).
    instances::InstanceBatches{VulkanInstanceBatch{Tri}}

    # Host-side bounds, updated by push!.
    root_aabb::Raycore.Bounds3

    # Built by sync!.
    hw_tlas::Union{Nothing, LavaTLAS}
    tri_gpu::Union{Nothing, LavaArray{Tri, 1}}
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}
    # Driver records, every batch's in order, written by `writeinstances!`. Fresh
    # per rebuild (the `LavaTLAS` owns it), written in place by a refit.
    combined_instance_buf::Union{Nothing, LavaArray{VulkanInstanceRecord, 1}}
    # The same instances as `Raycore.InstanceRecord`s, which hits read transforms
    # from; written in place by a refit so a plan's adapted form sees it.
    combined_records::Union{Nothing, LavaArray{Raycore.InstanceRecord, 1}}

    # Adapted form owned by sync!; `nothing` before the first sync!.
    static_tlas::Any   # Union{Nothing, AdaptedAccel{VulkanTLAS{Tri}}}

    # Topology dirty: a push!/delete! changed the batch list, full rebuild needed.
    dirty::Bool
    # Transforms dirty: queued updates or kernel-written records; a refit suffices.
    transforms_dirty::Bool

    # Queued transform updates (Mat3x4f device arrays) by handle, last write wins;
    # consumed by sync!.
    pending_updates::Dict{Raycore.TLASHandle, Any}

    # The scene's procedural payload or `nothing`, passed to the traversal
    # through `AdaptedAccel` (`procedural_candidate` in `raytracing/accel.jl`).
    procedural::Any
end

# The portable constructor, for callers that do not name a backend.
HWTLAS{Tri}(backend::LavaBackend; kw...) where {Tri} = VulkanTLAS{Tri}(backend; kw...)

"""
    VulkanTLAS{Tri}(backend::LavaBackend; bq=backend.dispatch_bq) -> VulkanTLAS{Tri}

An empty VulkanTLAS whose builds and traces run on `bq`.
"""
function VulkanTLAS{Tri}(backend::LavaBackend; bq::SubmitChannel{<:VulkanQueue}=backend.dispatch_bq) where {Tri}
    VulkanTLAS{Tri}(
        backend, bq,
        LavaBLAS[],
        InstanceBatches{VulkanInstanceBatch{Tri}}(),
        Raycore.Bounds3(),
        nothing, nothing, nothing,
        nothing,                  # combined_instance_buf
        nothing,                  # combined_records
        nothing,
        true,                     # dirty
        false,                    # transforms_dirty
        Dict{Raycore.TLASHandle, Any}(),  # pending_updates
        nothing,                  # procedural
    )
end

"""
    VulkanTLAS(backend::LavaBackend; bq=backend.dispatch_bq) -> VulkanTLAS{Triangle{UInt32}}

A `VulkanTLAS{Raycore.Triangle{UInt32}}`.
"""
VulkanTLAS(backend::LavaBackend; bq::SubmitChannel{<:VulkanQueue}=backend.dispatch_bq) =
    VulkanTLAS{Raycore.Triangle{UInt32}}(backend; bq)

"""The host-side AdaptedAccel of a synced VulkanTLAS."""
function AdaptedAccel(hwtlas::VulkanTLAS{Tri}) where Tri
    # `nothing` for `scene`: Vulkan binds the TLAS as a descriptor.
    AdaptedAccel(hwtlas, hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri),
                 nothing, hwtlas.procedural, hwtlas.combined_records)
end

# The walk stops here for every owner (`::Closed`, recordings included): the
# arrays are held through `AdaptedAccel`'s fields, and recursing into the TLAS
# reaches the context, whose cyclic closures overflow the stack.
@inline holdleaves!(::Closed, ::VulkanTLAS) = nothing

"""
    Adapt.adapt_structure(to, hwtlas::VulkanTLAS) -> AdaptedAccel

Sync, then return `hwtlas.static_tlas`: the host form, which keeps the `VulkanTLAS`
in `accel.hwtlas` for descriptor binding. Only a `LavaAdaptor` makes the device
form (method below).
"""
function Adapt.adapt_structure(to, hwtlas::VulkanTLAS{Tri}) where Tri
    Raycore.sync!(hwtlas)
    return hwtlas.static_tlas::AdaptedAccel
end

"""Drop the host owner and adapt traversal arrays; keep the out-of-band scene value.
Descriptor discovery uses the original unadapted hwtlas reference."""
Adapt.adapt_structure(to::LavaAdaptor, accel::AdaptedAccel) =
    adaptaccel(to, accel, identity)

"""
    Raycore.wait_for_gpu!(hwtlas::VulkanTLAS) -> hwtlas

Block until all work on `hwtlas.bq` has completed (`flush!`, not the backend-wide
`KA.synchronize`).
"""
function Raycore.wait_for_gpu!(hwtlas::VulkanTLAS)
    flush!(hwtlas.bq)
    return hwtlas
end

# Native primitives used by core's batch registration and mutation algorithms.
function buildgeometry!(t::VulkanTLAS, vertices)
    indices = UInt32.(0:length(vertices)-1)
    return build_accel!(t.bq) do ctx
        build_blas(ctx, vertices, indices)
    end
end

instancearray(t::VulkanTLAS, records::AbstractVector) = LavaArray(records; bq=t.bq)
instancearray(::VulkanTLAS, records::LavaArray) = records

# Queue the already validated buffer. Last write wins per live handle; sync!
# emits updates and the records-to-native conversion in the same submission.
function updatebatch!(t::VulkanTLAS, batch, transforms)
    t.pending_updates[batch.handle] = transforms
    return nothing
end

# Reuse `prev` for `data` by resize and copy, else allocate on `bq` (the TLAS's
# queue). At least one element: a trace binds these, a zero-length buffer has no
# device address, and an empty scene is valid (the element is never read).
function reuse_or_alloc(prev, data::AbstractArray{T}, bq) where T
    n = max(length(data), 1)
    if prev isa LavaArray{T}
        resize!(prev, n)
        isempty(data) || copyto!(prev, 1, data, 1, length(data))
        return prev
    end
    isempty(data) && return LavaArray{T, 1}(undef, n; bq)
    return LavaArray(data; bq)
end

# The driver record: transform, id (low 24 bits), mask ANDed with `keep`, and the
# batch's geometry as `(BLAS address, SBT offset)`.
@inline instancerecord(::Type{VulkanInstanceRecord}, r, (blas_address, sbt), keep::UInt32) =
    VulkanInstanceRecord(r.transform, (r.id & 0x00ffffff) | ((r.mask & keep) << 24),
                         sbt & 0x00ffffff, blas_address)

"""
    writeinstances!(hwtlas, native, shading)

Write the queued transform updates into their batches' records, then every
batch's instances into `native` and `shading` ([`instances_kernel!`](@ref)),
in one submission on the TLAS's channel, which the build or refit after it is
ordered behind. One submission, not one per batch: a scene is one batch per plot.
"""
function writeinstances!(hwtlas::VulkanTLAS, native::LavaArray{VulkanInstanceRecord, 1},
                         shading::LavaArray{Raycore.InstanceRecord, 1})
    oneshot!(hwtlas.bq; tag = :instances) do e
        updated = false
        for (handle, transforms) in hwtlas.pending_updates
            batch = batchof(hwtlas.instances, handle)
            batch === nothing && continue   # deleted before this sync!
            emitkernel!(e, update_instance_records_kernel!, batch.records, transforms;
                        ndrange = batch.n)
            updated = true
        end
        # The updates write records the conversion below reads.
        updated && emitpreparebarrier!(e)
        offset = 0
        for b in hwtlas.instances
            b.n == 0 && continue
            emitkernel!(e, instances_kernel!, native, shading, b.records,
                        Int32(offset), Int32(b.n), (b.blas.address, b.sbt_offset),
                        b.hidden ? UInt32(0) : UInt32(0xff); ndrange = b.n)
            offset += b.n
        end
    end
    empty!(hwtlas.pending_updates)
    return nothing
end

# Drop BLASes no batch references (by identity) and return them. Built
# `LavaTLAS`es keep their own copy of the list until their uses retire.
function compact_blas_list!(t::VulkanTLAS)
    used = Base.IdSet{LavaBLAS}()
    for batch in t.instances
        push!(used, batch.blas)
    end
    dropped = LavaBLAS[]
    filter!(t.blas_list) do blas
        keep = blas in used
        keep || push!(dropped, blas)
        return keep
    end
    return dropped
end

function rebuild_hw_tlas_from_batch!(hwtlas::VulkanTLAS{Tri}) where {Tri}
    # Drop unused BLASes (deleted batches may have left some unreferenced).
    dropped_blases = compact_blas_list!(hwtlas)

    # No early return when empty: a zero-instance TLAS is valid and every ray
    # misses it. A ray query with no TLAS bound is undefined and loses the device.

    # Fresh instance buffers per rebuild: the old `LavaTLAS` frees its records
    # with itself, which would free a shared array under the live one. At least
    # one record: the build reads the address, and a trace binds the shading copy.
    total = ninstances(hwtlas.instances)
    native = LavaArray{VulkanInstanceRecord, 1}(undef, max(total, 1);
                                                bq = hwtlas.bq, extra_usage = AS_INPUT_USAGE)
    shading = LavaArray{Raycore.InstanceRecord, 1}(undef, max(total, 1); bq = hwtlas.bq)
    writeinstances!(hwtlas, native, shading)

    # After compaction `blas_list` is exactly what the instances reference; the
    # TLAS keeps it so a trace holds every BLAS.
    hw_tlas = build_accel!(hwtlas.bq) do ctx
        build_tlas(ctx, native, total; allow_update=true, blases=hwtlas.blas_list)
    end

    metadata = trianglemetadata(Tri, hwtlas.instances)
    all_tris, hit_offsets = metadata.triangles, metadata.hits

    tri_gpu = reuse_or_alloc(hwtlas.tri_gpu, all_tris, hwtlas.bq)
    off_gpu = reuse_or_alloc(hwtlas.off_gpu, hit_offsets, hwtlas.bq)

    return (hw_tlas, tri_gpu, off_gpu, native, shading, dropped_blases)
end

canrefit(t::VulkanTLAS) = t.hw_tlas !== nothing && t.hw_tlas.update_scratch !== nothing

# MODE_UPDATE_KHR refit from what the records hold: queued updates or a caller's
# kernel writes (`Raycore.refit!`). In place, so the adapted form a plan holds
# reads the same arrays.
function build!(t::VulkanTLAS, ::Refit)
    native = t.combined_instance_buf::LavaArray{VulkanInstanceRecord, 1}
    writeinstances!(t, native, t.combined_records::LavaArray{Raycore.InstanceRecord, 1})
    build_accel!(t.bq) do ctx
        refit_tlas!(ctx, t.hw_tlas, native, ninstances(t.instances))
    end
    return t
end

# A topology change or the first build. Queued transform updates are written as
# part of it (`writeinstances!`). The build waits for itself (`build_accel!`);
# replaced structures are freed by core's deferred retirement.
function build!(t::VulkanTLAS, ::Rebuild)
    hw_tlas, tri_gpu, off_gpu, native, shading, dropped_blases = rebuild_hw_tlas_from_batch!(t)
    old_hw_tlas = t.hw_tlas
    old_shading = t.combined_records
    # Release `tri_gpu`/`off_gpu` only if the rebuild replaced them.
    old_tri_gpu = t.tri_gpu === tri_gpu ? nothing : t.tri_gpu
    old_off_gpu = t.off_gpu === off_gpu ? nothing : t.off_gpu
    t.hw_tlas = hw_tlas
    t.tri_gpu = tri_gpu
    t.off_gpu = off_gpu
    t.combined_instance_buf = native
    t.combined_records = shading
    # The old driver records go with `old_hw_tlas`, whose preserves own them.
    old_hw_tlas === nothing || unsafe_free!(old_hw_tlas)
    old_shading === nothing || unsafe_free!(old_shading)
    old_tri_gpu === nothing || unsafe_free!(old_tri_gpu)
    old_off_gpu === nothing || unsafe_free!(old_off_gpu)
    foreach(unsafe_free!, dropped_blases)
    return t
end

# Inline ray-query `closest_hit` / `any_hit` on `AdaptedAccel`. They return
# Raycore's software tuple `(hit, primitive, t, bary, inst_custom_idx)`, so one
# kernel runs on either traversal. The kernel must be compiled with
# `enable_ray_query=true` (set when `lava_launch!` gets `tlas=`), which binds the
# TLAS descriptor at set 0, binding 0.

# SPIR-V ray-query intersection kinds. Named because `1` means AABB for a
# candidate and triangle for a committed hit.
const CANDIDATE_AABB     = UInt32(1)
const COMMITTED_TRIANGLE  = UInt32(1)
const COMMITTED_GENERATED = UInt32(2)

# The candidate verbs (`raytracing/accel.jl`) on the inline ray query. Overlays in
# Lava's method table: a Mac compiles this backend and Metal's, and plain methods
# would be defined twice.
@lava_device_override @inline candidate_primitive_index() = Int(lava_ray_query_get_primitive_index(false)) + 1
@lava_device_override @inline candidate_object_ray() = (
    Vec3f(lava_ray_query_get_object_ray_origin(false, 1),
          lava_ray_query_get_object_ray_origin(false, 2),
          lava_ray_query_get_object_ray_origin(false, 3)),
    Vec3f(lava_ray_query_get_object_ray_direction(false, 1),
          lava_ray_query_get_object_ray_direction(false, 2),
          lava_ray_query_get_object_ray_direction(false, 3)))
@lava_device_override @inline commit_intersection!(t) = lava_ray_query_generate_intersection(Float32(t))

@inline function rq_collect(accel::AdaptedAccel)
    # The running best procedural hit, kept by the shader; `nothing` for a
    # triangles-only scene, which folds the loop body away.
    best = procedural_miss(accel.procedural)
    while lava_ray_query_proceed()
        # Opaque triangles commit themselves; an AABB candidate is answered here.
        if lava_ray_query_get_type(false) == CANDIDATE_AABB
            best = procedural_candidate(accel.procedural, best)
        end
    end

    kind = lava_ray_query_get_type(true)  # committed
    if kind == COMMITTED_GENERATED
        t = lava_ray_query_get_t(true)
        prim_idx = lava_ray_query_get_primitive_index(true)
        inst_custom_idx = lava_ray_query_get_instance_custom_index(true)
        prim = procedural_commit(accel.procedural, best, prim_idx, t, accel.empty)
        # Same tuple shape and primitive type as a triangle hit (`procedural_commit`).
        return (true, prim, t, procedural_bary(accel.procedural, best), inst_custom_idx)
    end
    if kind != COMMITTED_TRIANGLE
        return (false, accel.empty, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    t = lava_ray_query_get_t(true)
    inst_id = lava_ray_query_get_instance_id(true)
    inst_custom_idx = lava_ray_query_get_instance_custom_index(true)
    prim_idx = lava_ray_query_get_primitive_index(true)
    bx, by = lava_ray_query_get_barycentrics(true)

    tri = hittriangle(accel, inst_id, prim_idx)

    bary = SVector{3,Float32}(1f0 - bx - by, bx, by)
    return (true, tri, t, bary, inst_custom_idx)
end

# `mask` is the ray's cull mask; the driver compares its low 8 bits with each
# instance's mask. `% UInt32` avoids an error branch in every shader.
@propagate_inbounds function Raycore.closest_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    lava_ray_query_init(UInt32(0), mask % UInt32,
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return rq_collect(accel)
end

@propagate_inbounds function Raycore.any_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    # RayFlagsTerminateOnFirstHitKHR (4): stop at the first commit.
    lava_ray_query_init(UInt32(4), mask % UInt32,
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return rq_collect(accel)
end

"""
    tlasset!(owner, ctx, pipeline, tlas) -> VkDescriptorSet

The descriptor set binding `tlas` for `pipeline`'s layout, owned by the owner of
the commands that bind it. A [`Recording`](@ref), whose commands bake the set,
keeps one per (layout, structure) pair for as long as it can be submitted; a
one-shot allocates one per dispatch and holds its pool. Not a keyed cache:
`Raycore.sync!` makes a new `LavaTLAS` per rebuild, so one would grow without
bound or evict pools still in flight. Only `release!` frees.
"""
function tlasset!(rec::Recording, ctx::LavaDevice, pipeline::LavaComputePipeline, tlas)
    layout = pipeline.descriptor_set_layout::VK.DescriptorSetLayout
    for (l, t, _, set) in rec.sets
        (l === layout && t === tlas) && return set
    end
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(ctx, pipeline, tlas)
    push!(rec.sets, (layout, tlas, desc_pool, desc_set))
    return desc_set
end

function tlasset!(o::OneShot, ctx::LavaDevice, pipeline::LavaComputePipeline, tlas)
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(ctx, pipeline, tlas)
    # The pool must outlive the submission that binds the set.
    hold!(o, desc_pool)
    return desc_set
end

function bindtlas!(e::Emitter, pipeline::LavaComputePipeline, tlas::VulkanTLAS)
    lava_tlas = tlas.hw_tlas::LavaTLAS
    set = tlasset!(e.owner, e.ctx, pipeline, lava_tlas)
    VK.cmd_bind_descriptor_sets(e.cmd, VK.PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout, UInt32(0), [set], UInt32[])
    # One hold covers both levels: the `LavaTLAS` references every `LavaBLAS`,
    # and `syncbuf!(::LavaTLAS)` records each level's storage.
    hold!(e, lava_tlas)
    return nothing
end

# The SBT path needs VK_KHR_ray_tracing_pipeline, which a ray-query device may
# lack (`supports_rt_pipeline` in `raytracing/api.jl`).
supports_rt_pipeline(b::LavaBackend) = vk_context(b).rt_pipeline_properties !== nothing
supports_hwtlas(b::LavaBackend) = vk_context(b).ray_query_available
supports_rt_pipeline(t::VulkanTLAS) = supports_rt_pipeline(t.backend)

# The inline ray query answers boxes through the candidate verbs above.
supports_procedural_traversal(b::LavaBackend) = vk_context(b).ray_query_available
supports_procedural_traversal(t::VulkanTLAS) = supports_procedural_traversal(t.backend)

# ============================================================================
# VulkanTLAS — concrete hardware-accelerated HWTLAS living in Lava
# ============================================================================
#
# Single GPU-resident-instances code path: every push! produces an
# `VulkanInstanceBatch{Tri}` that owns a `LavaArray{Raycore.InstanceRecord, 1}`, the
# portable records a kernel may write (`Raycore.instance_buffer`). Mutations
# (update_transform!/update_transforms!) write them GPU-side via compute kernels
# and flag `transforms_dirty`. `sync!` decides rebuild-vs-refit from the dirty
# flags and writes the `VulkanInstanceRecord`s the driver builds from.

import Raycore
import Adapt
using GeometryBasics
using StaticArrays: SVector, SMatrix
using Base: @propagate_inbounds
import KernelAbstractions
const KA = KernelAbstractions

const Mat4f = SMatrix{4, 4, Float32, 16}

# `unsafe_free!` here is GPUArrays' (imported in the extension), extended for
# the acceleration structures in `raytracing/acceleration.jl`; a backing that
# may be `nothing` is checked at the call, there is no method for `nothing`.

const VulkanInstanceBatch{Tri} =
    InstanceBatch{Tri,LavaBLAS,LavaArray{Raycore.InstanceRecord,1}}

# ============================================================================
# VulkanTLAS struct
# ============================================================================

"""
    VulkanTLAS{Tri} <: Raycore.AbstractAccel

Lava-native hardware-accelerated HWTLAS.  Concretely typed on `Tri` (the
per-primitive triangle type, typically `Raycore.Triangle{UInt32}`).

Build geometry with `push!(hwtlas, mesh, transform)`, then call
`Raycore.sync!(hwtlas)` to upload and build the Vulkan AS.  The adapted form
lives in `hwtlas.static_tlas` as a `AdaptedAccel{VulkanTLAS{Tri}}`.

# Mutation contract

`update_transform!` / `update_transforms!` write the batch's GPU-resident
`records` via a compute kernel and flag `transforms_dirty`; a caller's own
kernel may write them too (`Raycore.instance_buffer`), and `Raycore.refit!`
commits that. The next `sync!` decides between full rebuild (topology change,
`dirty=true`) and `MODE_UPDATE_KHR` refit (`transforms_dirty=true`).  No
CPU-side staging.

# Adapted-form invariant

`sync!(hwtlas)` is the sole owner of `hwtlas.static_tlas`.  It rebuilds as
efficiently as possible — in place via `resize!`/`copyto!` where the backing
buffer can be reused, reallocated only when a buffer grew — and stores the
result in `hwtlas.static_tlas`.  `sync!` MAY reassign `hwtlas.static_tlas`
when a buffer was reallocated.

Every consumer that hands the accel to a raytracing dispatch MUST go through
`hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` (which reads /
refreshes `hwtlas.static_tlas`) per dispatch.  Both are cheap; `sync!` did
the heavy lifting.  **NEVER cache the `AdaptedAccel` returned by `adapt`
across mutations** — consumers that cache silently see stale geometry.

# Non-blocking sync!

`sync!(hwtlas)` does NOT call `KA.synchronize(backend)`.  Old backings are
dropped via `unsafe_free!`, which RECORDS the destroy (`Mantle.retire!`) and
lets core run it once the submission that named the storage has passed. What
makes that correct on the BDA path is that a trace HOLDS every structure it
reads, so nothing in flight can be the last reference.

For a CPU-blocking drain use `Raycore.wait_for_gpu!(hwtlas)`, which calls
`flush!(hwtlas.bq)` (waits on the VulkanTLAS's own queue specifically, not
the backend-wide queue the Raycore default `wait_for_gpu!` uses).
"""
mutable struct VulkanTLAS{Tri} <: BatchedTLAS{Tri}
    backend::LavaBackend

    # submit channel for all AS builds and RT dispatches on this VulkanTLAS.
    bq::SubmitChannel{<:VulkanQueue}

    # Geometry (accumulated on push!)
    blas_list::Vector{LavaBLAS}

    # Instance batches -- N instances of one BLAS each, transforms in a GPU buffer.
    # Every push! produces one batch; per-mesh push! batches simply have n=1.
    #
    # The order, the handles and the reindex on delete are core's
    # (`raytracing/batches.jl`); native types parameterize the shared batch.
    # The list, the handle→index `Dict` and the counter are not per backend.
    instances::InstanceBatches{VulkanInstanceBatch{Tri}}

    # Bounding box (CPU-side, updated on push!)
    root_aabb::Raycore.Bounds3

    # Built on sync! — Lava-specific
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, HardwareAccel{Vector{Tri}}}
    tri_gpu::Union{Nothing, LavaArray{Tri, 1}}
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}
    # What the driver builds from: one `VulkanInstanceRecord` per instance, every
    # batch's in order, written from the batches' records at each build and
    # refit (`writeinstances!`). Fresh per rebuild, because the `LavaTLAS` built
    # from it owns it; written in place by a refit.
    combined_instance_buf::Union{Nothing, LavaArray{VulkanInstanceRecord, 1}}
    # The same instances as `Raycore.InstanceRecord`s, which is what a hit reads
    # its transform from (`AdaptedAccel.instances`). Written in place by a
    # refit, so the adapted form a plan holds sees the refit.
    combined_records::Union{Nothing, LavaArray{Raycore.InstanceRecord, 1}}

    # GPU-adapted form, owned by sync!.  Consumers read this via
    # `hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` per dispatch.
    # `nothing` until first sync!.
    static_tlas::Any   # Union{Nothing, AdaptedAccel{VulkanTLAS{Tri}}}

    # Topology dirty: a push!/delete! changed the batch list, full rebuild needed.
    dirty::Bool
    # Transforms dirty: update_transforms! queued at least one pending update;
    # refit (MODE_UPDATE_KHR) is enough once they are flushed in sync!.
    transforms_dirty::Bool

    # Queued transform updates: handle → arguments for the per-batch kernel.
    # Populated by update_transforms!; consumed (and cleared) by sync!.
    # Last write wins per handle (Dict key); values are Mat3x4f device arrays.
    pending_updates::Dict{Raycore.TLASHandle, Any}

    # The scene's procedural geometry, or `nothing`. Lives beside `blas_list`
    # because that is what it is — geometry — and it reaches the kernel through
    # `AdaptedAccel`, which is what the traversal loop reads. See
    # `procedural_candidate` in `raytracing/accel.jl`.
    procedural::Any
end

"""
    VulkanTLAS{Tri}(backend::LavaBackend; bq=backend.dispatch_bq) -> VulkanTLAS{Tri}

Construct an empty VulkanTLAS parametrised on triangle type `Tri`.
"""
# The portable spelling. `HWTLAS{Tri}(backend)` is how a caller that must not name
# a backend asks for one — Hikari's `default_accel` does exactly this — and each
# backend answers with its own concrete type.
HWTLAS{Tri}(backend::LavaBackend; kw...) where {Tri} = VulkanTLAS{Tri}(backend; kw...)

function VulkanTLAS{Tri}(backend::LavaBackend; bq::SubmitChannel{<:VulkanQueue}=backend.dispatch_bq) where {Tri}
    VulkanTLAS{Tri}(
        backend, bq,
        LavaBLAS[],
        InstanceBatches{VulkanInstanceBatch{Tri}}(),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing,
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

Default constructor — narrows to `Triangle{UInt32}`.
"""
VulkanTLAS(backend::LavaBackend; bq::SubmitChannel{<:VulkanQueue}=backend.dispatch_bq) =
    VulkanTLAS{Raycore.Triangle{UInt32}}(backend; bq)

# ============================================================================
# AdaptedAccel
# ============================================================================

"""Build the CPU-form AdaptedAccel from a synced VulkanTLAS."""
function AdaptedAccel(hwtlas::VulkanTLAS{Tri}) where Tri
    # `nothing` for `scene`: Vulkan binds the TLAS as a descriptor, so the kernel
    # does not carry a handle to it. See `AdaptedAccel` in `raytracing/accel.jl`.
    AdaptedAccel(hwtlas, hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri),
                 nothing, hwtlas.procedural, hwtlas.combined_records)
end

# pin_leaves! stops at VulkanTLAS — its LavaArray contents (`tri_gpu` / `off_gpu`)
# are already directly exposed as fields on AdaptedAccel, so the walker pins
# them via that path.  Without this stop, the @generated walker recurses into
# VulkanTLAS → submit channel → ctx → submit channel → … and blows the stack.
#
# `::Closed`, and not one concrete owner: a `Recording` owns pins as much as a
# batch does, and naming only the batch lets a recorded hardware-RT plan fall
# through to the generic walker and blow the stack. The stack was 53 320 frames
# of
# `holdleaves!(::Recording, ::VK.Instance)`, whose `destructor` field is a
# closure over the instance itself. Every owner has to stop here, which is what
# the abstract type is for.
@inline holdleaves!(::Closed, ::VulkanTLAS) = nothing

# ============================================================================
# Adapt.adapt_structure
# ============================================================================

"""
    Adapt.adapt_structure(to, hwtlas::VulkanTLAS) -> AdaptedAccel

Ensure `sync!` has run, then return the cached `hwtlas.static_tlas` — the
CPU-form AdaptedAccel that still holds the live `VulkanTLAS` reference plus
the GPU triangle/offset arrays.  Hikari's CPU dispatch code reaches through
`accel.hwtlas` for descriptor binding / sync helpers; that path must keep
working after `Adapt.adapt(::LavaBackend, hwtlas)`.

Kernel-form adaptation (drop `hwtlas`, strip arrays to LavaDeviceArray) only
fires when the adaptor is a `LavaAdaptor` — see the specialised method below.
"""
function Adapt.adapt_structure(to, hwtlas::VulkanTLAS{Tri}) where Tri
    Raycore.sync!(hwtlas)
    return hwtlas.static_tlas::AdaptedAccel
end

"""Drop the host owner and adapt traversal arrays; keep the out-of-band scene value.
Descriptor discovery uses the original unadapted hwtlas reference."""
Adapt.adapt_structure(to::LavaAdaptor, accel::AdaptedAccel) =
    adaptaccel(to, accel, identity)

# ============================================================================
# Accessors
# ============================================================================

# ============================================================================
# wait_for_gpu!
# ============================================================================

"""
    Raycore.wait_for_gpu!(hwtlas::VulkanTLAS) -> hwtlas

Block until all pending GPU work on `hwtlas.bq` has completed.
Uses `flush!` rather than `KA.synchronize(backend)`.
"""
function Raycore.wait_for_gpu!(hwtlas::VulkanTLAS)
    flush!(hwtlas.bq)
    return hwtlas
end

# ============================================================================
# Mutation API
# ============================================================================

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

# ============================================================================
# Rebuild / refit helpers + sync!
# ============================================================================

# Try to reuse `prev` as the GPU sink for `data` via capacity-aware resize+copyto.
# `bq` is the device the fresh allocation lands on when `prev` cannot be reused —
# the TLAS's own queue, never the process default.
#
# Never shorter than one element: a trace BINDS these, and a zero-length buffer
# has no device address to bind. An empty scene is a legitimate scene — every
# ray misses it — so the element exists to be addressable and is never read.
function _reuse_or_alloc(prev, data::AbstractArray{T}, bq) where T
    n = max(length(data), 1)
    if prev isa LavaArray{T}
        resize!(prev, n)
        isempty(data) || copyto!(prev, 1, data, 1, length(data))
        return prev
    end
    isempty(data) && return LavaArray{T, 1}(undef, n; bq)
    return LavaArray(data; bq)
end

"""
    vkinstances_kernel!(native, shading, records, offset, n, blas_address, sbt, keep)

One batch's instances, from the portable records a caller writes into the two
arrays a build reads: `shading[offset + i]` is `records[i]` as it is (what a hit
reads its transform from), and `native[offset + i]` the `VulkanInstanceRecord`
the driver builds from: the record's transform, its id in the low 24 bits and
its mask ANDed with `keep` (0 for a hidden batch, else `0xff`) in the high 8, the
batch's SBT offset, and the address of the BLAS the batch instances.
"""
function vkinstances_kernel!(native, shading, records, offset::Int32, n::Int32,
                             blas_address::UInt64, sbt::UInt32, keep::UInt32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing   # the launch is whole workgroups
    @inbounds r = records[i]
    j = offset + i
    @inbounds shading[j] = r
    @inbounds native[j] = VulkanInstanceRecord(r.transform,
                                               (r.id & 0x00ffffff) | ((r.mask & keep) << 24),
                                               sbt & 0x00ffffff, blas_address)
    return nothing
end

"""
    writeinstances!(hwtlas, native, shading)

Write the queued transform updates into their batches' records, then every
batch's instances into `native` and `shading` ([`vkinstances_kernel!`](@ref)),
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
            emitkernel!(e, vkinstances_kernel!, native, shading, b.records,
                        Int32(offset), Int32(b.n), b.blas.address, b.sbt_offset,
                        b.hidden ? UInt32(0) : UInt32(0xff); ndrange = b.n)
            offset += b.n
        end
    end
    empty!(hwtlas.pending_updates)
    return nothing
end

# Native TLAS objects copy the BLAS keepalive list at build time. Compact the
# current scene's list in place; old structures keep their own native owners
# until their queued uses retire. Each batch names its BLAS by object identity.
function _compact_blas_list!(t::VulkanTLAS)
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
    dropped_blases = _compact_blas_list!(hwtlas)

    # No early return for an empty instance list. A TLAS over ZERO instances is
    # a valid acceleration structure that every ray misses, and building it is
    # what lets a scene whose geometry is being drawn by the other renderer keep
    # its sky: the trace still runs, every ray escapes, and the environment
    # light paints them. Leaving `hw_tlas = nothing` instead left the ray query
    # with no acceleration structure bound, which is undefined behaviour and
    # cost the device.

    # Fresh instance buffers for every rebuild: the previous driver records are
    # solely owned by the soon-to-be-freed `hw_tlas.preserves`. Reusing them
    # across builds shares a single LavaArray identity between multiple HWTLAS
    # preserves lists, and the older HWTLAS's `unsafe_free!` would then free the
    # buffer out from under the live HWTLAS. One record minimum: the build reads
    # the buffer's ADDRESS before it reads `total` of them, and a trace binds the
    # shading copy.
    total = ninstances(hwtlas.instances)
    native = LavaArray{VulkanInstanceRecord, 1}(undef, max(total, 1);
                                                bq = hwtlas.bq, extra_usage = AS_INPUT_USAGE)
    shading = LavaArray{Raycore.InstanceRecord, 1}(undef, max(total, 1); bq = hwtlas.bq)
    writeinstances!(hwtlas, native, shading)

    # After the compaction above, `blas_list` is exactly what the instances
    # reference; the TLAS carries the list so a trace can pin every one.
    hw_tlas = build_accel!(hwtlas.bq) do ctx
        build_tlas(ctx, native, total; allow_update=true, blases=hwtlas.blas_list)
    end

    metadata = trianglemetadata(Tri, hwtlas.instances, Val(true))
    all_tris, hit_offsets = metadata.triangles, metadata.hits
    blas_offsets, per_inst_offsets = metadata.starts, metadata.raw

    hw_accel = if hwtlas.hw_accel isa HardwareAccel
        accel_prev = hwtlas.hw_accel
        accel_prev.tlas = hw_tlas
        accel_prev.triangle_data = all_tris
        accel_prev.blas_offsets = blas_offsets
        accel_prev.per_instance_tri_offsets = per_inst_offsets
        accel_prev
    else
        HardwareAccel(hw_tlas, all_tris, blas_offsets, per_inst_offsets; bq=hwtlas.bq)
    end

    tri_gpu = _reuse_or_alloc(hwtlas.tri_gpu, all_tris, hwtlas.bq)
    off_gpu = _reuse_or_alloc(hwtlas.off_gpu, hit_offsets, hwtlas.bq)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu, native, shading, dropped_blases)
end

"""
    Raycore.sync!(hwtlas::VulkanTLAS)

Single commit boundary. Decides between full rebuild (topology change) and
in-place refit (transforms-only update) from the dirty flags. No-op when
nothing changed.

Does NOT call KA.synchronize — uses Lava's timeline-based deferred free path.
"""
function Raycore.sync!(hwtlas::VulkanTLAS)
    # True no-op when nothing changed and static_tlas is already built.
    if !hwtlas.dirty && !hwtlas.transforms_dirty && hwtlas.static_tlas !== nothing
        return hwtlas
    end

    # An empty instance list takes the REBUILD path like any other topology
    # change — it builds an acceleration structure over zero instances, which
    # every ray misses. Dropping the structure instead (the previous behaviour)
    # left nothing for a ray query to be initialised with, and the device was
    # lost the first time a scene with an environment light had its last
    # traced plot handed to the rasteriser.

    if hwtlas.dirty
        # Topology change: full rebuild (also handles the "first sync after
        # construct" case since the constructor leaves dirty=true). Queued
        # transform updates are written as part of it (`writeinstances!`).
        hw_tlas, hw_accel, tri_gpu, off_gpu, native, shading, dropped_blases =
            rebuild_hw_tlas_from_batch!(hwtlas)

        old_hw_tlas = hwtlas.hw_tlas
        old_shading = hwtlas.combined_records
        # tri_gpu / off_gpu get reused-in-place when sizes permit -- do NOT
        # release unless the rebuild returned a different object.
        old_tri_gpu = hwtlas.tri_gpu === tri_gpu ? nothing : hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu === off_gpu ? nothing : hwtlas.off_gpu

        hwtlas.hw_tlas   = hw_tlas
        hwtlas.hw_accel  = hw_accel
        hwtlas.tri_gpu   = tri_gpu
        hwtlas.off_gpu   = off_gpu
        hwtlas.combined_instance_buf = native
        hwtlas.combined_records = shading
        hwtlas.dirty            = false
        hwtlas.transforms_dirty = false
        hwtlas.static_tlas = AdaptedAccel(hwtlas)
        # The old driver records go with `old_hw_tlas`, whose preserves own them.
        old_hw_tlas === nothing || unsafe_free!(old_hw_tlas)
        old_shading === nothing || unsafe_free!(old_shading)
        old_tri_gpu === nothing || unsafe_free!(old_tri_gpu)
        old_off_gpu === nothing || unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    # Transforms-only path: MODE_UPDATE_KHR refit, from whatever the records now
    # hold — queued updates, or what a caller's kernel wrote (`Raycore.refit!`).
    if hwtlas.transforms_dirty
        if hwtlas.hw_tlas === nothing || !hwtlas.hw_tlas.allow_update
            # Defensive: if the prior build wasn't refit-capable, fall back to
            # full rebuild.  Should not happen on normal use since
            # rebuild_hw_tlas_from_batch! always sets allow_update=true.
            hwtlas.dirty = true
            return Raycore.sync!(hwtlas)
        end
        native = hwtlas.combined_instance_buf::LavaArray{VulkanInstanceRecord, 1}
        # In place: the adapted form a plan holds reads this same array.
        writeinstances!(hwtlas, native, hwtlas.combined_records::LavaArray{Raycore.InstanceRecord, 1})
        build_accel!(hwtlas.bq) do ctx
            refit_tlas!(ctx, hwtlas.hw_tlas, native, ninstances(hwtlas.instances))
        end
        hwtlas.transforms_dirty = false
        # static_tlas wraps the same hw_tlas — reuse, just make sure it
        # exists (first-sync edge case).
        hwtlas.static_tlas === nothing && (hwtlas.static_tlas = AdaptedAccel(hwtlas))
        return hwtlas
    end

    # Both flags false but static_tlas was nothing — first sync on a fresh
    # VulkanTLAS with no batches yet.
    hwtlas.static_tlas = AdaptedAccel(hwtlas)
    return hwtlas
end

# ============================================================================
# Inline ray query closest_hit / any_hit on AdaptedAccel
# ============================================================================
#
# Polymorphic with the SW path's `Raycore.closest_hit(::StaticTLAS, ray, mask)`:
# same return tuple shape `(hit, primitive, t, bary, inst_custom_idx)`.  A
# kernel written against `Raycore.closest_hit(accel, ray[, mask])` runs unchanged
# on either backend; multiple dispatch picks the right traversal, and Raycore's
# two-argument form passes mask `0xff`.
#
# Lowers to OpRayQueryInitializeKHR/Proceed/Get*KHR via the lava_ray_query_*
# intrinsics.  The kernel must be compiled with `enable_ray_query=true`
# (auto-set when a `tlas=` kwarg is passed to `lava_launch!`) so the SPIR-V
# emitter binds the HWTLAS descriptor at set 0 binding 0.

# Candidate/committed intersection kinds, from the SPIR-V ray query spec. Named
# because `== UInt32(1)` means AABB in one of these enumerations and triangle in
# the other, and the two appear four lines apart.
const CANDIDATE_TRIANGLE = UInt32(0)
const CANDIDATE_AABB     = UInt32(1)
const COMMITTED_NONE      = UInt32(0)
const COMMITTED_TRIANGLE  = UInt32(1)
const COMMITTED_GENERATED = UInt32(2)

# The portable candidate accessors, on the inline ray query. See
# `raytracing/accel.jl` for what they are for. Overlays, in Lava's method table:
# a Mac compiles this tree and Metal's, and as plain methods the two backends'
# answers would be one method defined twice.
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
    # The running best procedural hit, kept by the SHADER. See
    # `procedural_candidate` for why the query cannot keep it for us.
    # `nothing` when the scene holds only triangles, and then this whole loop
    # body folds away to the empty one it used to be.
    best = procedural_miss(accel.procedural)
    while lava_ray_query_proceed()
        # Opaque triangles auto-commit; no any-hit decision for those. A box is
        # the traversal ASKING, and it gets answered here.
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
        # Same tuple shape, same primitive TYPE, as a triangle hit — see
        # `procedural_commit`. Nothing downstream needs to know this ray met a
        # box rather than a triangle.
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

# `mask` is the ray's cull mask, the ray query's `Cull Mask` operand: the driver
# compares its low 8 bits with each instance's mask. `% UInt32`, not a checked
# conversion, which would leave an error branch in every shader.
@propagate_inbounds function Raycore.closest_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    lava_ray_query_init(UInt32(0), mask % UInt32,
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return rq_collect(accel)
end

@propagate_inbounds function Raycore.any_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    # SPIR-V RayFlagsTerminateOnFirstHitKHR = 4 — exit traversal at first commit.
    lava_ray_query_init(UInt32(4), mask % UInt32,
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return rq_collect(accel)
end

"""
    tlasset!(owner, dev, pipeline, tlas) -> VkDescriptorSet

The descriptor set binding `tlas` for `pipeline`'s layout, owned by whoever owns
the commands that bind it.

A [`Recording`](@ref) keeps ONE per (layout, acceleration structure) pair and
holds its pool for as long as it can be submitted. A one-shot allocates a fresh
one per dispatch and pins it.

Owned, and not cached under a key: a cache keyed by
`(layout, objectid(LavaTLAS))` grows without bound, because `Raycore.sync!`
makes a new `LavaTLAS` per rebuild, and `WeakRef` eviction lags Julia's GC, so
pools are destroyed while their sets are still in flight. Here nothing evicts
and `release!` is the only thing that frees.

Allocating per dispatch is not the alternative for a recording: the set is baked
into the command buffer, so a fresh set per dispatch means N pools kept alive
for one plan, all naming the same acceleration structure.
"""
function tlasset!(rec::Recording, dev::VK.Device, pipeline::LavaComputePipeline, tlas)
    layout = pipeline.descriptor_set_layout::VK.DescriptorSetLayout
    for (l, t, _, set) in rec.sets
        (l === layout && t === tlas) && return set
    end
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(dev, pipeline, tlas)
    push!(rec.sets, (layout, tlas, desc_pool, desc_set))
    return desc_set
end

function tlasset!(o::OneShot, dev::VK.Device, pipeline::LavaComputePipeline, tlas)
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(dev, pipeline, tlas)
    # A one-shot runs once, so its set is not cached the way a recording's is —
    # but the pool that owns it must outlive the submission that binds it.
    hold!(o, desc_pool)
    return desc_set
end

function bindtlas!(e::Emitter, pipeline::LavaComputePipeline, tlas::VulkanTLAS)
    lava_tlas = tlas.hw_tlas::LavaTLAS
    set = tlasset!(e.owner, e.ctx.device, pipeline, lava_tlas)
    VK.cmd_bind_descriptor_sets(e.cmd, VK.PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout, UInt32(0), [set], UInt32[])
    # A ray query walks the top level into its bottom levels and reads their
    # storage, and one hold covers all of it: the `LavaTLAS` references every
    # `LavaBLAS` it instances, and `syncbuf!(::LavaTLAS)` records each level's
    # storage for ordering.
    hold!(e, lava_tlas)
    return nothing
end

# Vulkan drives an SBT through `vkCmdTraceRaysIndirect`, where the device has the
# pipeline extension. See `supports_rt_pipeline` in `raytracing/api.jl`. Asked of the
# device: `true` unconditionally sent a device with ray query and no pipeline
# extension into a pipeline it cannot create.
supports_rt_pipeline(b::LavaBackend) = vk_context(b).rt_pipeline_properties !== nothing
supports_hwtlas(b::LavaBackend) = vk_context(b).ray_query_available
supports_rt_pipeline(t::VulkanTLAS) = supports_rt_pipeline(t.backend)

# …and the inline ray query answers a BOX as well as a triangle: the three
# candidate verbs just above are what `procedural_candidate` is written against.
# Without ray query there is no traversal to answer from.
supports_procedural_traversal(b::LavaBackend) = vk_context(b).ray_query_available
supports_procedural_traversal(t::VulkanTLAS) = supports_procedural_traversal(t.backend)

# ============================================================================
# VulkanTLAS — concrete hardware-accelerated HWTLAS living in Lava
# ============================================================================
#
# Single GPU-resident-instances code path: every push! produces an
# `InstanceBatch{Tri}` that owns a `LavaArray{VulkanInstanceRecord, 1}`.
# Mutations (update_transform!/update_transforms!) write GPU-side via
# compute kernels and flag `transforms_dirty`.  `sync!` decides
# rebuild-vs-refit from the dirty flags.

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

# ============================================================================
# InstanceBatch struct
# ============================================================================

"""
    InstanceBatch{Tri}

A batch of N HWTLAS instances all referencing the same BLAS, with instance
records read from a GPU-resident `LavaArray{VulkanInstanceRecord, 1}` at
sync! / refit time. Returned `handle` lets callers track the batch in
`VulkanTLAS.handle_to_batch_idx` for delete!/refit.

`triangles` holds the per-triangle metadata for the BLAS (typically
`Vector{Triangle{UInt32}}`). Pass an empty vector for rayQuery-only
callers that don't need Hikari's per-triangle TriangleMeta lookup.
"""
struct InstanceBatch{Tri}
    blas::LavaBLAS
    instance_buf::LavaArray{VulkanInstanceRecord, 1}
    n::Int
    instance_mask::UInt8
    custom_index::UInt32      # low 24 bits of gl_InstanceCustomIndexEXT (mi_idx / instance_id)
    handle::Raycore.TLASHandle
    triangles::Vector{Tri}
    # SBT hit-group offset every instance in the batch shares.  0 maps to the
    # first chit slot (legacy single-chit pipelines); per-material pipelines
    # set this per push! to the material's slot index in
    # `RayTracingPipeline.closesthit_funcs`.
    sbt_offset::UInt32
end

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

`update_transform!` / `update_transforms!` write directly to the batch's
GPU-resident `instance_buf` via a compute kernel and flag
`transforms_dirty`.  The next `sync!` decides between full rebuild
(topology change, `dirty=true`) and `MODE_UPDATE_KHR` refit
(`transforms_dirty=true`).  No CPU-side staging.

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
dropped via `unsafe_free!`, which defers destruction through
`hwtlas.bq`'s timeline (`bq.deferred_as_frees` / `bq.deferred_frees`) when
prior dispatches are still in flight.  Phase-B pinning of RT closure leaves
(tri_gpu, off_gpu, hw_tlas, hw_accel) is what makes the timeline tracking
correct on the BDA path.

For a CPU-blocking drain use `Raycore.wait_for_gpu!(hwtlas)`, which calls
`vk_flush!(hwtlas.bq)` (waits on the VulkanTLAS's own queue specifically, not
the backend-wide queue the Raycore default `wait_for_gpu!` uses).
"""
mutable struct VulkanTLAS{Tri} <: HWTLAS{Tri}
    backend::LavaBackend

    # VulkanBatchQueue for all AS builds and RT dispatches on this VulkanTLAS.
    bq::VulkanBatchQueue

    # Geometry (accumulated on push!)
    blas_list::Vector{LavaBLAS}
    blas_triangles::Vector{Vector{Tri}}
    blas_offsets::Vector{UInt32}

    # Instance batches -- N instances of one BLAS each, transforms in a GPU buffer.
    # Every push! produces one batch; per-mesh push! batches simply have n=1.
    instance_batches::Vector{InstanceBatch{Tri}}

    # Handle management: handle -> index into instance_batches.
    handle_to_batch_idx::Dict{Raycore.TLASHandle, Int}
    next_handle_id::UInt32

    # Bounding box (CPU-side, updated on push!)
    root_aabb::Raycore.Bounds3

    # Built on sync! — Lava-specific
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, HardwareAccel{Vector{Tri}}}
    tri_gpu::Union{Nothing, LavaArray{Tri, 1}}
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}
    # Combined instance buffer: concatenation of every batch's instance_buf.
    # Allocated/grown in rebuild_hw_tlas_from_batch! and reused across syncs +
    # refits.
    combined_instance_buf::Union{Nothing, LavaArray{VulkanInstanceRecord, 1}}

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
    # Last write wins per handle (Dict key). Value is one of:
    #   AbstractVector{<:AbstractMatrix}                    — CPU matrix path
    #   (LavaArray{Point3f,1}, LavaArray{Vec4f,1}, Float32, UInt32)   — GPU uniform scale
    #   (LavaArray{Point3f,1}, LavaArray{Vec4f,1}, LavaArray{Vec3f,1}, UInt32) — GPU per-vec scale
    pending_updates::Dict{Raycore.TLASHandle, Any}
end

"""
    VulkanTLAS{Tri}(backend::LavaBackend; bq=backend.dispatch_bq) -> VulkanTLAS{Tri}

Construct an empty VulkanTLAS parametrised on triangle type `Tri`.
"""
# The portable spelling. `HWTLAS{Tri}(backend)` is how a caller that must not name
# a backend asks for one — Hikari's `default_accel` does exactly this — and each
# backend answers with its own concrete type.
HWTLAS{Tri}(backend::LavaBackend; kw...) where {Tri} = VulkanTLAS{Tri}(backend; kw...)

function VulkanTLAS{Tri}(backend::LavaBackend; bq::VulkanBatchQueue=backend.dispatch_bq) where {Tri}
    VulkanTLAS{Tri}(
        backend, bq,
        LavaBLAS[], Vector{Tri}[], UInt32[],
        InstanceBatch{Tri}[],
        Dict{Raycore.TLASHandle, Int}(),
        UInt32(1),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing,
        nothing,                  # combined_instance_buf
        nothing,
        true,                     # dirty
        false,                    # transforms_dirty
        Dict{Raycore.TLASHandle, Any}(),  # pending_updates
    )
end

"""
    VulkanTLAS(backend::LavaBackend; bq=backend.dispatch_bq) -> VulkanTLAS{Triangle{UInt32}}

Default constructor — narrows to `Triangle{UInt32}`.
"""
VulkanTLAS(backend::LavaBackend; bq::VulkanBatchQueue=backend.dispatch_bq) =
    VulkanTLAS{Raycore.Triangle{UInt32}}(backend; bq)

# ============================================================================
# AdaptedAccel
# ============================================================================

"""Build the CPU-form AdaptedAccel from a synced VulkanTLAS."""
function AdaptedAccel(hwtlas::VulkanTLAS{Tri}) where Tri
    # `nothing` for `scene`: Vulkan binds the TLAS as a descriptor, so the kernel
    # does not carry a handle to it. See `AdaptedAccel` in `raytracing/accel.jl`.
    AdaptedAccel{VulkanTLAS{Tri}, typeof(hwtlas.tri_gpu), typeof(hwtlas.off_gpu), Tri, Nothing}(
        hwtlas, hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri), nothing)
end

# pin_leaves! stops at VulkanTLAS — its LavaArray contents (`tri_gpu` / `off_gpu`)
# are already directly exposed as fields on AdaptedAccel, so the walker pins
# them via that path.  Without this stop, the @generated walker recurses into
# VulkanTLAS → VulkanBatchQueue → ctx → VulkanBatchQueue → … and blows the stack.
#
# `::Closed`, not one concrete owner. It was the batch alone, from when a batch
# was the only thing that could own a pin — so a `Recording` fell through to
# the generic walker and a hardware-RT plan blew the stack the first time it
# was RECORDED rather than launched. The stack was 53 320 frames of
# `pin_leaves!(::Recording, ::VK.Instance)`, whose `destructor` field is a
# closure over the instance itself. Every owner has to stop here, which is what
# the abstract type is for.
@inline pin_leaves!(::Closed, ::VulkanTLAS) = nothing

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

"""
Kernel-form adaptation: drops `hwtlas` (so the kernel sees a Nothing-typed
field, which is bitstype) and walks the array fields through the adaptor so
`LavaArray` becomes `LavaDeviceArray` for the BDA arg buffer.  The auto-
discovered HWTLAS binding is wired by `lava_launch!`/`ka_launch!` from the
pre-adapt `AdaptedAccel.hwtlas` field.
"""
function Adapt.adapt_structure(to::LavaAdaptor, accel::AdaptedAccel)
    AdaptedAccel(
        nothing,
        Adapt.adapt(to, accel.triangles),
        Adapt.adapt(to, accel.offsets),
        accel.empty,
        accel.scene,
    )
end

# ============================================================================
# Accessors
# ============================================================================

Raycore.world_bound(hwtlas::VulkanTLAS)    = hwtlas.root_aabb
Raycore.n_geometries(hwtlas::VulkanTLAS)   = length(hwtlas.blas_list)
Raycore.n_instances(hwtlas::VulkanTLAS)    = sum(b.n for b in hwtlas.instance_batches; init=0)

"""
    Raycore.instance_buffer(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle) -> LavaArray{VulkanInstanceRecord, 1}

Return the GPU instance buffer for the batch registered under `handle`. The
caller can write new instance records into the returned LavaArray (typically
via a compute kernel) and then call `Raycore.sync!(hwtlas)` to commit
the change to the underlying LavaTLAS via MODE_UPDATE_KHR.

Errors if the handle is not registered.
"""
function Raycore.instance_buffer(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle)
    idx = get(hwtlas.handle_to_batch_idx, handle, nothing)
    idx === nothing && error("instance_buffer: invalid or deleted handle.")
    return hwtlas.instance_batches[idx].instance_buf
end

# RayMakie compat: hwtlas.instances -> lightweight length-only view
struct HWTLASInstances
    n::Int
end
Base.isempty(x::HWTLASInstances) = x.n == 0
Base.length(x::HWTLASInstances) = x.n

function Base.getproperty(hwtlas::VulkanTLAS, s::Symbol)
    if s === :instances
        n = sum(b.n for b in getfield(hwtlas, :instance_batches); init=0)
        return HWTLASInstances(n)
    end
    return getfield(hwtlas, s)
end

# ============================================================================
# wait_for_gpu!
# ============================================================================

"""
    Raycore.wait_for_gpu!(hwtlas::VulkanTLAS) -> hwtlas

Block until all pending GPU work on `hwtlas.bq` has completed.
Uses Lava's `vk_flush!` rather than `KA.synchronize(backend)`.
"""
function Raycore.wait_for_gpu!(hwtlas::VulkanTLAS)
    vk_flush!(hwtlas.bq)
    return hwtlas
end

# ============================================================================
# Mutation API
# ============================================================================

function hwtlas_add_geometry!(hwtlas::VulkanTLAS{Tri}, mesh::GeometryBasics.Mesh) where {Tri}
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    verts = decompose(Point3f, nmesh)
    norms = Raycore.Normal3f.(Raycore.decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? Point2f[] : Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))

    has_meta = hasproperty(nmesh, :face_meta)
    n_faces = length(fs)

    cpu_triangles = let tris = Tri[]
        for i in 1:n_faces
            Raycore.is_degenerate_face(verts, indices, i) && continue
            meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
            push!(tris, Raycore.build_triangle(verts, norms, uvs, indices, i, meta))
        end
        tris
    end
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertex positions for BLAS build
    n_tris = length(cpu_triangles)
    blas_vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        vs = cpu_triangles[i].vertices
        for j in 1:3
            v = vs[j]
            blas_vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    blas_indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        blas_indices[i+1] = UInt32(i)
    end

    hw_blas = build_accel!(hwtlas.bq) do ctx
        build_blas(ctx, blas_vertices, blas_indices)
    end

    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = Raycore.union(hwtlas.root_aabb, Raycore.Bounds3(Point3f(v)))
        end
    end

    hwtlas.dirty = true
    return blas_idx
end

# Internal: register an InstanceBatch for `blas` with `n` instances initially
# populated from the CPU-side `records` Vector.  Returns the new handle.
function _register_batch!(hwtlas::VulkanTLAS{Tri}, blas::LavaBLAS,
                          records::Vector{VulkanInstanceRecord},
                          triangles::Vector{Tri},
                          instance_mask::UInt8,
                          sbt_offset::UInt32) where {Tri}
    n = length(records)
    instance_buf = LavaArray{VulkanInstanceRecord, 1}(undef, n; bq=hwtlas.bq, extra_usage=AS_INPUT_USAGE)
    Base.copyto!(instance_buf, records)
    handle = Raycore.TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    push!(hwtlas.instance_batches,
          InstanceBatch{Tri}(blas, instance_buf, n, instance_mask, UInt32(0), handle, triangles, sbt_offset))
    hwtlas.handle_to_batch_idx[handle] = lastindex(hwtlas.instance_batches)
    hwtlas.dirty = true
    return handle
end

function Base.push!(hwtlas::VulkanTLAS{Tri}, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff),
                    sbt_offset::UInt32=UInt32(0)) where {Tri}
    blas_idx  = hwtlas_add_geometry!(hwtlas, mesh)
    blas      = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    record    = VulkanInstanceRecord(mat4_to_vk_transform(transform), blas.address;
                                   custom_index=instance_id, mask=instance_mask,
                                   sbt_offset=sbt_offset)
    return _register_batch!(hwtlas, blas, [record], triangles, instance_mask, sbt_offset)
end

function Base.push!(hwtlas::VulkanTLAS{Tri}, mesh::GeometryBasics.Mesh, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                    instance_mask::UInt8=UInt8(0xff),
                    sbt_offset::UInt32=UInt32(0)) where {Tri}
    if instance_ids !== nothing && length(instance_ids) != length(transforms)
        throw(ArgumentError("instance_ids length $(length(instance_ids)) != transforms length $(length(transforms))"))
    end
    blas_idx  = hwtlas_add_geometry!(hwtlas, mesh)
    blas      = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    addr      = blas.address
    records = Vector{VulkanInstanceRecord}(undef, length(transforms))
    @inbounds for i in eachindex(transforms)
        iid = instance_ids === nothing ? UInt32(0) : UInt32(instance_ids[i])
        records[i] = VulkanInstanceRecord(mat4_to_vk_transform(transforms[i]), addr;
                                        custom_index=iid, mask=instance_mask,
                                        sbt_offset=sbt_offset)
    end
    return _register_batch!(hwtlas, blas, records, triangles, instance_mask, sbt_offset)
end

"""
    push!(hwtlas::VulkanTLAS, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
          instance_id::UInt32=UInt32(0)) -> TLASHandle

Register a pre-built `LavaBLAS` (e.g. from `build_blas_aabb`) as a new
geometry + instance.  No triangle data is associated; accordingly the
`hw_accel` / ray-tracing pipeline path is not usable on this VulkanTLAS after
this call.  Use the compute-rayQuery path (`lava_launch!` with `tlas=hwtlas`)
instead.
"""
function Base.push!(hwtlas::VulkanTLAS{Tri}, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff),
                    sbt_offset::UInt32=UInt32(0)) where {Tri}
    # Register the pre-built BLAS — no triangles.
    push!(hwtlas.blas_list, blas)
    push!(hwtlas.blas_triangles, Tri[])
    blas_idx = length(hwtlas.blas_list)
    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    record = VulkanInstanceRecord(mat4_to_vk_transform(transform), blas.address;
                                custom_index=instance_id, mask=instance_mask,
                                sbt_offset=sbt_offset)
    return _register_batch!(hwtlas, blas, [record], Tri[], instance_mask, sbt_offset)
end

"""
    push!(tlas::VulkanTLAS{Tri}, blas::LavaBLAS,
          instance_buf::LavaArray{VulkanInstanceRecord, 1};
          n::Integer, instance_mask::UInt8,
          triangles::Vector{Tri}) -> Raycore.TLASHandle

Register an N-instance batch in the HWTLAS. All N instances reference the
same `blas`; per-instance transforms / custom_indices live in `instance_buf`
and are written by a GPU compute kernel (see `write_grain_instances_kernel`).

`n` defaults to `length(instance_buf)`. Pass a smaller value when the buffer
is pre-allocated larger than the current live instance count.

`triangles` supplies the BLAS's per-triangle metadata for Hikari's path tracer
(`tri_gpu` / `off_gpu` lookup). Pass empty (the default) for rayQuery-only
callers -- those don't need per-triangle metadata.

Returns one `TLASHandle` for the whole batch. Subsequent `sync!` builds
the underlying `LavaTLAS` with `allow_update=true` so per-frame refits work.
"""
function Base.push!(tlas::VulkanTLAS{Tri}, blas::LavaBLAS,
                    instance_buf::LavaArray{VulkanInstanceRecord, 1};
                    n::Integer = length(instance_buf),
                    instance_mask::UInt8 = UInt8(0xff),
                    custom_index::UInt32 = UInt32(0),
                    triangles::Vector{Tri} = Tri[],
                    sbt_offset::UInt32 = UInt32(0)) where {Tri}
    n_int = Int(n)
    n_int <= length(instance_buf) || error(
        "push!: n=$n_int exceeds instance_buf length $(length(instance_buf))")
    handle = Raycore.TLASHandle(tlas.next_handle_id)
    tlas.next_handle_id += UInt32(1)
    push!(tlas.instance_batches, InstanceBatch{Tri}(blas, instance_buf, n_int, instance_mask, custom_index, handle, triangles, sbt_offset))
    tlas.handle_to_batch_idx[handle] = lastindex(tlas.instance_batches)
    tlas.dirty = true
    return handle
end

"""
    push!(hwtlas::VulkanTLAS{Tri}, mesh::GeometryBasics.Mesh,
          instance_buf::LavaArray{VulkanInstanceRecord, 1};
          n::Integer, instance_mask::UInt8) -> Raycore.TLASHandle

Build a BLAS from `mesh` (or reuse a cached one) and register an N-instance
batch backed by the GPU-resident `instance_buf`. The per-instance transforms
and custom_indices are read from `instance_buf` at `sync!` time -- typically
written by a compute kernel such as `write_meshscatter_instances_kernel`.

`n` defaults to `length(instance_buf)`. Pass a smaller value when the buffer
is pre-allocated larger than the current live instance count.

Returns one `TLASHandle` for the whole batch.
"""
function Base.push!(hwtlas::VulkanTLAS{Tri}, mesh::GeometryBasics.Mesh,
                    instance_buf::LavaArray{VulkanInstanceRecord, 1};
                    n::Integer = length(instance_buf),
                    instance_mask::UInt8 = UInt8(0xff),
                    custom_index::UInt32 = UInt32(0),
                    sbt_offset::UInt32 = UInt32(0)) where {Tri}
    n_int = Int(n)
    n_int <= length(instance_buf) || error(
        "push!: n=$n_int exceeds instance_buf length $(length(instance_buf))")
    blas_idx = hwtlas_add_geometry!(hwtlas, mesh)
    blas = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    handle = Raycore.TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    push!(hwtlas.instance_batches, InstanceBatch{Tri}(blas, instance_buf, n_int, instance_mask, custom_index, handle, triangles, sbt_offset))
    hwtlas.handle_to_batch_idx[handle] = lastindex(hwtlas.instance_batches)
    hwtlas.dirty = true
    return handle
end

# ============================================================================
# GPU update kernel + update_transform!/update_transforms!
# ============================================================================

KA.@kernel cpu=false function update_instance_records_kernel!(
        records,
        @Const(transforms),
        blas_address::UInt64,
        cim::UInt32,
        sof::UInt32)
    i = @index(Global, Linear)
    @inbounds records[i] = VulkanInstanceRecord(transforms[i], cim, sof, blas_address)
end

"""
    Raycore.update_transforms!(hwtlas::VulkanTLAS, handle::TLASHandle,
                               transforms::LavaArray{Mat3x4f, 1})

Queue a bulk transform update for every instance in `handle`'s batch.
`transforms` must be a GPU-resident `LavaArray{Mat3x4f}`.
`length(transforms)` must equal the batch size. The actual kernel dispatch
happens in the next `sync!`, which issues a `MODE_UPDATE_KHR` refit after
applying all pending updates.
"""
function Raycore.update_transforms!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle,
                                    transforms::LavaArray{Mat3x4f, 1})
    haskey(hwtlas.handle_to_batch_idx, handle) || error("Invalid handle")
    batch = hwtlas.instance_batches[hwtlas.handle_to_batch_idx[handle]]
    length(transforms) == batch.n || error(
        "Transform count $(length(transforms)) != batch.n $(batch.n)")
    hwtlas.pending_updates[handle] = transforms
    hwtlas.transforms_dirty = true
    return nothing
end

# CPU-array overload: upload to GPU then delegate.
function Raycore.update_transforms!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle,
                                    transforms::AbstractVector{Mat3x4f})
    Raycore.update_transforms!(hwtlas, handle, LavaArray(collect(transforms); bq=hwtlas.bq))
end

# Mat4f convenience: mirrors Raycore's HWTLAS overloads — accept the natural
# homogeneous 4×4 form and convert at the boundary.
function Raycore.update_transforms!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle,
                                    transforms::AbstractVector{Mat4f})
    Raycore.update_transforms!(hwtlas, handle, map(Raycore.mat4_to_mat3x4, transforms))
end

"""
    update_transform!(hwtlas::VulkanTLAS, handle::TLASHandle, transform)

Set every instance in `handle`'s batch to the same transform. Accepts
`Mat3x4f` (canonical) or `Mat4f` (auto-converted). Returns true if the
handle was valid.
"""
function Raycore.update_transform!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle, transform::Mat3x4f)
    haskey(hwtlas.handle_to_batch_idx, handle) || return false
    batch = hwtlas.instance_batches[hwtlas.handle_to_batch_idx[handle]]
    Raycore.update_transforms!(hwtlas, handle, LavaArray(fill(transform, batch.n); bq=hwtlas.bq))
    return true
end

Raycore.update_transform!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle, transform::Mat4f) =
    Raycore.update_transform!(hwtlas, handle, Raycore.mat4_to_mat3x4(transform))

function _apply_pending_update!(batch::InstanceBatch, transforms::LavaArray{Mat3x4f, 1})
    backend = KA.get_backend(batch.instance_buf)
    cim = (batch.custom_index & 0x00FFFFFF) | (UInt32(batch.instance_mask) << 24)
    # Preserve the batch's SBT hit-group offset across refits (8-bit flags = 0).
    sof = batch.sbt_offset & 0x00FFFFFF
    update_instance_records_kernel!(backend)(
        batch.instance_buf, transforms,
        batch.blas.address, cim, sof;
        ndrange = batch.n)
end

# ============================================================================
# delete!
# ============================================================================

function Base.delete!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle)::Bool
    idx = get(hwtlas.handle_to_batch_idx, handle, nothing)
    idx === nothing && return false
    deleteat!(hwtlas.instance_batches, idx)
    delete!(hwtlas.handle_to_batch_idx, handle)
    # Reindex other handles whose batch index shifted left by one.
    for (h, j) in hwtlas.handle_to_batch_idx
        j > idx && (hwtlas.handle_to_batch_idx[h] = j - 1)
    end
    hwtlas.dirty = true
    return true
end

# ============================================================================
# Rebuild / refit helpers + sync!
# ============================================================================

# Try to reuse `prev` as the GPU sink for `data` via capacity-aware resize+copyto.
# `bq` is the device the fresh allocation lands on when `prev` cannot be reused —
# the TLAS's own queue, never the process default.
function _reuse_or_alloc(prev, data::AbstractArray{T}, bq) where T
    if prev isa LavaArray{T}
        resize!(prev, length(data))
        copyto!(prev, data)
        return prev
    end
    return LavaArray(data; bq)
end

# Allocate-or-reuse a combined LavaArray{VulkanInstanceRecord} that fits all
# `total` records, then GPU-copy each batch's instance_buf into the right
# offset.  The combined buffer is what build_tlas / refit_tlas! sees.
function _concat_batch_instances!(hwtlas::VulkanTLAS{Tri}) where {Tri}
    total = 0
    for batch in hwtlas.instance_batches
        total += batch.n
    end
    combined = hwtlas.combined_instance_buf
    if combined === nothing || length(combined) < total
        combined = LavaArray{VulkanInstanceRecord, 1}(undef, total;
                                                       bq=hwtlas.bq, extra_usage=AS_INPUT_USAGE)
        hwtlas.combined_instance_buf = combined
    end
    inst_offset = 0
    for batch in hwtlas.instance_batches
        # GPU->GPU copy at the right offset.  copyto! on LavaArray uses
        # vkCmdCopyBuffer (no CPU staging).
        Base.copyto!(combined, inst_offset + 1, batch.instance_buf, 1, batch.n)
        inst_offset += batch.n
    end
    return combined, total
end

# Compact `blas_list` / `blas_triangles` / `blas_offsets` to drop entries
# no longer referenced by any live batch.  Returns the dropped BLASes so the
# caller can `unsafe_free!` them after the rebuild.
function _compact_blas_list!(hwtlas::VulkanTLAS{Tri}) where {Tri}
    n_blas = length(hwtlas.blas_list)
    n_blas == 0 && return LavaBLAS[]
    used = Set{Int}()
    # Identify each BLAS by reference identity (===) since multiple batches
    # may share the same LavaBLAS object.
    for batch in hwtlas.instance_batches
        for (i, blas) in enumerate(hwtlas.blas_list)
            blas === batch.blas && push!(used, i)
        end
    end
    length(used) == n_blas && return LavaBLAS[]

    dropped = LavaBLAS[]
    new_blas       = LavaBLAS[]
    new_triangles  = Vector{Tri}[]
    new_offsets    = UInt32[]
    running_offset = UInt32(0)
    for i in 1:n_blas
        if i in used
            push!(new_blas, hwtlas.blas_list[i])
            push!(new_triangles, hwtlas.blas_triangles[i])
            push!(new_offsets, running_offset)
            running_offset += UInt32(length(hwtlas.blas_triangles[i]))
        else
            push!(dropped, hwtlas.blas_list[i])
        end
    end
    hwtlas.blas_list      = new_blas
    hwtlas.blas_triangles = new_triangles
    hwtlas.blas_offsets   = new_offsets
    return dropped
end

function rebuild_hw_tlas_from_batch!(hwtlas::VulkanTLAS{Tri}) where {Tri}
    # Drop unused BLASes (deleted batches may have left some unreferenced).
    dropped_blases = _compact_blas_list!(hwtlas)

    if isempty(hwtlas.instance_batches)
        # Caller (sync!) handles the empty path before us; assert defensively.
        return (nothing, nothing, nothing, nothing, dropped_blases)
    end

    # Rebuild always starts with a fresh combined buffer: the previous one
    # (if any) is now solely owned by the soon-to-be-freed `hw_tlas.preserves`.
    # Reusing it across builds shares a single LavaArray identity between
    # multiple HWTLAS preserves lists, and the older HWTLAS's `unsafe_free!` would
    # then free the buffer out from under the live HWTLAS.
    hwtlas.combined_instance_buf = nothing
    combined, total_n = _concat_batch_instances!(hwtlas)

    # After the compaction above, `blas_list` is exactly what the instances
    # reference; the TLAS carries the list so a trace can pin every one.
    hw_tlas = build_accel!(hwtlas.bq) do ctx
        build_tlas(ctx, combined, total_n; allow_update=true, blases=hwtlas.blas_list)
    end

    # Concatenate triangle metadata across batches.  Each batch's instances all
    # reference one BLAS, so all instances of batch i share the same offset
    # into all_tris (= the running tri_offset where batch i's triangles begin).
    all_tris = Tri[]
    per_inst_offsets = UInt32[]
    blas_offsets = UInt32[]
    tri_offset = UInt32(0)
    for batch in hwtlas.instance_batches
        push!(blas_offsets, tri_offset)
        append!(all_tris, batch.triangles)
        for _ in 1:batch.n
            push!(per_inst_offsets, tri_offset)
        end
        tri_offset += UInt32(length(batch.triangles))
    end

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
    off_gpu = _reuse_or_alloc(hwtlas.off_gpu, per_inst_offsets, hwtlas.bq)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu, dropped_blases)
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

    # Empty-topology path: drop the prior AS without rebuilding.
    if hwtlas.dirty && isempty(hwtlas.instance_batches)
        # Even with no batches we may still have unreferenced BLASes (e.g.,
        # the user delete!'d every handle).  Compact + free.
        dropped_blases = _compact_blas_list!(hwtlas)
        old_hw_tlas = hwtlas.hw_tlas
        old_tri_gpu = hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu
        hwtlas.hw_tlas    = nothing
        hwtlas.hw_accel   = nothing
        hwtlas.tri_gpu    = nothing
        hwtlas.off_gpu    = nothing
        hwtlas.dirty            = false
        hwtlas.transforms_dirty = false
        empty!(hwtlas.pending_updates)
        hwtlas.static_tlas = AdaptedAccel(hwtlas)
        old_hw_tlas === nothing || unsafe_free!(old_hw_tlas)
        old_tri_gpu === nothing || unsafe_free!(old_tri_gpu)
        old_off_gpu === nothing || unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    # Apply any queued transform updates to instance bufs before reading them.
    had_pending = !isempty(hwtlas.pending_updates)
    if had_pending
        for (handle, update) in hwtlas.pending_updates
            idx = get(hwtlas.handle_to_batch_idx, handle, nothing)
            idx === nothing && continue  # handle was deleted before sync!
            _apply_pending_update!(hwtlas.instance_batches[idx], update)
        end
        empty!(hwtlas.pending_updates)
    end

    if hwtlas.dirty
        # Topology change: full rebuild (also handles the "first sync after
        # construct" case since the constructor leaves dirty=true).
        # _concat_batch_instances! does vkCmdCopyBuffer from each instance_buf;
        # build_as_on_gpu_impl only barriers on AS-to-AS (not SHADER_WRITE).
        # If we just dispatched pending kernels, flush so their writes are
        # visible to the GPU-GPU copies. Rare path (topology + transform
        # update in the same frame).
        had_pending && vk_flush!(hwtlas.bq)
        hw_tlas, hw_accel, tri_gpu, off_gpu, dropped_blases =
            rebuild_hw_tlas_from_batch!(hwtlas)

        old_hw_tlas = hwtlas.hw_tlas
        # tri_gpu / off_gpu get reused-in-place when sizes permit -- do NOT
        # release unless the rebuild returned a different object.
        old_tri_gpu = hwtlas.tri_gpu === tri_gpu ? nothing : hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu === off_gpu ? nothing : hwtlas.off_gpu

        hwtlas.hw_tlas   = hw_tlas
        hwtlas.hw_accel  = hw_accel
        hwtlas.tri_gpu   = tri_gpu
        hwtlas.off_gpu   = off_gpu
        hwtlas.dirty            = false
        hwtlas.transforms_dirty = false
        hwtlas.static_tlas = AdaptedAccel(hwtlas)
        old_hw_tlas === nothing || unsafe_free!(old_hw_tlas)
        old_tri_gpu === nothing || unsafe_free!(old_tri_gpu)
        old_off_gpu === nothing || unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    # Transforms-only path: MODE_UPDATE_KHR refit.
    if hwtlas.transforms_dirty
        if hwtlas.hw_tlas === nothing || !hwtlas.hw_tlas.allow_update
            # Defensive: if the prior build wasn't refit-capable, fall back to
            # full rebuild.  Should not happen on normal use since
            # rebuild_hw_tlas_from_batch! always sets allow_update=true.
            hwtlas.dirty = true
            return Raycore.sync!(hwtlas)
        end
        combined, total_n = _concat_batch_instances!(hwtlas)
        build_accel!(hwtlas.bq) do ctx
            refit_tlas!(ctx, hwtlas.hw_tlas, combined, total_n)
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
# Polymorphic with the SW path's `Raycore.closest_hit(::StaticTLAS, ray)`:
# same return tuple shape `(hit, primitive, t, bary, inst_custom_idx)`.  A
# kernel written against `Raycore.closest_hit(accel, ray)` runs unchanged on
# either backend; multiple dispatch picks the right traversal.
#
# Lowers to OpRayQueryInitializeKHR/Proceed/Get*KHR via the lava_ray_query_*
# intrinsics.  The kernel must be compiled with `enable_ray_query=true`
# (auto-set when a `tlas=` kwarg is passed to `lava_launch!`) so the SPIR-V
# emitter binds the HWTLAS descriptor at set 0 binding 0.

@inline function _hw_rq_collect(accel::AdaptedAccel)
    while lava_ray_query_proceed()
        # Opaque triangles auto-commit; no any-hit decision here.
    end
    kind = lava_ray_query_get_type(true)  # committed
    if kind != UInt32(1)  # not RayQueryCommittedIntersectionTriangleKHR
        return (false, accel.empty, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    t = lava_ray_query_get_t(true)
    inst_id = lava_ray_query_get_instance_id(true)
    inst_custom_idx = lava_ray_query_get_instance_custom_index(true)
    prim_idx = lava_ray_query_get_primitive_index(true)
    bx, by = lava_ray_query_get_barycentrics(true)

    @inbounds tri_idx = Int(accel.offsets[inst_id + UInt32(1)]) + Int(prim_idx) + 1
    @inbounds tri = accel.triangles[tri_idx]

    bary = SVector{3,Float32}(1f0 - bx - by, bx, by)
    return (true, tri, t, bary, inst_custom_idx)
end

@propagate_inbounds function Raycore.closest_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay)
    o = ray.o; d = ray.d
    lava_ray_query_init(UInt32(0), UInt32(0xFF),
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return _hw_rq_collect(accel)
end

@propagate_inbounds function Raycore.any_hit(accel::AdaptedAccel, ray::Raycore.AbstractRay)
    o = ray.o; d = ray.d
    # SPIR-V RayFlagsTerminateOnFirstHitKHR = 4 — exit traversal at first commit.
    lava_ray_query_init(UInt32(4), UInt32(0xFF),
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return _hw_rq_collect(accel)
end

"""
    tlasset!(owner, dev, pipeline, tlas) -> VkDescriptorSet

The descriptor set binding `tlas` for `pipeline`'s layout, owned by whoever owns
the commands that bind it.

A [`Recording`](@ref) keeps ONE per (layout, acceleration structure) pair and
holds its pool for as long as it can be submitted. A one-shot allocates a fresh
one per dispatch and pins it, which is what both used to do.

Per-dispatch allocation was defended as removing a class of bug, and the bug was
real: a cache keyed by `(layout, objectid(LavaTLAS))` grew without bound because
`Raycore.sync!` makes a new `LavaTLAS` per rebuild, and its `WeakRef` eviction
lagged Julia's GC, so pools were destroyed while their sets were still in flight.
Both halves of that are about a cache with no owner. This one has exactly one:
nothing evicts, and `release!` is the only thing that frees.

The recorded case is also where the per-dispatch version stopped being a lifetime
fix at all — a recording bakes the set into its command buffer, so allocating a
fresh set while writing it means N pools kept alive for one plan, all naming the
same acceleration structure.
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
    pin!(o, desc_pool)
    return desc_set
end

function bindtlas!(e::Emitter, pipeline::LavaComputePipeline, tlas::VulkanTLAS)
    lava_tlas = tlas.hw_tlas::LavaTLAS
    set = tlasset!(e.owner, e.ctx.device, pipeline, lava_tlas)
    VK.cmd_bind_descriptor_sets(e.cmd, VK.PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout, UInt32(0), [set], UInt32[])
    # A ray query walks the HWTLAS into its BLASes and reads their storage, so
    # every one of them is held — `pintrace!` is core's statement of that.
    pintrace!(e, lava_tlas)
    return nothing
end

# Vulkan drives an SBT through `vkCmdTraceRaysIndirect`. See
# `supports_rt_pipeline` in `raytracing/api.jl`.
supports_rt_pipeline(::LavaBackend) = true
supports_rt_pipeline(::VulkanTLAS) = true

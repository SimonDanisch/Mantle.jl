# The hardware top-level acceleration structure, on every backend. Core owns the batches, their
# mutation, the commit decision, the instance writes, the rebuild and the refit; a backend builds,
# refits and releases its native structure and packs its driver's instance record, through the
# verbs declared below.

"""
    BatchedTLAS{Tri,D,B} <: HWTLAS{Tri}

The hardware top-level structure of device `D`, over bottom-level structures of the backend's
type `B` (`blastype`), holding triangles of type `Tri`. `push!` adds a batch of instances of
one geometry, `Raycore.sync!` builds: a rebuild after `push!`/`delete!`/a visibility or opacity
change, an in-place refit after `update_transforms!` or after a kernel wrote
`Raycore.instance_buffer` and `Raycore.refit!` was called. Read the adapted form from
`static_tlas` per dispatch: a rebuild replaces it.
"""
mutable struct BatchedTLAS{Tri,D,B} <: HWTLAS{Tri}
    # The device it is built on. `backend`, as Raycore's software `TLAS` names it, so
    # scene code (Hikari's procedural payloads) allocates for either kind alike.
    backend::D
    # Geometry, accumulated by push!; after a rebuild exactly what the instances name.
    blas_list::Vector{B}
    instances::InstanceBatches{InstanceBatch{Tri,B,Any}}
    root_aabb::Raycore.Bounds3
    # The backend's built structure (`build_tlas`), which owns `native`; `nothing` before a build.
    built::Any
    # Every batch's driver records in build order (`nativerecords`), which a refit rewrites.
    native::Any
    # The same instances as `Raycore.InstanceRecord`s, which hits read transforms from; written
    # in place by a refit, so an adapted form a plan holds sees it.
    records::Any
    # Every batch's shading triangles and each instance's offset into them (`trianglemetadata`).
    tri_gpu::Any
    off_gpu::Any
    # The adapted form, `nothing` before the first sync!.
    static_tlas::Any
    # A push!/delete!/visibility or opacity change: a rebuild.
    dirty::Bool
    # Queued updates or kernel-written records: a refit suffices.
    transforms_dirty::Bool
    # Transform updates by batch handle, last write wins, written at the next sync!.
    pending::Dict{Raycore.TLASHandle,Any}
    # The scene's procedural payload or `nothing`, passed to the traversal through `AdaptedAccel`.
    procedural::Any
end

"""
    HWTLAS{Tri}(device) -> BatchedTLAS

An empty hardware top-level structure on `device`, built on its batch queue.
"""
HWTLAS{Tri}(device::Device) where {Tri} =
    BatchedTLAS{Tri,typeof(device),blastype(device)}(device, blastype(device)[],
        InstanceBatches{InstanceBatch{Tri,blastype(device),Any}}(), Raycore.Bounds3(),
        nothing, nothing, nothing, nothing, nothing, nothing, true, false,
        Dict{Raycore.TLASHandle,Any}(), nothing)

# ── The backend's verbs ─────────────────────────────────────────────────────
# Besides the build verbs under `build_blas`/`build_tlas`/`refit_tlas!` (`raytracing/api.jl`)
# and `instancerecord` (`raytracing/instances.jl`).

"""    blastype(device) -> Type

The type of the backend's bottom-level structures (`build_blas`)."""
function blastype end

"""    nativerecords(device, n) -> device array

`n` uninitialised driver instance records, which `instancerecord` packs and `build_tlas` reads."""
function nativerecords end

"""    geometryvalue(blas, index) -> isbits

What a driver instance record names its geometry by: the structure `blas`, at one-based `index`
into the list `build_tlas` was given (a device address, or the index itself)."""
function geometryvalue end

"""    scenevalue(tlas) -> isbits or nothing

A built top-level structure as a kernel value where the backend's query takes one; `nothing`
where it is bound out of band."""
function scenevalue end

# ── Registration and mutation ───────────────────────────────────────────────

function geometryindex!(t::BatchedTLAS, blas)
    i = findfirst(b -> b === blas, t.blas_list)
    i === nothing || return i
    push!(t.blas_list, blas)
    return length(t.blas_list)
end

batchgeometry!(t::BatchedTLAS, blas, triangles) = (geometryindex!(t, blas), triangles)

function batchgeometry!(t::BatchedTLAS{Tri}, mesh::GeometryBasics.Mesh, _) where {Tri}
    triangles = meshtriangles(Tri, mesh)
    vertices = trianglevertices(triangles)
    blas = build_accel!(batchqueue(t.backend)) do ctx
        build_blas(ctx, vertices, UInt32.(0:length(vertices) - 1))
    end
    i = geometryindex!(t, blas)
    for tri in triangles, v in tri.vertices
        t.root_aabb = Raycore.union(t.root_aabb, Raycore.Bounds3(Point3f(v)))
    end
    return i, triangles
end

"""The device array a batch's records live in: a device array as it is, else uploaded."""
instancearray(t::BatchedTLAS, records) = isdevicearray(records) ? records : devicearray(t.backend, records)

function addbatch!(t::BatchedTLAS, i::Int, records, n::Int, triangles, opaque::Bool)
    B = eltype(t.instances)
    handle = register!(t.instances, h -> B(t.blas_list[i], records, n, h, triangles, false, opaque))
    t.dirty = true
    return handle
end

"""
    push!(t::BatchedTLAS, geometry, records; n=length(records), triangles, opaque=true) -> handle

Register one batch of instance records; only `records[1:n]` are instances. A mesh
builds its BLAS, a pre-built BLAS is registered once by identity. An invalid `n`
is refused before any geometry is built. Native device arrays are kept, others uploaded.
`opaque = false` makes `Raycore.transparent_hits` record the batch's hits rather
than stop at them (`Raycore.set_opaque!`).
"""
function Base.push!(t::BatchedTLAS{Tri}, geometry,
                    records::AbstractVector{Raycore.InstanceRecord};
                    n::Integer = length(records), triangles::Vector{Tri} = Tri[],
                    opaque::Bool = true) where {Tri}
    count = Int(n)
    0 <= count <= length(records) || throw(ArgumentError(
        "push!: n=$count must be between 0 and $(length(records)) records"))
    i, metadata = batchgeometry!(t, geometry, triangles)
    return addbatch!(t, i, instancearray(t, records), count, metadata, opaque)
end

function Base.push!(t::BatchedTLAS, geometry, transform::Mat4f = Mat4f(I);
                    instance_id::UInt32 = UInt32(0), instance_mask::UInt8 = UInt8(0xff),
                    opaque::Bool = true)
    return push!(t, geometry, instancerecords([transform], [instance_id], instance_mask); opaque)
end

function Base.push!(t::BatchedTLAS, geometry, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                    instance_mask::UInt8 = UInt8(0xff), opaque::Bool = true)
    records = instancerecords(transforms, instance_ids, instance_mask)
    return push!(t, geometry, records; opaque)
end

function requirebatch(t::BatchedTLAS, handle::Raycore.TLASHandle, operation)
    batch = batchof(t.instances, handle)
    batch === nothing && throw(ArgumentError("$operation: unknown handle $handle"))
    return batch
end

# `Mat3x4f` device arrays pass as they are; anything else, device `Mat4f` arrays included, is
# converted on the host.
transformbuffer(t::BatchedTLAS, transforms::AbstractVector) =
    instancearray(t, Mat3x4f[m isa Mat3x4f ? m : Raycore.mat4_to_mat3x4(m) for m in Array(transforms)])
transformbuffer(t::BatchedTLAS, transforms::AbstractVector{Mat3x4f}) = instancearray(t, transforms)

"""
    Raycore.update_transforms!(t::BatchedTLAS, handle, transforms)

Queue new transforms for every instance of the batch `handle` names, written into its records at
the next `sync!` in the same submission as the refit; a later update of the batch replaces it.
"""
function Raycore.update_transforms!(t::BatchedTLAS, handle::Raycore.TLASHandle,
                                    transforms::AbstractVector)
    b = requirebatch(t, handle, "update_transforms!")
    length(transforms) == b.n || throw(ArgumentError(
        "update_transforms!: $(length(transforms)) transforms for a batch of $(b.n) instances"))
    t.pending[handle] = transformbuffer(t, transforms)
    t.transforms_dirty = true
    return nothing
end

function Raycore.update_transform!(t::BatchedTLAS, handle::Raycore.TLASHandle, transform)
    b = requirebatch(t, handle, "update_transform!")
    canonical = transform isa Mat3x4f ? transform : Raycore.mat4_to_mat3x4(transform)
    return Raycore.update_transforms!(t, handle, fill(canonical, b.n))
end

"""Map a flat instance range onto exactly one complete batch, before mutation."""
function Raycore.update_instance_transforms!(t::BatchedTLAS, transforms::AbstractVector,
                                             n_to_update::Integer, first_idx::Integer = 1)
    n, i0 = Int(n_to_update), Int(first_idx)
    n > 0 || return nothing
    base = 0
    for b in t.instances
        if i0 <= base + b.n
            (i0 == base + 1 && n == b.n) || error(
                "update_instance_transforms!: instances $(i0):$(i0 + n - 1) do not " *
                "cover exactly one batch (batch at $(base + 1):$(base + b.n)). " *
                "A $(nameof(typeof(t))) updates whole batches; address it by handle instead.")
            return Raycore.update_transforms!(t, b.handle, transforms)
        end
        base += b.n
    end
    error("update_instance_transforms!: instance index $(i0) past the end " *
          "($(base) instances in $(length(t.instances)) batches).")
end

function Base.delete!(t::BatchedTLAS, handle::Raycore.TLASHandle)
    delete!(t.instances, handle) || return false
    delete!(t.pending, handle)
    t.dirty = true
    return true
end

Raycore.set_visible!(t::BatchedTLAS, handle::Raycore.TLASHandle, visible::Bool) =
    setbatch!(t, handle, :hidden, !visible)
Raycore.set_opaque!(t::BatchedTLAS, handle::Raycore.TLASHandle, opaque::Bool) =
    setbatch!(t, handle, :opaque, opaque)

function setbatch!(t::BatchedTLAS, handle::Raycore.TLASHandle, field::Symbol, value::Bool)
    changed = setbatch!(t.instances, handle, field, value)
    changed === nothing && return false
    changed && (t.dirty = true)
    return true
end

Raycore.world_bound(t::BatchedTLAS) = t.root_aabb
Raycore.n_geometries(t::BatchedTLAS) = length(t.blas_list)
Raycore.n_instances(t::BatchedTLAS) = ninstances(t.instances)
Raycore.instance_buffer(t::BatchedTLAS, handle::Raycore.TLASHandle) = recordsof(t.instances, handle)
Raycore.refit!(t::BatchedTLAS) = (t.transforms_dirty = true; Raycore.sync!(t))

"""Block until every build of `t` has completed."""
Raycore.wait_for_gpu!(t::BatchedTLAS) = (flush!(batchqueue(t.backend)); t)

supports_procedural_traversal(t::BatchedTLAS) = supports_procedural_traversal(t.backend)

# ── Building ────────────────────────────────────────────────────────────────

"""Build modes of [`build!`](@ref): an in-place update of the built structure, or a new one."""
struct Refit end
struct Rebuild end

"""
    writeinstances!(t, native, shading)

Write the queued transform updates into their batches' records, then every batch's instances, in
build order, into `native` (driver records) and `shading` ([`instances_kernel!`](@ref)), in one
submission, which the build or refit after it is ordered behind: a scene is one batch per plot.
"""
function writeinstances!(t::BatchedTLAS, native, shading; workgroup_size = (64, 1, 1))
    oneshot!(batchqueue(t.backend); tag = :instances) do e
        updated = false
        for (handle, transforms) in t.pending
            b = batchof(t.instances, handle)
            (b === nothing || b.n == 0) && continue
            emitkernel!(e, update_instance_records_kernel!, b.records, transforms; ndrange = b.n,
                        workgroup_size)
            updated = true
        end
        # The updates write records the conversion below reads.
        updated && emitpreparebarrier!(e)
        position = IdDict{Any,Int}(blas => k for (k, blas) in enumerate(t.blas_list))
        offset = 0
        for b in t.instances
            b.n == 0 && continue
            emitkernel!(e, instances_kernel!, native, shading, b.records, Int32(offset), Int32(b.n),
                geometryvalue(b.blas, position[b.blas]), b.hidden ? UInt32(0) : UInt32(0xff), b.opaque;
                ndrange = b.n, workgroup_size)
            offset += b.n
        end
    end
    empty!(t.pending)
    return nothing
end

"""Drop the geometry no batch names any more, and return it."""
function compactgeometry!(t::BatchedTLAS)
    used = Base.IdSet{Any}(b.blas for b in t.instances)
    dropped = filter(blas -> !(blas in used), t.blas_list)
    filter!(blas -> blas in used, t.blas_list)
    return dropped
end

"""`data` on the device, at least one element: kernels bind it whatever the scene holds."""
nonempty(device, data::Vector, filler) = devicearray(device, isempty(data) ? [filler] : data)

"""
    build!(t::BatchedTLAS, ::Refit)
    build!(t::BatchedTLAS, ::Rebuild)

Write the instances and refit the built structure in place, so an adapted form a plan holds sees
the change; or build a new structure and its traversal arrays, retiring the replaced ones.
"""
function build!(t::BatchedTLAS, ::Refit)
    writeinstances!(t, t.native, t.records)
    build_accel!(batchqueue(t.backend)) do ctx
        refit_tlas!(ctx, t.built)
    end
    return t
end

function build!(t::BatchedTLAS{Tri}, ::Rebuild) where {Tri}
    dropped = compactgeometry!(t)
    total = ninstances(t.instances)
    # At least one record: the build reads their address, and a trace binds the shading copy.
    native = nativerecords(t.backend, max(total, 1))
    shading = devicearray(t.backend, Raycore.InstanceRecord, max(total, 1))
    writeinstances!(t, native, shading)
    # A zero-instance structure is valid and every ray misses it; a query with none bound is not.
    built = build_accel!(batchqueue(t.backend)) do ctx
        build_tlas(ctx, native, total; allow_update = true, blases = t.blas_list)
    end
    metadata = trianglemetadata(Tri, t.instances)
    old = (t.built, t.records, t.tri_gpu, t.off_gpu)
    t.built, t.native, t.records = built, native, shading
    t.tri_gpu = nonempty(t.backend, metadata.triangles, Raycore.empty_triangle(Tri))
    t.off_gpu = nonempty(t.backend, metadata.hits, NOTRIANGLES)
    # The old driver records go with the old structure, which owns them.
    foreach(x -> x === nothing || unsafe_free!(x), old)
    foreach(unsafe_free!, dropped)
    return t
end

"""
    Raycore.sync!(t::BatchedTLAS) -> t

Commit pending changes: nothing when clean, an in-place refit when only
transforms changed, a rebuild otherwise.
"""
function Raycore.sync!(t::BatchedTLAS)
    built = t.static_tlas !== nothing
    built && !t.dirty && !t.transforms_dirty && return t
    if built && !t.dirty
        build!(t, Refit())
    else
        build!(t, Rebuild())
        t.static_tlas = AdaptedAccel(t)
    end
    t.dirty = false
    t.transforms_dirty = false
    return t
end

"""The host form of a synced structure, which keeps `t` for the backend to bind."""
AdaptedAccel(t::BatchedTLAS{Tri}) where {Tri} =
    AdaptedAccel(t, t.tri_gpu, t.off_gpu, Raycore.empty_triangle(Tri), scenevalue(t.built),
                 t.procedural, t.records)

"""Sync, then the adapted form: `t` adapts to what a kernel traverses."""
function Adapt.adapt_structure(to, t::BatchedTLAS)
    Raycore.sync!(t)
    return t.static_tlas::AdaptedAccel
end

# Held through what it adapts to: `static_tlas` leads back to `t`, and the device is cyclic.
holdleaves!(holder, t::BatchedTLAS) = holdleaves!(holder, t.static_tlas)

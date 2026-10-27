# Metal's incremental hardware TLAS and its device-side traversal: the query verbs
# (`raytracing/accel.jl`) on Metal's intersection query, through the AIR intrinsics
# Metal.jl wraps (`Metal.intersection_query`).

# Metal cannot infer residency for a structure a kernel names only by its
# `gpuResourceID`, and a non-resident structure makes rays miss silently. Through
# Metal.jl's per-queue residency set rather than `add_allocation!`: the set holds
# strong references, and only allocations Metal.jl tracks leave it when freed
# (`forget_resident!`).
function make_resident!(d::MetalDevice, resources...)
    Metal.install_queue_residency!(cmdqueue(d), d.dev)
    for r in resources
        r === nothing && continue
        Metal.make_persistently_resident!(r, d.dev)
    end
    return nothing
end

"""Metal's query object, allocated in thread memory for one query."""
struct MetalRayQuery
    q::Metal.IntersectionQuery
end

# The structure travels as its `gpuResourceID` (`AdaptedAccel.scene`). Forced opaque
# for `query_begin`; a transparent query leaves it to the instances (`writeinstances!`).
@inline function query_reset(accel::AdaptedAccel{<:Any, <:Metal.MtlDeviceArray}, ray::Raycore.AbstractRay,
                             mask::Integer, forced_opacity::UInt32, anyhit::Bool)
    q = Metal.intersection_query()
    o = ray.o; d = ray.d
    Metal.query_reset!(q, (Float32(o[1]), Float32(o[2]), Float32(o[3])),
                       (Float32(d[1]), Float32(d[2]), Float32(d[3])),
                       Float32(ray.t_min), Float32(ray.t_max), accel.scene, mask % UInt32,
                       forced_opacity, anyhit)
    return MetalRayQuery(q)
end

@inline query_begin(accel::AdaptedAccel{<:Any, <:Metal.MtlDeviceArray}, ray::Raycore.AbstractRay,
                    mask::Integer, anyhit::Bool) =
    query_reset(accel, ray, mask, Metal.FORCED_OPACITY_OPAQUE, anyhit)
@inline query_begin_transparent(accel::AdaptedAccel{<:Any, <:Metal.MtlDeviceArray}, ray::Raycore.AbstractRay,
                                mask::Integer) =
    query_reset(accel, ray, mask, Metal.FORCED_OPACITY_NONE, true)

@inline query_next!(m::MetalRayQuery) = Metal.query_next!(m.q)
@inline candidate_is_box(m::MetalRayQuery) = Metal.candidate_type(m.q) == Metal.INTERSECTION_BOUNDING_BOX
@inline candidate_primitive_index(m::MetalRayQuery) = Int(Metal.candidate_primitive_id(m.q)) + 1
# The candidate's ray is in the candidate instance's object space.
@inline candidate_object_ray(m::MetalRayQuery) =
    (Vec3f(Metal.candidate_ray_origin(m.q)...), Vec3f(Metal.candidate_ray_direction(m.q)...))
@inline candidate_ids(m::MetalRayQuery) =
    (Metal.candidate_instance_id(m.q), Metal.candidate_primitive_id(m.q), Metal.candidate_user_instance_id(m.q))
@inline function candidate_triangle(m::MetalRayQuery)
    u, v = Metal.candidate_triangle_barycentric(m.q)
    return (Metal.candidate_triangle_distance(m.q), u, v)
end
@inline commit_triangle!(m::MetalRayQuery) = Metal.commit_triangle!(m.q)
@inline commit_intersection!(m::MetalRayQuery, t) = Metal.commit_bounding_box!(m.q, Float32(t))

@inline function committed_kind(m::MetalRayQuery)
    kind = Metal.committed_type(m.q)
    kind == Metal.INTERSECTION_TRIANGLE && return Raycore.HIT_TRIANGLE
    kind == Metal.INTERSECTION_BOUNDING_BOX && return Raycore.HIT_GENERATED
    return Raycore.HIT_MISS
end
# `instance_id` is the instance INDEX, which picks the instance's triangles;
# `user_instance_id` its CUSTOM index (the descriptor's `userID`), which Hikari reads
# as a per-instance material override. Two different numbers, as on Vulkan.
@inline committed_ids(m::MetalRayQuery) =
    (Metal.committed_distance(m.q), Metal.committed_instance_id(m.q),
     Metal.committed_primitive_id(m.q), Metal.committed_user_instance_id(m.q))
@inline committed_barycentrics(m::MetalRayQuery) = Metal.committed_triangle_barycentric(m.q)
@inline query_end!(m::MetalRayQuery) = Metal.query_release!(m.q)

# Without procedural geometry no candidate of an opaque query needs an answer, and
# Metal traverses in one `intersect` call instead of an intersection query, its
# native spelling of the same traversal: the query was 5-10% slower on an M5 in
# subsurface and alpha-cutout scenes, which trace most. The loop in `query_record`
# then never runs. A transparent query, whose candidates are recorded, is a query.
"""Metal's answer to a traversal with no candidates: one `intersect` call's result."""
struct MetalIntersection
    r::Metal.IntersectionResult
end

@inline function query_begin(accel::AdaptedAccel{<:Any, <:Metal.MtlDeviceArray, <:Any, <:Any, <:Any, Nothing},
                             ray::Raycore.AbstractRay, mask::Integer, anyhit::Bool)
    o = ray.o; d = ray.d
    r = Metal.intersect((Float32(o[1]), Float32(o[2]), Float32(o[3])),
                        (Float32(d[1]), Float32(d[2]), Float32(d[3])),
                        Float32(ray.t_min), Float32(ray.t_max), accel.scene, mask % UInt32,
                        Metal.FORCED_OPACITY_OPAQUE, true, anyhit)
    return MetalIntersection(r)
end

@inline query_next!(::MetalIntersection) = false
@inline committed_kind(m::MetalIntersection) =
    m.r.type == Metal.INTERSECTION_TRIANGLE ? Raycore.HIT_TRIANGLE : Raycore.HIT_MISS
@inline committed_ids(m::MetalIntersection) =
    (m.r.distance, m.r.instance_id, m.r.primitive_id, m.r.user_instance_id)
@inline committed_barycentrics(m::MetalIntersection) = m.r.barycentric
@inline query_end!(::MetalIntersection) = nothing

"""One batch of instances that all reference the same BLAS."""
const MetalInstanceBatch{Tri} =
    InstanceBatch{Tri,Int,MtlVector{Raycore.InstanceRecord}}

"""
    MetalHWTLAS{Tri} <: BatchedTLAS{Tri}

Metal's incremental acceleration structure, with `VulkanTLAS`'s contract: `push!`
geometry, `Raycore.sync!` to build, read the adapted form from `static_tlas`.
Core's `BatchedTLAS` owns registration and batch mutation; this type owns native
builds, record packing, residency and traversal.
"""
mutable struct MetalHWTLAS{Tri} <: BatchedTLAS{Tri}
    backend::MetalDevice

    blas_list::Vector{MetalBLAS}
    # Order, handles and reindex on delete are core's (`raytracing/batches.jl`).
    instances::Mantle.InstanceBatches{MetalInstanceBatch{Tri}}

    root_aabb::Raycore.Bounds3

    built::Any            # Union{Nothing, MetalTLAS{Tri}}
    tri_gpu::Any          # Union{Nothing, MtlVector{Tri}}
    off_gpu::Any          # Union{Nothing, MtlVector{UInt32}}
    # Every batch's records in build order, read by hits for the instance
    # transform (`AdaptedAccel.instances`). A refit writes them in place, so an
    # adapted form a plan holds sees the change.
    records::Any          # Union{Nothing, MtlVector{Raycore.InstanceRecord}}
    static_tlas::Any

    # The scene's procedural geometry or `nothing`, which folds the box branch out of
    # the traversal. Set by the caller that pushes an AABB BLAS (Hikari's
    # `push!(scene, mesh, ::FEMMaterial)`), passed to kernels through `AdaptedAccel`.
    procedural::Any

    dirty::Bool
    transforms_dirty::Bool
end

HWTLAS{Tri}(d::MetalDevice; kw...) where {Tri} = MetalHWTLAS{Tri}(d; kw...)

MetalHWTLAS{Tri}(d::MetalDevice) where {Tri} =
    MetalHWTLAS{Tri}(d,
        MetalBLAS[],
        Mantle.InstanceBatches{MetalInstanceBatch{Tri}}(),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing, nothing, nothing,
        true, false)

Raycore.wait_for_gpu!(t::MetalHWTLAS) = (waitidle(t.backend); t)

AdaptedAccel(t::MetalHWTLAS{Tri}) where {Tri} =
    AdaptedAccel(t, t.tri_gpu, t.off_gpu, Raycore.empty_triangle(Tri),
                 reinterpret(UInt64, t.built.handle.gpuResourceID), t.procedural, t.records)

function Adapt.adapt_structure(to, t::MetalHWTLAS)
    Raycore.sync!(t)
    return t.static_tlas::AdaptedAccel
end

# Held through its adapted form: `static_tlas` leads back to `t`, so the generic
# walker would recurse forever.
Mantle.holdleaves!(holder, t::MetalHWTLAS) = Mantle.holdleaves!(holder, t.static_tlas)

"""Adapt the arrays, dropping the host `hwtlas` owner."""
Adapt.adapt_structure(to::Metal.Adaptor, accel::AdaptedAccel) = adaptaccel(to, accel)

# `vertices` comes packed from core, three consecutive vertices per triangle.
function buildgeometry!(t::MetalHWTLAS, vertices)
    vbuf = MTL.MTLBuffer(t.backend.dev, sizeof(vertices); storage = Metal.SharedStorage)
    GC.@preserve vertices unsafe_copyto!(convert(Ptr{NTuple{3,Float32}}, MTL.contents(vbuf)),
                                        pointer(vertices), length(vertices))
    return build_blas(t.backend, vbuf, length(vertices) ÷ 3)
end

geometryref(::MetalHWTLAS, i::Int) = i
instancearray(::MetalHWTLAS, records::AbstractVector) = MtlArray(records)
instancearray(::MetalHWTLAS, records::MtlArray) = records

# Writes the transforms into the batch's records on the device at once; the
# refit happens at `sync!`.
function updatebatch!(t::MetalHWTLAS, batch, transforms)
    batch.n == 0 || KI.Kernel(t.backend, update_instance_records_kernel!)(
        batch.records, transforms; ndrange = batch.n)
    return t
end

# The descriptor Metal builds from, naming geometry `index` (zero-based) in the
# structure's list, opaque or not whatever its geometry was built as.
@inline instancerecord(::Type{MetalInstanceDescriptor}, r, index::UInt32, keep::UInt32, opaque::Bool) =
    MetalInstanceDescriptor(r, index, opaque ? UInt32(MTL.MTLAccelerationStructureInstanceOptionOpaque) :
                                              UInt32(MTL.MTLAccelerationStructureInstanceOptionNonOpaque), keep)

"""
    writeinstances!(t, descriptors, shading)

Write every batch's instances with [`instances_kernel!`](@ref) and wait: the
build or refit that follows runs on a command buffer of its own, which nothing
orders after Metal.jl's batch.
"""
function writeinstances!(t::MetalHWTLAS, descriptors::MtlArray, shading::MtlArray)
    offset = 0
    for b in t.instances
        b.n == 0 && continue
        KI.Kernel(t.backend, instances_kernel!)(descriptors, shading, b.records,
            Int32(offset), Int32(b.n), UInt32(b.blas - 1),
            b.hidden ? UInt32(0) : UInt32(0xff), b.opaque; ndrange = b.n)
        offset += b.n
    end
    KA.synchronize(t.backend)
    return nothing
end

# A transform-only change refits in place, keeping the structure and its
# `gpuResourceID` (why it is built `refittable`). A rebuild makes a new structure
# and `static_tlas`, while Hikari keeps tracing its cached adapted scene until
# `accel.dirty`, so a moved mesh would render unchanged.
canrefit(t::MetalHWTLAS) =
    t.built !== nothing && t.built.refittable && t.built.count == ninstances(t.instances)

# Structure and shading records take the same transforms in place: recorded
# kernels retain the records' device view.
function build!(t::MetalHWTLAS, ::Refit)
    writeinstances!(t, t.built.instances, t.records)
    refit_tlas!(t.backend, t.built)
    return t
end

function build!(t::MetalHWTLAS{Tri}, ::Rebuild) where {Tri}
    total = ninstances(t.instances)
    # An empty scene is built too (every ray misses): `adapt_structure` asserts
    # `static_tlas::AdaptedAccel`. One TLAS instance per record, naming its
    # batch's BLAS by index into `blas_list`; a batch without triangles gets offset
    # `NOTRIANGLES`, so its hits answer the empty triangle, not the next batch's.
    metadata = Mantle.trianglemetadata(Tri, t.instances)

    # At least one element, as on Vulkan: kernels bind these whatever the scene holds.
    descriptors = MtlArray{MetalInstanceDescriptor}(undef, max(total, 1))
    t.records = MtlArray{Raycore.InstanceRecord}(undef, max(total, 1))
    writeinstances!(t, descriptors, t.records)
    t.built = build_accel!(t.backend, t.blas_list, descriptors, total; refittable = true)
    t.tri_gpu = MtlArray(metadata.triangles)
    t.off_gpu = MtlArray(metadata.hits)

    # Non-resident structures are not mapped for the dispatch: every ray misses silently.
    make_resident!(t.backend, t.built.handle, (b.handle for b in t.blas_list)...)
    return t
end

# AABB geometry: the query offers each box to the payload's `procedural_candidate`.
supports_procedural_traversal(::MetalDevice) = true
# Every Apple-silicon GPU builds and traverses `MTLAccelerationStructure`s.
supports_hwtlas(::MetalDevice) = true
supports_procedural_traversal(::MetalHWTLAS) = true

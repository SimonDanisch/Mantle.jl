# Metal's half of core's `BatchedTLAS` (`raytracing/mutation.jl`): the native verbs a
# build and a trace need, and the query verbs (`raytracing/accel.jl`) on Metal's
# intersection query, through the AIR intrinsics Metal.jl wraps (`Metal.intersection_query`).

"""Metal's query object, allocated in thread memory for one query."""
struct MetalRayQuery
    q::Metal.IntersectionQuery
end

# The structure travels as its `gpuResourceID` (`AdaptedAccel.scene`). Forced opaque
# for `query_begin`; a transparent query leaves it to the instances (`instancerecord`).
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

blastype(::MetalDevice) = MetalBLAS

nativerecords(d::MetalDevice, n::Integer) = devicearray(d, MetalInstanceDescriptor, Int(n))

# An instance names its BLAS by zero-based index into `instancedAccelerationStructures`.
geometryvalue(::MetalBLAS, index::Integer) = UInt32(index - 1)

# The query takes the structure as its `gpuResourceID`.
scenevalue(t::MetalTLAS) = reinterpret(UInt64, t.handle.gpuResourceID)

"""Adapt the arrays, dropping the host `hwtlas` owner."""
Adapt.adapt_structure(to::Metal.Adaptor, accel::AdaptedAccel) = adaptaccel(to, accel)

# Forced opacity per instance, whatever its geometry was built as; a transparent query
# leaves it to the instances.
@inline instancerecord(::Type{MetalInstanceDescriptor}, transform, id::UInt32, mask::UInt32,
                       index::UInt32, opaque::Bool) =
    MetalInstanceDescriptor(Tuple(transpose(transform)),
                            opaque ? UInt32(MTL.MTLAccelerationStructureInstanceOptionOpaque) :
                                     UInt32(MTL.MTLAccelerationStructureInstanceOptionNonOpaque),
                            mask, UInt32(0), index, id)

# AABB geometry: the query offers each box to the payload's `procedural_candidate`.
supports_procedural_traversal(::MetalDevice) = true
# Every Apple-silicon GPU builds and traverses `MTLAccelerationStructure`s.
supports_hwtlas(::MetalDevice) = true

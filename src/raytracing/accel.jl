"""Traversal data shared by hardware backends. `hwtlas` is the native owner on the
host and `nothing` after device adaptation. `triangles` and per-instance `offsets`
feed [`hittriangle`](@ref), `empty` is its miss value, `scene` an optional in-band
native binding, `procedural` the non-triangle intersection payload, and `instances`
the world transforms. Offsets and instances follow native build order."""
struct AdaptedAccel{H, T, O, Tri, S, P, I} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
    triangles::T
    offsets::O
    empty::Tri
    # In-band traversal binding (Metal); Vulkan binds the TLAS separately.
    scene::S
    # `nothing` removes the procedural branch from the traversal.
    procedural::P
    instances::I
end

"""
    tracescall(accel) -> Bool

Whether a ray query from a kernel through `accel` is a function call already: Metal's
traversal is a linked function its in-band `scene` binding feeds, Vulkan's an inline ray
query, which binds the TLAS out of band (`scene` is `nothing`), and a software BVH is
plain code. A caller outlines its own trace around an inline query to give the query a
register frame of its own; around a call that is a second call boundary, and Hikari's
shadow trace rendered 1.6x slower outlined on an M1 Pro.
"""
tracescall(accel) = false
tracescall(::AdaptedAccel{<:Any, <:Any, <:Any, <:Any, Nothing}) = false
tracescall(::AdaptedAccel) = true

"""Project traversal fields to device values, `scene` through the backend's
`sceneconvert`, in field order. The cyclic native `hwtlas` owner is dropped, never adapted."""
@inline function adaptaccel(to, accel::AdaptedAccel, sceneconvert)
    return AdaptedAccel(nothing,
        Adapt.adapt(to, accel.triangles), Adapt.adapt(to, accel.offsets),
        accel.empty, sceneconvert(accel.scene), Adapt.adapt(to, accel.procedural),
        Adapt.adapt(to, accel.instances))
end

# Holds the exposed leaves, not `hwtlas`: the native owner is cyclic through `static_tlas`.
@inline function holdleaves!(holder, a::AdaptedAccel)
    holdleaves!(holder, a.triangles)
    holdleaves!(holder, a.offsets)
    holdleaves!(holder, a.scene)
    holdleaves!(holder, a.procedural)
    holdleaves!(holder, a.instances)
    return nothing
end

"""
Triangle offset of an instance whose batch has no triangles (a pre-built BLAS
pushed without them). [`hittriangle`](@ref) answers the miss value for it instead
of reading another batch's triangle or past the end of the array.
"""
const NOTRIANGLES = typemax(UInt32)

"""
    hittriangle(accel::AdaptedAccel, inst_id, prim_idx) -> triangle

The hit triangle in world space, looked up by 0-based instance and primitive index
and moved by its instance transform: the BLAS holds object-space triangles, and
shading works on world-space rays. `accel.empty` for a [`NOTRIANGLES`](@ref)
instance; the hit's `t`, barycentrics and custom index are still valid.
"""
@inline function hittriangle(accel::AdaptedAccel, inst_id::UInt32, prim_idx::UInt32)
    @inbounds off = accel.offsets[inst_id + UInt32(1)]
    off == NOTRIANGLES && return accel.empty
    @inbounds tri = accel.triangles[Int(off) + Int(prim_idx) + 1]
    return toworld(accel.instances, inst_id, tri)
end
@inline toworld(::Nothing, inst_id, tri) = tri
@inline toworld(instances, inst_id, tri) =
    @inbounds Raycore.transform(instances[inst_id + UInt32(1)].transform, tri)



"""
    AccelBuildContext(bq, into, preserves)

What a `build_accel!` callback records builds into: the submit channel `bq`, the
backend's emitter `into` over the build command buffer, and `preserves`, objects
that must outlive the build submission.
"""
mutable struct AccelBuildContext{Q,E}
    bq::Q
    into::E
    preserves::Vector{Any}
end

# Procedural geometry. A payload implements `procedural_miss`, `procedural_candidate`
# and `procedural_commit`; `Nothing` implements them as identities, so a
# triangles-only scene compiles to the plain traversal. The shader keeps its own
# best hit because an inline ray query carries no hit attributes for a generated
# intersection.

"""
    procedural_miss(payload)

The initial value of the traversal's running best hit.
"""
@inline procedural_miss(::Nothing) = nothing

"""
    procedural_candidate(payload, best) -> best

Answer the AABB candidate the traversal offers and return the running best. Solve
in object space with [`candidate_object_ray`](@ref), and call
[`commit_intersection!`](@ref) only when `t` improves on `best`, so the shader's
best is the one traversal commits.
"""
@inline procedural_candidate(::Nothing, best) = best

"""
    procedural_commit(payload, best, prim_idx, t, empty) -> primitive

The primitive for a committed generated intersection, built from `best`. Must
return the type of `empty`: anything else makes `closest_hit` return a `Union` in
every integrator, also for scenes without procedural geometry. A triangle record
(vertices, normals, tangents, uv) can describe the local tangent plane of any
parametric surface.
"""
@inline procedural_commit(::Nothing, best, prim_idx, t, empty) = empty

# Candidate verbs: a payload calls them, a backend implements them, so a payload
# never names a backend's ray-query intrinsics.

"""
    candidate_primitive_index() -> Int

Which primitive of the current BLAS the traversal is offering, 1-based.
"""
function candidate_primitive_index end

"""
    candidate_object_ray() -> (o::Vec3f, d::Vec3f)

The candidate's ray in object space, as traversal has it. Transforming the
world-space ray instead applies the instance transform twice.
"""
function candidate_object_ray end

"""
    commit_intersection!(t)

Report a hit at `t` on the current candidate. Call it only for a `t` that improves
on the running best.
"""
function commit_intersection! end

"""
    procedural_bary(payload, best) -> SVector{3,Float32}

Barycentrics of the hit on the primitive `procedural_commit` returned: `(1, 0, 0)`
for one built with the hit at its first vertex. The surface parameter travels in
the primitive's `uv`.
"""
@inline procedural_bary(::Nothing, best) = SVector{3,Float32}(1f0, 0f0, 0f0)

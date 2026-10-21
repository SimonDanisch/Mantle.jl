"""Host/device traversal data shared by hardware backends. Host hwtlas retains the
native owner; device adaptation drops it. Triangles and per-instance offsets feed
hittriangle, empty is its miss sentinel, scene is an optional in-band native
binding, procedural supplies non-triangle intersection data, and instances carry
world transforms. Offsets and instances follow native build order."""
struct AdaptedAccel{H, T, O, Tri, S, P, I} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
    triangles::T
    offsets::O
    empty::Tri
    # In-band traversal binding (Metal); Vulkan binds the TLAS separately.
    scene::S
    # Non-triangle intersection data; nothing removes the procedural branch.
    procedural::P
    instances::I
end

"""Project traversal fields to device values. The backend supplies sceneconvert;
invoke it in field order and never adapt the cyclic native hwtlas owner."""
@inline function adaptaccel(to, accel::AdaptedAccel, sceneconvert)
    return AdaptedAccel(nothing,
        Adapt.adapt(to, accel.triangles), Adapt.adapt(to, accel.offsets),
        accel.empty, sceneconvert(accel.scene), Adapt.adapt(to, accel.procedural),
        Adapt.adapt(to, accel.instances))
end

# Optional traversal fields default to nothing.
AdaptedAccel(hwtlas, triangles, offsets, empty) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, nothing, nothing, nothing)
AdaptedAccel(hwtlas, triangles, offsets, empty, scene) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, scene, nothing, nothing)
AdaptedAccel(hwtlas, triangles, offsets, empty, scene, procedural) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, scene, procedural, nothing)

# Hold exposed resource leaves; the cached native owner forms a cycle via static_tlas.
@inline function holdleaves!(holder, a::AdaptedAccel)
    holdleaves!(holder, a.triangles)
    holdleaves!(holder, a.offsets)
    holdleaves!(holder, a.scene)
    holdleaves!(holder, a.procedural)
    holdleaves!(holder, a.instances)
    return nothing
end

"""
The per-instance triangle offset of an instance whose batch has no triangles: a
pre-built bottom-level structure pushed without them. Its hits have nothing to
look up, and [`hittriangle`](@ref) answers the miss sentinel for them instead of
reading another batch's triangle (or past the end of the array).
"""
const NOTRIANGLES = typemax(UInt32)

"""
    hittriangle(accel::AdaptedAccel, inst_id, prim_idx) -> triangle

The triangle a hit landed on, in world space: looked up in the flat triangle
array by instance and primitive index, then moved by its instance's transform.
The BLAS stores triangles in object space, and everything downstream (normals,
tangents, the side a shadow ray starts on) is computed against world-space rays.

`accel.empty` for an instance of a batch with no triangles ([`NOTRIANGLES`](@ref)):
the hit's distance, barycentrics and custom index are still right, there is just
no triangle record to hand back.
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
    AccelBuildContext(queue)

Scratch state reused across acceleration-structure builds: the queue the builds
are submitted on, and the objects that must outlive the build.

Was `ASBuildContext` in the Vulkan backend and then an abstract type here. It is
one shared struct: both fields are portable, the second trivially and the first
once the submission channel stopped being per-backend. Building is the expensive
part of ray tracing and every backend wants to amortise it, so the context is
API rather than an implementation detail one of them happens to have.
"""
mutable struct AccelBuildContext{Q,E}
    bq::Q
    # Where the builds are written: the backend's emitter over the closed
    # command buffer `build_accel!` opened for them, submitted on the way out.
    into::E
    preserves::Vector{Any}
end
AccelBuildContext(bq, into) = AccelBuildContext(bq, into, Any[])

# ── Procedural geometry ─────────────────────────────────────────────────────
#
# An acceleration structure can hold boxes as well as triangles, and a box is
# only an assertion that something is inside it. What that something IS, and
# where a ray meets it, is the shader's answer — given once here so that both
# the traversal and the geometry that comes out of it stay one code path.
#
# A procedural payload implements three methods. `Nothing` implements them as
# the identity, which is what makes a triangles-only scene compile to exactly
# the traversal that was here before:
#
#   procedural_miss(p)                 the "no hit yet" value the loop carries
#   procedural_candidate(p, best)      answer ONE box the traversal offers
#   procedural_commit(p, best, ...)    turn the winning answer into a primitive
#
# Why the loop carries `best` itself rather than reading it back from the
# query: an inline ray query carries no hit attributes for a GENERATED
# intersection — `get_barycentrics` is triangles only. The shader keeps its own
# best hit, and only ever generating an IMPROVING `t` is what makes that local
# best the one traversal finally commits.

"""
    procedural_miss(payload)

The initial value of the traversal's running best hit.
"""
@inline procedural_miss(::Nothing) = nothing

"""
    procedural_candidate(payload, best)

Answer the AABB candidate the ray query is currently offering, and return the
running best. Solve in OBJECT space and ask the QUERY for the ray
(`lava_ray_query_get_object_ray_origin` / `…_direction`) rather than
transforming one — the query already has it, and an instance transform applied
twice is a bug with no symptom until something is instanced.

Call `lava_ray_query_generate_intersection(t)` only when `t` improves on
`best`; that is what makes the shader's local best the committed one.
"""
@inline procedural_candidate(::Nothing, best) = best

"""
    procedural_commit(payload, best, prim_idx, t, empty) -> primitive

The primitive for a committed generated intersection, built from the best hit
the loop kept.

**It must return the same type as `empty`.** `closest_hit` is called from every
integrator in the renderer, and a payload that returns a type of its own makes
its return a `Union` — including for scenes that have no procedural geometry at
all, since the compiler cannot know the generated branch is unreachable. One
concrete type is what keeps that from leaking into every consumer.

That is not a restriction in practice: a triangle record carries vertices,
normals, tangents and uv, which is enough to describe the exact local tangent
plane of any parametric surface. See Hikari's `fem.jl` for what that looks like.
"""
@inline procedural_commit(::Nothing, best, prim_idx, t, empty) = empty

# The three things answering a candidate needs, spelled portably. A payload
# calls these; a BACKEND implements them. Without them Hikari would have to name
# `lava_ray_query_*` to write an intersection, which is Vulkan leaking through
# two layers and would make Metal a second implementation of the same idea
# rather than a second answer to the same three questions.

"""
    candidate_primitive_index() -> Int

Which primitive of the current BLAS the traversal is offering, 1-based.
"""
function candidate_primitive_index end

"""
    candidate_object_ray() -> (o::Vec3f, d::Vec3f)

The candidate's ray in OBJECT space.

Ask for it rather than transforming the world-space ray: the traversal already
has it, and applying the instance transform a second time is a bug with no
symptom at all until something is instanced.
"""
function candidate_object_ray end

"""
    commit_intersection!(t)

Report a hit at `t` on the current candidate.

Only ever call this for a `t` that IMPROVES on the best so far — that is what
makes the shader's own running best the one the traversal finally commits.
"""
function commit_intersection! end

"""
    procedural_bary(payload, best) -> SVector{3,Float32}

The barycentric coordinate of the hit ON the primitive `procedural_commit`
returned. A primitive built to put the hit at its first vertex reports
`(1, 0, 0)`; the surface's own parameter travels in the primitive's `uv`, which
is where every other primitive reports it too.
"""
@inline procedural_bary(::Nothing, best) = SVector{3,Float32}(1f0, 0f0, 0f0)

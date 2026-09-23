# The CPU-side view of a built acceleration structure.
#
# `AdaptedAccel` (was `HWAdaptedAccel`) reaches its backend's HWTLAS through a
# type parameter, so it never names one — which is why it is here and the
# `HWTLAS` it wraps is only declared.
"""
    AdaptedAccel{H, T, O, Tri, S} <: Raycore.AbstractAdaptedAccel

GPU-adapted form of `VulkanTLAS`.  Carries the kernel-side data needed by
`Raycore.closest_hit(::AdaptedAccel, ray)` / `any_hit` (which lower to
`OpRayQueryInitializeKHR` etc.):

  * `triangles` — flat array of all BLAS triangles, concatenated.
  * `offsets`   — per-instance offset into `triangles`, indexed by `gl_InstanceID + 1`.
  * `empty`     — sentinel triangle returned on miss.
  * `hwtlas`    — CPU-side `VulkanTLAS` reference for callers that need it
                  (descriptor binding, sync, RT pipeline path); `nothing` in
                  the kernel-form produced by `Adapt.adapt`.

Constructed by `sync!(hwtlas)` from the live VulkanTLAS state.  The kernel-form
(returned by `Adapt.adapt(LavaAdaptor, accel)`) drops `hwtlas` (Nothing) and
adapts the array fields to `LavaDeviceArray`.
"""
struct AdaptedAccel{H, T, O, Tri, S, P} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
    triangles::T
    offsets::O
    empty::Tri
    # How the KERNEL reaches the acceleration structure, when the backend needs
    # it in-band rather than bound beside the dispatch.
    #
    # Vulkan binds its TLAS as a descriptor, so a ray query names nothing and
    # this stays `nothing` — a ghost type, so the kernel-side struct is byte-for
    # byte what it was. Metal has no descriptor for an acceleration structure
    # reachable from a Julia kernel: traversal is a linked MSL function that
    # takes the structure through an argument buffer, and this carries the
    # pointer to it.
    scene::S
    # The geometry that is NOT triangles: an AABB per primitive plus a solve
    # that the traversal loop runs when the acceleration structure hands it a
    # box. `nothing` for a triangles-only scene, and then every branch reading
    # it folds away — the traversal is byte-for-byte the one that was here.
    #
    # See `procedural_candidate` / `procedural_commit` for what a value has to
    # implement. It is deliberately not a triangle: a primitive that reports
    # `(t, ξ)` is what lets an isoparametric FEM element be intersected exactly
    # instead of being tessellated first.
    procedural::P
end

# Shorter forms for the backends and call sites that need neither of the
# trailing fields. Keeping them means adding a field did not touch a single
# existing construction site.
AdaptedAccel(hwtlas, triangles, offsets, empty) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, nothing, nothing)
AdaptedAccel(hwtlas, triangles, offsets, empty, scene) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, scene, nothing)



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

"""Traversal data shared by hardware backends. `hwtlas` is the native owner on the
host and `nothing` after device adaptation. `triangles` and per-instance `offsets`
feed [`hittriangle`](@ref), `empty` is its miss value, `scene` the structure as a
kernel value where the backend needs one, `procedural` the non-triangle intersection payload, and `instances`
the world transforms. Offsets and instances follow native build order."""
struct AdaptedAccel{H, T, O, Tri, S, P, I} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
    triangles::T
    offsets::O
    empty::Tri
    # The structure as a kernel value where the backend's query takes one (Metal's
    # `gpuResourceID`); `nothing` where it is bound out of band (Vulkan's descriptor).
    scene::S
    # `nothing` removes the procedural branch from the traversal.
    procedural::P
    instances::I
end

"""Project traversal fields to device values, in field order. The cyclic native
`hwtlas` owner is dropped, never adapted."""
@inline function adaptaccel(to, accel::AdaptedAccel)
    return AdaptedAccel(nothing,
        Adapt.adapt(to, accel.triangles), Adapt.adapt(to, accel.offsets),
        accel.empty, Adapt.adapt(to, accel.scene), Adapt.adapt(to, accel.procedural),
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

# Procedural geometry. A payload implements `procedural_miss`, `procedural_candidate`,
# `procedural_opaque` and `procedural_commit`; `Nothing` implements them as
# identities, so a triangles-only scene compiles to the plain traversal. The shader
# keeps its own best hit because a ray query carries no hit attributes for a
# generated intersection.

"""
    procedural_miss(payload)

The initial value of the traversal's running best hit: a hit whose `t` no hit
beats, as every best hit has its distance in a field `t`.
"""
@inline procedural_miss(::Nothing) = nothing

"""
    procedural_candidate(payload, best, q) -> best

Answer the box candidate query `q` offers: the nearest hit in the box nearer than
`best.t`, or `best`. Solve in object space with
[`candidate_object_ray`](@ref)`(q)`. The traversal commits what improves on `best`
(`commit_intersection!`); the payload commits nothing.
"""
@inline procedural_candidate(::Nothing, best, q) = best

"""
    procedural_opaque(payload) -> Bool

Whether the payload's surfaces block what `Raycore.transparent_hits` traces, or are
recorded, as a triangle of an instance pushed with `opaque = false` is.
"""
@inline procedural_opaque(::Nothing) = true

"""
    procedural_commit(payload, best, prim_idx, t, empty) -> primitive

The primitive for a committed generated intersection, built from `best`. Must
return the type of `empty`: anything else makes `closest_hit` return a `Union` in
every integrator, also for scenes without procedural geometry. A triangle record
(vertices, normals, tangents, uv) can describe the local tangent plane of any
parametric surface.
"""
@inline procedural_commit(::Nothing, best, prim_idx, t, empty) = empty

# ── The ray query ───────────────────────────────────────────────────────────
#
# One traversal for every hardware backend: core walks the query, a backend
# implements the verbs on its own query object (an inline ray query on Vulkan,
# an intersection query on Metal). The verbs are device code.

"""
    query_begin(accel, ray, mask, anyhit) -> q

Start a query of `ray` through `accel` that takes every triangle as opaque. `mask`
is ANDed with each instance's mask; with `anyhit` the traversal ends at the first
committed intersection.
"""
function query_begin end

"""
    query_begin_transparent(accel, ray, mask) -> q

Start a query of `ray` through `accel` in which the triangles of instances that are
not opaque come back as candidates, and which ends at the first committed
intersection. What `Raycore.transparent_hits` traces.
"""
function query_begin_transparent end

"""
    query_next!(q) -> Bool

Advance to the next candidate the traversal cannot decide alone; `false` once it is
done. Opaque triangles never appear: traversal commits them itself.
"""
function query_next! end

"""
    candidate_is_box(q) -> Bool

Whether the current candidate is a box of procedural geometry; else it is a
triangle of an instance that is not opaque.
"""
function candidate_is_box end

"""
    candidate_primitive_index(q) -> Int

Which primitive of the current BLAS the candidate is, 1-based.
"""
function candidate_primitive_index end

"""
    candidate_object_ray(q) -> (o::Vec3f, d::Vec3f)

The candidate's ray in object space, as traversal has it. Transforming the
world-space ray instead applies the instance transform twice.
"""
function candidate_object_ray end

"""
    candidate_ids(q) -> (instance, primitive, custom)

The candidate's 0-based instance index and primitive index, and the instance's
custom index.
"""
function candidate_ids end

"""
    candidate_triangle(q) -> (t, u, v)

The candidate triangle's distance and barycentrics. Not for a box.
"""
function candidate_triangle end

"""
    commit_triangle!(q)

Accept the current triangle candidate as a hit.
"""
function commit_triangle! end

"""
    commit_intersection!(q, t)

Report a hit at `t` on the current box candidate. Called only for a `t` that
improves on the running best.
"""
function commit_intersection! end

"""
    committed_kind(q) -> UInt32

What the query committed: `Raycore.HIT_MISS`, `HIT_TRIANGLE` or `HIT_GENERATED`.
"""
function committed_kind end

"""
    committed_ids(q) -> (t, instance, primitive, custom)

The committed intersection's distance, 0-based instance index and primitive index,
and the instance's custom index.
"""
function committed_ids end

"""
    committed_barycentrics(q) -> (u, v)

The committed triangle's barycentrics. Only for a `HIT_TRIANGLE`.
"""
function committed_barycentrics end

"""
    query_end!(q)

Release what the query holds. After the committed intersection is read.
"""
function query_end! end

# The payload's answer to a box candidate, committed when it is nearer than `best`.
@inline improve!(q, ::Nothing, ::Nothing) = nothing
@inline function improve!(q, best, b)
    b.t < best.t || return best
    commit_intersection!(q, b.t)
    return b
end

"""
    query_record(accel, ray, mask, anyhit) -> Raycore.HitRecord

The traversal every backend runs: procedural boxes answered by the payload, then the
committed intersection as a hit record. A generated hit keeps the payload's best,
which builds its primitive.
"""
@inline function query_record(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer, anyhit::Bool)
    q = query_begin(accel, ray, mask, anyhit)
    # `nothing` for a triangles-only scene, which folds the loop body away.
    best = procedural_miss(accel.procedural)
    while query_next!(q)
        candidate_is_box(q) && (best = improve!(q, best, procedural_candidate(accel.procedural, best, q)))
    end
    kind = committed_kind(q)
    rec = if kind == Raycore.HIT_MISS
        Raycore.HitRecord()
    else
        t, inst, prim, custom = committed_ids(q)
        if kind == Raycore.HIT_GENERATED
            Raycore.HitRecord(t, packbest(best), inst, prim, custom, Raycore.HIT_GENERATED)
        else
            u, v = committed_barycentrics(q)
            Raycore.HitRecord(t, u, v, inst, prim, custom)
        end
    end
    query_end!(q)
    return rec
end

@propagate_inbounds Raycore.closest_hit_record(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer) =
    query_record(accel, ray, mask, false)
@propagate_inbounds Raycore.any_hit_record(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer) =
    query_record(accel, ray, mask, true)

"""
    query_transparent(accel, ray, mask, Val(N), limit) -> Raycore.TransparentHits{N}

`Raycore.transparent_hits` on every backend: each triangle candidate (of an instance
that is not opaque) is recorded with `Raycore.record`, and a procedural hit too
unless the payload is opaque; anything committed (an opaque triangle, an opaque
payload's hit, a hit past `limit`) ends the traversal and blocks the ray.
"""
@inline function query_transparent(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer,
                                   ::Val{N}, limit::Integer) where {N}
    q = query_begin_transparent(accel, ray, mask)
    p = accel.procedural
    th = Raycore.TransparentHits{N}()
    while query_next!(q)
        if candidate_is_box(q)
            miss = procedural_miss(p)
            b = procedural_candidate(p, miss, q)
            if procedural_opaque(p)
                improve!(q, miss, b)
            elseif b.t < miss.t && ray.t_min <= b.t <= ray.t_max
                inst, prim, custom = candidate_ids(q)
                th = Raycore.record(th, Raycore.HitRecord(b.t, packbest(b), inst, prim, custom,
                                                          Raycore.HIT_GENERATED), limit)
                th.blocked && commit_intersection!(q, b.t)
            end
        else
            inst, prim, custom = candidate_ids(q)
            t, u, v = candidate_triangle(q)
            th = Raycore.record(th, Raycore.HitRecord(t, u, v, inst, prim, custom), limit)
            th.blocked && commit_triangle!(q)
        end
    end
    blocked = committed_kind(q) != Raycore.HIT_MISS
    query_end!(q)
    return Raycore.TransparentHits{N}(th.hits, th.count, blocked)
end

@propagate_inbounds Raycore.transparent_hits(accel::AdaptedAccel, ray::Raycore.AbstractRay, mask::Integer,
                                             n::Val, limit::Integer) =
    query_transparent(accel, ray, mask, n, limit)

# A payload's running best hit as four floats: what a `Raycore.HitRecord` keeps of a
# generated hit. Packed per field because `reinterpret` refuses non-primitive
# structs on the GPU.

@inline as_f32(x::Float32) = x
@inline as_f32(x::UInt32)  = reinterpret(Float32, x)
@inline as_f32(x::Int32)   = reinterpret(Float32, x)
@inline from_f32(::Type{Float32}, x::Float32) = x
@inline from_f32(::Type{UInt32},  x::Float32) = reinterpret(UInt32, x)
@inline from_f32(::Type{Int32},   x::Float32) = reinterpret(Int32, x)

"""
    packbest(b) -> NTuple{4,Float32}

A payload's running best hit as four floats.
"""
@generated function packbest(b::B) where {B}
    n = fieldcount(B)
    n <= 4 || error("a procedural payload's best hit must have at most 4 fields " *
                    "to fit a hit record; `$B` has $n")
    vals = [i <= n ? :(as_f32(getfield(b, $i))) : :(0f0) for i in 1:4]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, vals...))
end

"""
    unpackbest(B, v) -> B

The inverse of [`packbest`](@ref).
"""
@generated function unpackbest(::Type{B}, v::NTuple{4,Float32}) where {B}
    args = [:(from_f32($(fieldtype(B, i)), v[$i])) for i in 1:fieldcount(B)]
    return Expr(:block, Expr(:meta, :inline), Expr(:call, B, args...))
end

# A record expands to what it hit: the world-space triangle, or the primitive a
# generated hit's payload builds from its best. `procedural === nothing` folds the
# generated branch to `empty`.
@inline besttype(accel::AdaptedAccel) = typeof(procedural_miss(accel.procedural))

@inline function Raycore.hit_primitive(accel::AdaptedAccel, r::Raycore.HitRecord)
    r.kind == Raycore.HIT_GENERATED &&
        return procedural_commit(accel.procedural, unpackbest(besttype(accel), r.a), r.primitive, r.t, accel.empty)
    return hittriangle(accel, r.instance, r.primitive)
end

@inline function Raycore.hit_barycentric(accel::AdaptedAccel, r::Raycore.HitRecord)
    r.kind == Raycore.HIT_GENERATED &&
        return procedural_bary(accel.procedural, unpackbest(besttype(accel), r.a))
    return SVector{3,Float32}(1f0 - r.a[1] - r.a[2], r.a[1], r.a[2])
end

@inline Raycore.miss_primitive(accel::AdaptedAccel) = accel.empty

"""
    procedural_bary(payload, best) -> SVector{3,Float32}

Barycentrics of the hit on the primitive `procedural_commit` returned: `(1, 0, 0)`
for one built with the hit at its first vertex. The surface parameter travels in
the primitive's `uv`.
"""
@inline procedural_bary(::Nothing, best) = SVector{3,Float32}(1f0, 0f0, 0f0)

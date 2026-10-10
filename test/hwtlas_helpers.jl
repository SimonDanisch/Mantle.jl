# Shared by the HWTLAS tests (`test_hwtlas_*.jl`, `test_holdleaves_stops_at_tlas.jl`).
#
# What an acceleration structure holds is seen the way a renderer sees it: by
# TRACING rays through it with `Raycore.closest_hit` in an ordinary kernel. That
# is the same call on every backend — Vulkan lowers it to an inline ray query,
# Metal to a linked MSL intersector — so a test written against the probe below
# names none, and checks what a caller would notice instead of a backend's
# instance records or buffers.
#
# Each test includes this with an `isdefined` guard so a second include into the
# same module is a no-op. Not called `translation`: Makie exports that name, so in
# a `Main` that had loaded Makie the guard found Makie's and skipped this file,
# and every call went to `Makie.translation(::Transformable)`.

import Adapt, Raycore, Mantle
import KernelAbstractions
import LinearAlgebra
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Point3f, Vec3f
using StaticArrays: SMatrix

"""Translation as a `Mat4f` (SMatrix{4,4,Float32,16})."""
tlastranslation(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)

# One ray per work item: whether it hit, at what distance, the instance's CUSTOM
# index (`instance_id` / `instance_ids` at `push!`), and the centroid of the
# triangle it landed on, in world space. The triangle is looked up through the
# per-instance offset into the scene's flat triangle array and moved by the
# instance's transform, so a wrong offset or a stale transform shows up here.
@inline function probeone!(hits, ts, ids, centroids, origins, dirs, accel, mask, i)
    ray = Raycore.Ray(o = origins[i], d = dirs[i], t_max = 1f30)
    hit, tri, t, _, id = Raycore.closest_hit(accel, ray, mask)
    v = tri.vertices
    hits[i] = hit ? Int32(1) : Int32(0)
    ts[i] = hit ? t : -1f0
    ids[i] = hit ? id : UInt32(0)
    centroids[i] = (v[1] + v[2] + v[3]) / 3f0
end

# Every instance the default mask sees, which is every instance pushed without
# one: the two-argument `closest_hit` is mask `0xff`.
@kernel function tlasprobe!(hits, ts, ids, centroids, @Const(origins), @Const(dirs), accel)
    i = @index(Global)
    probeone!(hits, ts, ids, centroids, origins, dirs, accel, 0xff, i)
end

# The rays traced with cull mask `mask`: only instances sharing a bit with it.
@kernel function tlasmaskprobe!(hits, ts, ids, centroids, @Const(origins), @Const(dirs), accel, mask)
    i = @index(Global)
    probeone!(hits, ts, ids, centroids, origins, dirs, accel, mask, i)
end

"""
    TLASProbe(backend, origins, dirs)

Rays from `origins[i]` along `dirs[i]`, uploaded once and traced as often as
asked. The outputs are allocated here too, so a loop that traces every
iteration does not allocate per trace — which would be noise in the tests that
bound memory growth.
"""
struct TLASProbe{B,O,D,H,T,I,C}
    backend::B
    origins::O
    dirs::D
    hits::H
    ts::T
    ids::I
    centroids::C
end

function TLASProbe(be, origins::AbstractVector, dirs::AbstractVector)
    n = length(origins)
    length(dirs) == n || throw(ArgumentError("$(length(dirs)) directions for $n origins"))
    TLASProbe(be,
              Mantle.devicearray(be, Point3f.(origins)),
              Mantle.devicearray(be, Vec3f.(dirs)),
              KernelAbstractions.zeros(be, Int32, n),
              KernelAbstractions.zeros(be, Float32, n),
              KernelAbstractions.zeros(be, UInt32, n),
              KernelAbstractions.allocate(be, Point3f, n))
end

Base.length(p::TLASProbe) = length(p.hits)

"""The adapted form a kernel takes: `Adapt.adapt` syncs an HWTLAS and hands back
the `AdaptedAccel` it owns; an `AdaptedAccel` is traced as it is."""
adapted(be, t::Mantle.HWTLAS) = Adapt.adapt(be, t)
adapted(be, a::Mantle.AdaptedAccel) = a

"""
    tlastrace!(probe, accel; mask = nothing) -> probe

Launch the probe against `accel` (an HWTLAS or an `AdaptedAccel`) and return
WITHOUT waiting for it. [`tlasresult`](@ref) waits and reads. `mask` is the
rays' cull mask; `nothing` traces with the two-argument `closest_hit`.
"""
function tlastrace!(p::TLASProbe, accel; mask = nothing)
    if mask === nothing
        tlasprobe!(p.backend)(p.hits, p.ts, p.ids, p.centroids, p.origins, p.dirs,
                              adapted(p.backend, accel); ndrange = length(p))
    else
        tlasmaskprobe!(p.backend)(p.hits, p.ts, p.ids, p.centroids, p.origins, p.dirs,
                                  adapted(p.backend, accel), UInt32(mask); ndrange = length(p))
    end
    return p
end

"""
    tlasresult(probe) -> (hit, t, id, centroid)

Wait for the backend and read the last trace back: `hit::Vector{Bool}`, the
distance (`-1` for a miss), the custom index (`0` for a miss) and the hit
triangle's world-space centroid.
"""
function tlasresult(p::TLASProbe)
    KernelAbstractions.synchronize(p.backend)
    return (hit = Array(p.hits) .== Int32(1), t = Array(p.ts), id = Array(p.ids),
            centroid = Array(p.centroids))
end

"""Trace and read back, in one."""
tlastrace(p::TLASProbe, accel; mask = nothing) = tlasresult(tlastrace!(p, accel; mask))
tlastrace(be, accel, origins::AbstractVector, dirs::AbstractVector; mask = nothing) =
    tlastrace(TLASProbe(be, origins, dirs), accel; mask)

# ── The host reference ───────────────────────────────────────────────────────

"""
    nearesthit(o, d, triangles) -> (hit, t, k)

What a trace from `o` along `d` should report against `triangles`, a list of
world-space vertex triples: the nearest one it meets (Möller–Trumbore), the
distance to it, and its index into `triangles`; `(false, 0f0, 0)` for a miss.
The probe and this share nothing past the ray, so they cannot agree by accident.

Rays are meant to be aimed well inside a triangle: on an edge the reference and
a watertight hardware traversal may legitimately decide differently.
"""
function nearesthit(o, d, triangles)
    best_t, best_k = Inf32, 0
    dir = Vec3f(d)
    for (k, (a, b, c)) in enumerate(triangles)
        e1, e2 = Vec3f(b - a), Vec3f(c - a)
        p = LinearAlgebra.cross(dir, e2)
        det = LinearAlgebra.dot(e1, p)
        abs(det) < 1f-12 && continue
        s = Vec3f(o - a)
        q = LinearAlgebra.cross(s, e1)
        u = LinearAlgebra.dot(s, p) / det
        v = LinearAlgebra.dot(dir, q) / det
        t = LinearAlgebra.dot(e2, q) / det
        (u >= 0f0 && v >= 0f0 && u + v <= 1f0 && 0f0 < t < best_t) || continue
        best_t, best_k = t, k
    end
    return best_k == 0 ? (false, 0f0, 0) : (true, best_t, best_k)
end

"""A vertex triple moved by `offset`."""
movedby(tri, offset) = map(v -> Point3f(v + Vec3f(offset...)), tri)

"""The centroid of a vertex triple, to compare with the probe's."""
tricentroid(tri) = (tri[1] + tri[2] + tri[3]) / 3f0

"""
Point location against a procedural (AABB) bottom-level structure: which of 1000
overlapping boxes contains a point, answered by a zero-length ray.

Setup: 1000 random boxes of side 1 in [-10, 10]^3 and 64 random query points. A
ray of length zero starts at each point; the hardware offers every box the point
might be in as a candidate, and the candidate answer (`procedural_candidate`
below) checks the point against that box's own bounds and commits `t = 0` only
when it is inside. So a ray hits exactly when the point is inside some box, and
the box it reports must contain it. The host answers the same question by brute
force.

The candidate's own bounds check is the point of the test. A driver may offer a
box whose bounds do not strictly contain the point (degenerate zero-length ray
edge cases); counting candidates instead of checking them would count those.

This used to count every containing box per point, by walking the candidates of
an inline ray query by hand. The portable procedural protocol answers the
CLOSEST hit, and Metal's traversal keeps a candidate's answer only when its `t`
improves, so the count is not observable through it. What is kept is what the
count stood for: every box containing the point is offered (a point inside only
one box misses if that box is skipped), and the primitive index of a committed
box is the right one.
"""

using Test, KernelAbstractions, Random
import Raycore, Adapt
using GeometryBasics: Point3f, Vec3f
using StaticArrays: SVector
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""The payload: the boxes' bounds, one array per coordinate, indexed by primitive."""
struct BoxElems{A}
    lo_x::A; lo_y::A; lo_z::A
    hi_x::A; hi_y::A; hi_z::A
end

Adapt.adapt_structure(to, e::BoxElems) = BoxElems(
    Adapt.adapt(to, e.lo_x), Adapt.adapt(to, e.lo_y), Adapt.adapt(to, e.lo_z),
    Adapt.adapt(to, e.hi_x), Adapt.adapt(to, e.hi_y), Adapt.adapt(to, e.hi_z))

"""The running best: the committed `t` and the 1-based box it is on."""
struct BoxHit
    t::Float32
    cell::UInt32
end

Mantle.procedural_miss(::BoxElems) = BoxHit(Inf32, UInt32(0))

# Object space is world space here (one instance, identity transform). Commit only
# for a `t` that improves on `best`, as the protocol requires: the first box that
# contains the point commits 0, and nothing improves on that.
function Mantle.procedural_candidate(e::BoxElems, best::BoxHit)
    cell = Mantle.candidate_primitive_index()
    o, _ = Mantle.candidate_object_ray()
    @inbounds inside = e.lo_x[cell] <= o[1] <= e.hi_x[cell] &&
                       e.lo_y[cell] <= o[2] <= e.hi_y[cell] &&
                       e.lo_z[cell] <= o[3] <= e.hi_z[cell]
    if inside && 0f0 < best.t
        Mantle.commit_intersection!(0f0)
        # `cell % UInt32`, not `UInt32(cell)`: the checked conversion's error path
        # needs kernel state, which a Metal visible function does not have.
        return BoxHit(0f0, cell % UInt32)
    end
    return best
end

Mantle.procedural_commit(::BoxElems, ::BoxHit, prim, t, empty) = empty
# The box rides out in the barycentric's first slot.
Mantle.procedural_bary(::BoxElems, b::BoxHit) = SVector{3,Float32}(Float32(b.cell), 0f0, 0f0)

@kernel function aabb_locate!(hits, cells, ts, @Const(qs), accel)
    i = @index(Global)
    @inbounds q = qs[i]
    r = Raycore.Ray(o = q, d = Vec3f(1, 0, 0), t_min = 0f0, t_max = 0f0)
    hit, _prim, t, bary, _inst = Raycore.closest_hit(accel, r)
    @inbounds hits[i] = hit ? Int32(1) : Int32(0)
    @inbounds cells[i] = hit ? unsafe_trunc(UInt32, bary[1]) : UInt32(0)
    @inbounds ts[i] = hit ? t : -1f0
end

in_aabb(p, a::Mantle.AABB) = all(a.min .<= p .<= a.max)

@testset "AABB BLAS: a zero-length ray locates a point among overlapping boxes" begin
    if !(Mantle.supports_hwtlas(TESTBACKEND) && Mantle.supports_procedural_traversal(TESTBACKEND))
        @info "no procedural hardware traversal on this backend; skipping"
        @test_skip false
        return
    end
    dev = Mantle.Device(TESTBACKEND)

    Random.seed!(42)
    n_aabbs = 1000
    aabbs = Mantle.AABB[]
    for _ in 1:n_aabbs
        c = Point3f(rand(Float32) * 20f0 - 10f0,
                    rand(Float32) * 20f0 - 10f0,
                    rand(Float32) * 20f0 - 10f0)
        h = 0.5f0
        push!(aabbs, Mantle.AABB(c .- h, c .+ h))
    end
    n_q = 64
    qx = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]
    qy = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]
    qz = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]
    qs = [Point3f(qx[i], qy[i], qz[i]) for i in 1:n_q]

    blas = Mantle.build_accel!(Mantle.batchqueue(dev)) do ctx
        Mantle.build_blas_aabb(ctx, aabbs)
    end
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(hw, blas; instance_id = UInt32(0))
    # Before the first `sync!`: the adapted accel is built with the payload in it.
    coord(f) = Mantle.devicearray(TESTBACKEND, Float32[f(a) for a in aabbs])
    hw.procedural = BoxElems(coord(a -> a.min[1]), coord(a -> a.min[2]), coord(a -> a.min[3]),
                             coord(a -> a.max[1]), coord(a -> a.max[2]), coord(a -> a.max[3]))
    Raycore.sync!(hw)
    accel = Adapt.adapt(TESTBACKEND, hw)

    d_q = Mantle.devicearray(TESTBACKEND, qs)
    d_h = Mantle.devicearray(TESTBACKEND, zeros(Int32, n_q))
    d_c = Mantle.devicearray(TESTBACKEND, zeros(UInt32, n_q))
    d_t = Mantle.devicearray(TESTBACKEND, fill(-2f0, n_q))
    aabb_locate!(TESTBACKEND)(d_h, d_c, d_t, d_q, accel; ndrange = n_q)
    KA.synchronize(TESTBACKEND)
    gh, gc, gt = Array(d_h), Array(d_c), Array(d_t)

    inside = [any(a -> in_aabb(q, a), aabbs) for q in qs]
    # Both outcomes occur, or the assertions below are vacuous for one of them.
    @test count(inside) > 0
    @test count(.!inside) > 0

    @test gh == Int32.(inside)
    for i in findall(inside)
        @test 1 <= gc[i] <= n_aabbs && in_aabb(qs[i], aabbs[gc[i]])
        @test gt[i] == 0f0
    end
end

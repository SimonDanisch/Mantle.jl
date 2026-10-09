"""
What a hardware hit REPORTS beyond "hit, at t": where on the triangle, which
instance, what bounds the search, and that every ray of a batch gets its own
answer. `Raycore.closest_hit` from a KernelAbstractions kernel on
`Adapt.adapt(backend, HWTLAS)`, gated on `Mantle.supports_hwtlas`.

These were Metal's tests of `trace_closest_hits!`, the MSL traversal kernel that
backend carries because `metal::raytracing::intersector<>` is a C++ template with
no AIR symbol to call. The questions they asked are not Metal's, so they are
asked here of the portable path on every backend. Hit or miss against a host
reference, the nearest of two, `t_max`, and agreement with the software BVH are
`test_trace_closest_hit.jl`; custom instance indices through `push!` and a refit
are `test_trace_hwtlas.jl`.
"""

using Test, Mantle, Raycore, GeometryBasics, KernelAbstractions
using Adapt
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function ray_probe!(hits, ts, us, vs, insts, @Const(origins), @Const(dirs),
                            @Const(tmins), @Const(tmaxs), accel)
    i = @index(Global)
    ray = Raycore.Ray(o = origins[i], d = dirs[i], t_min = tmins[i], t_max = tmaxs[i])
    hit, tri, t, bary, inst = Raycore.closest_hit(accel, ray)
    @inbounds hits[i]  = hit ? Int32(1) : Int32(0)
    @inbounds ts[i]    = hit ? t : -1f0
    @inbounds us[i]    = hit ? bary[2] : -1f0
    @inbounds vs[i]    = hit ? bary[3] : -1f0
    @inbounds insts[i] = inst
end

"""Trace one ray per origin; a NamedTuple of host arrays back."""
function trace(be, accel, origins::Vector{Point3f}, dirs::Vector{Vec3f};
               tmin = zeros(Float32, length(origins)), tmax = fill(1f30, length(origins)))
    n = length(origins)
    o = Mantle.devicearray(be, origins); d = Mantle.devicearray(be, dirs)
    lo = Mantle.devicearray(be, tmin); hi = Mantle.devicearray(be, tmax)
    hits = KA.zeros(be, Int32, n); ts = KA.zeros(be, Float32, n)
    us = KA.zeros(be, Float32, n); vs = KA.zeros(be, Float32, n)
    insts = KA.zeros(be, UInt32, n)
    ray_probe!(be)(hits, ts, us, vs, insts, o, d, lo, hi, accel; ndrange = n)
    KA.synchronize(be)
    return (hit = Array(hits), t = Array(ts), u = Array(us), v = Array(vs), inst = Array(insts))
end

# One triangle, (0,0,z)-(1,0,z)-(0,1,z): for a point (x, y) on it the
# barycentrics of the second and third vertex ARE x and y.
unit_triangle(z) = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
    Point3f[Point3f(0, 0, z), Point3f(1, 0, z), Point3f(0, 1, z)], [GLTriangleFace(1, 2, 3)]))

function one_triangle(be; z = 0f0, transform = Mat4f(LinearAlgebra.I))
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(t, unit_triangle(z), transform)
    Raycore.sync!(t)
    return t
end

@testset "what a hardware hit reports" begin
    be = TESTBACKEND
    if !Mantle.supports_hwtlas(be)
        @info "no hardware acceleration structure on this backend; skipping"
        @test_skip Mantle.supports_hwtlas(be)
        return
    end
    down = Vec3f(0, 0, -1)

    @testset "the barycentrics identify the point" begin
        t = one_triangle(be)
        # Three inside, three outside. (0.9, 0.9) is outside because 0.9+0.9 > 1 —
        # a bounding-box test would wrongly report it as a hit.
        inside  = [(0.2f0, 0.2f0), (0.1f0, 0.1f0), (0.3f0, 0.3f0)]
        outside = [(2f0, -2f0), (-1f0, 1f0), (0.9f0, 0.9f0)]
        pts = vcat(inside, outside)
        r = trace(be, Adapt.adapt(be, t), [Point3f(x, y, 5) for (x, y) in pts],
                  fill(down, length(pts)))
        for i in 1:3
            @test r.hit[i] == 1
            @test r.t[i] ≈ 5f0 atol = 1e-5
            # For this triangle the barycentrics are the xy of the point.
            @test r.u[i] ≈ pts[i][1] atol = 1e-5
            @test r.v[i] ≈ pts[i][2] atol = 1e-5
        end
        @test r.hit[4:6] == Int32[0, 0, 0]
    end

    @testset "the nearest hit reports its own instance" begin
        # Two parallel triangles, each with a custom index. A traversal that
        # returns any hit rather than the closest passes every test with a single
        # triangle in the scene, and one that kept the near distance but the far
        # record would report the wrong instance.
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(t, unit_triangle(0f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(10))  # far
        push!(t, unit_triangle(1f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(11))  # near
        Raycore.sync!(t)
        accel = Adapt.adapt(be, t)

        r = trace(be, accel, [Point3f(0.25, 0.25, 5)], [down])
        @test r.hit[1] == 1
        @test r.t[1] ≈ 4f0 atol = 1e-5         # the z=1 triangle, not the z=0 one
        @test r.inst[1] == UInt32(11)

        # Shooting the other way finds the far one first.
        r2 = trace(be, accel, [Point3f(0.25, 0.25, -5)], [Vec3f(0, 0, 1)])
        @test r2.t[1] ≈ 5f0 atol = 1e-5
        @test r2.inst[1] == UInt32(10)
    end

    @testset "tmin and tmax bound the search" begin
        t = one_triangle(be)
        # The triangle is at t = 5 along this ray.
        o = fill(Point3f(0.25, 0.25, 5), 3)
        r = trace(be, Adapt.adapt(be, t), o, fill(down, 3);
                  tmin = Float32[0, 6, 0], tmax = Float32[4, 1f30, 1f30])
        @test r.hit[1] == 0      # stops short
        @test r.hit[2] == 0      # starts past
        @test r.hit[3] == 1      # unbounded
    end

    @testset "a batch big enough to span many workgroups" begin
        # A count that is not a multiple of any workgroup size, so the tail is a
        # partial group and whatever guards it is what keeps it from writing past
        # the end. 10_007 is prime. Alternate hit/miss so a kernel that wrote a
        # constant would be caught.
        t = one_triangle(be)
        n = 10_007
        o = [iseven(i) ? Point3f(0.25, 0.25, 5) : Point3f(9, 9, 5) for i in 1:n]
        r = trace(be, Adapt.adapt(be, t), o, fill(down, n))
        @test count(==(1), r.hit) == count(iseven, 1:n)
        @test all(r.hit[i] == 1 for i in 2:2:n)
        @test all(r.hit[i] == 0 for i in 1:2:n)
    end

    @testset "an instance is traced where its transform puts it" begin
        # `Mat3x4f` (three rows of four) and Metal's packed 4x3 (four columns of
        # three) are both twelve `Float32` and TRANSPOSES of each other:
        # reinterpreting one as the other type-checks, runs, and silently shears
        # every instance, which nothing downstream would report — geometry would
        # just be in the wrong place. The translation column is the part a sheared
        # matrix gets subtly wrong rather than obviously.
        t = one_triangle(be; transform = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, 7,8,9,1))
        accel = Adapt.adapt(be, t)
        r = trace(be, accel, [Point3f(7.25, 8.25, 14), Point3f(0.25, 0.25, 5)], fill(down, 2))
        @test r.hit == Int32[1, 0]              # where it was moved to, not where it was
        @test r.t[1] ≈ 5f0 atol = 1e-4
        @test r.u[1] ≈ 0.25f0 atol = 1e-4
        @test r.v[1] ≈ 0.25f0 atol = 1e-4
    end
end

"""
`Raycore.closest_hit` and `Raycore.any_hit` on a hardware acceleration structure,
called from an ordinary kernel.

This is how Hikari traces with `hw_accel = true` on every backend: build a
`Mantle.HWTLAS`, adapt it to the backend, and call the same `closest_hit(accel, ray)`
its integrators call on the software BVH. Multiple dispatch picks the traversal
(an inline ray query on Vulkan, a linked intersector on Metal), so the contract
pinned here is that a kernel written once gets the same answers from either.

Three scenes, merged from five Vulkan files that each tested a step of the
migration to this API:

  * One triangle against a host Möller–Trumbore reference. This is the scene
    `mwe_hw_rt_blank.jl` reduced the Windows AMDVLK blank render to: with
    `hw_accel = true` RayDemo's Crown and bunny_cloud scenes came back empty,
    every ray a miss. A traversal that misses everything passes any test that
    only looks at misses, so both outcomes are asserted.
  * Two triangles, near and far, traced with two `t_max`: `closest_hit` must
    report the near one, and `any_hit` may report either until `t_max` excludes
    the far one.
  * The same kernel on the hardware structure and on the software BVH: hit flags
    and distances must agree ray by ray. That is what lets Hikari's integrators
    swap one for the other.
"""

using Test, KernelAbstractions
import Raycore, Adapt
import LinearAlgebra
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function ch_probe!(ts, hits, @Const(origins), accel, tmax)
    i = @index(Global)
    @inbounds o = origins[i]
    r = Raycore.Ray(o = o, d = Vec3f(0, 0, 1), t_min = 0f0, t_max = tmax)
    hit, _prim, t, _bary, _inst = Raycore.closest_hit(accel, r)
    @inbounds ts[i] = hit ? t : -1f0
    @inbounds hits[i] = hit ? UInt32(1) : UInt32(0)
end

@kernel function ch_closest_any!(cls, anys, @Const(origins), accel, tmax)
    i = @index(Global)
    @inbounds o = origins[i]
    r = Raycore.Ray(o = o, d = Vec3f(0, 0, 1), t_min = 0f0, t_max = tmax)
    ch, _cp, ct, _cb, _ci = Raycore.closest_hit(accel, r)
    ah, _ap, at, _ab, _ai = Raycore.any_hit(accel, r)
    @inbounds cls[i] = ch ? ct : -1f0
    @inbounds anys[i] = ah ? at : -1f0
end

"""A mesh of the triangles `(v1, v2, v3)` listed flat in `verts`."""
function trimesh(verts::Vector{Point3f})
    faces = [GLTriangleFace(k, k + 1, k + 2) for k in 1:3:length(verts)]
    GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
end

"""The 8x8 grid of ray origins at z = 0, 0.25 apart, centred on the origin."""
gridorigins() = [Point3f((i % 8 - 3.5f0) * 0.25f0, (i ÷ 8 - 3.5f0) * 0.25f0, 0f0) for i in 0:63]

"""The hit distance of a ray against one triangle, or -1 for a miss."""
function moller_trumbore(o, d, v0, v1, v2)
    e1 = v1 - v0
    e2 = v2 - v0
    h = LinearAlgebra.cross(d, e2)
    a = LinearAlgebra.dot(e1, h)
    abs(a) < 1f-7 && return -1f0
    f = 1f0 / a
    s = o - v0
    u = f * LinearAlgebra.dot(s, h)
    (u < 0f0 || u > 1f0) && return -1f0
    q = LinearAlgebra.cross(s, e1)
    v = f * LinearAlgebra.dot(d, q)
    (v < 0f0 || u + v > 1f0) && return -1f0
    t = f * LinearAlgebra.dot(e2, q)
    return t < 0f0 ? -1f0 : t
end

"""Run `ch_probe!` over `origins`; the hit distances and the hit flags."""
function probe(accel, origins; tmax = 1f4)
    n = length(origins)
    d_o = Mantle.devicearray(TESTBACKEND, origins)
    d_t = Mantle.devicearray(TESTBACKEND, fill(-2f0, n))
    d_h = Mantle.devicearray(TESTBACKEND, fill(UInt32(99), n))
    ch_probe!(TESTBACKEND)(d_t, d_h, d_o, accel, tmax; ndrange = n)
    KA.synchronize(TESTBACKEND)
    return Array(d_t), Array(d_h)
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "closest_hit on one triangle agrees with Möller–Trumbore" begin
    v0, v1, v2 = Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5)
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(hw, trimesh([v0, v1, v2]))
    Raycore.sync!(hw)

    origins = gridorigins()
    ref = [moller_trumbore(o, Vec3f(0, 0, 1), v0, v1, v2) for o in origins]
    miss = ref .< 0f0
    @test count(miss) > 0
    @test count(.!miss) > 0

    ts, hits = probe(Adapt.adapt(TESTBACKEND, hw), origins)
    @test all(==(UInt32(0)), hits[miss])
    @test all(<(0f0), ts[miss])
    @test all(==(UInt32(1)), hits[.!miss])
    @test maximum(abs.(ts[.!miss] .- ref[.!miss])) < 1f-3
end

@testset "closest_hit and any_hit honour t_max" begin
    # Two triangles with the same footprint, at z = 2 and z = 5.
    tri(z) = [Point3f(-1, -1, z), Point3f(1, -1, z), Point3f(0, 1, z)]
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(hw, trimesh(tri(2f0)))
    push!(hw, trimesh(tri(5f0)))
    Raycore.sync!(hw)

    accel = Adapt.adapt(TESTBACKEND, hw)
    # The adapted form is core's type and still knows its structure, which is
    # what `supports_rt_pipeline(::AdaptedAccel)` and Hikari's trace pass read.
    @test accel isa Mantle.AdaptedAccel
    @test accel.hwtlas === hw

    origins = gridorigins()
    n = length(origins)
    expected = [moller_trumbore(o, Vec3f(0, 0, 1), tri(2f0)...) >= 0f0 for o in origins]
    @test count(expected) > 0
    @test count(expected) < n

    function trace(tmax)
        d_o = Mantle.devicearray(TESTBACKEND, origins)
        d_c = Mantle.devicearray(TESTBACKEND, fill(-2f0, n))
        d_a = Mantle.devicearray(TESTBACKEND, fill(-2f0, n))
        ch_closest_any!(TESTBACKEND)(d_c, d_a, d_o, accel, tmax; ndrange = n)
        KA.synchronize(TESTBACKEND)
        return Array(d_c), Array(d_a)
    end

    # t_max = 10: both triangles in range. closest_hit reports the near one;
    # any_hit may commit either, which order is the implementation's.
    cls, anys = trace(10f0)
    @test all(i -> expected[i] ? isapprox(cls[i], 2f0; atol = 1f-3) : cls[i] < 0f0, 1:n)
    @test all(i -> expected[i] ? anys[i] >= 0f0 : anys[i] < 0f0, 1:n)

    # t_max = 3: only the near triangle is in range, so both report it.
    cls, anys = trace(3f0)
    @test all(i -> expected[i] ? isapprox(cls[i], 2f0; atol = 1f-3) : cls[i] < 0f0, 1:n)
    @test all(i -> expected[i] ? isapprox(anys[i], 2f0; atol = 1f-3) : anys[i] < 0f0, 1:n)
end

@testset "one kernel, hardware structure and software BVH agree" begin
    # One mesh, two faces at z = 5 and z = 8.
    mesh = trimesh([Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5),
                    Point3f(-1, -1, 8), Point3f(1, -1, 8), Point3f(0, 1, 8)])
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(hw, mesh)
    Raycore.sync!(hw)
    sw = Raycore.TLAS(TESTBACKEND)
    push!(sw, mesh)
    Raycore.sync!(sw)

    origins = gridorigins()
    t_hw, h_hw = probe(Adapt.adapt(TESTBACKEND, hw), origins)
    t_sw, h_sw = probe(Adapt.adapt(TESTBACKEND, sw), origins)
    # Something is hit, or the agreement is vacuous.
    @test count(==(UInt32(1)), h_sw) > 0
    @test h_hw == h_sw
    @test all(isapprox.(t_hw, t_sw; atol = 1f-3))
end

else
    @info "no hardware acceleration structure on this backend; skipping"
end

"""
The hardware acceleration structure, and the traversal a Julia kernel does
against it: `Mantle.HWTLAS{Tri}(backend)` built by `push!`/`sync!`, adapted with
`Adapt.adapt(backend, tlas)`, and traced with `Raycore.closest_hit` inside an
ordinary KernelAbstractions kernel. Gated on `Mantle.supports_hwtlas`.

Written against Metal first, where three things in that chain fail silently
rather than loudly, and each has an assertion below that every backend has to
pass:

  * residency — on Metal a TLAS bound through an argument buffer is not
    automatically resident, and a non-resident structure reports every ray as a
    miss. `test_trace_closest_hit.jl` traces one kernel against the hardware
    structure and the software BVH and asserts they agree ray by ray, which is
    where that shows; every testset here that expects a hit catches it too;
  * the hit record crossing from the traversal back into Julia is an ABI that no
    compiler checks (the distances and the instance index below come through it);
  * the fifth return value is the instance CUSTOM index, not the instance's
    position. Returning the latter type-checks, runs, and quietly gives every
    instance past the first the wrong medium interface in Hikari.
"""

using Test, Mantle, Raycore, GeometryBasics, KernelAbstractions
using Adapt
# QUALIFIED below, not `using LinearAlgebra`: `normalize` would collide with an
# earlier binding of the same name in whatever module included this first.
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function hw_probe!(hits, ts, insts, @Const(dirs), accel, O)
    i = @index(Global)
    hit, tri, t, bary, inst = Raycore.closest_hit(accel, Raycore.Ray(o = O, d = dirs[i], t_max = 1f30))
    @inbounds hits[i]  = hit ? Int32(1) : Int32(0)
    @inbounds ts[i]    = hit ? t : -1f0
    @inbounds insts[i] = inst
end

# Deterministic directions from `O` towards the unit cube around the origin.
function probe_dirs(n, O)
    s = Ref(987)
    nr() = (s[] = (1103515245 * s[] + 12345) % 2147483648; Float32(s[]) / 2147483648f0)
    [Vec3f(LinearAlgebra.normalize(Point3f(2*(nr()-0.5f0), 2*(nr()-0.5f0), 0) - O)) for _ in 1:n]
end

"""Trace `dirs` from `O` against `accel` with `hw_probe!`; host arrays back."""
function probe(be, accel, dirs, O)
    n = length(dirs)
    d_d = Mantle.devicearray(be, dirs)
    h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
    hw_probe!(be)(h_d, t_d, i_d, d_d, accel, O; ndrange = n)
    KA.synchronize(be)
    return Array(h_d), Array(t_d), Array(i_d)
end

# One triangle, (0,0,z)-(1,0,z)-(0,1,z).
unit_triangle(z) = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
    Point3f[Point3f(0, 0, z), Point3f(1, 0, z), Point3f(0, 1, z)], [GLTriangleFace(1, 2, 3)]))

# Column-major translate along x.
xshift(x) = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, x,0,0,1)

@testset "hardware acceleration structure" begin
    be = TESTBACKEND
    if !Mantle.supports_hwtlas(be)
        @info "no hardware acceleration structure on this backend; skipping"
        @test_skip Mantle.supports_hwtlas(be)
        return
    end
    dev = Mantle.Device(be)

    @testset "HWTLAS is reachable through the portable constructor" begin
        # `HWTLAS{Tri}(backend)` is what Hikari's `default_accel` calls; a caller
        # must never name the backend's own type.
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        @test t isa Mantle.HWTLAS{Raycore.Triangle{UInt32}}
        @test Raycore.n_geometries(t) == 0
        @test Raycore.n_instances(t) == 0

        mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
        h = push!(t, mesh)
        @test h isa Raycore.TLASHandle
        @test Raycore.n_geometries(t) == 1

        Raycore.sync!(t)
        @test Raycore.n_instances(t) == 1
        b = Raycore.world_bound(t)
        @test all(b.p_min .< -0.9f0) && all(b.p_max .> 0.9f0)

        # `sync!` is the sole owner of the adapted form, and a second one with
        # nothing dirty must not rebuild.
        a1 = Adapt.adapt(be, t)
        a2 = Adapt.adapt(be, t)
        @test a1 === a2
    end

    @testset "several instances, and the custom index stays 0" begin
        # Two instances of one mesh, offset along x. A traversal that returned
        # the instance's POSITION here would give 1 for the second instance, and
        # Hikari would look up a medium interface that does not exist.
        mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0))
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(t, mesh, Mat4f(LinearAlgebra.I))
        push!(t, mesh, xshift(1.5f0))
        Raycore.sync!(t)
        @test Raycore.n_geometries(t) == 2
        @test Raycore.n_instances(t) == 2

        # One ray at each sphere.
        O = Point3f(0, 0, 4)
        dirs = [Vec3f(0, 0, -1), Vec3f(LinearAlgebra.normalize(Point3f(1.5f0, 0, 0) - O))]
        hits, _, ids = probe(be, Adapt.adapt(be, t), dirs, O)
        @test hits == Int32[1, 1]                # both instances are hit
        @test all(==(UInt32(0)), ids)            # …and neither reports a custom index
    end

    @testset "a custom index set by push! comes back from the hit" begin
        # What `meshscatter!` depends on. Hikari bakes every face of the marker
        # mesh with medium interface 0 ("inherit") and hands each instance's
        # material over as `instance_ids`; `resolve_mi_idx` reads it back as the
        # fifth value of `closest_hit`. Until Metal's user-ID instance descriptor
        # was wired up, `push!` dropped the ids and the traversal returned 0, so
        # every sphere inherited interface 0 and rendered BLACK — on the default
        # `hw_accel = true` path, while the software TLAS drew them correctly.
        #
        # Both spellings: one instance with `instance_id` and a batch with
        # `instance_ids`. That the ids survive a refit, which rewrites every
        # instance record in place, is `test_hwtlas_mesh_update.jl`.
        mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0))
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(t, mesh, xshift(0f0); instance_id = UInt32(7))
        push!(t, mesh, [xshift(1.5f0), xshift(3f0)]; instance_ids = UInt32[11, 42])
        Raycore.sync!(t)
        @test Raycore.n_instances(t) == 3

        O = Point3f(0, 0, 4)
        at(x) = Vec3f(LinearAlgebra.normalize(Point3f(x, 0, 0) - O))
        hits, _, ids = probe(be, Adapt.adapt(be, t), [at(0f0), at(1.5f0), at(3f0)], O)
        @test hits == Int32[1, 1, 1]
        @test ids == UInt32[7, 11, 42]

        # A length mismatch is refused before anything is added.
        ng = Raycore.n_geometries(t)
        @test_throws ArgumentError push!(t, mesh, [xshift(9f0)]; instance_ids = UInt32[1, 2])
        @test Raycore.n_geometries(t) == ng
    end

    @testset "a refit moves even a one-triangle structure" begin
        # Metal reports a refit scratch size of ZERO for a structure this small,
        # even when it was built to be refitted. Gating the refit on "scratch > 0"
        # refused a perfectly legal refit, which is what the first version of
        # that backend did. Pinned from the outside: the triangle is traced where
        # it was moved to, and no longer where it was.
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        h = push!(t, unit_triangle(0f0), [Mat4f(LinearAlgebra.I)])
        Raycore.sync!(t)
        O = Point3f(0.25f0, 0.25f0, 5f0)
        down = [Vec3f(0, 0, -1)]
        hits, ts, _ = probe(be, Adapt.adapt(be, t), down, O)
        @test hits == Int32[1]
        @test ts[1] ≈ 5f0 atol = 1e-4

        Raycore.update_transforms!(t, h, [xshift(2f0)])
        Raycore.sync!(t)
        @test probe(be, Adapt.adapt(be, t), down, O)[1] == Int32[0]
        moved, mts, _ = probe(be, Adapt.adapt(be, t), down, Point3f(2.25f0, 0.25f0, 5f0))
        @test moved == Int32[1]
        @test mts[1] ≈ 5f0 atol = 1e-4

        # A refit with the wrong number of transforms is refused rather than
        # silently truncated.
        @test_throws ArgumentError Raycore.update_transforms!(t, h, [xshift(1f0), xshift(2f0)])
    end

    @testset "an empty structure syncs and adapts" begin
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        Raycore.sync!(t)
        @test Raycore.n_instances(t) == 0
        @test Adapt.adapt(be, t) isa Mantle.AdaptedAccel
    end

    @testset "the ray-tracing pipeline answer is the backend's, on every handle" begin
        # Hikari picks its trace path off this, asking whichever of the backend,
        # the TLAS or the adapted accel it holds. A triangles-only accel answers
        # what its backend does; an untyped default answering a handle no backend
        # wrote a method for would be a wrong answer, not a missing one.
        t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        @test Mantle.supports_rt_pipeline(t) == Mantle.supports_rt_pipeline(be)
        push!(t, GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0)))
        Raycore.sync!(t)
        @test Mantle.supports_rt_pipeline(Adapt.adapt(be, t)) == Mantle.supports_rt_pipeline(be)
    end

    @testset "a recorded plan traces what a direct launch traces" begin
        # An inline ray query is a call inside the SHADER, so nothing about the
        # dispatch says the kernel traces — and on Metal an indirect command
        # buffer that did not declare `supportRayTracing` refuses it by returning
        # a MISS. Every ray came back empty, the image was black, and there was no
        # error anywhere. It cost a benchmark run that came out 24x faster than
        # the software path because it rendered nothing.
        #
        # Read against the same kernel launched directly rather than against a
        # fixed number: what is at stake is whether the recorded one traversed at
        # all.
        mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
        O = Point3f(0, 0, 4)
        n = 256
        dirs = probe_dirs(n, O)
        hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(hw, mesh)
        Raycore.sync!(hw)
        accel = Adapt.adapt(be, hw)

        wh, wt, _ = probe(be, accel, dirs, O)

        g = Mantle.Graph(dev)
        d = Mantle.Buffer(dev, dirs)
        hits = Mantle.Buffer(dev, zeros(Int32, n))
        ts = Mantle.Buffer(dev, zeros(Float32, n))
        insts = Mantle.Buffer(dev, zeros(UInt32, n))
        Mantle.dispatch!(g, hw_probe!, (hits, ts, insts, d, accel, O), n; name = "trace")
        plan = Mantle.record!(Mantle.Plan(g))
        @test Mantle.recorded(plan) == Mantle.recordable(dev, plan)
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        rh, rt = Array(Mantle.storage(hits)), Array(Mantle.storage(ts))

        # Vacuous otherwise: a traversal that misses everything agrees with itself.
        @test sum(wh) > n ÷ 4
        @test rh == wh
        @test rt == wt
        Mantle.free!(plan)
    end
end

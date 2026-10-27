# `Raycore.transparent_hits` means the same on the software structure and the
# hardware one: one scene of see-through layers over an opaque floor, pushed into
# both, traced by the same kernel, records the same hits in the same order and
# blocks the same rays. What a shadow ray through foliage onto the ground does.
#
# Closest hits take every instance as opaque on both: an instance pushed with
# `opaque = false` is still where a camera ray stops. Without the opaque ray flag
# a hardware query offers its triangles as candidates, which the closest-hit loop
# leaves uncommitted, and the camera ray goes through.

using Test, Mantle, Raycore, Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Point3f, Vec3f, Mat4f, GLTriangleFace
import GeometryBasics
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

@kernel function transparent_hits_probe!(out, closest, @Const(origins), accel, t_max, limit)
    i = @index(Global)
    r = Raycore.Ray(o = origins[i], d = Vec3f(0, 0, -1), t_max = t_max)
    @inbounds out[i] = Raycore.transparent_hits(accel, r, 0xff, Val(4), limit)
    @inbounds closest[i] = Raycore.closest_hit_record(accel, r, 0xff)
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "transparent_hits: the same on Raycore.TLAS and HWTLAS" begin
    be = TESTBACKEND
    quad(z) = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
        [Point3f(-1, -1, z), Point3f(1, -1, z), Point3f(1, 1, z), Point3f(-1, 1, z)],
        [GLTriangleFace(1, 2, 3), GLTriangleFace(1, 3, 4)]))
    I4 = Mat4f(LinearAlgebra.I)
    sw = Raycore.TLAS(be)
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    layers = Dict{Int, Tuple{Raycore.TLASHandle, Raycore.TLASHandle}}()
    for z in (2, 5, 0, 3, 1, 4)
        layers[z] = (push!(sw, quad(Float32(z)), I4; instance_id = UInt32(10 + z), opaque = false),
                     push!(hw, quad(Float32(z)), I4; instance_id = UInt32(10 + z), opaque = false))
    end
    push!(sw, quad(-3f0), I4; instance_id = UInt32(99))
    push!(hw, quad(-3f0), I4; instance_id = UInt32(99))

    origins = [Point3f(x, y, 10) for x in (-0.6f0, 0.1f0, 0.7f0) for y in (-0.5f0, 0.2f0, 0.8f0)]
    push!(origins, Point3f(5, 5, 10))                       # past everything
    function shoot(accel; t_max = 12f0, limit = 10)
        Raycore.sync!(accel)
        out = Adapt.adapt(be, fill(Raycore.TransparentHits{4}(), length(origins)))
        closest = Adapt.adapt(be, fill(Raycore.HitRecord(), length(origins)))
        transparent_hits_probe!(be, 64)(out, closest, Adapt.adapt(be, origins), Adapt.adapt(be, accel),
                                        t_max, limit; ndrange = length(origins))
        KA.synchronize(be)
        return Array(out), Array(closest)
    end
    distances(th) = [h.t for h in th.hits[1:min(th.count, 4)]]
    ids(th) = [h.custom for h in th.hits[1:min(th.count, 4)]]

    for (t_max, limit, blocked) in ((12f0, 10, false), (20f0, 10, true), (12f0, 3, true))
        s, sc = shoot(sw; t_max, limit)
        h, hc = shoot(hw; t_max, limit)
        for i in 1:length(origins) - 1
            @test s[i].blocked == h[i].blocked == blocked
            if !blocked
                # Four kept; the hardware counts all six, the software stops at four.
                @test s[i].count >= 4 && h[i].count >= 4
                @test ids(s[i]) == ids(h[i]) == [15, 14, 13, 12]
                @test isapprox(distances(s[i]), [5, 6, 7, 8]; atol = 1f-4)
                @test isapprox(distances(h[i]), [5, 6, 7, 8]; atol = 1f-4)
            end
            # Closest hits stop at the top layer, see-through or not.
            @test sc[i].kind == hc[i].kind == Raycore.HIT_TRIANGLE
            @test isapprox(sc[i].t, 5f0; atol = 1f-4) && isapprox(hc[i].t, 5f0; atol = 1f-4)
            @test sc[i].custom == hc[i].custom == 15
        end
        @test !s[end].blocked && !h[end].blocked && s[end].count == h[end].count == 0
        @test sc[end].kind == hc[end].kind == Raycore.HIT_MISS
    end

    # Made opaque, a layer stops the ray; made see-through again, it is recorded.
    # On the hardware the flag is the instance's, written at the next build.
    for (i, accel) in ((1, sw), (2, hw))
        @test Raycore.set_opaque!(accel, layers[2][i], true)
    end
    @test all(th -> th.blocked, first(shoot(sw))[1:end-1])
    @test all(th -> th.blocked, first(shoot(hw))[1:end-1])
    for (i, accel) in ((1, sw), (2, hw))
        @test Raycore.set_opaque!(accel, layers[2][i], false)
    end
    @test !any(th -> th.blocked, first(shoot(hw))[1:end-1])
    # Hidden and see-through: neither recorded nor blocking.
    @test Raycore.set_visible!(hw, layers[4][2], false)
    h, _ = shoot(hw)
    @test all(th -> !th.blocked && th.count == 5, h[1:end-1])
    @test !Raycore.set_opaque!(hw, Raycore.TLASHandle(UInt32(999)), true)
end

end

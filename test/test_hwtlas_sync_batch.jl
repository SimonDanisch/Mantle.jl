# `sync!` over batches of instances a kernel writes: one batch, then several,
# each with its own record array. Asserted by tracing: after `sync!` every
# instance of every batch is there with its own id, the masks the records carry
# keep the batches apart, and a refit of the several-batch structure moves all
# of them — which is what being built for refit means to a caller.

using Test, Mantle, Raycore, Adapt
using GeometryBasics: Point3f, Vec3f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const Tri = Raycore.Triangle{UInt32}

if Mantle.supports_hwtlas(TESTBACKEND)

unit_blas() = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
    Mantle.build_blas(ctx, [(-0.4f0, -0.4f0, 0f0), (0.4f0, -0.4f0, 0f0), (0f0, 0.4f0, 0f0)],
                      UInt32[0, 1, 2])
end

"""`length(xs)` records at `(x, y, 0)` with ids `first_id …` and one mask."""
hostrecords(xs, y, first_id, mask) =
    [Raycore.InstanceRecord(tlastranslation(x, y, 0), UInt32(first_id + k - 1), UInt32(mask))
     for (k, x) in enumerate(xs)]
records_at(xs, y, first_id, mask) = Mantle.devicearray(TESTBACKEND, hostrecords(xs, y, first_id, mask))

downonto(ps) = TLASProbe(TESTBACKEND, [Point3f(p[1], p[2], 1) for p in ps], fill(Vec3f(0, 0, -1), length(ps)))

@testset "sync! with one instance batch" begin
    xs = Float32.(0:7)
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    push!(tlas, unit_blas(), records_at(xs, 0f0, 1, 0x02))
    Raycore.sync!(tlas)
    @test Adapt.adapt(TESTBACKEND, tlas) isa Mantle.AdaptedAccel
    r = tlastrace(downonto([(x, 0f0) for x in xs]), tlas)
    @test all(r.hit)
    @test r.id == UInt32.(1:8)
    @test all(t -> isapprox(t, 1f0; atol = 1f-3), r.t)
end

@testset "sync! over several batches, and a refit of all of them" begin
    blas = unit_blas()
    xa = Float32.(0:3)                 # 4 instances at y = 0, mask 0x02
    xb = Float32.(10:15)               # 6 instances at y = 5, mask 0x04
    a = records_at(xa, 0f0, 1, 0x02)
    b = records_at(xb, 5f0, 101, 0x04)
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    push!(tlas, blas, a)
    push!(tlas, blas, b)
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == length(xa) + length(xb)

    places = vcat([(x, 0f0) for x in xa], [(x, 5f0) for x in xb])
    probe = downonto(places)
    r = tlastrace(probe, tlas)
    @test all(r.hit)
    @test r.id == UInt32[1:4; 101:106]
    # Each batch's records carry its mask.
    @test tlastrace(probe, tlas; mask = 0x02).hit == [trues(4); falses(6)]
    @test tlastrace(probe, tlas; mask = 0x04).hit == [falses(4); trues(6)]

    # Rewrite both arrays one unit up in y and refit: every instance moves.
    copyto!(a, hostrecords(xa, 1f0, 1, 0x02))
    copyto!(b, hostrecords(xb, 6f0, 101, 0x04))
    before = Adapt.adapt(TESTBACKEND, tlas)
    Raycore.refit!(tlas)
    @test Adapt.adapt(TESTBACKEND, tlas) === before
    @test !any(tlastrace(probe, tlas).hit)
    r = tlastrace(downonto([(p[1], p[2] + 1f0) for p in places]), tlas)
    @test all(r.hit)
    @test r.id == UInt32[1:4; 101:106]
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end

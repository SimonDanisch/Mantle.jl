# `push!(tlas, blas, records; n)`: one batch of `n` instances of a pre-built BLAS,
# their transforms, custom indices and masks in a device array of
# `Raycore.InstanceRecord`s. Asserted through what a caller sees: the handle, the
# instance count, the array behind the handle, and rays — every one of the `n`
# instances is hit with its own id, the records past `n` are not instances, and
# the records' masks are the instances' masks.

using Test, Mantle, Raycore
using GeometryBasics: Point3f, Vec3f, Mat4f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "push!(hwtlas, blas, records) -- registration" begin
    # One triangle around the origin of its instance, in the plane z = 0.
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
        Mantle.build_blas(ctx, [(-0.4f0, -0.4f0, 0f0), (0.4f0, -0.4f0, 0f0), (0f0, 0.4f0, 0f0)],
                          UInt32[0, 1, 2])
    end
    # 100 records, of which the batch is the first 60; instance `i` at (i, 0, 0).
    total, n = 100, 60
    records = Mantle.devicearray(TESTBACKEND,
        [Raycore.InstanceRecord(tlastranslation(i, 0, 0), UInt32(i), UInt32(0x02)) for i in 1:total])

    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    handle = push!(tlas, blas, records; n = n)
    @test handle isa Raycore.TLASHandle
    @test Raycore.n_instances(tlas) == n
    @test Raycore.n_geometries(tlas) == 1
    @test Raycore.instance_buffer(tlas, handle) === records

    Raycore.sync!(tlas)
    probe = TLASProbe(TESTBACKEND, [Point3f(i, 0, 1) for i in 1:total], fill(Vec3f(0, 0, -1), total))
    r = tlastrace(probe, tlas)
    @test r.hit == [i <= n for i in 1:total]       # only the first `n` are instances
    @test r.id[1:n] == UInt32.(1:n)                 # each with its own custom index
    @test all(t -> isapprox(t, 1f0; atol = 1f-3), r.t[1:n])
    # The records' mask is the instances' mask.
    @test tlastrace(probe, tlas; mask = 0x02).hit == r.hit
    @test !any(tlastrace(probe, tlas; mask = 0x04).hit)

    # `n` past the end of the array is refused.
    @test_throws ArgumentError push!(tlas, blas, records; n = total + 1)
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end

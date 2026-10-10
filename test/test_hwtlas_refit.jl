# `Raycore.refit!(tlas)` after a kernel rewrote a batch's instance records: the
# structure takes what was written, in place.
#
# The records are written by `write_grain_instances_kernel`, a kernel, which is
# the point: nothing on the host knows the new transforms, so a structure that
# only refits from what the host last uploaded keeps tracing the old positions.
# "In place" is asserted as what it buys a caller: the adapted form a kernel or a
# plan holds stays the same object and sees the new positions, which is the
# reason a refit is worth having over a rebuild.

using Test, Mantle, Raycore, Adapt
import KernelInterface as KI
import KernelAbstractions as KA
using GeometryBasics: Point3f, Vec3f, Vec4f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "refit! takes the records a kernel rewrote" begin
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
        Mantle.build_blas(ctx, [(-1f0, -1f0, 0f0), (1f0, -1f0, 0f0), (0f0, 1f0, 0f0)], UInt32[0, 1, 2])
    end
    n = 4
    radius = 1f0
    quats = Mantle.devicearray(TESTBACKEND, [Vec4f(0, 0, 0, 1) for _ in 1:n])
    phys = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, n)
    rend = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, n)
    writegrains!(positions) =
        KI.Kernel(TESTBACKEND, Mantle.write_grain_instances_kernel)(
            Mantle.devicearray(TESTBACKEND, positions), quats, radius, phys, rend; ndrange = n)

    at = [Point3f(3f0 * (i - 1), 0f0, 0f0) for i in 1:n]
    writegrains!(at)
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(tlas, blas, phys)
    Raycore.sync!(tlas)
    before = Adapt.adapt(TESTBACKEND, tlas)

    down(ps) = TLASProbe(TESTBACKEND, [p + Vec3f(0, 0, 5) for p in ps], fill(Vec3f(0, 0, -1), length(ps)))
    @test all(tlastrace(down(at), tlas).hit)

    # Refit cycle: new positions written by the kernel, then refit!.
    moved = [p + Vec3f(100f0, 0f0, 0f0) for p in at]
    writegrains!(moved)
    @test Raycore.refit!(tlas) === tlas
    # Refit reuses the adapted form: what a kernel held before it still traces.
    @test Adapt.adapt(TESTBACKEND, tlas) === before
    r = tlastrace(down(moved), before)
    @test all(r.hit)
    @test all(t -> isapprox(t, 5f0; atol = 1f-3), r.t)
    @test r.id == UInt32.(0:n-1)                    # the ids the kernel wrote
    @test !any(tlastrace(down(at), before).hit)     # nothing left at the old places
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end

# The grain instance writer: one kernel writing the instance records of two
# batches, a grain's physics geometry and its rendering geometry, as
# `Raycore.InstanceRecord`s — the record a kernel writes on every backend.
#
# No `write_meshscatter_instances_kernel` testset: commit 9e0ec1d ("instance
# transform cleanup", 2026-05-06) deleted that
# kernel and its `_pervec` variant on purpose, replacing them with
# `Raycore.update_transforms!` / `_apply_pending_update!`. The test kept importing
# the deleted symbol and failed 8 assertions for three months without anyone
# noticing, because this file was not registered in runtests.jl. It is now.
#
# The records themselves are checked first, read back; then what they are FOR,
# traced: pushed as two batches of two geometries, a ray with the physics mask
# finds the physics geometry and one with the rendering mask the other. That
# trace replaces the assertion this file made on Vulkan alone, that each record
# held its BLAS's device address: a portable record names no geometry, the batch
# it is pushed in does.

using Test, Mantle, Raycore
import KernelInterface as KI
import KernelAbstractions as KA
using GeometryBasics: Point3f, Vec3f, Vec4f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const NGRAINS = 4
const RADIUS = 0.5f0

"""Grain `i` at `(2i, 0, 0)`, unrotated: both records, written by the kernel."""
function grain_records()
    positions = Mantle.devicearray(TESTBACKEND, [Point3f(2f0 * i, 0f0, 0f0) for i in 1:NGRAINS])
    quats = Mantle.devicearray(TESTBACKEND, [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:NGRAINS])   # identity
    phys = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, NGRAINS)
    rend = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, NGRAINS)
    # A plain KernelInterface kernel since Mantle stopped defining `@kernel`s,
    # so it is launched the way `src/` launches its siblings, `KI.Kernel`.
    KI.Kernel(TESTBACKEND, Mantle.write_grain_instances_kernel)(positions, quats, RADIUS, phys, rend;
                                                               ndrange = NGRAINS)
    KA.synchronize(TESTBACKEND)
    return phys, rend
end

@testset "write_grain_instances_kernel -- 4 grains, identity rotations" begin
    phys, rend = grain_records()
    phys_cpu, rend_cpu = Array(phys), Array(rend)
    for i in 1:NGRAINS
        # Both share the transform: identity rotation x radius scale x translation
        # by the grain's position. `Tuple(transform)` is the twelve floats in the
        # order a `Mat3x4f` stores them, the row-major 3x4 rows one after another.
        expected_t = (RADIUS, 0f0, 0f0, 2f0 * i,
                      0f0, RADIUS, 0f0, 0f0,
                      0f0, 0f0, RADIUS, 0f0)
        @test Tuple(phys_cpu[i].transform) == expected_t
        @test Tuple(rend_cpu[i].transform) == expected_t
        # Custom index i - 1 on both.
        @test phys_cpu[i].id == UInt32(i - 1)
        @test rend_cpu[i].id == UInt32(i - 1)
        # The masks that keep the two apart.
        @test phys_cpu[i].mask == UInt32(0x02)
        @test rend_cpu[i].mask == UInt32(0x04)
    end
end

if Mantle.supports_hwtlas(TESTBACKEND)
    @testset "each batch instances its own geometry, told apart by mask" begin
        # Physics: a triangle in the grain's plane z = 0. Rendering: the same
        # triangle half a radius below. Object space, so both are scaled by the
        # record's transform.
        tri(z) = [(-1f0, -1f0, z), (1f0, -1f0, z), (0f0, 1f0, z)]
        physblas, rendblas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
            (Mantle.build_blas(ctx, tri(0f0), UInt32[0, 1, 2]),
             Mantle.build_blas(ctx, tri(-1f0), UInt32[0, 1, 2]))
        end
        phys, rend = grain_records()
        tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
        push!(tlas, physblas, phys)
        push!(tlas, rendblas, rend)
        Raycore.sync!(tlas)

        probe = TLASProbe(TESTBACKEND, [Point3f(2f0 * i, 0f0, 5f0) for i in 1:NGRAINS],
                          fill(Vec3f(0, 0, -1), NGRAINS))
        p = tlastrace(probe, tlas; mask = 0x02)
        @test all(p.hit)
        @test all(t -> isapprox(t, 5f0; atol = 1f-3), p.t)                # z = 0
        @test p.id == UInt32.(0:NGRAINS-1)
        r = tlastrace(probe, tlas; mask = 0x04)
        @test all(r.hit)
        @test all(t -> isapprox(t, 5f0 + RADIUS; atol = 1f-3), r.t)       # z = -radius
        @test r.id == UInt32.(0:NGRAINS-1)
        @test !any(tlastrace(probe, tlas; mask = 0x01).hit)
    end
end

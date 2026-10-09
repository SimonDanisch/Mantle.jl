# The device's cooperative-matrix shape query must honour the operand type, not
# just the extents.
#
# It must not match on M, N and K alone: a device reports the same extents for
# different component types — the AMD Radeon 8060S lists 16x16x16 four ways:
# (Float16 -> Float32), (Float16 -> Float16), (UInt8 -> Int32) and
# (Int8 -> Int32) — so an extent-only match answers "yes" for Float16 on hardware
# that only implements the integer forms, and the kernel then emits
# cooperative-matrix instructions the device cannot execute.
#
# Asked through what a kernel reads — `caps(...)`'s shape table, `supports`, and
# the `tile` the GEMM kernels are written against — and against whatever the
# running device reports rather than a vendor's table, so it means the same thing
# everywhere. That the table matches the driver's own query is
# `test_device_caps.jl`; this is that the query built on it keeps the type.

using Test
using KernelInterface: MatrixShape, SubgroupScope, supports, matrix_shapes
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "coopmat shape query honours the operand type" begin
    c = Mantle.caps(TESTBACKEND)
    shapes = matrix_shapes(c)

    if isempty(shapes)
        @info "no cooperative-matrix support on this device; skipping"
        @test_skip !isempty(shapes)
    else
        # Every reported (extent, operand type) pair must be found...
        for s in shapes
            @test supports(c, s)
        end

        # ...and an operand type the device reports for NO shape must not match,
        # however many shapes share its extents.
        reported = Set(s.ab for s in shapes)
        for ty in (Float16, Float32, Float64, Int8, UInt8)
            ty in reported && continue
            for s in shapes
                @test !supports(c, MatrixShape(ty, s.acc, s.M, s.N, s.K, s.scope))
            end
        end

        # Extents the device never reports must not match either.
        @test !supports(c, MatrixShape(Float16, Float32, 7, 7, 7, SubgroupScope()))

        # A type no cooperative matrix holds is not a shape.
        @test !supports(c, MatrixShape(ComplexF32, ComplexF32, 16, 16, 16, SubgroupScope()))

        # What the bug did to a kernel: `coopmat` and `tile` must name an
        # fp16 -> fp32 instruction the table really has, since that is the one the
        # GEMM kernels emit. An extent-only match let `coopmat` say yes where only
        # the integer forms exist.
        @test supports(c, MatrixShape(Float16, Float32, c.tile, c.tile, c.tile, SubgroupScope()))
    end
end

# `coopmat_shape` must honour the operand type, not just the extents.
#
# It takes `::Type{T}` and used to ignore it entirely, matching on M, N and K
# alone. A device reports the same extents for different component types — the
# AMD Radeon 8060S lists 16x16x16 four ways: (Float16 -> Float32),
# (Float16 -> Float16), (UInt8 -> Int32) and (Int8 -> Int32) — so an
# extent-only match answers "yes" for Float16 on hardware that only implements
# the integer forms, and the kernel then emits cooperative-matrix instructions
# the device cannot execute.
#
# Written against whatever the running device reports rather than hardcoding a
# vendor's table, so it means the same thing everywhere.

using Test, Lava, Mantle
@testset "coopmat_shape honours the operand type" begin
    ctx = Mantle.vk_context()

    if !ctx.coopmat_available
        @info "no cooperative-matrix support on this device; skipping"
        @test_skip ctx.coopmat_available
    else
        shapes = ctx.coopmat_shapes
        @test !isempty(shapes)

        # Every reported (extent, operand type) pair must be found...
        for s in shapes
            s.scope == Mantle.VK_SCOPE_SUBGROUP || continue
            T = findfirst(t -> Mantle.vkcomponenttype(t) == s.ab_type,
                          (Float16, Float32, Float64,
                           Int8, Int16, Int32, Int64,
                           UInt8, UInt16, UInt32, UInt64))
            T === nothing && continue
            ty = (Float16, Float32, Float64, Int8, Int16, Int32, Int64,
                  UInt8, UInt16, UInt32, UInt64)[T]
            @test Mantle.coopmat_shape(ctx, ty, s.M, s.N, s.K)
        end

        # ...and an operand type the device reports for NO shape must not match,
        # however many shapes share its extents.
        reported = Set(s.ab_type for s in shapes)
        for ty in (Float16, Float32, Float64, Int8, UInt8)
            Mantle.vkcomponenttype(ty) in reported && continue
            for s in shapes
                @test !Mantle.coopmat_shape(ctx, ty, s.M, s.N, s.K)
            end
        end

        # Extents the device never reports must not match either.
        @test !Mantle.coopmat_shape(ctx, Float16, 7, 7, 7)

        # A type with no VkComponentTypeKHR mapping is not a shape.
        @test !Mantle.coopmat_shape(ctx, ComplexF32, 16, 16, 16)
        @test Mantle.vkcomponenttype(ComplexF32) === nothing
    end
end

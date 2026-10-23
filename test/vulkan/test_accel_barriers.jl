using Test, Mantle, Raycore, GeometryBasics, GPUArrays
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "..", "hwtlas_helpers.jl"))

# A BLAS built from device arrays reads them in place, through views that start
# one element in: a build or refit that dropped the views' offset reads the
# sentinel vertex and moves the triangle away from the probe ray.
function check_accel_barriers(ctx)
    dev = ctx
    if !Mantle.supports_hwtlas(dev)
        @test_skip Mantle.supports_hwtlas(dev)
        return nothing
    end
    be = Mantle.backend(dev)
    @testset "a BLAS built from offset device views refits in place" begin
        vertices = [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0), (0f0, 1f0, 0f0)]
        moved = [(x, y, 2f0) for (x, y, _) in vertices]
        vb = Mantle.LavaArray(vcat([(9f0, 9f0, 9f0)], vertices, [(8f0, 8f0, 8f0)]);
                              bq = dev.default_bq, extra_usage = Mantle.AS_INPUT_USAGE)
        ib = Mantle.LavaArray(UInt32[99, 0, 1, 2, 99]; bq = dev.default_bq, extra_usage = Mantle.AS_INPUT_USAGE)
        vs = GPUArrays.derive(NTuple{3,Float32}, vb, (3,), 1)
        ids = GPUArrays.derive(UInt32, ib, (3,), 1)
        blas = Mantle.build_accel!(dev.default_bq) do build
            Mantle.build_blas(build, vs, ids; allow_update = true)
        end
        tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(tlas, blas)
        Raycore.sync!(tlas)
        probe = TLASProbe(be, [Point3f(0.25f0, 0.25f0, -1f0)], [Vec3f(0, 0, 1)])
        @test only(tlastrace(probe, tlas).t) ≈ 1f0 atol = 1f-3
        Mantle.build_accel!(dev.default_bq) do build
            Mantle.refit_blas!(build, blas, moved)
        end
        Raycore.refit!(tlas)
        @test only(tlastrace(probe, tlas).t) ≈ 3f0 atol = 1f-3
        @test Mantle.bda_address(first(blas.preserves)) == Mantle.bda_address(vs)
        @test Array(vb) == vcat([(9f0, 9f0, 9f0)], moved, [(8f0, 8f0, 8f0)])
        @test_throws ArgumentError Mantle.build_accel!(dev.default_bq) do build
            Mantle.refit_blas!(build, blas, vcat(moved, moved))
        end
        Mantle.waitidle(dev)
        Mantle.drain!(dev.default_bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
end

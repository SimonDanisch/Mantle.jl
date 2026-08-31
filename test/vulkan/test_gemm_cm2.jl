"""
The coopmat2 GEMM: workgroup-scope matrices and tensor-addressed loads.

Not routed to — `coopmat_gemm!` still runs the staged kernel — so this is what
keeps it honest until `tools/gemm_cm2_ab.jl` decides whether it ships. A kernel
that is wrong and unused is worse than no kernel: it looks available.

The ragged shape is the point of the middle case. Every extent divides nothing,
which the staged kernel refuses outright and this one takes because the clamping
layout pads each load with zeros — and a zero contributes nothing to a sum. If
that ever stops holding, this is where it shows.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "coopmat2 GEMM (workgroup scope)" begin
    back = LavaBackend()
    dev = Lava.caps(back)

    @testset "the tiling is a device question" begin
        # Legal only where the device reports workgroup-scope shapes at all.
        @test (MVE.gemm_cm2_tiling(dev) === nothing) == isempty(dev.wggran)
        none = Lava.DeviceCaps(dev; wggran = NTuple{4,Int}[])
        @test MVE.gemm_cm2_tiling(none) === nothing
        @test !MVE.gemm_cm2_fits(none, 64, 64, 32, 256)

        # An invented device, so these assert the RULE and not this card.
        d = Lava.DeviceCaps(dev; wggran = [(128, 32, 16, 16), (256, 32, 32, 16)])
        @test MVE.gemm_cm2_fits(d, 64, 64, 32, 256)
        @test !MVE.gemm_cm2_fits(d, 64, 16, 32, 256)   # BM 16 < N granularity 32
        @test !MVE.gemm_cm2_fits(d, 64, 64, 8, 256)    # BK 8  < K granularity 16
        @test !MVE.gemm_cm2_fits(d, 64, 64, 32, 64)    # 64 invocations unreported
        # …and whatever the chooser returns must satisfy its own predicate. That
        # is the property worth pinning: a chooser handing back a tiling nothing
        # can run is the one failure this pair exists to prevent.
        for c in (dev, d, Lava.DeviceCaps(dev; wggran = [(256, 32, 32, 16)]))
            t = MVE.gemm_cm2_tiling(c)
            t === nothing || @test MVE.gemm_cm2_fits(c, t...)
        end
    end

    if !isempty(dev.wggran)
        @testset "a tiling the device forbids is refused, not launched" begin
            A = KA.allocate(back, Float16, 64, 64)
            B = KA.allocate(back, Float16, 64, 64)
            C = KA.allocate(back, Float32, 64, 64)
            # `BK = 8` is below every reported K granularity. Before this check it
            # reached `vkCreateComputePipelines` and failed there, naming neither
            # the tiling nor the rule.
            @test_throws ArgumentError MVE.coopmat_gemm_cm2!(C, A, B, 64, 64, 64;
                                                              tiling = (64, 64, 8, 256))
            A = B = C = nothing; GC.gc()
        end
    end

    if isempty(dev.wggran)
        @info "no workgroup-scope cooperative matrices here; kernel not exercised" dev
    else
        @testset "C = A*B, both destination types" begin
            for (M, N, K) in ((128, 192, 96), (100, 130, 70), (576, 512, 576))
                a = Float16.(reshape(sin.(range(0, 11, M * K)), M, K) .* 0.5)
                b = Float16.(reshape(cos.(range(0, 7, K * N)), K, N) .* 0.5)
                A = KA.allocate(back, Float16, M, K); copyto!(A, a)
                B = KA.allocate(back, Float16, K, N); copyto!(B, b)
                want = Float32.(a) * Float32.(b)
                scale = maximum(abs, want)
                for (T, tol) in ((Float32, 1e-4), (Float16, 3e-3))
                    C = KA.allocate(back, T, M, N); fill!(C, T(NaN))
                    @test MVE.coopmat_gemm_cm2!(C, A, B, M, N, K) !== nothing
                    KA.synchronize(back)
                    got = Float32.(Array(C))
                    # A kernel that writes nothing also matches a zero reference.
                    @test all(isfinite, got)
                    @test maximum(abs, got) > 1e-3
                    @test maximum(abs, got .- want) / scale < tol
                end
                A = B = nothing; GC.gc()
            end
        end

        @testset "the subgroup-scope, register-blocked form" begin
            # `gemm_cm2_sg!`: tensor loads into `16x16` SUBGROUP tiles with the
            # staged kernel's 4x4 register block. The other half of the
            # experiment the file's header describes, and unrouted for the same
            # reason — but it has to be correct to be evidence.
            for (M, N, K) in ((256, 512, 128), (100, 130, 70))
                a = Float16.(reshape(sin.(range(0, 11, M * K)), M, K) .* 0.5)
                b = Float16.(reshape(cos.(range(0, 7, K * N)), K, N) .* 0.5)
                A = KA.allocate(back, Float16, M, K); copyto!(A, a)
                B = KA.allocate(back, Float16, K, N); copyto!(B, b)
                want = Float32.(a) * Float32.(b)
                for nw in (2, 4)
                    C = KA.allocate(back, Float32, M, N); fill!(C, Float32(NaN))
                    @test MVE.coopmat_gemm_cm2_sg!(C, A, B, M, N, K; nw) !== nothing
                    KA.synchronize(back)
                    got = Float32.(Array(C))
                    @test all(isfinite, got)
                    @test maximum(abs, got) > 1e-3
                    @test maximum(abs, got .- want) / maximum(abs, want) < 1e-4
                end
                A = B = nothing; GC.gc()
            end
        end

        @testset "it writes only its own tile" begin
            # The clamping store must stop at `M` and `N` rather than at the tile
            # — the edge case a ragged shape exercises, checked directly here by
            # leaving a NaN margin around the destination.
            M, N, K = 100, 130, 64
            a = Float16.(reshape(sin.(range(0, 5, M * K)), M, K) .* 0.5)
            b = Float16.(reshape(cos.(range(0, 3, K * N)), K, N) .* 0.5)
            A = KA.allocate(back, Float16, M, K); copyto!(A, a)
            B = KA.allocate(back, Float16, K, N); copyto!(B, b)
            big = KA.allocate(back, Float32, M, N + 8); fill!(big, Float32(NaN))
            C = view(big, :, 1:N)
            @test MVE.coopmat_gemm_cm2!(C, A, B, M, N, K) !== nothing
            KA.synchronize(back)
            g = Array(big)
            @test all(isfinite, g[:, 1:N])
            @test all(isnan, g[:, (N + 1):(N + 8)])
            A = B = big = nothing; GC.gc()
        end
    end
end

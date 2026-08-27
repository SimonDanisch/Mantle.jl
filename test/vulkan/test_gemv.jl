# `Mantle.gemv!` — batch-1 matrix-vector, ported from llama.cpp's mul_mat_vec.comp.
#
# The oracle is a Float64 host `A' * B`, which for a GEMV is exact enough to be a
# reference rather than a competitor — unlike the FFT there is no algorithmic
# restructuring to hide a bug behind, so the risk here is not the arithmetic but
# the *decomposition*: which lane accumulates which slice of K, and whether the
# two-level reduction (subgroup, then across subgroups through shared memory)
# loses a partial.
#
# So the shapes below are chosen to break that decomposition, not to be
# representative:
#
#   * `K` not a multiple of `4 * BLOCK` — exercises the tail bounds check, which
#     is the one place a lane reads past the end.
#   * `N` not a multiple of `NROWS` — the last workgroup has idle columns, and a
#     missing guard there writes outside `C`.
#   * `K < 4` — fewer elements than one thread's step.
#   * `N < NROWS` — fewer output columns than one workgroup handles.
#   * `BLOCK` at 32 exactly — one subgroup, so the cross-subgroup tree has a
#     single entry and an off-by-one in `nsub` shows up as a zero rather than a
#     wrong value.

using Test, Lava, KernelAbstractions, LinearAlgebra
const KA = KernelAbstractions

"Host reference in Float64. `A` is length K, `B` is (K, N), result is length N."
gemvref(A::Vector{Float32}, B::Matrix{Float32}) = vec(Float64.(A)' * Float64.(B))

relerr(got, want) = maximum(abs, got .- want) / max(maximum(abs, want), eps())

@testset "gemv" begin
    backend = LavaBackend()

    function check(K, N; nrows = nothing, block = nothing)
        ha = Float32.(randn(K))
        hb = Float32.(randn(K, N))
        A = KA.allocate(backend, Float32, K); copyto!(A, ha)
        B = KA.allocate(backend, Float32, K, N); copyto!(B, hb)
        C = KA.allocate(backend, Float32, N); fill!(C, 0.0f0)
        Mantle.gemv!(C, A, B; nrows, block)
        KA.synchronize(backend)
        relerr(Float64.(Array(C)), gemvref(ha, hb))
    end

    @testset "the decoder's own shapes" begin
        for (K, N) in ((1280, 1280), (1280, 5120), (5120, 1280), (1280, 51866))
            @test check(K, N) < 1.0f-5
        end
    end

    @testset "shapes that break the decomposition" begin
        # tails in K, tails in N, and both at once
        for (K, N) in ((1, 1), (3, 1), (7, 13), (129, 7), (1023, 33),
                       (4096 + 1, 5), (5, 4096 + 1), (1279, 1281))
            @test check(K, N) < 1.0f-5
        end
    end

    @testset "every (nrows, block) agrees with every other" begin
        # The configuration is a performance knob and must never be a
        # correctness one. If `gemv_config` later picks a different pair on
        # another device, this is what says that is safe.
        for (K, N) in ((1280, 1280), (1023, 33))
            ref = nothing
            for nrows in (1, 2, 4, 8), block in (32, 64, 128, 256)
                e = check(K, N; nrows, block)
                @test e < 1.0f-5
            end
        end
    end

    # ── the other layout: matrix (M, K), reached as `transpose(W)`
    #
    # This is the one DNNKernels' graphs actually produce (`hoistpermutes`
    # materialises weights contiguous along the OUTPUT axis), so it is the one
    # the decoder's speed depends on. Same oracle, and deliberately the same
    # awkward shapes: the failure modes are different — the tail is in `M` now,
    # not `K`, and the reduction is inside a thread rather than across a subgroup
    # — but a shape that breaks one decomposition is a fine probe for the other.
    "Host reference: `W` is (M, K), `x` is length K, result is length M."
    gemvnref(W::Matrix{Float32}, x::Vector{Float32}) = Float64.(W) * Float64.(x)

    @testset "transposed layout" begin
        function checkn(M, K; tm = nothing, block = nothing, unroll = nothing,
                        withbias = false, epi = identity)
            hw = Float32.(randn(M, K))
            hx = Float32.(randn(K))
            hb = withbias ? Float32.(randn(M)) : nothing
            W = KA.allocate(backend, Float32, M, K); copyto!(W, hw)
            x = KA.allocate(backend, Float32, K); copyto!(x, hx)
            C = KA.allocate(backend, Float32, M); fill!(C, 0.0f0)
            b = nothing
            if withbias
                b = KA.allocate(backend, Float32, M); copyto!(b, hb)
            end
            Mantle.gemv!(C, x, transpose(W); bias = b, epilogue = epi,
                       tm, block, unroll)
            KA.synchronize(backend)
            want = gemvnref(hw, hx)
            withbias && (want .+= Float64.(hb))
            relerr(Float64.(Array(C)), epi.(want))
        end

        @testset "the decoder's own shapes" begin
            for (M, K) in ((1280, 1280), (5120, 1280), (1280, 5120), (51866, 1280))
                @test checkn(M, K) < 1.0f-5
            end
        end

        @testset "shapes that break the decomposition" begin
            # M not a multiple of TM leaves idle rows in the last workgroup; K
            # shorter than one unrolled step skips the main loop entirely; K
            # shorter than the K-group count leaves whole groups with no work at
            # all, and their zero partials must still be summed.
            for (M, K) in ((1, 1), (1, 1280), (31, 7), (33, 3), (1279, 1281),
                           (65, 5), (7, 4097), (4097, 7))
                @test checkn(M, K) < 1.0f-5
            end
        end

        @testset "bias and epilogue land in the store" begin
            for (M, K) in ((1280, 1280), (33, 129))
                @test checkn(M, K; withbias = true) < 1.0f-5
                @test checkn(M, K; withbias = true, epi = v -> max(v, 0.0f0)) < 1.0f-5
            end
        end

        @testset "every (tm, block, unroll) agrees with every other" begin
            for (M, K) in ((1280, 1280), (1279, 129))
                for tm in (32,), block in (32, 64, 128, 256), unroll in (1, 2, 4)
                    @test checkn(M, K; tm, block, unroll) < 1.0f-5
                end
            end
        end

        @testset "it beats mul! at one column" begin
            # The reason this second kernel exists. `mul!` on this shape measured
            # 3.5% of roofline: one thread per output is 1280 threads, five
            # workgroups, and the device is latency-bound however well it reads.
            M, K, iters = 1280, 1280, 50
            hw = fill(0.01f0, M, K)
            W = KA.allocate(backend, Float32, M, K); copyto!(W, hw)
            x = KA.allocate(backend, Float32, K); fill!(x, 0.01f0)
            x2 = KA.allocate(backend, Float32, K, 1); fill!(x2, 0.01f0)
            C = KA.allocate(backend, Float32, M); fill!(C, 0.0f0)
            C2 = KA.allocate(backend, Float32, M, 1); fill!(C2, 0.0f0)
            rg() = (for _ in 1:iters; Mantle.gemv!(C, x, transpose(W)); end)
            rm() = (for _ in 1:iters; mul!(C2, W, x2); end)
            for _ in 1:3; rg(); rm(); end
            KA.synchronize(backend)
            tg = tm_ = Inf
            for _ in 1:5
                KA.synchronize(backend); s = time_ns(); rg()
                KA.synchronize(backend); tg = min(tg, time_ns() - s)
                KA.synchronize(backend); s = time_ns(); rm()
                KA.synchronize(backend); tm_ = min(tm_, time_ns() - s)
            end
            @info "gemv(transposed) vs mul! at ($M,$K)" gemv_ms=tg/iters/1e6 mul_ms=tm_/iters/1e6 speedup=tm_/tg
            @test tm_ / tg > 1.5
        end
    end

    @testset "dimension mismatches are caught" begin
        A = KA.allocate(backend, Float32, 8)
        B = KA.allocate(backend, Float32, 8, 4)
        @test_throws DimensionMismatch Mantle.gemv!(KA.allocate(backend, Float32, 5), A, B)
        @test_throws DimensionMismatch Mantle.gemv!(KA.allocate(backend, Float32, 4),
                                                  KA.allocate(backend, Float32, 9), B)
    end

    @testset "it beats mul! at M = 1" begin
        # The reason the kernel exists. Interleaved, both live kernels, same
        # allocation — a sequential A/B on this box drifts by 5-10%.
        #
        # NOT a roofline claim: at these sizes most of `B` is L2-resident, and an
        # earlier version of this measurement reported 186% "of roofline" because
        # the denominator was a DRAM copy. The claim is only the ratio.
        K, N, iters = 1280, 1280, 50
        A = KA.allocate(backend, Float32, K); fill!(A, 0.01f0)
        A2 = KA.allocate(backend, Float32, 1, K); fill!(A2, 0.01f0)
        B = KA.allocate(backend, Float32, K, N); fill!(B, 0.01f0)
        C = KA.allocate(backend, Float32, N); fill!(C, 0.0f0)
        C2 = KA.allocate(backend, Float32, 1, N); fill!(C2, 0.0f0)
        rg() = (for _ in 1:iters; Mantle.gemv!(C, A, B); end)
        rm() = (for _ in 1:iters; mul!(C2, A2, B); end)
        for _ in 1:3; rg(); rm(); end
        KA.synchronize(backend)
        tg = tm = Inf
        for _ in 1:5
            KA.synchronize(backend); s = time_ns(); rg()
            KA.synchronize(backend); tg = min(tg, time_ns() - s)
            KA.synchronize(backend); s = time_ns(); rm()
            KA.synchronize(backend); tm = min(tm, time_ns() - s)
        end
        @info "gemv vs mul! at (1,$K)@($K,$N)" gemv_ms=tg/iters/1e6 mul_ms=tm/iters/1e6 speedup=tm/tg
        # Measured 3.5x. The floor is deliberately far below that: this asserts
        # the kernel is doing its job, not the exact number, which moves with the
        # driver and with whatever else holds the card.
        @test tm / tg > 1.5
    end
end

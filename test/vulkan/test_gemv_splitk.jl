# A GEMV must run at memory bandwidth, whatever its row count.
#
# `N == 1` is the whole of autoregressive decode. `staged_gemm_ok` declines it,
# so it fell to `strided_gemm_kernel!` — one invocation per output row — and its
# throughput then tracked M rather than the bytes it had to read. On a Radeon
# 8060S, fp16 weights, against a plain read kernel that reaches 211 GB/s:
#
#     shape (M x K)               one-per-row    split-K
#     26624 x  5120  gate/up            220.8      208.2   (already saturated)
#    250624 x  5120  lm_head            210.9      212.3   (already saturated)
#      8192 x  5120  q_proj             105.0      163.2
#      5120 x  8192  o_proj              71.3      202.7
#      5120 x 26624  down                62.5      210.5   3.4x
#      1024 x  5120  k_proj              23.4       59.6
#
# Over K2 Horizon 32B's 449 decode GEMVs that is 634 -> 344 ms per token,
# against a 318 ms bandwidth bound.
#
# What is pinned here is CORRECTNESS, not the throughput: the split changes the
# summation order (S partial sums, added in a second pass) and accumulates in
# fp32 regardless of the destination type, so it must stay at least as accurate
# as the single-pass kernel it replaced. Thresholds are far from both the fp32
# result (~1e-7) and fp16 output rounding (~3e-4) so this pins behaviour rather
# than a measurement of one machine.

using Test, Mantle, KernelAbstractions, LinearAlgebra, Random

const KA = KernelAbstractions

@testset "split-K GEMV" begin
    back = Mantle.LavaBackend()
    Random.seed!(20260911)

    @testset "gemv_split rule" begin
        # Wide enough to fill the device on rows alone: no split, no reduction.
        @test Mantle.gemv_split(26624, 5120) == 1
        @test Mantle.gemv_split(250624, 5120) == 1
        # Too few rows: split until there are enough threads.
        @test Mantle.gemv_split(1024, 5120) > 1
        @test Mantle.gemv_split(5120, 26624) > 1
        # Never below a chunk worth a thread — a short K cannot be split far.
        @test Mantle.gemv_split(64, 128) * 64 <= 128
    end

    # Shapes on both sides of the rule, plus a ragged one that divides evenly
    # into neither the split nor the unroll-by-four.
    for (M, K) in ((1024, 5120), (5120, 8192), (5120, 26624), (8192, 5120),
                   (26624, 5120), (777, 999), (33, 7))
        Wh = rand(Float16, M, K) .- Float16(0.5)
        xh = rand(Float16, K, 1) .- Float16(0.5)
        W = KA.allocate(back, Float16, M, K); copyto!(W, Wh)
        x = KA.allocate(back, Float16, K, 1); copyto!(x, xh)
        # fp64 on the SAME fp16 inputs, so the only error measured is the
        # reduction's, not the inputs'.
        want = Float64.(Wh) * Float64.(xh)
        scale = maximum(abs, want)

        @testset "$(M)x$(K)" begin
            C32 = KA.allocate(back, Float32, M, 1); fill!(C32, 0f0)
            mul!(C32, W, x, 1f0, 0f0); KA.synchronize(back)
            @test maximum(abs, Float64.(Array(C32)) .- want) / scale < 1e-5

            # An fp16 destination must still reduce in fp32 and round once.
            C16 = KA.allocate(back, Float16, M, 1); fill!(C16, Float16(0))
            mul!(C16, W, x, 1f0, 0f0); KA.synchronize(back)
            @test maximum(abs, Float64.(Array(C16)) .- want) / scale < 1e-3

            # alpha/beta belong to the reduction pass, not the partials: a
            # partial sum scaled by alpha and then added S times is alpha*S.
            C2 = KA.allocate(back, Float32, M, 1); fill!(C2, 1f0)
            mul!(C2, W, x, 2f0, 3f0); KA.synchronize(back)
            @test maximum(abs, Float64.(Array(C2)) .- (2 .* want .+ 3)) / scale < 1e-5
        end
    end
end

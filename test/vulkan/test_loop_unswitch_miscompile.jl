# Regression: loop unswitching → double-loop → NVIDIA miscompile.
#
# When a loop contains a branch on a loop-invariant condition, LLVM's
# SimpleLoopUnswitchPass duplicates the loop into two bodies. After our
# StructurizeCFG pipeline that becomes two sequential guarded loops, which
# NVIDIA's shader compiler miscompiles: a comparison inside one of the bodies
# silently returns the wrong result (valid SPIR-V, spirv-val clean, wrong
# answer). This corrupted anything with a loop-invariant branch in a loop —
# notably every `Base.isless`-based merge-sort binary search, so
# `AcceleratedKernels.sort!` lost data with no error.
#
# Fixed in `compiler/target.jl` by dropping SimpleLoopUnswitchPass from the
# Lava loop-optimizer pipeline (keeping all other loop opts). These tests fail
# (wrong results, not crashes) if unswitching is ever re-enabled.

using Test, Lava, Mantle
using Lava.KernelAbstractions
using Random
const KA = Lava.KernelAbstractions

@testset "loop-unswitch miscompile regression" begin
    backend = MVE.LavaBackend()
    bq = MVE.vk_context().default_bq
    bs = 256
    n = 512

    # ── 1. Bare loop-invariant branch (no isless): the minimal trigger. ──
    @kernel inbounds=true cpu=false unsafe_indices=true function uw_kernel!(
            @Const(arr), @Const(flags), out, m::Int32)
        i = @index(Global, Linear)
        @inbounds f = flags[i] != Int32(0)
        acc = Int32(0); j = Int32(0)
        while j < m
            @inbounds x = arr[j + 0x1]
            if f                       # loop-invariant branch → unswitch candidate
                acc += x < 0.5f0  ? Int32(1) : Int32(0)
            else
                acc += x < 0.25f0 ? Int32(1) : Int32(0)
            end
            j += Int32(1)
        end
        out[i] = acc
    end
    arrcpu = rand(MersenneTwister(7), Float32, n)
    flags  = Int32[isodd(i) ? 1 : 0 for i in 1:n]
    a_g = MVE.LavaArray(arrcpu); f_g = MVE.LavaArray(flags)
    o_g = MVE.LavaArray(fill(Int32(-9), n))
    uw_kernel!(backend, bs)(a_g, f_g, o_g, Int32(n), ndrange=(n,)); MVE.vk_flush!(bq)
    lo = count(<(0.25f0), arrcpu); hi = count(<(0.5f0), arrcpu)
    expected = [flags[i] != 0 ? hi : lo for i in 1:n]
    @test Array(o_g) == expected

    # ── 2. isless-based binary search in a loop (the AK.sort kernel shape). ──
    @kernel inbounds=true cpu=false unsafe_indices=true function lb_kernel!(@Const(arr), out, m::Int32)
        i = @index(Global, Linear)
        @inbounds v = arr[i]
        left = Int32(0); right = m
        while right > left + Int32(1)
            mid = left + ((right - left) >> 0x1)
            @inbounds am = arr[mid + 0x1]
            if isless(am, v); left = mid; else; right = mid; end
        end
        out[i] = left
    end
    sorted = sort(rand(MersenneTwister(11), Float32, n))
    s_g = MVE.LavaArray(sorted); b_g = MVE.LavaArray(fill(Int32(-9), n))
    lb_kernel!(backend, bs)(s_g, b_g, Int32(n), ndrange=(n,)); MVE.vk_flush!(bq)
    truth = [let l=Int32(0), r=Int32(n)
        while r > l + Int32(1)
            mm = l + ((r - l) >> 0x1)
            isless(sorted[mm + 1], sorted[i]) ? (l = mm) : (r = mm)
        end; l end for i in 1:n]
    @test Array(b_g) == truth
end

# AK.sort! end-to-end — the original corruption. Multi-pass merge sort is the
# real exercise of the isless binary search across data sizes.
@testset "AcceleratedKernels.sort! multiset preservation" begin
    import AcceleratedKernels as AK
    backend = MVE.LavaBackend()
    for nn in (1024, 4000, 100_000), T in (Float32, UInt32)
        cpu = rand(MersenneTwister(3), T, nn)
        g = MVE.LavaArray(cpu)
        AK.sort!(g); KA.synchronize(backend)
        got = Array(g)
        @test issorted(got)
        @test sort(got) == sort(cpu)   # multiset preserved (no lost/duplicated data)
    end
end

# Regression: NVIDIA miscompiles a shared multi-use PSB OpPtrAccessChain.
#
# Two loads from the SAME array at DIFFERENT, loop-varying indices, inside a
# loop, both read 0 on NVIDIA. Root cause: 1-based indexing emits a
# loop-invariant `base + (-1)` GEP that LICM hoists into the preheader; the two
# loads then chain off that SHARED intermediate OpPtrAccessChain. NVIDIA returns
# 0 for every such load (the SPIR-V is valid and spirv-val passes — it's a
# driver miscompile of a multi-use intermediate PSB access chain).
#
# This corrupted any kernel doing `v[a]` and `v[b]` from one array in a loop —
# notably AcceleratedKernels.sortperm! (indirect comparator `v[ix]<v[iy]`), whose
# block sort returned the identity permutation.
#
# Fixed in `compiler/spirv/emit.jl` (`emit_psb_ptr_arithmetic!`): chained PSB
# access chains are folded into a single OpPtrAccessChain from the root with a
# combined index, so the intermediate is never shared. These tests read 0
# (not crash) if the fold regresses.

using Test, Lava, Mantle
using Lava.KernelAbstractions
using Random
const KA = Lava.KernelAbstractions

@testset "PSB shared-access-chain miscompile regression" begin
    backend = MVE.LavaBackend()
    bq = MVE.vk_context().default_bq
    bs = 256

    # Minimal: two loads from one array at different loop-varying indices.
    @kernel inbounds=true cpu=false unsafe_indices=true function two_loads!(out, @Const(w), m::Int32, len::Int32)
        i = @index(Global, Linear)
        acc = 0f0; k = Int32(0)
        while k < m
            @inbounds acc += w[(UInt32(i) + UInt32(k))         % UInt32(len) + 0x1]
            @inbounds acc += w[(UInt32(i) + UInt32(k) + 0x32)  % UInt32(len) + 0x1]
            k += Int32(1)
        end
        out[i] = acc
        nothing
    end
    n = 256; w = Float32.(1:n)
    out = MVE.LavaArray(fill(-1f0, n))
    two_loads!(backend, bs)(out, MVE.LavaArray(w), Int32(3), Int32(n), ndrange=(n,))
    MVE.vk_flush!(bq)
    expected = [ Float32(sum(w[((i + k) % n) + 1] + w[((i + k + 50) % n) + 1] for k in 0:2)) for i in 1:n ]
    @test Array(out) == expected
    @test !all(Array(out) .== 0)   # the bug made every element 0
end

# AK.sortperm! end-to-end — the original corruption (indirect comparator does two
# loads `v[ix] < v[iy]` from the value array per comparison).
@testset "AcceleratedKernels.sortperm!" begin
    import AcceleratedKernels as AK
    backend = MVE.LavaBackend()
    for nn in (256, 1024, 4000, 100_000)
        v = rand(MersenneTwister(nn + 1), Float32, nn)
        vg = MVE.LavaArray(v)
        ix = MVE.LavaArray(collect(UInt32(1):UInt32(nn)))
        AK.sortperm!(ix, vg); KA.synchronize(backend)
        @test Array(ix) == sortperm(v)
    end
end

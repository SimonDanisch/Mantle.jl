# A cooperative matrix spanning the WHOLE WORKGROUP, not one subgroup.
#
# `VK_NV_cooperative_matrix2`'s `cooperativeMatrixWorkgroupScope`. The value of a
# `32 x 64` fp32 accumulator is 64 components per lane at subgroup scope and 8 at
# workgroup scope with 256 invocations — which is the mechanism behind
# `flash_attn_cm2.comp` holding `O`, `L` and `M` at once where our subgroup-scope
# kernel is gated by a step at 128 registers.
#
# Both scopes run here, on the same operands against the same CPU reference. The
# subgroup row is the CONTROL: if it fails too, the fault is in the test rather
# than in scope, which is exactly the distinction the flash MWE's first failure
# needed and could not make.
#
# Shapes are M=32, K=16, N=64 — all different, because a square case cannot tell
# `A'*B'` from `(B*A)'` and cannot see a shape mapping at all. They also satisfy
# this device's flexible-dimensions granularity **at 256 invocations**: M 32,
# N 32, K 16 for fp16 x fp16 -> fp32. That table is per workgroup size, so the
# launch below is not a free choice.
#
#     julia --project=. dev/Lava/test/mwe_workgroup_scope.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions

const M = 32       # rows of the accumulator
const K = 16       # the shared extent
const N = 64       # columns of the accumulator

"A 2-D clamping layout over a column-major (R, C) array; the layout's dims are
REVERSED, the tensor's last dimension being the fastest-varying one."
lay(R, C) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(C), Int32(R))),
        (Int32(0), Int32(0)), (Int32(C), Int32(R)))

# One kernel, not two: the scope is a type parameter, so the same body serves
# both and the comparison cannot accidentally be between two different programs.
@kernel cpu = false function tgemm_scope!(out, @Const(A), @Const(B), ::Val{SC}) where {SC}
    za = Lava.coopmat_zero(Lava.CoopMatrix{Float16,M,K,Lava.MatrixA,SC})
    zb = Lava.coopmat_zero(Lava.CoopMatrix{Float16,K,N,Lava.MatrixB,SC})
    ma = Lava.tensor_load(za, UInt64(pointer(A)), lay(K, M))   # A is K x M in memory
    mb = Lava.tensor_load(zb, UInt64(pointer(B)), lay(N, K))   # B is N x K in memory
    acc = Lava.coopmat_zero(Lava.CoopMatrix{Float32,M,N,Lava.Accumulator,SC})
    r = Lava.coopmat_muladd(ma, mb, acc)
    Mantle.copyto!(pointer(out), 1, M, r)
end

ctx = MVE.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    A = KA.allocate(back, Float16, K, M)
    B = KA.allocate(back, Float16, N, K)
    a = Float16.(reshape(1:(K * M), K, M) ./ 128)
    b = Float16.(reshape((N * K):-1:1, N, K) ./ 256)
    copyto!(A, a); copyto!(B, b)
    want = Float32.(a') * Float32.(b')            # M x N

    # (label, scope, invocations). The invocation count is part of the contract
    # at workgroup scope: 256 is what this device's granularity table pairs with
    # M=32, N=32.
    runs = [("subgroup (control)", Mantle.SubgroupScope,  32),
            ("workgroup",          Mantle.WorkgroupScope, 256)]
    # In a function, not a bare `for`: a top-level loop assigning to `ok` gets
    # its own local and the accumulated result is lost.
    ok = let
        allgood = true
        for (label, SC, nt) in runs
            out = KA.allocate(back, Float32, M, N)
            fill!(out, Float32(NaN))
            tgemm_scope!(back, nt)(out, A, B, Val(SC); ndrange = nt)
            KA.synchronize(back)
            got = Array(out)
            e = maximum(abs.(got .- want)) / maximum(abs.(want))
            good = isfinite(e) && e < 1e-2
            allgood &= good
            @printf("%-20s %3d threads  max rel err %.3e  %s\n",
                    label, nt, e, good ? "OK" : "MISMATCH")
        end
        allgood
    end
    println()
    ok && println("Workgroup scope works: one matrix over $(256) invocations, " *
                  "$(M*N ÷ 256) accumulator components a lane instead of $(M*N ÷ 32).")
end

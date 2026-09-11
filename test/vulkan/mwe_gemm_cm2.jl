# A GEMM with no shared memory: workgroup-scope matrices and tensor-addressed
# loads, which is `mul_mm_cm2.comp`'s structure.
#
# `addmm` is now the encoder's biggest bucket — 57.8 ms of 126.8 serialised,
# against attention's 28.9 after the flash port — and the staged kernel it runs
# reaches 35 TFLOP/s against the 44 a cuBLAS-class kernel gets on the same
# shapes. This is the same treatment that took attention 43.2 -> 28.9.
#
# **No views, and that is not a simplification of the reference.** `mul_mm_cm2`
# loads B through a transposing view because ggml's arrays are row-major. Ours
# are column-major, so a Julia `(r, c)` array IS the tensor `(c, r)`, and the
# natural product is the transposed one:
#
#     C = A*B   in Julia (M,K)x(K,N)   is   C̃ = B̃·Ã   in tensor terms
#
# So B supplies the A-operand and A the B-operand, every load is plain, and the
# store is plain too. `mwe_tensor_gemm_nonsquare.jl` is where that mapping was
# pinned; this is the first kernel to lean on it for both operands at once.
#
# Ragged shapes are in the list deliberately: the clamping layout pads the loads
# with zeros, which contribute nothing to a sum, so this kernel takes extents
# that divide nothing — where the staged kernel needs `M`, `N` and `K` on the
# tile and hands the rest to a slower path.
#
#     julia --project=. dev/Lava/test/mwe_gemm_cm2.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions

# The kernel itself lives in `Mantle.gemm_cm2!` — this file drives it rather than
# holding a second copy. A kernel written twice is a kernel that drifts, and the
# copy in a test file is the one nobody updates.

function run(M, N, K; tol = 3e-2)
    back = LavaBackend()
    a = Float16.(reshape(sin.(range(0, 11, M * K)), M, K) .* 0.5)
    b = Float16.(reshape(cos.(range(0, 7, K * N)), K, N) .* 0.5)
    A = KA.allocate(back, Float16, M, K); copyto!(A, a)
    B = KA.allocate(back, Float16, K, N); copyto!(B, b)
    C = KA.allocate(back, Float32, M, N); fill!(C, Float32(NaN))

    Mantle.coopmat_gemm_cm2!(C, A, B, M, N, K)
    KA.synchronize(back)
    got = Array(C)
    want = Float32.(a) * Float32.(b)
    e = maximum(abs.(got .- want)) / maximum(abs.(want))
    ok = all(isfinite, got) && e < tol
    @printf("%5d x %5d x %5d   max rel err %.3e  %s\n", M, N, K, e, ok ? "OK" : "MISMATCH")
    ok
end

ctx = Mantle.vk_context()
if isempty(Mantle.caps(Mantle.vk_context()).wggran)
    @info "no workgroup-scope cooperative matrices here — nothing to run"
else
    ok = run(128, 192, 96)          # every extent divides its tile
    ok &= run(100, 130, 70)         # none of them do: clamped loads and store
    ok &= run(576, 4096, 576)       # one of SAM 2's own `addmm` shapes
    println()
    ok && println("A tensor-addressed GEMM at workgroup scope, correct on extents " *
                  "that divide nothing.")
end

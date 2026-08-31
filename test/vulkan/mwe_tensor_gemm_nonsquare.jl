# Which product does tensor addressing compute when the shape is NOT square?
#
# `mwe_tensor_gemm.jl` answers this for 16x16 and reports `A'*B'`. A square case
# cannot separate `A'*B'` from `(B*A)'` — they have the same shape and, for that
# test's operands, the same entries are being compared either way. Flash attention
# is non-square everywhere that matters (`S` is Br x Bc, `O` is Br x E), and this
# project has already been bitten once by a transpose "only a non-square shape
# reveals", so the arrangement gets pinned here before a kernel depends on it.
#
# M, N and K are three different values, so every candidate that is not the right
# one is either the wrong shape (and cannot even be compared) or numerically far.
#
#     julia --project=. dev/Lava/test/mwe_tensor_gemm_nonsquare.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const M = 32       # rows of the A-side operand in memory
const K = 16       # the shared extent
const N = 48       # cols of the B-side operand in memory

"A 2-D clamping layout over a column-major (R, C) array. The layout's dims are
REVERSED — the tensor's last dimension is fastest-varying and Julia's first is —
which is what `mwe_tensor_orientation.jl` measured."
lay(R, C) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(C), Int32(R))),
        (Int32(0), Int32(0)), (Int32(C), Int32(R)))

@kernel cpu = false function tgemm_ns!(out, @Const(A), @Const(B))
    za = Lava.coopmat_zero(AMg{Float16,M,K,Lava.MatrixA})
    zb = Lava.coopmat_zero(AMg{Float16,K,N,Lava.MatrixB})
    ma = Lava.tensor_load(za, UInt64(pointer(A)), lay(K, M))   # A is K x M in memory
    mb = Lava.tensor_load(zb, UInt64(pointer(B)), lay(N, K))   # B is N x K in memory
    acc = Lava.coopmat_zero(AMg{Float32,M,N,Lava.Accumulator})
    r = Lava.coopmat_muladd(ma, mb, acc)
    Mantle.copyto!(pointer(out), 1, M, r)
end

ctx = MVE.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    # A stored K x M, B stored N x K: the shapes that make `A'` be M x K and `B'`
    # be K x N, i.e. the arrangement `mwe_tensor_gemm.jl` reported.
    A = KA.allocate(back, Float16, K, M)
    B = KA.allocate(back, Float16, N, K)
    a = Float16.(reshape(1:(K * M), K, M) ./ 128)
    b = Float16.(reshape((N * K):-1:1, N, K) ./ 256)
    copyto!(A, a); copyto!(B, b)
    out = KA.allocate(back, Float32, M, N)
    fill!(out, Float32(NaN))

    tgemm_ns!(back, 32)(out, A, B; ndrange = 32)
    KA.synchronize(back)
    got = Array(out)

    println("out is $(size(got)); all finite = ", all(isfinite, got))
    # `A'*B'` and `(B*A)'` are the SAME matrix — `(BA)ᵀ ≡ AᵀBᵀ` — so listing both
    # as rival candidates discriminates nothing, and an earlier version of this
    # file reported "MATCH" twice and looked like a test that had failed to
    # decide. What a non-square run actually pins is the SHAPE MAPPING, which a
    # 16x16 case cannot show at all.
    cands = ("A'*B' == (B*A)'" => Float32.(a') * Float32.(b'),)
    for (nm, want) in cands
        if size(want) != size(got)
            @printf("  %-9s shape %s != %s — not this one\n", nm, size(want), size(got))
            continue
        end
        e = maximum(abs.(got .- want)) / max(maximum(abs.(want)), eps(Float32))
        @printf("  %-9s max rel err %.3e %s\n", nm, e, e < 1e-2 ? "  <== MATCH" : "")
    end
    println()
    println("SHAPE MAPPING, which is the part only a non-square run establishes:")
    println("  a logical M x K operand must live in memory as K x M column-major,")
    println("  loaded as AcceleratedMatrix{_,M,K,MatrixA}   (here A stored $(size(a)) -> $M x $K)")
    println("  a logical K x N operand lives in memory as N x K,")
    println("  loaded as AcceleratedMatrix{_,K,N,MatrixB}   (here B stored $(size(b)) -> $K x $N)")
    println("  and the accumulator is M x N                 (here $(size(got)))")
    println()
    println("So: store every operand TRANSPOSED relative to the logical matrix.")
    println("That is what a flash kernel needs for S = Q·Kᵀ and O += P·V, where")
    println("Br, Bc and E are all different and a square check would prove nothing.")
end

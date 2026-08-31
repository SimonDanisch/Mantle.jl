# One attention block through the coopmat2 primitives, end to end.
#
# This is the skeleton `flash_attn_cm2.comp` is built from, at the smallest size
# that exercises every piece: a tensor-loaded Q, a tensor-loaded K read through a
# TRANSPOSING VIEW (no staged transpose), the product, an in-fragment row
# reduction, a per-element `exp`, the second product, and a row sum.
#
# Deliberately NOT the whole algorithm: one key block, so there is no running
# maximum and no rescale. The online-softmax machinery is the part we already
# have working in the cm1 kernel; what is unproven is that these five primitives
# compose, and that is what this answers. The final division by the row sum is
# done on the host so the kernel stays the thing under test.
#
#     julia --project=. dev/Lava/test/mwe_flash_cm2.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const E = 16     # head dimension  — one GEMM tile
const L = 16     # sequence length — one query block, one key block

# Top-level, not closures: `coopmat_reduce`/`coopmat_perelement` name the function
# in the instruction and there is no operand slot for a captured environment.
rmax(x::Float32, y::Float32) = max(x, y)
rsum(x::Float32, y::Float32) = x + y
pneg(::UInt32, ::UInt32, e::Float32) = -e
pexp(::UInt32, ::UInt32, e::Float32) = exp(e)

"2-D clamping layout over a column-major (nrow, ncol) array; the tensor's dims
are the REVERSE, because its last dimension is fastest-varying and Julia's first."
lay(nrow, ncol) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(ncol), Int32(nrow))),
        (Int32(0), Int32(0)), (Int32(ncol), Int32(nrow)))

@kernel cpu = false function flashblock!(O, rowsum, @Const(Q), @Const(K), @Const(V))
    # Q is (E, L) in memory and wants to be the logical L x E A-operand, which is
    # exactly the pinned mapping (a logical M x K operand lives as K x M).
    zq = Lava.coopmat_zero(AMg{Float16,L,E,Lava.MatrixA})
    mq = Lava.tensor_load(zq, UInt64(pointer(Q)), lay(E, L))

    # K is ALSO (E, L), so as the B-operand it is the wrong way round — the
    # transposing view reads it in place instead of staging a copy.
    zk = Lava.coopmat_zero(AMg{Float16,E,L,Lava.MatrixB})
    # Created immediately before its use: `mwe_tensor_view.jl` does the same and
    # works, and with other calls in between the view reached the load as a
    # CONSTANT instead of the created value (spirv-val: "does not have a tensor
    # view type"). Kept adjacent until that is understood.
    vt = Lava.tensor_view(Val(2), Val((1, 0)))
    mk = Lava.tensor_load(zk, UInt64(pointer(K)), lay(E, L), vt)

    S = Lava.coopmat_muladd(mq, mk, Lava.coopmat_zero(AMg{Float32,L,L,Lava.Accumulator}))

    # softmax, in fragment: row max, subtract (as add-of-negated), exp, row sum
    mx  = Lava.coopmat_reduce(rmax, AMg{Float32,L,L,Lava.Accumulator}, S,
                              Val(Lava.CoopMatReduce.Row))
    P   = Lava.coopmat_perelement(pexp, Lava.coopmat_add(S, Lava.coopmat_perelement(pneg, mx)))
    rs  = Lava.coopmat_reduce(rsum, AMg{Float32,L,L,Lava.Accumulator}, P,
                              Val(Lava.CoopMatReduce.Row))

    # O = P·V. V is (E, L) in memory and the B-operand wants (N, K) = (E, L), so
    # this one needs no view.
    zv = Lava.coopmat_zero(AMg{Float16,L,E,Lava.MatrixB})
    mv = Lava.tensor_load(zv, UInt64(pointer(V)), lay(E, L))
    Pa = Lava.coopmat_convert(AMg{Float16,L,L,Lava.MatrixA}, P)
    acc = Lava.coopmat_muladd(Pa, mv, Lava.coopmat_zero(AMg{Float32,L,E,Lava.Accumulator}))

    Mantle.copyto!(pointer(O), 1, L, acc)
    Mantle.copyto!(pointer(rowsum), 1, L, rs)
end

ctx = MVE.vk_context()
if !(ctx.coopmat2.tensor_addressing && ctx.coopmat2.per_element_operations)
    @info "device lacks coopmat2 tensor addressing / per-element ops — nothing to run"
else
    back = LavaBackend()
    q = Float16.(0.20 .* sin.(range(0, 9, E * L))); q = reshape(q, E, L)
    k = Float16.(0.20 .* cos.(range(0, 7, E * L))); k = reshape(k, E, L)
    v = Float16.(0.20 .* sin.(range(0, 5, E * L))); v = reshape(v, E, L)
    Q = KA.allocate(back, Float16, E, L); copyto!(Q, q)
    K = KA.allocate(back, Float16, E, L); copyto!(K, k)
    V = KA.allocate(back, Float16, E, L); copyto!(V, v)
    O  = KA.allocate(back, Float32, L, E); fill!(O, Float32(NaN))
    RS = KA.allocate(back, Float32, L, L); fill!(RS, Float32(NaN))

    flashblock!(back, 32)(O, RS, Q, K, V; ndrange = 32)
    KA.synchronize(back)
    got  = Array(O)
    rsum_got = Array(RS)[:, 1]                    # smeared along the row
    out  = got ./ rsum_got                        # the division, done on the host

    # CPU reference: scores[lq, lk] = Σ_e q[e,lq] k[e,lk], softmax over lk, then ·V
    qf, kf, vf = Float32.(q), Float32.(k), Float32.(v)
    S = qf' * kf                                   # L x L
    P = exp.(S .- maximum(S, dims = 2))
    P = P ./ sum(P, dims = 2)
    want = P * vf'                                 # L x E

    err = maximum(abs.(out .- want)) / maximum(abs.(want))
    @printf("O is %s, all finite = %s\n", string(size(got)), all(isfinite, got))
    @printf("flash block vs CPU softmax attention: max rel err %.3e  %s\n",
            err, err < 2e-2 ? "OK" : "MISMATCH")
    err < 2e-2 && println("\nThe cm2 primitive chain composes: tensor load, transposing view,")
    err < 2e-2 && println("coopmat product, in-fragment row reduce, per-element exp, second product.")
end

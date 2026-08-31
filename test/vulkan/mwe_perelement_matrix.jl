# A per-element op whose extra operand is a MATRIX.
#
# `OpCooperativeMatrixPerElementOpNV` allows a cooperative matrix in the operand
# list, and the callback is then handed that matrix's CORRESPONDING ELEMENT.
# `flash_attn_cm2.comp` uses it for `coopMatPerElementNV(M, rowmax, Max, Mold)` —
# the elementwise maximum of two matrices in ONE pass.
#
# Why it is worth a capability: an ablation of our flash kernel puts **70% of its
# time in the softmax's per-element passes**, for about one eightieth of the
# arithmetic. Without a matrix operand, `max(a, b)` is three passes (negate,
# relu, add) and `exp(S - M)` is two.
#
# The three-pass identity is computed alongside as the CONTROL: both must agree,
# and it is the formulation the flash kernel uses today, so this file is also the
# proof that replacing it changes nothing numerically.
#
#     julia --project=. dev/Lava/test/mwe_perelement_matrix.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix
const T = 16

# Top-level, not closures, and not `@noinline`: the thunk is what the instruction
# names and these should melt into it.
pmax2(::UInt32, ::UInt32, a::Float32, b::Float32) = max(a, b)
pneg(::UInt32, ::UInt32, e::Float32) = -e
pnrelu(::UInt32, ::UInt32, e::Float32) = max(-e, 0.0f0)

lay(nrow, ncol) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(ncol), Int32(nrow))),
        (Int32(0), Int32(0)), (Int32(ncol), Int32(nrow)))

@kernel cpu = false unsafe_indices = true function maxpair!(one, three, @Const(A), @Const(B))
    z() = Lava.coopmat_zero(AMg{Float32,T,T,Lava.Accumulator})
    ma = Lava.tensor_load(z(), UInt64(pointer(A)), lay(T, T))
    mb = Lava.tensor_load(z(), UInt64(pointer(B)), lay(T, T))

    # ONE pass, the matrix as the extra operand.
    m1 = Lava.coopmat_perelement(pmax2, ma, mb)

    # THREE passes, the identity the flash kernel uses today:
    #   max(a, b) = a + relu(b - a), anchored on `a`.
    d = Lava.coopmat_add(mb, Lava.coopmat_perelement(pneg, ma))
    m3 = Lava.coopmat_add(ma, Lava.coopmat_perelement(pnrelu,
                                Lava.coopmat_perelement(pneg, d)))

    Mantle.copyto!(pointer(one), 1, T, m1)
    Mantle.copyto!(pointer(three), 1, T, m3)
end

ctx = MVE.vk_context()
if !ctx.coopmat2.per_element_operations
    @info "no coopmat2 per-element operations on this device — nothing to run"
else
    back = LavaBackend()
    a = Float32.(reshape(sin.(range(0, 9, T * T)), T, T))
    b = Float32.(reshape(cos.(range(0, 7, T * T)), T, T))
    A = KA.allocate(back, Float32, T, T); copyto!(A, a)
    B = KA.allocate(back, Float32, T, T); copyto!(B, b)
    O1 = KA.allocate(back, Float32, T, T); fill!(O1, Float32(NaN))
    O3 = KA.allocate(back, Float32, T, T); fill!(O3, Float32(NaN))

    maxpair!(back, 32)(O1, O3, A, B; ndrange = 32)
    KA.synchronize(back)
    g1, g3 = Array(O1), Array(O3)
    want = max.(a', b')                       # the loads transpose; see the gemm MWE

    e1 = maximum(abs.(g1 .- want))
    e3 = maximum(abs.(g3 .- want))
    @printf("one pass  (matrix operand)  max abs err %.3e  %s\n", e1, e1 < 1e-6 ? "OK" : "MISMATCH")
    @printf("three pass (the identity)   max abs err %.3e  %s\n", e3, e3 < 1e-6 ? "OK" : "CONTROL FAILED")
    @printf("the two agree to            %.3e\n", maximum(abs.(g1 .- g3)))
    e1 < 1e-6 && e3 < 1e-6 &&
        println("\nA matrix may be an operand: max(a,b) is one pass where the identity is three.")
end

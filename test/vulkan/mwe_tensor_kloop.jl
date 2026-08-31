# The smallest GEMM with a real K-loop through tensor addressing.
#
# Purpose is not speed — it is to pin the SLICE OFFSET ORDER, which nothing so
# far has been able to: every earlier test used a symmetric offset `(off, off)`
# on a square tensor, so "dimension 0 is the array's column" and "dimension 0 is
# the array's row" are indistinguishable. A K-loop walks one axis and not the
# other, so it separates them.
#
# Computes Ct = B' * A' one 16x16 tile at a time, accumulating over K, because
# `muladd(load(P), load(Q)) == P' * Q'` and there is no row-major store to
# global — so the natural output of this path is the TRANSPOSE of A*B.
#
#     julia --project=. dev/Lava/test/mwe_tensor_kloop.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMk = Lava.AcceleratedMatrix

const TK = 16
const MK = 16          # A is MK x KK, B is KK x NK
const NK = 16
const KK = 64          # four k-steps — enough that a wrong axis walks off

# `which` selects the offset convention so both can be measured rather than
# argued about: 1 = (koff, 0) on both operands, 2 = (0, koff).
#
# It is a `Val`, NOT a runtime Int, and that is a finding rather than a style
# choice: a runtime branch makes an `OpPhi` over the two `tensor_slice` results,
# and a tensor LAYOUT handle is an i32 in LLVM whose SPIR-V type is opaque. The
# phi emitter derives its result type from the LLVM type and produces
#
#     OpPhi's result type '%uint' does not match incoming value type
#
# Cooperative matrices avoid this via `state.coopmat_value_types`; nothing
# equivalent exists for layouts. A GEMM that selects a layout across a branch
# would hit it — see the note in compiler/spirv/coopmat.jl.
@kernel cpu = false function kloop_kernel!(out, @Const(A), @Const(B), ::Val{which}) where {which}
    accm = Lava.coopmat_zero(AMk{Float32,TK,TK,Lava.Accumulator})
    zb = Lava.coopmat_zero(AMk{Float16,TK,TK,Lava.MatrixA})
    za = Lava.coopmat_zero(AMk{Float16,TK,TK,Lava.MatrixB})
    # B is (KK, NK) column-major -> tensor dims reversed = (NK, KK)
    lb = Lava.tensor_setdim(
             Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
             (Int32(NK), Int32(KK)))
    # A is (MK, KK) column-major -> dims (KK, MK)
    la = Lava.tensor_setdim(
             Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
             (Int32(KK), Int32(MK)))
    for k in 0:((KK ÷ TK) - 1)
        ko = Int32(k * TK)
        sb = which == 1 ? Lava.tensor_slice(lb, (ko, Int32(0)), (Int32(TK), Int32(TK))) :
                          Lava.tensor_slice(lb, (Int32(0), ko), (Int32(TK), Int32(TK)))
        sa = which == 1 ? Lava.tensor_slice(la, (Int32(0), ko), (Int32(TK), Int32(TK))) :
                          Lava.tensor_slice(la, (ko, Int32(0)), (Int32(TK), Int32(TK)))
        bt = Lava.tensor_load(zb, UInt64(pointer(B)), sb)
        at = Lava.tensor_load(za, UInt64(pointer(A)), sa)
        accm = Lava.coopmat_muladd(bt, at, accm)
    end
    Mantle.copyto!(pointer(out), 1, TK, accm)
end

ctx = MVE.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device"
else
    back = LavaBackend()
    WG = MVE.device_subgroup_size(ctx)
    a = Float16.(reshape(1:(MK * KK), MK, KK) ./ 512)
    b = Float16.(reshape((KK * NK):-1:1, KK, NK) ./ 512)
    A = KA.allocate(back, Float16, MK, KK); copyto!(A, a)
    B = KA.allocate(back, Float16, KK, NK); copyto!(B, b)
    out = KA.allocate(back, Float32, TK, TK)

    ref = permutedims(Float32.(a) * Float32.(b))     # Ct = (A*B)'
    for which in (1, 2)
        fill!(out, NaN32)
        kloop_kernel!(back, (Int(WG),))(out, A, B, Val(which); ndrange = (Int(WG),))
        KA.synchronize(back)
        o = Array(out)
        rel = maximum(abs.(o .- ref)) / maximum(abs.(ref))
        @printf("offset convention %d: max rel err %.3e  %s\n", which, rel,
                rel < 1e-2 ? "<== MATCHES (A*B)'" : "")
    end
    println("\nExactly one convention should match. If both do, the test is symmetric")
    println("somewhere and pins nothing; if neither, the operand roles are also wrong.")
end

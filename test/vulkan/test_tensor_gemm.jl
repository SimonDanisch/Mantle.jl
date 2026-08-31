using Test, Lava, KernelAbstractions, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

# What does a product of two TENSOR-LOADED operands actually compute?
#
# `test_tensor_load.jl` shows a load returns the right elements and
# `mwe_tensor_orientation.jl` pins which layout dimension is which. Neither
# answers this, because the load hands back the TRANSPOSE of the block it
# addressed — so the product is some transpose of `A*B`, and a GEMM cannot be
# routed through tensor addressing until it is known which.
#
# MEASURED, exactly:
#
#     coopmat_muladd(tensor_load(P), tensor_load(Q))  ==  P' * Q'
#
# bit-exact for fp16 operands accumulated in fp32. So to compute `A*B` through
# this path the operands must be handed over already transposed — or, equivalently,
# `muladd(load(B), load(A)) == (A*B)'`.
#
# The three rejected candidates are asserted too, and that is the point rather
# than thoroughness: all four produce finite, fully-written output, so a test
# that only checked "did it compute something sane" would pass on every one of
# them. Only equality against ONE reference and inequality against the others
# pins the orientation.
const TGt = 16          # one tile, one subgroup — the smallest thing that multiplies

@kernel cpu = false function tensorgemm_kernel!(out, @Const(A), @Const(B))
    # Dims REVERSED relative to the Julia array: the tensor's last dimension is
    # fastest-varying, Julia's first is. See `mwe_tensor_orientation.jl`.
    lay = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(TGt), Int32(TGt))),
            (Int32(0), Int32(0)), (Int32(TGt), Int32(TGt)))
    za = Lava.coopmat_zero(AMg{Float16,TGt,TGt,Lava.MatrixA})
    zb = Lava.coopmat_zero(AMg{Float16,TGt,TGt,Lava.MatrixB})
    ma = Lava.tensor_load(za, UInt64(pointer(A)), lay)
    mb = Lava.tensor_load(zb, UInt64(pointer(B)), lay)
    acc = Lava.coopmat_zero(AMg{Float32,TGt,TGt,Lava.Accumulator})
    Mantle.copyto!(pointer(out), 1, TGt, Lava.coopmat_muladd(ma, mb, acc))
end

@testset "a product of tensor-loaded operands is P' * Q'" begin
    ctx = MVE.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        WG = MVE.device_subgroup_size(ctx)

        A = KA.allocate(back, Float16, TGt, TGt)
        B = KA.allocate(back, Float16, TGt, TGt)
        # Asymmetric, non-constant operands: a symmetric one would let a
        # transposed reading match by accident, which is the whole failure mode.
        a = Float16.(reshape(1:(TGt * TGt), TGt, TGt) ./ 32)
        b = Float16.(reshape((TGt * TGt):-1:1, TGt, TGt) ./ 64)
        copyto!(A, a); copyto!(B, b)
        out = KA.allocate(back, Float32, TGt, TGt)
        fill!(out, -1.0f0)

        tensorgemm_kernel!(back, (Int(WG),))(out, A, B; ndrange = (Int(WG),))
        KA.synchronize(back)
        o = Array(out)
        af, bf = Float32.(a), Float32.(b)

        @test all(isfinite, o)
        @test !any(==(-1.0f0), o)              # every element written

        # Exact, not approximate: these magnitudes accumulate in fp32 without
        # rounding, and an exact assertion cannot be satisfied by a near-miss.
        @test o == transpose(af) * transpose(bf)

        # The negative control, and it is the reason this file exists. Each of
        # these is finite and fully written; only the comparison separates them.
        @test o != af * bf
        @test o != permutedims(af * bf)
        @test o != bf * af
    end
end

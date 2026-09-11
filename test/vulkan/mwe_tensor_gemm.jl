# The first PRODUCT through tensor addressing: does a matrix loaded by
# `OpCooperativeMatrixLoadTensorNV` multiply correctly?
#
# `test_tensor_load.jl` proves a load returns the right elements and
# `mwe_tensor_orientation.jl` pins which layout dimension is which. Neither says
# what `coopmat_muladd` then computes, because the load hands back the TRANSPOSE
# of the block it addressed — so the product of two tensor-loaded operands is
# some transpose of A*B, and which one is the question a GEMM has to answer.
#
# This does not assume an answer. It computes one product on the device and
# compares it against ALL FOUR candidate references, printing which match. A test
# that accepted "any of them" would pin nothing; the point is to find out which,
# and then encode that.
#
#     julia --project=. dev/Lava/test/mwe_tensor_gemm.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const TG = 16          # one tile, one subgroup — the smallest thing that multiplies

# A and B are TG x TG column-major. The layout gets its dims REVERSED, which is
# what `mwe_tensor_orientation.jl` measured: for a column-major (R, C) array the
# tensor is (C, R), because the tensor's LAST dimension is fastest-varying and
# Julia's FIRST is.
@kernel cpu = false function tgemm_kernel!(out, @Const(A), @Const(B))
    la = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(TG), Int32(TG))),
            (Int32(0), Int32(0)), (Int32(TG), Int32(TG)))
    za = Lava.coopmat_zero(AMg{Float16,TG,TG,Lava.MatrixA})
    zb = Lava.coopmat_zero(AMg{Float16,TG,TG,Lava.MatrixB})
    ma = Lava.tensor_load(za, UInt64(pointer(A)), la)
    mb = Lava.tensor_load(zb, UInt64(pointer(B)), la)
    acc = Lava.coopmat_zero(AMg{Float32,TG,TG,Lava.Accumulator})
    r = Lava.coopmat_muladd(ma, mb, acc)
    Mantle.copyto!(pointer(out), 1, TG, r)
end

ctx = Mantle.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    WG = Mantle.device_subgroup_size(ctx)

    A = KA.allocate(back, Float16, TG, TG)
    B = KA.allocate(back, Float16, TG, TG)
    # Values that make every candidate distinguishable: a symmetric or constant
    # operand would let a transposed reading match by accident.
    a = Float16.(reshape(1:(TG * TG), TG, TG) ./ 32)
    b = Float16.(reshape((TG * TG):-1:1, TG, TG) ./ 64)
    copyto!(A, a); copyto!(B, b)
    out = KA.allocate(back, Float32, TG, TG)
    fill!(out, -1.0f0)

    tgemm_kernel!(back, (Int(WG),))(out, A, B; ndrange = (Int(WG),))
    KA.synchronize(back)
    o = Array(out)

    af, bf = Float32.(a), Float32.(b)
    cands = ("A*B" => af * bf, "(A*B)'" => permutedims(af * bf),
             "A'*B'" => transpose(af) * transpose(bf), "B*A" => bf * af)
    @printf("all finite = %s   any unwritten = %s\n",
            all(isfinite, o), any(==(-1.0f0), o))
    for (name, ref) in cands
        # fp16 operands accumulated in fp32: compare with a tolerance scaled to
        # the magnitudes involved, not exactly.
        rel = maximum(abs.(o .- ref)) / max(1f-6, maximum(abs.(ref)))
        @printf("  %-8s max rel err %.3e   %s\n", name, rel, rel < 1e-2 ? "<== MATCH" : "")
    end
    println("\nWhichever matches is the arrangement a GEMM must use; the others read")
    println("a differently-oriented operand and still produce finite numbers.")
end

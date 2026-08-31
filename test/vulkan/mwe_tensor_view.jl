# Does a transposing `TensorView` read an operand in place?
#
# `mwe_tensor_gemm_nonsquare.jl` pins the plain arrangement: a logical `M x K`
# operand must live in memory as `K x M`. That is a problem for attention,
# because with `(E, L, H, B)` slabs BOTH Q and K sit in memory as `(E, L)` — so
# whichever one is the B operand is the wrong way round, and the reference
# (`flash_attn_cm2.comp`) solves it with `tensorViewNV<2, false, 1, 0>` rather
# than by staging a transposed copy.
#
# NON-SQUARE on purpose. A transpose is precisely the thing a square case cannot
# tell you anything about, and this project has already shipped a store whose
# transpose "only a non-square shape reveals".
#
#     julia --project=. dev/Lava/test/mwe_tensor_view.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const R = 16      # tile rows
const C = 32      # tile cols — different, so a transpose cannot hide

lay(nrow, ncol) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(ncol), Int32(nrow))),
        (Int32(0), Int32(0)), (Int32(ncol), Int32(nrow)))

# Plain load: memory (C, R) comes back as an R x C matrix.
@kernel cpu = false function tv_plain!(out, @Const(A))
    z = Lava.coopmat_zero(AMg{Float32,R,C,Lava.Accumulator})
    m = Lava.tensor_load(z, UInt64(pointer(A)), lay(R, C))
    Mantle.copyto!(pointer(out), 1, R, m)
end

# Through a transposing view: memory (R, C) should ALSO come back R x C.
@kernel cpu = false function tv_transposed!(out, @Const(A))
    z = Lava.coopmat_zero(AMg{Float32,R,C,Lava.Accumulator})
    v = Lava.tensor_view(Val(2), Val((1, 0)))
    m = Lava.tensor_load(z, UInt64(pointer(A)), lay(R, C), v)
    Mantle.copyto!(pointer(out), 1, R, m)
end

ctx = MVE.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    want = Float32.(reshape(1:(R * C), R, C) ./ 64)      # the R x C matrix we want back

    # plain: store it transposed, as the pinned mapping requires
    At = KA.allocate(back, Float32, C, R); copyto!(At, permutedims(want))
    o1 = KA.allocate(back, Float32, R, C); fill!(o1, Float32(NaN))
    tv_plain!(back, 32)(o1, At; ndrange = 32)

    # view: store it the natural way round and let the view do the transpose
    An = KA.allocate(back, Float32, R, C); copyto!(An, want)
    o2 = KA.allocate(back, Float32, R, C); fill!(o2, Float32(NaN))
    tv_transposed!(back, 32)(o2, An; ndrange = 32)
    KA.synchronize(back)

    g1, g2 = Array(o1), Array(o2)
    e1 = maximum(abs.(g1 .- want)) / maximum(abs.(want))
    e2 = maximum(abs.(g2 .- want)) / maximum(abs.(want))
    @printf("plain load (memory %s)      max rel err %.3e %s\n",
            string(size(At)), e1, e1 < 1e-2 ? "OK" : "MISMATCH")
    @printf("view load  (memory %s)      max rel err %.3e %s\n",
            string(size(An)), e2, e2 < 1e-2 ? "OK" : "MISMATCH")
    println(all(isfinite, g2) ? "" : "  view result has non-finite entries")
    println("\nIf the view row is OK, K can be read in place and the flash port")
    println("needs no transposed staging copy.")
end

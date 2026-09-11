# Storing a matrix THROUGH A TRANSPOSING VIEW — the last primitive a
# tensor-addressed GEMM needs.
#
# `mul_mm_cm2.comp` ends with
#
#     coopMatStoreTensorNV(mat_d, data_d, pos_d,
#                          sliceTensorLayoutNV(tensorLayoutD, ...), tensorViewTranspose)
#
# because its accumulator is `M x N` while the destination is laid out `N x M`.
# Without a viewed store that transpose has to be a staging pass, which is most
# of what the tensor-addressed path exists to remove.
#
# NON-SQUARE, and that is the whole design of this file: a `32x16` matrix stored
# plainly lands in a `(16, 32)` array and stored through the view in a `(32, 16)`
# one, so the two cannot be confused and a wrong orientation cannot accidentally
# match. The plain store is the control — if it fails too, the fault is the test.
#
#     julia --project=. dev/Lava/test/mwe_tensor_storeview.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const R = 32       # rows of the matrix
const C = 16       # columns of the matrix

"2-D clamping layout over a column-major (nrow, ncol) array; the tensor's dims
are the REVERSE, its last dimension being the fastest-varying one."
lay(nrow, ncol) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(ncol), Int32(nrow))),
        (Int32(0), Int32(0)), (Int32(ncol), Int32(nrow)))

@kernel cpu = false unsafe_indices = true function storeview!(plain, viewed, @Const(A))
    # An `R x C` matrix lives in a `(C, R)` array — the pinned mapping.
    m = Lava.tensor_load(Lava.coopmat_zero(AMg{Float32,R,C,Lava.Accumulator}),
                         UInt64(pointer(A)), lay(C, R))
    Lava.tensor_store(m, UInt64(pointer(plain)), lay(C, R))
    vt = Lava.tensor_view(Val(2), Val((1, 0)))
    Lava.tensor_store(m, UInt64(pointer(viewed)), lay(R, C), vt)
end

ctx = Mantle.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    a = Float32.(reshape(1:(R * C), C, R) ./ 7)          # (C, R): the matrix is a'
    A = KA.allocate(back, Float32, C, R); copyto!(A, a)
    P = KA.allocate(back, Float32, C, R); fill!(P, Float32(NaN))
    V = KA.allocate(back, Float32, R, C); fill!(V, Float32(NaN))

    storeview!(back, 32)(P, V, A; ndrange = 32)
    KA.synchronize(back)
    gp, gv = Array(P), Array(V)

    ep = maximum(abs.(gp .- a)) / maximum(abs.(a))
    ev = maximum(abs.(gv .- permutedims(a))) / maximum(abs.(a))
    @printf("plain store  -> %s   max rel err %.3e  %s\n", string(size(gp)), ep,
            ep < 1e-6 ? "OK (control)" : "MISMATCH — the test is wrong, not the view")
    @printf("viewed store -> %s   max rel err %.3e  %s\n", string(size(gv)), ev,
            ev < 1e-6 ? "OK" : "MISMATCH")
    ep < 1e-6 && ev < 1e-6 &&
        println("\nA GEMM can now write an N x M destination from an M x N accumulator " *
                "without a staging pass.")
end

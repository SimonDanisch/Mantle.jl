# A tensor layout over a slab of a LARGER array — `OpTensorLayoutSetStrideNV`.
#
# Without a stride a layout describes a packed tensor: the driver takes the
# distance between successive rows to be the row length it was given. That is
# true of a whole dense array and false of a window into one, which is what
# attention's `q`, `k` and `v` are — a permuted view of a single packed
# `(E, L, H, B)` block, where consecutive `L` are `stride(q, 2)` apart.
#
# The run WITHOUT the stride is kept and expected to MISMATCH. It is the negative
# control: a packed test array would pass either way and prove nothing about the
# instruction, which is the trap `mwe-must-match-the-real-types` records.
#
#     julia --project=. dev/Lava/test/mwe_tensor_stride.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const EB = 24      # rows the array actually has
const E  = 16      # rows the matrix wants — a WINDOW, so the slab is not packed
const L  = 32      # columns of the array = rows of the matrix

@kernel cpu = false unsafe_indices = true function tload!(out, @Const(A),
                                                          ::Val{USE}) where {USE}
    l = Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT))
    l = Lava.tensor_setdim(l, (Int32(L), Int32(E)))
    # The only difference between the two runs. Strides are innermost-last, in
    # elements: successive `L` are `EB` apart, successive `E` are adjacent.
    l = USE ? Lava.tensor_setstride(l, (Int32(EB), Int32(1))) : l
    l = Lava.tensor_slice(l, (Int32(0), Int32(0)), (Int32(L), Int32(E)))
    m = Lava.tensor_load(Lava.coopmat_zero(AMg{Float32,L,E,Lava.Accumulator}),
                         UInt64(pointer(A)), l)
    Mantle.copyto!(pointer(out), 1, L, m)
end

ctx = Mantle.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    a = Float32.(reshape(1:(EB * L), EB, L))
    A = KA.allocate(back, Float32, EB, L); copyto!(A, a)
    want = a[1:E, :]'                                  # L x E, the window transposed
    for use in (false, true)
        out = KA.allocate(back, Float32, L, E); fill!(out, Float32(NaN))
        tload!(back, 32)(out, A, Val(use); ndrange = 32)
        KA.synchronize(back)
        got = Array(out)
        e = maximum(abs.(got .- want)) / maximum(abs.(want))
        match = isfinite(e) && e < 1e-6
        @printf("setstride %-5s  max rel err %.3e  %s\n", use, e,
                use ? (match ? "OK" : "MISMATCH — the instruction is wrong") :
                      (match ? "UNEXPECTED MATCH — this test proves nothing" :
                               "mismatches, as it must (negative control)"))
    end
end

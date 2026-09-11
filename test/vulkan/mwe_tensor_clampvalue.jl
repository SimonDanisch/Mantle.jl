# What does `OpTensorLayoutSetClampValueNV` actually fill with?
#
# A `TENSOR_CLAMP_CONSTANT` load substitutes a constant outside the tensor, and
# that constant defaults to zero. It can be set — but the GLSL signature is
# `setTensorLayoutClampValueNV(layout, uint)` for a matrix of ANY component type,
# and it does not say how the 32 bits relate to an fp16 element. Two readings are
# equally plausible from the signature alone:
#
#   BITS      the operand is the element's bit pattern zero-extended, so filling
#             an fp16 matrix with 1.0 means passing 0x3C00
#   NUMERIC   the operand is converted, so filling with 1.0 means passing 1
#
# They are not confusable in the result: under BITS, passing 1 fills with 6.0e-8
# (the smallest fp16 subnormal), and under NUMERIC, passing 0x3C00 fills with
# 15360. So one run of each settles it, and `tensor_clampbits` is the answer in
# the form callers should use.
#
# This matters because ONE is the useful fill. Attention loads `V` as `Bc x EP`
# out of a slab only `E` wide; with a fill of one, the padding columns of
# `P x V` come out as the row sum of `P` — a per-key-block `coopmat_reduce`
# removed rather than made cheaper.
#
#     julia --project=. dev/Lava/test/mwe_tensor_clampvalue.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const E = 16       # rows/cols the matrix wants
const EA = 10      # rows the array actually has — 6 rows of every load are FILL

@kernel cpu = false unsafe_indices = true function cload!(out, @Const(A),
                                                          v::Int32, ::Val{SET}) where {SET}
    l = Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT))
    l = Lava.tensor_setdim(l, (Int32(E), Int32(EA)))
    l = Lava.tensor_setstride(l, (Int32(EA), Int32(1)))
    # The only difference between the runs. Unset, the fill is the documented
    # default of zero — which is also the negative control: if the fill never
    # changed, every arm would report zero and the instruction would be inert.
    l = SET ? Lava.tensor_setclampvalue(l, v) : l
    l = Lava.tensor_slice(l, (Int32(0), Int32(0)), (Int32(E), Int32(E)))
    m = Lava.tensor_load(Lava.coopmat_zero(AMg{Float32,E,E,Lava.Accumulator}),
                         UInt64(pointer(A)), l)
    Mantle.copyto!(pointer(out), 1, E, m)
end

ctx = Mantle.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device — nothing to run"
else
    back = LavaBackend()
    a = Float32.(reshape(1:(EA * E), EA, E))
    A = KA.allocate(back, Float32, EA, E); copyto!(A, a)
    # The fill occupies columns EA+1:E of the loaded (E x E) matrix — the part of
    # the slice past the array's own extent.
    fillof(got) = got[:, (EA + 1):E]
    runs = [("default (unset)",     Int32(0),                                false),
            ("bits of 1.0f0",       Lava.tensor_clampbits(1.0f0, Float32),   true),
            ("numeric 1",           Int32(1),                                true),
            ("bits of 2.5f0",       Lava.tensor_clampbits(2.5f0, Float32),   true)]
    for (name, v, set) in runs
        o = KA.allocate(back, Float32, E, E); fill!(o, Float32(NaN))
        cload!(back, 32)(o, A, v, Val(set); ndrange = 32)
        KA.synchronize(back)
        got = Array(o)
        f = fillof(got)
        inrange = maximum(abs.(got[:, 1:EA] .- a')) # the real elements must survive
        @printf("%-18s operand 0x%08x  fill = %-12g  in-range err %.2e\n",
                name, reinterpret(UInt32, v), first(f), inrange)
        all(==(first(f)), f) || println("    ^ fill is NOT uniform — the read is wrong")
    end
    println("\nBITS reading is confirmed iff `bits of 1.0f0` fills with 1 and",
            " `numeric 1` fills with ~1.4e-45.")
end

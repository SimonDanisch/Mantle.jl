using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMc = Lava.AcceleratedMatrix

# A CLAMPING layout makes an out-of-range block legal, and this is the claim the
# whole tensor-addressing port rests on.
#
# `gemm_padn` pads N 1500 -> 1536, `GEMM_BLOCK` rounds M, `padtile`/`crsextent`
# round K, and `gemm_divides` declines shapes outright — every one of them exists
# because `OpCooperativeMatrixLoadKHR` has no bounds checking, so extents must be
# arranged to divide the tile. Under `TENSOR_CLAMP_CONSTANT` the driver
# bounds-checks the load itself.
#
# MEASURED, on a 17x17 tensor (an extent that divides nothing) read with a 16x16
# tile at offsets that run off the end:
#
#     offset 0   0 out-of-range    in-range correct
#     offset 4  87 out-of-range    in-range correct   out-of-range == 0.0 exactly
#     offset 8 175 out-of-range    in-range correct   out-of-range == 0.0 exactly
#
# No fault, no garbage, exact zeros — and a zero contributes nothing to a dot
# product, so an unpadded extent Just Works for a GEMM.
#
# THE OBJECT IS NOT WHAT OUT-OF-RANGE ELEMENTS KEEP. An earlier note here said it
# was. The `%object` operand is loaded full of a sentinel below precisely so the
# two answers are distinguishable, and the out-of-range elements come back 0, not
# the sentinel. Zeroing the object would have made this untestable — zero is also
# the plausible clamp constant.
const EXT_C = 17
const TC = 16
const SENT = -999.0f0

@kernel cpu = false function tensorclamp_kernel!(out, @Const(src), @Const(sentinel), off::Int32)
    lay = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT_C), Int32(EXT_C))),
            (off, off), (Int32(TC), Int32(TC)))
    obj = AMc{Float32,TC,TC,Lava.Accumulator}(pointer(sentinel), 1, TC)
    Mantle.copyto!(pointer(out), 1, TC,
                 Lava.tensor_load(obj, UInt64(pointer(src)), lay))
end

@testset "a clamping layout makes an unpadded extent legal" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        WG = Mantle.device_subgroup_size(ctx)
        src = KA.allocate(back, Float32, EXT_C, EXT_C)
        s = Float32.(reshape(1:(EXT_C * EXT_C), EXT_C, EXT_C))
        copyto!(src, s)
        sentinel = KA.allocate(back, Float32, TC, TC); fill!(sentinel, SENT)
        out = KA.allocate(back, Float32, TC, TC)

        for off in Int32.((0, 4, 8))
            fill!(out, NaN32)
            tensorclamp_kernel!(back, (Int(WG),))(out, src, sentinel, off;
                                                  ndrange = (Int(WG),))
            KA.synchronize(back)
            o = Array(out)
            # The load transposes (dims are reversed for a column-major array),
            # so tile (i,j) comes from s[off+j, off+i].
            inr = [(off + j <= EXT_C && off + i <= EXT_C) for i in 1:TC, j in 1:TC]
            @test all(o[i, j] == s[off + j, off + i]
                      for i in 1:TC, j in 1:TC if inr[i, j])
            oo = [o[i, j] for i in 1:TC, j in 1:TC if !inr[i, j]]
            if !isempty(oo)
                @test all(==(0.0f0), oo)      # the clamp constant
                @test !any(==(SENT), oo)      # NOT the object's value
                @test !any(isnan, oo)         # and it was written at all
            end
        end
    end
end


# ── The clamp CONSTANT is settable, and the operand is BITS ──────────────────
#
# `OpTensorLayoutSetClampValueNV` takes a 32-bit integer for a matrix of any
# component type, and the signature alone does not say whether those bits are
# the element's representation or a number to convert. It is the representation,
# which is why callers go through `tensor_clampbits` — passing `1` for an fp16
# matrix fills it with 6.0e-8, not 1.0, and a kernel reading that fill would get
# an answer wrong by a factor of 1.6e7 with nothing anywhere reporting an error.
#
# The `numeric 1` arm below is the negative control. Without it the test could
# not tell the BITS reading from a driver that ignores the instruction: both
# would leave `bits of 1.0f0` filling with 1.0 only by coincidence of the
# default being zero.
#
# A fill of ONE is the useful one: `attn_flash_cm2!` loads `V` as `Bc x EP` from
# a slab only `E` wide, and with a fill of one the padding columns of `P·V` come
# out as the row sum of `P` — a `coopmat_reduce` per key block deleted rather
# than made cheaper. See `OSUM` there.
const EA_V = 10        # rows the array has; the load asks for EXT_V and the rest is fill

@kernel cpu = false unsafe_indices = true function clampvalue_kernel!(out, @Const(A),
                                                                     v::Int32,
                                                                     ::Val{SET}) where {SET}
    l = Lava.tensor_setstride(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(TC), Int32(EA_V))),
            (Int32(EA_V), Int32(1)))
    l = SET ? Lava.tensor_setclampvalue(l, v) : l
    l = Lava.tensor_slice(l, (Int32(0), Int32(0)), (Int32(TC), Int32(TC)))
    Mantle.copyto!(pointer(out), 1, TC,
                 Lava.tensor_load(Lava.coopmat_zero(AMc{Float32,TC,TC,Lava.Accumulator}),
                                  UInt64(pointer(A)), l))
end

@testset "the clamp constant is settable, and it is a bit pattern" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        a = Float32.(reshape(1:(EA_V * TC), EA_V, TC))
        A = KA.allocate(back, Float32, EA_V, TC); copyto!(A, a)

        function fillvalue(v, set)
            out = KA.allocate(back, Float32, TC, TC); fill!(out, NaN32)
            clampvalue_kernel!(back, 32)(out, A, v, Val(set); ndrange = 32)
            KA.synchronize(back)
            got = Array(out)
            # The real elements have to survive whatever the fill is set to.
            @test maximum(abs, got[:, 1:EA_V] .- a') < 1e-6
            f = got[:, (EA_V + 1):TC]
            @test all(==(first(f)), f)        # uniform, or the read is wrong
            first(f)
        end

        @test fillvalue(Int32(0), false) == 0.0f0                       # documented default
        @test fillvalue(Lava.tensor_clampbits(1.0f0, Float32), true) == 1.0f0
        @test fillvalue(Lava.tensor_clampbits(2.5f0, Float32), true) == 2.5f0
        # NUMERIC would make this 1.0; BITS makes it the smallest subnormal.
        @test fillvalue(Int32(1), true) == reinterpret(Float32, UInt32(1))
        @test fillvalue(Int32(1), true) != 1.0f0
    end
end

@testset "tensor_clampbits is the element's representation" begin
    # Host-side and exact, so it holds on a machine with no such device — and it
    # is the half of the fp16 story the GPU arms above cannot show, since they
    # run in fp32.
    @test Lava.tensor_clampbits(1.0f0, Float16) == Int32(0x3C00)
    @test Lava.tensor_clampbits(1.0f0, Float32) == Int32(reinterpret(UInt32, 1.0f0))
    @test reinterpret(Float16, UInt16(Lava.tensor_clampbits(1.0f0, Float16))) === Float16(1)
    # The mistake the type parameter exists to prevent.
    @test reinterpret(Float16, UInt16(1)) !== Float16(1)
end

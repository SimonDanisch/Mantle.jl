# Does a CLAMPING layout make an out-of-range block legal?
#
# This is the claim the whole port rests on. `gemm_padn` pads N 1500 -> 1536, the
# `GEMM_BLOCK` pad rounds M, `padtile`/`crsextent` round K, and `gemm_divides`
# declines shapes outright — all of it exists because a cooperative-matrix load
# has no bounds checking, so the extents must be arranged to divide the tile.
# Under `gl_CooperativeMatrixClampModeConstantNV` the driver is supposed to
# bounds-check the load itself. If that works, every one of those goes away.
#
# So: a tensor whose extent divides NOTHING (17 x 17), and a 16x16 tile read at
# an offset that runs off the end. Two things to learn, and neither is guessable:
#
#   1. does it fault / corrupt, or is it defined?
#   2. what do the out-of-range elements contain — the clamp constant, or the
#      `%object` matrix's existing value?
#
# The object is loaded full of a SENTINEL rather than zeroed, because zero is
# also the plausible clamp constant and the two answers would be
# indistinguishable.
#
#     julia --project=. dev/Lava/test/mwe_tensor_clamp.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMc = Lava.AcceleratedMatrix

const EXT_C = 17          # divides neither 16 nor anything else
const TC = 16
const SENT = -999.0f0

@kernel cpu = false function clamp_kernel!(out, @Const(src), @Const(sentinel), off::Int32)
    lay = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT_C), Int32(EXT_C))),
            (off, off), (Int32(TC), Int32(TC)))
    # %object = a matrix full of SENT, so "kept the object" and "wrote a zero
    # constant" are distinguishable in the output.
    obj = AMc{Float32,TC,TC,Lava.Accumulator}(pointer(sentinel), 1, TC)
    m = Lava.tensor_load(obj, UInt64(pointer(src)), lay)
    Mantle.copyto!(pointer(out), 1, TC, m)
end

ctx = Mantle.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device"
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
        clamp_kernel!(back, (Int(WG),))(out, src, sentinel, off; ndrange = (Int(WG),))
        KA.synchronize(back)
        o = Array(out)
        # With dims reversed the load transposes, so element (i,j) of the tile
        # comes from s[off+j, off+i] when that is in range.
        inrange = [(off + j <= EXT_C && off + i <= EXT_C) for i in 1:TC, j in 1:TC]
        nout = count(!, inrange)
        ok = all(o[i, j] == s[off + j, off + i] for i in 1:TC, j in 1:TC if inrange[i, j])
        oo = nout == 0 ? Float32[] : [o[i, j] for i in 1:TC, j in 1:TC if !inrange[i, j]]
        @printf("offset %2d  out-of-range %3d  in-range correct=%s", off, nout, ok)
        if nout > 0
            u = unique(oo)
            @printf("  out-of-range values: %s%s", string(u[1:min(3, end)]),
                    length(u) > 3 ? " …" : "")
            @printf("  %s", all(==(SENT), oo) ? "(kept the object)" :
                            all(==(0.0f0), oo) ? "(clamp constant 0)" : "(SOMETHING ELSE)")
        end
        println()
    end
    println("\nIf the in-range half stays correct and the out-of-range half is DEFINED,")
    println("an unpadded extent is legal and the padding apparatus is unnecessary.")
end

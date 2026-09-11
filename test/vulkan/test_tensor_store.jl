using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMs = Lava.AcceleratedMatrix

# `OpCooperativeMatrixStoreTensorNV` — the mirror of the tensor load, and the
# half that makes a ragged OUTPUT legal.
#
# `test_tensor_clamp.jl` established that a clamping layout bounds-checks READS,
# so a GEMM can consume unpadded operands. That is only half a GEMM: the result
# still has to be written, and `OpCooperativeMatrixStoreKHR` has no bounds check
# either, so a destination tile straddling the edge of an M x N output runs past
# the end. This asserts the store clamps too.
#
# WHY THE TEST IS SHAPED THIS WAY. A store test that only checks "the in-range
# elements are right" passes even if the store wrote the whole 16x16 tile and
# trampled its neighbours — which is exactly the bug being guarded against. So
# the destination is allocated LARGER than the tensor and pre-filled with a
# sentinel, and the assertion is two-sided: in-range elements changed, and
# every byte outside the layout's extent is still the sentinel.
#
# Orientation follows `test_tensor_load.jl`: the tensor's LAST dimension is the
# fastest-varying, so a `(EXT, EXT)` layout over a column-major Julia array
# addresses the transpose of the leading block. That was measured and pinned
# there; it is assumed here rather than re-derived.
const EXT_S = 37          # divides nothing
const TS = 16
const SENT_S = -777.0f0

@kernel cpu = false function tensorstore_kernel!(dst, @Const(src), off::Int32)
    lay = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT_S), Int32(EXT_S))),
            (off, off), (Int32(TS), Int32(TS)))
    # A plain (non-tensor) load of a full in-range tile, so the VALUES are not
    # themselves in question — only where they land.
    m = AMs{Float32,TS,TS,Lava.Accumulator}(pointer(src), 1, TS)
    Lava.tensor_store(m, UInt64(pointer(dst)), lay)
end

@testset "OpCooperativeMatrixStoreTensorNV clamps writes" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        WG = Int(Mantle.device_subgroup_size(ctx))
        src = KA.allocate(back, Float32, TS, TS)
        copyto!(src, Float32.(reshape(1:(TS * TS), TS, TS)))

        # offset 0 is fully in range; 24 and 32 straddle the 37-element extent.
        for off in (0, 24, 32)
            dst = KA.allocate(back, Float32, EXT_S, EXT_S)
            fill!(dst, SENT_S)
            tensorstore_kernel!(back, (WG,))(dst, src, Int32(off); ndrange = (WG,))
            KA.synchronize(back)
            d = Array(dst)

            inrange = EXT_S - off                 # rows/cols the tile actually covers
            expected_written = min(TS, inrange)^2
            written = count(!=(SENT_S), d)

            @test written == expected_written
            # Nothing outside the tile's footprint moved. Written as a scan over
            # the whole array rather than a slice comparison, so a store that
            # wrapped around or wrote a wrong stride is caught too.
            for j in 1:EXT_S, i in 1:EXT_S
                covered = (off < i <= off + TS) && (off < j <= off + TS)
                covered || @test d[i, j] == SENT_S
            end
        end

        # And the fully in-range case must actually deliver the source values, or
        # everything above is satisfied by a store that writes nothing at all.
        dst = KA.allocate(back, Float32, EXT_S, EXT_S)
        fill!(dst, SENT_S)
        tensorstore_kernel!(back, (WG,))(dst, src, Int32(0); ndrange = (WG,))
        KA.synchronize(back)
        d = Array(dst)
        s = Array(src)
        @test count(!=(SENT_S), d) == TS * TS
        # Transposed, per the orientation note above.
        @test d[1:TS, 1:TS] == permutedims(s)
    end
end

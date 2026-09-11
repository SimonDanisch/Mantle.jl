# The cooperative-matrix GEMM is only valid on a 32-lane subgroup.
#
# The block kernel computes its subgroup index as `lane ÷ 32` (gemm.jl) and
# GEMM_WORKGROUP is sized as "2 subgroups" on the same assumption. Cooperative
# matrix operations are subgroup-scoped, so on a device with a different wave
# width those lanes do not form a subgroup: on wave64 hardware exactly HALF the
# output tile is written — bit-exact where written, zero elsewhere — which is a
# silently wrong answer, not a crash. It reproduced down to a single 16x16x16
# tile with nbatch=1.
#
# `coopmat_gemm_available()` therefore requires the device subgroup size to match,
# and `mul!` falls back to `gemmlaunch!`, which is correct at any wave width.
#
# Written against whatever the running device reports, so it is meaningful on
# wave32 hardware (where the coopmat path is taken) and wave64 (where it is not).

using Test, Lava, LinearAlgebra, KernelAbstractions
const KA = KernelAbstractions

@testset "coopmat GEMM requires a matching subgroup size" begin
    ctx = Mantle.vk_context()
    sg = Mantle.device_subgroup_size(ctx)
    @test sg > 0

    # The kernel may only be used where a subgroup really is GEMM_SUBGROUP wide:
    # either natively, or because the pipeline pinned it. Never on a wave width
    # that is neither.
    if Mantle.coopmat_gemm_available(ctx)
        @test sg == Mantle.GEMM_SUBGROUP || Mantle.can_require_subgroup_size(ctx, Mantle.GEMM_SUBGROUP)
    else
        @test !(sg == Mantle.GEMM_SUBGROUP || Mantle.can_require_subgroup_size(ctx, Mantle.GEMM_SUBGROUP)) ||
              !Mantle.coopmat_shape(ctx, Float16, Mantle.GEMM_TILE, Mantle.GEMM_TILE, Mantle.GEMM_TILE)
    end

    # A pinnable device must actually be pinned, and only for coopmat modules —
    # the whole guarantee rests on `get_compute_pipeline` recognising the
    # capability, so check the scanner in both directions against real SPIR-V.
    if Mantle.can_require_subgroup_size(ctx, Mantle.GEMM_SUBGROUP)
        c = Mantle.subgroup_size_control(ctx)
        @test c.compute
        @test c.min <= Mantle.GEMM_SUBGROUP <= c.max
    end
    # Hand-built modules so the parser is pinned exactly, with no compile cost.
    spv(caps...) = collect(reinterpret(UInt8, UInt32[
        0x07230203, 0x00010600, 0, 1, 0,                 # magic, version, gen, bound, schema
        # OpCapability is opcode 17, word count 2
        (vcat(([UInt32(2 << 16 | 17), UInt32(c)] for c in caps)...))...,
        UInt32(3 << 16 | 14), UInt32(0), UInt32(1),      # OpMemoryModel ends the section
    ]))
    @test Mantle.spirv_declares_capability(spv(Lava.Cap.CooperativeMatrixKHR),
                                          Lava.Cap.CooperativeMatrixKHR)
    @test Mantle.spirv_declares_capability(spv(UInt32(1), Lava.Cap.CooperativeMatrixKHR),
                                          Lava.Cap.CooperativeMatrixKHR)   # not just the first
    @test !Mantle.spirv_declares_capability(spv(UInt32(1)), Lava.Cap.CooperativeMatrixKHR)
    @test !Mantle.spirv_declares_capability(UInt8[], Lava.Cap.CooperativeMatrixKHR)
    @test !Mantle.spirv_declares_capability(UInt8[0x01, 0x02, 0x03],           # not word-aligned
                                          Lava.Cap.CooperativeMatrixKHR)
    bad = collect(spv(Lava.Cap.CooperativeMatrixKHR)); bad[1] = 0x00        # wrong magic
    @test !Mantle.spirv_declares_capability(bad, Lava.Cap.CooperativeMatrixKHR)

    # Whichever path `mul!` picks, the answer must be right — this is the case
    # that silently returned a half-written tile.
    M, N, K = 64, 64, 32
    hA = Float16.(reshape(sin.(range(0, 3, M * K)), M, K))
    hB = Float16.(reshape(cos.(range(0, 4, K * N)), K, N))
    A = Mantle.LavaArray(hA); B = Mantle.LavaArray(hB)
    C = Mantle.LavaArray(zeros(Float32, M, N))
    LinearAlgebra.mul!(C, A, B, 1.0f0, 0.0f0)
    KA.synchronize(LavaBackend())
    got = Array(C)
    want = Float32.(hA) * Float32.(hB)

    @test count(!iszero, got) == length(got)          # nothing left unwritten
    @test maximum(abs, got .- want) / maximum(abs, want) < 1e-2
end

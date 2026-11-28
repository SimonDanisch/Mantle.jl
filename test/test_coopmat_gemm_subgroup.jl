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
# So the GEMM is only offered where a subgroup really is 32 lanes wide — natively,
# or because the backend pins cooperative-matrix pipelines to 32 — and `mul!`
# falls back to a kernel that is correct at any wave width everywhere else.
#
# What a user sees if that guard breaks is a half-written product, so that is what
# is asserted, through `mul!`, on whatever path the device takes. The guard's
# parts are not asserted here: the pinned width is `caps(...).coopmatsubgroup`
# (`test_device_caps.jl`), and whether a pin is honoured is
# `test_subgroup_size_pinning.jl`.

using Test, LinearAlgebra, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "coopmat GEMM requires a matching subgroup size" begin
    @test Mantle.caps(TESTBACKEND).subgroup > 0

    # Whichever path `mul!` picks, the answer must be right — this is the case
    # that silently returned a half-written tile.
    M, N, K = 64, 64, 32
    hA = Float16.(reshape(sin.(range(0, 3, M * K)), M, K))
    hB = Float16.(reshape(cos.(range(0, 4, K * N)), K, N))
    A = Mantle.devicearray(TESTBACKEND, hA); B = Mantle.devicearray(TESTBACKEND, hB)
    C = Mantle.devicearray(TESTBACKEND, zeros(Float32, M, N))
    LinearAlgebra.mul!(C, A, B, 1.0f0, 0.0f0)
    KA.synchronize(TESTBACKEND)
    got = Array(C)
    want = Float32.(hA) * Float32.(hB)

    @test count(!iszero, got) == length(got)          # nothing left unwritten
    @test maximum(abs, got .- want) / maximum(abs, want) < 1e-2
end

# Moved from `test_coopmat_subgroup_pin.jl`: a real cooperative-matrix GEMM builds
# and computes correctly at the width its pipeline runs at. `mul!` on fp16
# operands with an fp32 destination is the shipped route onto `coopmat_gemm!`,
# whose kernels are written at the staged GEMM's tile.
@testset "coopmat GEMM is correct at the pinned width" begin
    if !hasstagedgemm()
        @info "no cooperative matrices at the staged GEMM's tile on this device; skipping"
    else
        M = N = K = 128
        A = Mantle.devicearray(TESTBACKEND, Float16.(randn(Float32, M, K) .* 0.1f0))
        B = Mantle.devicearray(TESTBACKEND, Float16.(randn(Float32, K, N) .* 0.1f0))
        C = Mantle.devicearray(TESTBACKEND, zeros(Float32, M, N))
        ref = Float32.(Array(A)) * Float32.(Array(B))
        LinearAlgebra.mul!(C, A, B)
        KA.synchronize(TESTBACKEND)
        got = Array(C)
        @test maximum(abs, got) > 1e-3                       # it wrote something
        @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3
    end
end

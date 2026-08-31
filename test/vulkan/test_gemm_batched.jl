"""
Batched cooperative-matrix GEMM.

`nbatch` runs `nbatch` independent `M x N = (M x K) * (K x N)` products from
contiguous slabs in one dispatch. It exists because attention needs one GEMM per
(head, batch): SAM 2's windowed attention is 128 of them per pass, and issuing
those separately would add ~11 500 dispatches to an encode that currently makes
1972 — the recording cost alone would swamp the tensor-core win.

Batching is a decomposition of the subgroup index rather than a second kernel,
so `nbatch = 1` must remain *exactly* the old arithmetic. The first testset is
what pins that.
"""

using Test, Lava, KernelAbstractions, LinearAlgebra
const KA = KernelAbstractions

"Reference: `nb` independent products, on the host."
function hostref(Ah, Bh, M, N, K, nb)
    C = zeros(Float32, M, N, nb)
    for b in 1:nb
        A = reshape(view(Ah, ((b-1)*M*K + 1):(b*M*K)), M, K)
        B = reshape(view(Bh, ((b-1)*K*N + 1):(b*K*N)), K, N)
        C[:, :, b] = Float32.(A) * Float32.(B)
    end
    C
end

function rungemm(M, N, K, nb)
    back = LavaBackend()
    Ah = Float16.(reshape(sin.(range(0, 3, M * K * nb)), M * K * nb))
    Bh = Float16.(reshape(cos.(range(0, 4, K * N * nb)), K * N * nb))
    A = KA.allocate(back, Float16, M * K * nb); copyto!(A, Ah)
    B = KA.allocate(back, Float16, K * N * nb); copyto!(B, Bh)
    C = KA.allocate(back, Float32, M * N * nb); fill!(C, 0.0f0)
    MVE.coopmat_gemm!(C, A, B, M, N, K; nbatch = nb)
    KA.synchronize(back)
    got = reshape(Array(C), M, N, nb)
    want = hostref(Ah, Bh, M, N, K, nb)
    err = maximum(abs, got .- want) / max(maximum(abs, want), eps(Float32))
    A = B = C = nothing; GC.gc()
    err
end

@testset "batched coopmat GEMM" begin
    if !MVE.coopmat_gemm_available()
        @info "skipping: the coopmat GEMM needs a $(MVE.GEMM_SUBGROUP)-lane subgroup and needs the device to report a $(MVE.GEMM_TILE)^3 Float16 shape; this device has subgroup=$(MVE.device_subgroup_size()), shape=$(MVE.coopmat_shape(MVE.vk_context(), Float16, MVE.GEMM_TILE, MVE.GEMM_TILE, MVE.GEMM_TILE))"
    else
        @testset "nbatch = 1 is unchanged" begin
            for (M, N, K) in ((64, 64, 32), (128, 256, 64), (256, 64, 80))
                @test rungemm(M, N, K, 1) < 1e-2
            end
        end

        @testset "batches are independent and correctly placed" begin
            for (M, N, K, nb) in ((64, 64, 32, 2), (64, 64, 32, 8),
                                  (128, 128, 80, 4),      # K = 80: E = 72 padded
                                  (256, 256, 80, 8))      # SAM 2 windowed, one head-batch group
                @test rungemm(M, N, K, nb) < 1e-2
            end
        end

        @testset "the shape heuristic accounts for the batch" begin
            # Not cosmetic: on the unbatched target, SAM 2's windowed attention
            # picks a 5-way split that runs at 1.2 TFLOP/s against 11.6 without
            # one — 1.14 ms versus 0.12 for the same product.
            @test MVE.coopmat_gemm_shape(256, 256, 80; nbatch = 128)[2] == 1
            @test MVE.coopmat_gemm_shape(256, 80, 256; nbatch = 128)[2] == 1
            # …and an unbatched call still splits, which is what the skinny
            # im2col convolutions depend on.
            @test MVE.coopmat_gemm_shape(128, 256, 2304)[2] > 1
            @test MVE.coopmat_gemm_shape(128, 256, 2304; nbatch = 1) ==
                  MVE.coopmat_gemm_shape(128, 256, 2304)
        end

        @testset "a wrong batch would not pass" begin
            # Guard against the reference and the kernel agreeing by accident:
            # batch 2 must differ from batch 1 for the test to have teeth.
            back = LavaBackend()
            M = N = K = 64; nb = 2
            Ah = Float16.(vcat(fill(1.0, M * K), fill(2.0, M * K)))
            Bh = Float16.(fill(1.0, K * N * nb))
            A = KA.allocate(back, Float16, M * K * nb); copyto!(A, Ah)
            B = KA.allocate(back, Float16, K * N * nb); copyto!(B, Bh)
            C = KA.allocate(back, Float32, M * N * nb); fill!(C, 0.0f0)
            MVE.coopmat_gemm!(C, A, B, M, N, K; nbatch = nb)
            KA.synchronize(back)
            got = reshape(Array(C), M, N, nb)
            @test all(≈(Float32(K)), got[:, :, 1])
            @test all(≈(Float32(2K)), got[:, :, 2])
            A = B = C = nothing; GC.gc()
        end
    end
end

"""
The Tensile kernel (`src/array/gemm_tensile.jl`): correct through the ordinary
`coopmat_gemm!` entry, and taken exactly when `tensile_gemm_ok` says so.

The shapes are chosen for the kernel's own edges rather than for speed: the
smallest K it accepts (two k steps, so the paired loop never runs), one and two
loop trips, and column-block counts that are not a multiple of the workgroup
group, so the last group of `tensile_tile` is narrow.

Every product pins `blk_split = (4, 1)`. Left to `coopmat_gemm_shape`, small
shapes plan a split K for the register-blocked kernel, which the Tensile kernel
(walking the whole of K, like the staged family) declines — so a test of small
shapes without the pin passes on whatever kernel ran instead, which is why each
case also asserts which kernel that was.
"""

const PLAN = (; blk_split = (4, 1))

using Test, KernelAbstractions
const KA = KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))

"Relative error of `coopmat_gemm!` against a Float32 product, the epilogue applied after rounding to `T`."
function tensile_error(M, N, K; T = Float16, bias = nothing, epilogue = identity)
    hA = rand(Float16, M, K) .- Float16(0.5)
    hB = rand(Float16, K, N) .- Float16(0.5)
    A = Mantle.devicearray(TESTBACKEND, hA)
    B = Mantle.devicearray(TESTBACKEND, hB)
    C = KA.allocate(TESTBACKEND, T, M, N)
    fill!(C, zero(T))
    dbias = bias === nothing ? nothing : Mantle.devicearray(TESTBACKEND, bias)
    gemmkernel(C, A, B, M, N, K; bias = dbias, epilogue) === Mantle.coopmat_gemm_tensile_kernel! ||
        error("M=$M N=$N K=$K did not take the Tensile kernel")
    Mantle.coopmat_gemm!(C, A, B, M, N, K; bias = dbias, epilogue, PLAN...)
    KA.synchronize(TESTBACKEND)
    ref = Float32.(hA) * Float32.(hB)
    bias === nothing || (ref .+= Float32.(bias))
    ref = Float32.(epilogue.(T.(ref)))
    return maximum(abs.(Float32.(Array(C)) .- ref)) / maximum(abs.(ref))
end

"The kernel `coopmat_gemm!` would launch for this product."
gemmkernel(C, A, B, M, N, K; kw...) = first(Mantle.coopmat_gemm_launches(C, A, B, M, N, K; PLAN..., kw...)).kern

relu(x) = max(x, zero(x))

if hasstagedgemm()
    @testset "Tensile GEMM" begin
        @testset "M$M N$N K$K" for (M, N, K) in [(128, 128, 32),     # two k steps: no loop trip
                                                 (256, 384, 64),     # one trip
                                                 (384, 1152, 96),    # 9 column blocks: a group of 1
                                                 (640, 1408, 512),   # 11: a group of 3
                                                 (1024, 640, 3072)]
            @test tensile_error(M, N, K) < 2e-3
        end
        @testset "bias, epilogue, fp32 destination" begin
            @test tensile_error(256, 384, 64; bias = Float16.(rand(256) .- 0.5)) < 2e-3
            @test tensile_error(256, 384, 64; bias = Float32.(rand(256) .- 0.5)) < 2e-3
            @test tensile_error(384, 1152, 96; epilogue = relu) < 2e-3
            @test tensile_error(384, 1152, 96; T = Float32) < 1e-4
            @test tensile_error(384, 1152, 96; T = Float32, bias = Float32.(rand(384) .- 0.5), epilogue = relu) < 1e-4
        end

        @testset "taken exactly when it applies" begin
            tensile = Mantle.coopmat_gemm_tensile_kernel!
            alloc(T, dims...) = KA.allocate(TESTBACKEND, T, dims...)
            A, B, C = alloc(Float16, 256, 64), alloc(Float16, 64, 384), alloc(Float16, 256, 384)
            @test gemmkernel(C, A, B, 256, 384, 64) === tensile
            @test gemmkernel(alloc(Float32, 256, 384), A, B, 256, 384, 64) === tensile
            # Another kernel asked for by name.
            @test gemmkernel(C, A, B, 256, 384, 64; tensile = false) !== tensile
            @test gemmkernel(C, A, B, 256, 384, 64; staged = false) !== tensile
            @test gemmkernel(C, A, B, 256, 384, 64; tiling = first(Mantle.GEMM_TILINGS)) !== tensile
            # Shapes and operands it does not take.
            @test gemmkernel(C, alloc(Float16, 256, 48), alloc(Float16, 48, 384), 256, 384, 48) !== tensile  # K % 32
            @test gemmkernel(alloc(Float16, 256, 320), A, alloc(Float16, 64, 320), 256, 320, 64) !== tensile  # N % 128
            @test gemmkernel(alloc(Float32, 256, 384), alloc(Float32, 256, 64), alloc(Float32, 64, 384),
                             256, 384, 64) !== tensile                                                    # fp32 operands
            @test !Mantle.tensile_gemm_ok(C, A, B, 256, 384, 64, 2, 1)                                   # batched
            @test !Mantle.tensile_gemm_ok(C, A, B, 256, 384, 64, 1, 2)                                   # split K
            # A base not known to be 16-byte aligned: the 128-bit loads step from it.
            @test Mantle.basealignment(A) >= 16
            shifted = view(A, 2:256, :)
            @test Mantle.basealignment(shifted) < 16
            @test !Mantle.tensile_gemm_ok(C, shifted, B, 256, 384, 64, 1, 1)
        end
    end
end

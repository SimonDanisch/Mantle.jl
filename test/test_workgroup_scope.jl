"""
Workgroup-scope cooperative matrices — `Scope = Workgroup` on
`OpTypeCooperativeMatrixKHR`, which `VK_NV_cooperative_matrix2` enables.

One matrix spans every invocation of the workgroup instead of one subgroup, so a
`64 x 80` fp32 accumulator is 40 components a lane at 128 invocations against
160 at subgroup scope. That is what lets `attn_flash_cm2!` hold `O`, `L` and `M`
at once, and it is the whole reason the type carries a scope parameter.

A device has them exactly when `Mantle.caps(b).wggran` is not empty; the table
pairs each workgroup size with the `(M, N, K)` multiples a matrix must be at.

**Both scopes run the same kernel body here**, parameterised by the scope, so the
subgroup row is a control: if it fails too, the fault is the test rather than
scope.

Shapes are `M=32, K=16, N=64` — all different, because a square case cannot tell
a product from a transposed reading of it and cannot see a shape mapping at all.

The operands are loaded with the portable `CoopMatrix(src, offset, stride)` load,
column-major as they are stored. Tensor-addressed loads into workgroup-scope
matrices are what `test_gemm_cm2.jl` and DNNKernels' `test_flash_cm2.jl` run.

The kernel is at TOP LEVEL, not inside the `@testset`. A `@kernel` that reaches
a local closure captures it, and the compile fails several layers from the cause.
"""

using Test, KernelAbstractions
using KernelInterface: CoopMatrix, AcceleratedMatrix, WorkgroupMatrix, MatrixA, MatrixB,
    Accumulator, SubgroupScope, WorkgroupScope, matrixscope, coopmat_muladd, wggranularity
import KernelInterface
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

const WGT_M = 32
const WGT_K = 16
const WGT_N = 64

# One body, two scopes: the comparison cannot accidentally be between two
# different programs. `A` is stored `M x K` and `B` `K x N`, column-major.
@kernel cpu=false unsafe_indices=true function wgt_gemm!(
        out, @Const(A), @Const(B), ::Val{SC}) where {SC}
    ma = CoopMatrix{Float16,WGT_M,WGT_K,MatrixA,SC}(pointer(A), 1, WGT_M)
    mb = CoopMatrix{Float16,WGT_K,WGT_N,MatrixB,SC}(pointer(B), 1, WGT_K)
    acc = zero(CoopMatrix{Float32,WGT_M,WGT_N,Accumulator,SC})
    copyto!(pointer(out), 1, WGT_M, muladd(ma, mb, acc))
end

"""
The largest workgroup the device pairs with a workgroup-scope `WGT_M x WGT_N x
WGT_K` product, or `nothing`. The size is part of the contract — launch at one the
table does not pair with these extents and the driver rejects the pipeline.
"""
function wgtgroup(c)
    fits = [nt for (nt, m, n, k) in c.wggran
            if WGT_M % m == 0 && WGT_N % n == 0 && WGT_K % k == 0 && nt <= c.workgrouplimit]
    return isempty(fits) ? nothing : maximum(fits)
end

"KernelInterface's own methods of `f` that could take arguments of these types."
kimethods(f, types...) =
    filter(m -> m.module === KernelInterface, collect(methods(f, Tuple{types...})))

@testset "workgroup-scope cooperative matrices" begin
    c = Mantle.caps(TESTBACKEND)

    @testset "the device reports its shapes, and they pair with a workgroup size" begin
        # Empty means no workgroup-scope matrices at all, so `isempty` is the
        # capability test — a kernel cannot learn "may I?" without also learning
        # "at what shapes?". A device with them has subgroup-scope ones too.
        isempty(c.wggran) || @test c.coopmat && c.tile > 0
        for (nt, m, n, k) in c.wggran
            @test nt > 0 && ispow2(nt)
            # Granularities are multiples, so each is a multiple of the
            # cooperative-matrix tile.
            @test m % c.tile == 0 && n % c.tile == 0 && k % c.tile == 0
            # …and the lookup kernels use reads the same row.
            @test wggranularity(c, nt) == (m, n, k)
        end
        # …and it coarsens as the workgroup grows, which is the property the
        # flash kernel's workgroup choice depends on.
        for i in 2:length(c.wggran)
            @test c.wggran[i][1] > c.wggran[i - 1][1]
            @test c.wggran[i][2] >= c.wggran[i - 1][2]
            @test c.wggran[i][3] >= c.wggran[i - 1][3]
        end
    end

    @testset "the type keeps the scopes apart" begin
        # Mixing scopes is a method error at the call site, not a module the
        # driver rejects — the reason scope is a type parameter.
        sg = AcceleratedMatrix{Float16,16,16,MatrixA}
        wg = WorkgroupMatrix{Float16,16,16,MatrixB}
        acc = AcceleratedMatrix{Float32,16,16,Accumulator}
        @test sg !== wg
        @test matrixscope(sg) === SubgroupScope
        @test matrixscope(wg) === WorkgroupScope
        # `muladd` is what a kernel writes. Base keeps an untyped
        # `muladd(x, y, z) = x*y + z` fallback, so it is KernelInterface's method
        # that must not match — and must match when the scopes agree, or the
        # negative proves nothing.
        @test !isempty(kimethods(muladd, sg, AcceleratedMatrix{Float16,16,16,MatrixB}, acc))
        @test isempty(kimethods(muladd, sg, wg, acc))
        # …and no backend lowers a mixed product either.
        @test isempty(methods(coopmat_muladd, Tuple{sg, wg, acc}))
    end

    if isempty(c.wggran)
        @info "no workgroup-scope cooperative matrices here; not exercised" c.wggran
    else
        back = TESTBACKEND
        nt = wgtgroup(c)
        @test nt !== nothing        # some workgroup size takes these extents
        A = KA.allocate(back, Float16, WGT_M, WGT_K)
        B = KA.allocate(back, Float16, WGT_K, WGT_N)
        a = Float16.(reshape(1:(WGT_M * WGT_K), WGT_M, WGT_K) ./ 128)
        b = Float16.(reshape((WGT_K * WGT_N):-1:1, WGT_K, WGT_N) ./ 256)
        copyto!(A, a); copyto!(B, b)
        want = Float32.(a) * Float32.(b)

        # The control runs as one subgroup; the workgroup row at the size the
        # device's table pairs with these extents.
        runs = Any[("subgroup (control)", SubgroupScope, c.subgroup)]
        nt === nothing || push!(runs, ("workgroup", WorkgroupScope, nt))
        @testset "$label" for (label, SC, n) in runs
            out = KA.allocate(back, Float32, WGT_M, WGT_N); fill!(out, Float32(NaN))
            wgt_gemm!(back, n)(out, A, B, Val(SC); ndrange = n)
            KA.synchronize(back)
            got = Array(out)
            @test all(isfinite, got)
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- want) / maximum(abs, want) < 1e-3
        end
    end
end

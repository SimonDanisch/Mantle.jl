"""
Workgroup-scope cooperative matrices — `Scope = Workgroup` on
`OpTypeCooperativeMatrixKHR`, which `VK_NV_cooperative_matrix2` enables.

One matrix spans every invocation of the workgroup instead of one subgroup, so a
`64 x 80` fp32 accumulator is 40 components a lane at 128 invocations against
160 at subgroup scope. That is what lets `attn_flash_cm2!` hold `O`, `L` and `M`
at once, and it is the whole reason the type carries a scope parameter.

**Both scopes run the same kernel body here**, parameterised by the scope, so the
subgroup row is a control: if it fails too, the fault is the test rather than
scope. `mwe_workgroup_scope.jl` is the same check as a script with a printout.

Shapes are `M=32, K=16, N=64` — all different, because a square case cannot tell
`A'*B'` from `(B*A)'` and cannot see a shape mapping at all.

The kernel and its layout helper are at TOP LEVEL, not inside the `@testset`.
A `@kernel` that reaches a local closure captures it, and the compile fails with
a `LavaCompilationError` several layers from the cause.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

const WGT_M = 32
const WGT_K = 16
const WGT_N = 64

"2-D clamping layout over a column-major `(nrow, ncol)` array; the tensor's dims
are the REVERSE, its last dimension being the fastest-varying one."
wgtlay(nrow, ncol) = Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Int32(ncol), Int32(nrow))),
        (Int32(0), Int32(0)), (Int32(ncol), Int32(nrow)))

# One body, two scopes: the comparison cannot accidentally be between two
# different programs.
@kernel cpu=false unsafe_indices=true function wgt_gemm!(
        out, @Const(A), @Const(B), ::Val{SC}) where {SC}
    za = Lava.coopmat_zero(Lava.CoopMatrix{Float16,WGT_M,WGT_K,Lava.MatrixA,SC})
    zb = Lava.coopmat_zero(Lava.CoopMatrix{Float16,WGT_K,WGT_N,Lava.MatrixB,SC})
    ma = Lava.tensor_load(za, UInt64(pointer(A)), wgtlay(WGT_K, WGT_M))
    mb = Lava.tensor_load(zb, UInt64(pointer(B)), wgtlay(WGT_N, WGT_K))
    acc = Lava.coopmat_zero(Lava.CoopMatrix{Float32,WGT_M,WGT_N,Lava.Accumulator,SC})
    Mantle.copyto!(pointer(out), 1, WGT_M, Lava.coopmat_muladd(ma, mb, acc))
end

@testset "workgroup-scope cooperative matrices" begin
    ctx = MVE.vk_context()
    dev = MVE.caps(MVE.vk_context())

    @testset "the device reports its shapes, and they pair with a workgroup size" begin
        # Empty means no workgroup-scope matrices at all, so `isempty` is the
        # capability test — a kernel cannot learn "may I?" without also learning
        # "at what shapes?".
        @test ctx.coopmat2.workgroup_scope == !isempty(dev.wggran)
        for (nt, m, n, k) in dev.wggran
            @test nt > 0 && ispow2(nt)
            # Granularities are multiples, so each is a multiple of the
            # cooperative-matrix tile.
            @test m % MVE.GEMM_TILE == 0 && n % MVE.GEMM_TILE == 0 &&
                  k % MVE.GEMM_TILE == 0
        end
        # …and it coarsens as the workgroup grows, which is the property the
        # flash kernel's workgroup choice depends on.
        for i in 2:length(dev.wggran)
            @test dev.wggran[i][1] > dev.wggran[i - 1][1]
            @test dev.wggran[i][2] >= dev.wggran[i - 1][2]
            @test dev.wggran[i][3] >= dev.wggran[i - 1][3]
        end
    end

    @testset "the type keeps the scopes apart" begin
        # Mixing scopes is a method error at the call site, not a module the
        # driver rejects — the reason scope is a type parameter.
        sg = Lava.AcceleratedMatrix{Float16,16,16,Lava.MatrixA}
        wg = Lava.WorkgroupMatrix{Float16,16,16,Lava.MatrixB}
        @test sg !== wg
        @test Lava.matrixscope(sg) === Mantle.SubgroupScope
        @test Lava.matrixscope(wg) === Mantle.WorkgroupScope
        @test isempty(methods(Lava.coopmat_muladd,
                              Tuple{sg, wg,
                                    Lava.AcceleratedMatrix{Float32,16,16,Lava.Accumulator}}))
    end

    if isempty(dev.wggran)
        @info "no workgroup-scope cooperative matrices here; not exercised" dev
    else
        back = LavaBackend()
        A = KA.allocate(back, Float16, WGT_K, WGT_M)   # an M x K matrix lives as (K, M)
        B = KA.allocate(back, Float16, WGT_N, WGT_K)
        a = Float16.(reshape(1:(WGT_K * WGT_M), WGT_K, WGT_M) ./ 128)
        b = Float16.(reshape((WGT_N * WGT_K):-1:1, WGT_N, WGT_K) ./ 256)
        copyto!(A, a); copyto!(B, b)
        want = Float32.(a') * Float32.(b')

        @testset "$label" for (label, SC, nt) in
                (("subgroup (control)", Mantle.SubgroupScope, 32),
                 ("workgroup", Mantle.WorkgroupScope, 256))
            out = KA.allocate(back, Float32, WGT_M, WGT_N); fill!(out, Float32(NaN))
            wgt_gemm!(back, nt)(out, A, B, Val(SC); ndrange = nt)
            KA.synchronize(back)
            got = Array(out)
            @test all(isfinite, got)
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- want) / maximum(abs, want) < 1e-3
        end
    end
end

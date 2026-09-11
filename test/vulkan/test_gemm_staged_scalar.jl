"""
The staged scalar GEMM: `mul!`'s fp32 path.

Ported from the `#else` (non-`COOPMAT`) branch of llama.cpp's `mul_mm.comp`,
vendored at `reference/mul_mm/`. The cooperative-matrix branch of that same
shader was already Lava's staged coopmat GEMM, and the two share their staging,
so this is the other half of a port rather than a new kernel.

It replaced `strided_gemm_kernel!`, which declared no `@localmem` at all: one
invocation per output element with K walked in global memory, two global loads
per `muladd`. That measured **0.448 TFLOP/s at 2048^3** on a Radeon 8060S against
14.6 for the fp16 cooperative-matrix path, and it got *worse* with size, which is
what a kernel whose working set outgrows cache does.

There is no fp32 shortcut this could have used instead. The driver reports
fourteen cooperative-matrix shapes and `FLOAT32` appears in none of them as an A
or B type, only as an accumulator, matching AMD's documented RDNA3 WMMA input set
of f16/bf16/iu8/iu4. So a tiled scalar GEMM is the answer rather than a
workaround for a missing one, and this file is mostly about the two things such a
kernel gets wrong: ragged extents, and being used when it should not be.

What is pinned here:

  * every combination of ragged extent, transpose and `alpha`/`beta`, because the
    `FAST` specialisation means the guarded and unguarded paths are *different
    compiled kernels* and a test that only exercises exact multiples would check
    one of them;
  * that the dispatch gate is on TILE COUNT and not on the extents. Gating on `M`
    and `N` separately was tried, and it sent the 64 x 1370 plane to the
    per-element kernel because of its 64 rows. That plane is the single largest
    operation in Depth Anything's forward pass and the staged kernel wins it
    1.63x, so the wrong gate silently gave back most of the point of the kernel.
"""

using Test, Lava, KernelAbstractions, LinearAlgebra, Random
using LinearAlgebra: mul!
const KA = KernelAbstractions

back = LavaBackend()

"Run `C = α*op(A)*op(B) + β*C` on device and on the host, return relative error."
function gemmerr(M, K, N; tA = false, tB = false, α = 1.0f0, β = 0.0f0)
    Ah = rand(Float32, tA ? (K, M) : (M, K)) .- 0.5f0
    Bh = rand(Float32, tB ? (N, K) : (K, N)) .- 0.5f0
    Ch = rand(Float32, M, N) .- 0.5f0
    Ad = KA.allocate(back, Float32, size(Ah)...); copyto!(Ad, Ah)
    Bd = KA.allocate(back, Float32, size(Bh)...); copyto!(Bd, Bh)
    Cd = KA.allocate(back, Float32, M, N);        copyto!(Cd, Ch)
    mul!(Cd, tA ? transpose(Ad) : Ad, tB ? transpose(Bd) : Bd, α, β)
    KA.synchronize(back)
    ref = α .* ((tA ? Ah' : Ah) * (tB ? Bh' : Bh)) .+ β .* Ch
    maximum(abs.(Array(Cd) .- ref)) / max(maximum(abs.(ref)), 1f-6)
end

@testset "staged scalar GEMM" begin

    # The `FAST` type parameter splits this kernel in two at compile time, so
    # both arms need exercising. Exact multiples of (BM, BN, BK) = (64, 64, 32)
    # with unit row strides take the unguarded arm; everything else does not.
    @testset "exact tiling (the FAST arm)" begin
        for (M, K, N) in ((256, 128, 256), (512, 256, 512), (1024, 32, 256))
            @test gemmerr(M, K, N) < 1e-5
        end
    end

    # Ragged extents that DO reach the staged kernel. Asserting that is the point:
    # a shape can be ragged and still be too small to tile, in which case it goes
    # to the per-element kernel and proves nothing about the guarded arm. Three of
    # the shapes originally in this list did exactly that.
    @testset "ragged extents on the guarded arm" begin
        for (M, K, N) in ((193, 97, 257),        # ragged in all three
                          (1536, 384, 1370),     # ragged N only
                          (1370, 64, 1370),      # ragged M and N
                          (64, 1370, 1370))      # ragged K and N, M exactly one tile
            f32(a, b) = KA.allocate(back, Float32, a, b)
            @test Mantle.staged_gemm_ok(f32(M, N), f32(M, K), f32(K, N), M, N)
            @test gemmerr(M, K, N) < 1e-5
        end
    end

    # ...and ragged extents that fall to the per-element kernel, which still has
    # to be right. These are the shapes too small to tile.
    @testset "ragged extents on the per-element kernel" begin
        for (M, K, N) in ((100, 37, 73), (65, 33, 65), (63, 5, 7), (1, 1, 1))
            f32(a, b) = KA.allocate(back, Float32, a, b)
            @test !Mantle.staged_gemm_ok(f32(M, N), f32(M, K), f32(K, N), M, N)
            @test gemmerr(M, K, N) < 1e-5
        end
    end

    # A transposed operand reaches the kernel as swapped strides via
    # `gemmstrides`, so it has a non-unit row stride and must take the guarded
    # arm even when the extents tile exactly.
    @testset "transposes and alpha/beta" begin
        @test gemmerr(256, 256, 256; tA = true) < 1e-5
        @test gemmerr(256, 256, 256; tB = true) < 1e-5
        @test gemmerr(256, 256, 256; tA = true, tB = true) < 1e-5
        @test gemmerr(384, 32, 384; tA = true, tB = true, α = 1.5f0, β = -0.5f0) < 1e-5
        @test gemmerr(192, 64, 192; α = 2.5f0, β = 0.5f0) < 1e-5
        # beta = 0 must OVERWRITE rather than accumulate, including where C held
        # values before the call.
        @test gemmerr(256, 64, 256; α = 1.0f0, β = 0.0f0) < 1e-5
    end

    # The gate. These are the assertions that would have caught the extent-based
    # version, which is why they name shapes rather than just counts.
    @testset "dispatch gate is on tile count" begin
        f32(M, N) = (KA.allocate(back, Float32, M, N))
        ok(M, K, N) = Mantle.staged_gemm_ok(f32(M, N), f32(M, K), f32(K, N), M, N)

        # Too few tiles to fill the device: the per-element kernel wins.
        @test !ok(64, 64, 64)          # 1 tile
        @test !ok(192, 192, 192)       # 9 tiles
        @test !ok(512, 128, 1)         # matvec, 8 tiles

        # Enough tiles but almost all of them discarded. A vector destination at
        # M = 2048 is 32 tiles, which passes the count, while computing a 64-wide
        # column for one useful column: 64x the work, and measured 0.61x. The
        # tile count alone does not catch this; `SGEMM_MAXWASTE` does.
        @test cld(2048, Mantle.SGEMM_BM) * cld(1, Mantle.SGEMM_BN) >= Mantle.SGEMM_MINTILES
        @test !ok(2048, 128, 1)     # 64x waste, measured 0.61x
        @test !ok(4096, 64, 8)      # 8x waste,  measured 0.79x

        # ...but a shape only just past a tile boundary is admitted. At 2.02x
        # waste this measured 1.46x, and an earlier `SGEMM_MAXWASTE` of 2 was
        # rejecting it.
        @test ok(65, 512, 1370)

        # Enough tiles, including the skinny plane an extent-based gate rejected.
        @test ok(256, 256, 256)        # 16 tiles, exactly SGEMM_MINTILES
        @test ok(64, 1370, 1370)       # 22 tiles: the attn*V plane, M = 64
        @test ok(1370, 64, 1370)       # 484 tiles: the Q*K' plane
        @test ok(2048, 2048, 2048)

        # The boundary is the constant, not a coincidence of these shapes.
        @test Mantle.SGEMM_MINTILES == 16
        @test !ok(64, 64, 64 * (Mantle.SGEMM_MINTILES - 1))
        @test ok(64, 64, 64 * Mantle.SGEMM_MINTILES)
    end

    # dtype: the shared blocks are Float32, so anything wider has to keep the
    # per-element kernel or it would silently round through Float32.
    @testset "only fp32 destinations take the staged path" begin
        M = N = K = 512
        c32 = KA.allocate(back, Float32, M, N)
        a32 = KA.allocate(back, Float32, M, K)
        b32 = KA.allocate(back, Float32, K, N)
        @test Mantle.staged_gemm_ok(c32, a32, b32, M, N)

        c64 = KA.allocate(back, Float64, M, N)
        a64 = KA.allocate(back, Float64, M, K)
        b64 = KA.allocate(back, Float64, K, N)
        @test !Mantle.staged_gemm_ok(c64, a64, b64, M, N)

        # ...and a Float64 product still computes correctly, on the other kernel.
        Ah = rand(Float64, 128, 64) .- 0.5; Bh = rand(Float64, 64, 128) .- 0.5
        Ad = KA.allocate(back, Float64, 128, 64); copyto!(Ad, Ah)
        Bd = KA.allocate(back, Float64, 64, 128); copyto!(Bd, Bh)
        Cd = KA.allocate(back, Float64, 128, 128)
        mul!(Cd, Ad, Bd); KA.synchronize(back)
        @test maximum(abs.(Array(Cd) .- Ah * Bh)) < 1e-10
    end

    # fp16 operands into an fp32 destination. `mul!` prefers cooperative matrices
    # for this, but only when the device has them AND all three extents land on
    # the 16-wide tile; everything else arrives here. That is the whole fp16 story
    # on a device with no cooperative-matrix support at all, so it must work and
    # must accumulate in fp32 rather than in fp16.
    @testset "fp16 operands, fp32 destination" begin
        for (M, K, N) in ((512, 128, 512),      # tiles exactly
                          (500, 130, 500))      # and does not
            Ah = rand(Float16, M, K); Bh = rand(Float16, K, N)
            Ad = KA.allocate(back, Float16, M, K); copyto!(Ad, Ah)
            Bd = KA.allocate(back, Float16, K, N); copyto!(Bd, Bh)
            Cd = KA.allocate(back, Float32, M, N)
            @test Mantle.staged_gemm_ok(Cd, Ad, Bd, M, N)
            mul!(Cd, Ad, Bd); KA.synchronize(back)
            # Reference in Float32 over the same fp16 inputs: this asserts the
            # accumulation is fp32, since an fp16 one would drift far past this.
            ref = Float32.(Ah) * Float32.(Bh)
            @test maximum(abs.(Array(Cd) .- ref)) / maximum(abs.(ref)) < 1e-3
        end
    end

    # A randomised net, because this file is the ONLY coverage this kernel has.
    # GPUArrays' own `linalg/mul!/matrix-matrix` suite runs 456 cases and every
    # one is 4x4, so the dispatch gate sends all of them to the per-element
    # kernel: they prove the fallback still works and say nothing at all about
    # the staged path. Indexing bugs in a tiled kernel hide in the shapes nobody
    # thought to enumerate, so enumerate them randomly instead.
    @testset "randomised shapes against a host reference" begin
        rng = MersenneTwister(0xC0FFEE)      # fixed: a failure has to be re-runnable
        checked = 0
        for _ in 1:14
            M = rand(rng, 200:600); N = rand(rng, 200:600); K = rand(rng, 16:300)
            f32(a, b) = KA.allocate(back, Float32, a, b)
            Mantle.staged_gemm_ok(f32(M, N), f32(M, K), f32(K, N), M, N) || continue
            checked += 1
            @test gemmerr(M, K, N) < 1e-4
        end
        # If the gate ever tightens enough to reject all of these, the testset
        # would silently assert nothing.
        @test checked >= 10
    end

    # The per-element kernel reduces in `gemmaccum(eltype(C))`, not in `eltype(C)`.
    # It used to do the latter, so an fp16 destination summed the whole K loop in
    # fp16. That is reachable and not rare: `mul!`'s cooperative-matrix gate and
    # `staged_gemm_ok` both require an fp32 destination, so every fp16-into-fp16
    # product lands there. Instrumenting MatAnyone found 29 of its 132 `matmul!`
    # calls doing exactly this at K = 256..769.
    #
    # The tolerances below are the point. Reducing in fp32 and storing once lands
    # at fp16's own store floor, ~3e-4 (eps(Float16)/2); reducing in fp16 grows as
    # sqrt(K) and reaches 4.2e-3 by K = 512. 1.5e-3 separates them cleanly at
    # every K here and would have failed before the fix.
    #
    # NOTE for anyone A/B-ing this: redefining `gemmaccum` in a live session does
    # NOT change the compiled kernel. It is `@inline`d into a `@kernel` whose
    # SPIR-V is cached on the kernel function and its argument types, so the
    # helper's body is baked in. Compare across processes, not within one.
    @testset "fp16 destinations reduce in fp32" begin
        @test Mantle.gemmaccum(Float16) === Float32
        @test Mantle.gemmaccum(Float32) === Float32
        @test Mantle.gemmaccum(Float64) === Float64
        for (M, K, N) in ((256, 256, 16), (256, 256, 1024),   # MatAnyone's shapes
                          (1024, 257, 256), (1024, 769, 64),
                          (64, 512, 64))                       # long enough to expose it
            Ah = rand(Float16, M, K) .- Float16(0.5)
            Bh = rand(Float16, K, N) .- Float16(0.5)
            A = KA.allocate(back, Float16, M, K); copyto!(A, Ah)
            B = KA.allocate(back, Float16, K, N); copyto!(B, Bh)
            C = KA.allocate(back, Float16, M, N)
            # fp16 operands into an fp16 destination: declines coopmat (fp32
            # destination required) and declines the staged kernel (likewise), so
            # this is the per-element kernel by construction.
            @test !Mantle.staged_gemm_ok(C, A, B, M, N)
            mul!(C, A, B); KA.synchronize(back)
            ref = Float64.(Ah) * Float64.(Bh)
            @test maximum(abs.(Float64.(Array(C)) .- ref)) / maximum(abs, ref) < 1.5e-3
        end
    end

    # The tiling has to stay internally consistent: the reference derives WNITER
    # from the others, and a config where it does not come out a positive integer
    # is silently wrong rather than an error.
    @testset "tiling identities hold" begin
        @test Mantle.SGEMM_WNITER * (Mantle.SGEMM_WARP * Mantle.SGEMM_TM *
              Mantle.SGEMM_TN * Mantle.SGEMM_WMITER) == Mantle.SGEMM_WM * Mantle.SGEMM_WN
        @test Mantle.SGEMM_WG == Mantle.SGEMM_NUMWARPS * Mantle.SGEMM_WARP
        @test Mantle.SGEMM_WSUBM * Mantle.SGEMM_WMITER == Mantle.SGEMM_WM
        @test Mantle.SGEMM_WSUBN * Mantle.SGEMM_WNITER == Mantle.SGEMM_WN
        # Staging must divide the workgroup, or a block is left partly unwritten.
        @test (Mantle.SGEMM_BM * Mantle.SGEMM_BK) % Mantle.SGEMM_WG == 0
        @test (Mantle.SGEMM_BK * Mantle.SGEMM_BN) % Mantle.SGEMM_WG == 0
        # Each invocation's register block covers exactly its share of the tile.
        @test Mantle.SGEMM_WMITER * Mantle.SGEMM_TM * Mantle.SGEMM_WNITER * Mantle.SGEMM_TN *
              Mantle.SGEMM_WG == Mantle.SGEMM_BM * Mantle.SGEMM_BN
        # `SHMEM_STRIDE` is the SCALAR branch's `BK/2 + 1`, in floats. The
        # coopmat branch's `BK/2 + 4` (Lava's `GEMM_PAD = 4`) is a different
        # number for a different access pattern and must not be reused here.
        @test Mantle.SGEMM_SHF == Mantle.SGEMM_BK + 2
    end
end

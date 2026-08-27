"""
The staged GEMM against the register-blocked one, on shapes that expose its
assumptions — and **every tiling in `GEMM_TILINGS`, individually**.

That last part is not thoroughness for its own sake. `(3, 2, 2, 4, 32, 8)` — a
96 x 128 block — silently lost 4 of every 32 k-terms per row, while 96 x 64 and
160 x 128 were correct, and nothing about the tiling predicted which. The cause
was not the tiling at all but the shape of the kernel's k-loop: see
`test_shared_index_division.jl`, where `while` loses 92% of a staged block that
a counted `for` keeps intact, on identical instructions.

So a tiling is trusted because it was run, never because it looks like one that
was.

The other bug this file exists for: **the staged kernel does not split K.** It
walks the whole of it and writes one plane. But `coopmat_gemm_shape` picks
`splitk` from the shape, and at 64x64x64 it picks 4 — so the caller allocates
four partial planes and sums them, three of which the staged kernel never wrote.
The result came back with 0.83 relative error, and only on small shapes: every
shape the kernel had ever been benchmarked on chooses `splitk == 1`. A GEMM test
suite made of the shapes a GEMM is fast at is a suite that cannot find this.
"""

using Test, Lava, DNNKernels, KernelAbstractions
const KA = KernelAbstractions

"Relative error of `matmul!` against a Float32 CPU reference, over a slice."
function gemmerr(backend, ws, M, N, K; staged::Bool, withbias::Bool, gemmkw...)
    # `matmul!` takes a context, not a `(backend, ws)` pair — DNNKernels'
    # kernel-library refactor moved every entry point onto `Ctx`, which is one
    # argument where there were two. This test is in the OTHER repo and so was
    # not caught by that change's own suite; the symptom here was a MethodError
    # per shape, then a device loss from the repeated failures, which reads like
    # a Lava bug and is not one.
    ctx = DNNKernels.Ctx(backend; ws)
    hA = rand(Float16, M, K) .- Float16(0.5)
    hB = rand(Float16, K, N) .- Float16(0.5)
    hbias = Float16.(rand(M) .- 0.5)
    A = KA.allocate(backend, Float16, M, K); copyto!(A, hA)
    B = KA.allocate(backend, Float16, K, N); copyto!(B, hB)
    bias = nothing
    if withbias
        bias = KA.allocate(backend, Float16, M)
        copyto!(bias, hbias)
    end
    out = KA.allocate(backend, Float16, M, N)
    fill!(out, zero(Float16))
    # Passed, not set. These were `Mantle.GEMM_STAGED[]` and friends, saved and
    # restored around the call — and a test that threw before the `finally` left
    # the setting on for every later test in the process.
    DNNKernels.reset!(ws)
    DNNKernels.matmul!(ctx, out, A, B, bias; gemm = (; staged, gemmkw...))
    KA.synchronize(backend)
    rows = 1:min(M, 24)
    ref = Float32.(hA[rows, :]) * Float32.(hB)
    withbias && (ref = ref .+ Float32.(hbias[rows]))
    return maximum(abs.(Float32.(Array(out)[rows, :]) .- ref)) / maximum(abs.(ref))
end

"""
Every k-term accumulated, or not.

`A` and `B` all ones makes each output element the count of products that
actually landed, so a result of 28 where K is 32 says *which* is wrong rather
than merely *that* it is — and an integer count survives fp16 exactly, where a
relative error on random data can hide four missing terms in the tolerance.
"""
function stagedcount(backend, cfg, K; blocks = 2)
    M = Mantle.gemm_bm(cfg) * blocks
    N = Mantle.gemm_bn(cfg) * blocks
    A = KA.allocate(backend, Float16, M, K); fill!(A, one(Float16))
    B = KA.allocate(backend, Float16, K, N); fill!(B, one(Float16))
    C = KA.allocate(backend, Float16, M, N); fill!(C, Float16(-1))
    wg = Mantle.gemm_wg(cfg)
    Mantle.GEMM_STAGED_KERNELS[cfg](backend, wg)(
        C, A, B, nothing, identity, Val(M), Val(N), Val(K);
        ndrange = (M ÷ Mantle.gemm_bm(cfg)) * (N ÷ Mantle.gemm_bn(cfg)) * wg)
    KA.synchronize(backend)
    g = Float32.(Array(C))
    (correct = all(==(Float32(K)), g), K = K, seen = sort(unique(g)))
end

@testset "staged GEMM" begin
    backend = LavaBackend()
    ws = DNNKernels.Workspace(backend)

    # Small shapes first, because they are the ones whose plan splits K.
    shapes = [(64, 64, 64), (64, 64, 32), (64, 64, 128), (128, 128, 64),
              (128, 64, 64), (64, 128, 64), (192, 192, 64),
              (576, 4096, 576), (2304, 4096, 576)]

    @testset "both paths, with and without bias" begin
        for (M, N, K) in shapes, staged in (false, true), withbias in (false, true)
            # fp16 destination and fp16 accumulate through the tensor cores, so
            # the tolerance is the format's.
            @test gemmerr(backend, ws, M, N, K; staged, withbias) < 2.0f-2
        end
    end

    @testset "both staging widths accumulate every k-term, at every K" begin
        # `vec2` picks between scalar and `vec2` staging buffers, and both are
        # generated for every tiling — so both have to be swept, not just
        # whichever is currently the default. Passed per call now; it used to be
        # a global set inside a `try`.
        for v2 in (false, true)
            for cfg in Mantle.GEMM_TILINGS,
                K in (32, 64, 96, 128, 288, 576, 2304)
                K % Mantle.gemm_bk(cfg) == 0 || continue
                v2 && !haskey(Mantle.GEMM_STAGED_V2_KERNELS, cfg) && continue
                @test gemmerr(backend, ws, Mantle.gemm_bm(cfg) * 2,
                              Mantle.gemm_bn(cfg) * 2, K;
                              staged = true, withbias = false, vec2 = v2) < 2.0f-2
            end
        end
    end

    @testset "every shipped tiling accumulates every k-term, at every K" begin
        # The K sweep is the point. The hazard in `test_shared_index_division.jl`
        # depends on the trip count as well as the geometry — the same kernel is
        # exact at K = 32 and loses terms at K = 64 — so checking one size proves
        # nothing about the others. 32 to 2304 is one iteration to seventy-two.
        for cfg in Mantle.GEMM_TILINGS,
            K in (32, 64, 96, 128, 160, 288, 320, 576, 1152, 2304)
            K % Mantle.gemm_bk(cfg) == 0 || continue
            r = stagedcount(backend, cfg, K)
            r.correct || @info "tiling $cfg lost k-terms" K = r.K seen = r.seen
            @test r.correct
        end
    end

    @testset "the tiling chooser only returns blocks that divide the shape" begin
        # No save/restore: `gemm_tiling` takes the forced tiling as an argument,
        # and not passing one IS the unforced case.
        for (M, N, K) in vcat(shapes, [(288, 16384, 1152), (1152, 16384, 288),
                                       (2304, 4096, 576), (48, 64, 64)])
            c = Mantle.gemm_tiling(M, N, K)
            c === nothing && continue
            @test M % Mantle.gemm_bm(c) == 0
            @test N % Mantle.gemm_bn(c) == 0
            @test K % Mantle.gemm_bk(c) == 0
            @test c in Mantle.GEMM_TILINGS
        end
    end

    @testset "the guard names the calls the staged kernel may take" begin
        for (M, N, K) in shapes
            splitk = Mantle.coopmat_gemm_shape(M, N, K)[2]
            Mantle.staged_gemm_tiling(M, N, K, 1, splitk; staged = true) === nothing ||
                @test splitk == 1
        end
        @test Mantle.staged_gemm_tiling(64, 64, 64, 1, 4; staged = true) === nothing  # split: refused
        @test Mantle.staged_gemm_tiling(64, 64, 64, 2, 1; staged = true) === nothing  # batched: refused
        @test Mantle.staged_gemm_tiling(48, 40, 64, 1, 1; staged = true) === nothing  # no block divides
    end

    # `gemm_padn` decides what a caller that pads `N` should pad it *to*. Rounding
    # to the 16-wide cooperative-matrix tile makes the instruction legal and the
    # staged kernel inapplicable, which is a factor of several on a shape whose
    # token count happens to land between two blocks.
    @testset "padding N lands on the block, not on the tile" begin
        # Whisper's encoder, the case that found this. 1500 tokens: cld(1500,16)*16
        # is 1504, which no tiling's block divides; 1536 is 12 x 128.
        for (M, K) in ((1280, 1280), (5120, 1280), (1280, 5120))
            @test Mantle.gemm_padn(M, 1500, K) == 1536
            c = Mantle.gemm_tiling(M, Mantle.gemm_padn(M, 1500, K), K)
            @test c !== nothing
            @test haskey(Mantle.GEMM_STAGED_KERNELS, c)
            # The old rounding is what it must beat, and it is only 32 columns away.
            @test Mantle.gemm_tiling(M, cld(1500, 16) * 16, K) === nothing
        end

        # Already on a block: no padding at all, so a shape that was fast stays
        # fast and pays nothing.
        @test Mantle.gemm_padn(1280, 1536, 1280) == 1536
        @test Mantle.gemm_padn(1280, 4096, 1280) == 4096

        # The padded width is always a multiple of the block the chooser then
        # picks — the two ask the same predicates in the same order, and a
        # divergence would pad to a width nothing uses.
        for M in (64, 96, 128, 192, 256, 576, 1280, 2304, 5120),
            K in (32, 64, 288, 576, 1152, 1280, 2304, 5120),
            N in (1, 7, 40, 100, 1500, 4095)
            NP = Mantle.gemm_padn(M, N, K)
            @test NP >= N
            c = Mantle.gemm_tiling(M, NP, K)
            c === nothing || @test NP % Mantle.gemm_bn(c) == 0
        end

        # No tiling can take these `M`/`K` at all, so there is nothing to pad for
        # and the tile rounding is what is left.
        @test Mantle.gemm_padn(48, 40, 64) == 48        # M = 48 divides no block
        @test Mantle.gemm_padn(64, 40, 48) == 48        # K = 48 is not a multiple of 32

        # `slack` is the point of the rule, not a detail of it: a small `N` pads
        # to more work than the faster kernel can pay back, so it must NOT be
        # padded. N = 48 would go to 64 (1.33x) and N = 8 to 64 (8x).
        @test Mantle.gemm_padn(1280, 48, 1280) == 48
        @test Mantle.gemm_padn(1280, 8, 1280) == 16
        # ...and the same shape with the bar removed does pad, so the assertions
        # above are testing the bar rather than an unreachable branch.
        @test Mantle.gemm_padn(1280, 48, 1280; slack = 10) == 64
        # Either side of the bar, measured against the tile rounding the padding
        # replaces rather than against `N` itself:
        @test Mantle.gemm_padn(1280, 208, 1280) == 256   # 256/208 = 1.23, just inside
        @test Mantle.gemm_padn(1280, 72, 1280) == 80     # 128/80  = 1.60, refused
        # Already on a 64-wide block: nothing to pay, nothing to refuse.
        @test Mantle.gemm_padn(1280, 192, 1280) == 192
    end
end

@testset "the aliasing rule keeps the 96-row block off its bad stride" begin
    # `gemm_aliasing` exists because a *weighted mean over shapes* said the wrong
    # thing: 96 x 128 is the faster block on four of SAM 2's six `addmm` shapes
    # and collapses on the fifth, and the fifth was heavy enough to lose it the
    # average. The collapse is exactly at `K % 256 == 0` — measured at seventeen
    # values of K on either side of it, with everything but K held fixed.
    #
    # Asserted as *selection*, not as speed: a timing assertion on this card is a
    # coin flip (the clock idles at 210 MHz of 2265) and would fail for reasons
    # that have nothing to do with the rule.
    let
        c96 = (3, 2, 2, 4, 32, 8)
        @test Mantle.gemm_bm(c96) == 96 && !ispow2(Mantle.gemm_bm(c96))
        @test Mantle.gemm_aliasing(c96, 2304)          # 256 * 9
        @test Mantle.gemm_aliasing(c96, 1024)
        @test !Mantle.gemm_aliasing(c96, 2208)         # one step below, and fine
        @test !Mantle.gemm_aliasing(c96, 2400)         # one step above, and fine
        # A power-of-two block is never refused, whatever K does.
        for c in Mantle.GEMM_TILINGS, K in (256, 1024, 2304, 4096)
            ispow2(Mantle.gemm_bm(c)) && @test !Mantle.gemm_aliasing(c, K)
        end

        # The rule as the picker applies it: no shape whose K is a multiple of
        # 256 may come back with a non-power-of-two block.
        for K in (256, 512, 1024, 1536, 2048, 2304, 2560, 3072), M in (576, 1152, 2304)
            c = Mantle.gemm_tiling(M, 4096, K)
            c === nothing && continue
            @test ispow2(Mantle.gemm_bm(c))
        end
        # ...and where it is allowed, the 96-row block is what gets picked, since
        # it now leads the table.
        for K in (576, 1152, 1728, 2880), M in (576, 1152, 2304)
            @test Mantle.gemm_tiling(M, 4096, K) == c96
        end

        # SAM 2's own six, as the encoder runs them.
        @test Mantle.gemm_tiling(2304, 4096,  576) == c96
        @test Mantle.gemm_tiling( 576, 4096, 2304) == (2, 2, 2, 4, 32, 8)   # the bad K
        @test Mantle.gemm_tiling(1728, 4096,  576) == c96
        @test Mantle.gemm_tiling( 576, 4096,  576) == c96
        @test Mantle.gemm_tiling( 288, 16384, 1152) == c96
        @test Mantle.gemm_tiling(1152, 16384,  288) == c96
    end
end

@testset "the vec2 switch's other side still compiles and agrees" begin
    # `GEMM_VEC2` picks between two staged kernels, and `coopmat_gemm!` falls
    # back to the v1 one for any config with no v2 variant — so v1 is reachable
    # in a default build, not only when the switch is flipped. It had been
    # unreachable in practice for long enough that adding an argument to the
    # kernel family missed it, and the miss surfaced as a `MethodError` at
    # launch: "no method matching ... ::typeof(identity), ::Val{288}".
    #
    # A switch whose other side is broken is not a switch. This runs it.
    let
        # `blk_split = (1, 1)` below because the default plan splits K four ways
        # at this size, and an epilogue on a split-K plane is refused — correctly,
        # a plane is a partial sum. Forcing one plane is what puts this on the
        # staged path at all, which is the path under test.
        M, N, K = 256, 256, 128
        hA = rand(Float16, M, K) .- Float16(0.5)
        hB = rand(Float16, K, N) .- Float16(0.5)
        A, B = Mantle.LavaArray(hA), Mantle.LavaArray(hB)
        bias = Mantle.LavaArray(rand(Float16, M) .- Float16(0.5))
        for withbias in (false, true), epi in (identity, x -> x * 2.0f0)
            outs = map((true, false)) do vec2
                C = KA.allocate(LavaBackend(), Float32, M, N); fill!(C, 0f0)
                Mantle.coopmat_gemm!(C, A, B, M, N, K; blk_split = (1, 1), staged = true,
                                   vec2, bias = withbias ? bias : nothing, epilogue = epi)
                KA.synchronize(LavaBackend())
                Array(C)
            end
            @test maximum(abs, outs[1]) > 1e-3          # both computed something
            # The two stage through differently shaped shared tiles but do the
            # same arithmetic in the same order, so this is exact.
            @test outs[1] == outs[2]
        end
        A = B = bias = nothing; GC.gc()
    end
end

@testset "the narrow-index kernel agrees with the wide one" begin
    # `GEMM_NARROW` swaps the staging addresses to `Int32`. It is worth -1.2%
    # weighted over SAM 2's shapes, which is small enough that the only thing
    # standing between it and a silent wrong answer is this: the two kernels must
    # produce **bit-identical** output, not merely close output. They compute the
    # same products in the same order, so anything other than equality is a bug
    # in the address arithmetic, and a narrowing bug shows up as a few wrong
    # tiles rather than as garbage.
    back = LavaBackend()
    let
        @testset "M$M N$N K$K" for (M, N, K) in
                [(4096, 2304, 576), (4096, 576, 2304), (1024, 1152, 288),
                 (256, 256, 256), (512, 128, 64), (2048, 576, 576)]
            A = Mantle.LavaArray(Float16.(reshape(0.2 .* sin.(range(0, 9, M * K)), M, K)))
            B = Mantle.LavaArray(Float16.(reshape(0.2 .* cos.(range(0, 7, K * N)), K, N)))
            C = KA.allocate(back, Float16, M, N)

            fill!(C, Float16(0)); Mantle.coopmat_gemm!(C, A, B, M, N, K; narrow_ok = false)
            KA.synchronize(back); wide = copy(Array(C))

            fill!(C, Float16(0)); Mantle.coopmat_gemm!(C, A, B, M, N, K; narrow_ok = true)
            KA.synchronize(back); narrow = copy(Array(C))

            @test narrow == wide
            @test any(!iszero, wide)          # and both actually computed something
            A = B = C = nothing; GC.gc()
        end

        # The guard that decides which one may run. `M*K`, `K*N` and `M*N` all
        # have to fit an Int32; the wide kernel is the fallback above that.
        @test Mantle.gemm_fits32(4096, 2304, 576)
        @test Mantle.gemm_fits32(16384, 1152, 288)             # the largest here, 18.9M
        @test !Mantle.gemm_fits32(65536, 65536, 65536)
        @test !Mantle.gemm_fits32(1 << 16, 2, 1 << 16)         # M*K alone overflows
        @test !Mantle.gemm_fits32(2, 1 << 16, 1 << 16)         # K*N alone overflows
        @test !Mantle.gemm_fits32(1 << 16, 1 << 16, 2)         # M*N alone overflows
        # The boundary is exclusive because the indices are 1-based.
        @test !Mantle.gemm_fits32(typemax(Int32), 1, 1)
        @test Mantle.gemm_fits32(typemax(Int32) - 1, 1, 1)
    end
end

@testset "the double-buffered kernel is bit-identical to the single-buffered one" begin
    # It is off by default (it measures no faster — see `GEMM_STAGED_DB_KERNELS`),
    # so nothing else in the suite reaches it and it would rot silently. What has
    # to hold is EQUALITY, not closeness: the two differ only in which staging
    # buffer a k-block lands in, so any difference at all is an indexing bug.
    #
    # The K values are chosen for the pipeline's edges: one block (the prologue
    # does everything and the loop's re-stage is pure overhead), two blocks, and
    # an ODD block count, where the last tile is read out of buffer 1.
    back = LavaBackend()
    for (M, N, K) in ((192, 256, 32), (192, 256, 64), (192, 256, 96),
                      (576, 512, 576), (96, 128, 32))
        for c in Mantle.GEMM_TILINGS
            (Mantle.gemm_divides(c, M, N, K) && !Mantle.gemm_aliasing(c, K)) || continue
            haskey(Mantle.GEMM_STAGED_DB_KERNELS, c) || continue
            a = Float16.(0.05f0 .* randn(Float32, M, K))
            b = Float16.(0.05f0 .* randn(Float32, K, N))
            A = KA.allocate(back, Float16, M, K); copyto!(A, a)
            B = KA.allocate(back, Float16, K, N); copyto!(B, b)
            C = KA.allocate(back, Float16, M, N)

            fill!(C, Float16(NaN))
            Mantle.coopmat_gemm!(C, A, B, M, N, K; tiling = c, doublebuf = false)
            KA.synchronize(back); one_ = copy(Array(C))

            fill!(C, Float16(NaN))
            Mantle.coopmat_gemm!(C, A, B, M, N, K; tiling = c, doublebuf = true)
            KA.synchronize(back); two = copy(Array(C))

            @test two == one_
            @test any(!iszero, one_)          # and it computed something
            A = B = C = nothing; GC.gc()
        end
    end
end

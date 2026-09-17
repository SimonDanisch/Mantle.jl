# Empty pool blocks must be returned to the driver without waiting for an OOM.
#
# `GPU_LIVE_BYTES` tracks pool *capacity*, and blocks used to be handed back only
# on an allocation-failure retry. `maybe_collect`'s gate is a ratio against the
# device heap, so on an iGPU with a large shared heap a few GB of dead blocks is
# only ~20 % and never trips it — the pool ratchets up to the high-water mark of
# the largest workload and stays there. Across a multi-scene run that is
# gigabytes of system RAM the rest of the machine still needs, and it showed up
# as a driver timeout partway through a 5-scene sweep.
#
# `maybe_trim_pool!` adds an absolute-capacity trigger so dead capacity is
# released on its own terms.

# `Mantle`, not `Lava`: `LavaBackend` is Mantle's and `Lava` does not export it,
# so the first line of the first testset threw `UndefVarError` — this file has not
# run since the tests were moved onto Mantle's names.
using Test, Mantle, KernelAbstractions
using Mantle: LavaBackend
const KA = KernelAbstractions

@testset "empty pool blocks are trimmed without an OOM" begin
    be = LavaBackend()
    ctx = Mantle.vk_context()

    # Against this file's own baseline, not zero: `gpu_live_bytes` is process
    # wide, and every region another file retired without a trim is still live
    # in it. What this checks is that the capacity THIS testset created comes
    # back.
    base = Mantle.gpu_live_bytes()
    threshold = Mantle.mempolicy(Mantle.vk_context()).trim_threshold
    target = base + threshold + 256 * 1024 * 1024
    let arrays = Mantle.LavaArray[]
        while Mantle.gpu_live_bytes() < target
            a = KA.allocate(be, Float32, 4_000_000)   # 16 MB each
            fill!(a, 1.0f0)
            push!(arrays, a)
        end
        KA.synchronize(be)
        empty!(arrays)
    end

    grown = Mantle.gpu_live_bytes()
    @test grown - base >= threshold

    # Defeat the rate limiter so the test doesn't depend on wall-clock timing.
    Mantle.mempolicy(Mantle.vk_context()).last_trim = 0.0
    Mantle.maybe_trim_pool!(ctx)

    trimmed = Mantle.gpu_live_bytes()
    @test trimmed < grown                     # capacity actually came back
    @test trimmed - base < threshold          # and it was THIS testset's

    # And the allocator still works afterwards — blocks were returned, not corrupted.
    b = KA.allocate(be, Float32, 1024)
    fill!(b, 2.0f0)
    KA.synchronize(be)
    @test all(Array(b) .== 2.0f0)
end

# The case the testset above cannot reach, and the one that mattered.
#
# It calls `KA.synchronize` before dropping its arrays, so every buffer's
# timeline value has already signalled, `vk_free!` destroys each on the spot, and
# the trim finds empty blocks. Drop them while a batch is still *recording* and
# nothing is destroyed at all: `vk_free!` takes its `pins > 0` branch, sets
# `free_requested` and returns, and the block keeps its `live_count` until the
# flush inside `quiesce_before_reclaim!` releases the pin. (A buffer with
# in-flight work takes a third branch onto core's retired list, released by the
# drain in the same call.)
#
# `trim_gpu_pool!` used to gate on `any(b -> isempty(b.live), blocks)` *before*
# that call — a precondition it establishes itself — so it returned `(0, 0)` and
# kept everything. A graph evaluator is nothing but this shape, dispatches
# recorded and not flushed until the output is read: TRELLIS.2's 30-block torso
# left 190 blocks and 12 410 MiB resident with 0 blocks empty, and 12 750 MiB of
# it was reclaimable. Nothing had leaked; the trim was refusing to look.
#
# 60 unsynchronised dispatches is the smallest thing that reproduces it, and the
# assertion is on the state *before* the trim as well as the bytes after, so this
# fails loudly if a future change makes the workload stop reproducing rather than
# passing on a technicality.
@testset "an explicit trim flushes before it decides there is nothing to do" begin
    be = LavaBackend()
    ctx = Mantle.vk_context()
    Mantle.trim_gpu_pool!(ctx)                  # from a known floor
    # …which is a floor and not zero in the full suite; see the first testset.
    base2 = Mantle.gpu_live_bytes()

    @kernel function grind!(a)
        i = @index(Global)
        x = a[i]
        for _ in 1:2000
            x = x * 1.0000001f0 + 1f-7
        end
        a[i] = x
    end

    let arrays = Mantle.LavaArray[]
        for _ in 1:60
            a = KA.allocate(be, Float32, 4_000_000)          # 16 MB
            fill!(a, 1.0f0)
            grind!(be, 256)(a; ndrange = length(a))          # recorded, NOT synchronised
            push!(arrays, a)
        end
        empty!(arrays)
    end
    GC.gc(true)

    grown = Mantle.gpu_live_bytes()
    @test grown - base2 > 256 * 1024 * 1024
    # The state the old gate mishandled — every block still counted as live even
    # though every reference to its contents is gone.
    @test !any(b -> isempty(b.live), Mantle.poolblocks(ctx))

    blocks, bytes = Mantle.trim_gpu_pool!(ctx)
    @test blocks > 0
    @test bytes > 0
    # Half of what THIS testset added, for the reason the first one gives.
    @test Mantle.gpu_live_bytes() - base2 < (grown - base2) ÷ 2

    # And the allocator still works — blocks were returned, not corrupted.
    b = KA.allocate(be, Float32, 1024)
    fill!(b, 2.0f0)
    KA.synchronize(be)
    @test all(Array(b) .== 2.0f0)
end

@testset "trim is rate-limited" begin
    ctx = Mantle.vk_context()
    Mantle.mempolicy(Mantle.vk_context()).last_trim = time()            # just trimmed
    before = Mantle.gpu_live_bytes()
    Mantle.maybe_trim_pool!(ctx)                # must be a no-op, not a stall
    @test Mantle.gpu_live_bytes() == before
end

# The other rate limit, and the one that was missing: a soft-cap collection is
# spaced by what it COSTS, not only by how often it may run.
#
# `gc_mingap` is 20 ms. An incremental collection on a heap holding a couple of
# GiB of GPU-backed arrays takes ~63 ms, so the gap alone permitted spending
# three quarters of the clock inside the allocator — and it did: RIFE at
# 1920x1152 sits at 2094.6 MiB against the 2048 MiB default cap, collected six
# times a run, and 142 ms of work was reported as a p50 of 519 ms with a spread
# out to 912. The pool was 2% over the cap the whole time, so there was nothing
# to win either.
@testset "a soft-cap collection is limited by what it cost" begin
    ctx = Mantle.vk_context()
    p = Mantle.mempolicy(ctx)
    bq = ctx.default_bq
    was = (p.gc_last, p.gc_lastcost, p.gc_budget)
    try
        # A collection that cost 100 ms, on a 5% budget, is a 2 s gap — so one
        # 20 ms after it is refused, where `gc_mingap` alone would allow it.
        p.gc_budget = 0.05
        p.gc_lastcost = 0.1
        p.gc_last = time() - 0.02
        @test !Mantle.collect_for_pool!(bq)
        # Past the cost-derived gap it runs again.
        p.gc_last = time() - 2.5
        @test Mantle.collect_for_pool!(bq)
        # …and what it just cost is remembered, which is what spaces the next one.
        @test p.gc_lastcost > 0.0
        # `gc_mingap` still applies when a collection was cheap: the budget can
        # only ever make the gap LONGER, never shorter.
        p.gc_lastcost = 0.0
        p.gc_last = time()
        @test !Mantle.collect_for_pool!(bq)
        # A budget of 1 is no bound at all, which is what this did before.
        p.gc_budget = 1.0
        p.gc_lastcost = 0.1
        p.gc_last = time() - 0.15
        @test Mantle.collect_for_pool!(bq)
    finally
        p.gc_last, p.gc_lastcost, p.gc_budget = was
    end
end

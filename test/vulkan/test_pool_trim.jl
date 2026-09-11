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

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "empty pool blocks are trimmed without an OOM" begin
    be = LavaBackend()
    ctx = Mantle.vk_context()

    # Grow the pool past the trim threshold, then drop every reference.
    target = Mantle.mempolicy(Mantle.vk_context()).trim_threshold + 256 * 1024 * 1024
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
    @test grown >= Mantle.mempolicy(Mantle.vk_context()).trim_threshold

    # Defeat the rate limiter so the test doesn't depend on wall-clock timing.
    Mantle.mempolicy(Mantle.vk_context()).last_trim = 0.0
    Mantle.maybe_trim_pool!(ctx)

    trimmed = Mantle.gpu_live_bytes()
    @test trimmed < grown                     # capacity actually came back
    @test trimmed < Mantle.mempolicy(Mantle.vk_context()).trim_threshold

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
    @test grown > 256 * 1024 * 1024
    # The state the old gate mishandled — every block still counted as live even
    # though every reference to its contents is gone.
    @test !any(b -> isempty(b.live), Mantle.poolblocks(ctx))

    blocks, bytes = Mantle.trim_gpu_pool!(ctx)
    @test blocks > 0
    @test bytes > 0
    @test Mantle.gpu_live_bytes() < grown ÷ 2

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

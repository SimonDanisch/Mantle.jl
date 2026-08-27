# A captured sequence must survive a garbage collection.
#
# `capture` records command buffers once and `replay!` re-submits them, so a
# replay depends on everything the recording named still being there — and on
# nothing having moved. A collection between the two is not exotic: any caller
# doing host work between replays (an editor redrawing, a script building a
# result array) will trigger one.
#
# **This passes today, at 1 thread and at 32.** It is a guard on the invariant,
# not a reproduction — and that it passes is itself the finding: two dependent
# dispatches, one submission, one capture, one collection is NOT enough to break
# a replay.
#
# The failing case is SAM 2's decoder, which captures on the first click and
# replays on every one after: four clicks in a row pass, and the second fails
# with `GPUVM fault detected at address 0x100000000` the moment a `GC.gc(true)`
# is inserted between them. The same sequence with `replaydecode = false` — the
# identical work, recorded fresh each time — passes with the same collections.
# So a replay is what cannot survive the collection, and something this file does
# not yet do is required to provoke it. The untested differences, in the order
# worth trying: a capture spanning several submissions (SAM 2's spans four
# command buffers), enough push constants to fill more than one arg slab, and
# fresh recording interleaved between replays (each click uploads its prompt
# before replaying).
#
# What has been ruled out, so the next person does not re-run these:
#
#   * buffer lifetime. Of the `LavaArray`s the capture pins, ZERO have their
#     buffer destroyed by the collection, and no BDA changes.
#   * memory returning to the driver. `gpu_live_bytes` is identical across the GC.
#   * command-buffer recycling. None of `seq.cmd_bufs` appears in
#     `bq.free_cmd_bufs` afterwards.
#   * pool size. Trimming the pool from 4.4 GB to 1.3 GB before the replay
#     changes nothing; the fault address is unrelated to the pool's extent.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function capgc_bump!(dst, @Const(src), k::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] + k
end

@testset "a captured sequence survives a garbage collection" begin
    backend = LavaBackend()
    bq = Mantle.vk_context().default_bq
    n = 1 << 16
    src = KA.allocate(backend, Float32, n); fill!(src, 1f0)
    dst = KA.allocate(backend, Float32, n); fill!(dst, 0f0)
    KA.synchronize(backend)

    # Two dependent launches, so the sequence is more than a single dispatch.
    seq = Mantle.capture(bq) do
        capgc_bump!(backend)(dst, src, 1f0; ndrange = n)
        capgc_bump!(backend)(dst, dst, 10f0; ndrange = n)
    end
    KA.synchronize(backend)
    @test all(==(12f0), Array(dst))

    # A replay with nothing in between: the path that already worked.
    fill!(dst, 0f0); KA.synchronize(backend)
    Mantle.replay!(seq)
    KA.synchronize(backend)
    @test all(==(12f0), Array(dst))

    # …and the same replay with a collection first. Host allocation is what a
    # real caller does between replays; `GC.gc(true)` just makes it deterministic.
    fill!(dst, 0f0); KA.synchronize(backend)
    GC.gc(true)
    Mantle.replay!(seq)
    KA.synchronize(backend)
    @test all(==(12f0), Array(dst))
end

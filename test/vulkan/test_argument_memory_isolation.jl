using Test
using Lava, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# No two launches may be handed the same argument bytes while both can still
# run. That is the property; these are the two ways it was broken.
#
# `sweep_retired_batches!` reset the arg/indirect slab cursors whenever
# `in_flight` drained to empty — including when called opportunistically from
# the ALLOCATION path in the middle of recording a batch. The already-recorded
# dispatches kept their packed args at slab offsets below the cursor; after the
# reset, the next dispatch's args overwrote them, so the earlier dispatches
# read garbage arg buffers (wrong buffer pointers) when their batch submitted.
#
# Latent because `in_flight` only drains mid-recording when the GPU runs
# ahead of the host. First surfaced through Hikari's volpath bounce-loop
# early-exit (2026-06-10): the mid-loop synchronize let every subsequent
# sample's first trace dispatch read a clobbered queue pointer — whole
# samples rendered black, nondeterministically.
#
# There is no cursor to reset now: a launch's arguments are a `Region` owned by
# the one-shot that recorded it, released only once that one-shot's submission
# has passed on the timeline. The tests stay, because they pin the property
# rather than the mechanism, and the second one is the reason the mechanism
# changed.
#
# The sequence below reproduces the original deterministically:
#   1. drain, then launch dispatch A (submits itself → in_flight)
#   2. launch dispatch B (its own submission, args at a fresh region)
#   3. wait (WITHOUT sweeping) until A's submission completes
#   4. allocate → opportunistic sweep retires A → in_flight loses A →
#      the old code reset the cursors HERE, with B's submission still
#      holding its own region
#   5. launch dispatch C then D
#   6. synchronize: verify every dispatch wrote ITS OWN array

@kernel function fill_value_kernel!(arr, v::Int32)
    i = @index(Global)
    @inbounds arr[i] = v
end

# Heavy variant: keeps the GPU busy for tens of milliseconds so the submitted
# batch is still in flight while the host records the next dispatches — that
# in-flight-ness is what holds the hazard window open (see sequence below).
@kernel function heavy_fill_kernel!(arr, v::Int32)
    i = @index(Global)
    acc = Float32(i)
    for _ in 1:20_000
        acc = muladd(acc, 1.0000001f0, 0.5f0)
    end
    @inbounds arr[i] = v + (acc > 0f0 ? Int32(0) : Int32(1))
end

@testset "a mid-recording sweep must not reclaim a recorded dispatch's arguments" begin
    backend = Mantle.defaultbackend()
    bq = backend.dispatch_bq
    n = 4096

    a = KA.allocate(backend, Int32, n); KA.fill!(a, Int32(0))
    b = KA.allocate(backend, Int32, n); KA.fill!(b, Int32(0))
    c = KA.allocate(backend, Int32, n); KA.fill!(c, Int32(0))
    KA.synchronize(backend)

    k! = fill_value_kernel!(backend, 256)
    heavy! = heavy_fill_kernel!(backend, 256)

    # 1. Launch A (slow — keeps the GPU busy). It submits itself immediately;
    #    its one-shot owns its own argument region until ITS submission passes.
    heavy!(a, Int32(1); ndrange=n)
    target = Mantle.driver(bq).next_timeline

    # 2. Launch B — a separate submission with its own region, taken out
    #    while A is still in flight.
    k!(b, Int32(2); ndrange=n)

    # 3. Wait for A's submission to complete WITHOUT sweeping, so it sits
    #    completed-but-unreclaimed in in_flight.
    deadline = time() + 10.0
    while Mantle.query_timeline(bq) < target
        time() > deadline && error("timeout waiting for submitted batch")
        sleep(0.001)
    end

    # 4. Allocation → vk_alloc/pool_alloc → drain! retires A's
    #    submission. B's own submission is unaffected: its region belongs to
    #    B's one-shot and is only released once B's submission has passed.
    #    (Must be a real allocation — ≤64 B takes the unified fast path
    #    without a sweep.)
    big = KA.allocate(backend, Int32, 1 << 20)

    # 5. Two more launches, each its own submission with its own region — a
    #    sweep between launches can never hand B's bytes to D.
    k!(c, Int32(3); ndrange=n)
    d = KA.allocate(backend, Int32, n); KA.fill!(d, Int32(0))
    k!(d, Int32(4); ndrange=n)

    # 6. Wait for everything and verify each dispatch wrote ITS OWN array.
    KA.synchronize(backend)
    @test all(==(1), Array(a))
    @test all(==(2), Array(b))   # fails pre-fix: B ran with D's args
    @test all(==(3), Array(c))
    @test all(==(4), Array(d))
end

# The same hazard, reached without any submit at all — which is the shape the
# graphics path actually has. `pack_gfx_args` takes an arg buffer for a draw and
# only then records the draw. With nothing in flight and no draw recorded yet,
# a sweep used to reset the pool: the buffer just handed to draw k is handed out
# again to draw k+1, and the frame submits with two draws pointing at one
# argument buffer. On screen that is a null buffer device address and a GPUVM
# fault at 0x0.
#
# It took a hundred plots and no per-frame flush to hit in the wild — many draws
# per frame is what makes a sweep land between a handout and its draw. Here it is
# three lines, because "has this batch recorded a dispatch" was never the right
# question: what matters is who owns the bytes, and the answer is now a `Region`
# the one-shot holds until it is reclaimed.
@testset "two handouts are two disjoint blocks, whatever the sweep does" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    Mantle.flush!(ctx.default_bq)                      # nothing in flight, nothing recorded

    # A submission that has completed but has not been swept yet: that is what
    # the sweep drains, and draining is what used to reset the cursors.
    backend = Mantle.defaultbackend()
    k! = fill_value_kernel!(backend, 256)
    e0 = KA.allocate(backend, Int32, 4)
    k!(e0, Int32(0); ndrange=4)
    target = Mantle.driver(bq).next_timeline
    while Mantle.query_timeline(bq) < target; end   # polling does not sweep

    # `get_arg_buffer` takes the OWNER of the bytes, not the queue: what decides
    # when they may be handed out again is who holds them, and a queue does not.
    a = Ref{Any}(nothing)
    b = Ref{Any}(nothing)
    o = Mantle.oneshot(bq) do e
        a[] = Mantle.get_arg_buffer(e.owner, 256)   # a draw's arguments, not yet recorded
        Mantle.drain!(bq)                           # sweeps: outstanding drains here
        b[] = Mantle.get_arg_buffer(e.owner, 256)   # the next draw's arguments
    end

    # Disjoint, not ordered: the pool fits a request into the best free span it
    # has, so the second handout is under no obligation to be above the first.
    @test b[].address >= a[].address + a[].size || a[].address >= b[].address + b[].size

    # Never submitted: core gives the one-shot back and the scratch with it.
    Mantle.release!(bq, o)
end

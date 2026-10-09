"""
`repeat!` — a loop whose body is declared and compiled ONCE and run `n` times,
with an optional gate that lets the DEVICE decide how many of them run.

With a gate, iterations past `count[]` are discarded at execution, so nothing
about the trip count reaches the host. That is the shape a wavefront bounce loop
wants, and it is why the lowering is a per-iteration predicate rather than
device-generated commands: DGC repeats a dispatch with no barriers between the
repetitions, which cannot express a loop whose iteration k+1 reads what k wrote.

Three properties, and the third is the one that makes it usable:

  * the count comes from a buffer, and a shader may write it;
  * the same compiled plan runs a different number of iterations when that
    buffer changes, with no rebuild;
  * **iterations are ORDERED** — a discarded iteration does not disturb the
    ordering of the ones that run. The body below is `x[i] = x[i] * 2 + 1` on
    the same buffer every time, which is order-sensitive and value-sensitive: run
    it k times from 0 and you get `2^k - 1`. Any missed barrier, any iteration
    run twice or out of order, gives a number that is not of that form.

`2^k - 1` is also why the counter is not just "how many ran": 3 iterations gives
7 and 7 iterations gives 127, so an off-by-one is not a small error in the
result, it is a different order of magnitude.

Without a gate every iteration runs, and the body tells them apart by the index
`f` is handed, a device `Int32` counting from 1.

A gated pass holding a HOST-sized dispatch needs a device that can discard
fixed-size recorded work (`supportspredicate`): Vulkan's conditional rendering,
or a backend that walks its passes on the host. The testsets built that way run
only where it can; on a device that cannot, the first one asserts the plan is
refused at build instead. A gated pass whose dispatches are all sized on the
device runs everywhere.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function repeat_step!(x)
    i = @index(Global)
    @inbounds x[i] = x[i] * Int32(2) + Int32(1)
end

"""Writes the trip count on the DEVICE, from a value only the device has."""
@kernel function repeat_decide!(count, src)
    @inbounds count[1] = src[1]
end

"""`x .= 2x + 1`, `count[]` times, recorded `maxiters` times."""
function _repeatplan(dev, x, count, src, maxiters, n)
    g = Mantle.Graph(dev)
    # The count itself is produced by a kernel, so the host never supplies it.
    Mantle.dispatch!(g, repeat_decide!, (count, src), 1; name = "decide")
    Mantle.repeat!(g, maxiters, count) do i
        Mantle.dispatch!(g, repeat_step!, (x,), n; name = "step")
    end
    Mantle.record!(Mantle.Plan(g))
end

@testset "repeat!: the device decides the trip count" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev)
        n, maxiters = 64, 8

        x = Mantle.Buffer(dev, zeros(Int32, n))
        count = Mantle.Buffer(dev, zeros(Int32, 1))
        src = Mantle.Buffer(dev, zeros(Int32, 1))
        pl = Base.invokelatest(_repeatplan, dev, x, count, src, maxiters, n)

        # ONE compiled plan, run for every trip count. `src` is the only thing that
        # moves, and it moves on the device.
        for k in (0, 1, 3, maxiters)
            copyto!(Mantle.storage(x), zeros(Int32, n))
            copyto!(Mantle.storage(src), Int32[k])
            Mantle.waitidle(dev)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            got = Array(Mantle.storage(x))
            @test all(==(Int32(2)^k - Int32(1)), got)
        end

        # A count above what was recorded cannot run more than was recorded — the
        # loop is bounded by `maxiters`, not by whatever a shader happens to write.
        copyto!(Mantle.storage(x), zeros(Int32, n))
        copyto!(Mantle.storage(src), Int32[maxiters + 5])
        Mantle.waitidle(dev)
        Mantle.run!(pl)
        Mantle.waitidle(dev)
        @test all(==(Int32(2)^maxiters - Int32(1)), Array(Mantle.storage(x)))

        Mantle.free!(pl)
    else
        # A gated pass holding a host-sized dispatch, on a device that cannot
        # discard fixed-size recorded work: refused at build, not run wrongly.
        n = 64
        x = Mantle.Buffer(dev, zeros(Int32, n))
        count = Mantle.Buffer(dev, zeros(Int32, 1))
        src = Mantle.Buffer(dev, zeros(Int32, 1))
        @test_throws ArgumentError Base.invokelatest(_repeatplan, dev, x, count, src, 8, n)
    end
end

# Baked, because that is the point: a loop whose count the device decides is what
# lets a whole render be ONE recording. If `repeat!` only worked interpreted it
# would have bought nothing — the host would still be deciding, just later.
@testset "repeat! survives recording" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev)
        n, maxiters = 64, 8
        x = Mantle.Buffer(dev, zeros(Int32, n))
        count = Mantle.Buffer(dev, zeros(Int32, 1))
        src = Mantle.Buffer(dev, zeros(Int32, 1))
        pl = Base.invokelatest(_repeatplan, dev, x, count, src, maxiters, n)
        Mantle.record!(pl)
        Mantle.waitidle(dev)

        for k in (2, 5, 0, 7)
            copyto!(Mantle.storage(x), zeros(Int32, n))
            copyto!(Mantle.storage(src), Int32[k])
            Mantle.waitidle(dev)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            @test all(==(Int32(2)^k - Int32(1)), Array(Mantle.storage(x)))
        end
        Mantle.free!(pl)
    end
end

# THE property that makes this a loop rather than a bounded repeat: the gate is
# re-read before every iteration, so the BODY can end it.
#
# The first design expanded a count into one flag per iteration in a single pass
# before the body ran. That cannot express a bounce loop at all — it fixes the
# trip count at a moment when nothing yet knows it, because rays die as the loop
# runs. This testset fails outright against that design, which is the point of
# having it: the body below empties the thing the gate watches.
#
# `budget` starts at 3 and each iteration decrements it. With a per-iteration
# gate exactly 3 iterations run whatever `maxiters` is; with an up-front gate all
# 8 do, because the flag was computed when `budget` was still 3.
@kernel function repeat_drain!(x, budget)
    i = @index(Global)
    @inbounds x[i] += Int32(1)
    # One invocation owns the counter; every other lane only touches `x`.
    i == 1 && (@inbounds budget[1] -= Int32(1))
end

@testset "repeat!: the body ends its own loop" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev)
        n, maxiters = 32, 8
        x = Mantle.Buffer(dev, zeros(Int32, n))
        budget = Mantle.Buffer(dev, zeros(Int32, 1))

        g = Mantle.Graph(dev)
        Mantle.repeat!(g, maxiters; while_nonzero = budget) do i
            Mantle.dispatch!(g, repeat_drain!, (x, budget), n; name = "drain")
        end
        pl = Mantle.record!(Mantle.Plan(g))

        for start in (0, 1, 3, maxiters, maxiters + 4)
            copyto!(Mantle.storage(x), zeros(Int32, n))
            copyto!(Mantle.storage(budget), Int32[start])
            Mantle.waitidle(dev)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            # Bounded by what was recorded, however much budget there was.
            want = Int32(min(start, maxiters))
            @test all(==(want), Array(Mantle.storage(x)))
            # …and the loop stopped because the budget ran out, not because it was
            # counted: what is left is exactly the unspent remainder.
            @test Array(Mantle.storage(budget))[1] == Int32(max(0, start - maxiters))
        end
        Mantle.free!(pl)
    end
end

# A loop long enough to cross the queue's own submission threshold.
#
# A recorder that ends the command buffer whenever a dispatch takes it past a
# split or submit threshold is what breaks this. A conditional-rendering scope
# spans a pass, so an end landing between its `begin` and its `end` leaves an
# unmatched begin in the buffer that went and an unmatched end in the next —
# and the GPU hung on it.
#
# Hikari found this before this test did: the fused sample ran at `max_depth` 8
# (47 dispatches) and hung at 16 (~95), with a submit threshold at 64 in
# between. Core cuts a recording into pieces only BETWEEN passes (by the plan's
# budget, and at a loop body's edge), never inside one; the loop below runs past
# where such thresholds sat, so this keeps pinning the same boundary.
#
# **A regression here HANGS rather than fails.** The wait is a foreign call that
# does not return and cannot be interrupted, so there is no error to catch and
# nothing useful in a stack trace. That is what makes it worth pinning.
@testset "repeat! survives a loop longer than any submission threshold" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev)
        # Comfortably past both thresholds (64 dispatches to submit, 3000 to
        # split), which is where a recording would want to end mid-scope.
        maxiters = 2 * 64 + 8
        n = 32
        x = Mantle.Buffer(dev, zeros(Int32, n))
        budget = Mantle.Buffer(dev, zeros(Int32, 1))

        g = Mantle.Graph(dev)
        Mantle.repeat!(g, maxiters; while_nonzero = budget) do i
            Mantle.dispatch!(g, repeat_drain!, (x, budget), n; name = "drain")
        end
        pl = Mantle.record!(Mantle.Plan(g))
        # The body is compiled once, whatever the count: the host-write updates, the
        # index reset, then gate, drain and advance.
        @test length(pl.passes) == 5

        for start in (3, maxiters - 1)
            copyto!(Mantle.storage(x), zeros(Int32, n))
            copyto!(Mantle.storage(budget), Int32[start])
            Mantle.waitidle(dev)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            @test all(==(Int32(start)), Array(Mantle.storage(x)))
        end
        Mantle.free!(pl)
    end
end

@testset "repeat! wants at most one gate and at least one iteration" begin
    dev0 = Mantle.Device(TESTBACKEND)
    g0 = Mantle.Graph(dev0); c0 = Mantle.Buffer(dev0, zeros(Int32, 1))
    @test_throws ArgumentError Mantle.repeat!(_ -> nothing, g0, 4, c0; while_nonzero = c0)
    # …and `maxiters` has to name at least one iteration.
    dev = Mantle.Device(TESTBACKEND)
    g2 = Mantle.Graph(dev)
    c2 = Mantle.Buffer(dev, zeros(Int32, 1))
    @test_throws ArgumentError Mantle.repeat!(_ -> nothing, g2, 0, c2)
end

# The gate is folded into the prepare: a discarded iteration writes zero groups
# for every device-sized dispatch of its pass, so a backend with no way to
# discard recorded commands still runs nothing. A gated loop whose dispatches are
# all sized on the device therefore runs on EVERY device, whatever it can discard,
# and the refusal guards what the fold cannot cover: fixed-size work in a gated
# pass, on a device that cannot discard it.
#
# The Vulkan version switched the device's conditional rendering off for the
# duration, so that the prepare's zero was the whole answer on a device that has
# it. That switch is a driver field. The zero itself is asserted, on every
# device, by `test_discarded_iteration_prepare.jl`, which reads the indirect slot
# of a discarded iteration.
@kernel function repeat_step_sized!(x, n)
    i = @index(Global)
    @inbounds if i <= Int(n[1])
        x[i] = x[i] * Int32(2) + Int32(1)
    end
end

@testset "repeat!: a device-sized body is discarded on every device" begin
    dev = Mantle.Device(TESTBACKEND)
    n, maxiters = 64, 8
    x = Mantle.Buffer(dev, zeros(Int32, n))
    nbuf = Mantle.Buffer(dev, Int32[n])
    count = Mantle.Buffer(dev, zeros(Int32, 1))
    src = Mantle.Buffer(dev, Int32[3])
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, repeat_decide!, (count, src), 1; name = "decide")
    Mantle.repeat!(g, maxiters, count) do i
        # With a ceiling, which a recording on every device can be narrowed from.
        Mantle.dispatch!(g, repeat_step_sized!, (x, nbuf), Mantle.DeviceRange(nbuf; max = n);
                         name = "step")
    end
    pl = Mantle.record!(Mantle.Plan(g))
    Mantle.run!(pl)
    Mantle.waitfor!(pl)
    # Three iterations of `2x + 1` from 0 give 7; all eight would give 255.
    @test all(==(Int32(7)), Array(Mantle.storage(x)))
    Mantle.free!(pl)

    # A gated pass with a host-sized dispatch cannot be discarded by the fold:
    # refused at build on a device that cannot discard it either.
    Mantle.supportspredicate(dev) ||
        @test_throws ArgumentError Base.invokelatest(_repeatplan, dev, x, count, src, maxiters, n)
end

# ── the compile-once loop ─────────────────────────────────────────────────────
#
# Without a gate every iteration runs; the body reads the index to differ per
# iteration. Two transients carry across iterations and two are scoped to one:
#
#   * `t` is read by the body before it is written (an accumulator), and read
#     after the loop, so it lives across the whole body and keeps its value
#     from one iteration to the next;
#   * `w` and `w2` are written and then read inside an iteration, so the placer
#     may reuse their memory — and the head barrier of the next iteration is
#     what orders that reuse.
#
# After `n` iterations `t = 1 + … + n` and `out2 = 2(1 + … + n)`. A body that ran
# with a stale index, a carried value that was aliased away, or an iteration
# that read the previous one's `w` gives a different number.
@kernel function loop_zero!(t)
    i = @index(Global)
    @inbounds t[i] = Int32(0)
end
@kernel function loop_accum!(t, index)
    i = @index(Global)
    @inbounds t[i] += index[1]
end
@kernel function loop_twice!(w, index)
    i = @index(Global)
    @inbounds w[i] = index[1] * Int32(2)
end
@kernel function loop_addfrom!(out, w)
    i = @index(Global)
    @inbounds out[i] += w[i]
end
@kernel function loop_copy!(out, t)
    i = @index(Global)
    @inbounds out[i] = t[i]
end

function loopgraph(dev, n, iters)
    g = Mantle.Graph(dev)
    out1 = Mantle.Buffer(dev, zeros(Int32, n))
    out2 = Mantle.Buffer(dev, zeros(Int32, n))
    t = Mantle.Transient.Buffer(g, Int32, n)
    Mantle.dispatch!(g, loop_zero!, (t,), n; name = "zero t")
    Mantle.dispatch!(g, loop_zero!, (out2,), n; name = "zero out2")
    Mantle.repeat!(g, iters) do i
        Mantle.dispatch!(g, loop_accum!, (t, i), n; name = "accumulate")
        w = Mantle.Transient.Buffer(g, Int32, n)
        Mantle.dispatch!(g, loop_twice!, (w, i), n; name = "twice")
        w2 = Mantle.Transient.Buffer(g, Int32, n)
        Mantle.dispatch!(g, loop_copy!, (w2, w), n; name = "copy")
        Mantle.dispatch!(g, loop_addfrom!, (out2, w2), n; name = "add")
    end
    Mantle.dispatch!(g, loop_copy!, (out1, t), n; name = "read t")
    return g, out1, out2
end

@testset "repeat!: a host count, the index, and transients across iterations" begin
    dev = Mantle.Device(TESTBACKEND)
    n = 256
    for iters in (1, 7, 50)
        g, out1, out2 = loopgraph(dev, n, iters)
        pl = Mantle.record!(Mantle.Plan(g))
        # Compiled once, for every count: the host-write updates, two zeroings,
        # the index reset, five body passes (the four above and the advance) and
        # the read. One iteration is no loop, and has neither reset nor advance.
        @test length(pl.passes) == (iters == 1 ? 8 : 10)
        @test length(pl.loops) == (iters == 1 ? 0 : 1)
        want = iters * (iters + 1) ÷ 2
        for _ in 1:3
            Mantle.run!(pl)
            Mantle.waitfor!(pl)
            @test all(==(Int32(want)), Array(Mantle.storage(out1)))
            @test all(==(Int32(2want)), Array(Mantle.storage(out2)))
        end
        Mantle.free!(pl)
    end
end

# Where the recording is cut and what goes in one submission are core's, and a
# loop is cut at its body's edge so a run can submit the body once per
# iteration. The results cannot depend on either.
#
#   * `budget = Inf`: before, body, after; all iterations in ONE submission.
#   * a budget no pass fits: every pass its own piece and every iteration of
#     every body pass its own submission — 4 before + 7 × 5 + 1 after.
#   * the default: measured by the first run, one pass per submission, and
#     re-recorded at the start of the second into the budget partition, which
#     for work this small is the three pieces in one submission again.
#
# The last two need a device that can time its passes (`timestamps`). One that
# cannot never measures, so no budget cuts it: the three pieces, in one
# submission, whatever the budget.
@testset "repeat!: iterations are cut into submissions by the budget" begin
    dev = Mantle.Device(TESTBACKEND)
    n, iters = 256, 7
    nbatches(pl) = length(pl.recording.batches)
    timed = Mantle.timestamps(dev)
    whole = [(3, 1), (3, 1)]
    # (pieces, submissions) after the first run and after the second.
    for (budget, after) in ((Inf, whole),
                            (1e-9, timed ? [(10, 40), (10, 40)] : whole),
                            (Mantle.submissionbudget(dev), timed ? [(10, 40), (3, 1)] : whole))
        g, out1, out2 = loopgraph(dev, n, iters)
        pl = Mantle.record!(Mantle.Plan(g; budget))
        for r in 1:2
            Mantle.run!(pl)
            Mantle.waitfor!(pl)
            @test (length(pl.partition), nbatches(pl)) == after[r]
            @test all(==(Int32(28)), Array(Mantle.storage(out1)))
            @test all(==(Int32(56)), Array(Mantle.storage(out2)))
        end
        Mantle.free!(pl)
    end
end

@testset "repeat! refuses what one recording of the body cannot express" begin
    dev = Mantle.Device(TESTBACKEND)
    x = Mantle.Buffer(dev, zeros(Int32, 4))
    # Nested loops: a pass belongs to one loop.
    g = Mantle.Graph(dev)
    @test_throws ArgumentError Mantle.repeat!(g, 2) do i
        Mantle.repeat!(g, 2) do j
            Mantle.dispatch!(g, loop_zero!, (x,), 4; name = "inner")
        end
    end
    # A `when!` region inside a body, and a body inside one.
    g = Mantle.Graph(dev)
    flag = Ref(true)
    @test_throws ArgumentError Mantle.repeat!(g, 2) do i
        Mantle.when!(g, flag) do
            Mantle.dispatch!(g, loop_zero!, (x,), 4; name = "conditional")
        end
    end
    g = Mantle.Graph(dev)
    @test_throws ArgumentError Mantle.when!(g, flag) do
        Mantle.repeat!(g, 2) do i
            Mantle.dispatch!(g, loop_zero!, (x,), 4; name = "looped")
        end
    end
end

# One iteration is a gated region: no loop in the plan, no index to reset or
# advance, and a recording that is not cut around it. VideoEditor gates every
# optional effect of its chain this way.
@testset "repeat!(g, 1; gate) is a gated region, not a loop" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev)
        n = 32
        x = Mantle.Buffer(dev, zeros(Int32, n))
        flag = Mantle.Buffer(dev, Int32[0])
        g = Mantle.Graph(dev)
        Mantle.repeat!(g, 1; while_nonzero = flag) do _
            Mantle.dispatch!(g, repeat_drain!, (x, flag), n; name = "once")
        end
        pl = Mantle.record!(Mantle.Plan(g; budget = Inf))
        @test isempty(pl.loops)
        @test [pp.pass.name for pp in pl.passes] == ["updates", "repeat!/gate", "once"]
        for start in (0, 1, 0, 1)
            copyto!(Mantle.storage(x), zeros(Int32, n))
            copyto!(Mantle.storage(flag), Int32[start])
            Mantle.waitidle(dev)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            @test all(==(Int32(start)), Array(Mantle.storage(x)))
        end
        Mantle.free!(pl)
    end
end

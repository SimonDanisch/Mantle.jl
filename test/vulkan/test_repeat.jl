"""
`repeat!` — a loop recorded once, whose trip count the DEVICE decides.

The body is recorded `maxiters` times and the iterations past `count[]` are
discarded at execution, so nothing about the trip count reaches the host. That is
the shape a wavefront bounce loop wants, and it is why the lowering is a
per-iteration predicate rather than device-generated commands: DGC repeats a
dispatch with no barriers between the repetitions, which cannot express a loop
whose iteration k+1 reads what k wrote.

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
"""

using Test, Mantle, Lava, KernelAbstractions
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
    Mantle.compute!(g, "decide") do p
        Mantle.use(p, src; read = true)
        Mantle.use(p, count; write = true)
        Mantle.dispatch!(p, repeat_decide!, (count, src), 1)
    end
    Mantle.repeat!(g, maxiters, count) do i
        Mantle.compute!(g, "step-$i") do p
            Mantle.use(p, x; read = true, write = true)
            Mantle.dispatch!(p, repeat_step!, (x,), n)
        end
    end
    Mantle.record!(Mantle.Plan(g))
end

@testset "repeat!: the device decides the trip count" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    @test Mantle.supportspredicate(dev)
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
end

# Baked, because that is the point: a loop whose count the device decides is what
# lets a whole render be ONE recording. If `repeat!` only worked interpreted it
# would have bought nothing — the host would still be deciding, just later.
@testset "repeat! survives recording" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
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
    dev = Mantle.Device(Mantle.VulkanAPI())
    n, maxiters = 32, 8
    x = Mantle.Buffer(dev, zeros(Int32, n))
    budget = Mantle.Buffer(dev, zeros(Int32, 1))

    g = Mantle.Graph(dev)
    Mantle.repeat!(g, maxiters; while_nonzero = budget) do i
        Mantle.compute!(g, "drain-$i") do p
            Mantle.use(p, x; read = true, write = true)
            Mantle.use(p, budget; read = true, write = true)
            Mantle.dispatch!(p, repeat_drain!, (x, budget), n)
        end
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

# A loop long enough to cross the queue's own submission threshold.
#
# The open batch's recorder used to end the command buffer whenever a dispatch
# took it past its split or submit threshold. A conditional-rendering scope
# spans a pass, so an end landing between its `begin` and its `end` left an
# unmatched begin in the buffer that went and an unmatched end in the next —
# and the GPU hung on it.
#
# Hikari found this before this test did: the fused sample ran at `max_depth` 8
# (47 dispatches) and hung at 16 (~95), with the submit threshold at 64 in
# between. The thresholds are gone with the open batch — a plan's recording is
# one command buffer, whole, and nothing cuts it — and the loop below is sized
# past where they used to sit, so this keeps pinning the same boundary.
#
# **A regression here HANGS rather than fails.** The wait is a foreign call that
# does not return and cannot be interrupted, so there is no error to catch and
# nothing useful in a stack trace. That is what makes it worth pinning.
@testset "repeat! survives a loop longer than any submission threshold was" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    # Comfortably past both of the old thresholds (64 dispatches to submit,
    # 3000 to split), where the recording used to want to end mid-scope.
    maxiters = 2 * 64 + 8
    n = 32
    x = Mantle.Buffer(dev, zeros(Int32, n))
    budget = Mantle.Buffer(dev, zeros(Int32, 1))

    g = Mantle.Graph(dev)
    Mantle.repeat!(g, maxiters; while_nonzero = budget) do i
        Mantle.compute!(g, "drain-$i") do p
            Mantle.use(p, x; read = true, write = true)
            Mantle.use(p, budget; read = true, write = true)
            Mantle.dispatch!(p, repeat_drain!, (x, budget), n)
        end
    end
    pl = Mantle.record!(Mantle.Plan(g))
    @test length(pl.passes) > 64          # the old boundary is crossed

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

# A backend that cannot discard recorded work must say so at build. Running every
# recorded iteration instead is the one wrong answer that still produces output,
# and on a bounce loop it would be a slower render rather than a broken one —
# invisible until someone measured it.
#
# The host backend is not such a device: it never records, so it predicates by
# reading the flag between passes — and that read must GATE, not just exist.
@testset "repeat! on the host predicates by reading the flag" begin
    host = Mantle.Device(Mantle.HostAPI())
    @test Mantle.supportspredicate(host)
    n = 8
    x = Mantle.Buffer(host, zeros(Int32, n))
    count = Mantle.Buffer(host, zeros(Int32, 1))
    src = Mantle.Buffer(host, Int32[3])
    pl = _repeatplan(host, x, count, src, 7, n)
    Mantle.run!(pl)
    # Three iterations of `2x + 1` from 0 give 2^3 - 1. Ungated it would be
    # 2^7 - 1, so a flag nobody reads is a different order of magnitude, not a
    # rounding error.
    @test all(==(Int32(7)), Array(Mantle.storage(x)))
    Mantle.free!(pl)
end

@testset "repeat! wants exactly one gate and at least one iteration" begin
    # Neither both gates nor neither.
    dev0 = Mantle.Device(Mantle.VulkanAPI())
    g0 = Mantle.Graph(dev0); c0 = Mantle.Buffer(dev0, zeros(Int32, 1))
    @test_throws ArgumentError Mantle.repeat!(_ -> nothing, g0, 4)
    @test_throws ArgumentError Mantle.repeat!(_ -> nothing, g0, 4, c0; while_nonzero = c0)
    # …and `maxiters` has to name at least one iteration.
    dev = Mantle.Device(Mantle.VulkanAPI())
    g2 = Mantle.Graph(dev)
    c2 = Mantle.Buffer(dev, zeros(Int32, 1))
    @test_throws ArgumentError Mantle.repeat!(_ -> nothing, g2, 0, c2)
end

# The gate is folded into the prepare: a discarded iteration writes zero groups
# for every device-sized dispatch of its pass, so a backend with no way to
# discard recorded commands still runs nothing. Pinned by switching Vulkan's
# conditional rendering off for the duration — `withpredicate` then emits no
# scope and the prepare's zero is the whole answer — and by the refusal that
# guards what the fold cannot cover: fixed-size work in a gated pass.
@kernel function repeat_step_sized!(x, n)
    i = @index(Global)
    @inbounds if i <= Int(n[1])
        x[i] = x[i] * Int32(2) + Int32(1)
    end
end

@testset "repeat!: the fold discards without conditional rendering" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    ctx = MVE.vk_context()
    had = ctx.conditional_rendering_available
    ctx.conditional_rendering_available = false
    try
        @test !Mantle.supportspredicate(dev)
        n, maxiters = 64, 8
        x = Mantle.Buffer(dev, zeros(Int32, n))
        nbuf = Mantle.Buffer(dev, Int32[n])
        count = Mantle.Buffer(dev, zeros(Int32, 1))
        src = Mantle.Buffer(dev, Int32[3])
        g = Mantle.Graph(dev)
        Mantle.compute!(g, "decide") do p
            Mantle.use(p, src; read = true)
            Mantle.use(p, count; write = true)
            Mantle.dispatch!(p, repeat_decide!, (count, src), 1)
        end
        Mantle.repeat!(g, maxiters, count) do i
            Mantle.compute!(g, "step-$i") do p
                Mantle.use(p, x; read = true, write = true)
                Mantle.use(p, nbuf; read = true)
                Mantle.dispatch!(p, repeat_step_sized!, (x, nbuf), Mantle.DeviceRange(nbuf))
            end
        end
        pl = Mantle.record!(Mantle.Plan(g))
        Mantle.run!(pl)
        Mantle.waitfor!(pl)
        # Three iterations of `2x + 1` from 0 give 7; all eight would give 255.
        @test all(==(Int32(7)), Array(Mantle.storage(x)))
        Mantle.free!(pl)

        # A gated pass with a host-sized dispatch cannot be discarded by the fold,
        # and this backend can no longer discard it either: refused at build.
        @test_throws ArgumentError _repeatplan(dev, x, count, src, maxiters, n)
    finally
        ctx.conditional_rendering_available = had
    end
end

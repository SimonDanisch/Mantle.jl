"""
Memory named by a submission still in flight must be honoured, not raced.

This is the intermittent `vkWaitSemaphores` hang — six occurrences over two
months, recorded as "not reproducible" — and it was one missing case in the
Vulkan backend's `vk_free!`: a buffer the open batch had recorded against but
never submitted read as idle and was destroyed under the batch. There is no open
batch now; every launch HOLDS what it names into its own closed command buffer
and submits it at once, and the hold is what keeps the buffer: a free that
arrives while the submission is in flight is RECORDED (core's `retire!`), the
buffer stays alive, and the destroy runs once the submission has passed. The
window before a recording is submitted at all is core's `Stamp.holders`, pinned
without a device in `test_hold_lifetime.jl`.

Asserted directly rather than by trying to provoke the hang. Reproducing it takes
a Julia GC landing mid-flight — 60 SAM 2 decodes with the collector live
did it within 15, and with collections confined to safe points never did — which
is a fine way to *find* a bug and a terrible way to guard one.

What is asserted, on every backend, is the pool's ledger: a `Mantle.Buffer` freed
while a launch that reads it is still running stays handed out — nobody else can
be given those bytes — until the device is past the launch, and then it comes
back, however many submissions after the free have finished first; one that
nothing in flight names comes back at the first reclaim after the
device is idle. The test read the Vulkan buffer's state machine (`ALIVE -> DEAD`,
`holders`) for the same two facts. A backend array freed or dropped mid-flight is
checked by the result the launch computes from it.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# Spins `iters` dependent steps — nothing can fold them away — and only then
# reads `src`, so the read happens after the host has freed it, while the
# launch is still on the device.
@kernel function spinread!(out, @Const(src), iters)
    i = @index(Global)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    @inbounds out[i] = src[i] + (acc == Int32(-7) ? 1f0 : 0f0)
end

"""
Iterations of `spinread!` that keep this device busy for about `ms`
milliseconds: long enough that a launch is reliably still in flight when the host
does the next thing, without spinning on a host flag, which can hang a device.
Measured once, after a launch that compiles the kernel.
"""
function spinsfor(ms)
    out = KA.zeros(TESTBACKEND, Float32, 64)
    src = KA.zeros(TESTBACKEND, Float32, 64)
    spinread!(TESTBACKEND, 64)(out, src, Int32(1000); ndrange = 64)
    KA.synchronize(TESTBACKEND)
    t = @elapsed begin
        spinread!(TESTBACKEND, 64)(out, src, Int32(1_000_000); ndrange = 64)
        KA.synchronize(TESTBACKEND)
    end
    return Int32(clamp(round(Int, 1_000_000 * ms / (1000 * t)), 1, typemax(Int32)))
end

@kernel function bump!(x)
    i = @index(Global)
    @inbounds x[i] += 1f0
end

"""Is `r` still handed out by its block — bytes the pool may give nobody else?"""
islive(r) = get(r.block.live, Mantle.offset(r), -1) == Mantle.offset(r) + length(r)

"""The pool's two phases around a wait: what was freed is stamped, the device
finishes, and what it has finished with is released."""
function reclaimidle!(dev)
    sp = Mantle.pool(dev)
    Mantle.reclaim!(sp, dev)
    Mantle.waitidle(dev)
    Mantle.reclaim!(sp, dev)
    return nothing
end

"""A launch that reads a backend array nothing else references once this returns."""
@noinline function readdropped!(out, n, iters)
    src = Mantle.devicearray(TESTBACKEND, fill(5f0, n))
    spinread!(TESTBACKEND, 64)(out, src, iters; ndrange = n)
    return nothing
end

@testset "a free while the memory is in flight is honoured, not raced" begin
    dev = Mantle.Device(TESTBACKEND)
    sp = Mantle.pool(dev)
    n = 64
    iters = spinsfor(200)

    # Quiesce, so nothing left over decides the outcome — and warm the free and
    # reclaim path, so that compiling it cannot outlast the launch below.
    let warm = Mantle.Buffer(dev, zeros(Float32, n))
        Mantle.free!(warm)
        reclaimidle!(dev)
    end

    @testset "named by a launch still in flight: kept, released once it passes" begin
        a = Mantle.Buffer(dev, fill(5f0, n))
        out = Mantle.Buffer(dev, zeros(Float32, n))
        r = Mantle.region(a.store)

        # Slow launch: it is still running when the free below lands.
        spinread!(TESTBACKEND, 64)(Mantle.storage(out), Mantle.storage(a), iters; ndrange = n)
        Mantle.free!(a)

        # The request is recorded and nothing is released: the first reclaim
        # stamps the region with what the device has been given, and the second
        # asks again and finds the launch still running.
        Mantle.reclaim!(sp, dev)
        Mantle.reclaim!(sp, dev)
        @test islive(r)

        # And nothing leaks: once the device is past the launch, the region that
        # was owed comes back.
        KA.synchronize(TESTBACKEND)
        reclaimidle!(dev)
        @test !islive(r)
        # …and the launch read what it was given.
        @test all(==(5f0), Array(out))
    end

    @testset "a submission after the free does not stand for launches before it" begin
        # `k` small launches, then the slow one, the free, and a recorded plan's
        # submission. On a backend that batches launches the slow one lands at
        # every position of its batch, the last included — Metal.jl commits at 32
        # operations — and a batch committed with it leaves nothing open to put a
        # signal behind it: the plan's submission was signalled past the
        # still-running launch on Metal 4, and the region came back at once. Every
        # `k` up to 40, so no backend's threshold is named. The small launches go
        # FIRST because they finish at once: behind the slow one they pile up, and
        # on RADV the 33rd unfinished submission blocked in `vkQueueSubmit2` until
        # the slow one was done — after which handing the region back is right.
        # The pause after the plan is what makes it deterministic: the plan's own
        # submission is done in well under a millisecond, and asked straight after
        # `run!` the reclaim usually came before that signal too.
        g = Mantle.Graph(dev)
        tmp = Mantle.Buffer(dev, zeros(Float32, n))
        Mantle.dispatch!(g, bump!, (tmp,), n; name = "bump")
        plan = Mantle.record!(Mantle.Plan(g))
        Mantle.run!(plan)
        small = KA.zeros(TESTBACKEND, Float32, n)
        KA.synchronize(TESTBACKEND)
        short = spinsfor(100)
        early = Int[]
        for k in 0:40
            a = Mantle.Buffer(dev, fill(5f0, n))
            out = Mantle.Buffer(dev, zeros(Float32, n))
            r = Mantle.region(a.store)
            for _ in 1:k
                bump!(TESTBACKEND, 64)(small; ndrange = n)
            end
            spinread!(TESTBACKEND, 64)(Mantle.storage(out), Mantle.storage(a), short; ndrange = n)
            Mantle.free!(a)
            Mantle.run!(plan)
            sleep(0.02)
            Mantle.reclaim!(sp, dev)
            Mantle.reclaim!(sp, dev)
            islive(r) || push!(early, k)
            KA.synchronize(TESTBACKEND)
            reclaimidle!(dev)
        end
        @test isempty(early)
    end

    @testset "nothing in flight names it: released at the first reclaim" begin
        # The other side, so the guard cannot be satisfied by deferring
        # everything forever — which would leak instead of hanging.
        b = Mantle.Buffer(dev, zeros(Float32, n))
        r = Mantle.region(b.store)
        fill!(b, 1f0)
        KA.synchronize(TESTBACKEND)          # nothing left in flight naming it
        Mantle.free!(b)
        reclaimidle!(dev)
        @test !islive(r)
    end

    @testset "a backend array freed or dropped mid-flight is still read correctly" begin
        # The Vulkan case the hang came from: a backend array, not a pool buffer,
        # freed while the launch naming it runs — explicitly, and by the collector.
        src = Mantle.devicearray(TESTBACKEND, fill(5f0, n))
        out = KA.zeros(TESTBACKEND, Float32, n)
        spinread!(TESTBACKEND, 64)(out, src, iters; ndrange = n)
        Mantle.unsafe_free!(src)
        canary = Mantle.devicearray(TESTBACKEND, fill(-1f0, n))   # the bytes a premature free would hand out
        KA.synchronize(TESTBACKEND)
        @test all(==(5f0), Array(out))
        @test all(==(-1f0), Array(canary))

        out2 = KA.zeros(TESTBACKEND, Float32, n)
        readdropped!(out2, n, iters)
        GC.gc(true); GC.gc(true)
        canary2 = Mantle.devicearray(TESTBACKEND, fill(-1f0, n))
        KA.synchronize(TESTBACKEND)
        @test all(==(5f0), Array(out2))
        @test all(==(-1f0), Array(canary2))
    end
end

"""
Capture a dispatch sequence once, re-submit it without recording.

The motivating case is an inference step whose launch sequence is identical
every iteration: recording it costs more host time than running it costs GPU
time, so replaying the recorded command buffers removes the larger half.

Checks, in order: a replay computes what the recording computed; a replay
observes input written between replays (the command buffer refers to the buffer,
not to a snapshot of it); repeated replays accumulate exactly like repeated
recordings; and replay is substantially cheaper on the host.
"""

using Test, KernelAbstractions, Lava, Statistics
const KA = KernelAbstractions

@kernel function addc!(a, c)
    i = @index(Global, Linear)
    @inbounds a[i] = a[i] + c
end

@kernel function scale!(dst, @Const(src), f)
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i] * f
end

@testset "capture / replay" begin
    be = LavaBackend()
    bq = be.bq
    n = 4096
    a = KA.allocate(be, Float32, n)
    d = KA.allocate(be, Float32, n)
    ka = addc!(be, 256)
    ks = scale!(be, 256)

    # One "step": 20 dependent dispatches, then a scaled copy out.
    step!() = begin
        for _ in 1:20
            ka(a, 1.0f0; ndrange = n)
        end
        ks(d, a, 2.0f0; ndrange = n)
    end

    fill!(a, 0.0f0)
    step!()
    KA.synchronize(be)
    @test all(collect(a) .== 20.0f0)
    @test all(collect(d) .== 40.0f0)

    # Capture executes once itself, so `a` advances by another 20.
    fill!(a, 0.0f0)
    KA.synchronize(be)
    seq = Mantle.capture(step!, bq)
    KA.synchronize(be)
    @test all(collect(a) .== 20.0f0)
    @test all(collect(d) .== 40.0f0)
    @test !isempty(seq.cmd_bufs)

    # A replay must do exactly what another recording would have done.
    Mantle.replay!(seq)
    KA.synchronize(be)
    @test all(collect(a) .== 40.0f0)
    @test all(collect(d) .== 80.0f0)

    # Input written between replays is observed: the recorded commands name the
    # buffer, they do not carry a copy of it.
    fill!(a, 100.0f0)
    KA.synchronize(be)
    Mantle.replay!(seq)
    KA.synchronize(be)
    @test all(collect(a) .== 120.0f0)
    @test all(collect(d) .== 240.0f0)

    # Repeated replays accumulate like repeated recordings.
    fill!(a, 0.0f0)
    KA.synchronize(be)
    for _ in 1:5
        Mantle.replay!(seq)
    end
    KA.synchronize(be)
    @test all(collect(a) .== 100.0f0)

    # Recording again on the same queue must not disturb a captured sequence —
    # this is what `reserve_arg_slabs!` protects.
    fill!(a, 0.0f0)
    step!()
    KA.synchronize(be)
    Mantle.replay!(seq)
    KA.synchronize(be)
    @test all(collect(a) .== 40.0f0)

    # Host cost: replaying should be far cheaper than recording.
    hostcost(f; iters = 50) = begin
        f(); KA.synchronize(be)
        best = Inf
        for _ in 1:5
            t = time_ns()
            for _ in 1:iters; f(); end
            best = min(best, (time_ns() - t) / 1e6 / iters)
            KA.synchronize(be)
        end
        best
    end
    rec = hostcost(step!)
    rep = hostcost(() -> Mantle.replay!(seq))
    @info "capture/replay host cost" record_ms=round(rec, digits=4) replay_ms=round(rep, digits=4) speedup=round(rec/rep, digits=1)
    @test rep < rec / 3
end

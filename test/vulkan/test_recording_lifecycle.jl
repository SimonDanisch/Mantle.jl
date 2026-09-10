"""
What a recording has to survive between the run that wrote it and the runs that
submit it.

Three files stood here — `test_capture_replay.jl`, `test_capture_gc.jl` and
`test_replay_interleaved.jl` — and all three drove `MVE.capture(f, bq)` over ad
hoc KernelAbstractions launches. That API is gone: `capture` recorded by running
the ordinary recorder and collecting whatever `submit!` sealed on the way past,
which is why it also EXECUTED, why it needed `bq.capturing` for four separate
decisions, and why `flush!` had to refuse while one was open. A plan's commands
are written into a command buffer of its own now.

The properties are the same ones and they are asserted against a plan:

  * a run computes what the recording says, and repeated runs accumulate;
  * a run observes input written between runs — the commands name the buffer,
    they do not carry a copy of it;
  * a recording survives a garbage collection;
  * a run interleaves with ordinary ad hoc recording on the same queue without
    desyncing the timeline;
  * submitting a recording is far cheaper on the host than writing one.

The GC case is a guard on the invariant rather than a reproduction, and that it
passes is itself the finding. The failing case it came from is SAM 2's decoder,
which captured on the first click and replayed on every one after: four clicks
in a row pass, and the second fails with `GPUVM fault detected at address
0x100000000` the moment a `GC.gc(true)` is inserted between them. What has been
ruled out, so the next person does not re-run it: buffer lifetime (of the
`LavaArray`s pinned, ZERO have their buffer destroyed by the collection and no
BDA changes), memory returning to the driver (`gpu_live_bytes` is identical
across the GC), command-buffer recycling, and pool size.

The interleaving case is a real bug this pins. Opening the old batch handed it
`MVE.driver(bq).next_timeline + 1` as its signal value, reserving it, and `submit!`
later asserts the reservation still holds; a submission path that bumped the same
counter left any open batch with a stale reservation and the next `submit!` died
with `AssertionError: batch signal desync: 859 vs 860`, naming neither. The shape
below is the minimum that reproduces it: record without draining, then run, then
record again. Draining between the two hides it entirely.
"""

using Test, Mantle, Lava, KernelAbstractions, Statistics
const KA = KernelAbstractions

@kernel function reclife_add!(a, c)
    i = @index(Global, Linear)
    @inbounds a[i] = a[i] + c
end

@kernel function reclife_scale!(dst, @Const(src), f)
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i] * f
end

"""Twenty dependent adds over `a`, then a scaled copy into `d`."""
function _stepplan(dev, a, d, n)
    g = Mantle.Graph(dev)
    for k in 1:20
        Mantle.compute!(g, "add$k") do p
            Mantle.use(p, a; read = true, write = true)
            Mantle.dispatch!(p, reclife_add!, (a, 1.0f0), n)
        end
    end
    Mantle.compute!(g, "scale") do p
        Mantle.dispatch!(p, reclife_scale!, (Mantle.use(p, d; write = true),
                                             Mantle.use(p, a; read = true), 2.0f0), n)
    end
    Mantle.Plan(g)
end

@testset "a recording runs, accumulates and reads its inputs" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 4096
    a = Mantle.Buffer(dev, zeros(Float32, n))
    d = Mantle.Buffer(dev, zeros(Float32, n))
    pl = Base.invokelatest(_stepplan, dev, a, d, n)

    Mantle.record!(pl)
    KA.synchronize(be)
    # Nothing ran while it was written.
    @test all(==(0.0f0), Array(Mantle.storage(a)))

    Mantle.run!(pl)
    KA.synchronize(be)
    @test all(==(20.0f0), Array(Mantle.storage(a)))
    @test all(==(40.0f0), Array(Mantle.storage(d)))

    # Input written between runs is observed: the recorded commands name the
    # buffer, they do not carry a copy of it.
    fill!(Mantle.storage(a), 100.0f0)
    KA.synchronize(be)
    Mantle.run!(pl)
    KA.synchronize(be)
    @test all(==(120.0f0), Array(Mantle.storage(a)))
    @test all(==(240.0f0), Array(Mantle.storage(d)))

    # Repeated runs accumulate, and several of them are in flight at once —
    # which is what makes `SIMULTANEOUS_USE` on the recording load-bearing: the
    # same command buffer is submitted again before the device has finished the
    # previous submission of it.
    fill!(Mantle.storage(a), 0.0f0)
    KA.synchronize(be)
    for _ in 1:5
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    @test all(==(100.0f0), Array(Mantle.storage(a)))

    # …and a collection between runs changes nothing.
    fill!(Mantle.storage(a), 0.0f0)
    KA.synchronize(be)
    GC.gc(true)
    Mantle.run!(pl)
    KA.synchronize(be)
    @test all(==(20.0f0), Array(Mantle.storage(a)))

    Mantle.free!(pl)
end

@testset "a run interleaves with ad hoc recording on the same queue" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 1024
    a = Mantle.Buffer(dev, zeros(Float32, n))
    d = Mantle.Buffer(dev, zeros(Float32, n))
    pl = Base.invokelatest(_stepplan, dev, a, d, n)
    Mantle.record!(pl)

    scratch = KA.allocate(be, Float32, n)
    # Not zeroed by the allocator: on a pristine pool fresh pages read 0 and
    # this test passes by luck, but once earlier work has recycled bytes
    # through the pool, `scratch` starts at whatever they last held. The
    # interleave below asserts +3 four times and +5 once, so it must start at 0.
    KA.fill!(scratch, 0f0)
    bump = reclife_add!(be, 256)

    # No `synchronize` between the two: this is the case that desynced. The ad
    # hoc launch leaves a batch open with a reserved timeline value, and the run
    # goes into that same batch.
    for _ in 1:4
        bump(scratch, 3.0f0; ndrange = n)
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    @test all(==(12.0f0), Array(scratch))
    @test all(==(80.0f0), Array(Mantle.storage(a)))

    # The desync surfaced on the NEXT submit, not on the run, so the assertion is
    # that ordinary work keeps going afterwards.
    Mantle.run!(pl)
    bump(scratch, 5.0f0; ndrange = n)
    KA.synchronize(be)
    @test all(==(17.0f0), Array(scratch))

    Mantle.free!(pl)
end

@testset "submitting a recording is cheaper than writing one" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 4096
    a = Mantle.Buffer(dev, zeros(Float32, n))
    d = Mantle.Buffer(dev, zeros(Float32, n))

    # `record!` on a fresh plan of the same graph, against `run!` on one already
    # recorded. Both build the same 21 dispatches; only the first writes command
    # buffers for them.
    hostcost(f; iters) = begin
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

    plans = Mantle.Plan[]
    writecost = hostcost(iters = 5) do
        pl = Base.invokelatest(_stepplan, dev, a, d, n)
        Mantle.record!(pl)
        push!(plans, pl)
    end

    hot = Base.invokelatest(_stepplan, dev, a, d, n)
    Mantle.record!(hot)
    runcost = hostcost(() -> Mantle.run!(hot); iters = 50)

    @info "recording host cost" record_ms=round(writecost, digits = 4) run_ms=round(runcost, digits = 4) speedup=round(writecost / runcost, digits = 1)
    @test runcost < writecost / 3

    for pl in plans
        Mantle.free!(pl)
    end
    Mantle.free!(hot)
end

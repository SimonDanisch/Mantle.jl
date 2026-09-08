"""
A command buffer is never open when a Mantle call returns.

Every path that reaches the device — an ad hoc launch, an upload, a readback,
a plan's run, a frame — closes what it wrote inside the call and hands it over
the moment it is closed. The queue holds nothing between calls but what is in
flight: no open batch anything could append to, no list of closed-but-
unsubmitted work, no threshold deciding the fate of either. This file pins
that from the outside.

Two of the assertions are structural rather than behavioural, and say why. The
barrier at the head of every closed buffer cannot be observed as a race on
this driver: an MWE with an undeclared hazard — two launches, the second
reading what the first wrote, no barrier anywhere — showed zero errors on
RADV, and so did its positive control. So what is pinned is that the barrier
is WRITTEN, counted on the context as it is emitted, once per closed buffer.
And "no open command buffer" is pinned by the queue having nowhere to keep
one: the fields that held it are gone, and every closed buffer the queue knows
about is sealed.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function ccb_bump!(a)
    i = @index(Global)
    @inbounds a[i] += 1f0
end

@kernel function ccb_add!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

"""Every closed buffer the queue holds, in flight or pooled."""
function allclosed(bq)
    cs = Any[]
    for o in bq.outstanding
        sub = o.payload
        append!(cs, sub.oneshots)
        append!(cs, sub.recordings)
    end
    append!(cs, bq.free_oneshots)
    cs
end

"""What one call left behind: how many submissions, head barriers and timeline
values it produced, and whether anything is open afterwards."""
function observe(f, bq)
    d = bq.ctx.diag
    f0, h0, t0 = d.flush_counter[], d.head_barriers[], bq.next_timeline
    f()
    (submits = d.flush_counter[] - f0, barriers = d.head_barriers[] - h0,
     tokens = Int(bq.next_timeline - t0), open = any(c -> c.open, allclosed(bq)))
end

@testset "the queue has nowhere to keep an open command buffer" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    bq = Mantle.batchqueue(dev)
    for f in (:active_batch, :free_batches, :free_cmd_bufs, :as_cmd_buf, :as_fence,
              :staging, :auto_submit_threshold, :cb_split_threshold)
        @test !hasfield(typeof(bq), f)
    end
    # One record of what is in flight: the submission rides in `outstanding` as
    # the token's payload, and there is no second list swept against the same
    # counter by a second function.
    @test !hasfield(typeof(bq), :in_flight)
    @test hasfield(typeof(bq), :outstanding)
    @test eltype(bq.outstanding) == Mantle.Outstanding{UInt64, MVE.Submission}
    @test hasfield(typeof(bq), :free_oneshots)
end

@testset "every call closes and submits what it wrote" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    bq = Mantle.batchqueue(dev)
    be = Mantle.backend(dev)
    n = 256
    a = KA.allocate(be, Float32, n)
    k = ccb_bump!(be, 64)

    # An ad hoc launch: one one-shot, one submission, one head barrier.
    r = observe(bq) do
        k(a; ndrange = n)
    end
    @test r == (submits = 1, barriers = 1, tokens = 1, open = false)
    sub = last(bq.outstanding).payload
    @test length(sub.oneshots) == 1 && isempty(sub.recordings)

    # An upload and a download.
    b = Mantle.Buffer(dev, zeros(Float32, n))
    r = observe(bq) do
        Mantle.update!(b, fill(3f0, n))
    end
    @test r.submits == 1 && r.barriers == 1 && !r.open
    r = observe(bq) do
        @test Array(b) == fill(3f0, n)
    end
    @test r.submits == 1 && r.barriers == 1 && !r.open

    # A run of a recorded plan with nothing pending: exactly the recording,
    # which opened with its own head barrier when it was recorded.
    kref = Mantle.GPURef(dev, Int32(2))
    out = Mantle.Buffer(dev, zeros(Int32, n))
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.use(p, kref; read = true)
        Mantle.dispatch!(p, ccb_add!, (out, kref), n)
    end
    pl = Base.invokelatest(Mantle.Plan, g)
    r = observe(bq) do
        Mantle.record!(pl)
    end
    @test r == (submits = 0, barriers = 1, tokens = 0, open = false)
    r = observe(bq) do
        Mantle.run!(pl)
    end
    @test r == (submits = 1, barriers = 0, tokens = 1, open = false)
    sub = last(bq.outstanding).payload
    @test isempty(sub.oneshots) && length(sub.recordings) == 1
    @test only(sub.recordings) === pl.recording

    # With a store pending: the store's one-shot in front of the recording, in
    # ONE submission.
    kref[] = Int32(5)
    r = observe(bq) do
        Mantle.run!(pl)
    end
    @test r == (submits = 1, barriers = 1, tokens = 1, open = false)
    sub = last(bq.outstanding).payload
    @test length(sub.oneshots) == 1 && length(sub.recordings) == 1
    Mantle.waitfor!(pl)
    @test Array(out) == fill(Int32(7), n)

    # And after everything has passed, the sweep leaves nothing in flight and
    # every pooled one-shot clean.
    Mantle.flush!(bq)
    @test isempty(bq.outstanding)
    @test all(o -> !o.open && isempty(o.pinned) && isempty(o.regions), bq.free_oneshots)
    Mantle.free!(pl)
end

@testset "two uploads back to back are both correct with nothing waited on between" begin
    # The staging bytes belong to the one-shot that copies out of them; a
    # second upload never reuses a buffer the first is still reading. This is
    # what a queue-level staging buffer could not promise once every transfer
    # became its own submission.
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 1 << 16
    x, y = rand(Float32, n), rand(Float32, n)
    bx = Mantle.Buffer(dev, zeros(Float32, n))
    by = Mantle.Buffer(dev, zeros(Float32, n))
    Mantle.update!(bx, x)
    Mantle.update!(by, y)
    @test Array(bx) == x
    @test Array(by) == y
    for _ in 1:8
        x .= rand(Float32, n); y .= rand(Float32, n)
        Mantle.update!(bx, x)
        Mantle.update!(by, y)
    end
    @test Array(bx) == x
    @test Array(by) == y
end

@testset "a token beyond the timeline is refused, not waited for" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    bq = Mantle.batchqueue(dev)
    @test_throws MVE.LavaError Mantle.waitfor!(dev, bq.next_timeline + 1)
end

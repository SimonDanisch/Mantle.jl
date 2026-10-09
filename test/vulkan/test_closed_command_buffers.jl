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
And "no open command buffer" is pinned by the queue having nowhere to keep one:
it has no field for one, and every closed buffer it knows about is sealed.

**Why this file is still Vulkan's.** Everything here is counted: submissions,
head barriers and timeline values per call, read off the Vulkan context's
diagnostics (`ctx.diag.flush_counter`, `head_barriers`) and queue. No portable
count of submissions exists, so the counting assertions of three files live here
until one does: this file's, "a run is one submission" from
`test_recorded_run_semantics.jl`, and "a run's token is out the moment `run!`
returns, and neither wait submits" from `test_waitidle_submits.jl`. Their
results — the numbers a user sees — are asserted on every backend by those two
files and by `test_uploads_back_to_back.jl`, which took this file's two-uploads
testset.
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

"""Every closed buffer the channel holds, in flight or pooled."""
function allclosed(bq)
    cs = Any[]
    for o in bq.outstanding
        sub = o.payload
        sub.recording === nothing || push!(cs, sub.recording)
        # A plan's recording is not the payload — it is submitted again next
        # run — so it rides in the holds like anything else the commands name.
        # `nothing` when the submission held nothing at all, which is what a
        # recorded run with no store to land is.
        for h in something(sub.holds, ())
            h isa Mantle.Closed && push!(cs, h)
        end
    end
    append!(cs, bq.free)
    cs
end

"""What one call left behind: how many submissions, head barriers and timeline
values it produced, and whether anything is open afterwards."""
function observe(f, bq)
    d = Mantle.ctxof(bq).diag
    f0, h0, t0 = d.flush_counter[], d.head_barriers[], Mantle.driver(bq).next_timeline
    f()
    (submits = d.flush_counter[] - f0, barriers = d.head_barriers[] - h0,
     tokens = Int(Mantle.driver(bq).next_timeline - t0), open = any(c -> c.open, allclosed(bq)))
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
    @test eltype(bq.outstanding) ==
          Mantle.Outstanding{UInt64, Mantle.Submission{Union{Nothing,Mantle.OneShot}}}
    @test hasfield(typeof(bq), :free)
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
    @test sub.recording isa Mantle.OneShot

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
    Mantle.dispatch!(g, ccb_add!, (out, kref), n; name = "add")
    # `budget = Inf`: recorded whole, never measured. A plan on the device's
    # budget submits once per pass on its first run and is re-recorded after it
    # (see `partitionranges`), and this counts the steady state.
    pl = Base.invokelatest(Mantle.Plan, g; budget = Inf)
    r = observe(bq) do
        Mantle.record!(pl)
    end
    @test r == (submits = 0, barriers = 1, tokens = 0, open = false)
    r = observe(bq) do
        Mantle.run!(pl)
    end
    @test r == (submits = 1, barriers = 0, tokens = 1, open = false)
    sub = last(bq.outstanding).payload
    @test sub.recording === nothing
    # No hold frame at all, not an empty one — see `unhold!(::SubmitChannel,
    # ::Nothing)`: a submission that holds nothing is handed none, so a run in
    # a loop that never waits allocates zero.
    @test sub.holds === nothing

    # With a store pending: the store's one-shot in front of the recording, in
    # ONE submission.
    kref[] = Int32(5)
    r = observe(bq) do
        Mantle.run!(pl)
    end
    @test r == (submits = 1, barriers = 1, tokens = 1, open = false)
    sub = last(bq.outstanding).payload
    @test sub.recording isa Mantle.OneShot
    @test any(x -> x === pl.recording, sub.holds)
    Mantle.waitfor!(pl)
    @test Array(out) == fill(Int32(7), n)

    # And after everything has passed, the sweep leaves nothing in flight and
    # every pooled one-shot clean.
    Mantle.flush!(bq)
    @test isempty(bq.outstanding)
    @test all(o -> !o.open && isempty(o.sync) && isempty(o.regions), bq.free)
    Mantle.free!(pl)
end

@testset "a token beyond the timeline is refused, not waited for" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    bq = Mantle.batchqueue(dev)
    @test_throws Mantle.LavaError Mantle.waitfor!(dev, Mantle.driver(bq).next_timeline + 1)
end

# ── from test_recorded_run_semantics.jl ──────────────────────────────────────
#
# A run is one submission: the recording, and nothing else when nothing is
# pending.
#
# Two ways to get this wrong, both of them a queue reaching the driver more than
# once per run: force-closing whatever batch is open, submitting it, then
# submitting the recording behind a hand-rolled wait on the newest outstanding
# timeline value; or appending the run to an open batch that goes when something
# else asks, so the run submits nothing at all. A run is handed over the moment
# it is called, as ONE `vkQueueSubmit2` carrying the recording, and when a store
# is waiting, the one-shot that lands it in front. Two passes, so the count is
# not accidentally right because there is only one thing to submit.
@kernel function ccb_addref!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

function ccb_twopassplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    for name in ("a", "b")
        Mantle.dispatch!(g, ccb_addref!, (out, kref), n; name = name)
    end
    Mantle.Plan(g; budget = Inf)
end

@testset "a run with nothing pending submits exactly the recording" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.backend(dev)
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(1))
    pl = Base.invokelatest(ccb_twopassplan, dev, out, kref, n)
    Mantle.record!(pl)
    KA.synchronize(be)

    bq = Mantle.batchqueue(dev)
    runs = 10
    before = Mantle.ctxof(bq).diag.flush_counter[]
    for i in 1:runs
        kref[] = Int32(i)
        Mantle.run!(pl)
        # A store was pending: its one-shot went in front of the recording, in
        # the same submission.
        # The payload is core's `Submission`: the one-shot the submission GIVES
        # BACK, and the holds it must outlive — which include the recording,
        # because that is submitted again next run and stays the plan's.
        sub = last(bq.outstanding).payload
        @test sub.recording isa Mantle.OneShot
        @test any(x -> x === pl.recording, sub.holds)
    end
    @test Mantle.ctxof(bq).diag.flush_counter[] - before == runs
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Int32(2 * sum(1:runs)), n)

    # Nothing pending: the recording alone, and nothing allocated for it.
    before = Mantle.ctxof(bq).diag.flush_counter[]
    Mantle.run!(pl)
    @test Mantle.ctxof(bq).diag.flush_counter[] - before == 1
    sub = last(bq.outstanding).payload
    @test sub.recording === nothing
    # Nothing was opened, so there is no hold frame and nothing to hold: the
    # plan owns the recording and `release!` is what waits for it. `nothing`
    # and not an empty list — a submission that holds nothing is handed none,
    # which is what makes a recorded run allocate zero bytes
    # (`test_run_allocates_nothing.jl`, and RayMakie's render loop).
    @test sub.holds === nothing
    KA.synchronize(be)
    # And nothing is left claiming to be in flight once the device has caught up.
    Mantle.sweep!(bq)
    @test isempty(Mantle.outstanding(bq))

    Mantle.free!(pl)
end

# ── from test_waitidle_submits.jl ────────────────────────────────────────────
#
# There is no open batch: a run is submitted the moment `run!` is called, and
# the token it answers with is out before `run!` returns — the token is the
# queue's newest the moment it exists — and `waitidle` and `waitfor!` then wait
# for it without submitting anything more.
@testset "a run's token is out when run! returns, and waiting submits nothing" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(7))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, ccb_addref!, (out, kref), n; name = "bump")
    pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g; budget = Inf))
    bq = Mantle.batchqueue(dev)
    lasttoken(pl) = Mantle.argtoken(pl.args)

    flushes = Mantle.ctxof(bq).diag.flush_counter[]
    Mantle.run!(pl)
    tok = lasttoken(pl)
    # Submitted the moment it was run: the token is the queue's newest, and
    # the run was one submission.
    @test tok == Mantle.driver(bq).next_timeline
    @test Mantle.ctxof(bq).diag.flush_counter[] == flushes + 1
    Mantle.waitidle(dev)
    @test Mantle.passed(dev, tok)
    # …and nothing was submitted to get there: there was nothing left to hand over.
    @test Mantle.ctxof(bq).diag.flush_counter[] == flushes + 1

    for k in Int32[3, 5, 11]
        kref[] = k
        Mantle.run!(pl)
        tok = lasttoken(pl)
        @test tok == Mantle.driver(bq).next_timeline       # out the moment the run returned
        Mantle.waitfor!(pl)
        @test Mantle.passed(dev, tok)
    end

    # Once the device has caught up it is a no-op, not another submission.
    flushes = Mantle.ctxof(bq).diag.flush_counter[]
    Mantle.waitfor!(pl)
    @test Mantle.ctxof(bq).diag.flush_counter[] == flushes
    Mantle.free!(pl)
end

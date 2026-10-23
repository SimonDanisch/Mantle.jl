# `run!(recorded_plan)` allocates nothing.
#
# The whole point of recording: the per-run host work is the pending updates,
# the pending move patches, the arena claim and one `submit!` — and with none
# pending it is a handful of field reads and two vector pushes, none of which
# may allocate. A run that allocates is a frame loop paying GC pauses, and the
# regression that reintroduces it is small and quiet: an `Any`-typed field that
# boxes the timeline value it is handed, a `lock do` closure that escapes
# inlining, a fresh info struct per submission. This test is the cliff
# detector: it fails on the FIRST byte, and the allocation profiler's stack
# says whose.
#
# `waitfor!` is in the loop because a renderer's is: it submits the batch the
# run sealed and blocks on the timeline, so the measurement covers the full
# steady-state submission cycle, not just the seal.
#
# Counted by `allocations_per_run` (`runallocations.jl`), which says why it counts
# the allocation profiler's stacks rather than `@allocated`.

using Test, Mantle, Profile
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle
include(joinpath(@__DIR__, "spinwork.jl"))
include(joinpath(@__DIR__, "runallocations.jl"))

KA.@kernel function _allocfree_step!(a)
    i = KA.@index(Global)
    @inbounds a[i] = a[i] + 1f0
end

function _allocfree_plan(dev, n; budget)
    a = Mantle.Buffer(dev, zeros(Float32, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, _allocfree_step!, (a,), n; name = "step")
    pl = Mantle.Plan(g; budget)
    Mantle.record!(pl)
    return pl, a
end

@testset "run! of a recorded plan allocates nothing" begin
    dev = Mantle.Device(TESTBACKEND)
    # `budget = Inf` is recorded whole and never measures; the device's budget
    # measures itself in the warm-up and is re-recorded from it, and the steady
    # state after that has to be just as free.
    for budget in (Inf, Mantle.submissionbudget(dev))
        pl, a = _allocfree_plan(dev, 256; budget)
        per_run = allocations_per_run(pl)
        # …and the run still ran: 300 cycles of +1 from 0.
        @test all(==(300f0), Array(Mantle.storage(a)))
        @test per_run == 0
        # A device that times its passes measured every one of them; one without
        # timestamps keeps a single submission and has nothing to measure.
        isfinite(budget) && Mantle.timestamps(dev) && @test !any(isnan, pl.passcost)
        Mantle.free!(pl)
    end
end

# A loop: the body's pieces are submitted once per iteration, from vectors the
# recording kept, and a measured loop is still one submission.
function _allocfree_loopplan(dev, n, iters)
    a = Mantle.Buffer(dev, zeros(Float32, n))
    g = Mantle.Graph(dev)
    Mantle.repeat!(g, iters) do i
        Mantle.dispatch!(g, _allocfree_step!, (a,), n; name = "step")
    end
    pl = Mantle.Plan(g)
    Mantle.record!(pl)
    return pl, a
end

@testset "run! of a recorded loop allocates nothing" begin
    dev = Mantle.Device(TESTBACKEND)
    pl, a = _allocfree_loopplan(dev, 256, 5)
    per_run = allocations_per_run(pl)
    @test all(==(1500f0), Array(Mantle.storage(a)))
    @test length(pl.recording.batches) == 1
    @test per_run == 0
    Mantle.free!(pl)
end

# The same cycle with GPU work that outlasts the wait's poll, so every `waitfor!` hands
# its wait to a worker thread (`Mantle.hostwait`), and handing it over has to be as free.
# The dispatches above are short enough that a wait reaches a worker only now and then:
# 2 waits in 200 did on an RTX 4000, and their 308 bytes each failed the test above at
# 3 bytes per run, while a machine that polls a little longer would have passed it.
@testset "run! of a recorded plan allocates nothing when its wait reaches a worker" begin
    dev = Mantle.Device(TESTBACKEND)
    pl = spinplan(dev, spinsteps(dev, 0.01))
    @test allocations_per_run(pl) == 0
    Mantle.free!(pl)
end

# A plan that syncs MANY buffers is not buildable here: a `compute!` dispatch
# names its external arrays by address and pins none (see the submission
# refactor notes), so only a trace plan stamps dozens of buffers a run. That
# zero — a per-buffer stamp at submit boxes 48 bytes each — is pinned by
# Hikari's `test_sample_is_one_run.jl` on a real ray-tracing plan.

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
# Everything measured lives inside `cycle`/`measure` — locals, not globals —
# because a non-const global read boxes its way into the count and a fresh
# world age recompiles into it too.
#
# Counted by the allocation profiler's STACKS, not by `@allocated`. `@allocated`
# reads the process-wide counter, and `waitfor!` waits: whatever else runs during
# that wait is counted as the run's. Inside a bt session that is its own timer task,
# 112 bytes every ~250 ms, which failed this test at 1.1-1.7 bytes per run while
# the profiler recorded no allocation from the run at all. An allocation is the
# run's when one of the frames on its stack is in a package the run goes through.

using Test, Mantle, Profile
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

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

function _allocfree_measure(pl)
    cycle() = (Mantle.run!(pl); Mantle.waitfor!(pl))
    # Warm: compile the path, grow every pool and vector to steady state.
    for _ in 1:50
        cycle()
    end
    GC.gc()
    for _ in 1:50
        cycle()
    end
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 1 for _ in 1:200
        cycle()
    end
    ours = filter(byrun, Profile.Allocs.fetch().allocs)
    # Whose, when there is one: the stack is the whole point of counting this way.
    for a in first(ours, 3)
        @info "an allocation by the run" a.type a.size stack = first(a.stacktrace, 16)
    end
    return sum(a -> a.size, ours; init = 0) / 200
end

"""Whether an allocation was made by the code under test: one of its frames is in the
SOURCE of a package a run goes through (`src/` or `lib/`, so not this test file)."""
byrun(a) = any(f -> occursin(RUN_PACKAGES, string(f.file)), a.stacktrace)

const RUN_PACKAGES = r"/(Mantle|Metal|Lava|ObjectiveC|KernelAbstractions|KernelInterface|GPUArrays|Vulkan)/([^/]+/)?(src|lib)/"

@testset "run! of a recorded plan allocates nothing" begin
    dev = Mantle.Device(TESTBACKEND)
    # `budget = Inf` is recorded whole and never measures; the device's budget
    # measures itself in the warm-up and is re-recorded from it, and the steady
    # state after that has to be just as free.
    for budget in (Inf, Mantle.submissionbudget(dev))
        pl, a = _allocfree_plan(dev, 256; budget)
        per_run = _allocfree_measure(pl)
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
    per_run = _allocfree_measure(pl)
    @test all(==(1500f0), Array(Mantle.storage(a)))
    @test length(pl.recording.batches) == 1
    @test per_run == 0
    Mantle.free!(pl)
end

# A plan that syncs MANY buffers is not buildable here: a `compute!` dispatch
# names its external arrays by address and pins none (see the submission
# refactor notes), so only a trace plan stamps dozens of buffers a run. That
# zero — a per-buffer stamp at submit boxes 48 bytes each — is pinned by
# Hikari's `test_sample_is_one_run.jl` on a real ray-tracing plan.

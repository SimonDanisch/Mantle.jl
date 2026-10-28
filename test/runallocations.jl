# What a recorded plan's steady-state cycle allocates: `test_run_allocates_nothing.jl`
# and `metal/test_mtl4_hostwait.jl` count with it. Included after `using Profile`.
#
# Everything measured lives inside `cycle`/`allocations_per_run` — locals, not globals —
# because a non-const global read boxes its way into the count and a fresh world age
# recompiles into it too.
#
# Counted by the allocation profiler's STACKS, not by `@allocated`. `@allocated` reads
# the process-wide counter, and `waitfor!` waits: whatever else runs during that wait is
# counted as the run's. Inside a bt session that is its own timer task, 112 bytes every
# ~250 ms, which failed `test_run_allocates_nothing.jl` at 1.1-1.7 bytes per run while
# the profiler recorded no allocation from the run at all. Which allocations are the
# run's is `byrun`'s question.

"""
    allocations_per_run(pl) -> Float64

The bytes the run allocates per `run!`/`waitfor!` cycle of the recorded plan `pl`, at
steady state, logging the stacks of the first few allocations.
"""
function allocations_per_run(pl)
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
        @info "an allocation by the run" a.type a.size stack = Text(join(first(a.stacktrace, 24), "\n"))
    end
    return sum(a -> a.size, ours; init = 0) / 200
end

"""
Whether an allocation was made by the code under test: walking out from the
allocation, a frame in the SOURCE of a package a run goes through (`src/` or `lib/`,
so not this test file) comes before any frame of the event loop.

The event loop is the line between the run and everything else that happens while it
waits. A wait yields, and the scheduler's `process_events` runs whatever libuv has
pending on the waiting task's stack — a bt session's timer callback looks up its
`cfunction` there, 32 + 24 + 24 bytes under the run's own `poll_completed`. Those
callbacks belong to whoever registered them.

GPUToolbox is among the packages because a wait that outlasts its poll runs on one of
GPUToolbox's worker threads, and what GPUToolbox allocates there has no other package's
frame on its stack.
"""
function byrun(a)
    for f in a.stacktrace
        f.func in EVENT_LOOP && return false
        occursin(RUN_PACKAGES, string(f.file)) && return true
    end
    return false
end

const EVENT_LOOP = (:uv_run, :ijl_process_events, :process_events)

const RUN_PACKAGES = r"/(Mantle|Metal|Lava|ObjectiveC|KernelAbstractions|KernelInterface|GPUArrays|Vulkan|GPUToolbox)/([^/]+/)?(src|lib)/"

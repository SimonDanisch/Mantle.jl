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

using Test, Mantle, Lava
import KernelAbstractions as KA

KA.@kernel function _allocfree_step!(a)
    i = KA.@index(Global)
    @inbounds a[i] = a[i] + 1f0
end

function _allocfree_plan(dev, n)
    a = Mantle.Buffer(dev, zeros(Float32, n))
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "step") do p
        Mantle.use(p, a; read = true, write = true)
        Mantle.dispatch!(p, _allocfree_step!, (a,), n)
    end
    pl = Mantle.Plan(g)
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
    bytes = @allocated for _ in 1:200
        cycle()
    end
    return bytes / 200
end

@testset "run! of a recorded plan allocates nothing" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    pl, a = _allocfree_plan(dev, 256)
    per_run = _allocfree_measure(pl)
    # …and the run still ran: 300 cycles of +1 from 0.
    @test all(==(300f0), Array(Mantle.storage(a)))
    @test per_run == 0
    Mantle.free!(pl)
end

# A plan that syncs MANY buffers is not buildable here: a `compute!` dispatch
# names its external arrays by address and pins none (see the submission
# refactor notes), so only a trace plan stamps dozens of buffers a run. That
# zero — `sync_access!`'s stamp used to box 48 bytes a buffer — is pinned by
# Hikari's `test_sample_is_one_run.jl` on a real ray-tracing plan.

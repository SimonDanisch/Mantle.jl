"""
`record!` writes commands. It does not run the plan.

This was false under `bake!`, and the docstring on `capture` said so out loud —
"Run `f` once, recording and executing it normally". Callers had to compensate:
Hikari's note on why its volumetric path is not baked lists, among the traps,
that "`bake!` RUNS the plan as it captures it (so the accumulate pass would
contribute a spurious sample)". A graph that models every operation should not
need the caller to correct for when they happen; deciding that is the graph's
whole job.

**It was never a Vulkan constraint.** `vkEndCommandBuffer` and `vkQueueSubmit`
are separate calls and a command buffer may be submitted zero times. The cause
was ours: `capture` ran the ordinary recorder and collected whatever `submit!`
sealed on the way past, so a recording was a side effect of executing. `record!`
writes into a command buffer of its own and ends it; nothing reaches the queue.

The assertion is a counter the plan increments. After `record!` it must still
read its initial value — nothing ran — and after `run!` it must have moved.
Reading it is the only honest test: a submission count or a timeline value would
say something about bookkeeping, and the question is whether the GPU did the
work.

The plan below is deliberately more than one dispatch. Under `bake!` a recording
could span several submit boundaries (the submit threshold split a long one),
so an implementation that sealed only the LAST batch would pass a single-dispatch
test and execute everything before it. There is one command buffer now, which is
what makes that unreachable rather than merely untriggered — and this asserts it
directly.

A note on the kernel name, because it cost a suite run to find: `@kernel function
NAME` also defines a helper called `_NAME` (KernelAbstractions `macros.jl:53`)
and wraps the whole constructor block in `if !@isdefined(_NAME)`. So a kernel
named `capture_bump!` collides with the helper of a kernel named `bump!` — which
`test_arena_recording.jl` defines — finds itself already defined, and never gets its
`capture_bump!(dev)` constructor at all. It fails much later and somewhere else, as
`MethodError: no method matching capture_bump!(::LavaBackend)` from inside `kernelfor`,
with the three-argument helper listed as the only candidate. Reading that as a
world-age problem is the obvious mistake and `invokelatest` does not fix it. No
leading underscore on a kernel name.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function capture_bump!(c)
    i = @index(Global)
    @inbounds c[i] += Int32(1)
end

"""
The plan under test: four passes, each adding one to the same counter.
"""
function _bumpplan(dev, counter, n)
    g = Mantle.Graph(dev)
    for k in 1:4
        Mantle.compute!(g, "bump$k") do p
            Mantle.use(p, counter; read = true, write = true)
            Mantle.dispatch!(p, capture_bump!, (counter,), n)
        end
    end
    Mantle.Plan(g)
end

@testset "record! writes without executing" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 64

    counter = Mantle.Buffer(dev, zeros(Int32, n))
    pl = Base.invokelatest(_bumpplan, dev, counter, n)

    @test all(==(Int32(0)), Array(Mantle.storage(counter)))

    Mantle.record!(pl)
    KA.synchronize(be)
    # THE assertion. Under `bake!` this read 4: the plan had run as a side effect
    # of being recorded.
    @test Array(Mantle.storage(counter)) == zeros(Int32, n)
    @test Mantle.recorded(pl)

    # ONE command buffer, which is what makes the paragraph above structural.
    # `bake!` produced a LIST, cut by the submit threshold firing mid-capture
    # — measured at five per recording on Hikari's fused sample, a threshold
    # about when to submit applied while nothing was being submitted. It was
    # then one per argument slot, and now it is one, because nothing rewrites a
    # plan's argument memory between runs.
    @test pl.recording isa MVE.Recording
    @test !pl.recording.open

    # …and the recording is real: running it does the work.
    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(counter)) == fill(Int32(4), n)

    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(counter)) == fill(Int32(8), n)

    Mantle.free!(pl)
end

@testset "run! never records: an unrecorded plan is refused, record! records once" begin
    # `record!` IS something a caller has to do, once, after building the plan:
    # `run!` submits that recording and never records for itself, so a plan
    # that was never recorded is refused rather than recorded on the owning
    # thread's first frame. Every run after the one `record!` submits the same
    # command buffer.
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 64
    counter = Mantle.Buffer(dev, zeros(Int32, n))
    pl = Base.invokelatest(_bumpplan, dev, counter, n)

    @test !Mantle.recorded(pl)
    @test_throws ArgumentError Mantle.run!(pl)
    @test !Mantle.recorded(pl)
    Mantle.record!(pl)
    @test Mantle.recorded(pl)
    Mantle.run!(pl)
    first = pl.recording
    for _ in 1:5
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    # Same object, so nothing was recorded again.
    @test pl.recording === first
    @test Array(Mantle.storage(counter)) == fill(Int32(24), n)

    Mantle.free!(pl)
end

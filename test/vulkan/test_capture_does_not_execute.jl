"""
`bake!` records. It does not run the plan.

This was false, and the docstring on `capture` said so out loud — "Run `f` once,
recording and executing it normally". Callers had to compensate: Hikari's note on
why its volumetric path is not baked lists, among the traps, that "`bake!` RUNS
the plan as it captures it (so the accumulate pass would contribute a spurious
sample)". A graph that models every operation should not need the caller to
correct for when they happen; deciding that is the graph's whole job.

**It was never a Vulkan constraint.** `vkEndCommandBuffer` and `vkQueueSubmit`
are separate calls and a command buffer may be submitted zero times — which
Mantle already relied on, since `cb_begin_flags` begins capture buffers with
`SIMULTANEOUS_USE` rather than `ONE_TIME_SUBMIT` precisely so they can be
replayed. The cause was ours: sealing a batch and queueing it were one function,
`submit!`, and `capture` called it only to reach the collection step buried in
the middle. So a capture executed as a side effect of being collected.

The assertion is a counter the plan increments. After `bake!` it must still read
its initial value — nothing ran — and after `run!` it must have moved. Reading it
is the only honest test: `submissions` or a timeline value would say something
about bookkeeping, and the question is whether the GPU did the work.

A capture can span several submit boundaries (`auto_submit_threshold` splits a
long recording), so the plan below is deliberately more than one dispatch: a
version that only sealed the LAST batch would pass a single-dispatch test and
execute everything before it.

A note on the kernel name, because it cost a suite run to find: `@kernel function
NAME` also defines a helper called `_NAME` (KernelAbstractions `macros.jl:53`)
and wraps the whole constructor block in `if !@isdefined(_NAME)`. So a kernel
named `capture_bump!` collides with the helper of a kernel named `bump!` — which
`test_arena_bake.jl` defines — finds itself already defined, and never gets its
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

Several passes on purpose — the recording then crosses more than one batch
boundary, so an implementation that sealed only the TRAILING batch into the
capture would execute the first three and still pass a single-pass test.

A function, and reached through `invokelatest`, which is how every graph-building
helper in this suite is called — see `test_arena_bake.jl`.
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

@testset "bake! records without executing" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 64

    counter = Mantle.Buffer(dev, zeros(Int32, n))
    pl = Base.invokelatest(_bumpplan, dev, counter, n)

    @test all(==(Int32(0)), Array(Mantle.storage(counter)))

    Mantle.bake!(pl)
    KA.synchronize(be)
    # THE assertion. Under the old capture this read 4: the plan had run as a
    # side effect of being recorded.
    @test Array(Mantle.storage(counter)) == zeros(Int32, n)

    # …and the recording is real: replaying it does the work.
    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(counter)) == fill(Int32(4), n)

    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(counter)) == fill(Int32(8), n)

    Mantle.free!(pl)
end

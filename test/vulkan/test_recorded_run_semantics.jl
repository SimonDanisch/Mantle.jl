"""
Recording changes WHEN commands are built. It does not change what a run means.

Three properties, and all three have been false at some point.

**`record!` writes without executing.** `capture` used to fall through into
`submit!`, so a plan ran as a side effect of being captured. It was never a
Vulkan constraint — `vkEndCommandBuffer` and `vkQueueSubmit2` are separate calls
— it was `submit!` doing two jobs, with the collection step capture needed buried
between them. `test_record_does_not_execute.jl` is that property on its own.

**A per-run value reaches the run.** This is what a `GPURef` is for. A dispatch
is packed with the ref's device ADDRESS, once, at `record!`; the value behind it
is written by a store — `ref[] = x` — which lands as a command ahead of the
recording in the run's own submission. So the same recording reads a different
number every run, and it does so without the host touching argument memory at
all.

It was a `Ref` argument, re-read and REPACKED every run, and that cost the
argument ring: 602 host stores per run on Hikari's fused sample to move 268
bytes, into memory an in-flight submission could still be reading, so the plan
needed three copies of its arguments and a wait to rotate between them. One
indirection deletes all of it.

**And argument memory does NOT move.** The other half of the same statement, and
the one that has to be pinned or the first is meaningless: the bytes a plan's
dispatches read their arguments from are written at `record!` and not again. A
run with stores between leaves them byte-identical, and a `Ref` — which used to
mean the opposite — is refused at the declaration, however deep in the
arguments it sits.

The assertion for the first is arithmetic: the same graph, run with a sequence of
values, must accumulate their sum. There is no unbaked half to compare against
and that is the point — the old test would have passed if both paths were wrong
the same way.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function baked_addk!(out, k)
    i = @index(Global)
    @inbounds out[i] += k
end

# A `GPURef` arrives as a one-element device array, so the kernel reads through
# it. That is the indirection, and it is the whole mechanism: the address in the
# packed arguments never changes, only what is behind it.
@kernel function baked_addref!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

"""The graph under test: adds `kref`'s current value to `out` on every run."""
function _refplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.use(p, kref; read = true)
        Mantle.dispatch!(p, baked_addref!, (out, kref), n)
    end
    Mantle.Plan(g)
end

# Graph builders go through `invokelatest`, as they do everywhere in this suite.
# On the kernel NAME, which is not a free choice: see the note in
# `test_record_does_not_execute.jl`. No leading underscore.
_plan(f, args...) = Base.invokelatest(f, args...)

"""Two passes over one buffer, so a submission count is not right by accident."""
function _twopassplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    for name in ("a", "b")
        Mantle.compute!(g, name) do p
            Mantle.use(p, out; read = true, write = true)
            Mantle.use(p, kref; read = true)
            Mantle.dispatch!(p, baked_addref!, (out, kref), n)
        end
    end
    Mantle.Plan(g)
end

"""The same graph with a constant where the `GPURef` was: nothing can move."""
function _constplan(dev, out, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.dispatch!(p, baked_addk!, (out, Int32(5)), n)
    end
    Mantle.Plan(g)
end

"""The plan's argument memory, byte for byte."""
argbytes(pl) = copy(unsafe_wrap(Array, pl.args.ptr, length(pl.args.store)))

@testset "a recorded run reads the value its GPURef holds now" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    # Distinct, and none a multiple of another, so a run that read a neighbour's
    # value cannot land on the right sum by accident.
    steps = Int32[3, 7, 11, 13, 17, 19, 23, 29]
    reference = fill(sum(steps), n)

    out_b = Mantle.Buffer(dev, zeros(Int32, n))
    kref_b = Mantle.GPURef(dev, Int32(0))
    pl_b = _plan(_refplan, dev, out_b, kref_b, n)
    # Both are read by the pass, so both are what a store can be waiting on.
    @test pl_b.hostwritten == (out_b, kref_b)

    kref_b[] = steps[1]
    Mantle.record!(pl_b)
    KA.synchronize(be)
    # `record!` wrote commands and did not run — and did not land the store.
    @test Array(Mantle.storage(out_b)) == zeros(Int32, n)
    @test Mantle.isdirty(kref_b)
    # ONE recording. It was one per argument slot, and the slots existed because
    # a run rewrote argument bytes on the host; nothing does.
    @test pl_b.recording isa MVE.Recording

    for k in steps
        kref_b[] = k
        Mantle.run!(pl_b)
    end
    KA.synchronize(be)

    # THE assertion. A frozen value gives `length(steps)*steps[1]`; a store
    # that landed behind the recording that reads it gives something in between.
    @test Array(Mantle.storage(out_b)) == reference
    # Consumed: nothing is waiting once a run has taken it.
    @test !Mantle.anydirty(pl_b.hostwritten)

    Mantle.free!(pl_b)
end

# A run is one submission: the recording, and nothing else when nothing is
# pending.
#
# `replay!` used to reach the queue twice per run: force-close whatever batch was
# open, submit it, then submit the recording behind a hand-rolled wait on the
# newest outstanding timeline value. Then a run was appended to an open batch
# that went when something else asked — a flush, a readback, a threshold — so
# a run submitted nothing at all and the test here asserted exactly that. There
# is no open batch now: a run is handed over the moment it is called, as ONE
# `vkQueueSubmit2` carrying the recording — and, when a store is waiting, the
# one-shot that lands it in front. Two passes, so the count is not accidentally
# right because there is only one thing to submit.
@testset "a run with nothing pending submits exactly the recording" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(1))
    pl = _plan(_twopassplan, dev, out, kref, n)
    Mantle.record!(pl)
    KA.synchronize(be)

    bq = Mantle.batchqueue(dev)
    runs = 10
    before = MVE.ctxof(bq).diag.flush_counter[]
    for i in 1:runs
        kref[] = Int32(i)
        Mantle.run!(pl)
        # A store was pending: its one-shot went in front of the recording, in
        # the same submission.
        # The payload is core's `Submission`: the one-shot the submission GIVES
        # BACK, and the holds it must outlive — which include the recording,
        # because that is submitted again next run and stays the plan's.
        sub = last(bq.outstanding).payload
        @test sub.recording isa MVE.OneShot
        @test any(x -> x === pl.recording, sub.holds)
    end
    @test MVE.ctxof(bq).diag.flush_counter[] - before == runs
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Int32(2 * sum(1:runs)), n)

    # Nothing pending: the recording alone, and nothing allocated for it.
    before = MVE.ctxof(bq).diag.flush_counter[]
    Mantle.run!(pl)
    @test MVE.ctxof(bq).diag.flush_counter[] - before == 1
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

# A plan with nothing stored does no host work between `run!` and the queue: no
# store is waiting, so `emitupdates!` writes nothing and the run is one `submit!`.
@testset "a plan with nothing stored writes nothing per run" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    pl = _plan(_constplan, dev, out, n)
    Mantle.record!(pl)
    KA.synchronize(be)

    # `out` is host-writable — every declared buffer is — and clean.
    @test pl.hostwritten == (out,)
    @test !Mantle.anydirty(pl.hostwritten)
    for _ in 1:4
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Int32(20), n)

    Mantle.free!(pl)
end

# The invariant this whole design rests on, asserted directly: the plan's
# argument memory is written at `record!` and never again. Stores between runs
# land as commands in the run's submission and touch only the resource they
# name; the pointer packed into the arguments stays what it was.
@testset "argument memory is byte-identical across runs with stores between" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(0))
    pl = _plan(_twopassplan, dev, out, kref, n)
    Mantle.record!(pl)
    before = argbytes(pl)
    @test !isempty(before)
    for k in Int32[5, 9, 13, 21]
        kref[] = k
        Mantle.run!(pl)
    end
    Mantle.waitfor!(pl)
    @test argbytes(pl) == before
    @test Array(Mantle.storage(out)) == fill(Int32(2 * (5 + 9 + 13 + 21)), n)
    Mantle.free!(pl)
end

# The contract that makes the first testset mean something: a `Ref` used to
# mean "read this fresh every run", and honouring that is what the write plan,
# the argument ring and `rebind!` all existed for. Nothing rewrites argument
# memory now, so a `Ref` would silently freeze at whatever it held when the
# plan recorded — and it is refused at the declaration instead, wherever it
# sits in the arguments. Nested is the case that matters: a `Ref` inside a
# camera struct is what a renderer actually passed.
struct RefHolder
    r::Base.RefValue{Int32}
end

# A mutable struct is a handle that owns its insides, and the walker stops at
# it: an acceleration structure holds the driver's objects, which reference
# each other in cycles, and walking into one overflowed the stack on every
# hardware-traced plan. So a cycle is fine, and so is a `Ref` behind a handle.
mutable struct CyclicHandle
    self::Any
    r::Base.RefValue{Int32}
end

@testset "a Ref argument is refused, nested or not" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        @test_throws ArgumentError Mantle.dispatch!(p, baked_addk!, (out, Ref(Int32(5))), n)
        @test_throws ArgumentError Mantle.dispatch!(p, baked_addk!, (out, (Ref(Int32(5)),)), n)
        @test_throws ArgumentError Mantle.dispatch!(p, baked_addk!, (out, (k = Ref(Int32(5)),)), n)
        @test_throws ArgumentError Mantle.dispatch!(p, baked_addk!, (out, RefHolder(Ref(Int32(5)))), n)
        # …and the message names the replacement.
        err = try
            Mantle.dispatch!(p, baked_addk!, (out, Ref(Int32(5))), n)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("GPURef", err.msg)
        # A value is accepted; so is a resource, whose insides are its own, and
        # so is a mutable handle — even one that references itself.
        Mantle.dispatch!(p, baked_addk!, (out, Int32(5)), n)
        handle = CyclicHandle(nothing, Ref(Int32(5)))
        handle.self = handle
        Mantle.dispatch!(p, baked_addk!, (out, handle), n)
    end
    @test length(only(g.passes).dispatches) == 2
    Mantle.free!(out)
end

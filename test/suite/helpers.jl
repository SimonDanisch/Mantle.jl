# What every area file uses: the case macro, the device being tested, counters
# and memory figures read through Mantle's test hooks, and the instruments for
# concurrency cases (slow kernels, blocking host functions, watchdogs,
# canaries).
#
# Test hooks the refactor adds to Mantle for this suite (ledger P0, "suite
# hooks"; none of them is user API):
#   Mantle.diagnostics(dev) -> Diagnostics   counters since the last reset:
#       plancompiles, pipelinecompiles, kernelcompiles, records (recorded
#       parts, BuildRecordings, profile variants and pieces included),
#       prefixrecords, eagerrecords (temporary native encodings), submissions (per channel),
#       allocations (acquire! calls), hostwaits
#   Mantle.resetdiagnostics!(dev)
#   Mantle.reserved(dev), Mantle.mapped(dev)  pool bytes
#   Mantle.liveuploads(dev), Mantle.livecells(dev)
#   Mantle.pendingof(x) -> Vector{Change}     a resource's pending requests
#   Mantle.steplog!(dev, on) / Mantle.steplog(dev)   prefix steps applied:
#       (kind, target, seq, token) per step, in emission order
#   Mantle.checkrecord!(dev, on), withrecordguard(f, dev), readresourcesnapshot(a)
#       reject semantic resource reads during recording; cellaccesses(g) is a compiled-origin audit
#   Mantle.checklocks!(dev, on)               assert the lock order and that
#       no sync-state, channel, pool or window lock is held in a host wait

using Test, Mantle, Random
import KernelInterface as KI        # kernels are plain functions over KI's intrinsics

"""The suite's settings: the one global, set by runsuite."""
struct SuiteConfig
    phase::Base.RefValue{Int}
    long::Base.RefValue{Bool}
    device::Base.RefValue{Union{Nothing,Mantle.Device}}
    executed::Base.RefValue{Int}
    fixtures::Base.RefValue{NamedTuple}
end
const SUITE = SuiteConfig(Ref(0), Ref(false), Ref{Union{Nothing,Mantle.Device}}(nothing), Ref(0),
                          Ref{NamedTuple}((; devices = (), driverpairs = ())))

testdevice() = SUITE.device[]
devicename(dev) = string(nameof(typeof(dev)))

# The phase-1 vertical slice supplies Device/MantleArray/Graph/run! and the
# diagnostics hooks. Every case runs at its own phase: a missing foundation
# must fail visibly instead of making the first five phase gates empty.

"""
    @case id phase [needs...] begin … end

One case: a testset named by its id. Skipped when its
ledger phase is later than the suite's, when it is `:long` and long cases are
off, or when the device lacks a capability in `needs` (`:rt`, `:sparse`,
`:multiqueue`, `:window`, `:vendorview`, `:metal`,
`:lavapipe`, `:graphics`).
"""
macro case(id, phase, args...)
    body = last(args)
    needs = collect(args[1:end-1])
    quote
        local _id = $(esc(id))
        local _needs = Symbol[$(map(esc, needs)...)]
        if $(esc(phase)) > SUITE.phase[] || (:long in _needs && !SUITE.long[]) ||
           !all(n -> n === :long || has(testdevice(), n), _needs)
            @testset "$_id" begin
                @test_skip $(string(id))
            end
        else
            SUITE.executed[] += 1
            @testset "$_id" begin
                $(esc(body))
            end
        end
    end
end

# Capabilities, from caps(dev) (field names are G3's; written here once).
has(dev, ::Val{:rt}) = caps(dev).raytracing
has(dev, ::Val{:graphics}) = caps(dev).graphics       # render passes and draws (not CUDA, HIP)
has(dev, ::Val{:sparse}) = caps(dev).sparseresidency
has(dev, ::Val{:multiqueue}) = length(unique(q.state for q in Mantle.channelinfo(dev))) > 1
has(dev, ::Val{:multicompute}) = length(unique(q.state for q in Mantle.channelinfo(dev) if q.compute)) > 1
has(dev, ::Val{:transferqueue}) = any(Mantle.channelinfo(dev)) do q
    q.transfer && any(c -> c.compute && c.state !== q.state, Mantle.channelinfo(dev))
end
has(dev, ::Val{:window}) = caps(dev).display
has(dev, ::Val{:vendorview}) = caps(dev).api === :cuda
has(dev, ::Val{:metal}) = caps(dev).api === :metal
has(dev, ::Val{:lavapipe}) = occursin("llvmpipe", lowercase(Mantle.name(dev)))
# Fixture requirements are distinct from device features. A CPU-only pass
# cannot satisfy the default-hardware-device or paired-driver cases.
defaultcandidate(dev) = caps(dev).api in (:vulkan, :metal) && !caps(dev).software
has(dev, ::Val{:defaultgpu}) = any(defaultcandidate, SUITE.fixtures[].devices)
has(dev, ::Val{:nodefaultgpu}) = !has(dev, :defaultgpu)
has(dev, ::Val{:twodevices}) = length(SUITE.fixtures[].devices) > 1
vulkanpeers(dev) = [v for (cu, v) in SUITE.fixtures[].driverpairs if cu === dev]
has(dev, ::Val{:vulkanpeer}) = !isempty(vulkanpeers(dev))

has(dev, n::Symbol) = has(dev, Val(n))

"""The devices the suite runs on: every GPU, and lavapipe where it is installed."""
suitedevices() = collect(Mantle.devices())

# One replacement helper for immutable caps. Unknown overrides must fail:
# silently ignoring a misspelled field invalidates fault/queue policy tests.
function withcaps(c; kw...)
    all(k -> k in propertynames(c), keys(kw)) || throw(ArgumentError("unknown capability override"))
    return replacecaps(c, (; kw...))
end
replacecaps(c::NamedTuple, replacements) = merge(c, replacements)
replacecaps(c, replacements) = typeof(c)((get(replacements, f, getfield(c, f)) for f in fieldnames(typeof(c)))...)

# ── Counters ──

"""Counters of `f()` alone: reset, run, read."""
function counted(f, dev = testdevice())
    Mantle.resetdiagnostics!(dev)
    v = f()
    return v, Mantle.diagnostics(dev)
end
submitcount(d) = sum(values(d.submissions))   # not `submissions`: that is Mantle's, for a graph
"""The channels that submitted something, from a Diagnostics (counted per channel)."""
channelsused(d) = count(>(0), values(d.submissions))
"""
The segments of a compiled graph: `submissions(g)` lists stages (cut by host
nodes), each with its segments (cut by budget and queue kind; api.md 2.3).
"""
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))

# ── Queue policy ──

"""
`f()` with `dev`'s queue policy set to `SpreadQueues()`: every submission
takes the next queue of its kind, so dependent work crosses queues wherever
the device has more than one (recording-plan.md, Queues; api.md 2.1
`queuepolicy!`). The previous policy is restored afterwards.
"""
function spreadqueues(f, dev = testdevice())
    old = Mantle.queuepolicy!(dev, Mantle.SpreadQueues())
    try
        return f()
    finally
        Mantle.queuepolicy!(dev, old)
    end
end
"""`f()` under `spreadqueues` when `spread`, else under the default policy."""
withqueues(f, spread::Bool, dev = testdevice()) = spread ? spreadqueues(f, dev) : f()

"""Memory figures, for leak cases: compare before and after, after a sweep."""
function memory(dev = testdevice())
    GC.gc(true)
    Mantle.waitidle(dev)
    Mantle.sweep!(dev)
    Mantle.trim!(dev)  # empty native caches, not trim!(array) live-page unmapping
    return (; reserved = Mantle.reserved(dev), mapped = Mantle.mapped(dev),
            uploads = Mantle.liveuploads(dev), cells = Mantle.livecells(dev))
end

# ── Instruments for concurrency cases ──

"""
A kernel that keeps the GPU busy for about `ms` milliseconds and then writes
`value` into `out[1]`: a submission that is reliably still in flight when the
host does the next thing (without spinning on a host flag,
which can hang a device). Calibrated once per device.
"""
function slowwrite!(g_or_dev, out::MantleArray{Int32}, value::Integer; ms = 200)
    iters = calibratedspin(testdevice(), ms)
    dispatch!(g_or_dev, spinwrite_kernel!, (out, Int32(value), Int32(iters)), 1)
end
function spinwrite_kernel!(out, value, iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i       # dependent chain: not optimized away
    end
    out[1] = value + (acc == Int32(-7) ? Int32(1) : Int32(0))
    return
end
const SPINS = Dict{Mantle.Device,Int}()
function calibratedspin(dev, ms)
    per_ms = get!(SPINS, dev) do
        out = MantleArray(dev, Int32, 1)
        t = @elapsed (dispatch!(dev, spinwrite_kernel!, (out, Int32(0), Int32(10^7)), 1); Array(out))
        round(Int, 10^7 / (1000t))
    end
    return max(1, ms * per_ms)
end

"""
A host function that blocks until `letgo!(gate)`: holds a
run between two stages for as long as the test needs.
"""
struct Gate
    entered::Channel{Nothing}
    release::Channel{Nothing}
end
Gate() = Gate(Channel{Nothing}(1), Channel{Nothing}(1))
blockinghost(gate::Gate) = HostCall(_ -> (put!(gate.entered, nothing); take!(gate.release)))
waitentered(gate::Gate; timeout = 30) = withtimeout(() -> take!(gate.entered), timeout)
letgo!(gate::Gate) = put!(gate.release, nothing)

"""
`f()` on another task; fails the test (a likely deadlock) if it has not
returned after `timeout` seconds.
"""
function withtimeout(f, timeout::Real)
    t = Threads.@spawn f()
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("no return after $timeout s: likely a deadlock")
    return fetch(t)
end

"""
An array filled with a sentinel next to the arrays a case writes: a write
past an end, or into memory a released array gave back, changes it.
"""
canary(dev, n = 4096) = (a = MantleArray(dev, UInt32, n); fill!(a, 0xC0FFEE00); a)
intact(c) = all(==(0xC0FFEE00), Array(c))

"""Deterministic test data."""
testdata(T, n; seed = 1) = rand(Xoshiro(seed), T, n)

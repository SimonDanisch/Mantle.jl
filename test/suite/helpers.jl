# What every area file uses: the case macro, the device being tested, counters
# and memory figures read through Mantle's test hooks, and the instruments for
# concurrency cases (slow kernels, blocking host functions, watchdogs,
# canaries).
#
# Test hooks the refactor adds to Mantle for this suite (ledger P0, "suite
# hooks"; none of them is user API):
#   Mantle.diagnostics(dev) -> Diagnostics   counters since the last reset:
#       plancompiles, pipelinecompiles, kernelcompiles, records (recorded
#       parts, AccelCommands and pieces included), submissions (per channel),
#       allocations (acquire! calls), hostwaits
#   Mantle.resetdiagnostics!(dev)
#   Mantle.reserved(dev), Mantle.mapped(dev)  pool bytes
#   Mantle.liveuploads(dev), Mantle.livecells(dev)
#   Mantle.pendingof(x) -> Vector{Change}     a resource's pending requests
#   Mantle.steplog!(dev, on) / Mantle.steplog(dev)   prefix steps applied:
#       (kind, target, seq, token) per step, in emission order
#   Mantle.checklocks!(dev, on)               assert the lock order and that
#       no sync-state, channel, pool or window lock is held in a host wait
#                                             (catalogue FI-8)

using Test, Mantle, Random

"""The suite's settings: the one global, set by runsuite."""
struct SuiteConfig
    phase::Base.RefValue{Int}
    long::Base.RefValue{Bool}
    device::Base.RefValue{Any}
end
const SUITE = SuiteConfig(Ref(0), Ref(false), Ref{Any}(nothing))

testdevice() = SUITE.device[]
devicename(dev) = string(nameof(typeof(dev)))

# The API every case uses (MantleArray, Graph(dev), run!(g)) arrives in ledger
# phase 6 (P6.A1). A case keeps the phase of the contract it checks, and runs
# from phase 6 on at the earliest.
const APIPHASE = 6

"""
    @case id phase [needs...] begin … end

One catalogue case: a testset named by its catalogue id. Skipped when its
ledger phase is later than the suite's, when it is `:long` and long cases are
off, or when the device lacks a capability in `needs` (`:rt`, `:sparse`,
`:multiqueue`, `:window`, `:separateheap`, `:vendorview`, `:metal`,
`:lavapipe`, `:graphics`).
"""
macro case(id, phase, args...)
    body = last(args)
    needs = collect(args[1:end-1])
    quote
        local _needs = Symbol[$(map(esc, needs)...)]
        if max($(esc(phase)), APIPHASE) > SUITE.phase[] || (:long in _needs && !SUITE.long[]) ||
           !all(n -> n === :long || has(testdevice(), n), _needs)
            @testset $(esc(id)) begin
                @test_skip $(string(id))
            end
        else
            @testset $(esc(id)) begin
                $(esc(body))
            end
        end
    end
end

# Capabilities, from caps(dev) (field names are G3's; written here once).
has(dev, ::Val{:rt}) = caps(dev).raytracing
has(dev, ::Val{:graphics}) = caps(dev).graphics       # render passes and draws (not CUDA, HIP)
has(dev, ::Val{:sparse}) = caps(dev).sparseresidency
has(dev, ::Val{:multiqueue}) = length(Mantle.queuekinds(dev)) > 1
has(dev, ::Val{:window}) = caps(dev).display
has(dev, ::Val{:separateheap}) = caps(dev).separatehostheap
has(dev, ::Val{:vendorview}) = dev isa Mantle.CudaDevice
has(dev, ::Val{:metal}) = dev isa Mantle.MetalDevice
has(dev, ::Val{:lavapipe}) = occursin("llvmpipe", lowercase(Mantle.name(dev)))
has(dev, n::Symbol) = has(dev, Val(n))

"""The devices the suite runs on: every GPU, and lavapipe where it is installed."""
suitedevices() = collect(Mantle.devices())

# ── Counters ──

"""Counters of `f()` alone: reset, run, read."""
function counted(f, dev = testdevice())
    Mantle.resetdiagnostics!(dev)
    v = f()
    return v, Mantle.diagnostics(dev)
end
submitcount(d) = sum(values(d.submissions))   # not `submissions`: that is Mantle's, for a graph

"""Memory figures, for leak cases: compare before and after, after a sweep."""
function memory(dev = testdevice())
    GC.gc(true)
    Mantle.waitidle(dev)
    Mantle.sweep!(dev)
    return (; reserved = Mantle.reserved(dev), mapped = Mantle.mapped(dev),
            uploads = Mantle.liveuploads(dev), cells = Mantle.livecells(dev))
end

# ── Instruments for concurrency cases ──

"""
A kernel that keeps the GPU busy for about `ms` milliseconds and then writes
`value` into `out[1]`: a submission that is reliably still in flight when the
host does the next thing (catalogue FI-G, without spinning on a host flag,
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
const SPINS = Dict{Any,Int}()
function calibratedspin(dev, ms)
    per_ms = get!(SPINS, dev) do
        out = MantleArray(dev, Int32, 1)
        t = @elapsed (dispatch!(dev, spinwrite_kernel!, (out, Int32(0), Int32(10^7)), 1); Array(out))
        round(Int, 10^7 / (1000t))
    end
    return max(1, ms * per_ms)
end

"""
A host function that blocks until `letgo!(gate)` (catalogue FI-H): holds a
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

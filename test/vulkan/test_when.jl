"""
`when!`: a branch the HOST decides, recorded once and skipped at submission.

A graph is declared and recorded with every branch in it, and each run submits
only the branches it wants. What makes that cheap is that a recording is already
a sequence of independently submittable pieces (`RecordingParts`), so a branch
that does not run is a piece that is not submitted — not a dispatch whose writes
are discarded.

That is the whole difference from `repeat!`'s device predicate, which is the
right tool when only the GPU knows the answer and whose own docstring prices it:
"what a discarded iteration still costs is its barriers, its gate dispatch and
its predicate test". A `when!` region that does not run costs one boolean read.

The assertions below are the four places this can be quietly wrong.
"""

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
const M = Mantle

function when_addone!(a, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds a[i] += 1.0f0)
    return nothing
end

"""`a .= v`. A transient is NOT zeroed, so a region writing one has to SET it
rather than accumulate — the accumulating version passed alone and failed after
another test had used the pool, which is the whole reason to run them together."""
function when_set!(a, v::Float32, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds a[i] = v)
    return nothing
end

"""`dst .= src`, so a reader can be told apart from a writer."""
function when_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds dst[i] = src[i])
    return nothing
end

const WHEN_N = 64

@testset "a when! region runs or does not, from one recording" begin
    dev = M.Device(TESTBACKEND)
    buf = M.Buffer(dev, zeros(Float32, WHEN_N))
    flag = Ref(true)

    g = M.Graph(dev)
    M.dispatch!(g, when_addone!, (buf, Int32(WHEN_N)), WHEN_N; group = 64, name = "always")
    M.when!(g, flag) do
        M.dispatch!(g, when_addone!, (buf, Int32(WHEN_N)), WHEN_N; group = 64, name = "maybe")
    end
    plan = M.Plan(g)
    M.record!(plan)

    # Two pieces: the unconditional head and the region. The condition is
    # recorded per piece, not per pass, because a piece is the unit that can be
    # left unsubmitted.
    @test length(plan.recording.parts) == 2
    @test plan.recording.conds[1] === nothing
    @test plan.recording.conds[2] === flag

    # ONE recording, both answers.
    fill!(M.storage(buf), 0.0f0); flag[] = true
    M.run!(plan); M.waitidle(dev)
    @test all(==(2.0f0), Array(M.storage(buf)))

    fill!(M.storage(buf), 0.0f0); flag[] = false
    M.run!(plan); M.waitidle(dev)
    @test all(==(1.0f0), Array(M.storage(buf)))

    # And back, so nothing about the first run is sticky.
    fill!(M.storage(buf), 0.0f0); flag[] = true
    M.run!(plan); M.waitidle(dev)
    @test all(==(2.0f0), Array(M.storage(buf)))

    M.free!(plan); M.free!(buf)
end

@testset "a reader after a skipped region still sees the write before it" begin
    # The hazard the whole design rests on. `a` is written unconditionally,
    # written again inside the region, and READ after it. With the region
    # skipped the reader must see the first write — which means the barrier the
    # graph derived for the reader has to survive its writer not being
    # submitted. A barrier waiting on a write that did not happen is a wait that
    # is already satisfied; a MISSING barrier would be a race that reads the
    # older value only sometimes.
    dev = M.Device(TESTBACKEND)
    a = M.Buffer(dev, zeros(Float32, WHEN_N))
    out = M.Buffer(dev, zeros(Float32, WHEN_N))
    flag = Ref(false)

    g = M.Graph(dev)
    M.dispatch!(g, when_addone!, (a, Int32(WHEN_N)), WHEN_N; group = 64, name = "write")
    M.when!(g, flag) do
        M.dispatch!(g, when_addone!, (a, Int32(WHEN_N)), WHEN_N; group = 64, name = "write-again")
    end
    M.dispatch!(g, when_copy!, (out, a, Int32(WHEN_N)), WHEN_N; group = 64, name = "read")
    plan = M.Plan(g)
    M.record!(plan)
    # Three pieces: the region splits the unconditional work either side of it.
    @test length(plan.recording.parts) == 3

    for (on, want) in ((false, 1.0f0), (true, 2.0f0))
        fill!(M.storage(a), 0.0f0); fill!(M.storage(out), 0.0f0)
        flag[] = on
        M.run!(plan); M.waitidle(dev)
        @test all(==(want), Array(M.storage(out)))
    end

    M.free!(plan); M.free!(a); M.free!(out)
end

@testset "a transient only a when! region uses is still placed" begin
    # `Liveness` refuses a transient no pass uses, and "no pass that RAN" is a
    # different question from "no pass in the graph" — the placement is decided
    # once, at record, for every branch. So a transient written and read only
    # inside a region has to survive planning whether or not that region is ever
    # submitted, and the run with it off must not trip over it.
    dev = M.Device(TESTBACKEND)
    out = M.Buffer(dev, zeros(Float32, WHEN_N))
    flag = Ref(false)

    g = M.Graph(dev)
    tmp = M.Transient.Buffer(g, Float32, (WHEN_N,))
    M.when!(g, flag) do
        M.dispatch!(g, when_set!, (tmp, 3.0f0, Int32(WHEN_N)), WHEN_N;
                    group = 64, name = "fill-tmp")
        M.dispatch!(g, when_copy!, (out, tmp, Int32(WHEN_N)), WHEN_N; group = 64, name = "tmp-out")
    end
    plan = M.Plan(g)
    M.record!(plan)
    # The graph opens on a conditional region, so the head gets an empty piece of
    # its own — `emithead!` carries the head barrier and the profiler's pool
    # reset, and a piece that might not be submitted cannot hold them.
    @test length(plan.recording.parts) == 2
    @test plan.recording.conds[1] === nothing

    fill!(M.storage(out), 7.0f0); flag[] = false
    M.run!(plan); M.waitidle(dev)
    @test all(==(7.0f0), Array(M.storage(out)))     # nothing ran, nothing changed

    flag[] = true
    M.run!(plan); M.waitidle(dev)
    @test all(==(3.0f0), Array(M.storage(out)))

    M.free!(plan); M.free!(out)
end

@testset "a plan with no condition records exactly as it did" begin
    # The property that keeps `when!` free for everyone not using it: with no
    # condition and no `maxpasses`, the plan takes the single-recording path and
    # is not a `RecordingParts` at all.
    dev = M.Device(TESTBACKEND)
    buf = M.Buffer(dev, zeros(Float32, WHEN_N))
    g = M.Graph(dev)
    M.dispatch!(g, when_addone!, (buf, Int32(WHEN_N)), WHEN_N; group = 64, name = "plain")
    plan = M.Plan(g)
    M.record!(plan)
    @test !(plan.recording isa M.RecordingParts)

    M.run!(plan); M.waitidle(dev)
    @test all(==(1.0f0), Array(M.storage(buf)))
    M.free!(plan); M.free!(buf)
end

@testset "when! refuses to nest" begin
    # A pass carries ONE condition and nesting would need their conjunction —
    # the same reason `repeat!` refuses, and the same message shape.
    dev = M.Device(TESTBACKEND)
    buf = M.Buffer(dev, zeros(Float32, WHEN_N))
    outer, inner = Ref(true), Ref(true)
    g = M.Graph(dev)
    @test_throws "nested" M.when!(g, outer) do
        M.when!(g, inner) do
            M.dispatch!(g, when_addone!, (buf, Int32(WHEN_N)), WHEN_N; group = 64, name = "n")
        end
    end
    M.free!(buf)
end

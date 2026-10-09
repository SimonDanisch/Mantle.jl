"""
What a recorded plan has to get right across runs, asserted on values.

Each of these was a Metal test of that backend's recorder or queue, written where
the bug was found; none of them is about Metal. A graph with a transient between
two passes, a store landing ahead of the pass that reads it, a gated loop run back
to back and reopened by a store, an upload ordered against the first run that
reads it, a moved buffer staying patched, a frame whose host cost does not grow
with the plan, a rebuilt plan that compiles nothing, and a library call between
two recorded dispatches: the same assertions on every backend, gated only on what
the device can do (`supportspredicate` for a gate the device decides,
`runscalls` for a library call in a plan).
"""

using Test, Mantle
using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: mul!
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle

@kernel function rp_scale!(o, @Const(a), s)
    i = @index(Global, Linear)
    @inbounds o[i] = a[i] * s
end

@kernel function rp_add!(o, @Const(x), @Const(y))
    i = @index(Global, Linear)
    @inbounds o[i] = x[i] + y[i]
end

@kernel function rp_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function rp_accumulate!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

# One invocation owns the gate; every lane adds.
@kernel function rp_gate!(dst, @Const(src), flag)
    i = @index(Global)
    @inbounds dst[i] += src[i]
    i == 1 && (@inbounds flag[1] -= Int32(1))
end

@kernel function rp_bump!(counter, flag)
    i = @index(Global)
    @inbounds if i == 1
        counter[1] += 1.0f0
        flag[1] -= Int32(1)
    end
end

@kernel function rp_fill!(dst, v::Float32)
    i = @index(Global)
    @inbounds dst[i] = v
end

# The consumer for the call tests: it has to READ what the call wrote, so the
# ordering is exercised rather than asserted.
@kernel function rp_trace!(out, @Const(c), n::Int32)
    i = @index(Global)
    @inbounds if i == 1
        t = zero(eltype(c))
        for kk in Int32(1):n
            t += c[kk, kk]
        end
        out[1] = t
    end
end

@testset "a two-pass graph with a transient between them" begin
    # Passes declared, the phases run, `Place` aliases the transient into an
    # arena, and the run computes what the graph says. None of that is a
    # backend's: it is `src/graph/` and `src/phases.jl`.
    dev = M.Device(TESTBACKEND)
    g = M.Graph(dev)
    n = 1024
    inp = M.Buffer(dev, collect(1.0f0:Float32(n)))
    out = M.Buffer(dev, zeros(Float32, n))
    mid = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, rp_scale!, (mid, inp, 3.0f0), n; name = "scale")
    M.dispatch!(g, rp_add!, (out, mid, inp), n; name = "add")
    @test length(M.passes(g)) == 2
    @test length(g.transients) == 1

    plan = M.record!(M.Plan(g))
    # Every plan opens with the `updates` pass, on every backend: pending host
    # stores land there, ahead of anything that reads them. Spelled as the kinds
    # rather than a count, because a count says nothing about which pass moved.
    @test [p.pass.kind for p in plan.passes] == [:update, :compute, :compute]
    @test [p.pass.name for p in plan.passes] == ["updates", "scale", "add"]
    # The transient was placed into an arena, not allocated on its own.
    @test M.peakbytes(plan) >= n * sizeof(Float32)

    M.run!(plan)
    M.waitfor!(plan)
    want = collect(1.0f0:Float32(n)) .* 3.0f0 .+ collect(1.0f0:Float32(n))
    @test Array(M.storage(out)) ≈ want
    # Re-runnable: the second run must produce the same answer.
    M.run!(plan)
    M.waitfor!(plan)
    @test Array(M.storage(out)) ≈ want
    M.free!(plan)

    # The ordering the graph derived is the ordering that ran: reading `mid`
    # before the write gives `0 + inp`, and after it `5*inp + inp`.
    g2 = M.Graph(dev)
    inp2 = M.Buffer(dev, fill(2.0f0, 256))
    out2 = M.Buffer(dev, zeros(Float32, 256))
    mid2 = M.Transient.Buffer(g2, Float32, 256)
    M.dispatch!(g2, rp_scale!, (mid2, inp2, 5.0f0), 256; name = "producer")
    M.dispatch!(g2, rp_add!, (out2, mid2, inp2), 256; name = "consumer")
    plan2 = M.record!(M.Plan(g2))
    M.run!(plan2)
    M.waitfor!(plan2)
    @test all(≈(12.0f0), Array(M.storage(out2)))      # 5*2 + 2, not 0 + 2
    M.free!(plan2)
end

@testset "a store lands once, ahead of the pass that reads it" begin
    # `run!` walked `pp.dispatches` and nothing else, and the update pass HAS no
    # dispatches — landing the writes a store is holding is its entire body. So
    # every store was silently dropped on every KernelAbstractions backend.
    dev = M.Device(TESTBACKEND)
    g = M.Graph(dev)
    n = 256
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.dispatch!(g, rp_copy!, (dst, src), n; name = "copy")
    plan = M.record!(M.Plan(g))
    # The update pass is a pass, dispatchless or not, and it is FIRST — that
    # ordering is what makes the write visible to the reader in the same frame.
    @test length(plan.passes) == 2
    @test first(plan.passes).pass.kind === :update
    @test isempty(first(plan.passes).dispatches)

    # Not fired: nothing is written and the kernel copies the zeros.
    M.run!(plan); M.waitfor!(plan)
    @test all(iszero, Array(M.storage(dst)))
    # Written: the store lands before the dispatch that declared the read.
    src[:] = fill(3.0f0, n)
    M.run!(plan); M.waitfor!(plan)
    @test Array(M.storage(dst)) == fill(3.0f0, n)
    # Consumed: a store that is not made again writes nothing next frame, so the
    # reader keeps seeing the last value rather than a stale one reappearing.
    src[:] = fill(7.0f0, n)
    M.run!(plan); M.waitfor!(plan)
    @test Array(M.storage(dst)) == fill(7.0f0, n)
    M.run!(plan); M.waitfor!(plan)
    @test Array(M.storage(dst)) == fill(7.0f0, n)
    M.free!(plan)
end

@testset "a gated loop runs back to back, and a store reopens it" begin
    # A `repeat!` gate is a value the device writes and the host never sees, and
    # a recording that gates is not read-only: the gate's state lives in the
    # recording's own memory, written inside the run and read later in the SAME
    # run. Two runs of one recording in flight at once are two writers of those
    # bytes. Metal's MTL4 queue does no hazard tracking between command buffers,
    # so overlapping runs corrupted each other's state and the command processor
    # stopped on one that no longer named a valid range: no error, just a
    # submission that never signalled and a host that blocked forever.
    #
    # It needs THREE things at once, which is why everything smaller passed: a
    # gated plan, several runs in flight, and enough work per run that they really
    # do overlap. `run!` without `waitfor!` in between is what a multi-sample
    # frame does.
    dev = M.Device(TESTBACKEND)
    if !M.supportspredicate(dev)
        @info "this device cannot discard fixed-size gated work; skipping"
        @test_skip M.supportspredicate(dev)
    else
        n = 1 << 20
        iters = 8
        g = M.Graph(dev)
        a = M.Buffer(dev, zeros(Float32, n))
        b = M.Buffer(dev, fill(1.0f0, n))
        flag = M.Buffer(dev, Int32[iters])
        M.repeat!(g, iters; while_nonzero = flag) do i
            M.dispatch!(g, rp_gate!, (a, b, flag), n; name = "it")
        end
        plan = M.record!(M.Plan(g))

        # Open: every iteration runs, and the last one shuts the gate.
        M.run!(plan); M.waitfor!(plan)
        @test Array(M.storage(a))[1] == Float32(iters)
        @test Array(M.storage(flag))[1] == 0

        # Back to back, filling whatever the queue holds in flight. The flag is
        # NOT reopened in between, deliberately: a host write to it between runs
        # would race the runs still in flight. With the gate shut the iterations
        # are discarded, but the gate's own writers still run in every run and
        # still write the bytes that are read later in it — which is the hazard.
        for _ in 1:12
            M.run!(plan)
        end
        M.waitfor!(plan)
        # Nothing more ran, and nothing corrupted what had.
        @test all(==(Float32(iters)), Array(M.storage(a)))
        @test Array(M.storage(flag))[1] == 0

        # Reopened to three by a store, which lands in front of the recording in
        # the run's own submission: exactly three more, then shut again.
        M.update!(flag, Int32[3])
        M.run!(plan); M.waitfor!(plan)
        @test Array(M.storage(a))[1] == Float32(iters + 3)
        @test Array(M.storage(flag))[1] == 0
        M.run!(plan); M.waitfor!(plan)
        @test Array(M.storage(a))[1] == Float32(iters + 3)
        M.free!(plan)
    end
end

@testset "an upload is ordered against the run that reads it" begin
    # Metal's MTL4 path had two queues live: the MTL4 one for a recording, and the
    # plain one for host updates and every kernel launch Metal.jl makes. Metal
    # orders command buffers only WITHIN a queue, so the upload of
    # `Buffer(dev, data)` landed AFTER the dispatch that read it — every frame
    # showed the previous frame's answer, and at four million elements the blits
    # overlapped a 9 ms run and forty chained adds came out at ten. Silent, not a
    # race a barrier fixes, so it is pinned as a VALUE, small and large.
    dev = M.Device(TESTBACKEND)
    for n in (256, 1 << 20)
        g = M.Graph(dev)
        a = M.Buffer(dev, zeros(Float32, n))
        b = M.Buffer(dev, fill(1.0f0, n))
        for i in 1:8
            M.dispatch!(g, rp_accumulate!, (a, b), n; name = "add-$i")
        end
        pl = M.record!(M.Plan(g))
        # The FIRST run is the one that catches it: that is when the uploads are
        # still pending, so the frame holds both an upload and a recording.
        for run in 1:3
            M.run!(pl)
            M.waitfor!(pl)
            @test Array(M.storage(a))[1] == Float32(8 * run)
        end
        M.free!(pl)
    end
end

@testset "a moved buffer stays patched on the runs after the move" begin
    # `resize!` past capacity gives a `Buffer` fresh storage and RETIRES the old
    # region rather than freeing it, so a recording left pointing at the old
    # address keeps reading well-formed, stale bytes: no fault, no warning, a
    # number that is merely wrong. On Metal that was the state of things until
    # 2026-09-14, because the move was announced through a hook whose default did
    # nothing and that backend never implemented. `test_recorded_move_patch.jl`
    # pins the first run after a move; this pins the store that follows it and
    # the run after that, with nothing left to patch.
    dev = M.Device(TESTBACKEND)
    n = 256
    a = M.Buffer(dev, zeros(Float32, n))
    b = M.Buffer(dev, fill(1.0f0, n))
    g = M.Graph(dev)
    M.dispatch!(g, rp_accumulate!, (a, b), n; name = "add")
    pl = M.record!(M.Plan(g))
    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 1.0f0

    before = M.region(b.store)
    resize!(b, 4n)
    @test M.region(b.store) !== before          # it really moved
    M.update!(b, fill(2.0f0, 4n))
    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 3.0f0       # 2.0f0 would be the retired storage
    # Still patched on the run after: a patch is a write into the argument
    # memory, not a state the next run has to redo.
    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 5.0f0
    M.free!(pl)
end

@testset "a recorded run's host cost does not grow with the plan" begin
    # The property a recording exists for: a run submits what was recorded, and
    # none of that grows with the graph. Twenty times the dispatches has to cost
    # the host the same, or something is walking the passes again. Read as a
    # MINIMUM over several runs, because `@allocated` on a path that touches the
    # driver is noisy upward and never downward. The bound is loose on purpose:
    # what it catches is the regression that puts per-dispatch work back, which
    # was 147 KB for 34 commands on Metal before its recording existed.
    # `test_run_allocates_nothing.jl` holds the stronger, zero bound.
    dev = M.Device(TESTBACKEND)
    function chainplan(k)
        g = M.Graph(dev)
        n = 256
        a = M.Buffer(dev, zeros(Float32, n))
        b = M.Buffer(dev, fill(1.0f0, n))
        for i in 1:k
            M.dispatch!(g, rp_accumulate!, (a, b), n; name = "add-$i")
        end
        return M.record!(M.Plan(g))
    end
    function runbytes(plan)
        for _ in 1:5
            M.run!(plan)
        end
        M.waitfor!(plan)
        bs = [(@allocated M.run!(plan)) for _ in 1:9]
        M.waitfor!(plan)
        return minimum(bs)
    end
    small = chainplan(2)
    big = chainplan(40)
    b_small = runbytes(small)
    b_big = runbytes(big)
    @test b_small < 4096
    @test b_big <= b_small + 64
    M.free!(small)
    M.free!(big)
end

@testset "a rebuilt plan compiles nothing" begin
    # A loop's body is declared once, and a plan built again from the same
    # declarations finds every kernel it needs already compiled. On Metal each
    # recorded dispatch was named by its argument offset, the name was part of the
    # compile key, and a rebuilt plan linked every dispatch again: 124 links and
    # 138 s before the first GOLD frame of the isubd demo, 9 links and 21 s after.
    dev = M.Device(TESTBACKEND)
    if !M.supportspredicate(dev)
        @test_skip M.supportspredicate(dev)
    else
        function bumps()
            g = M.Graph(dev)
            counter = M.Buffer(dev, Float32[0])
            flag = M.Buffer(dev, Int32[4])
            M.repeat!(g, 4; while_nonzero = flag) do i
                M.dispatch!(g, rp_bump!, (counter, flag), 1; name = "bump")
            end
            return M.record!(M.Plan(g)), counter
        end
        plan, counter = bumps()
        M.run!(plan); M.waitfor!(plan)
        @test Array(M.storage(counter))[1] == 4f0
        M.free!(plan)

        M.resetkernelcompiles!(dev)
        rebuilt, counter2 = bumps()
        M.run!(rebuilt); M.waitfor!(rebuilt)
        @test Array(M.storage(counter2))[1] == 4f0
        @test M.kernelcompiles(dev).misses == 0
        M.free!(rebuilt)
    end
end

# ── A library call is a HOLE in the recording ────────────────────────────────
#
# A device that answers `runscalls` may hold a call in a plan: host work that
# submits or encodes its own, here `mul!`. A recording cannot hold one, so it
# stops at the call and resumes after it. What is silent when it is wrong is the
# ordering: Metal runs command buffers in COMMIT order, so a library that commits
# a buffer of its own while the frame's is still open runs BEFORE everything
# already in it. A dispatch on each side of the calls, read back, is the only way
# that shows. `test_declared_call.jl` covers a single call and the refusal on a
# device that cannot run one.

@testset "a library call between two dispatches keeps its place" begin
    dev = M.Device(TESTBACKEND)
    if !M.runscalls(dev)
        @test_skip M.runscalls(dev)
    else
        n = 32
        B = M.Buffer(dev, ones(Float32, n, n))
        A = M.Buffer(dev, zeros(Float32, n, n))
        C = M.Buffer(dev, zeros(Float32, n, n))
        D = M.Buffer(dev, zeros(Float32, n, n))
        tr = M.Buffer(dev, zeros(Float32, 1))

        g = M.Graph(dev)
        # Every step reads what the one before it wrote, so nothing here is right
        # by accident: a dispatch, then two adjacent calls, then a dispatch.
        M.dispatch!(g, rp_fill!, (A, 3.0f0), n * n; name = "pre")
        M.dispatch!(g, mul!, (C, A, B); name = "gemm1")
        M.dispatch!(g, mul!, (D, C, B); name = "gemm2")
        M.dispatch!(g, rp_trace!, (tr, D, Int32(n)), 1; name = "post")
        plan = M.record!(M.Plan(g))

        want = 3.0f0 * n * n            # A is 3, B is ones: C is 3n, D is 3n*n
        M.run!(plan)
        M.waitfor!(plan)
        @test all(==(3.0f0 * n), Array(M.storage(C)))
        @test all(==(want), Array(M.storage(D)))
        @test Array(M.storage(tr))[1] == want * n

        # Again from zeroed outputs. A run runs the calls again — the recording
        # holds a hole, not the library's kernels — so the second frame has to
        # reach the same answer through the same holes.
        fill!(M.storage(C), 0.0f0)
        fill!(M.storage(D), 0.0f0)
        fill!(M.storage(tr), 0.0f0)
        M.run!(plan)
        M.waitfor!(plan)
        @test all(==(want), Array(M.storage(D)))
        @test Array(M.storage(tr))[1] == want * n
        M.free!(plan)
    end
end

@testset "a call inside a gated pass is refused by name" begin
    # Whether a gated pass runs is a value the device writes and the host never
    # reads, so there is no way to NOT run a host call for a discarded iteration.
    # Refused at record, naming the pass — not run anyway.
    dev = M.Device(TESTBACKEND)
    if !(M.runscalls(dev) && M.supportspredicate(dev))
        @test_skip M.runscalls(dev)
    else
        n = 32
        A = M.Buffer(dev, fill(2.0f0, n, n))
        B = M.Buffer(dev, ones(Float32, n, n))
        C = M.Buffer(dev, zeros(Float32, n, n))
        flag = M.Buffer(dev, Int32[1])
        g = M.Graph(dev)
        M.repeat!(g, 2; while_nonzero = flag) do i
            M.dispatch!(g, mul!, (C, A, B); name = "gated-gemm")
        end
        err = try
            M.record!(M.Plan(g))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("gated pass", err.msg)
        @test occursin("gated-gemm", err.msg)
    end
end

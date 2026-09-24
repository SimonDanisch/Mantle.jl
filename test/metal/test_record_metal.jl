# Recording a plan on Metal: one indirect command buffer, replayed.
#
# `record!` on this backend writes the plan's compute commands into an
# `MTLIndirectCommandBuffer` once, and every `run!` replays it — one
# `executeCommandsInBuffer` per segment and nothing else. There is no
# dispatch-by-dispatch Julia in a frame, no KernelAbstractions, no `mtlconvert`
# and no argument re-adaptation. What that costs in correctness is the four
# properties below, each of which is silent when it is wrong:
#
#   * commands in one replay are CONCURRENT unless one carries a barrier, so a
#     consumer can read what its producer has not written yet;
#   * a `repeat!` gate is a value the device writes and the host never sees, so a
#     discarded iteration has to discard itself — through the execution range a
#     one-thread kernel writes ahead of the segment;
#   * a plan holding a render pass cannot be recorded at all, and has to keep
#     running the way it always did rather than half-recorded;
#   * a `DeviceRange` is dispatched at its CEILING, so a range without one is
#     refused by name rather than dispatched at a million threads.

using Test, Mantle, Metal
using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: mul!

const REC_DEV = Mantle.Device(Mantle.MetalAPI())

@kernel function rec_fill!(dst, v::Float32)
    i = @index(Global)
    @inbounds dst[i] = v
end

@kernel function rec_add!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

# The consumer for the call tests: it has to READ what the call wrote, so the
# ordering is exercised rather than asserted.
@kernel function rec_trace!(out, @Const(c), n::Int32)
    i = @index(Global)
    @inbounds if i == 1
        t = zero(eltype(c))
        for kk in Int32(1):n
            t += c[kk, kk]
        end
        out[1] = t
    end
end

@kernel function rec_scale_ref!(dst, s)
    i = @index(Global)
    @inbounds dst[i] = Float32(i) * s[1]
end

@kernel function rec_bump!(counter, flag)
    i = @index(Global)
    @inbounds if i == 1
        counter[1] += 1.0f0
        flag[1] -= Int32(1)
    end
end

@kernel function rec_mark!(dst, n)
    i = @index(Global)
    @inbounds if i <= n[1]
        dst[i] = 1.0f0
    end
end

@testset "a compute plan is recorded, and every run replays it" begin
    n = 64
    g = Mantle.Graph(REC_DEV)
    a = Mantle.Buffer(REC_DEV, zeros(Float32, n))
    b = Mantle.Buffer(REC_DEV, fill(2.0f0, n))
    Mantle.dispatch!(g, rec_fill!, (a, 1.0f0), n; name = "fill")
    Mantle.dispatch!(g, rec_add!, (a,
                                       b), n; name = "add")
    plan = Mantle.record!(Mantle.Plan(g))
    @test plan.recording !== nothing

    # Two passes, one segment, two commands: nothing here is gated, so the whole
    # recording is one `executeCommandsInBuffer`.
    rec = plan.recording
    @test rec.ncommands == 2
    @test length(rec.segments) == 1

    # THREE runs, and the same answer each time. Every one of them is the same
    # recording: `fill` writes 1, `add` adds 2. A run that came out 5 would be the
    # two commands racing — `add` reading 3 from the run before while `fill`
    # overwrote it — which is what the barrier the graph derived prevents, and
    # what a single run cannot tell apart from a correct one.
    for _ in 1:3
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        @test all(≈(3.0f0), Array(Mantle.storage(a)))
    end
end

@testset "a GPURef written between runs reaches the recorded commands" begin
    n = 32
    g = Mantle.Graph(REC_DEV)
    dst = Mantle.Buffer(REC_DEV, zeros(Float32, n))
    s = Mantle.GPURef(REC_DEV, 1.0f0)
    Mantle.dispatch!(g, rec_scale_ref!, (dst,
                                             s), n; name = "scale")
    plan = Mantle.record!(Mantle.Plan(g))

    # The recording holds the ref's ADDRESS, not its value — that is the whole
    # contract of `record!`, and the update lands in the run's own submission
    # ahead of the replay.
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test Array(Mantle.storage(dst)) ≈ collect(1.0f0:Float32(n))

    Mantle.update!(s, 10.0f0)
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test Array(Mantle.storage(dst)) ≈ collect(1.0f0:Float32(n)) .* 10
end

@testset "a repeat! gate discards its iterations on the device" begin
    for (start, want) in ((Int32(2), 2.0f0), (Int32(4), 4.0f0), (Int32(0), 0.0f0))
        g = Mantle.Graph(REC_DEV)
        counter = Mantle.Buffer(REC_DEV, Float32[0])
        flag = Mantle.Buffer(REC_DEV, Int32[start])
        Mantle.repeat!(g, 4; while_nonzero = flag) do i
            Mantle.dispatch!(g, rec_bump!,
                                 (counter,
                                  flag), 1; name = "bump-$i")
        end
        plan = Mantle.record!(Mantle.Plan(g))

        # ONE execute per iteration, not two. The head segment holds the range
        # reset, the first gate and its range writer; every gated segment then
        # absorbs the NEXT iteration's gate and writer, so a discarded iteration
        # takes the following gate down with it — which is the right answer,
        # because a discarded iteration writes nothing the gate reads.
        @test length(plan.recording.segments) == 5
        @test count(s -> s.slot >= 0, plan.recording.segments) == 4
        @test plan.recording.segments[1].slot == -1

        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        @test Array(Mantle.storage(counter))[1] == want
        @test Array(Mantle.storage(flag))[1] == 0

        # The execution ranges the device wrote: a length for the iterations that
        # ran, zero for the ones the gate discarded. Read back rather than
        # inferred, because a gate that never closed and a body that did nothing
        # produce the same counter.
        ranges = Array(plan.recording.ranges)
        @test count(k -> ranges[2k] != 0, 1:4) == Int(want)

        # Run it again with the flag now at zero: nothing runs, and the counter
        # does not move. The recording is unchanged — only what the device read
        # out of it is.
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        @test Array(Mantle.storage(counter))[1] == want
    end
end

@testset "an unrolled loop compiles its kernel once, and a rebuild links nothing" begin
    # `repeat!` declares its body once per iteration. Each recorded dispatch used
    # to be named by its argument offset, and the name is part of the compile
    # key: four copies were four compile jobs, four metallibs and four native
    # links. And an indirect pipeline was never cached, so a REBUILT plan linked
    # every dispatch again. Hikari's bounce loop is such a body, and a RayMakie
    # material switch rebuilds its plans: 124 links and 138 s before the first
    # GOLD frame of the isubd demo, 9 links and 21 s after.
    recorded(plan) = [d for pp in plan.passes for d in pp.dispatches
                      if d isa Mantle.MetalRecordedDispatch && occursin("rec_bump", d.name)]
    function bumps()
        g = Mantle.Graph(REC_DEV)
        counter = Mantle.Buffer(REC_DEV, Float32[0])
        flag = Mantle.Buffer(REC_DEV, Int32[4])
        Mantle.repeat!(g, 4; while_nonzero = flag) do i
            Mantle.dispatch!(g, rec_bump!, (counter, flag), 1; name = "bump-$i")
        end
        return Mantle.record!(Mantle.Plan(g)), counter
    end
    plan, counter = bumps()
    ds = recorded(plan)
    @test length(ds) == 4
    @test length(unique(d.name for d in ds)) == 1        # one compile key
    @test all(d -> d.kernel.pipeline === ds[1].kernel.pipeline, ds)
    # And it still runs: a shared pipeline is named by four commands.
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test Array(Mantle.storage(counter))[1] == 4f0

    rebuilt, _ = bumps()
    @test all(d -> d.kernel.pipeline === ds[1].kernel.pipeline, recorded(rebuilt))
end

@testset "a DeviceRange records at its ceiling" begin
    cap = 256
    g = Mantle.Graph(REC_DEV)
    dst = Mantle.Buffer(REC_DEV, zeros(Float32, cap))
    n = Mantle.Buffer(REC_DEV, Int32[77])
    Mantle.dispatch!(g, rec_mark!, (dst,
                                        n),
                         Mantle.DeviceRange(n; max = cap), name = "mark")
    plan = Mantle.record!(Mantle.Plan(g))
    @test plan.recording !== nothing
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    # The command covers the whole ceiling and the kernel's own `i <= n[1]` is
    # what bounds it — the count is never read on the host.
    @test count(==(1.0f0), Array(Mantle.storage(dst))) == 77
end

@testset "a DeviceRange with no ceiling is refused by name" begin
    g = Mantle.Graph(REC_DEV)
    dst = Mantle.Buffer(REC_DEV, zeros(Float32, 64))
    n = Mantle.Buffer(REC_DEV, Int32[8])
    Mantle.dispatch!(g, rec_mark!, (dst,
                                        n),
                         Mantle.DeviceRange(n); name = "mark")
    plan = Mantle.Plan(g)
    # Refused, and the message says what to do about it. The plan still RUNS —
    # core's interpreted launch reads the count on the host — which is why this
    # is an error at `record!` and not at `Plan`.
    @test_throws "max" Mantle.record!(plan)
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test count(==(1.0f0), Array(Mantle.storage(dst))) == 8
end

# ── Hardware traversal, from a recorded plan ─────────────────────────────────
#
# An inline ray query is a call inside the SHADER, so nothing about the dispatch
# says the kernel traces — and an indirect command buffer that did not declare
# `supportRayTracing` refuses it by returning a MISS. Every ray comes back empty,
# the image is black, and there is no error anywhere. It cost a benchmark run
# that came out 24x faster than the software path because it rendered nothing.
#
# Read against the SAME graph walked rather than against a fixed number: the
# hardware and software traversals agree to about a ulp, and what is at stake
# here is whether the recorded one traversed at all.

using Raycore, Adapt, KernelAbstractions, GeometryBasics
import LinearAlgebra
const REC_KA = KernelAbstractions

@kernel function rec_hw_probe!(hits, ts, dirs, accel, O)
    i = @index(Global)
    hit, tri, t, bary, inst =
        Raycore.closest_hit(accel, Raycore.Ray(o = O, d = dirs[i], t_max = 1f30))
    @inbounds hits[i] = hit ? Int32(1) : Int32(0)
    @inbounds ts[i] = hit ? t : -1f0
end

@testset "a recorded plan traces a hardware acceleration structure" begin
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1.0f0))
    O = GeometryBasics.Point3f(0, 0, 4)
    n = 256
    s = Ref(987)
    nr() = (s[] = (1103515245 * s[] + 12345) % 2147483648; Float32(s[]) / 2147483648f0)
    dirs = [GeometryBasics.Vec3f(LinearAlgebra.normalize(
                GeometryBasics.Point3f(2 * (nr() - 0.5f0), 2 * (nr() - 0.5f0), 0) - O))
            for _ in 1:n]

    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, mesh)
    Raycore.sync!(hw)
    accel = Adapt.adapt(be, hw)

    function traceplan(dirs, accel, O, n; record::Bool)
        g = Mantle.Graph(REC_DEV)
        d = Mantle.Buffer(REC_DEV, dirs)
        hits = Mantle.Buffer(REC_DEV, zeros(Int32, n))
        ts = Mantle.Buffer(REC_DEV, zeros(Float32, n))
        Mantle.dispatch!(g, rec_hw_probe!,
                             (hits,
                              ts,
                              d, accel, O), n; name = "trace")
        plan = Mantle.Plan(g)
        record && Mantle.record!(plan)
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        return plan, Array(Mantle.storage(hits)), Array(Mantle.storage(ts))
    end

    walked, wh, wt = traceplan(dirs, accel, O, n; record = false)
    recorded, rh, rt = traceplan(dirs, accel, O, n; record = true)

    @test walked.recording === nothing
    @test recorded.recording !== nothing
    # Vacuous otherwise: a traversal that misses everything agrees with itself.
    @test sum(wh) > n ÷ 4
    @test rh == wh
    @test rt == wt
end

# ── What a replayed frame costs the host ─────────────────────────────────────
#
# The property the whole recording exists for: a frame is ONE command buffer, one
# encoder, one `useResources`, one `executeCommandsInBuffer` per segment and one
# commit — and none of that grows with the graph. Twenty times the dispatches has
# to cost the host the same, or something is walking the passes again.
#
# Read as a MINIMUM over several runs, because `@allocated` on a path that touches
# the driver is noisy upward and never downward. The bound is loose on purpose: what
# it catches is the regression that puts per-dispatch work back, which was 147 KB
# for 34 commands before the recording existed.

@kernel function rec_chain!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

@testset "a replayed frame does not grow with the plan" begin
    function chainplan(k)
        g = Mantle.Graph(REC_DEV)
        n = 256
        a = Mantle.Buffer(REC_DEV, zeros(Float32, n))
        b = Mantle.Buffer(REC_DEV, fill(1.0f0, n))
        for i in 1:k
            Mantle.dispatch!(g, rec_chain!,
                                 (a,
                                  b), n; name = "add-$i")
        end
        return Mantle.record!(Mantle.Plan(g))
    end
    function runbytes(plan)
        for _ in 1:5
            Mantle.run!(plan)
        end
        Mantle.waitfor!(plan)
        bs = [(@allocated Mantle.run!(plan)) for _ in 1:9]
        Mantle.waitfor!(plan)
        return minimum(bs)
    end

    small = chainplan(2)
    big = chainplan(40)
    @test small.recording.ncommands == 2
    @test big.recording.ncommands == 40
    # One segment each: nothing here is gated, so the whole recording is one
    # `executeCommandsInBuffer` however many dispatches it holds.
    @test length(small.recording.segments) == 1
    @test length(big.recording.segments) == 1

    b_small = runbytes(small)
    b_big = runbytes(big)
    @test b_small < 4096
    @test b_big <= b_small + 64
end

# ── A library call is a HOLE in the recording ─────────────────────────────────
#
# `runscalls` is `true` on this queue, so a plan may hold a `Mantle.Call`: host work
# that submits or encodes its own — here `mul!`, which reaches MPSGraph. An indirect
# command buffer cannot hold one, so the recording stops at the call and resumes
# after it, and a frame is one `executeCommandsInBuffer:` per SEGMENT with the calls
# run between them.
#
# What is silent when it is wrong is the ordering. Metal runs command buffers in
# COMMIT order and encoders in creation order, so a library that commits a buffer of
# its own while the frame's is still open runs BEFORE everything already in it. The
# tests below put a dispatch on each side of a call and read the result, which is the
# only way that shows up.

@testset "a library call is a hole in the recording" begin
    n = 32
    ones_h = ones(Float32, n, n)
    B = Mantle.Buffer(REC_DEV, ones_h)
    A = Mantle.Buffer(REC_DEV, zeros(Float32, n, n))
    C = Mantle.Buffer(REC_DEV, zeros(Float32, n, n))
    D = Mantle.Buffer(REC_DEV, zeros(Float32, n, n))
    tr = Mantle.Buffer(REC_DEV, zeros(Float32, 1))

    g = Mantle.Graph(REC_DEV)
    # Every step reads what the one before it wrote, so nothing here is right by
    # accident: a dispatch, then two calls, then a dispatch.
    Mantle.dispatch!(g, rec_fill!, (A, 3.0f0), n * n; name = "pre")
    Mantle.dispatch!(g, mul!, (C, A, B); name = "gemm1")
    Mantle.dispatch!(g, mul!, (D, C, B); name = "gemm2")
    Mantle.dispatch!(g, rec_trace!, (tr, D, Int32(n)), 1; name = "post")
    plan = Mantle.record!(Mantle.Plan(g))

    # TWO segments and two calls, both in the same hole: the calls are adjacent in
    # the plan, so there is no command between them to reopen the buffer for.
    @test length(plan.recording.segments) == 2
    @test map(first, plan.recording.calls) == [1, 1]
    # And the calls took no room in the buffer — the two commands are the two
    # dispatches, one per segment.
    @test plan.recording.ncommands == 2
    @test plan.recording.segments[1] == Mantle.MetalSegment(1, 1, -1)
    @test plan.recording.segments[2] == Mantle.MetalSegment(2, 2, -1)

    want = 3.0f0 * n * n            # A is 3, B is ones: C is 3n, D is 3n*n
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test all(==(3.0f0 * n), Array(Mantle.storage(C)))
    @test all(==(want), Array(Mantle.storage(D)))
    @test Array(Mantle.storage(tr))[1] == want * n

    # Again from zeroed outputs. A replay runs the calls again — the recording holds
    # a hole, not the library's kernels — so the second frame has to reach the same
    # answer through the same holes.
    fill!(Mantle.storage(C), 0.0f0)
    fill!(Mantle.storage(D), 0.0f0)
    fill!(Mantle.storage(tr), 0.0f0)
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test all(==(want), Array(Mantle.storage(D)))
    @test Array(Mantle.storage(tr))[1] == want * n
    Mantle.free!(plan)
end

@testset "a call before the first command" begin
    n = 32
    A = Mantle.Buffer(REC_DEV, fill(2.0f0, n, n))
    B = Mantle.Buffer(REC_DEV, ones(Float32, n, n))
    C = Mantle.Buffer(REC_DEV, zeros(Float32, n, n))
    tr = Mantle.Buffer(REC_DEV, zeros(Float32, 1))

    g = Mantle.Graph(REC_DEV)
    Mantle.dispatch!(g, mul!, (C, A, B); name = "gemm")
    Mantle.dispatch!(g, rec_trace!, (tr, C, Int32(n)), 1; name = "post")
    plan = Mantle.record!(Mantle.Plan(g))

    # Registered at segment ZERO, which is what "before the first segment" is.
    @test map(first, plan.recording.calls) == [0]
    @test length(plan.recording.segments) == 1

    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    @test all(==(2.0f0 * n), Array(Mantle.storage(C)))
    @test Array(Mantle.storage(tr))[1] == 2.0f0 * n * n
    Mantle.free!(plan)
end

@testset "a call inside a gated pass is refused by name" begin
    n = 32
    A = Mantle.Buffer(REC_DEV, fill(2.0f0, n, n))
    B = Mantle.Buffer(REC_DEV, ones(Float32, n, n))
    C = Mantle.Buffer(REC_DEV, zeros(Float32, n, n))
    flag = Mantle.Buffer(REC_DEV, Int32[1])

    g = Mantle.Graph(REC_DEV)
    Mantle.repeat!(g, 2; while_nonzero = flag) do i
        Mantle.dispatch!(g, mul!, (C, A, B); name = "gated-gemm")
    end
    # Whether a gated pass runs is a value the device writes and the host never
    # reads, so there is no way to NOT run a host call for a discarded iteration.
    # Refused at record, naming the pass — not run anyway.
    err = try
        Mantle.record!(Mantle.Plan(g))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("gated pass", err.msg)
    @test occursin("gated-gemm", err.msg)
end

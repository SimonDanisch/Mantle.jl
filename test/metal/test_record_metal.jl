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

const REC_DEV = Mantle.Device(Mantle.MetalAPI())

@kernel function rec_fill!(dst, v::Float32)
    i = @index(Global)
    @inbounds dst[i] = v
end

@kernel function rec_add!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
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
    Mantle.compute!(g, "fill") do p
        Mantle.dispatch!(p, rec_fill!, (Mantle.use(p, a; write = true), 1.0f0), n)
    end
    Mantle.compute!(g, "add") do p
        Mantle.dispatch!(p, rec_add!, (Mantle.use(p, a; read = true, write = true),
                                       Mantle.use(p, b; read = true)), n)
    end
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
    Mantle.compute!(g, "scale") do p
        Mantle.dispatch!(p, rec_scale_ref!, (Mantle.use(p, dst; write = true),
                                             Mantle.use(p, s; read = true)), n)
    end
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
            Mantle.compute!(g, "bump-$i") do p
                Mantle.dispatch!(p, rec_bump!,
                                 (Mantle.use(p, counter; read = true, write = true),
                                  Mantle.use(p, flag; read = true, write = true)), 1)
            end
        end
        plan = Mantle.record!(Mantle.Plan(g))

        # Four iterations, each a gate pass and a gated pass: four ungated
        # segments (the gate dispatch and the range writer behind it) and four
        # gated ones.
        @test length(plan.recording.segments) == 8
        @test count(s -> s.slot >= 0, plan.recording.segments) == 4

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

@testset "a DeviceRange records at its ceiling" begin
    cap = 256
    g = Mantle.Graph(REC_DEV)
    dst = Mantle.Buffer(REC_DEV, zeros(Float32, cap))
    n = Mantle.Buffer(REC_DEV, Int32[77])
    Mantle.compute!(g, "mark") do p
        Mantle.dispatch!(p, rec_mark!, (Mantle.use(p, dst; write = true),
                                        Mantle.use(p, n; read = true)),
                         Mantle.DeviceRange(n; max = cap))
    end
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
    Mantle.compute!(g, "mark") do p
        Mantle.dispatch!(p, rec_mark!, (Mantle.use(p, dst; write = true),
                                        Mantle.use(p, n; read = true)),
                         Mantle.DeviceRange(n))
    end
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
        Mantle.compute!(g, "trace") do p
            Mantle.dispatch!(p, rec_hw_probe!,
                             (Mantle.use(p, hits; write = true),
                              Mantle.use(p, ts; write = true),
                              Mantle.use(p, d; read = true), accel, O), n)
        end
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
            Mantle.compute!(g, "add-$i") do p
                Mantle.dispatch!(p, rec_chain!,
                                 (Mantle.use(p, a; read = true, write = true),
                                  Mantle.use(p, b; read = true)), n)
            end
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

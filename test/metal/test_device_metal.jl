# The device/queue seam, and the one property it exists to protect.
#
# `MetalDevice{Q}` is parametric in its submission queue because that is the only
# thing two Metal generations disagree about. A `LegacyQueue` hands work to
# Metal.jl's batched `MTLCommandQueue` and can only answer "has the GPU finished"
# by draining the device; an `MTL4Queue` owns its command buffers and signals a
# shared event per submit, so the same question is a load. Allocation, storage
# modes, budgets and the eleven pool primitives are the same either way.
#
# What is pinned here is the SEAM: two verbs (`opensubmit!`, `closesubmit!`) plus
# the three timeline verbs, both generations constructible on ONE machine, and
# nothing outside `device.jl` naming a generation. A third generation should be a
# third queue type and no edit anywhere else — and if that stops being true, the
# last test in this file is the one that says so.

using Test
using Mantle
using Metal
using KernelAbstractions: @kernel, @index, @Const
const M = Mantle
const MTLm = Metal.MTL

DEV_SEAM = M.Device(Mantle.MetalAPI())

# Both, on whatever machine this is. The process device gets whichever generation
# `metalqueue` chose; the other is built here by name, which is the whole reason
# the generation is a runtime parameter rather than a precompile-time constant —
# otherwise the path this Mac does not use goes untested until someone else's
# machine finds the regression.
#
# Never allocated from: a second `MetalDevice` is a second `Pool` over the same
# `MTLDevice`, which is two allocators over one memory. These exist to show that
# the constructor takes the queue, and to run the verbs against both.
seamdevices() = begin
    mtldev = Metal.device()
    ds = Any[DEV_SEAM]
    push!(ds, Mantle.MetalDevice(mtldev, Mantle.LegacyQueue(mtldev)))
    MTLm.supports_mtl4(mtldev) && push!(ds, Mantle.MetalDevice(mtldev, Mantle.MTL4Queue(mtldev)))
    ds
end

@testset "the device is parametric in its queue" begin
    @test DEV_SEAM isa Mantle.MetalDevice
    # Concrete, which is the point of the parameter: the queue is named by the
    # type rather than the field being a UnionAll box.
    @test isconcretetype(typeof(DEV_SEAM))
    @test typeof(DEV_SEAM) === Mantle.MetalDevice{typeof(DEV_SEAM.queue)}
    @test DEV_SEAM.queue isa Union{Mantle.LegacyQueue, Mantle.MTL4Queue}

    # The fields a generation owns are the queue's, not the device's. A timeline
    # counter on `MetalDevice` is what the split removed, and putting one back is
    # how the two would start disagreeing about a field again.
    devfields = fieldnames(Mantle.MetalDevice)
    @test :queue in devfields
    for f in (:next, :completed, :event, :recording)
        @test !(f in devfields)
    end
    @test hasfield(Mantle.LegacyQueue, :next)
    @test hasfield(Mantle.MTL4Queue, :next)
end

@testset "both generations are constructible here" begin
    ds = seamdevices()
    # The process device plus a legacy one, plus MTL4 where the device has it.
    @test length(ds) >= 2
    @test any(d -> d.queue isa Mantle.LegacyQueue, ds)
    @test MTLm.supports_mtl4(Metal.device()) == any(d -> d.queue isa Mantle.MTL4Queue, ds)
    for d in ds
        @test isconcretetype(typeof(d))
        @test d.dev === Metal.device()
        # Every generation owns a plain queue, for the two paths that submit a
        # command buffer of their own: building an acceleration structure and
        # compiling the traversal pipeline. Both are setup, not frame work.
        @test Mantle.cmdqueue(d) isa MTLm.MTLCommandQueue
        @test M.batchqueue(d).queue === Mantle.cmdqueue(d)
    end
end

# ── The in-flight depth is correctness, not tuning ───────────────────────────
#
# Metal.jl blocks in `flush!` once its queue has `inflight` command buffers still
# running, and its default is three. Mantle was relying on that default without
# saying so, which meant `JULIA_METAL_COMMAND_BATCHING_INFLIGHT` — or the matching
# preference — could raise it under us.
#
# Raising it renders BLACK. Measured at eight: the crown comes out at mean luma
# 0.0 against 0.74, reproducibly, alternating 8/3/8/3 in one process. The host
# runs ahead of the GPU and mutates state an in-flight replay is still reading.
# Nothing in this suite caught it — all 9781 assertions passed while the image was
# black — so the invariant is pinned directly instead.
@testset "the in-flight depth is pinned, not inherited" begin
    for d in seamdevices()
        @test M.batchqueue(d).inflight == 3
    end
    # Even when Metal.jl's own default has been raised.
    was = Metal.command_batching_inflight()
    try
        @eval Metal command_batching_inflight() = 32
        q = Base.invokelatest(Mantle.LegacyQueue, Metal.device())
        @test M.batchqueue(q).inflight == 3
    finally
        @eval Metal command_batching_inflight() = $was
    end
end

@testset "the legacy timeline counts retirements" begin
    q = Mantle.LegacyQueue(Metal.device())
    @test M.fence(q) == 1
    @test M.fence(q) == 2
    # Monotone, which is what `Pool.retiring_at` relies on to drain a PREFIX
    # rather than walking its whole list.
    fs = [M.fence(q) for _ in 1:8]
    @test issorted(fs) && allunique(fs)
    @test !M.passed(q, fs[end])
    # This generation can only answer by draining the device, which is the one
    # thing the MTL4 timeline below exists to stop doing.
    @test M.waitfor(q, fs[end])
    @test M.passed(q, fs[end])
    @test !M.passed(q, fs[end] + 1)
end

if MTLm.supports_mtl4(Metal.device())
    @testset "the MTL4 timeline reads a shared event" begin
        q = Mantle.MTL4Queue(Metal.device())
        # A READ, not an allocation: "the value the next submission will signal",
        # which is what a region retired now has to wait for. It does not move
        # until something is submitted.
        @test M.fence(q) == 1
        @test M.fence(q) == 1
        @test q.event.signaledValue == 0
        # And it PASSES, because nothing has been submitted: the token names the
        # submission after the last one, and on an idle queue everything that
        # could be reading those bytes has finished. A region retired on an idle
        # device is reusable, and asking for `f` itself instead would leave the
        # pool's backlog stuck until something unrelated was submitted.
        @test M.passed(q, 1)
        @test M.waitfor(q, 1)
    end
end

@testset "a submission is opened and closed, on every generation" begin
    for d in seamdevices()
        # Opening a submission takes NO resource list. It used to, because legacy
        # declared them on the encoder with `useResource` while MTL4 needed its
        # residency set complete beforehand — so the list went in here and each
        # generation did what it needed. Both were paying per frame for something
        # permanent: `ensureresident!` grants residency where the list is built,
        # which is only when a block comes or goes. Taking no list is the point,
        # since a list that had not been granted could not then be passed.
        sub = Mantle.opensubmit!(d)
        @test Mantle.encoder(sub) isa Union{MTLm.MTLComputeCommandEncoder,
                                         MTLm.MTL4ComputeCommandEncoder}
        before = d.queue.next
        tok = Mantle.closesubmit!(d, sub)
        # The token comes OUT of the close, because the queue that signals an
        # event per submit is the one that knows which value this run will reach.
        @test tok isa UInt64
        @test tok > before
        @test M.waitfor(d, tok)
        @test M.passed(d, tok)
    end
end

# ── The seam holds ────────────────────────────────────────────────────────────
#
# The refactor's actual claim: adding a generation touches `device.jl` and
# nothing else. `replay!`, the recorder, the pool primitives, graphics and the
# hardware TLAS are written against the verbs and name no generation.
#
# A grep, because that is what the claim is. The failure this catches is the
# cheap one — someone reaching for `d.queue.mtl` or a `LegacyQueue` method from
# `record.jl` because it was two characters shorter than adding a verb, which
# works perfectly until the second queue type is the one running.

@testset "no generation is named outside device.jl" begin
    src = joinpath(pkgdir(Mantle), "src", "metal")
    offenders = Tuple{String,Int,String}[]
    for f in readdir(src; join = true)
        endswith(f, ".jl") || continue
        basename(f) == "device.jl" && continue
        for (i, line) in enumerate(eachline(f))
            # Comments may say "legacy" — the point is that no CODE names it.
            startswith(strip(line), "#") && continue
            occursin(r"\bLegacyQueue\b|\bLegacySubmission\b|\bMTL4Queue\b|\bMTL4Submission\b", line) &&
                push!(offenders, (basename(f), i, strip(line)))
        end
    end
    @test isempty(offenders)
    isempty(offenders) || @info "generation named outside device.jl" offenders
end

# ── The bug this whole path is built around ──────────────────────────────────
#
# Two queues are live on the MTL4 path: the MTL4 one for a replay, and the plain
# one for the host updates a plan applies as blits and for every kernel launch
# Metal.jl makes. Metal orders command buffers only WITHIN a queue, so without a
# shared timeline the upload of `Buffer(dev, data)` lands AFTER the dispatch that
# reads it — every frame shows the previous frame's answer, and at four million
# elements the 32 MB of blits overlap a 9 ms replay and forty chained adds come
# out at ten.
#
# It is silent, it is not a race that a barrier fixes, and it does not reproduce
# under `MTL_SHADER_VALIDATION=1`, so it is pinned as a VALUE.

# ── The driver defect, refused rather than waited on ─────────────────────────
#
# `executeCommandsInBuffer:` on an MTL4 compute encoder stops completing after
# about a second of replayed GPU work: the timeline freezes, `MTL4CommitFeedback`
# reports nothing, the system log is empty and the next submission blocks for
# ever. Ninety lines of Metal.jl reproduce it with no Mantle in them
# (`Metal/research/mtl4_icb_stall.jl`), and every workaround at the API level was
# tried — both execute forms, a fresh allocator and command buffer per submission,
# forty distinct indirect command buffers, one submission in flight, no feedback
# handler, `supportMTLEvent`, `ray_tracing` off, a `Private` buffer, a barrier
# after the execute, a fresh queue per submission.
#
# So a recorded plan does not go through that queue as a REPLAY. It goes through
# as an ENCODED frame instead: the recorder captures every dispatch's pipeline,
# argument table, grid and barrier, and a run puts them on the encoder directly.
# Three sends per dispatch against one per segment, and the two routes are one
# predicate apart — `canreplay`, in `device.jl`, which is also the only file that
# knows a generation exists.
#
# A `repeat!` gate survives the crossing because a threadgroup count can be read
# from memory: where a replay shortens an execution range to nothing, an encoded
# frame writes a grid of zeros and the dispatch runs no threadgroups. Same
# one-thread writer, same "the host never learns whether the iteration ran".

@kernel function seam_gatecount!(dst, @Const(src), flag)
    i = @index(Global)
    @inbounds dst[i] += src[i]
    i == 1 && (@inbounds flag[1] -= Int32(1))
end

# A second `MetalDevice` is a second `Pool` over one `MTLDevice`, so these stay
# small and are the only tests here that allocate from one.
function seam_gatedplan(dev, n, iters)
    g = M.Graph(dev)
    a = M.Buffer(dev, zeros(Float32, n))
    b = M.Buffer(dev, fill(1.0f0, n))
    flag = M.Buffer(dev, Int32[iters])
    M.repeat!(g, iters; while_nonzero = flag) do i
        M.compute!(g, "it-$i") do p
            M.dispatch!(p, seam_gatecount!,
                        (M.use(p, a; read = true, write = true),
                         M.use(p, b; read = true),
                         M.use(p, flag; read = true, write = true)), n)
        end
    end
    return M.record!(M.Plan(g)), a, flag
end

if MTLm.supports_mtl4(Metal.device())
    @testset "an MTL4 queue encodes a recording instead of replaying it" begin
        mtldev = Metal.device()
        @test Mantle.canreplay(Mantle.LegacyQueue(mtldev))
        @test !Mantle.canreplay(Mantle.MTL4Queue(mtldev))
        # Exactly one of the two, and not both: a queue that replays captures
        # nothing at record time and one that encodes captures everything.
        @test !Mantle.encodes(Mantle.LegacyQueue(mtldev))
        @test Mantle.encodes(Mantle.MTL4Queue(mtldev))
    end

    @testset "a gated plan runs the same on both generations" begin
        mtldev = Metal.device()
        for q in (Mantle.MTL4Queue(mtldev), Mantle.LegacyQueue(mtldev))
            dev = Mantle.MetalDevice(mtldev, q)
            Mantle.adoptqueue!(dev)
            pl, a, flag = seam_gatedplan(dev, 1 << 10, 8)
            rec = pl.recording
            @test count(s -> s.slot >= 0, rec.segments) == 8
            # Captured only where it is needed, and then every command inside a
            # gated segment takes its grid from memory.
            if Mantle.encodes(q)
                @test length(rec.encoded) == rec.ncommands
                @test count(c -> c.gridoff >= 0, rec.encoded) > 0
                # Both ends of every gated run carry a barrier: the count is
                # written by a dispatch and read by the command processor, so
                # nothing inside a shader can order it.
                for sg in rec.segments
                    sg.slot < 0 && continue
                    @test rec.encoded[sg.first].barrier
                    sg.last < length(rec.encoded) && @test rec.encoded[sg.last + 1].barrier
                end
            else
                @test isempty(rec.encoded)
            end

            # Open: every iteration runs, and the last one shuts the gate.
            M.run!(pl); M.waitfor!(pl)
            @test Array(M.storage(a))[1] == 8.0f0
            @test Array(M.storage(flag))[1] == 0

            # Shut, and back to back so the ring fills: the iterations are
            # discarded and nothing corrupts what had already run.
            for _ in 1:12; M.run!(pl); end
            M.waitfor!(pl)
            @test Array(M.storage(a))[1] == 8.0f0
            @test Array(M.storage(flag))[1] == 0

            # Reopened to three: exactly three more, then shut again.
            M.update!(flag, Int32[3])
            M.run!(pl); M.waitfor!(pl)
            @test Array(M.storage(a))[1] == 11.0f0
            @test Array(M.storage(flag))[1] == 0
            M.run!(pl); M.waitfor!(pl)
            @test Array(M.storage(a))[1] == 11.0f0
        end
        # Handed back, so the rest of the suite runs on the device it built its
        # buffers on. Not required for correctness — every run adopts its own
        # device's queue and `adopt_queue!` drains the one it leaves — but a test
        # that leaves the process pointing at its own scratch device is a test that
        # will be blamed for the next unrelated failure.
        Mantle.adoptqueue!(DEV_SEAM)
    end

    # A recording is shaped for ONE of the two routes, and only one of the two
    # mismatches is loud. A replay-shaped recording has no dispatches to encode and
    # says so. An ENCODE-shaped one replays perfectly well and silently discards
    # every iteration of every `repeat!`, because its execution ranges were never
    # written and stayed at the zeros `openrecording` left — a black frame with
    # nothing to read. Both are refused, and the second is the reason why.
    #
    # Neither is reachable through the public API today (a plan's device is fixed
    # when it records), which is exactly why it wants a test: the day something
    # re-records a plan onto another device, this says so instead of going quiet.
    @testset "a recording is refused by the route it was not made for" begin
        mtldev = Metal.device()
        legacy = Mantle.MetalDevice(mtldev, Mantle.LegacyQueue(mtldev))
        mtl4 = Mantle.MetalDevice(mtldev, Mantle.MTL4Queue(mtldev))
        pl_legacy, = seam_gatedplan(legacy, 256, 2)
        pl_mtl4, = seam_gatedplan(mtl4, 256, 2)

        @test Mantle.canrun(legacy.queue, pl_legacy.recording)
        @test Mantle.canrun(mtl4.queue, pl_mtl4.recording)
        @test !Mantle.canrun(mtl4.queue, pl_legacy.recording)
        @test !Mantle.canrun(legacy.queue, pl_mtl4.recording)
        # And `replay!` is where that is asked, so neither runs a command.
        @test_throws ArgumentError Mantle.replay!(mtl4, pl_legacy.recording)
        @test_throws ArgumentError Mantle.replay!(legacy, pl_mtl4.recording)
        Mantle.adoptqueue!(DEV_SEAM)
    end
end

# ── The queue is adopted by every run, not only by one that opens a front ────
#
# Core opens a front only for a run with host stores or address patches to put in
# front of its recording, so a STEADY frame of a recorded plan never reaches
# `openrun`. With `adoptqueue!` only there, a device that was not the process
# device ran every frame on whatever queue Metal.jl had handed the task — and a
# `Buffer(dev, data)` uploaded on that queue landed AFTER the first frame that
# read it. One wrong frame, every frame after it right, nothing reported.
#
# Both halves are pinned: the device below is built by name and allocated from
# BEFORE it adopts anything, which is the order that used to fail.

@kernel function seam_adopt!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

@testset "a run adopts this device's queue whatever core opens" begin
    mtldev = Metal.device()
    for q in (Mantle.LegacyQueue(mtldev), Mantle.MTL4Queue(mtldev))
        MTLm.supports_mtl4(mtldev) || q isa Mantle.LegacyQueue || continue
        dev = Mantle.MetalDevice(mtldev, q)
        g = M.Graph(dev)
        a = M.Buffer(dev, zeros(Float32, 64))
        b = M.Buffer(dev, fill(2.0f0, 64))
        M.compute!(g, "fill") do p
            M.dispatch!(p, seam_adopt!, (M.use(p, a; read = true, write = true),
                                         M.use(p, b; read = true)), 64)
        end
        pl = M.record!(M.Plan(g))
        # THE FIRST RUN is the whole test.
        M.run!(pl); M.waitfor!(pl)
        @test Array(M.storage(a))[1] == 2.0f0
        @test Metal.adopted_queue[] === Mantle.batchqueue(dev)
    end
    Mantle.adoptqueue!(DEV_SEAM)
end

# ── …and so does RECORDING, which is not host work either ────────────────────
#
# `openrecording` fills three arrays with a kernel launch and `closerecording!`
# blits the gate template, so a `record!` on a device nothing has adopted puts all
# of it on whatever queue Metal.jl handed the task — and the frames, which run on
# THIS device's queue, are ordered against it by nothing.
#
# `ranges` and `grids` survive that: `emithead!`'s reset rewrites them inside every
# frame. The TEMPLATE does not. It is written once at record time and read by every
# gate for ever after, so a blit the first frames are not ordered behind is a plan
# whose gates open onto a grid of zeros and which silently runs nothing at all.
#
# The device below is therefore built by name and NEVER adopted before `record!`,
# which is the order that used to leave the blit on the wrong queue.

@testset "recording adopts this device's queue too" begin
    mtldev = Metal.device()
    for q in (Mantle.LegacyQueue(mtldev), Mantle.MTL4Queue(mtldev))
        MTLm.supports_mtl4(mtldev) || q isa Mantle.LegacyQueue || continue
        Mantle.adoptqueue!(DEV_SEAM)          # point the process somewhere else first
        dev = Mantle.MetalDevice(mtldev, q)
        g = M.Graph(dev)
        a = M.Buffer(dev, zeros(Float32, 256))
        b = M.Buffer(dev, fill(1.0f0, 256))
        flag = M.Buffer(dev, Int32[4])
        M.repeat!(g, 4; while_nonzero = flag) do i
            M.compute!(g, "it-$i") do p
                M.dispatch!(p, seam_gatecount!,
                            (M.use(p, a; read = true, write = true),
                             M.use(p, b; read = true),
                             M.use(p, flag; read = true, write = true)), 256)
            end
        end
        pl = M.Plan(g)
        @test Metal.adopted_queue[] === Mantle.batchqueue(DEV_SEAM)
        M.record!(pl)
        @test Metal.adopted_queue[] === Mantle.batchqueue(dev)
        # THE FIRST RUN is the whole test: four iterations, not zero.
        M.run!(pl); M.waitfor!(pl)
        @test Array(M.storage(a))[1] == 4.0f0
        @test Array(M.storage(flag))[1] == 0
    end
    Mantle.adoptqueue!(DEV_SEAM)
end

# ── And the one that only the crown found ────────────────────────────────────
#
# A recording is not read-only. A `repeat!` gate's execution range lives in the
# recording's own buffer, written by a one-thread dispatch inside the replay and
# read by the command processor later in the SAME replay — so two replays of one
# recording in flight at once are two writers of those bytes.
#
# The legacy path never has to say so: everything it submits is on one queue and
# Metal runs a queue's command buffers in commit order. MTL4 does no hazard
# tracking between command buffers, so overlapping replays corrupt each other's
# ranges and the command processor stops on one that no longer names a valid
# range: no error, nothing in the system log, just a submission that never
# signals and a host that blocks forever in the next `opensubmit!`.
#
# It needs THREE things at once, which is why everything smaller passed: a gated
# plan, several submissions in flight, and enough work per frame that they really
# do overlap. `run!` without `waitfor!` in between is what a multi-sample frame
# does, and it is what this asks for.

@kernel function seam_gate!(dst, @Const(src), flag)
    i = @index(Global)
    @inbounds dst[i] += src[i]
    i == 1 && (@inbounds flag[1] -= Int32(1))
end

@testset "replays of one recording do not overlap" begin
    n = 1 << 20
    iters = 8
    g = M.Graph(DEV_SEAM)
    a = M.Buffer(DEV_SEAM, zeros(Float32, n))
    b = M.Buffer(DEV_SEAM, fill(1.0f0, n))
    flag = M.Buffer(DEV_SEAM, Int32[iters])
    M.repeat!(g, iters; while_nonzero = flag) do i
        M.compute!(g, "it-$i") do p
            M.dispatch!(p, seam_gate!,
                        (M.use(p, a; read = true, write = true),
                         M.use(p, b; read = true),
                         M.use(p, flag; read = true, write = true)), n)
        end
    end
    plan = M.record!(M.Plan(g))
    @test count(s -> s.slot >= 0, plan.recording.segments) == iters

    # One run with the gate open: every iteration runs, and the last one closes it.
    M.run!(plan); M.waitfor!(plan)
    @test Array(M.storage(a))[1] == Float32(iters)
    @test Array(M.storage(flag))[1] == 0

    # Now back to back, filling the ring. The flag is NOT reopened, deliberately:
    # a host write to it between runs would race the replays still in flight and
    # make the count nondeterministic, which would be the test's bug rather than
    # the backend's. With the gate closed the iterations are discarded, but the
    # RANGE WRITERS still run in every replay and still write the bytes the
    # command processor reads — which is the hazard. Before the submissions were
    # ordered against each other, this did not return at all.
    runs = 12
    for _ in 1:runs
        M.run!(plan)
    end
    M.waitfor!(plan)
    # Nothing more ran, and nothing corrupted what had.
    @test Array(M.storage(a))[1] == Float32(iters)
    @test Array(M.storage(flag))[1] == 0
    # Every submission signalled: the ring never blocked on one that did not.
    DEV_SEAM.queue isa Mantle.MTL4Queue &&
        @test DEV_SEAM.queue.event.signaledValue >= DEV_SEAM.queue.next - 1
end

@kernel function seam_chain!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

@testset "an upload is ordered against the replay that reads it" begin
    for n in (256, 1 << 20)
        g = M.Graph(DEV_SEAM)
        a = M.Buffer(DEV_SEAM, zeros(Float32, n))
        b = M.Buffer(DEV_SEAM, fill(1.0f0, n))
        for i in 1:8
            M.compute!(g, "add-$i") do p
                M.dispatch!(p, seam_chain!,
                            (M.use(p, a; read = true, write = true),
                             M.use(p, b; read = true)), n)
            end
        end
        pl = M.record!(M.Plan(g))
        # The FIRST run is the one that catches it: that is when the plan's host
        # writes are still pending, so the frame holds both an upload and a
        # replay.
        for run in 1:3
            M.run!(pl)
            M.waitfor!(pl)
            @test Array(M.storage(a))[1] == Float32(8 * run)
        end
    end
end

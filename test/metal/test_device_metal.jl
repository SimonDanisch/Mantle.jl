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
        # What a submission touches is an argument to OPENING it, because the two
        # generations need it at different moments and only one of those is
        # expressible after the fact: legacy declares on the encoder
        # (`useResource`, which is hazard tracking as much as residency), MTL4
        # must have its residency set complete before the command buffer names it.
        sub = Mantle.opensubmit!(d, MTLm.MTLBuffer[])
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

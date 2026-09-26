# The ROCm backend: the same graph, the same five analyses, on HIP.
#
# Beside `test/vulkan/` and `test/metal/`, and it is the same kind of file — what
# one backend answers differently, checked against what core decided. The three
# things pinned here are the three that were WRONG in the first version of
# `ext/MantleROCmExt.jl` and that nothing else would have caught:
#
#   1. A view's address. `ROCArray.offset` is in bytes in AMDGPU 2.2.1 as
#      installed and in elements in a 2.2.1 checkout beside it, so a backend
#      that picks one silently places every transient at a quarter of its
#      offset on the other. The symptom was a chain that came back holding
#      8.0, 9.0, 10.0 and 11.0 in the four quarters of a buffer.
#   2. That a run's work lands on the stream the waits use. It did not, because
#      `AMDGPU.stream!(f, s)` never binds `s` (see `onstream`), and a wait on an
#      empty stream returns at once — which reported 5.7 TB/s on a part whose
#      measured rate is 279 GB/s.
#   3. That a captured graph holds ADDRESSES and not values. A `hipGraph` is the
#      submit baked; if it baked the data instead, every run after the first
#      would be a replay of the first one's answer.
#
# Requires AMDGPU.jl, which Mantle weak-depends on: `import AMDGPU` loads
# `MantleROCmExt` and there is nothing else to do.
#
# `include`ing the extension file by hand is what a missing `[weakdeps]` /
# `[extensions]` pair in `Project.toml` forces, since nothing loads
# it otherwise. That include is not a substitute: it defines the extension's
# methods in `Main`, so `Base.get_extension(Mantle, :MantleROCmExt)` answers
# `nothing` and `Mantle.openrecording` ends up with TWO methods for two
# `ROCmDevice` types that are not the same type. With the extension declared,
# asking for it by name is the whole of it.

using Test, LinearAlgebra
import Mantle, AMDGPU, KernelAbstractions, KernelInterface
using KernelAbstractions: @kernel, @index, @Const
const M = Mantle
const KI = KernelInterface

const RE = Base.get_extension(Mantle, :MantleROCmExt)
RE === nothing && error("test_rocm.jl: `import AMDGPU` did not load " *
                        "MantleROCmExt. Check `[extensions]` in Mantle's Project.toml.")

@kernel function rocadd!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1.0f0
end

function rocmacrofree!(dst, src)
    i = Int(KI.get_global_id().x)
    i <= length(dst) && (@inbounds dst[i] = src[i] + 1.0f0)
    return
end

@testset "ROCm: core infers a macro-free kernel through AMDGPU" begin
    dev = M.Device(M.ROCmAPI())
    # The algorithm stays in Mantle. ROCm supplies only the compiler
    # interpreter needed to inspect the IR; neither the top-level routing nor
    # KA's ordinary signature construction is copied into the extension.
    @test which(M.kerneltouches,
                (typeof(dev), typeof(rocmacrofree!), Tuple, Int, Nothing)).module === M
    @test which(M.kakernelaccesssignature,
                (typeof(dev), typeof(rocadd!), Tuple, Int, Nothing)).module === M
    @test which(M.kernelinterpreter,
                (typeof(dev), typeof(rocmacrofree!), Type{Tuple{}})).module === RE
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, 1024))
    dst = M.Transient.Buffer(g, Float32, 1024)
    M.dispatch!(g, rocmacrofree!, (dst, src), 1024; name = "macro-free")
    pl = M.Plan(g)
    M.record!(pl)
    M.run!(pl)
    @test all(==(2.0f0), Array(M.storage(dst)))
end

"A forced chain: each pass reads what the previous wrote."
function chain(dev, n::Integer, links::Integer; alias::Bool = true)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(0.0f0, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:links]
    prev = src
    for (k, dst) in enumerate(t)
        M.dispatch!(g, rocadd!, (dst,
                                     prev), n; name = "l$k")
        prev = dst
    end
    return g, t, M.Plan(g; alias)
end

@testset "ROCm: the graph runs, in order, on the bytes the placer chose" begin
    dev = M.Device(M.ROCmAPI())
    # Big enough that the transients land at nonzero offsets far apart — 4 MiB
    # a link, so a byte/element mixup puts a view 4x too close to the base and
    # the overlap is visible in the result rather than in a pointer.
    n, links = 1 << 20, 8
    _, t, pl = chain(dev, n, links)
    M.record!(pl)
    M.run!(pl)
    out = Array(M.storage(t[end]))
    # 8 exactly: only correct if every pass ran in the scheduled order, and if
    # each read the transient the placer gave the previous one.
    @test all(==(Float32(links)), out)
    # …and the placer really did overlap them. If this ever equals `naive` the
    # test above stops being a test of anything.
    @test M.peakbytes(pl) < M.naivebytes(pl)
    @test M.peakbytes(pl) == 2 * n * sizeof(Float32)
end

@testset "ROCm: a view sits where Mantle's region does" begin
    dev = M.Device(M.ROCmAPI())
    n, links = 1 << 20, 4
    _, t, pl = chain(dev, n, links)
    M.record!(pl); M.run!(pl)
    # Two slots, alternating, and the addresses have to differ by the offset
    # the placer computed — in BYTES. `rocview` checks this itself on every
    # view; this is the same claim from outside, on a real plan.
    for x in t
        a = M.storage(x)
        base = convert(Ptr{Float32}, a.buf[].mem)
        @test pointer(a) == base + x.offset
    end
    @test M.storage(t[1]).offset != M.storage(t[2]).offset
    @test M.storage(t[1]).offset == M.storage(t[3]).offset
end

@testset "ROCm: the waits cover the stream the work is on" begin
    dev = M.Device(M.ROCmAPI())
    # 16 MiB a link and eight links is 268 MB of traffic, which cannot happen
    # in less than a millisecond on this part. A wait that returns before the
    # work is done shows up as a physically impossible rate, which is exactly
    # how the borrowed-stream bug presented.
    n, links = 1 << 22, 8
    _, t, pl = chain(dev, n, links)
    M.record!(pl); M.run!(pl); M.waitidle(dev)
    t0 = time_ns()
    M.run!(pl)
    M.waitidle(dev)
    ms = (time_ns() - t0) / 1e6
    gb = links * n * sizeof(Float32) * 2 / 1e9
    # 2 TB/s is comfortably past anything this class of part can do, cache
    # included, so this fails only if the wait did not wait.
    @test gb / (ms / 1e3) < 2000
    @test all(==(Float32(links)), Array(M.storage(t[end])))
end

@testset "ROCm: a captured graph holds addresses, not values" begin
    dev = M.Device(M.ROCmAPI())
    n = 4096
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(0.0f0, n))
    dst = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, rocadd!, (dst,
                                 src), n; name = "one")
    pl = M.Plan(g)
    # `true`: every plan this backend accepts is recordable, and the ones it
    # cannot record it refuses out loud rather than handing back unrecorded.
    @test M.recordsplans(dev)
    M.record!(pl)
    @test pl.recording isa RE.ROCmRecording
    M.run!(pl); M.waitidle(dev)
    @test all(==(1.0f0), Array(M.storage(dst)))
    # A different input, into the same buffer, through the SAME recording: a
    # captured graph has to hold the ADDRESS and not the value.
    M.upload!(dev, src.store, 1, fill(41.0f0, n))
    M.run!(pl); M.waitidle(dev)
    @test all(==(42.0f0), Array(M.storage(dst)))
end

@testset "ROCm: capture acquires every argument on its own stream" begin
    dev = M.Device(M.ROCmAPI())
    n = 4096
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(6.0f0, n))
    dst = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, rocadd!, (dst, src), n; name = "foreign owner")
    pl = M.Plan(g)

    # Reproduce an argument last owned by another stream. AMDGPU's kernel
    # wrapper normally synchronises that owner while converting its arguments;
    # HIP forbids the synchronisation after capture has started. `record!` must
    # transfer ownership before opening capture, on the exact stream `dev` owns.
    foreign = AMDGPU.HIPStream()
    AMDGPU.rocconvert(M.storage(src), foreign)
    M.record!(pl)
    M.run!(pl); M.waitidle(dev)
    @test all(==(7.0f0), Array(M.storage(dst)))
end

@testset "ROCm: record! writes the plan down and does not run it" begin
    dev = M.Device(M.ROCmAPI())
    # A pass that ACCUMULATES, so a run that happens when it should not is
    # visible in the value rather than only in the timing. An early draft of this
    # backend walked the plan once uncaptured to get its kernels compiled and
    # captured a second walk: on an idempotent chain that is invisible, and on
    # SAM 2.1's encoder it overflowed an fp16 residual stream and returned `NaN`
    # from three of six outputs with nothing raised. Compiling in
    # `compile_dispatch` is what removed the extra walk.
    n = 1024
    g = M.Graph(dev)
    acc = M.Buffer(dev, zeros(Float32, n))
    M.dispatch!(g, rocadd!, (acc,
                                 acc), n; name = "bump")
    pl = M.Plan(g)
    M.record!(pl)
    M.waitidle(dev)
    @test all(==(0.0f0), M.download(dev, acc.store))
    @test pl.recording isa RE.ROCmRecording
    # One run, one increment. A capture that had executed as well as recorded, or
    # a graph holding two copies of the pass, shows up here on the first run.
    for k in 1:4
        M.run!(pl); M.waitidle(dev)
        @test all(==(Float32(k)), M.download(dev, acc.store))
    end
end

# `openrecording` here took `(dev, plan)` after core's grew a `passes` range on
# 2026-09-22, so it was a new function nothing called. Core's default declined,
# every ROCm plan ran unrecorded, and the `isa ROCmRecording` checks in this file
# failed along with the partition test below — which nobody ran; Qwen-Image's VAE
# found it by asking for `maxpasses`. Pinned by the method core actually calls.
@testset "ROCm: openrecording is the method core calls" begin
    @test which(M.openrecording, (RE.ROCmDevice, M.Plan, UnitRange{Int})).module === RE
end

@testset "ROCm: a dispatch is compiled once, at Pipelines time" begin
    dev = M.Device(M.ROCmAPI())
    _, _, pl = chain(dev, 1024, 3)
    # Not a `Launch`: core's interpreted bake re-enters KernelAbstractions and
    # rebuilds the iteration space per run, and a kernel compiled inside a
    # capture invalidates it. `compile_dispatch` answers with something already
    # compiled, which is what lets `record!` capture on the plan's own walk.
    ds = collect(Iterators.flatten(pp.dispatches for pp in pl.passes))
    @test length(ds) == 3
    @test all(d -> !(d isa M.Launch), ds)
    @test all(d -> M.argsize(d) == 0 && M.indirectindex(d) == 0, ds)
end

@testset "ROCm: capabilities are the native backend's exact floor" begin
    dev = M.Device(M.ROCmAPI())
    # `repeat!` over fixed-size work needs the host to read a device-written
    # flag between passes; core does that by scalar-indexing the predicate,
    # which AMDGPU refuses. A gated pass is therefore refused at graph build.
    @test !M.supportspredicate(dev)
    arch = first(split(AMDGPU.HIP.gcn_arch(dev.dev), ':'))
    if startswith(arch, "gfx11") || startswith(arch, "gfx12")
        @test M.caps(dev).coopmat
        @test M.caps(dev).tile == 16
        @test M.caps(dev).shapes ==
              [KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope())]

        # The advertised floor has to compile through its real consumer, not
        # merely exist in the capability table.  This takes the vector-packed
        # local-memory loads and split-K scratch/reduction paths.
        ah = rand(Float16, 64, 64)
        bh = rand(Float16, 64, 64)
        a, b = AMDGPU.ROCArray(ah), AMDGPU.ROCArray(bh)
        c = AMDGPU.zeros(Float32, 64, 64)
        M.coopmat_gemm!(c, a, b, 64, 64, 64)
        AMDGPU.synchronize()
        @test maximum(abs.(Array(c) .- Float32.(ah) * Float32.(bh))) < 0.01

        # And through the declared path. This exercises Mantle's access walk
        # over AMDGPU's native LLVMPtr arithmetic and WMMA loads/stores; an
        # immediate launch alone cannot catch a lost read/write declaration.
        gg = M.Graph(dev)
        ga, gb = M.Buffer(dev, ah), M.Buffer(dev, bh)
        blk_split = M.coopmat_gemm_shape(64, 64, 64)
        splitk = blk_split[2]
        gc = M.Transient.Buffer(gg, Float32, 64, 64, max(splitk, 1))
        M.coopmat_gemm_dispatch!(gg, gc, ga, gb, 64, 64, 64;
                                 blk_split, partials = gc, reduce = false,
                                 name = "declared coopmat")
        gp = M.Plan(gg)
        M.record!(gp); M.run!(gp); M.waitidle(dev)
        got = dropdims(sum(Array(M.storage(gc)); dims = 3); dims = 3)
        @test maximum(abs.(got .- Float32.(ah) * Float32.(bh))) < 0.01
    else
        @test !M.caps(dev).coopmat
        @test isempty(M.caps(dev).shapes)
    end
    @test isempty(M.caps(dev).wggran)
    @test M.caps(dev).subgroup == 32
    props = AMDGPU.HIP.properties(dev.dev)
    @test M.caps(dev).warps ==
          Int(props.maxThreadsPerMultiProcessor) ÷ Int(props.warpSize)
end


# "ROCm: record_into sees more than kernel launches" was here, deleted
# 2026-09-15 with the capture path. It pinned that a capture noticed the three
# things a kernel library submits which are not KernelAbstractions launches — a
# bare `@roc` (`mapreduce`), a `hipMemset` behind `fill!`, a rocBLAS `mul!` —
# because each one missed made a recorded plan silently wrong rather than
# noisy: the encoder replayed with its GEMMs absent and came back `NaN` in three
# of six outputs while the three constants stayed bit-identical.
#
# Nothing to notice now. A library call is declared —
# `dispatch!(p, BLASMul(...), (use(p, C; write = true), ...), 0)` — and
# `compile_dispatch` builds a `ROCmCallDispatch` from any kernel that is not a
# `KA.Kernel`. What the deleted testset was really guarding is covered by the
# graph being written down instead of discovered.


@testset "ROCm: a move invalidates the recording, and is then refused" begin
    dev = M.Device(M.ROCmAPI())
    # The Vulkan model is that a recorded plan SURVIVES its buffers resizing:
    # every device address it holds is eight bytes in the plan's argument memory,
    # so `notify_move!` patches them and the recording stands. A captured HIP
    # graph holds its kernel arguments where nothing can rewrite them — capture
    # does not hand back the node handles `hipGraphExecKernelNodeSetParams`
    # needs — so this backend answers `patchable` with `false` and core throws
    # the recording away instead.
    #
    # Before that answer existed, such a plan fell through `notify_move!`'s
    # `isempty(patchtab)` test (it has no patch table, because there is nowhere
    # for one to point) and went on running commands that named the freed
    # storage. Which is the bug `deviceaddress` replaced `resource_moved!` to
    # stop, so it is worth a test rather than a comment.
    @test !M.patchable(dev)
    n = 512
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(2.0f0, n); capacity = n)
    dst = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, rocadd!, (dst,
                                 src), n; name = "one")
    pl = M.Plan(g)
    M.record!(pl)
    @test pl.recording isa RE.ROCmRecording
    M.run!(pl); M.waitidle(dev)
    @test all(==(3.0f0), Array(M.storage(dst)))
    first = pl.recording

    # Past its capacity, so the store really moves and the addresses the graph
    # holds are freed.
    resize!(src, 2n)
    M.upload!(dev, src.store, 1, fill(40.0f0, 2n))
    M.waitidle(dev)
    @test pl.recording === nothing      # core invalidated it
    @test pl.stale

    # …and the next run REFUSES, because re-recording is not enough: the
    # compiled dispatch holds the array it resolved at `Pipelines` time, not a
    # pointer into argument memory the way Vulkan's does, so the walk it would
    # capture still names the previous storage. Saying so beats replaying that
    # buffer's answer, which is what happens without `checkresolved`.
    @test_throws ArgumentError M.run!(pl)
    # A plan built again against the resized buffer gives the new answer.
    g2 = M.Graph(dev)
    dst2 = M.Transient.Buffer(g2, Float32, n)
    M.dispatch!(g2, rocadd!, (dst2,
                                 src), n; name = "one")
    pl2 = M.Plan(g2); M.record!(pl2)
    M.run!(pl2); M.waitidle(dev)
    @test all(==(41.0f0), Array(M.storage(dst2)))
end

@testset "ROCm: replaying a baked plan allocates nothing" begin
    dev = M.Device(M.ROCmAPI())
    n, links = 4096, 16
    _, t, pl = chain(dev, n, links)
    M.record!(pl)
    # No argument memory on this backend, and that is not an omission. Vulkan's
    # `ArgMemory` is one packed blob of argument bytes laid out at compile, with
    # the recording holding pointers into it — which is what makes a moved
    # resource patchable there. This backend answers `argbytes` with nothing, so
    # `makeargmemory` builds none: a captured graph node carries its own kernel
    # arguments, and `compile_dispatch` resolved the arrays once, at `Pipelines`
    # time. `patchable` being `false` is the other half of the same fact.
    @test pl.args === nothing
    for _ in 1:4; M.run!(pl); end
    M.waitidle(dev)
    # The property baking is FOR. A walk of this plan builds a KA kernel object
    # and a launch configuration per dispatch and reaches the allocator for both
    # — measured 40,976 bytes for a 64-pass chain, about 640 a dispatch — where a
    # replay is one `hipGraphLaunch` and touches no host memory at all. A replay
    # that allocates is a replay that can be interrupted by a garbage collection
    # while the previous launch is still in flight.
    @test all(==(0), [(@allocated M.run!(pl)) for _ in 1:8])
    M.waitidle(dev)
    @test all(==(Float32(links)), Array(M.storage(t[end])))
end

@testset "ROCm: hipBLASLt bias epilogue is one recorded call" begin
    dev = M.Device(M.ROCmAPI())
    m, k, n = 48, 32, 64
    ah = randn(Float16, m, k)
    bh = randn(Float16, k, n)
    zh = randn(Float16, m)
    g = M.Graph(dev)
    a = M.Buffer(dev, ah)
    b = M.Buffer(dev, bh)
    z = M.Buffer(dev, zh)
    d = M.Transient.Buffer(g, Float16, m, n)
    call = M.librarygemm(dev, d, a, b, z, identity)
    if call === nothing
        # hipBLASLt is optional even when the HIP backend itself is present.
        @test dev.blaslt == C_NULL
    else
        M.dispatch!(g, call, (d, a, b, z); name = "gemm+bias")
        pl = M.Plan(g)
        # Initialise the library's implicit algorithm selection before stream
        # capture, just as DNNKernels' immediate first run does.
        call(M.storage(d), M.storage(a), M.storage(b), M.storage(z))
        M.waitidle(dev)
        M.record!(pl)
        fill!(M.storage(d), Float16(0))
        M.run!(pl); M.waitidle(dev)
        got = Float32.(Array(M.storage(d)))
        want = Float32.(ah) * Float32.(bh) .+ Float32.(zh)
        @test got ≈ want rtol=2e-3 atol=2e-2
        @test length(pl.passes) == 2 # updates plus the single fused call
    end
end

@testset "ROCm: a partitioned recording is several graphs, in order" begin
    dev = M.Device(M.ROCmAPI())
    # `record!(pl; maxpasses = N)` splits the baked commands into submissions of
    # at most N passes, so a workload longer than the driver's submission timeout
    # has completion points inside it. HIP has no equivalent of one submission
    # carrying several command buffers, so the pieces here are separate
    # `hipGraphLaunch`es on the same in-order stream — same ordering, same
    # completion points.
    #
    # This backend implements no `recordplan!` and no sequence type: the chunked
    # walk is core's `recordparts!` and the sequence is core's `RecordingParts`,
    # so all this backend answers is `submitrecording!` for ONE piece. Refusing a
    # partition outright is what follows from the walk living in the Vulkan backend
    # and nothing portable built one — which is also why `HorizonRunner`, whose
    # prefill asks for 64, could only be recorded there.
    n, links = 1024, 8
    _, ta, pa = chain(dev, n, links)
    M.record!(pa)
    M.run!(pa); M.waitidle(dev)
    want = Array(M.storage(ta[end]))
    @test all(==(Float32(links)), want)

    _, tb, pb = chain(dev, n, links)
    M.record!(pb; maxpasses = 3)
    @test pb.recording isa M.RecordingParts
    # `links` compute passes plus the update pass core adds for the host-written
    # source, in pieces of three.
    @test length(pb.passes) == links + 1
    @test length(pb.recording.parts) == cld(links + 1, 3)
    @test all(p -> p isa RE.ROCmRecording, pb.recording.parts)
    @test pb.record_maxpasses == 3
    M.run!(pb); M.waitidle(dev)
    # The dependencies hold ACROSS the split: every link reads what the previous
    # wrote, and a piece is a separate launch.
    @test Array(M.storage(tb[end])) == want
    # …and a second run replays the same pieces rather than the first one only.
    fill!(M.storage(tb[end]), 0.0f0)
    M.run!(pb); M.waitidle(dev)
    @test Array(M.storage(tb[end])) == want
end

@kernel function roc2d!(dst, @Const(src), n1)
    i, j = @index(Global, NTuple)
    k = (j - 1) * n1 + i
    @inbounds dst[k] = src[k] + 1.0f0
end

@testset "ROCm: a compiled dispatch is partitioned the way an eager launch is" begin
    dev = M.Device(M.ROCmAPI())
    # A 2-D iteration space, which is what makes this visible. `compile_dispatch`
    # has to reproduce the partition `ROCKernels` picks per launch: occupancy
    # gives a thread count, and `threads_to_workgroupsize` SHAPES it to the
    # ndrange. `KA.launch_config` has a fallback of its own for a kernel launched
    # without a workgroup size — `min(prod(ndrange), _max_group_size)` in the
    # first dimension and 1 in the rest — and it returns that in place of the
    # `nothing` it was given. A backend that then asks "was a size requested?" by
    # looking at the RETURNED value never sees `nothing` and never tunes.
    #
    # It is not a small difference, because the fallback's shape ignores the
    # extent of the first dimension: for (64, 1024) it groups 1024 threads along
    # an axis 64 long, so 1,048,576 workitems run for a 65,536-element space and
    # fifteen out of sixteen exit on the bounds check. SAM 2.1's encoder is 1,014
    # broadcast kernels launched exactly this way and it cost 29% of the whole
    # graph — 647.3 ms replayed against 454.9 ms eager, which read as "baking a
    # plan makes it slower" until this was the reason.
    dims = (64, 1024)
    n = prod(dims)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, n))
    dst = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, roc2d!, (dst,
                                src, dims[1]), dims; name = "k")
    pl = M.Plan(g)
    disp = only([d for pp in pl.passes for d in pp.dispatches
                 if d isa RE.ROCmCompiledDispatch])
    # The contract, against the kernel that will actually run rather than against
    # a number: the same two calls `ROCKernels` makes.
    tuned = AMDGPU.ROCKernels.threads_to_workgroupsize(
        AMDGPU.launch_configuration(disp.kernel).groupsize, dims)
    @test disp.groupsize == prod(tuned)
    # …and the consequence. Every dimension of this ndrange divides its group, so
    # a partition shaped to it launches the iteration space exactly.
    @test disp.groupsize * disp.gridsize == n
    M.record!(pl)
    M.run!(pl)
    @test all(==(2.0f0), Array(M.storage(dst)))
end

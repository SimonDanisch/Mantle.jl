# The device-owned arena and the recorded plan. Needs a GPU, but no display:
# every graph here is headless, which is also the only kind `record!` takes
# today.
using Mantle, Test, KernelAbstractions, Lava, Statistics
const M = Mantle
# The backend this run is for. `runtests.jl` includes this file once per
# available backend (`Mantle.eachbackend()`); a bare `include` from the REPL
# gets the default one. Nothing below names a backend, which is the point:
# these testsets check PORTABLE behaviour and used to check it on Vulkan only.
# Vulkan-gated until its three backend-specific assertions are split out; see
# the note beside its include in `runtests.jl`.
const TESTBACKEND = M.VulkanAPI()


@kernel function bump!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1f0
end

"""
A chain of `nstage + 1` transients: stage k reads t[k] and writes t[k+1], so only
adjacent ones are ever live together and the plan's peak is about two buffers
however long the chain is. `out` ends at `nstage + 2` — seed writes 1 and every
stage adds 1 — which is what makes a wrong answer legible rather than merely
different.
"""
function chainplan(dev, n, nstage)
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:(nstage + 1)]
    M.compute!(g, "seed") do p
        s = M.use(p, seed; read = true); d = M.use(p, t[1]; write = true)
        M.dispatch!(p, bump!, (d, s), n)
    end
    for k in 1:nstage
        M.compute!(g, "s$k") do p
            s = M.use(p, t[k]; read = true); d = M.use(p, t[k + 1]; write = true)
            M.dispatch!(p, bump!, (d, s), n)
        end
    end
    out = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "out") do p
        s = M.use(p, t[end]; read = true); d = M.use(p, out; write = true)
        M.dispatch!(p, bump!, (d, s), n)
    end
    (; g, out, seed, plan = M.record!(M.Plan(g)), keep = (seed, t))
end

"""Two transients whose lifetimes overlap, so they cannot share bytes. The
persistent buffer stays tiny on purpose: it is the *placement* that has to exceed
the device, not an allocation on the way to it."""
function fatplan(dev, n)
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, 16))
    t1 = M.Transient.Buffer(g, Float32, n)
    t2 = M.Transient.Buffer(g, Float32, n)
    M.compute!(g, "a") do p
        s = M.use(p, seed; read = true); d = M.use(p, t1; write = true)
        M.dispatch!(p, bump!, (d, s), 16)
    end
    M.compute!(g, "b") do p
        s = M.use(p, t1; read = true); d = M.use(p, t2; write = true)
        M.dispatch!(p, bump!, (d, s), 16)
    end
    (; plan = M.record!(M.Plan(g)), keep = (seed, t1, t2))
end

@kernel function add2!(d, @Const(x), @Const(y))
    i = @index(Global)
    @inbounds d[i] = x[i] + y[i]
end

# `pool_offset` below is the BACKEND's — it names an offset inside a
# `VkBuffer` — so this file is one of the ones 0.7 still owes `test/vulkan/`.
const E = Mantle

@testset "two plans in one process commit the max, not the sum" begin
    # The gate for the device-owned arena. It can only pass if the `Device` owns
    # the reservation: two plans that each allocate cannot share, whatever the
    # placer does inside either of them.
    #
    # `small` is built FIRST so that `big` has to grow the pool underneath it and
    # remap it. A correct answer from `small` afterwards is `remap!` having
    # re-materialised its transients into the new allocation — which is the half
    # of this that a test built in the other order would not reach.
    dev = M.Device(TESTBACKEND)
    small = Base.invokelatest(chainplan, dev, 250_000, 3)
    big   = Base.invokelatest(chainplan, dev, 2_000_000, 3)

    ps, pb = M.peakbytes(small.plan), M.peakbytes(big.plan)
    # The arena lives on the core `Pool` now, not on the backend's device: the
    # sharing this asserts is a property every backend gets, not Lava's.
    arena = M.pool(dev).arenas[E.Buffers()]
    @test arena.bytes == max(ps, pb)
    @test arena.bytes < ps + pb
    @test length(M.tenants!(arena)) == 2
    @test M.sharing(M.pool(dev), E.Buffers())

    # Interleaved, a hundred times: two plans alternating on one arena, each
    # run's recording opening with the global barrier that orders it behind
    # whatever ran last. The arena no longer remembers who that was — the
    # handover used to be decided at record time from its memory of the last
    # runner, so of two recorded plans alternating the first never got one.
    for _ in 1:100
        M.run!(small.plan)
        M.run!(big.plan)
    end
    KernelAbstractions.synchronize(M.backend(dev))
    @test all(==(5f0), Array(M.storage(small.out)))
    @test all(==(5f0), Array(M.storage(big.out)))
    @test !hasfield(M.Arena, :lastrun)
end

@testset "over budget fails at compile, with numbers" begin
    dev = M.Device(TESTBACKEND)
    # Whichever bound is binding on THIS device. `maxalloc` is 4 GB on the APU
    # this was written against and `typemax(Int)` — "no limit" — on NVIDIA, so a
    # test pinned to it passes on one machine and allocates 8 exabytes on the
    # other. `headroom` is the number the compiler actually checks against.
    n = M.headroom(M.pool(dev), dev, E.Buffers()) ÷ sizeof(Float32)
    err = try
        Base.invokelatest(fatplan, dev, n)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("does not fit", msg)
    @test occursin("Lower bound", msg)         # so a reader can tell placement from size
    @test occursin("Largest items", msg)       # and knows which buffer to drop
    # Nothing was allocated on the way to the throw: the check runs on the
    # placement, before `reserve!`.
    @test !haskey(M.pool(dev).arenas, E.Buffers()) ||
          M.pool(dev).arenas[E.Buffers()].bytes < n * sizeof(Float32)
end

@testset "a pass barrier carries mask tuples, not buffers" begin
    # This used to assert the opposite: that a `renameable` resource was widened
    # to a global memory barrier while everything else kept a barrier scoped to
    # its (VkBuffer, offset, size). The `renameable` case was the general one —
    # a recording cannot bake a handle for anything that can move, and baking
    # makes everything movable — so `build_pass_barrier` emits one
    # `VkMemoryBarrier2` per distinct `(waits, to)` tuple and no buffer barriers
    # at all. `barrierspan`, `barrierbuffer` and the special case are gone.
    #
    # `renameable` itself is gone with renaming: `a` and `b` are read by the pass
    # below, so `Plan` registers both as `CopyDst` of the update pass and writes
    # them in place, and neither can move its target — the distinction has
    # nothing left to decide. What is worth pinning is the tuples being `unique`
    # and never unioned: `a` and `b` are the same hazard and collapse into one
    # barrier, which is why three transitions produce two.
    dev = M.Device(TESTBACKEND)
    g = M.Graph(dev)
    a   = M.Buffer(dev, zeros(Float32, 1024))
    b   = M.Buffer(dev, zeros(Float32, 1024))
    out = M.Buffer(dev, zeros(Float32, 1024))
    M.compute!(g, "read") do p
        sa = M.use(p, a; read = true); sb = M.use(p, b; read = true)
        d  = M.use(p, out; write = true)
        M.dispatch!(p, add2!, (d, sa, sb), 1024)
    end
    plan = Base.invokelatest(M.Plan, g)

    pp = only(p for p in plan.passes if p.pass.name == "read")
    # Three transitions: `a` and `b` are both a copy made visible to a shader
    # read, and `out`'s is the store's copy made visible to a shader write.
    @test length(pp.pre) == 3
    v = pp.barrier.vks
    # Two of them, because the first two are the same tuple. A union of all three
    # would make the shader writes visible to a copy nothing performs and drag in
    # caches no hazard here touches.
    @test Int(v.memoryBarrierCount) == 2
    # And no handle, no offset, no range: a recording names no `VkBuffer`, which
    # is what lets a buffer move under a recorded plan without invalidating it.
    @test Int(v.bufferMemoryBarrierCount) == 0

    mb = unsafe_wrap(Array, v.pMemoryBarriers, Int(v.memoryBarrierCount))
    @test allunique((m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask)
                    for m in mb)
end

@testset "a recorded plan runs bit-exact, and records nothing" begin
    dev = M.Device(TESTBACKEND)
    s = Base.invokelatest(chainplan, dev, 20_000, 200)     # 202 passes, recorded at build
    @test M.recorded(s.plan)
    @test M.record!(s.plan) === s.plan                     # idempotent
    # ONE recording, not one per argument slot: nothing rewrites a plan's
    # argument memory after `record!`, so there is nothing for a ring to protect.
    @test s.plan.recording isa Mantle.Recording

    M.run!(s.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    want = copy(Array(M.storage(s.out)))

    # Dispatches the host recorded INTO THE FRAME'S BATCH, counted at submit.
    # This is the assertion, and it used to be a wall-clock ratio: baked had to
    # be three times faster than unbaked over thirty runs.
    #
    # That measurement stopped meaning what it said the day `run!` started
    # rotating argument slots for a baked plan too: a recorded run submitted
    # every run and waited when the host got `ARG_SLOTS` ahead of the device,
    # which is the same backpressure the interpreted path always had — so both
    # medians were GPU throughput for this graph (0.73 ms against 0.64 ms
    # measured). Nothing regressed; the clock was simply no longer measuring
    # host work. (The ring is gone now and so is the wait, but the point about
    # what a wall-clock ratio measures here stands.)
    #
    # Counting is better than timing anyway: it is exactly the claim in the name
    # of this testset, it is not a ratio anybody has to keep generous, and it does
    # not move on a shared machine. `test_recording_lifecycle.jl` owns the
    # host-cost comparison, where the two sides genuinely differ.
    diag = Mantle.ctxof(M.batchqueue(dev)).diag
    function batchrecorded(f, n)
        f()                                  # warm, and outside the count
        KernelAbstractions.synchronize(M.backend(dev))
        before = diag.total_dispatches[]
        for _ in 1:n
            f()
        end
        KernelAbstractions.synchronize(M.backend(dev))   # flushes the trailing batch
        diag.total_dispatches[] - before
    end

    fill!(M.storage(s.out), 0f0)
    KernelAbstractions.synchronize(M.backend(dev))
    @test batchrecorded(() -> M.run!(s.plan), 30) == 0
    KernelAbstractions.synchronize(M.backend(dev))
    @test Array(M.storage(s.out)) == want                  # bit-exact, not merely close
end

@testset "a recorded plan survives a full GC between runs" begin
    # A plan holds references to every resource its recording names, so a full
    # collection between two runs cannot free storage the command buffer points
    # at. `test_recording_lifecycle.jl` asserts the same thing on a smaller plan;
    # this is the 50 000-element chain.
    dev = M.Device(TESTBACKEND)
    p = Base.invokelatest(chainplan, dev, 50_000, 8)
    M.run!(p.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    want = copy(Array(M.storage(p.out)))

    M.record!(p.plan)
    for _ in 1:6
        GC.gc(true)
        fill!(M.storage(p.out), 0f0)
        KernelAbstractions.synchronize(M.backend(dev))
        M.run!(p.plan)
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(M.storage(p.out)) == want
    end
end

@testset "growing the arena under a recorded plan patches it" begin
    # The one way M1 and M5 interact. Plans share the device's arena, so a
    # later, larger plan grows it and remaps every tenant. A recorded tenant
    # used to make that an error: its command buffer held the addresses the old
    # allocation had, and re-materialising underneath it gave a run that read
    # freed storage, deterministically and quietly.
    #
    # Those addresses are not IN the command buffer — they are in the plan's
    # argument memory, at offsets the one pack recorded (`recpatch!`). So the
    # growth announces itself (`arena_moved!` → `notify_move!`) and the next
    # run writes the new addresses as commands in its own submission: the same
    # recording answers, with the right numbers.
    dev = M.Device(TESTBACKEND)
    small = Base.invokelatest(chainplan, dev, 100_000, 3)
    M.run!(small.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    M.record!(small.plan)
    rec = small.plan.recording

    big = Base.invokelatest(chainplan, dev, 4_000_000, 3)   # needs a much bigger arena
    M.run!(big.plan)
    KernelAbstractions.synchronize(M.backend(dev))

    M.run!(small.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    @test small.plan.recording === rec
    @test Array(M.storage(small.out)) == fill(Float32(3 + 2), 100_000)
    M.free!(small.plan)
    M.free!(big.plan)
end

@testset "a profiled plan reports both halves of a recorded run" begin
    dev = M.Device(TESTBACKEND)
    # `bake!` REFUSED a profiled plan, because `timings` measures host recording
    # per pass and a baked plan did not record — the numbers would have been the
    # capture's, reported forever. Recording is not a second mode any more: the
    # host number is what writing the pass cost, sampled once because it happens
    # once, and the GPU number comes from timestamps inside the recording and is
    # resampled on every run. Both are honest; neither is a frozen copy of the
    # other.
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, 64))
    out  = M.Buffer(dev, zeros(Float32, 64))
    M.compute!(g, "only") do p
        s = M.use(p, seed; read = true); d = M.use(p, out; write = true)
        M.dispatch!(p, bump!, (d, s), 64)
    end
    profiled = Base.invokelatest(M.Plan, g; profile = true)
    M.record!(profiled)
    for _ in 1:4
        M.run!(profiled)
    end
    KernelAbstractions.synchronize(M.backend(dev))
    t = M.timings(profiled)
    # Two passes: the update pass every plan with a declared buffer has, which
    # was never written (nothing was stored, so it has no host time and no
    # frame of GPU time), and the one that was.
    @test [x.name for x in t] == ["updates", "only"]
    @test t[2].host_ms > 0                 # the one recording
    @test t[2].samples >= 1                # …and at least one frame of GPU time
    M.free!(profiled)
end

@testset "two recorded plans alternating on one arena stay bit-exact" begin
    # Both recorded up front, so neither run emits anything: what orders b's
    # writes behind a's reads of the same bytes is the barrier each recording
    # opens with, and nothing else. The arena's record of who ran last,
    # consulted at record time, is gone; a run has nothing to claim.
    dev = M.Device(TESTBACKEND)
    a = Base.invokelatest(chainplan, dev, 50_000, 2)
    b = Base.invokelatest(chainplan, dev, 50_000, 2)
    M.record!(a.plan)
    M.record!(b.plan)
    for _ in 1:100
        M.run!(a.plan)
        M.run!(b.plan)
    end
    KernelAbstractions.synchronize(M.backend(dev))
    @test all(==(4f0), Array(M.storage(a.out)))
    @test all(==(4f0), Array(M.storage(b.out)))
    @test !isdefined(M, :takeover!)
end

@testset "an N-dimensional buffer reaches the GPU with its shape" begin
    # The Lava half of `Buffer(dev, T, dims)`: `deviceview` used to hard-code
    # `LavaArray{T,1}(…, (length(a),))`, so a 2-D region arrived at a kernel
    # flattened and every index had to be recomputed from a width the kernel had
    # to be told separately.
    dev = M.Device(TESTBACKEND)
    want = reshape(collect(1f0:12f0), 3, 4)
    b = M.Buffer(dev, want)
    @test size(b) == (3, 4)
    # Was `Mantle.LavaArray{Float32,2}`, which is one backend's array type in a
    # file that runs against all of them. What the property IS: a Buffer's
    # storage is the backend's own 2-D device array, whatever that is.
    @test M.storage(b) isa AbstractArray{Float32,2}
    @test size(M.storage(b)) == (3, 4)
    @test Array(b) == want
    M.free!(b)
end

# Slow enough that a submission is still in flight when the next line runs:
# every element spins, and the total is far past what a fence poll sees pass.
@kernel function arena_slow!(a)
    i = @index(Global)
    acc = 0f0
    for k in 1:2_000_000
        acc += sin(Float32(k) * 1f-3)
    end
    @inbounds a[i] = acc
end

@testset "a retired region waits for the device, then comes back" begin
    # The Lava half of `reclaim!`. Unlike the host, a fence here carries real
    # information: a region dropped while a batch is open may be named by a
    # command already recorded into it, so releasing on the spot would hand
    # those bytes to the next caller while the GPU is still reading them.
    #
    # So the first `reclaim!` only stamps, and the release waits for the
    # timeline. That ordering is the whole safety argument, and asserting the
    # count alone would pass just as well if it released immediately.
    dev = M.Device(TESTBACKEND)
    pool = M.pool(dev)
    b = M.Buffer(dev, fill(1f0, 4096))
    reserved = M.reserved(pool)
    # Anything this file retired earlier goes back first, so the counts below
    # are about `b` and `b3` rather than about the order of the testsets.
    while M.reclaim!(pool, dev; wait = true) > 0 end

    # Retire with a submission IN FLIGHT, which is the case that must wait: a
    # command the device is still running can name these bytes. A slow kernel
    # keeps the timeline behind the value the region is stamped with.
    scratch = KernelAbstractions.allocate(M.backend(dev), Float32, 16)
    arena_slow!(M.backend(dev), 16)(scratch; ndrange = 16)
    @test !isempty(dev.bq.outstanding)
    M.free!(b)
    @test M.reclaim!(pool, dev) == 0        # stamped, not released: it has not signalled

    # Wait, so the fence it was stamped with has passed.
    Mantle.flush!(dev.ctx.default_bq)
    KernelAbstractions.synchronize(M.backend(dev))
    @test M.reclaim!(pool, dev) == 1        # now
    @test M.reclaim!(pool, dev) == 0

    b2 = M.Buffer(dev, fill(2f0, 4096))
    @test M.reserved(pool) == reserved      # reused, not reallocated
    @test Array(b2) == fill(2f0, 4096)
    M.free!(b2)

    # And the idle case, which is the one that must NOT wait. `fence` used to
    # return `next_timeline + 1` unconditionally, so a region dropped by an
    # application that then submits nothing more waited for a signal nobody
    # would ever raise — the same leak this path removes, wearing a hat.
    KernelAbstractions.synchronize(M.backend(dev))
    @test M.idle(dev.bq)
    b3 = M.Buffer(dev, fill(3f0, 4096))
    while M.reclaim!(pool, dev; wait = true) > 0 end
    M.free!(b3)
    @test M.reclaim!(pool, dev) == 1        # nothing is in flight, so: now
end

# The device-owned arena and the baked plan. Needs a GPU, but no display: every
# graph here is headless, which is also the only kind `bake!` takes today.
using Mantle, Test, KernelAbstractions, Lava, Statistics
const M = Mantle

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
    (; g, out, seed, plan = M.Plan(g), keep = (seed, t))
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
    (; plan = M.Plan(g), keep = (seed, t1, t2))
end

@kernel function add2!(d, @Const(x), @Const(y))
    i = @index(Global)
    @inbounds d[i] = x[i] + y[i]
end

const E = Base.get_extension(Mantle, :MantleLavaExt)

@testset "two plans in one process commit the max, not the sum" begin
    # The gate for the device-owned arena. It can only pass if the `Device` owns
    # the reservation: two plans that each allocate cannot share, whatever the
    # placer does inside either of them.
    #
    # `small` is built FIRST so that `big` has to grow the pool underneath it and
    # remap it. A correct answer from `small` afterwards is `remap!` having
    # re-materialised its transients into the new allocation — which is the half
    # of this that a test built in the other order would not reach.
    dev = M.Device(Lava)
    small = Base.invokelatest(chainplan, dev, 250_000, 3)
    big   = Base.invokelatest(chainplan, dev, 2_000_000, 3)

    ps, pb = M.peakbytes(small.plan), M.peakbytes(big.plan)
    pool = only(values(dev.pools))
    @test pool.bytes == max(ps, pb)
    @test pool.bytes < ps + pb
    @test length(pool.tenants) == 2

    for _ in 1:3
        M.run!(small.plan)
        M.run!(big.plan)          # interleaved, so the handover barrier is exercised
    end
    KernelAbstractions.synchronize(M.backend(dev))
    @test all(==(5f0), Array(M.storage(small.out)))
    @test all(==(5f0), Array(M.storage(big.out)))
    @test E.handover!(small.plan, dev.bq)      # two live tenants -> a barrier
end

@testset "over budget fails at compile, with numbers" begin
    dev = M.Device(Lava)
    # `maxMemoryAllocationSize`, not the heap budget: an arena is one allocation,
    # and on this machine that limit is 4 GB against ~28 GB of budget, so it is
    # the binding one. Two of these cannot share bytes, so the arena needs both.
    n = E.maxalloc(dev) ÷ sizeof(Float32)
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
    @test !haskey(dev.pools, E.Buffers()) || dev.pools[E.Buffers()].bytes < n * sizeof(Float32)
end

@testset "a renameable Update gives up its scoped barrier" begin
    # `build_pass_barrier` bakes `st.buf[].buffer` at compile time, and `rename!`
    # points the resource at a different store — so a resource an `Update` can
    # move must get a global memory barrier instead. A *ranged* update writes in
    # place and keeps its scope, which is the discrimination being asserted.
    dev = M.Device(Lava)
    g = M.Graph(dev)
    a   = M.Buffer(dev, zeros(Float32, 1024))
    b   = M.Buffer(dev, zeros(Float32, 1024))
    out = M.Buffer(dev, zeros(Float32, 1024))
    M.Update(g, a)                   # whole buffer -> may rename
    M.Update(g, b; range = 1:10)     # ranged -> always in place
    M.compute!(g, "read") do p
        sa = M.use(p, a; read = true); sb = M.use(p, b; read = true)
        d  = M.use(p, out; write = true)
        M.dispatch!(p, add2!, (d, sa, sb), 1024)
    end
    plan = Base.invokelatest(M.Plan, g)

    @test E.renameable(g, a)
    @test !E.renameable(g, b)
    @test !E.renameable(g, out)

    pp = only(p for p in plan.passes if p.pass.name == "read")
    # One transition per resource; exactly the renameable one is widened. Counted
    # off the dependency rather than by handle: Lava suballocates, so all three
    # of these share a VkBuffer and comparing handles proves nothing.
    v = pp.barrier.vks
    @test Int(v.memoryBarrierCount) == 1
    @test Int(v.bufferMemoryBarrierCount) == 2

    # And a scoped barrier names the resource's range inside the pool buffer, not
    # offset 0 — `barrierspan` adds `pool_offset`, the same sum a copy computes.
    arr = unsafe_wrap(Array, v.pBufferMemoryBarriers, Int(v.bufferMemoryBarrierCount))
    want = UInt64(M.storage(b).buf[].pool_offset + M.storage(b).offset)
    @test any(bb -> UInt64(bb.offset) == want, arr)
    @test all(bb -> UInt64(bb.offset) != 0, arr)
end

@testset "a baked plan replays bit-exact, and records almost nothing" begin
    dev = M.Device(Lava)
    s = Base.invokelatest(chainplan, dev, 20_000, 200)     # 202 passes
    M.run!(s.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    want = copy(Array(M.storage(s.out)))
    @test !M.baked(s.plan)

    host(f, n) = (f(); [(t0 = time_ns(); f(); (time_ns() - t0) / 1e6) for _ in 1:n])
    un = host(() -> M.run!(s.plan), 30)
    KernelAbstractions.synchronize(M.backend(dev))

    M.bake!(s.plan)
    @test M.baked(s.plan)
    @test M.bake!(s.plan) === s.plan                       # idempotent

    fill!(M.storage(s.out), 0f0)
    KernelAbstractions.synchronize(M.backend(dev))
    bk = host(() -> M.run!(s.plan), 30)
    KernelAbstractions.synchronize(M.backend(dev))

    @test Array(M.storage(s.out)) == want                  # bit-exact, not merely close
    # The point of the whole exercise. Generous because it is a wall-clock median
    # on a shared machine; the measured figure is ~96%.
    @test median(sort(bk)) < median(sort(un)) / 3
end

@testset "a baked plan survives a full GC between invocations" begin
    # The property a plan has and a raw capture does not: a plan holds references
    # to every resource its recording names, so a full collection between two
    # replays cannot free storage the command buffer points at. `test_capture_gc.jl`
    # in Lava owns this for a raw capture; this is where it is owned for a plan.
    dev = M.Device(Lava)
    p = Base.invokelatest(chainplan, dev, 50_000, 8)
    M.run!(p.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    want = copy(Array(M.storage(p.out)))

    M.bake!(p.plan)
    for _ in 1:6
        GC.gc(true)
        fill!(M.storage(p.out), 0f0)
        KernelAbstractions.synchronize(M.backend(dev))
        M.run!(p.plan)
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(M.storage(p.out)) == want
    end
end

@testset "growing the arena under a baked plan is refused" begin
    # The one way M1 and M5 interact badly. Plans share the device's arena, so a
    # later, larger plan grows it and remaps every tenant — but a baked tenant
    # cannot be remapped: its recording holds the addresses the old allocation
    # had. Silently re-materialising underneath it gives a replay that reads
    # freed storage, deterministically and quietly. It has to be an error, and
    # the error has to name the order that avoids it.
    dev = M.Device(Lava)
    small = Base.invokelatest(chainplan, dev, 100_000, 3)
    M.run!(small.plan)
    KernelAbstractions.synchronize(M.backend(dev))
    M.bake!(small.plan)

    err = try
        Base.invokelatest(chainplan, dev, 4_000_000, 3)   # needs a much bigger arena
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("baked", sprint(showerror, err))
    @test occursin("before baking", sprint(showerror, err))
end

@testset "bake! refuses what it cannot record" begin
    dev = M.Device(Lava)
    # Profiling measures host recording per pass, and a baked plan does not
    # record — the numbers would be the capture's, reported forever.
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, 64))
    out  = M.Buffer(dev, zeros(Float32, 64))
    M.compute!(g, "only") do p
        s = M.use(p, seed; read = true); d = M.use(p, out; write = true)
        M.dispatch!(p, bump!, (d, s), 64)
    end
    profiled = Base.invokelatest(M.Plan, g; profile = true)
    @test_throws ArgumentError M.bake!(profiled)
end

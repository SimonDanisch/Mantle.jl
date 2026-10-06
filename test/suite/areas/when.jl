# when! (catalogue-core.md, table 6): launches in the block run only where the
# flag is nonzero on the GPU; the launches are indirect and the count is
# multiplied by the flag (by the head for a host-written flag, by a prepare
# node after a kernel-written one); no recompile or re-record for a flag change.

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: triu!

@kernel function when_inc!(a)
    i = @index(Global)
    @inbounds a[i] += one(eltype(a))
end

@kernel function when_setfirst!(x, v)
    @inbounds x[1] = v
end

@kernel function when_setindex!(a)
    i = @index(Global)
    @inbounds a[i] = i % eltype(a)
end

@kernel function when_double!(a)
    i = @index(Global)
    @inbounds a[i] *= 2
end

@kernel function when_add!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] + v
end

# Ray-tracing shaders of WHEN-11: one raygen invocation per trace adds 1.
when_raygen(hits) = (@inbounds hits[1] += one(eltype(hits)); nothing)
when_nohit(hits) = nothing

zeroed(dev, n) = (a = MantleArray(dev, Int32, n); fill!(a, Int32(0)); a)

"""A graph with one counter kernel, run once."""
function countergraph(dev)
    counter = zeroed(dev, 1)
    g = Graph(dev)
    dispatch!(g, when_inc!, (counter,), 1)
    run!(g)
    return g, counter
end

"""
K1 writes A, K2 in when! reads and writes A, K3 reads A, over 20 runs that
alternate the flag: K3 sees K2's write when the flag is 1 and K1's when it is
0, so the barrier K1 → K3 is never dropped.
"""
function checkconditionalchain(dev; n = 1 << 20)
    A, out = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    r = GPURef(dev, Int32(0))
    g = Graph(dev)
    dispatch!(g, when_setindex!, (A,), n)
    when!(g, r) do
        dispatch!(g, when_double!, (A,), n)
    end
    dispatch!(g, when_add!, (out, A, Int32(1)), n)
    for k in 1:20
        r[] = Int32(isodd(k))
        run!(g)
        @test Array(out) == (isodd(k) ? 2 : 1) .* (1:n) .+ 1
    end
end

"""A graph of `k` one-element launches on one array, plain or each in a when! block whose flag is 1."""
function tinylaunches(dev, k; conditional)
    x = zeroed(dev, 1)
    r = GPURef(dev, Int32(1))
    g = Graph(dev)
    launch() = dispatch!(g, when_inc!, (x,), 1)
    foreach(_ -> conditional ? when!(launch, g, r) : launch(), 1:k)
    return g, x
end

# ── 6. when! (WHEN) ──

# A:130, D:133-136: when!(g, GPURef(dev, Int32(1))); run; r[] = 0; run: the
# launch runs, then is skipped; the flag change compiles and records nothing.
@case "WHEN-01" 4 begin
    dev = testdevice(); n = 64
    acc = zeroed(dev, n)
    r = GPURef(dev, Int32(1))
    g = Graph(dev)
    when!(g, r) do
        dispatch!(g, when_inc!, (acc,), n)
    end
    run!(g)
    @test Array(acc) == fill(1, n)
    r[] = Int32(0)
    _, d = counted(() -> run!(g))
    @test Array(acc) == fill(1, n)
    @test d.plancompiles == 0
    @test d.records == 0
    @test d.kernelcompiles == 0
    r[] = Int32(1)
    run!(g)
    @test Array(acc) == fill(2, n)
end

# P:2288-2295: a flag in Readback memory written by a kernel of the graph: the
# prepare node after the writer applies it in the same run; Array(flag) after
# the run submits nothing (a wait and a memcpy).
@case "WHEN-02" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, Int32, 1; memory = Mantle.Readback())
    acc = zeroed(dev, n)
    r = GPURef(dev, Int32(1))
    g = Graph(dev)
    dispatch!(g, when_setfirst!, (flag, r), 1)
    when!(g, flag) do
        dispatch!(g, when_inc!, (acc,), n)
    end
    for (v, expected) in ((1, 1), (0, 1), (1, 2))
        r[] = Int32(v)
        run!(g)
        @test Array(acc) == fill(expected, n)
        f, d = counted(() -> Array(flag))
        @test f == [v]
        @test submitcount(d) == 0
    end
end

# A:130: a one-element device array flag changed by a store while a run is in
# flight: effective at the next run; neither the store nor that run waits on
# the host, and the run allocates nothing (an inline store needs no staging).
@case "WHEN-03" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, Int32[1])
    acc, slow = zeroed(dev, n), zeroed(dev, 1)
    g = Graph(dev)
    slowwrite!(g, slow, 1)
    when!(g, flag) do
        dispatch!(g, when_inc!, (acc,), n)
    end
    run!(g)
    run!(g)
    _, d = counted(() -> (flag[1:1] = Int32[0]))
    @test d.hostwaits == 0
    _, d = counted(() -> run!(g))
    @test d.hostwaits == 0
    @test d.plancompiles == 0
    @test d.allocations == 0
    @test Array(acc) == fill(2, n)
end

# A:130 vs P:2287: flags 2, -1, typemax(Int32), 0.5f0, NaN32.
# pending decision 2: nonzero means "runs once"; the GPU computes
# count × (flag ≠ 0). Two runs, two increments, whatever the value.
@case "WHEN-04" 4 begin
    dev = testdevice(); n = 64
    for v in (Int32(2), Int32(-1), typemax(Int32), 0.5f0, NaN32)
        acc = zeroed(dev, n)
        g = Graph(dev)
        when!(g, GPURef(dev, v)) do
            dispatch!(g, when_inc!, (acc,), n)
        end
        run!(g); run!(g)
        @test Array(acc) == fill(2, n)
    end
end

# I:368: a when! block inside another throws; the graph is unchanged.
@case "WHEN-05" 1 begin
    dev = testdevice(); n = 64
    acc, other = zeroed(dev, n), zeroed(dev, n)
    f1, f2 = MantleArray(dev, Int32[1]), MantleArray(dev, Int32[1])
    g, counter = countergraph(dev)
    @test_throws ArgumentError when!(g, f1) do
        dispatch!(g, when_inc!, (acc,), n)
        when!(g, f2) do
            dispatch!(g, when_inc!, (other,), n)
        end
    end
    run!(g)
    @test Array(acc) == zeros(n)
    @test Array(other) == zeros(n)
    @test Array(counter) == [2]
end

# A:478: a when! block holding render!, build!, a HostCall, an image copy or a
# repeat! throws; the graph and the holds are unchanged (the counter alone
# runs; a's memory returns after free!(a)).
@case "WHEN-06" 6 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    flag = MantleArray(dev, Int32[1])
    a = zeroed(dev, n)
    img, img2 = Mantle.Image(g, Float32, 16, 16), Mantle.Image(g, Float32, 16, 16)
    tlas = TLAS(dev)
    refused = (
        () -> (Mantle.render!(g, img => Mantle.Clear(0f0)) do p end),
        () -> build!(g, tlas),
        () -> dispatch!(g, HostCall(_ -> nothing), (a,); writes = ()),
        () -> copyto!(img2, img),
        () -> repeat!(i -> dispatch!(g, when_inc!, (a,), n), g, 2),
    )
    baseline = memory()
    for declare in refused
        @test_throws ArgumentError when!(g, flag) do
            dispatch!(g, when_inc!, (a,), n)
            declare()
        end
    end
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [2]
    @test all(iszero, Array(a))
    free!(a)
    @test memory().reserved < baseline.reserved
end

# P:103-108, I:628-637: K1 writes A; when! (flag 0 or 1) K2 reads and writes A;
# K3 reads A: K3 always sees the latest write.
@case "WHEN-07" 4 begin
    checkconditionalchain(testdevice())
end

# I:503: WHEN-07 as a CUDA graph: the edge K1 → K3 is kept in the transitive
# reduction (it would otherwise run only through the conditional K2).
@case "WHEN-08" 10 :vendorview begin
    checkconditionalchain(testdevice())
end

# P:2291-2292: a resizable flag array moved by resize! between runs: no
# recompile, no re-record; its value moved with it.
@case "WHEN-09" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, Int32[1])
    resize!(flag, 1)                                # the first resize!: resizable before g compiles
    acc = zeroed(dev, n)
    g = Graph(dev)
    when!(g, flag) do
        dispatch!(g, when_inc!, (acc,), n)
    end
    run!(g)
    resize!(flag, 1 << 20)                          # beyond its storage: a move
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
    @test Array(acc) == fill(2, n)
    flag[1:1] = Int32[0]
    run!(g)
    @test Array(acc) == fill(2, n)
end

# A:130: when!(dev, flag), the eager form. Unspecified; "graphs only" and pending
# decision 3 make a throw the plausible contract: when! has no device method,
# and the block's call does not run.
@case "WHEN-10" 1 begin
    dev = testdevice()
    a = zeroed(dev, 4)
    flag = MantleArray(dev, Int32[1])
    @test_throws MethodError when!(dev, flag) do
        fill!(a, Int32(1))
    end
    @test Array(a) == zeros(4)
end

# A:131: trace! in a when! block: the trace's count is multiplied by the flag.
@case "WHEN-11" 5 :rt begin
    dev = testdevice()
    hits = zeroed(dev, 1)
    r = GPURef(dev, Int32(1))
    rtpipe = RayTracingPipeline(; raygen = when_raygen, closest_hit = when_nohit, miss = when_nohit)
    tlas = TLAS(dev)
    g = Graph(dev)
    when!(g, r) do
        trace!(g, rtpipe, tlas, (hits,), 1)
    end
    for (v, expected) in ((1, 1), (0, 1), (1, 2))
        r[] = Int32(v)
        run!(g)
        @test Array(hits) == [expected]
    end
end

# I:369: a freed flag throws at declaration.
@case "WHEN-12" 2 begin
    dev = testdevice()
    flag = MantleArray(dev, Int32[1])
    acc = zeroed(dev, 4)
    g = Graph(dev)
    free!(flag)
    @test_throws ArgumentError when!(g, flag) do
        dispatch!(g, when_inc!, (acc,), 4)
    end
end

# P:1518, P:1556-1558: 10 000 tiny conditional launches against plain ones,
# interleaved runs: the overhead per launch is within the measured cost of an
# indirect over a direct small dependent dispatch (+194-244 ns RADV, +44-62 ns
# NVIDIA), with a factor 2 for margin. lavapipe has no measurement; there the
# case checks the results only.
@case "WHEN-13" 4 :long begin
    dev = testdevice(); k = 10_000
    plain, xp = tinylaunches(dev, k; conditional = false)
    conditional, xc = tinylaunches(dev, k; conditional = true)
    run!(plain); run!(conditional); Array(xp); Array(xc)
    tp, tc = Float64[], Float64[]
    for _ in 1:10
        push!(tp, @elapsed (run!(plain); Array(xp)))
        push!(tc, @elapsed (run!(conditional); Array(xc)))
    end
    @test Array(xp) == Array(xc) == [11k]
    bound = has(dev, :lavapipe) ? Inf : 2 * 244e-9
    @test (minimum(tc) - minimum(tp)) / k <= bound
end

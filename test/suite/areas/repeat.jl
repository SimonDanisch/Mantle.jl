# repeat! (catalogue-core.md, table 5): bodies declared once and limited to
# launches, the loop index written by the loop head, while_nonzero ending the
# loop on the GPU, hazards between iterations, the split between iterations.

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: triu!

@kernel function rep_inc!(a)
    i = @index(Global)
    @inbounds a[i] += one(eltype(a))
end

@kernel function rep_add!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] + v
end

@kernel function rep_setat!(out, i)
    @inbounds out[i] = i % eltype(out)
end

# One iteration of a while_nonzero loop: counts it, counts `left` down and
# clears the flag at 0.
@kernel function rep_countdown!(left, flag, count)
    @inbounds begin
        count[1] += one(eltype(count))
        left[1] -= one(eltype(left))
        left[1] == 0 && (flag[1] = zero(eltype(flag)))
    end
end

@kernel function rep_setflag!(flag, i)
    @inbounds flag[1] = isodd(i) % eltype(flag)
end

@kernel function rep_copyfirst!(out, i, @Const(x))
    @inbounds out[i] = x[1]
end

@kernel function rep_setfirst!(x, i)
    @inbounds x[1] = i % eltype(x)
end

@kernel function rep_addindex!(t, @Const(h), i)
    j = @index(Global)
    @inbounds t[j] = h[j] + i % eltype(t)
end

@kernel function rep_accumulate!(acc, @Const(t))
    j = @index(Global)
    @inbounds acc[j] += t[j]
end

# Ray-tracing shaders of REP-05: one raygen invocation per trace adds 1.
rep_raygen(hits) = (@inbounds hits[1] += one(eltype(hits)); nothing)
rep_nohit(hits) = nothing

# Shaders of REP-04, never compiled: the declaration throws first.
rep_vertex(a, i) = nothing
rep_fragment(a) = nothing

zeroed(dev, n) = (a = MantleArray(dev, Int32, n); fill!(a, Int32(0)); a)

"""A graph with one counter kernel, run once."""
function countergraph(dev)
    counter = zeroed(dev, 1)
    g = Graph(dev)
    dispatch!(g, rep_inc!, (counter,), 1)
    run!(g)
    return g, counter
end

# submissions(g) lists stages, each with its segments (shape open in api.md
# 2.3; the one the suite assumes).
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))

"""A graph whose repeat! body writes out[i] = i for i in 1:iterations."""
function indexloop(dev, iterations)
    out = zeroed(dev, iterations)
    g = Graph(dev)
    repeat!(g, iterations) do i
        dispatch!(g, rep_setat!, (out, i), 1)
    end
    return g, out
end

"""
The declarations refused in a repeat! or when! body: a render pass, a build,
a host call, an image copy, an insert! block, a nested repeat!. Each names `a`.
"""
function nonlaunches(g, dev, a)
    img, img2 = Mantle.Image(g, Float32, 16, 16), Mantle.Image(g, Float32, 16, 16)
    tlas = TLAS(dev)
    return (
        () -> (Mantle.render!(g, img => Mantle.Clear(0f0)) do p end),
        () -> build!(g, tlas),
        () -> dispatch!(g, HostCall(_ -> nothing), (a,); writes = ()),
        () -> copyto!(img2, img),
        () -> insert!(() -> dispatch!(g, rep_inc!, (a,), length(a)), g),
        () -> repeat!(j -> dispatch!(g, rep_inc!, (a,), length(a)), g, 2),
    )
end

# ── 5. repeat! (REP) ──

# A:129, P:365-367: repeat!(g, 50) do i; out[i] = i: out == 1:50; the body is
# recorded once (as many records for 100 iterations as for 50), and later runs
# compile and record nothing.
@case "REP-01" 1 begin
    dev = testdevice()
    g50, out50 = indexloop(dev, 50)
    g100, out100 = indexloop(dev, 100)
    _, d50 = counted(() -> run!(g50))
    _, d100 = counted(() -> run!(g100))
    @test Array(out50) == 1:50
    @test Array(out100) == 1:100
    @test d50.records == d100.records
    _, d = counted(() -> run!(g50))
    @test d.records == 0
    @test d.plancompiles == 0
end

# A:129, P:356-361: while_nonzero: run 1 ends after 7 iterations; flag and
# counter reset; run 2 ends after 3 (the loop start restores the counts the
# first run's loop head zeroed).
@case "REP-02" 1 begin
    dev = testdevice()
    left, flag, iterations = zeroed(dev, 1), zeroed(dev, 1), zeroed(dev, 1)
    g = Graph(dev)
    repeat!(g, 50; while_nonzero = flag) do i
        dispatch!(g, rep_countdown!, (left, flag, iterations), 1)
    end
    fill!(left, Int32(7)); fill!(flag, Int32(1))
    run!(g)
    @test Array(iterations) == [7]
    fill!(left, Int32(3)); fill!(flag, Int32(1))
    run!(g)
    @test Array(iterations) == [10]
end

# A:129, I:359-361: a body holding render!, build!, a HostCall, an image copy,
# insert!(g) or a nested repeat! throws at declaration; the graph and the holds
# are unchanged (the counter alone runs; a's memory returns after free!(a)).
@case "REP-03" 6 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    a = zeroed(dev, n)
    refused = nonlaunches(g, dev, a)
    baseline = memory()
    for declare in refused
        @test_throws ArgumentError repeat!(g, 3) do i
            dispatch!(g, rep_inc!, (a,), n)
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

# I:348-362 vs I:873-879: render! with a draw! over a device array in a body
# throws, and the draw's hold on the array is given back (ambiguity 12, a
# suspected leak; the case pins no leak).
@case "REP-04" 6 begin
    dev = testdevice(); n = 16 << 20
    g, _ = countergraph(dev)
    img = Mantle.Image(g, Float32, 16, 16)
    shaders = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(rep_vertex),
                                        fragment = Mantle.FragmentShader(rep_fragment))
    baseline = memory()
    a = MantleArray(dev, Float32, n)
    @test_throws ArgumentError repeat!(g, 2) do i
        Mantle.render!(g, img => Mantle.Clear(0f0)) do p
            Mantle.draw!(p, shaders, (a,), 3)
        end
    end
    free!(a)
    @test memory().reserved <= baseline.reserved
end

# A:131: trace! in a body the flag ends: the trace runs in each iteration up to
# the end and in none after it (its count zeroed by the loop head); two runs
# with 5 and 3 iterations match the host reference min(left, 20).
@case "REP-05" 5 :rt begin
    dev = testdevice()
    left, flag, iterations, hits = (zeroed(dev, 1) for _ in 1:4)
    rtpipe = RayTracingPipeline(; raygen = rep_raygen, closest_hit = rep_nohit, miss = rep_nohit)
    tlas = TLAS(dev)
    g = Graph(dev)
    repeat!(g, 20; while_nonzero = flag) do i
        dispatch!(g, rep_countdown!, (left, flag, iterations), 1)
        trace!(g, rtpipe, tlas, (hits,), 1)
    end
    for k in (5, 3)
        fill!(left, Int32(k)); fill!(flag, Int32(1)); fill!(hits, Int32(0))
        run!(g)
        @test Array(hits) == [min(k, 20)]
    end
end

# A:130: a when! block in a body whose flag a kernel writes each iteration
# (odd i): a prepare node per iteration; the launch runs only where the flag is nonzero.
@case "REP-06" 4 begin
    dev = testdevice(); iterations = 10
    flag, out = zeroed(dev, 1), zeroed(dev, iterations)
    g = Graph(dev)
    repeat!(g, iterations) do i
        dispatch!(g, rep_setflag!, (flag, i), 1)
        when!(g, flag) do
            dispatch!(g, rep_setat!, (out, i), 1)
        end
    end
    run!(g)
    @test Array(out) == [isodd(i) ? i : 0 for i in 1:iterations]
end

# I:546: the last body node writes x, the first reads it: each iteration sees
# the previous iteration's write.
@case "REP-07" 1 begin
    dev = testdevice(); iterations = 10
    x, out = zeroed(dev, 1), zeroed(dev, iterations)
    g = Graph(dev)
    repeat!(g, iterations) do i
        dispatch!(g, rep_copyfirst!, (out, i, x), 1)
        dispatch!(g, rep_setfirst!, (x, i), 1)
    end
    run!(g)
    @test Array(out) == 0:iterations-1
end

# P:2311-2315: a loop whose estimated cost (8n bytes per iteration at
# caps(dev).bandwidth) is three times today's 0.1 s budget, one iteration well
# within it: cut between iterations, more than one segment, the count exact.
@case "REP-08" 1 begin
    dev = testdevice(); n = 1 << 24
    a = zeroed(dev, n)
    iterations = ceil(Int, 0.3 * caps(dev).bandwidth / (8n))
    g = Graph(dev)
    repeat!(g, iterations) do i
        dispatch!(g, rep_inc!, (a,), n)
    end
    run!(g)
    @test segmentcount(g) > 1
    @test all(==(iterations), Array(a))
end

# I:604-605: a per-iteration transient beside a hostwritten array read in the
# body: the hostwritten bytes are not shared with the transient; exact over two runs.
@case "REP-09" 4 begin
    dev = testdevice(); n = 4096; iterations = 5
    g = Graph(dev)
    h = MantleArray(g, Int32, n; hostwritten = true)
    t = MantleArray(g, Int32, n)
    acc = zeroed(dev, n)
    repeat!(g, iterations) do i
        dispatch!(g, rep_addindex!, (t, h, i), n)
        dispatch!(g, rep_accumulate!, (acc, t), n)
    end
    expected = zeros(Int32, n)
    for seed in 1:2
        data = Int32.(testdata(Int16, n; seed))
        h[:] = data
        run!(g)
        expected .+= iterations .* data .+ Int32(sum(1:iterations))
        @test Array(acc) == expected
    end
end

# A:129: repeat!(g, 0) and repeat!(g, -1). Unspecified (ambiguity 18); the case
# pins the most plausible contract: 0 iterations run nothing, a negative count
# throws at declaration.
@case "REP-10" 1 begin
    dev = testdevice()
    a = zeroed(dev, 1)
    g = Graph(dev)
    repeat!(g, 0) do i
        dispatch!(g, rep_inc!, (a,), 1)
    end
    run!(g)
    @test Array(a) == [0]
    @test_throws ArgumentError repeat!(g, -1) do i
        dispatch!(g, rep_inc!, (a,), 1)
    end
end

# P:348-355: a body calling an operation with no graph method (GPUArrays'
# generic triu!) throws; the graph is unchanged. The plan says MethodError;
# GPUArrays' method reaches KA.get_backend first, which throws its own error,
# so any exception is accepted.
@case "REP-11" 6 begin
    dev = testdevice()
    g, counter = countergraph(dev)
    x = MantleArray(g, Float32, 8, 8)
    @test_throws Exception repeat!(g, 2) do i
        triu!(x)
    end
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [2]
end

# A:132: copyto! between graph arrays in a body is a copy kernel: allowed, exact.
@case "REP-12" 6 begin
    dev = testdevice(); n = 256; iterations = 3
    g = Graph(dev)
    t1, t2 = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    fill!(t1, Int32(0))
    repeat!(g, iterations) do i
        copyto!(t2, t1)
        dispatch!(g, rep_add!, (t1, t2, Int32(1)), n)
    end
    copyto!(out, t1)
    run!(g)
    @test Array(out) == fill(iterations, n)
end

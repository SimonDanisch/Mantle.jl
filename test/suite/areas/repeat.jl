# repeat! of a group (api.md 2.4; plan, Operations, "Groups are places in the
# order" and "Control flow is launches only"; internals.jl section 2, repeat!
# and checkloop, and section 4, LoopNode): a group runs n times at its place,
# holds launches only, the loop head writes the loop index, while_nonzero ends
# the loop on the GPU, hazards between iterations, the split between iterations.

using LinearAlgebra: triu!

function rep_inc!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds a[i] += one(eltype(a))
    return nothing
end

function rep_add!(dst, src, v)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = src[i] + v
    return nothing
end

function rep_setat!(out, i)
    KI.get_global_id().x == 1 || return nothing
    @inbounds out[i] = i % eltype(out)
    return nothing
end

# One iteration of a while_nonzero loop: counts it, counts `left` down and
# clears the count `flag` at 0.
function rep_countdown!(left, flag, count)
    KI.get_global_id().x == 1 || return nothing
    @inbounds begin
        count[1] += one(eltype(count))
        left[1] -= one(eltype(left))
        left[1] == 0 && (flag[1] = zero(eltype(flag)))
    end
    return nothing
end

function rep_setflag!(flag, i)
    KI.get_global_id().x == 1 || return nothing
    @inbounds flag[1] = isodd(i)
    return nothing
end

function rep_copyfirst!(out, i, x)
    KI.get_global_id().x == 1 || return nothing
    @inbounds out[i] = x[1]
    return nothing
end

function rep_setfirst!(x, i)
    KI.get_global_id().x == 1 || return nothing
    @inbounds x[1] = i % eltype(x)
    return nothing
end

function rep_addindex!(t, h, i)
    j = KI.get_global_id().x
    j <= length(t) || return nothing
    @inbounds t[j] = h[j] + i % eltype(t)
    return nothing
end

function rep_accumulate!(acc, t)
    j = KI.get_global_id().x
    j <= length(acc) || return nothing
    @inbounds acc[j] += t[j]
    return nothing
end

# Ray-tracing shaders of REP-05: one raygen invocation per trace adds 1.
rep_raygen(hits) = (@inbounds hits[1] += one(eltype(hits)); nothing)
rep_nohit(hits) = nothing

# Shaders of REP-03 and REP-04, never compiled: the declaration throws first.
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

"""A graph whose repeated group writes out[i] = i for i in 1:iterations."""
function indexloop(dev, iterations)
    out = zeroed(dev, iterations)
    g = Graph(dev)
    loop = Group(g, "index")
    dispatch!(loop, rep_setat!, (out, iteration(loop)), 1)
    repeat!(loop, iterations)
    return g, out
end

"""
What a repeated group refuses, each a declaration into a group `grp`:
`refused` throws `ArgumentError` there (a draw over `a`, a clear, a group: a
repeated group holds launches only); `nomethod` has no method for a `Group`
(a build, a host call naming `a`, an image copy).
"""
function nonlaunches(g, dev, a)
    img, img2 = Mantle.Image(g, Float32, 16, 16), Mantle.Image(g, Float32, 16, 16)
    shaders = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(rep_vertex),
                                        fragment = Mantle.FragmentShader(rep_fragment))
    tlas = TLAS(dev)
    refused = (
        grp -> Mantle.draw!(grp, img, shaders, (a,), 3),
        grp -> Mantle.clear!(grp, img, 0f0),
        grp -> Group(grp, "inner"),
    )
    nomethod = (
        grp -> build!(grp, tlas),
        grp -> dispatch!(grp, HostCall(_ -> nothing), (a,); writes = ()),
        grp -> copyto!(grp, img2, img),
    )
    return refused, nomethod
end

# ── 5. repeat! (REP) ──

# api.md 2.4 (repeat!, iteration): a group repeated 50 times writes
# out[i] = i: out == 1:50; the group is recorded once (as many records for 100
# iterations as for 50), and later runs compile and record nothing.
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

# api.md 2.4 (repeat!, while_nonzero); plan, "Control flow is launches only":
# while_nonzero: run 1 ends after 7 iterations; flag and counter reset; run 2
# ends after 3 (the loop start restores the counts the first run's loop head
# zeroed).
@case "REP-02" 1 begin
    dev = testdevice()
    left, flag, iterations = zeroed(dev, 1), zeroed(dev, 1), zeroed(dev, 1)
    g = Graph(dev)
    loop = Group(g, "countdown")
    dispatch!(loop, rep_countdown!, (left, flag, iterations), 1)
    repeat!(loop, 50; while_nonzero = flag)
    fill!(left, Int32(7)); fill!(flag, Int32(1))
    run!(g)
    @test Array(iterations) == [7]
    fill!(left, Int32(3)); fill!(flag, Int32(1))
    run!(g)
    @test Array(iterations) == [10]
end

# api.md 2.3-2.4 and section 7; internals.jl section 2 (checkloop, repeat!): a
# repeated group holds launches only. A draw, a clear or a group declared into
# it throws ArgumentError, when declared after repeat! and at repeat! for a
# group that already holds one; a build, a host call or an image copy has no
# method for a Group (MethodError). A refused declaration leaves the graph and
# the holds unchanged: the next run compiles nothing, the loop runs its
# launch, and a's memory returns after free!(a).
@case "REP-03" 6 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    loop = Group(g, "loop")
    dispatch!(loop, rep_inc!, (counter,), 1)
    repeat!(loop, 3)
    run!(g)                                         # counter: 1 + 1 + 3
    a = zeroed(dev, n)
    refused, nomethod = nonlaunches(g, dev, a)
    baseline = memory()
    for declare in refused
        @test_throws ArgumentError declare(loop)
    end
    for declare in nomethod
        @test_throws MethodError declare(loop)
    end
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [9]
    @test all(iszero, Array(a))
    free!(a)
    @test memory().reserved < baseline.reserved
    b = zeroed(dev, 16)
    for declare in first(nonlaunches(g, dev, b))
        held = Group(g, "held")
        declare(held)                               # allowed: held is not repeated
        @test_throws ArgumentError repeat!(held, 3)
    end
end

# internals.jl section 2 (addnode!: checkloop before name!): a draw over a
# device array declared into a repeated group throws, and takes no hold on the
# array (its memory returns after free!(a)).
@case "REP-04" 6 begin
    dev = testdevice(); n = 16 << 20
    g, _ = countergraph(dev)
    img = Mantle.Image(g, Float32, 16, 16)
    shaders = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(rep_vertex),
                                        fragment = Mantle.FragmentShader(rep_fragment))
    loop = Group(g, "loop")
    repeat!(loop, 2)
    baseline = memory()
    a = MantleArray(dev, Float32, n)
    @test_throws ArgumentError Mantle.draw!(loop, img, shaders, (a,), 3)
    free!(a)
    @test memory().reserved <= baseline.reserved
end

# api.md 2.4 (trace!): trace! in a repeated group the flag ends: the trace runs
# in each iteration up to the end and in none after it (its count zeroed by the
# loop head); two runs with 5 and 3 iterations match the host reference
# min(left, 20).
@case "REP-05" 5 :rt begin
    dev = testdevice()
    left, flag, iterations, hits = (zeroed(dev, 1) for _ in 1:4)
    rtpipe = RayTracingPipeline(; raygen = rep_raygen, closest_hit = rep_nohit, miss = rep_nohit)
    tlas = TLAS(dev)
    g = Graph(dev)
    loop = Group(g, "bounce")
    dispatch!(loop, rep_countdown!, (left, flag, iterations), 1)
    trace!(loop, rtpipe, tlas, (hits,), 1)
    repeat!(loop, 20; while_nonzero = flag)
    for k in (5, 3)
        fill!(left, Int32(k)); fill!(flag, Int32(1)); fill!(hits, Int32(0))
        run!(g)
        @test Array(hits) == [min(k, 20)]
    end
end

# api.md 2.4 (when!); plan, "Control flow is launches only": a when! launch in
# a repeated group whose Bool flag a kernel writes each iteration (odd i): a
# prepare node per iteration; the launch runs only where the flag is true.
@case "REP-06" 4 begin
    dev = testdevice(); iterations = 10
    flag = MantleArray(dev, [false])
    out = zeroed(dev, iterations)
    g = Graph(dev)
    loop = Group(g, "loop")
    i = iteration(loop)
    dispatch!(loop, rep_setflag!, (flag, i), 1)
    when!(loop, flag, rep_setat!, (out, i), 1)
    repeat!(loop, iterations)
    run!(g)
    @test Array(out) == [isodd(i) ? i : 0 for i in 1:iterations]
end

# internals.jl section 4 (compile! of a LoopNode, derivebodyhazards!): the last
# launch of the group writes x, the first reads it: each iteration sees the
# previous iteration's write.
@case "REP-07" 1 begin
    dev = testdevice(); iterations = 10
    x, out = zeroed(dev, 1), zeroed(dev, iterations)
    g = Graph(dev)
    loop = Group(g, "loop")
    i = iteration(loop)
    dispatch!(loop, rep_copyfirst!, (out, i, x), 1)
    dispatch!(loop, rep_setfirst!, (x, i), 1)
    repeat!(loop, iterations)
    run!(g)
    @test Array(out) == 0:iterations-1
end

# plan, "Submission split: estimated" (a loop whose iterations together exceed
# the submission budget is cut between iterations): a loop whose estimated
# cost (8n bytes per iteration at caps(dev).bandwidth) is three times the
# submission budget, one iteration well within it: cut between iterations,
# more than one segment, the count exact.
@case "REP-08" 1 begin
    dev = testdevice(); n = 1 << 24
    a = zeroed(dev, n)
    iterations = ceil(Int, 3 * Mantle.submissionbudget(dev) * caps(dev).bandwidth / (8n))
    g = Graph(dev)
    loop = Group(g, "loop")
    dispatch!(loop, rep_inc!, (a,), n)
    repeat!(loop, iterations)
    run!(g)
    @test segmentcount(g) > 1
    @test all(==(iterations), Array(a))
end

# api.md 2.5 (per-run host data): a per-iteration transient beside per-run host
# data, a device array h written by a store before each run and read in the
# repeated group: each run sees its store; exact over two runs.
@case "REP-09" 4 begin
    dev = testdevice(); n = 4096; iterations = 5
    g = Graph(dev)
    h = MantleArray(dev, Int32, n)                  # per-run host data: a device array written by stores
    t = MantleArray(g, Int32, n)
    acc = zeroed(dev, n)
    loop = Group(g, "loop")
    i = iteration(loop)
    dispatch!(loop, rep_addindex!, (t, h, i), n)
    dispatch!(loop, rep_accumulate!, (acc, t), n)
    repeat!(loop, iterations)
    expected = zeros(Int32, n)
    for seed in 1:2
        data = Int32.(testdata(Int16, n; seed))
        h[:] = data
        run!(g)
        expected .+= iterations .* data .+ Int32(sum(1:iterations))
        @test Array(acc) == expected
    end
end

# api.md 2.4 (repeat!): repeat!(group, 0) and repeat!(group, -1). The docs do
# not state either; the case pins the most plausible contract: 0 iterations
# run nothing, a negative count throws at repeat!.
@case "REP-10" 1 begin
    dev = testdevice()
    a = zeroed(dev, 1)
    g = Graph(dev)
    never = Group(g, "never")
    dispatch!(never, rep_inc!, (a,), 1)
    repeat!(never, 0)
    run!(g)
    @test Array(a) == [0]
    negative = Group(g, "negative")
    dispatch!(negative, rep_inc!, (a,), 1)
    @test_throws ArgumentError repeat!(negative, -1)
end

# api.md 2.3 (Group) and internals.jl section 2: an operation reaches a group
# only through a method that takes the group; GPUArrays' generic triu! has
# none, so triu!(group, x) is a MethodError and declares nothing. The graph is
# unchanged (an empty group added to a compiled graph recompiles nothing,
# internals.jl section 5, piecechange).
@case "REP-11" 6 begin
    dev = testdevice()
    g, counter = countergraph(dev)
    x = MantleArray(g, Float32, 8, 8)
    loop = Group(g, "loop")
    @test_throws MethodError triu!(loop, x)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [2]
end

# api.md 2.4 (copyto! with x on a graph): copyto! between graph arrays in a
# repeated group is a copy kernel: allowed, exact. How an operation reaches a
# group is open (internals.jl section 2); the case passes the group first, as
# examples.jl 19 does for the transformer's operations.
@case "REP-12" 6 begin
    dev = testdevice(); n = 256; iterations = 3
    g = Graph(dev)
    t1, t2 = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    fill!(t1, Int32(0))
    loop = Group(g, "loop")
    copyto!(loop, t2, t1)
    dispatch!(loop, rep_add!, (t1, t2, Int32(1)), n)
    repeat!(loop, iterations)
    copyto!(out, t1)
    run!(g)
    @test Array(out) == fill(iterations, n)
end

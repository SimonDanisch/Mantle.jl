# when! (api.md 2.4; plan, Operations, "Control flow is launches only";
# internals.jl section 2, when!): one launch per call, run only where its Bool
# flag is true on the GPU; the launch is indirect and its count is multiplied
# by the flag (by the head for a host-written flag, by a prepare node after a
# kernel-written one); no recompile or re-record for a flag change.


function when_inc!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds a[i] += one(eltype(a))
    return nothing
end

function when_setfirst!(x, v)
    KI.get_global_id().x == 1 || return nothing
    @inbounds x[1] = v
    return nothing
end

function when_setindex!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] = i % eltype(a)
    return nothing
end

function when_double!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] *= 2
    return nothing
end

function when_add!(dst, src, v)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = src[i] + v
    return nothing
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
K1 writes A, K2 as a when! launch reads and writes A, K3 reads A, over 20
runs that alternate the flag: K3 sees K2's write when the flag is true and
K1's when it is false, so the barrier K1 → K3 is never dropped.
"""
function checkconditionalchain(dev; n = 1 << 20)
    A, out = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    r = GPURef(dev, false)
    g = Graph(dev)
    dispatch!(g, when_setindex!, (A,), n)
    when!(g, r, when_double!, (A,), n)
    dispatch!(g, when_add!, (out, A, Int32(1)), n)
    for k in 1:20
        r[] = isodd(k)
        run!(g)
        @test Array(out) == (isodd(k) ? 2 : 1) .* (1:n) .+ 1
    end
end

"""A graph of `k` one-element launches on one array, plain or each a when! launch whose flag is true."""
function tinylaunches(dev, k; conditional)
    x = zeroed(dev, 1)
    r = GPURef(dev, true)
    g = Graph(dev)
    launch() = conditional ? when!(g, r, when_inc!, (x,), 1) : dispatch!(g, when_inc!, (x,), 1)
    foreach(_ -> launch(), 1:k)
    return g, x
end

# ── 6. when! (WHEN) ──

# api.md 2.4 (when!), 2.5 (GPURef); examples.jl 22: when!(g, GPURef(dev, true),
# …); run; r[] = false; run: the launch runs, then is skipped; the flag change
# compiles and records nothing.
@case "WHEN-01" 4 begin
    dev = testdevice(); n = 64
    acc = zeroed(dev, n)
    r = GPURef(dev, true)
    g = Graph(dev)
    when!(g, r, when_inc!, (acc,), n)
    run!(g)
    @test Array(acc) == fill(1, n)
    r[] = false
    _, d = counted(() -> run!(g))
    @test Array(acc) == fill(1, n)
    @test d.plancompiles == 0
    @test d.records == 0
    @test d.kernelcompiles == 0
    r[] = true
    run!(g)
    @test Array(acc) == fill(2, n)
end

# api.md 2.2 (unified = true); plan, "Control flow is launches only": a flag in
# unified memory written by a kernel of the graph: the prepare node after the
# writer applies it in the same run; Array(flag) after the run submits nothing
# (a wait and a memcpy).
@case "WHEN-02" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, Bool, 1; unified = true)
    acc = zeroed(dev, n)
    r = GPURef(dev, true)
    g = Graph(dev)
    dispatch!(g, when_setfirst!, (flag, r), 1)
    when!(g, flag, when_inc!, (acc,), n)
    for (v, expected) in ((true, 1), (false, 1), (true, 2))
        r[] = v
        run!(g)
        @test Array(acc) == fill(expected, n)
        f, d = counted(() -> Array(flag))
        @test f == [v]
        @test submitcount(d) == 0
    end
end

# api.md 2.2 (a[r] = data), 2.4 (when!): a one-element Bool device array flag
# changed by a store while a run is in flight: effective at the next run;
# neither the store nor that run waits on the host, and the run allocates
# nothing (an inline store needs no staging).
@case "WHEN-03" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, [true])
    acc, slow = zeroed(dev, n), zeroed(dev, 1)
    g = Graph(dev)
    slowwrite!(g, slow, 1)
    when!(g, flag, when_inc!, (acc,), n)
    run!(g)
    run!(g)
    _, d = counted(() -> (flag[1:1] = [false]))
    @test d.hostwaits == 0
    _, d = counted(() -> run!(g))
    @test d.hostwaits == 0
    @test d.plancompiles == 0
    @test d.allocations == 0
    @test Array(acc) == fill(2, n)
end

# api.md 2.4 (when!) and section 7; internals.jl section 2: the flag is a
# GPURef{Bool} or a one-element Bool array by the method's argument types; a
# GPURef of 2, -1, typemax(Int32), 0.5f0 or NaN32 is a MethodError, nothing is
# declared, and two runs leave acc untouched.
@case "WHEN-04" 4 begin
    dev = testdevice(); n = 64
    for v in (Int32(2), Int32(-1), typemax(Int32), 0.5f0, NaN32)
        acc = zeroed(dev, n)
        g = Graph(dev)
        @test_throws MethodError when!(g, GPURef(dev, v), when_inc!, (acc,), n)
        run!(g); run!(g)
        @test Array(acc) == zeros(n)
    end
end

# api.md 2.4 (HostCall, when!) and section 7: when! takes one launch or one
# draw; a host call under a flag is a MethodError. The graph and the holds are
# unchanged (the counter alone runs; a's memory returns after free!(a)).
@case "WHEN-06" 6 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    flag = MantleArray(dev, [true])
    a = zeroed(dev, n)
    baseline = memory()
    @test_throws MethodError when!(g, flag, HostCall(_ -> nothing), (a,); writes = ())
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [2]
    @test all(iszero, Array(a))
    free!(a)
    @test memory().reserved < baseline.reserved
end

# plan, "Control flow is launches only"; internals.jl section 4 (barriers!:
# merged over conditional launches): K1 writes A; a when! launch (flag false or
# true) K2 reads and writes A; K3 reads A: K3 always sees the latest write.
@case "WHEN-07" 4 begin
    checkconditionalchain(testdevice())
end

# plan, "Four device types, one model" (repeat! / conditional kernels): WHEN-07
# as a CUDA graph: the edge K1 → K3 is kept in the transitive reduction (it
# would otherwise run only through the conditional K2).
@case "WHEN-08" 10 :vendorview begin
    checkconditionalchain(testdevice())
end

# plan, "Control flow is launches only" (the flag is never a handle
# dependence): a resizable flag array moved by resize! between runs: no
# recompile, no re-record; its value moved with it.
@case "WHEN-09" 4 begin
    dev = testdevice(); n = 64
    flag = MantleArray(dev, [true])
    resize!(flag, 1)                                # the first resize!: resizable before g compiles
    acc = zeroed(dev, n)
    g = Graph(dev)
    when!(g, flag, when_inc!, (acc,), n)
    run!(g)
    resize!(flag, 1 << 20)                          # beyond its storage: a move
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
    @test Array(acc) == fill(2, n)
    flag[1:1] = [false]
    run!(g)
    @test Array(acc) == fill(2, n)
end

# plan, "Control flow is launches only" ("Conditions exist in graphs only");
# internals.jl section 11: when! takes a graph or a group, so when!(dev, flag,
# …) is a MethodError and the launch does not run.
@case "WHEN-10" 1 begin
    dev = testdevice()
    a = zeroed(dev, 4)
    flag = MantleArray(dev, [true])
    @test_throws MethodError when!(dev, flag, when_inc!, (a,), 4)
    @test Array(a) == zeros(4)
end

# api.md 2.4 (trace!): a trace as a when! launch: the trace's count is
# multiplied by the flag. The docs name trace! a when! launch without spelling
# the call; the case puts the flag after the graph, as for a kernel launch.
@case "WHEN-11" 5 :rt begin
    dev = testdevice()
    hits = zeroed(dev, 1)
    r = GPURef(dev, true)
    rtpipe = RayTracingPipeline(; raygen = when_raygen, closest_hit = when_nohit, miss = when_nohit)
    tlas = TLAS(dev)
    g = Graph(dev)
    when!(g, r, rtpipe, tlas, (hits,), 1)
    for (v, expected) in ((true, 1), (false, 1), (true, 2))
        r[] = v
        run!(g)
        @test Array(hits) == [expected]
    end
end

# internals.jl section 2 (declare!: usable on the flag and the arguments): a
# freed flag throws at declaration.
@case "WHEN-12" 2 begin
    dev = testdevice()
    flag = MantleArray(dev, [true])
    acc = zeroed(dev, 4)
    g = Graph(dev)
    free!(flag)
    @test_throws ArgumentError when!(g, flag, when_inc!, (acc,), 4)
end

# plan, "Control flow is launches only" (indirect launches): 10 000 tiny
# conditional launches against plain ones, interleaved runs: the overhead per
# launch is within the measured cost of an indirect over a direct small
# dependent dispatch (+194-244 ns RADV, +44-62 ns NVIDIA), with a factor 2 for
# margin. lavapipe has no measurement; there the case checks the results only.
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

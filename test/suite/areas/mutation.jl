# Graph mutation after compile (api.md 2.3; plan, "The graph and its plan";
# internals.jl section 2, Group and delete!, and section 5, applychanges!):
# declaring into a compiled graph and deleting a group of launches only mark
# the plan; the next run! recompiles; device arrays keep their contents; holds
# are counted per node.


function mut_inc!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds a[i] += one(eltype(a))
    return nothing
end

function mut_add!(dst, src, v)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = src[i] + v
    return nothing
end

function mut_fillfrom!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = src[1] % eltype(dst)
    return nothing
end

function mut_setindex!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] = i % eltype(a)
    return nothing
end

zeroed(dev, n) = (a = MantleArray(dev, Int32, n); fill!(a, Int32(0)); a)

"""A graph with one counter kernel outside any group, run once."""
function countergraph(dev)
    counter = zeroed(dev, 1)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (counter,), 1)
    run!(g)
    return g, counter
end

# ── 3. Graph mutation (MUT) ──

# api.md 2.3 (a graph is never replaced for a structure change): run!, declare
# a kernel, run!: one recompile, the device array the first kernel accumulates
# into keeps its contents. That the graph's windows and swapchains stay is a
# window case and not part of this one.
@case "MUT-01" 1 begin
    dev = testdevice(); n = 1024
    acc, other = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (acc,), n)
    run!(g)
    dispatch!(g, mut_inc!, (other,), n)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(acc) == fill(2, n)
    @test Array(other) == fill(1, n)
end

# api.md 2.3: five declarations between two runs: exactly one recompile.
@case "MUT-02" 1 begin
    dev = testdevice(); n = 256
    g, _ = countergraph(dev)
    arrs = [zeroed(dev, n) for _ in 1:5]
    foreach(a -> dispatch!(g, mut_inc!, (a,), n), arrs)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test all(a -> Array(a) == fill(1, n), arrs)
end

# api.md 2.3 (Group, delete!): h = Group(g, …) holding K1 and K2; runs;
# delete!(g, h); runs: K1 and K2 run, then stop; one recompile per change (a
# group of launches marks the plan).
@case "MUT-03" 1 begin
    dev = testdevice(); n = 256
    a, b = zeroed(dev, n), zeroed(dev, n)
    g, counter = countergraph(dev)
    h = Group(g, "block")
    dispatch!(h, mut_inc!, (a,), n)
    dispatch!(h, mut_inc!, (b,), n)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    run!(g)
    delete!(g, h)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    run!(g)
    @test Array(a) == fill(2, n)
    @test Array(b) == fill(2, n)
    @test Array(counter) == [5]
end

# api.md 2.3 (delete!: a resource loses the graph's hold once the plan that
# names it has retired); internals.jl section 2 (unname!): A named only by
# group h; free!(A) and delete!(g, h) while the old plan's run (a slow kernel,
# then a write of A) is in flight: A's storage is released only after that
# run's token, so a canary of A's size, allocated right after the recompile,
# is not written by it.
@case "MUT-04" 2 begin
    dev = testdevice(); n = 1 << 22
    slow = MantleArray(dev, Int32, 1)
    A = MantleArray(dev, UInt32, n)
    g, _ = countergraph(dev)
    h = Group(g, "block")
    slowwrite!(h, slow, 1)
    dispatch!(h, mut_fillfrom!, (A, slow), n)       # after the slow kernel
    run!(g)
    free!(A)
    delete!(g, h)
    run!(g)
    c = canary(dev, n)
    @test Array(slow) == [1]
    @test intact(c)
end

# internals.jl section 10 (name!: one hold per graph, counted per node): A
# named by group h and by a node outside it; free!(A); delete!(g, h): the
# graph keeps its hold and the outside node still reads A.
@case "MUT-05" 2 begin
    dev = testdevice(); n = 1 << 20
    A = MantleArray(dev, Int32.(1:n))
    out, other = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_add!, (out, A, Int32(1)), n)
    h = Group(g, "block")
    dispatch!(h, mut_add!, (other, A, Int32(2)), n)
    run!(g)
    free!(A)
    delete!(g, h)
    run!(g)
    c = canary(dev, n)
    run!(g)
    @test Array(out) == (1:n) .+ 1
    @test intact(c)
end

# api.md 2.3 (Group, delete!): K1 declared into group h, then the caller's code
# throws: h keeps K1 (a group is a value; nothing undoes a declaration).
# delete!(g, h) before the next run: K1 never runs and A's hold is given back
# (A's memory returns after free!(A)).
@case "MUT-06" 2 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    baseline = memory()
    A = MantleArray(dev, Int32, n)
    fill!(A, Int32(5))
    h = Group(g, "block")
    @test_throws ErrorException begin
        dispatch!(h, mut_inc!, (A,), n)
        error("thrown between declarations")
    end
    delete!(g, h)
    run!(g)
    @test all(==(5), Array(A))
    @test Array(counter) == [2]
    free!(A)
    @test memory().reserved <= baseline.reserved
end

# api.md 2.3-2.4 and section 7; internals.jl section 2 (Group, checkloop): a
# group made in a group is allowed and runs at its place; a group made in a
# repeated group throws ArgumentError (loops do not nest) and leaves that
# group unchanged.
@case "MUT-07" 1 begin
    dev = testdevice(); n = 16
    a, b = zeroed(dev, n), zeroed(dev, n)
    g, counter = countergraph(dev)
    outer = Group(g, "outer")
    inner = Group(outer, "inner")
    dispatch!(inner, mut_inc!, (a,), n)
    loop = Group(g, "loop")
    dispatch!(loop, mut_inc!, (b,), n)
    repeat!(loop, 2)
    @test_throws ArgumentError Group(loop, "inner")
    run!(g)
    @test Array(a) == fill(1, n)
    @test Array(b) == fill(2, n)
    @test Array(counter) == [2]
end

# internals.jl section 2 (delete!, Group.live): delete!(g, h) twice: the second
# throws ArgumentError, and the holds do not underflow: A, held by its
# creator, stays readable after a sweep.
@case "MUT-08" 1 begin
    dev = testdevice(); n = 64
    A = MantleArray(dev, Int32.(1:n))
    other = zeroed(dev, n)
    g, _ = countergraph(dev)
    h = Group(g, "block")
    dispatch!(h, mut_add!, (other, A, Int32(0)), n)
    run!(g)
    delete!(g, h)
    run!(g)
    @test_throws ArgumentError delete!(g, h)
    run!(g)
    memory()
    @test Array(A) == 1:n
end

# internals.jl section 2 (delete!): delete!(g2, h) with h a group of g1
# throws and changes no hold: h still deletes from g1, and A stays readable.
@case "MUT-09" 1 begin
    dev = testdevice(); n = 64
    A = MantleArray(dev, Int32.(1:n))
    other = zeroed(dev, n)
    g1, _ = countergraph(dev)
    g2, _ = countergraph(dev)
    h = Group(g1, "block")
    dispatch!(h, mut_add!, (other, A, Int32(1)), n)
    run!(g1)
    @test_throws ArgumentError delete!(g2, h)
    run!(g1)
    @test Array(other) == (1:n) .+ 1
    delete!(g1, h)
    run!(g1); run!(g2)
    memory()
    @test Array(A) == 1:n
end

# api.md 2.3 (delete!: nodes declared into the graph itself stay): a node
# outside any group, then 50 cycles of a group made and deleted with a run
# after each change: the outside node runs every time.
@case "MUT-10" 1 begin
    dev = testdevice(); n = 64
    always, extra = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (always,), n)
    for _ in 1:50
        h = Group(g, "block")
        dispatch!(h, mut_inc!, (extra,), n)
        run!(g)
        delete!(g, h)
        run!(g)
    end
    @test Array(always) == fill(100, n)
    @test Array(extra) == fill(50, n)
end

# api.md 2.3 (declaring takes the graph lock, so it waits while a run of g has
# stages left): T1 runs g, held in its host function; T2 declares into g: T2
# waits for T1's last stage; the next run recompiles.
@case "MUT-11" 3 begin
    dev = testdevice(); n = 64
    a, b = zeroed(dev, n), zeroed(dev, n)
    gate = Gate()
    g = Graph(dev)
    dispatch!(g, mut_inc!, (a,), n)
    dispatch!(g, blockinghost(gate), (a,); writes = ())
    runner = Threads.@spawn run!(g)
    waitentered(gate)
    declarer = Threads.@spawn dispatch!(g, mut_inc!, (b,), n)
    sleep(0.5)
    @test !istaskdone(declarer)
    letgo!(gate)
    withtimeout(() -> (wait(runner); wait(declarer)), 30)
    letgo!(gate)                                    # for the next run's host function
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(b) == fill(1, n)
    @test Array(a) == fill(2, n)
end

# api.md 2.3 (declaring from a host function of g throws); internals.jl
# section 2 (declaring, lockgraph): g's host function declares into g: run!
# throws, g is unchanged (the next run compiles nothing and the kernel never
# runs).
@case "MUT-12" 3 begin
    dev = testdevice(); n = 64
    a, b = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    tried = Ref(false)
    dispatch!(g, mut_inc!, (a,), n)
    dispatch!(g, HostCall(_ -> (tried[] || (tried[] = true; dispatch!(g, mut_inc!, (b,), n)); nothing)), (); writes = ())
    @test_throws ArgumentError run!(g)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(b) == zeros(n)
end

# api.md 2.3 (what must survive runs and recompiles is a device array the
# graph names): a device-array film accumulates over 100 runs with a group
# made or deleted every 10 runs: the sum is exact.
@case "MUT-13" 1 begin
    dev = testdevice(); n = 1024
    film, scratch = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (film,), n)
    h = nothing
    for k in 1:100
        if k % 10 == 0
            if h === nothing
                h = Group(g, "block")
                t = MantleArray(g, Int32, n)                # changes the placement too
                dispatch!(h, mut_add!, (t, film, Int32(1)), n)
                dispatch!(h, mut_add!, (scratch, t, Int32(0)), n)
            else
                delete!(g, h)
                h = nothing
            end
        end
        run!(g)
    end
    @test Array(film) == fill(100, n)
end

# api.md 2.2 (graph arrays are never persistent) and 2.3: `persistent` is an
# unknown keyword; state that outlives runs and recompiles is a device array,
# which keeps its count across a recompile.
@case "MUT-14" 1 begin
    dev = testdevice()
    g = Graph(dev)
    @test_throws MethodError MantleArray(g, Int32, 1; persistent = true)
    counter = zeroed(dev, 1)
    dispatch!(g, mut_inc!, (counter,), 1)
    run!(g)
    dispatch!(g, mut_inc!, (zeroed(dev, 1),), 1)
    run!(g)
    @test Array(counter) == [2]
end

# api.md section 5 (the recompile flag: set by a failed compile); internals.jl
# section 6 (recompile!): a group whose compile fails (FaultDevice throws from
# the next two compilekernel calls): both runs throw (the recompile mark
# stays); after delete!(g, h) g runs with its old structure; the failed plans
# leave no memory behind.
@case "MUT-15" 2 begin
    fd = FaultDevice(testdevice()); n = 64
    A = zeroed(fd, n)
    g, counter = countergraph(fd)
    baseline = memory(fd)
    k = calls(fd, :compilekernel)
    inject!(fd, :compilekernel, Throw(k + 1, ErrorException("injected compile failure")))
    inject!(fd, :compilekernel, Throw(k + 2, ErrorException("injected compile failure")))
    h = Group(g, "block")
    dispatch!(h, mut_inc!, (A,), n)
    @test_throws ErrorException run!(g)
    @test_throws ErrorException run!(g)
    delete!(g, h)
    run!(g)
    @test Array(counter) == [2]
    @test Array(A) == zeros(n)
    @test memory(fd).reserved <= baseline.reserved
end

# internals.jl section 6 (recompile!, discard!); resizing-and-raytracing.jl B.6
# (register!): T2 makes the first resize! of an array g launches over directly
# while T1's compile is held (FaultDevice blocks the next compilekernel call):
# register! returns Again, the plan is compiled again, the output covers the
# new length, and the discarded plan leaves no memory behind.
@case "MUT-16" 4 begin
    fd = FaultDevice(testdevice()); n = 1024
    baseline = memory(fd)
    a = MantleArray(fd, Int32, n)                   # fixed: launched over directly
    g = Graph(fd)
    dispatch!(g, mut_setindex!, (a,), launchrange(a))
    held = Channel{Nothing}(1)
    k = calls(fd, :compilekernel)
    inject!(fd, :compilekernel, Block(k + 1, held))
    runner = Threads.@spawn counted(() -> run!(g), fd)
    @test timedwait(() -> calls(fd, :compilekernel) > k, 30) === :ok
    resize!(a, 2n)
    put!(held, nothing)
    _, d = withtimeout(() -> fetch(runner), 60)
    @test d.plancompiles == 2
    @test Array(a) == 1:2n
    free!(g); free!(a)
    @test memory(fd).reserved <= baseline.reserved
end

# api.md 2.3 (the next run! compiles a new plan and retires the old one after
# its last run); internals.jl section 6 (retireplan!): a recompile while the
# old plan's run is in flight (a slow kernel, then kernels reading their
# arguments): the old argument block and recordings are released only after
# that run, so both runs' outputs are right; after free!(g) the memory
# returns.
@case "MUT-17" 2 begin
    dev = testdevice(); n = 1 << 20
    slow = MantleArray(dev, Int32, 1)
    out1, out2 = zeroed(dev, n), zeroed(dev, n)
    calibratedspin(dev, 200)
    baseline = memory()
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    slowwrite!(g, slow, 3)
    dispatch!(g, mut_fillfrom!, (t, slow), n)
    dispatch!(g, mut_add!, (out1, t, Int32(1)), n)
    run!(g)
    dispatch!(g, mut_add!, (out2, t, Int32(2)), n)
    run!(g)
    @test Array(out1) == fill(4, n)
    @test Array(out2) == fill(5, n)
    free!(g)
    @test memory().reserved <= baseline.reserved
end

# api.md 2.3 (run!: a run's first segment waits on the GPU for the previous
# run of g) and section 6: a read-modify-write counter in a device array; a
# group made or deleted before each of 1000 runs, no host waits: every run
# counts once, so a new plan's first run is ordered after the old plan's last
# run.
@case "MUT-18" 3 begin
    dev = testdevice()
    counter, scratch = zeroed(dev, 1), zeroed(dev, 64)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (counter,), 1)
    h = nothing
    for _ in 1:1000
        if h === nothing
            h = Group(g, "block")
            dispatch!(h, mut_inc!, (scratch,), 64)
        else
            delete!(g, h)
            h = nothing
        end
        run!(g)
    end
    @test Array(counter) == [1000]
end

# internals.jl section 2 (unname!) and section 10 (name!): 10 000 cycles of a
# group made and deleted, naming a device array, a transient and a value
# GPURef: memory figures back at their baseline after free!(A) (holds,
# namers, the graph's named set and the GPURef's plans do not grow).
@case "MUT-19" 2 :long begin
    dev = testdevice(); n = 1 << 16
    g, counter = countergraph(dev)
    baseline = memory()
    A = zeroed(dev, n)
    r = GPURef(dev, Int32(1))
    for _ in 1:10_000
        h = Group(g, "block")
        t = MantleArray(g, Int32, n)
        dispatch!(h, mut_add!, (t, A, r), n)
        dispatch!(h, mut_add!, (A, t, Int32(0)), n)
        run!(g)
        delete!(g, h)
        run!(g)
    end
    @test Array(A) == fill(10_000, n)
    @test Array(counter) == [20_001]
    free!(A)
    @test all(Tuple(memory()) .<= Tuple(baseline))
end

# api.md 2.3 (declaring takes the graph lock); internals.jl section 2
# (declaring): two tasks declare into their own groups of one graph at once:
# each declaration is serialized by the graph lock, and each group holds only
# its own nodes.
@case "MUT-20" 1 begin
    dev = testdevice(); n = 64
    arrs = [zeroed(dev, n) for _ in 1:4]
    g = Graph(dev)
    handles = Vector{Any}(undef, 2)
    tasks = map(1:2) do k
        Threads.@spawn begin
            h = handles[k] = Group(g, "task $k")
            dispatch!(h, mut_inc!, (arrs[2k-1],), n)
            sleep(0.2)                              # the other task declares meanwhile
            dispatch!(h, mut_inc!, (arrs[2k],), n)
        end
    end
    withtimeout(() -> foreach(wait, tasks), 60)
    run!(g)
    delete!(g, handles[1])
    run!(g)
    @test Array.(arrs) == [fill(1, n), fill(1, n), fill(2, n), fill(2, n)]
end

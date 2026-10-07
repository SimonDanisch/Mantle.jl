# Graph mutation after compile, decisions B3.16 (catalogue-core.md, table 3):
# declaring into a compiled graph and delete!(g, h) only mark the plan; the
# next run! recompiles; device arrays keep their contents; holds are counted
# per top-level node.

using KernelAbstractions: @kernel, @index, @Const

@kernel function mut_inc!(a)
    i = @index(Global)
    @inbounds a[i] += one(eltype(a))
end

@kernel function mut_add!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] + v
end

@kernel function mut_fillfrom!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[1] % eltype(dst)
end

@kernel function mut_setindex!(a)
    i = @index(Global)
    @inbounds a[i] = i % eltype(a)
end

zeroed(dev, n) = (a = MantleArray(dev, Int32, n); fill!(a, Int32(0)); a)

"""A graph with one counter kernel outside any block, run once."""
function countergraph(dev)
    counter = zeroed(dev, 1)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (counter,), 1)
    run!(g)
    return g, counter
end

# ── 3. Graph mutation, B3.16 (MUT) ──

# A:107-115, D:244-254: run!, declare a kernel, run!: one recompile, the device
# array the first kernel accumulates into keeps its contents. The window part
# of the row (window and swapchain kept) is a window case and not part of this one.
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

# A:108-110: five declarations between two runs: exactly one recompile.
@case "MUT-02" 1 begin
    dev = testdevice(); n = 256
    g, _ = countergraph(dev)
    arrs = [zeroed(dev, n) for _ in 1:5]
    foreach(a -> dispatch!(g, mut_inc!, (a,), n), arrs)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test all(a -> Array(a) == fill(1, n), arrs)
end

# A:119-120: h = insert!(g) do K1; K2 end; runs; delete!(g, h); runs: K1 and K2
# run, then stop; one recompile per change.
@case "MUT-03" 1 begin
    dev = testdevice(); n = 256
    a, b = zeroed(dev, n), zeroed(dev, n)
    g, counter = countergraph(dev)
    h = insert!(g) do
        dispatch!(g, mut_inc!, (a,), n)
        dispatch!(g, mut_inc!, (b,), n)
    end
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

# A:120, I:279-296: A named only by block h; free!(A) and delete!(g, h) while the
# old plan's run (a slow kernel, then a write of A) is in flight: A's storage is
# released only after that run's token, so a canary of A's size, allocated
# right after the recompile, is not written by it.
@case "MUT-04" 2 begin
    dev = testdevice(); n = 1 << 22
    slow = MantleArray(dev, Int32, 1)
    A = MantleArray(dev, UInt32, n)
    g, _ = countergraph(dev)
    h = insert!(g) do
        slowwrite!(g, slow, 1)
        dispatch!(g, mut_fillfrom!, (A, slow), n)  # after the slow kernel
    end
    run!(g)
    free!(A)
    delete!(g, h)
    run!(g)
    c = canary(dev, n)
    @test Array(slow) == [1]
    @test intact(c)
end

# I:2068-2074: A named by block h and by a node outside it; free!(A); delete!(g, h):
# the graph keeps its hold (counted per node) and the outside node still reads A.
@case "MUT-05" 2 begin
    dev = testdevice(); n = 1 << 20
    A = MantleArray(dev, Int32.(1:n))
    out, other = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_add!, (out, A, Int32(1)), n)
    h = insert!(g) do
        dispatch!(g, mut_add!, (other, A, Int32(2)), n)
    end
    run!(g)
    free!(A)
    delete!(g, h)
    run!(g)
    c = canary(dev, n)
    run!(g)
    @test Array(out) == (1:n) .+ 1
    @test intact(c)
end

# A:119, I:268-274: insert!(g) do K1; error() end throws; K1 is gone and A's hold
# is given back (A's memory returns after free!(A)).
@case "MUT-06" 2 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    baseline = memory()
    A = MantleArray(dev, Int32, n)
    fill!(A, Int32(5))
    @test_throws ErrorException insert!(g) do
        dispatch!(g, mut_inc!, (A,), n)
        error("thrown inside the block")
    end
    run!(g)
    @test all(==(5), Array(A))
    @test Array(counter) == [2]
    free!(A)
    @test memory().reserved <= baseline.reserved
end

# A:119, I:263,265: insert!(g) nested, inside repeat!, inside when!: each
# throws and the graph is unchanged.
@case "MUT-07" 1 begin
    dev = testdevice(); n = 16
    a = zeroed(dev, n)
    flag = MantleArray(dev, Int32[1])
    g, counter = countergraph(dev)
    block() = insert!(() -> dispatch!(g, mut_inc!, (a,), n), g)
    @test_throws ArgumentError insert!(g) do
        dispatch!(g, mut_inc!, (a,), n)
        block()
    end
    @test_throws ArgumentError repeat!(g, 2) do i
        block()
    end
    @test_throws ArgumentError when!(g, flag) do
        block()
    end
    run!(g)
    @test Array(a) == zeros(n)
    @test Array(counter) == [2]
end

# A:120: delete!(g, h) twice. Ambiguity 11: internals.jl (NodeBlock.live) throws
# on the second delete!, the catalogue reads a no-op; the case follows
# internals.jl. Either way the holds do not underflow: A, held by its creator,
# stays readable after a sweep.
@case "MUT-08" 1 begin
    dev = testdevice(); n = 64
    A = MantleArray(dev, Int32.(1:n))
    other = zeroed(dev, n)
    g, _ = countergraph(dev)
    h = insert!(g) do
        dispatch!(g, mut_add!, (other, A, Int32(0)), n)
    end
    run!(g)
    delete!(g, h)
    run!(g)
    @test_throws ArgumentError delete!(g, h)
    run!(g)
    memory()
    @test Array(A) == 1:n
end

# A:120: delete!(g2, h) with h from g1 throws and changes no hold: h still
# deletes from g1, and A stays readable.
@case "MUT-09" 1 begin
    dev = testdevice(); n = 64
    A = MantleArray(dev, Int32.(1:n))
    other = zeroed(dev, n)
    g1, _ = countergraph(dev)
    g2, _ = countergraph(dev)
    h = insert!(g1) do
        dispatch!(g1, mut_add!, (other, A, Int32(1)), n)
    end
    run!(g1)
    @test_throws ArgumentError delete!(g2, h)
    run!(g1)
    @test Array(other) == (1:n) .+ 1
    delete!(g1, h)
    run!(g1); run!(g2)
    memory()
    @test Array(A) == 1:n
end

# A:120: a node outside any block, then 50 insert/delete cycles with a run after
# each change: the outside node runs every time.
@case "MUT-10" 1 begin
    dev = testdevice(); n = 64
    always, extra = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (always,), n)
    for _ in 1:50
        h = insert!(g) do
            dispatch!(g, mut_inc!, (extra,), n)
        end
        run!(g)
        delete!(g, h)
        run!(g)
    end
    @test Array(always) == fill(100, n)
    @test Array(extra) == fill(50, n)
end

# A:114: T1 runs g, held in its host function; T2 declares into g: T2 waits for
# T1's last stage; the next run recompiles.
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

# A:114-115, I:238: g's host function declares into g: run! throws, g is
# unchanged (the next run compiles nothing and the kernel never runs).
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

# A:112-113: a device-array film accumulates over 100 runs with an insert! or
# delete! every 10 runs: the sum is exact.
@case "MUT-13" 1 begin
    dev = testdevice(); n = 1024
    film, scratch = zeroed(dev, n), zeroed(dev, n)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (film,), n)
    h = nothing
    for k in 1:100
        if k % 10 == 0
            if h === nothing
                h = insert!(g) do
                    t = MantleArray(g, Int32, n)            # changes the placement too
                    dispatch!(g, mut_add!, (t, film, Int32(1)), n)
                    dispatch!(g, mut_add!, (scratch, t, Int32(0)), n)
                end
            else
                delete!(g, h)
                h = nothing
            end
        end
        run!(g)
    end
    @test Array(film) == fill(100, n)
end

# A:71 vs A:112-113: a persistent graph array across a recompile.
# pending decision 1: `persistent` is dropped (an unknown keyword); state that
# outlives runs and recompiles is a device array, which keeps its count.
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

# I:1028-1058: a block whose compile fails [FI-1, the next two compilekernel
# calls throw]: both runs throw (the recompile mark stays); after delete!(g, h)
# g runs with its old structure; the failed plans leave no memory behind.
@case "MUT-15" 2 begin
    fd = FaultDevice(testdevice()); n = 64
    A = zeroed(fd, n)
    g, counter = countergraph(fd)
    baseline = memory(fd)
    k = calls(fd, :compilekernel)
    inject!(fd, :compilekernel, Throw(k + 1, ErrorException("injected compile failure")))
    inject!(fd, :compilekernel, Throw(k + 2, ErrorException("injected compile failure")))
    h = insert!(g) do
        dispatch!(g, mut_inc!, (A,), n)
    end
    @test_throws ErrorException run!(g)
    @test_throws ErrorException run!(g)
    delete!(g, h)
    run!(g)
    @test Array(counter) == [2]
    @test Array(A) == zeros(n)
    @test memory(fd).reserved <= baseline.reserved
end

# I:1038-1041, R:672-692: T2 makes the first resize! of an array g launches over
# directly while T1's compile is held [FI-7]: register! returns Again, the plan
# is compiled again, the output covers the new length, and the discarded plan
# leaves no memory behind.
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

# I:1059-1065: a recompile while the old plan's run is in flight (a slow kernel,
# then kernels reading their arguments): the old argument block and recordings
# are released only after that run, so both runs' outputs are right; after
# free!(g) the memory returns.
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

# P:72-73, I:1216, 1348: a read-modify-write counter in a device array; an
# insert! or delete! before each of 1000 runs, no host waits: every run counts
# once, so a new plan's first run is ordered after the old plan's last run
# (ambiguity 5: implied through the shared resource, not stated).
@case "MUT-18" 3 begin
    dev = testdevice()
    counter, scratch = zeroed(dev, 1), zeroed(dev, 64)
    g = Graph(dev)
    dispatch!(g, mut_inc!, (counter,), 1)
    h = nothing
    for _ in 1:1000
        if h === nothing
            h = insert!(g) do
                dispatch!(g, mut_inc!, (scratch,), 64)
            end
        else
            delete!(g, h)
            h = nothing
        end
        run!(g)
    end
    @test Array(counter) == [1000]
end

# D:244-254: 10 000 insert/delete cycles of a block naming a device array, a
# transient and a value GPURef: memory figures back at their baseline after
# free!(A) (holds, namers, the graph's named set and the GPURef's plans do not grow).
@case "MUT-19" 2 :long begin
    dev = testdevice(); n = 1 << 16
    g, counter = countergraph(dev)
    baseline = memory()
    A = zeroed(dev, n)
    r = GPURef(dev, Int32(1))
    for _ in 1:10_000
        h = insert!(g) do
            t = MantleArray(g, Int32, n)
            dispatch!(g, mut_add!, (t, A, r), n)
            dispatch!(g, mut_add!, (A, t, Int32(0)), n)
        end
        run!(g)
        delete!(g, h)
        run!(g)
    end
    @test Array(A) == fill(10_000, n)
    @test Array(counter) == [20_001]
    free!(A)
    @test all(Tuple(memory()) .<= Tuple(baseline))
end

# I:244-248 vs P:1440: two tasks declare insert! blocks into one graph at once:
# serialized by the graph lock, each handle holds only its own nodes
# (ambiguity 19; the case follows api.md 2.3, declaring takes the graph lock).
@case "MUT-20" 1 begin
    dev = testdevice(); n = 64
    arrs = [zeroed(dev, n) for _ in 1:4]
    g = Graph(dev)
    handles = Vector{Any}(undef, 2)
    tasks = map(1:2) do k
        Threads.@spawn handles[k] = insert!(g) do
            dispatch!(g, mut_inc!, (arrs[2k-1],), n)
            sleep(0.2)                              # the other task tries to declare meanwhile
            dispatch!(g, mut_inc!, (arrs[2k],), n)
        end
    end
    withtimeout(() -> foreach(wait, tasks), 60)
    run!(g)
    delete!(g, handles[1])
    run!(g)
    @test Array.(arrs) == [fill(1, n), fill(1, n), fill(2, n), fill(2, n)]
end

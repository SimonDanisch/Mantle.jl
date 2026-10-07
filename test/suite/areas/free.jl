# Free and release with pending changes (catalogue-resize-rt.md, table K:
# FR). Contracts: api.md:79, 156, 375, 482, 487; resizing-and-raytracing.jl
# B.8 (1145-1147), B.8b (1271-1282), B.10 (1312-1323) and 467-479 (WaitFor);
# internals.jl sections 6 (1037-1055), 9 (1945-1956) and 10 (2108-2179).
# Line numbers of 2026-10-05.
#
# Leaks are caught with memory() back at its baseline; use after free with
# canaries allocated right after the release while work that writes the
# released memory is still on the GPU. Traces are not checked here (api.md
# does not give the shader side of a trace); FR-08 and FR-09 check that runs
# building the TLAS go on and what is released when.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Mat4f, Point3f, TriangleFace

const FR_LEN = 2^20               # Int32 elements: 4 MiB
const FR_BIG = 64 * 2^20          # Int32 elements of a large array: 256 MiB

@kernel function fr_addone!(a)
    i = @index(Global)
    a[i] += one(eltype(a))
end

# Spins `iters` steps, then writes `value` into x[1].
@kernel function fr_spinwrite!(x, value, iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    x[1] = value + (acc == Int32(-7) ? Int32(1) : Int32(0))
end

"""A graph that adds one to device array `a` into a transient and copies it to `out`: it names both."""
function plusonegraph(dev, a, out)
    g = Graph(dev)
    t = MantleArray(g, eltype(a), length(a))
    t .= a .+ one(eltype(a))
    copyto!(out, t)
    return g
end

"""A FaultDevice over the suite's device whose pool and cell arena were used once."""
function faultdevice(dev = testdevice())
    d = FaultDevice(dev)
    free!(MantleArray(d, Int32, 1))
    return d
end

"""Caps `d`'s budget at what its pool holds now plus `headroom` bytes [FI-9]; returns the cap."""
function limit!(d, headroom)
    cap = Mantle.reserved(d) + headroom
    inject!(d, :budget, Answer(calls(d, :budget) + 1, cap))
    return cap
end

"""Waits for `t` (a watchdog fails the test after `timeout` s) and returns what it threw, or `nothing`."""
function thrownby(t::Task; timeout = 30)
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("task did not finish after $timeout s: likely a deadlock")
    return istaskfailed(t) ? t.result : nothing
end

translation(x, y, z) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1)

"""A BLAS over one triangle, with its vertex and face arrays."""
function blaswitharrays(dev)
    vertices = MantleArray(dev, Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    return BLAS(dev, vertices, faces), vertices, faces
end

"""Creates an array with a pending resize and store and drops it without free!."""
@noinline function droppending(dev, n)
    a = MantleArray(dev, Int32, n)
    resize!(a, 2n)
    a[1:n] = testdata(Int32, n)
    return nothing
end

# ── K. Free and release with pending changes (FR) ──

# resizing-and-raytracing.jl:1314-1321, api.md:79, 375: a pending resize past the storage and a store, only the creator's hold: free!(a) and a sweep make no submission and allocate nothing; upload region and cell go back, memory at the baseline.
@case "FR-01" 4 begin
    dev = testdevice()
    base = memory(dev)
    a = MantleArray(dev, Int32, FR_LEN)
    resize!(a, 4FR_LEN)
    a[1:FR_LEN] = testdata(Int32, FR_LEN)
    m, diag = counted(() -> (free!(a); memory(dev)), dev)
    @test submitcount(diag) == 0
    @test diag.allocations == 0
    @test m == base
end

# api.md:482: a graph holds a, a store is pending, free!(a): the handle throws, the graph's next run applies the store, memory comes back after free!(g).
@case "FR-02" 4 begin
    dev = testdevice()
    base = memory(dev)
    x = testdata(Int32, FR_LEN)
    a, o = MantleArray(dev, x), MantleArray(dev, Int32, FR_LEN)
    g = plusonegraph(dev, a, o)
    run!(g)
    a[1:4] = Int32[1, 2, 3, 4]
    free!(a)
    @test_throws ArgumentError (a[1:1] = Int32[0])
    run!(g)
    x[1:4] = 1:4
    @test Array(o) == x .+ Int32(1)
    free!(g); free!(o)
    @test memory(dev) == base
end

# api.md:79: free!(a) from 8 threads at once, 100 times: the creator's hold drops once and the array is released once (memory, cells included, at the baseline).
@case "FR-03" 2 begin
    dev = testdevice()
    base = memory(dev)
    for _ in 1:100
        a = MantleArray(dev, Int32, 1024)
        go = Threads.Event()
        tasks = [Threads.@spawn (wait(go); free!(a)) for _ in 1:8]
        notify(go)
        withtimeout(() -> foreach(wait, tasks), 30)
    end
    @test memory(dev) == base
end

# resizing-and-raytracing.jl:1317-1320: the last hold dropping between an eager call's prepare! and its locks (no hook there: 20 races of resize!-then-broadcast loops against free!): every call either runs or throws ArgumentError, and storage prepared for it goes back (memory at the baseline).
@case "FR-04" 4 begin
    dev = testdevice()
    base = memory(dev)
    for _ in 1:20
        a = MantleArray(dev, zeros(Int32, 1024))
        go = Threads.Event()
        caller = Threads.@spawn begin
            wait(go)
            for i in 1:50
                resize!(a, 1024 + 64i)
                a .+= Int32(1)
            end
        end
        dropper = Threads.@spawn (wait(go); free!(a))
        notify(go)
        withtimeout(() -> wait(dropper), 30)
        e = thrownby(caller)
        @test e === nothing || e isa ArgumentError
    end
    @test memory(dev) == base
end

# resizing-and-raytracing.jl:1271-1282: r[] = b, then free!(r) before any submission, while a launch through r still writes a: the rebind is dropped, a's hold goes with r's last tokens (canaries intact), b's hold drops; memory at the baseline.
@case "FR-05" 6 begin
    dev = testdevice()
    base = memory(dev)
    a, b = MantleArray(dev, Int32, FR_LEN), MantleArray(dev, Int32, FR_LEN)
    r = GPURef(dev, a)
    dispatch!(dev, fr_spinwrite!, (r, Int32(7), Int32(calibratedspin(dev, 500))), 1)
    r[] = b
    free!(r); free!(a); free!(b)
    cs = [canary(dev, FR_LEN) for _ in 1:4]
    Mantle.waitidle(dev)
    @test all(intact, cs)
    foreach(free!, cs)
    @test memory(dev) == base
end

# resizing-and-raytracing.jl:1322-1323, internals.jl:2168-2179: every reference to an array with a pending resize and store dropped without free!: released through the inbox at the next sweep, with no submission.
@case "FR-06" 4 begin
    dev = testdevice()
    base = memory(dev)
    droppending(dev, FR_LEN)
    m, diag = counted(() -> memory(dev), dev)
    @test submitcount(diag) == 0
    @test m == base
end

# resizing-and-raytracing.jl:1145-1147: a TLAS with three ranges and its pending Build, free!(t): the Build is never applied (no submission) and its storage and members' holds go; memory at the baseline once the BLAS and its arrays are freed too.
@case "FR-07" 5 :rt begin
    dev = testdevice()
    base = memory(dev)
    blas, vertices, faces = blaswitharrays(dev)
    t = TLAS(dev)
    foreach(i -> push!(t, blas, translation(3i, 0, 0)), 1:3)
    m, diag = counted(dev) do
        free!(t); free!(blas); free!(vertices); free!(faces)
        memory(dev)
    end
    @test submitcount(diag) == 0
    @test m == base
end

# api.md:156: free!(blas) while a range holds it: runs building the TLAS go on; after delete! of the range and a run, the BLAS goes with that Build's token (canaries intact); memory at the baseline.
@case "FR-08" 5 :rt begin
    dev = testdevice()
    base = memory(dev)
    blas, vertices, faces = blaswitharrays(dev)
    t = TLAS(dev)
    h = push!(t, blas, translation(0, 0, 0))
    g = Graph(dev)
    build!(g, t)
    run!(g)
    free!(blas); free!(vertices); free!(faces)    # the range holds the BLAS, the BLAS its arrays
    run!(g)
    delete!(t, h)
    run!(g)                                       # the Build without the range
    cs = [canary(dev, FR_LEN) for _ in 1:4]
    Mantle.waitidle(dev)
    @test all(intact, cs)
    free!(g); free!(t); foreach(free!, cs)
    @test memory(dev) == base
end

# api.md:482: free!(src) while a range holds it: the handle throws; the range keeps src and runs building the TLAS go on.
@case "FR-09" 5 :rt begin
    dev = testdevice()
    blas, vertices, faces = blaswitharrays(dev)
    src = MantleArray(dev, [translation(0, 0, 0), translation(3, 0, 0)])
    t = TLAS(dev)
    push!(t, blas, src)
    g = Graph(dev)
    build!(g, t)
    run!(g)
    free!(src)
    @test_throws ArgumentError (src[1:1] = [translation(1, 0, 0)])
    @test_throws ArgumentError fill!(src, translation(1, 0, 0))
    run!(g); run!(g)
    free!(g); free!(t); free!(blas); free!(vertices); free!(faces)
end

# internals.jl:1037-1055, resizing-and-raytracing.jl:467-479, api.md:487: g sits between its stages (its host function blocked) and launched directly over a; after resize!(a), another task's eager call on a waits for the run's end and a third task's free!(g) for the graph lock; both finish once the run ends.
@case "FR-10" 5 begin
    dev = testdevice()
    x = testdata(Int32, FR_LEN)
    a = MantleArray(dev, x)
    gate = Gate()
    g = Graph(dev)
    dispatch!(g, fr_addone!, (a,), launchrange(a))   # direct: a is fixed at compile
    t = MantleArray(g, Int32, 16)
    fill!(t, Int32(0))
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    fill!(t, Int32(1))                               # the stage after the host function
    runner = Threads.@spawn run!(g)
    waitentered(gate)
    resize!(a, 2FR_LEN)                              # only enqueues: never waits
    caller = Threads.@spawn (a .+= Int32(1); Array(view(a, 1:8)))
    freer = Threads.@spawn free!(g)
    sleep(0.5)
    @test !istaskdone(caller)
    @test !istaskdone(freer)
    letgo!(gate)
    withtimeout(() -> fetch(runner), 30)
    v = withtimeout(() -> fetch(caller), 30)
    withtimeout(() -> fetch(freer), 30)
    @test v == x[1:8] .+ Int32(2)
end

# internals.jl:1949-1950, 1956: an evicted array with a pending resize drops its last hold: its host copy (and restore storage, had a submitter prepared one; that moment has no hook) goes back: memory at the baseline.
@case "FR-11" 4 :separateheap begin
    d = faultdevice()
    base = memory(d)
    a = MantleArray(d, testdata(Int32, FR_BIG))
    limit!(d, 2FR_BIG)                               # half of a's bytes
    b = MantleArray(d, Int32, FR_BIG)                # evicts a
    resize!(a, 2FR_BIG)
    free!(a); free!(b)
    @test memory(d) == base
end

# api.md:79, internals.jl:2229-2230 (ambiguity A25, settled there): free!(a), then Array(a) and copyto!(host, a) while a graph still holds a: both throw; the graph keeps running.
@case "FR-12" 2 begin
    dev = testdevice()
    x = testdata(Int32, FR_LEN)
    a, o = MantleArray(dev, x), MantleArray(dev, Int32, FR_LEN)
    g = plusonegraph(dev, a, o)
    free!(a)
    @test_throws ArgumentError Array(a)
    @test_throws ArgumentError copyto!(zeros(Int32, FR_LEN), a)
    run!(g)
    @test Array(o) == x .+ Int32(1)
end

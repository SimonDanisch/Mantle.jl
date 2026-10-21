# Free and release with pending changes. Contracts: api.md 2.2 (`free!`),
# 2.6 (`delete!(tlas, r)`), 5 (the last hold) and 7; resizing-and-raytracing.jl
# B.4 (takepending, WaitFor), B.8 (free! of a TLAS and a BLAS) and B.10;
# internals.jl sections 6, 9 and 10.
#
# Leaks are caught with memory() back at its baseline; use after free with
# canaries allocated right after the release while work that writes the
# released memory is still on the GPU. Traces are not checked here (api.md
# does not give the shader side of a trace); FR-08 and FR-09 check that runs
# building the TLAS go on and what is released when.

using GeometryBasics: Mat4f, Point3f, TriangleFace

const FR_LEN = 2^20               # Int32 elements: 4 MiB
const FR_BIG = 64 * 2^20          # Int32 elements of a large array: 256 MiB

"""a[i] += 1."""
function fr_addone!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds a[i] += one(eltype(a))
    return nothing
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

"""Caps `d`'s budget at what its pool holds now plus `headroom` bytes; returns the cap."""
function limit!(d, headroom)
    cap = Mantle.reserved(d) + headroom
    inject!(d, :budget, BudgetLimit(calls(d, :budget) + 1, d, cap))
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

# ── Free and release with pending changes (FR) ──

# resizing-and-raytracing.jl B.10, api.md 2.2 (`free!`), 5 (the last hold): a pending resize past the storage and a store, only the creator's hold: free!(a) and a sweep make no submission and allocate nothing; upload region and cell go back, memory at the baseline.
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

# api.md 7 (free!(a) while graphs hold a), 2.2: a graph holds a, a store is pending, free!(a): the handle throws, the graph's next run applies the store, memory comes back after free!(g).
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

# api.md 2.2 (`free!(a)`: the creator's hold drops once, whatever the number of threads): free!(a) from 8 threads at once, 100 times: the creator's hold drops once and the array is released once (memory, cells included, at the baseline).
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

# resizing-and-raytracing.jl B.4 (prepare!, then the locks), B.10 (prepared storage released at the last hold): the last hold dropping between an eager call's prepare! and its locks (no hook there: 20 races of resize!-then-broadcast loops against free!): every call either runs or throws ArgumentError, and storage prepared for it goes back (memory at the baseline).
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

# resizing-and-raytracing.jl B.10 (every holder's finalizer pushes through the inbox), internals.jl section 10 (finalizers, settle!): every reference to an array with a pending resize and store dropped without free!: released through the inbox at the next sweep, with no submission.
@case "FR-06" 4 begin
    dev = testdevice()
    base = memory(dev)
    droppending(dev, FR_LEN)
    m, diag = counted(() -> memory(dev), dev)
    @test submitcount(diag) == 0
    @test m == base
end

# resizing-and-raytracing.jl B.8 (free!(t): the ranges' composite holds dropped with the TLAS's last tokens): a TLAS with three ranges and its pending Build, free!(t): the Build is never applied (no submission) and its storage and members' holds go; memory at the baseline once the BLAS and its arrays are freed too.
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

# api.md 2.6 (`delete!(tlas, r)`: the range's BLAS and arrays released with the token of the build that deactivated it): free!(blas) while a range holds it: runs building the TLAS go on; after delete! of the range and a run, the BLAS goes with that Build's token (canaries intact); memory at the baseline.
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

# api.md 2.2 (`free!`: composites that hold a keep using it), 7: free!(src) while a range holds it: the handle throws; the range keeps src and runs building the TLAS go on.
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

# internals.jl section 6, resizing-and-raytracing.jl B.4 (takepending, waitforruns), api.md 7 (a change that would make a running plan stale is applied after its last stage): g sits between its stages (its host function blocked) and launched directly over a; after resize!(a), another task's eager call on a waits for the run's end and a third task's free!(g) for the graph lock; both finish once the run ends.
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

# internals.jl sections 9 (an evicted array's pending resize, the last hold) and 10 (release!): an evicted array with a pending resize drops its last hold: its host copy (and restore storage, had a submitter prepared one; that moment has no hook) goes back: memory at the baseline.
@case "FR-11" 4 begin
    d = faultdevice()
    base = memory(d)
    a = MantleArray(d, testdata(Int32, FR_BIG))
    limit!(d, 2FR_BIG)                               # half of a's bytes
    b = MantleArray(d, Int32, FR_BIG)                # evicts a
    resize!(a, 2FR_BIG)
    free!(a); free!(b)
    @test memory(d) == base
end

# api.md 2.2 (`free!(a)`: later use through any handle throws), internals.jl section 11 (`Array`): free!(a), then Array(a) and copyto!(host, a) while a graph still holds a: both throw; the graph keeps running.
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

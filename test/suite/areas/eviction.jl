# Eviction, memory pressure and read-only inputs. Contracts: api.md 2.2
# (`readonly = true`, `trim!`), 6 (two graphs that each need 90% of the
# device) and 7 (eviction, pressure steps); recording-plan.md "Memory" (an
# idle resource can leave the device, pressure policy); internals.jl section 9
# (eviction and restore, acquire!).
#
# Pressure is forced through FaultDevice's `budget` answer: a
# FaultDevice has its own pool, so it holds only the case's arrays. The large
# arrays are 256 MiB, so each needs a block of its own and meets the budget
# check (a region that fits a block already held is never checked).
# Assumed of the hooks: Mantle.steplog entries have `kind` (a Symbol:
# :Restore, :InlineStore, :StagedStore, :MoveStorage) and `target` (the root
# array the user holds). Assumed of eviction: the host copy is host memory and
# does not count against budget(dev) (otherwise evicting frees nothing).
# Read-only inputs are `MantleArray(dev, host; readonly = true)`; a file is
# mapped by the caller (`Mmap.mmap` returns a plain `Array`) and passed the
# same way (api.md 2.2): Mantle does not know where the bytes come from.

import Mmap
using GeometryBasics: Vec2f, Vec4f, Point3f, TriangleFace

const EV_LEN = 64 * 2^20          # Int32 elements of a large array: 256 MiB
const EV_BYTES = 4 * EV_LEN
const EV_SAMPLES = 1024           # elements of an accumulator

"""acc[j] += x[(j - 1) * stride + 1]."""
function ev_addsample!(acc, x, stride)
    j = KI.get_global_id().x
    j <= length(acc) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds acc[j] += x[(j - 1) * stride + 1]
    return nothing
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



"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""The steps in `log` of `kind` on `x`."""
logged(log, kind, x) = [s for s in log if s.kind === kind && s.target === x]

"""The first `n` elements of `a`: a host read of only that many, through a view."""
firstvalues(a, n = 8) = Array(view(a, 1:n))

"""
Whether `a` is evicted: a host read of an evicted array copies from its host
copy and submits nothing, a read of a resident one is a submission (api.md
2.2, `Array(a)`; internals.jl section 9). Nothing may be pending on `a`.
"""
isevicted(a, d) = submitcount(last(counted(() -> firstvalues(a, 1), d))) == 0

"""Every `length(x) ÷ m`-th element of `x`, `m` of them: what ev_addsample! reads."""
samplesof(x, m = EV_SAMPLES) = x[1:(length(x) ÷ m):end][1:m]

"""What an accumulator holds after `runs` runs over weights with host data `xs`."""
expectedacc(runs, xs...) = Int32(runs) .* samplesof(reduce((p, q) -> p .+ q, xs))

accumulator(d) = MantleArray(d, zeros(Int32, EV_SAMPLES))

"""Adds a sample of `x` to `acc`: a node of graph `l`, or an eager call on device `l`."""
addsample!(l, acc, x) = dispatch!(l, ev_addsample!, (acc, x, Int32(length(x) ÷ length(acc))), length(acc))

"""
A model call on `d` with weights `ws` (device arrays of equal length): every
run sums them into a transient as large and adds a sample of it to `acc`.
"""
function modelgraph(d, acc, ws...)
    g = Graph(d)
    t = MantleArray(g, Int32, length(first(ws)))
    broadcast!(+, t, ws...)
    addsample!(g, acc, t)
    return g
end

"""A graph that adds a sample of device array `w` to `acc` every run, with no transient."""
readergraph(d, acc, w) = (g = Graph(d); addsample!(g, acc, w); g)

"""Writes and drops an array of `n` Int32 on `d` without free!: only the garbage collection step can reclaim it."""
@noinline unreferenced!(d, n) = (fill!(MantleArray(d, Int32, n), Int32(0)); nothing)

"""A read-only array on `d` over the file at `path`, mapped by the caller as `T` with `dims` (api.md 2.2)."""
mmaparray(d, T, path, dims) = MantleArray(d, Mmap.mmap(path, Array{T,length(dims)}, dims); readonly = true)

"""Overwrites the file at `path` in place: no truncation, so a mapping of it stays valid."""
overwrite(path, data) = open(io -> write(io, data), path, "r+")

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

"""The median host time of `n` calls of `run!(g)`, in seconds."""
function medianrun(g, n = 200)
    ts = [(t0 = time_ns(); run!(g); time_ns() - t0) for _ in 1:n]
    return sort(ts)[n ÷ 2 + 1] / 1e9
end

# EVICT-07's draw. api.md does not give the shader-side signatures; assumed
# here: the vertex stage gets the vertex index and the draw's arguments.
ev_vertex(i, verts) = Vec4f(verts[i][1], verts[i][2], 0f0, 1f0)
ev_fragment(_) = 1f0

# ── Eviction, pressure, read-only modes (EVICT) ──

# api.md 6, recording-plan.md "Memory" (an idle resource can leave the device): two graphs whose runs each need about 90% of the budget run alternately 50 times with no host wait: exact, no recompile, and every run restores the weights the other run evicted.
@case "EVICT-01" 4 begin
    d = faultdevice()
    x1, x2 = testdata(Int32, EV_LEN; seed = 1), testdata(Int32, EV_LEN; seed = 2)
    w1, w2 = MantleArray(d, x1), MantleArray(d, x2)
    acc1, acc2 = accumulator(d), accumulator(d)
    g1, g2 = modelgraph(d, acc1, w1), modelgraph(d, acc2, w2)
    limit!(d, EV_BYTES ÷ 5)               # both weights held now: a run needs its weights and a transient as large, 2W of 2.2W
    run!(g1); run!(g2)                    # compiles both
    _, log = stepsduring(d) do
        _, diag = counted(d) do
            for _ in 1:50
                run!(g1); run!(g2)
            end
        end
        @test diag.plancompiles == 0
    end
    @test Array(acc1) == expectedacc(51, x1)
    @test Array(acc2) == expectedacc(51, x2)
    @test length(logged(log, :Restore, w1)) >= 49
    @test length(logged(log, :Restore, w2)) >= 49
end

# internals.jl section 9 (evictable: the eviction copy waits for the last write): an array evicted while its last write is still on the GPU: the eviction copy waits for the write, and the restore brings back the latest contents.
@case "EVICT-02" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    slowwrite!(d, a, 7; ms = 500)         # a[1] = 7 after about 500 ms on the GPU
    b = MantleArray(d, Int32, EV_LEN)     # needs a's bytes: a is evicted with the write in flight
    x[1] = 7
    @test isevicted(a, d)
    @test Array(a) == x                   # the host copy holds the write
    free!(b)
    _, log = stepsduring(() -> (a .+= Int32(1)), d)
    @test length(logged(log, :Restore, a)) == 1
    @test firstvalues(a) == x[1:8] .+ Int32(1)
end

# api.md 7 (eviction: no running plan names it): an array a running plan names (the run held in its host function) is not evicted: the allocation throws; after the run the same allocation evicts it.
@case "EVICT-03" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    w, acc = MantleArray(d, x), accumulator(d)
    g = readergraph(d, acc, w)
    t = MantleArray(g, Int32, EV_SAMPLES)
    copyto!(t, acc)                       # the host function's input: after the kernel that reads w
    gate = Gate()
    dispatch!(g, blockinghost(gate), (t,); writes = ())
    limit!(d, EV_BYTES ÷ 2)
    runner = Threads.@spawn run!(g)
    waitentered(gate)                     # g's plan is running
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, EV_LEN)
    letgo!(gate)
    withtimeout(() -> fetch(runner), 30)
    b = MantleArray(d, Int32, EV_LEN)
    @test isevicted(w, d)
    @test Array(acc) == expectedacc(1, x)
end

# api.md 7 (eviction: nothing pending on it): an array with a pending store is not evicted; once a host read applied the store, it is.
@case "EVICT-04" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    a[1:4] = Int32[1, 2, 3, 4]
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, EV_LEN)
    @test firstvalues(a) == [Int32[1, 2, 3, 4]; x[5:8]]
    b = MantleArray(d, Int32, EV_LEN)
    @test isevicted(a, d)
end

# api.md 7 (eviction: not lent), 2.9: an array lent to withview is not evicted while the block runs; after the block it is.
@case "EVICT-05" 10 :vendorview begin
    # Needs faultdevice.jl to forward the device type's vendor-view verb once it has a name (api.md 10, vendor views).
    d = faultdevice()
    a = MantleArray(d, testdata(Int32, EV_LEN))
    limit!(d, EV_BYTES ÷ 2)
    cuda = cudamodule()
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, EV_LEN)
    end
    b = MantleArray(d, Int32, EV_LEN)
    @test isevicted(a, d)
end

# api.md 7 (eviction: no composite holds it), internals.jl section 10 (composite holds), resizing-and-raytracing.jl B.8 (a BLAS holds its vertex and index arrays): an array a composite holds (a BLAS's vertices) is never evicted; after free!(blas) it is.
@case "EVICT-06" 6 begin
    d = faultdevice()
    a = MantleArray(d, fill(Point3f(0, 0, 0), EV_BYTES ÷ 12))   # 256 MiB of vertices
    faces = MantleArray(d, [TriangleFace{UInt32}(1, 2, 3)])
    blas = BLAS(d, a, faces)
    limit!(d, EV_BYTES ÷ 2)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, EV_LEN)
    free!(blas)
    b = MantleArray(d, Int32, EV_LEN)
    @test isevicted(a, d)
end

# api.md 7 (eviction: no handle dependent), 2.6b (`draw!`: indices bound by handle): an index array a draw binds by handle is never evicted; once the plan that recorded it is gone, it is.
@case "EVICT-07" 6 begin
    d = faultdevice()
    verts = MantleArray(d, Vec2f[(-1, -1), (1, -1), (0, 1)])
    idx = MantleArray(d, UInt32.(rand(Xoshiro(7), 1:3, EV_LEN)))
    img = Image(d, Float32, 64, 64)
    pipeline = Rasterizer(; vertex = VertexShader(ev_vertex), fragment = FragmentShader(ev_fragment))
    g = Graph(d)
    clear!(g, img, 0f0)
    draw!(g, img, pipeline, (verts,), EV_LEN; indices = idx)
    run!(g)
    limit!(d, EV_BYTES ÷ 2)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, UInt32, EV_LEN)
    free!(g)
    b = MantleArray(d, UInt32, EV_LEN)
    @test isevicted(idx, d)
end

# internals.jl section 9 (a host read copies from the host copy): a host read of an evicted array reads its host copy and submits nothing.
@case "EVICT-09" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    b = MantleArray(d, Int32, EV_LEN)     # evicts a
    v, diag = counted(() -> Array(a), d)
    @test v == x
    @test submitcount(diag) == 0
end

# api.md 7, internals.jl section 9 (the last hold retires the host copy): free! of an evicted array (its last hold) releases its host copy: memory back at the baseline.
@case "EVICT-10" 4 begin
    d = faultdevice()
    base = memory(d)
    a = MantleArray(d, testdata(Int32, EV_LEN))
    limit!(d, EV_BYTES ÷ 2)
    b = MantleArray(d, Int32, EV_LEN)
    @test isevicted(a, d)
    free!(a); free!(b)
    @test memory(d) == base
end

# recording-plan.md "Memory" (while evicted, a store restores it first and is pended after the Restore): a store into an evicted array: the restore, then the store, in one submission (a host read's).
@case "EVICT-11" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    free!(MantleArray(d, Int32, EV_LEN))  # evicts a, then gives the bytes back
    a[1:4] = Int32[1, 2, 3, 4]
    (v, diag), log = stepsduring(() -> counted(() -> firstvalues(a), d), d)
    kinds = [s.kind for s in log if s.target === a]
    i, j = findfirst(==(:Restore), kinds), findfirst(in((:InlineStore, :StagedStore)), kinds)
    @test i !== nothing && j !== nothing && i < j
    @test submitcount(diag) == 1
    @test v == [Int32[1, 2, 3, 4]; x[5:8]]
end

# internals.jl section 9, resizing-and-raytracing.jl B.5 (resizesteps of Evicted), recording-plan.md "Memory": resize! of an evicted array restores it into storage of the new size, with no move after the restore.
@case "EVICT-12" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    free!(MantleArray(d, Int32, EV_LEN))
    resize!(a, EV_LEN + 1024)
    @test size(a) == (EV_LEN + 1024,)
    _, log = stepsduring(() -> firstvalues(a), d)
    @test length(logged(log, :Restore, a)) == 1
    @test isempty(logged(log, :MoveStorage, a))
    @test Array(view(a, EV_LEN-7:EV_LEN)) == x[end-7:end]
end

# internals.jl sections 9 (restore) and 11 (eager calls): an eager call on an evicted array restores it in the call's own submission.
@case "EVICT-13" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x)
    limit!(d, EV_BYTES ÷ 2)
    free!(MantleArray(d, Int32, EV_LEN))
    (_, diag), log = stepsduring(() -> counted(() -> (a .+= Int32(1)), d), d)
    @test submitcount(diag) == 1
    @test length(logged(log, :Restore, a)) == 1
    @test firstvalues(a) == x[1:8] .+ Int32(1)
end

# recording-plan.md "Memory" (a resource evicted after the run looked makes it start over): an array evicted after a run looked at its plan makes the run start over and restore it; runs stay exact while another task evicts in a loop (the window between the look and the locks has no hook, so this drives the race).
@case "EVICT-14" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    w, acc = MantleArray(d, x), accumulator(d)
    spare = MantleArray(d, zeros(Int32, EV_LEN))   # always evictable: the evicting task never runs out of room
    g = readergraph(d, acc, w)
    limit!(d, EV_BYTES ÷ 2)
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    evicter = Threads.@spawn begin
        n = 0
        while !stop[]
            spare .+= Int32(1)                     # restores spare, evicting w when g is idle
            free!(MantleArray(d, Int32, EV_LEN))   # evicts the least recently used of w and spare
            n += 1
        end
        n
    end
    _, log = stepsduring(() -> foreach(_ -> run!(g), 1:100), d)
    stop[] = true
    n = withtimeout(() -> fetch(evicter), 60)
    @test Array(acc) == expectedacc(101, x)
    @test firstvalues(spare) == fill(Int32(n), 8)
    @test !isempty(logged(log, :Restore, w))
end

# recording-plan.md "Memory" (a plan whose own compile evicted what it names restores it at its next attempt): a new plan whose own compile evicts what it names (no sparse residency, a caps override, so compile places the arena part whole) restores it at its next attempt, with no second compile.
@case "EVICT-15" 4 begin
    d = faultdevice()
    inject!(d, :caps, Answer(1, withcaps(caps(testdevice()); sparseresidency = false)))
    x = testdata(Int32, EV_LEN)
    w = MantleArray(d, x)                          # the least recently used
    spare = MantleArray(d, zeros(Int32, EV_LEN))
    acc = accumulator(d)
    g = modelgraph(d, acc, w)                      # a transient as large as w
    limit!(d, EV_BYTES ÷ 2)
    (_, diag), log = stepsduring(() -> counted(() -> run!(g), d), d)
    @test diag.plancompiles == 1
    @test length(logged(log, :Restore, w)) == 1
    @test isevicted(spare, d)
    @test Array(acc) == expectedacc(1, x)
end

# recording-plan.md "Memory" (the restore's allocation may evict others, never what the running plan names): a restore may evict the other graph's weights, never an array its own running plan names: g1 reads two weights, g2 one; alternating runs stay exact.
@case "EVICT-16" 4 begin
    d = faultdevice()
    half = EV_LEN ÷ 2
    xa, xb = testdata(Int32, half; seed = 1), testdata(Int32, half; seed = 2)
    x2 = testdata(Int32, EV_LEN; seed = 3)
    wa, wb, w2 = MantleArray(d, xa), MantleArray(d, xb), MantleArray(d, x2)
    acc1, acc2 = accumulator(d), accumulator(d)
    g1, g2 = modelgraph(d, acc1, wa, wb), modelgraph(d, acc2, w2)
    limit!(d, EV_BYTES ÷ 5)
    withtimeout(() -> foreach(_ -> (run!(g1); run!(g2)), 1:20), 120)
    @test Array(acc1) == expectedacc(20, xa, xb)
    @test Array(acc2) == expectedacc(20, x2)
end

# api.md 7 (pressure steps), recording-plan.md "Memory" (pressure policy, only in acquire!): pressure steps in order: collecting garbage before waiting for retiring storage, both before evicting an idle array, OutOfDeviceMemory last.
@case "EVICT-17" 4 begin
    d = faultdevice()
    a = MantleArray(d, testdata(Int32, EV_LEN))    # idle and evictable: only the eviction step may take it
    limit!(d, 2 * EV_BYTES + EV_BYTES ÷ 2)
    b = MantleArray(d, Int32, EV_LEN)
    unreferenced!(d, EV_LEN)
    slowwrite!(d, b, 1; ms = 2000); free!(b)       # retiring until its write passed
    c, diag = counted(() -> MantleArray(d, Int32, EV_LEN), d)
    @test diag.hostwaits == 0                      # collecting garbage was enough: no wait for b
    @test !isevicted(a, d)
    e, diag = counted(() -> MantleArray(d, Int32, EV_LEN), d)
    @test diag.hostwaits >= 1                      # waited for b's retirement instead of evicting
    @test !isevicted(a, d)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, 8 * EV_LEN)
end

# recording-plan.md "Memory" (acquire! submits nothing but the unmapping of idle pages and the eviction copies): an allocation within the budget submits and waits for nothing; under pressure Mantle submits only the eviction copies, one per evicted array.
@case "EVICT-18" 4 begin
    d = faultdevice()
    a1 = MantleArray(d, testdata(Int32, EV_LEN ÷ 2; seed = 1))
    a2 = MantleArray(d, testdata(Int32, EV_LEN ÷ 2; seed = 2))
    limit!(d, EV_BYTES ÷ 4)
    small, diag = counted(() -> MantleArray(d, Int32, 1024), d)
    @test submitcount(diag) == 0
    @test diag.hostwaits == 0
    b, diag = counted(() -> MantleArray(d, Int32, EV_LEN), d)
    @test submitcount(diag) == 2
    @test isevicted(a1, d) && isevicted(a2, d)
end

# internals.jl section 9 (tryacquire!: the budget check and the new block under one lock): two threads allocate and free 40% of the room in a loop: what the pool holds never exceeds the budget.
@case "EVICT-19" 2 begin
    d = faultdevice()
    cap = limit!(d, EV_BYTES)
    tasks = map(1:2) do _
        Threads.@spawn begin
            peak = 0
            for _ in 1:20
                a = MantleArray(d, Int32, 2 * EV_LEN ÷ 5)
                fill!(a, Int32(1))
                peak = max(peak, Mantle.reserved(d))
                free!(a)
            end
            peak
        end
    end
    @test maximum(withtimeout(() -> fetch.(tasks), 120)) <= cap
end

# api.md 7 (OutOfDeviceMemory after the pressure steps): one allocation larger than the whole budget throws OutOfDeviceMemory; an existing array stays intact and nothing the attempt prepared stays allocated.
@case "EVICT-20" 2 begin
    d = faultdevice()
    base = memory(d)
    x = testdata(Int32, 1024)
    a = MantleArray(d, x)
    cap = limit!(d, EV_BYTES)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, cap ÷ 2)   # twice the budget in bytes
    @test Array(a) == x
    free!(a)
    @test memory(d) == base
end

# recording-plan.md "Memory" (only a marked plan looks at what it names): a run whose plan is not marked walks nothing: its host time does not grow with 10 000 idle arrays on the eviction list (interleaved A/B).
@case "EVICT-21" 4 begin
    dev = testdevice()
    a = MantleArray(dev, zeros(Int32, 1024))
    g = Graph(dev)
    t = MantleArray(g, Int32, 1024)
    t .= a .+ Int32(1)
    copyto!(a, t)
    run!(g)
    few, many = Float64[], Float64[]
    for _ in 1:3
        push!(few, medianrun(g))
        idle = [MantleArray(dev, Int32, 16) for _ in 1:10_000]
        push!(many, medianrun(g))
        foreach(free!, idle)
    end
    @test minimum(many) <= 1.5 * minimum(few) + 2e-6
end

# internals.jl section 9 (evictfor!: least recently used first): the least recently used array is evicted first: accessed in the order b, c, a, an allocation needing one array's bytes evicts b only.
@case "EVICT-22" 4 begin
    d = faultdevice()
    a, b, c = (MantleArray(d, testdata(Int32, EV_LEN; seed = i)) for i in 1:3)
    firstvalues(b); firstvalues(c); firstvalues(a)
    limit!(d, EV_BYTES ÷ 2)
    e = MantleArray(d, Int32, EV_LEN)
    @test isevicted(b, d)
    @test !isevicted(c, d)
    @test !isevicted(a, d)
end

# api.md 2.2 (`readonly = true`: eviction drops the device copy, a restore uploads from `host`; an mmapped `host`): an array mapped from a file is evicted with no copy-out (no submission) and restored from the file, exactly.
@case "EVICT-23" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    path = tempname()
    write(path, x)
    a = mmaparray(d, Int32, path, (EV_LEN,))
    limit!(d, EV_BYTES ÷ 2)
    b, diag = counted(() -> MantleArray(d, Int32, EV_LEN), d)
    @test isevicted(a, d)
    @test submitcount(diag) == 0
    free!(b)
    acc = accumulator(d)
    addsample!(d, acc, a)                          # restores a from the file
    @test Array(acc) == expectedacc(1, x)
    free!(a)
    rm(path)
end

# api.md 2.2 (`readonly = true`), internals.jl section 9 (evictionhost!): a read-only array created from host data leaves only VRAM: evicted with no copy-out, and restored exactly although the caller zeroed its data after creation.
@case "EVICT-24" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    a = MantleArray(d, x; readonly = true)
    x .= 0
    limit!(d, EV_BYTES ÷ 2)
    b, diag = counted(() -> MantleArray(d, Int32, EV_LEN), d)
    @test isevicted(a, d)
    @test submitcount(diag) == 0
    free!(b)
    acc = accumulator(d)
    addsample!(d, acc, a)
    @test Array(acc) == expectedacc(1, testdata(Int32, EV_LEN))
end

# api.md 2.2 (`readonly = true`: a write throws at the call), 7: every write into a read-only array throws at its call (store, copyto!, resize!, fill!, broadcast, a declared copy or kernel); reads, eager and declared, work.
@case "EVICT-25" 4 begin
    dev = testdevice()
    x = testdata(Int32, 1024)
    a = MantleArray(dev, x; readonly = true)
    @test_throws ArgumentError (a[1:4] = Int32[1, 2, 3, 4])
    @test_throws ArgumentError copyto!(a, x)
    @test_throws ArgumentError resize!(a, 2048)
    @test_throws ArgumentError fill!(a, Int32(0))
    @test_throws ArgumentError (a .+= Int32(1))
    g = Graph(dev)
    t = MantleArray(g, Int32, 1024)
    @test_throws ArgumentError copyto!(a, t)
    @test_throws ArgumentError dispatch!(g, ev_addsample!, (a, t, Int32(1)), 1024)
    t .= a .+ Int32(1)
    o = MantleArray(dev, Int32, 1024)
    copyto!(o, t)
    run!(g)
    @test Array(o) == x .+ Int32(1)
    @test Array(a) == x
end

# api.md 2.2 (`host` or its file changing afterwards is undefined; nothing checks it): a mapped file changed on disk (same size) before its restore: the contents are undefined, but the restore completes and the device stays usable.
@case "EVICT-26" 4 begin
    d = faultdevice()
    path = tempname()
    write(path, testdata(Int32, EV_LEN))
    a = mmaparray(d, Int32, path, (EV_LEN,))
    limit!(d, EV_BYTES ÷ 2)
    free!(MantleArray(d, Int32, EV_LEN))           # evicts a
    overwrite(path, testdata(Int32, EV_LEN; seed = 2))
    acc = accumulator(d)
    addsample!(d, acc, a)                          # restores from the changed file
    @test length(Array(acc)) == EV_SAMPLES
    c = MantleArray(d, Int32, 1024)
    fill!(c, Int32(5))
    @test all(==(Int32(5)), Array(c))
    free!(a)
    rm(path)
end

# recording-plan.md "Memory" (`run!(dnn_1); run!(dnn_2)` back to back), examples.jl 40b: run!(g2) right after run!(g1), with g1's run still on the GPU, evicts g1's weights under that run and completes; both stay exact.
@case "EVICT-27" 4 begin
    d = faultdevice()
    x1, x2 = testdata(Int32, EV_LEN; seed = 1), testdata(Int32, EV_LEN; seed = 2)
    w1, w2 = MantleArray(d, x1), MantleArray(d, x2)
    acc1, acc2 = accumulator(d), accumulator(d)
    g1, g2 = modelgraph(d, acc1, w1), modelgraph(d, acc2, w2)
    slowwrite!(g1, MantleArray(d, Int32, 1), 1; ms = 500)   # each run of g1 stays on the GPU for about 500 ms
    limit!(d, EV_BYTES ÷ 5)
    run!(g1); run!(g2); run!(g1)                   # compiled; w2 is evicted now
    run!(g1)
    withtimeout(() -> run!(g2), 30)                # restores w2 by evicting w1 while g1's run reads it
    @test isevicted(w1, d)
    @test Array(acc1) == expectedacc(3, x1)
    @test Array(acc2) == expectedacc(2, x2)
end

# internals.jl section 9 (restore: a marked run restores in front of its own work): a plan marked for restore (its array evicted between runs) restores the array in front of its own work: one Restore, no recompile, exact.
@case "EVICT-28" 4 begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    w, acc = MantleArray(d, x), accumulator(d)
    g = readergraph(d, acc, w)
    run!(g)
    limit!(d, EV_BYTES ÷ 2)
    free!(MantleArray(d, Int32, EV_LEN))           # evicts w and marks g's plan
    @test isevicted(w, d)
    (_, diag), log = stepsduring(() -> counted(() -> run!(g), d), d)
    @test diag.plancompiles == 0
    @test length(logged(log, :Restore, w)) == 1
    @test Array(acc) == expectedacc(2, x)
end

# api.md 2.2 (`trim!`), 7, recording-plan.md "Memory" (pressure never unmaps pages beyond an array's size): pressure never unmaps a reserve's pages beyond the array's size (a view past the size still reads them); trim! does.
@case "EVICT-29" 4 :sparse begin
    d = faultdevice()
    n = EV_BYTES ÷ 12                              # Point3f elements: 256 MiB
    x = testdata(Point3f, n)
    a = MantleArray(d, Point3f, n; capacity = n)   # resizable from the start: a reserve
    copyto!(a, x)
    v = view(a, n ÷ 2 + 1:n)                       # past the size once a shrinks
    faces = MantleArray(d, [TriangleFace{UInt32}(1, 2, 3)])
    blas = BLAS(d, a, faces)                       # a composite holds a: pressure cannot evict it
    resize!(a, n ÷ 2)
    firstvalues(a)                                 # applies the store and the shrink
    m = Mantle.mapped(d)
    limit!(d, EV_BYTES ÷ 4)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, Int32, EV_LEN)
    @test Mantle.mapped(d) >= m
    @test Array(v) == x[n ÷ 2 + 1:end]
    trim!(a)
    firstvalues(a)                                 # the submission that applies the trim
    @test Mantle.mapped(d) < m
end

# recording-plan.md "Memory" (an idle graph holds address space, not memory): under pressure an idle graph's arena pages are unmapped before anything is evicted; its next run maps them again and is exact.
@case "EVICT-30" 2 :sparse begin
    d = faultdevice()
    x = testdata(Int32, EV_LEN)
    w, acc = MantleArray(d, x), accumulator(d)
    g = modelgraph(d, acc, w)                      # a transient as large as w
    run!(g)
    limit!(d, EV_BYTES ÷ 2)
    m = Mantle.mapped(d)
    b = MantleArray(d, Int32, EV_LEN)              # g is idle: its pages go back
    @test Mantle.mapped(d) < m
    @test !isevicted(w, d)
    free!(b)
    run!(g)
    @test Array(acc) == expectedacc(2, x)
end

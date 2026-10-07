# Arena parts (catalogue-core.md, table 8: ARENA). Contracts: recording-plan.md
# "Arrays" (where contents live) and "Memory" (arenas grow and shrink, idle
# graphs hold address space); internals.jl section 6 (holdparts!) and
# section 9 (unmapidle!, acquireunmapped!, bindunmapped!).
#
# Memory pressure is forced through FaultDevice's `budget` answer [FI-9]: a
# FaultDevice has its own pool, so its arenas hold only the case's graphs.
# Graphs on one device share the device-memory arena part by default (plan,
# Memory). Byte figures come from memory() and the pool hooks.

import KernelInterface as KI

"""a[i] = i."""
function arena_iota!(a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = eltype(a)(i)
    return
end

"""a[i] = v."""
function arena_set!(a, v)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = v
    return
end

"""dst[i] = src[i] + k, over the shorter of the two."""
function arena_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return
    @inbounds dst[i] = src[i] + k
    return
end

"""acc[i] += t[i] + z[1]."""
function arena_accumulate!(acc, t, z)
    i = KI.get_global_id().x
    i <= length(acc) || return
    @inbounds acc[i] += t[i] + z[1]
    return
end

"""
A graph whose one transient holds `n` Int32 (a peak of 4n bytes in the
device-memory part): t = 1:n, then its first 16 elements into the returned
device array.
"""
function peakgraph(dev, n)
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, 16)
    dispatch!(g, arena_iota!, (t,), n)
    dispatch!(g, arena_add!, (out, t, Int32(0)), 16)
    return g, out
end

"""`c` with the fields in `kw` replaced (caps(dev) is immutable)."""
withcaps(c; kw...) = typeof(c)((get(kw, f, getfield(c, f)) for f in fieldnames(typeof(c)))...)

"""Lowers `fd`'s budget to `bytes` from its next `budget` call on [FI-9]."""
lowerbudget!(fd, bytes) = inject!(fd, :budget, Answer(calls(fd, :budget) + 1, bytes))

"""Waits for `t` (a watchdog fails the test after `timeout` s) and returns what it threw, or `nothing`."""
function thrownby(t::Task; timeout = 30)
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("task did not finish after $timeout s: likely a deadlock")
    return istaskfailed(t) ? t.result : nothing
end
"""What `f()` throws on another task, or `nothing`; a deadlock fails the test."""
thrown(f; timeout = 30) = thrownby(Threads.@spawn(f()); timeout)

"""The value of `f()` run on another task, the exception it threw, or `:timeout`. No test of its own: for loops of many attempts."""
function outcome(f; timeout = 60)
    t = Threads.@spawn f()
    timedwait(() -> istaskdone(t), timeout) === :ok || return :timeout
    return istaskfailed(t) ? t.result : fetch(t)
end

"""Frees what an allocation attempt returned; an exception or a timeout holds nothing."""
freeresult!(a::MantleArray) = free!(a)
freeresult!(_) = nothing

# ── 8. Arena parts (ARENA) ──

# P:201-206: two graphs that do not run at the same time, peaks 32 MiB and 16 MiB: the second adds less than half its own peak to reserved (the shared part costs the largest tenant, not the sum).
@case "ARENA-01" 2 begin
    dev = testdevice()
    big, outb = peakgraph(dev, 2^23)
    small, outs = peakgraph(dev, 2^22)
    run!(big)
    m1 = memory()
    run!(small)
    m2 = memory()
    @test m2.reserved - m1.reserved < 2^23
    @test Mantle.peakbytes(big) >= 2^25
    @test Array(outb) == Int32.(1:16)
    @test Array(outs) == Int32.(1:16)
end

# P:1919-1920: a small graph (4 MiB peak) runs, then a big one (64 MiB); free!(big): reserved drops back to the small graph's figure (within 2 MiB of argument blocks and recordings), and the small graph still runs.
@case "ARENA-02" 2 begin
    dev = testdevice()
    small, outs = peakgraph(dev, 2^20)
    run!(small)
    ms = memory()
    big, outb = peakgraph(dev, 2^24)
    run!(big)
    mb = memory()
    @test mb.reserved - ms.reserved >= 2^25
    free!(big)
    m = memory()
    @test m.reserved <= ms.reserved + 2^21
    fill!(outs, Int32(0))
    run!(small)
    @test Array(outs) == Int32.(1:16)
end

# P:1359-1373, I:1930-1942: the budget lowered to what is reserved [FI-9]; a 24 MiB allocation unmaps the 32 MiB of pages only the idle graph g uses; g's next run (its plan marked) binds them again before it submits, and is exact.
# Assumes no pool block keeps 24 MiB free beside g's pages, so the allocation needs a new block and meets the budget.
@case "ARENA-03" 2 :sparse begin
    fd = FaultDevice(testdevice())
    g, out = peakgraph(fd, 2^23)
    run!(g)
    m1 = memory(fd)
    @test Array(out) == Int32.(1:16)
    lowerbudget!(fd, m1.reserved)
    ballast = MantleArray(fd, UInt8, 3 * 2^23)
    @test Mantle.mapped(fd) < m1.mapped
    free!(ballast)
    memory(fd)
    fill!(out, Int32(0))
    run!(g)
    @test Array(out) == Int32.(1:16)
    @test Mantle.mapped(fd) >= 2^25
end

# I:1933: pressure (budget lowered to what is reserved [FI-9]) while g's host function runs: the running plan's pages stay mapped (the allocation gets no room from them: OutOfDeviceMemory, or room from elsewhere), and stage 2 reads stage 1's 32 MiB transient exactly.
@case "ARENA-04" 3 :sparse begin
    fd = FaultDevice(testdevice()); n = 2^23
    gate = Gate()
    g = Graph(fd)
    t = MantleArray(g, Int32, n)
    h = MantleArray(g, Int32, 1)
    out = MantleArray(fd, Int32, 16)
    dispatch!(g, arena_iota!, (t,), n)
    dispatch!(g, arena_iota!, (h,), 1)
    dispatch!(g, blockinghost(gate), (h,); writes = ())
    dispatch!(g, arena_add!, (out, t, Int32(0)), 16)
    task = Threads.@spawn run!(g)
    waitentered(gate)
    m1 = Mantle.mapped(fd)
    lowerbudget!(fd, Mantle.reserved(fd))
    e = thrown(() -> MantleArray(fd, UInt8, 3 * 2^23))
    @test e === nothing || e isa Mantle.OutOfDeviceMemory
    @test Mantle.mapped(fd) == m1
    letgo!(gate)
    @test thrownby(task) === nothing
    @test Array(out) == Int32.(1:16)
end

# I:1990-2004: g's run allocates its unmapped pages before its locks while another task's allocations keep forcing pressure (budget lowered [FI-9]), so the gap g binds can widen in between (Again, then a retry). The interleaving cannot be forced through the hooks: 2000 runs against a task that allocates and frees 12 MiB in a loop. Every run that gets memory is exact (an OutOfDeviceMemory is an allowed outcome when both need memory at once); after both stop and g is freed, mapped is back at the baseline (no stray pages).
@case "ARENA-05" 2 :sparse :long begin
    fd = FaultDevice(testdevice())
    base = memory(fd)
    g, out = peakgraph(fd, 2^22)
    run!(g)
    m1 = memory(fd)
    lowerbudget!(fd, m1.reserved + 2^23)
    stop = Threads.Atomic{Bool}(false)
    presser = Threads.@spawn while !stop[]
        freeresult!(outcome(() -> MantleArray(fd, UInt8, 3 * 2^22)))
    end
    results = [outcome(() -> (fill!(out, Int32(0)); run!(g); Array(out))) for _ in 1:2000]
    stop[] = true
    @test thrownby(presser; timeout = 120) === nothing
    @test all(r -> r == Int32.(1:16) || r isa Mantle.OutOfDeviceMemory, results)
    @test count(==(Int32.(1:16)), results) > 0
    free!(g)
    @test memory(fd).mapped == base.mapped
end

# P:1356-1358: caps report no sparse residency [FI-10]; a graph compiled after the first needs more than the first part holds: it gets a new part (reserved grows by its need) and the first graph's addresses stay (its next run neither recompiles nor records, and is exact).
@case "ARENA-06" 2 begin
    inner = testdevice()
    fd = FaultDevice(inner)
    inject!(fd, :caps, Answer(1, withcaps(caps(inner); sparseresidency = false)))
    small, outs = peakgraph(fd, 2^20)
    run!(small)
    r1 = memory(fd).reserved
    big, outb = peakgraph(fd, 2^24)
    run!(big)
    r2 = memory(fd).reserved
    @test r2 - r1 >= 2^25
    fill!(outs, Int32(0))
    _, d = counted(() -> run!(small), fd)
    @test d.plancompiles == 0
    @test d.records == 0
    @test Array(outs) == Int32.(1:16)
    @test Array(outb) == Int32.(1:16)
end

# I:1059-1065: 50 recompiles that switch g between a plan with a 64 MiB transient (an insert! block) and one without: each old plan's tenancy is dropped where the new plan has none. Tenancy has no hook; its consequence is checked: after free!(g), reserved and mapped are back at the baseline.
@case "ARENA-07" 2 :sparse :long begin
    dev = testdevice(); n = 2^24
    base = memory(dev)
    g, out = peakgraph(dev, 2^16)
    for _ in 1:50
        h = insert!(g) do
            t = MantleArray(g, Int32, n)
            dispatch!(g, arena_iota!, (t,), n)
            dispatch!(g, arena_add!, (out, t, Int32(1)), 16)
        end
        run!(g)
        @test Array(out) == Int32.(2:17)
        delete!(g, h)
        run!(g)
        @test Array(out) == Int32.(1:16)
    end
    free!(g)
    m = memory(dev)
    @test m.reserved <= base.reserved + 2^21
    @test m.mapped <= base.mapped
end

# A:382: g1 and g2 share the device part, each run 10 000 times on its own thread; g1 holds the part across a host node between its kernels. Every run adds its transient (all 1 in g1, all 2 in g2) into its own device accumulator, so a transient another run clobbered shows in the sums; no deadlock.
@case "ARENA-08" 3 :long begin
    dev = testdevice(); n = 4096; runs = 10_000
    acc1 = MantleArray(dev, Int32, n)
    acc2 = MantleArray(dev, Int32, n)
    fill!(acc1, Int32(0))
    fill!(acc2, Int32(0))
    g1 = Graph(dev)
    t1 = MantleArray(g1, Int32, n)
    h1 = MantleArray(g1, Int32, 1)
    z1 = MantleArray(g1, Int32, 1)
    dispatch!(g1, arena_set!, (t1, Int32(1)), n)
    dispatch!(g1, arena_add!, (h1, t1, Int32(0)), 1)
    dispatch!(g1, HostCall((x, z) -> (z[1] = x[1] - x[1])), (h1, z1); writes = (z1,))
    dispatch!(g1, arena_accumulate!, (acc1, t1, z1), n)
    g2 = Graph(dev)
    t2 = MantleArray(g2, Int32, n)
    z2 = MantleArray(g2, Int32, 1)
    dispatch!(g2, arena_set!, (t2, Int32(2)), n)
    dispatch!(g2, arena_set!, (z2, Int32(0)), 1)
    dispatch!(g2, arena_accumulate!, (acc2, t2, z2), n)
    w1 = Threads.@spawn foreach(_ -> run!(g1), 1:runs)
    w2 = Threads.@spawn foreach(_ -> run!(g2), 1:runs)
    @test thrownby(w1; timeout = 1800) === nothing
    @test thrownby(w2; timeout = 1800) === nothing
    @test all(==(Int32(runs)), Array(acc1))
    @test all(==(Int32(2runs)), Array(acc2))
end

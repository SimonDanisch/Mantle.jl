# Ordering between graphs (catalogue-core.md, table 11: ORD). Between graphs a
# stage that writes a resource another plan names waits for that plan's
# complete last run; a stage that finds a resource in its inbox waits for the
# last runs of its writers (recording-plan.md, "Ordering across graphs and
# queues"; internals.jl section 8; decisions.md B3.14, B3.15).
#
# Every ordering case makes the correct value depend on the wait: the writer
# is slow (slowwrite!, which writes after a spin) or the reader spins before
# it reads, and the other side is fast, so a missing wait reads or leaves the
# wrong value. Kernels are compiled by a first run before the measured one, so
# a compile never stands in for a wait.
#
# Doc keys as in the catalogue: A api.md, P recording-plan.md, I internals.jl,
# R resizing-and-raytracing.jl, D experiments/plan-review/decisions.md.
# A second queue: Graph(dev; priority = 1) (the plan's open decision 12 leaves
# the keyword and its values open). A result read for a timing sits in
# Readback memory (api.md 2.2, name open): its read waits for the writer and
# submits nothing on another queue.

using KernelAbstractions: @kernel, @index

"""Elements of the arrays the repeated cases bump and sum: enough GPU work per run to overlap."""
const ORDN = 1 << 20

@kernel function ord_bump!(a)                # a[i] += 1: a read and a write of a
    i = @index(Global)
    @inbounds a[i] += Int32(1)
end

@kernel function ord_accumulate!(sums, src)  # sums[i] += src[i]: a read of src
    i = @index(Global)
    @inbounds sums[i] += src[i]
end

@kernel function ord_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function ord_addone!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i] + Int32(1)
end

@kernel function ord_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@kernel function ord_set!(a, v)              # a[1] = v
    @inbounds a[1] = v
end

@kernel function ord_readfirst!(out, src)    # out[1] = src[1]
    @inbounds out[1] = src[1]
end

@kernel function ord_sum!(out, a, t)         # out[1] = a[1] + t[1]
    @inbounds out[1] = a[1] + t[1]
end

# Spins, then out[1] = src[1]. The index depends on the spin, so the load is
# not issued before it (src has two elements).
@kernel function ord_slowread!(out, src, iters)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    @inbounds out[1] = src[Int32(1) + (acc == Int32(-7) ? Int32(1) : Int32(0))]
end

"""Spin iterations for about `ms` milliseconds on the device under test."""
ord_spin(ms) = Int32(min(calibratedspin(testdevice(), ms), typemax(Int32)))

"""A graph; on the second queue when `second` (priority, open decision 12 of the plan)."""
ord_graph(dev, second::Bool; kw...) = second ? Graph(dev; priority = 1, kw...) : Graph(dev; kw...)

"""A one-element result in Readback memory: reading it waits for its writer only."""
ord_readback(dev) = MantleArray(dev, Int32, 1; memory = Mantle.Readback())

"""The channels that submitted something, from a Diagnostics."""
ord_channels(d) = count(>(0), values(d.submissions))

ord_median(x) = sort(x)[cld(length(x), 2)]

"""A host function: dst[1] = src[1]."""
ord_hostcopy(src, dst) = (dst[1] = src[1]; nothing)

"""
Host time of one `run!` per submission of a graph that reads `k` device
arrays a second graph also names (so each is shared), nothing changed between
runs; and the host waits of those runs.
"""
function ord_hosttime(dev, k; runs = 30)
    arrays = [MantleArray(dev, Int32[i, 0]) for i in 1:k]
    g, other = Graph(dev), Graph(dev)
    t, u = MantleArray(g, Int32, 1), MantleArray(other, Int32, 1)
    for a in arrays
        dispatch!(g, ord_readfirst!, (t, a), 1)
        dispatch!(other, ord_readfirst!, (u, a), 1)
    end
    run!(other); run!(g); run!(g)        # g's second run handles what registration put into its inbox
    Mantle.waitidle(dev)
    times, d = counted(() -> [@elapsed(run!(g)) for _ in 1:runs])
    free!(g); free!(other); foreach(free!, arrays)
    return (; persubmission = ord_median(times) * runs / submitcount(d), hostwaits = d.hostwaits)
end

"""
The two graphs of ORD-08: g1 copies s into its transient and writes s + 1
back (a read, then a write of s); g2 copies s into its transient and adds it
into sums. Their transients share an arena part unless core separates them
for a second queue.
"""
function ord_crossgraphs(dev, s, sums, second; kw...)
    n = length(s)
    g1 = Graph(dev; kw...)
    t1 = MantleArray(g1, Int32, n)
    dispatch!(g1, ord_copy!, (t1, s), n)
    dispatch!(g1, ord_addone!, (s, t1), n)
    g2 = ord_graph(dev, second; kw...)
    t2 = MantleArray(g2, Int32, n)
    dispatch!(g2, ord_copy!, (t2, s), n)
    dispatch!(g2, ord_accumulate!, (sums, t2), n)
    return g1, g2
end

# ── 11. Ordering between graphs (ORD) ──

# P:587-606, D:225-243: P bumps a, C adds a into sums, 10^4 rounds back to back with no host wait, on one queue and on two; every C sees exactly the P before it.
@case "ORD-01" 3 :long begin
    dev = testdevice()
    rounds = 10^4
    for second in (false, true)
        second && !has(dev, :multiqueue) && continue
        a, sums = MantleArray(dev, zeros(Int32, ORDN)), MantleArray(dev, zeros(Int32, ORDN))
        p = Graph(dev); dispatch!(p, ord_bump!, (a,), ORDN)
        c = ord_graph(dev, second); dispatch!(c, ord_accumulate!, (sums, a), ORDN)
        run!(p); run!(c); Mantle.waitidle(dev)
        _, d = counted() do
            for _ in 2:rounds
                run!(p); run!(c)
            end
        end
        @test d.hostwaits == 0
        second && @test ord_channels(d) == 2
        @test all(==(rounds), Array(a))
        @test all(==(rounds * (rounds + 1) ÷ 2), Array(sums))
        free!(p); free!(c); free!(a); free!(sums)
    end
end

# P:595-597: C reads a slowly, then P writes a fast: P waits for C's whole last run, else C reads P's value.
@case "ORD-02" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, ord_slowread!, (out, a, ord_spin(400)), 1)
    p = Graph(dev); dispatch!(p, ord_set!, (a, Int32(7)), 1)
    run!(c); run!(p); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(1)), 2)
    run!(c); run!(p)
    @test Array(out) == [1]
    @test Array(a) == [7, 1]
end

# I:1545-1548: P1 writes a after a spin, P2 writes a at once, alternately three times: each P2 write lands after the P1 write before it.
@case "ORD-03" 3 begin
    dev = testdevice()
    a = MantleArray(dev, Int32[0])
    p1 = Graph(dev); slowwrite!(p1, a, 1; ms = 400)
    p2 = Graph(dev); dispatch!(p2, ord_set!, (a, Int32(2)), 1)
    run!(p1); run!(p2); Mantle.waitidle(dev)
    for _ in 1:3
        run!(p1); run!(p2)
        @test Array(a) == [2]
    end
end

# P:599-601: R1 and R2 only read a, each spinning 400 ms, on two queues: neither waits for the other, so both finish in about the time of one.
@case "ORD-04" 3 :multiqueue begin
    dev = testdevice()
    a = MantleArray(dev, Int32[3, 3])
    o1, o2 = ord_readback(dev), ord_readback(dev)
    spin = ord_spin(400)
    r1 = Graph(dev); dispatch!(r1, ord_slowread!, (o1, a, spin), 1)
    r2 = ord_graph(dev, true); dispatch!(r2, ord_slowread!, (o2, a, spin), 1)
    _, d = counted(() -> (run!(r1); run!(r2)))
    @test ord_channels(d) == 2
    Mantle.waitidle(dev)
    alone = @elapsed (run!(r1); Array(o1))
    both = @elapsed (run!(r1); run!(r2); Array(o1); Array(o2))
    @test both < 1.5alone
    @test Array(o1) == Array(o2) == [3]
end

# R:657-662: C is compiled and registered while P's slow write of a is in flight: registration puts a into C's inbox, so C's first run waits for P.
@case "ORD-05" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    dispatch!(dev, ord_readfirst!, (out, a), 1)       # C's kernel in the device's compiler cache
    p = Graph(dev); slowwrite!(p, a, 7; ms = 500)
    run!(p); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(0)), 2)
    run!(p)                                           # in flight
    c = Graph(dev); dispatch!(c, ord_readfirst!, (out, a), 1)
    run!(c)                                           # compiles, registers, runs
    @test Array(out) == [7]
end

# D:240-241: ten reader graphs of a register, run and are freed between runs of the writer P: P never recompiles, and reader k sees P's k-th run.
@case "ORD-06" 3 begin
    dev = testdevice(); n = 1024
    a = MantleArray(dev, zeros(Int32, n))
    p = Graph(dev); dispatch!(p, ord_bump!, (a,), n)
    run!(p); Mantle.waitidle(dev)
    outs = [MantleArray(dev, Int32[0]) for _ in 1:10]
    _, d = counted() do
        for out in outs
            c = Graph(dev); dispatch!(c, ord_readfirst!, (out, a), 1)
            run!(c); run!(p); free!(c)
        end
    end
    @test d.plancompiles == 10                        # the ten readers; P none
    @test map(o -> only(Array(o)), outs) == 1:10
    @test Array(a) == fill(Int32(11), n)
end

# R:667-671: P's plan writing a slowly is retired in flight (free!(P), then a recompile whose new plan no longer names a); C reads a next and waits for the retired plan's tokens.
@case "ORD-07" 3 begin
    dev = testdevice()
    a, b, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, ord_readfirst!, (out, a), 1)
    p = Graph(dev); slowwrite!(p, a, 7; ms = 500)
    run!(p); run!(c); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(0)), 2)
    run!(p); free!(p)                                 # freed in flight
    run!(c)
    @test Array(out) == [7]
    p = Graph(dev)
    h = insert!(p) do
        slowwrite!(p, a, 8; ms = 500)
    end
    dispatch!(p, ord_set!, (b, Int32(1)), 1)
    run!(p); run!(c); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(0)), 2)
    run!(p)                                           # in flight: writes 8 into a
    delete!(p, h); run!(p)                            # recompiled: the new plan names b only
    run!(c)
    @test Array(out) == [8]
end

# A:458, P:1914-1918: test_crossplan_order. g1 reads then writes s, g2 copies s and sums it, sharing s and an arena part, 10^4 rounds on one queue and on two: exact. Negative control Unordered(): wrong on two queues.
@case "ORD-08" 3 :long begin
    dev = testdevice()
    rounds = 10^4
    want = rounds * (rounds + 1) ÷ 2
    for second in (false, true), control in (false, true)
        second && !has(dev, :multiqueue) && continue
        s, sums = MantleArray(dev, zeros(Int32, ORDN)), MantleArray(dev, zeros(Int32, ORDN))
        kw = control ? (; ordering = Mantle.Unordered()) : (;)
        g1, g2 = ord_crossgraphs(dev, s, sums, second; kw...)
        for _ in 1:rounds
            run!(g1); run!(g2)
        end
        exact = all(==(rounds), Array(s)) && all(==(want), Array(sums))
        if !control
            @test exact
        elseif second
            @test !exact          # one queue may serialize the submissions anyway: asserted on two only
        end
        free!(g1); free!(g2); free!(s); free!(sums)
    end
end

# I:1613-1616: a graph whose stage writes nothing another plan names and whose inbox is empty: its host time per run does not grow from 10 to 2 000 named (shared, only read) arrays.
@case "ORD-09" 3 begin
    dev = testdevice()
    small, large = ord_hosttime(dev, 10), ord_hosttime(dev, 2000)
    @test small.hostwaits == 0 && large.hostwaits == 0
    @test large.persubmission < 2small.persubmission
end

# A:466, P:1934-1937: test_run_host_time. Plans naming 100, 2 500, 25 000 and 50 000 shared arrays, nothing changed: host time per run flat, the largest within 3x of the smallest.
@case "ORD-10" 3 :long begin
    dev = testdevice()
    times = [ord_hosttime(dev, k).persubmission for k in (100, 2_500, 25_000, 50_000)]
    @test maximum(times) < 3first(times)
end

# I:1606-1613: P writes a slowly; C's stage 1 takes a from its inbox without using it, stage 3 (after two host nodes) reads it: the wait found by stage 1 is kept for stage 3.
@case "ORD-11" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    p = Graph(dev); slowwrite!(p, a, 7; ms = 500)
    c = Graph(dev)
    t1, t2, t3, t4 = (MantleArray(c, Int32, 1) for _ in 1:4)
    dispatch!(c, ord_set!, (t1, Int32(0)), 1)                              # stage 1: does not use a
    dispatch!(c, HostCall(ord_hostcopy), (t1, t2); writes = (t2,))
    dispatch!(c, ord_copy!, (t3, t2), 1)                                   # stage 2
    dispatch!(c, HostCall(ord_hostcopy), (t3, t4); writes = (t4,))
    dispatch!(c, ord_sum!, (out, a, t4), 1)                                # stage 3: out = a[1] + 0
    run!(p); run!(c); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(0)), 2)
    run!(p); run!(c)
    @test Array(out) == [7]
end

# I:1213-1239: reader graphs register (40 compiles) on one task while P bumps a 200 times on another: P never recompiles, no run is lost, and each reader copies one whole state of a.
@case "ORD-12" 3 begin
    dev = testdevice()
    rounds, readers = 200, 40
    a = MantleArray(dev, zeros(Int32, ORDN))
    p = Graph(dev); dispatch!(p, ord_bump!, (a,), ORDN)
    run!(p); Mantle.waitidle(dev)
    outs = [MantleArray(dev, Int32, ORDN) for _ in 1:readers]
    Mantle.resetdiagnostics!(dev)
    writer = Threads.@spawn for _ in 1:rounds
        run!(p)
    end
    for out in outs
        c = Graph(dev); dispatch!(c, ord_copy!, (out, a), ORDN)
        run!(c); free!(c)
    end
    withtimeout(() -> wait(writer), 120)
    @test Mantle.diagnostics(dev).plancompiles == readers
    @test all(==(rounds + 1), Array(a))
    @test all(o -> allequal(Array(o)), outs)
end

# P:647-651: gx writes x from y, gy (second queue) writes y from x, 10^4 rounds: no GPU deadlock and exact (validation is the session's: the suite runs with the validation layers on).
@case "ORD-13" 3 :multiqueue :long begin
    dev = testdevice()
    rounds = 10^4
    x, y = MantleArray(dev, zeros(Int32, ORDN)), MantleArray(dev, zeros(Int32, ORDN))
    gx = Graph(dev); dispatch!(gx, ord_addone!, (x, y), ORDN)              # x = y + 1
    gy = ord_graph(dev, true); dispatch!(gy, ord_addone!, (y, x), ORDN)    # y = x + 1
    withtimeout(1200) do
        for _ in 1:rounds
            run!(gx); run!(gy)
        end
        Mantle.waitidle(dev)
    end
    @test all(==(2rounds - 1), Array(x))
    @test all(==(2rounds), Array(y))
end

# D:244-254 + D:225-243: delete! drops g's only (slow) read of a while g's old plan is in flight: P's write of a waits for it; after g recompiled, P no longer waits for g (timed where there are two queues).
@case "ORD-14" 3 begin
    dev = testdevice()
    mq = has(dev, :multiqueue)
    a, out, z = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32[0]), MantleArray(dev, Int32[0])
    pdone = ord_readback(dev)
    g = ord_graph(dev, mq)
    h = insert!(g) do
        dispatch!(g, ord_slowread!, (out, a, ord_spin(500)), 1)
    end
    slowwrite!(g, z, 1; ms = 500)                     # g's node outside the block: its later runs stay busy
    p = Graph(dev)
    dispatch!(p, ord_set!, (a, Int32(7)), 1)
    dispatch!(p, ord_set!, (pdone, Int32(1)), 1)
    run!(g); run!(p); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(1)), 2)
    run!(g)                                           # in flight: reads a after its spin
    delete!(g, h)                                     # the old plan names a until it retires
    run!(p)
    @test Array(out) == [1]
    run!(g); Mantle.waitidle(dev)                     # recompiled: the old plan retired
    if mq
        run!(g)                                       # 500 ms on z, on the other queue
        @test (@elapsed (run!(p); Array(pdone))) < 0.25
    end
end

# I:1649-1655: an eager slow write of a puts a into C's inbox; C's stage 1 does not use a, its stage 2 (after a host node) reads it: the item is waited for with C's use and kept for stage 2.
@case "ORD-15" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    c = Graph(dev)
    t1, t2 = MantleArray(c, Int32, 1), MantleArray(c, Int32, 1)
    dispatch!(c, ord_set!, (t1, Int32(0)), 1)
    dispatch!(c, HostCall(ord_hostcopy), (t1, t2); writes = (t2,))
    dispatch!(c, ord_sum!, (out, a, t2), 1)
    run!(c); Mantle.waitidle(dev)
    slowwrite!(dev, a, 9; ms = 500)                   # eager, in flight
    run!(c)
    @test Array(out) == [9]
end

# P:572-573, P:2316-2317: a priority frame graph beside a background graph busy for 500 ms: the frame's result is ready without waiting for the background (open decision 12 of the plan).
@case "ORD-16" 3 :multiqueue begin
    dev = testdevice()
    w, src, fout = MantleArray(dev, Int32[0]), MantleArray(dev, Int32[5, 5]), ord_readback(dev)
    background = Graph(dev); slowwrite!(background, w, 1; ms = 500)
    frame = Graph(dev; priority = 1); dispatch!(frame, ord_readfirst!, (fout, src), 1)
    run!(background); run!(frame); Mantle.waitidle(dev)
    run!(background)
    @test (@elapsed (run!(frame); Array(fout))) < 0.25
    @test Array(fout) == [5]
end

# I:1629-1634: free!(g) while g's slow read of a is in flight: unregister! puts a into P's inbox with g's last tokens, so P's write of a waits for that run.
@case "ORD-17" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32[0])
    g = Graph(dev); dispatch!(g, ord_slowread!, (out, a, ord_spin(500)), 1)
    p = Graph(dev); dispatch!(p, ord_set!, (a, Int32(7)), 1)
    run!(g); run!(p); Mantle.waitidle(dev)
    dispatch!(dev, ord_fill!, (a, Int32(1)), 2)
    run!(g)                                           # in flight
    free!(g)
    run!(p)
    @test Array(out) == [1]
    @test Array(a) == [7, 1]
end

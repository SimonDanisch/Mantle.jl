# Inbox, ordering with eager and host work (catalogue-core.md, table 12:
# INBOX). What is not a run (eager calls, host reads, sparse binding, the
# steps applying pending changes) records its accesses on the resource's sync
# state and puts the resource into the inbox of every plan naming it; a run
# handles only its inbox (recording-plan.md, "Ordering across graphs and
# queues", the inbox; internals.jl section 8, "With the host").
#
# As in ordering.jl, the correct value depends on the wait: one side spins
# before it writes or reads, the other is fast, and kernels are compiled by a
# first run before the measured one.
#
# Doc keys as in the catalogue: A api.md, P recording-plan.md, I internals.jl.
# A second queue: Graph(dev; priority = 1) (open decision 12 of the plan).

using KernelAbstractions: @kernel, @index

@kernel function inbox_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@kernel function inbox_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function inbox_readfirst!(out, src)  # out[1] = src[1]
    @inbounds out[1] = src[1]
end

@kernel function inbox_readat!(out, src, k)  # out[1] = src[k]
    @inbounds out[1] = src[k]
end

# Spins, then out[1] = src[1]; the index depends on the spin (src has two elements).
@kernel function inbox_slowread!(out, src, iters)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    @inbounds out[1] = src[Int32(1) + (acc == Int32(-7) ? Int32(1) : Int32(0))]
end

# Spins, then a[1] = a[1] + 1: one segment that reads, then writes a.
@kernel function inbox_slowbump!(a, iters)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    @inbounds a[1] = a[Int32(1) + (acc == Int32(-7) ? Int32(1) : Int32(0))] + Int32(1)
end

inbox_spin(ms) = Int32(min(calibratedspin(testdevice(), ms), typemax(Int32)))

inbox_median(x) = sort(x)[cld(length(x), 2)]

"""
Host time of one `run!` per submission of a graph reading `k` device arrays,
with an eager write into one of them before each run (one inbox item).
"""
function inbox_hosttime(dev, k; runs = 30)
    arrays = [MantleArray(dev, Int32[i, 0]) for i in 1:k]
    f = Graph(dev); t = MantleArray(f, Int32, 1)
    foreach(a -> dispatch!(f, inbox_readfirst!, (t, a), 1), arrays)
    run!(f); run!(f); Mantle.waitidle(dev)
    times, subs = Float64[], 0
    for v in 1:runs
        dispatch!(dev, inbox_fill!, (first(arrays), Int32(v)), 2)
        dt, d = counted(() -> @elapsed(run!(f)))
        push!(times, dt); subs += submitcount(d)
    end
    free!(f); foreach(free!, arrays)
    return inbox_median(times) * runs / subs
end

# ── 12. Inbox, ordering with eager and host work (INBOX) ──

# P:622-631: run!(C) reads a after a spin, then an eager write of a: the eager call waits for C's last run, else C reads the new value.
@case "INBOX-01" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, inbox_slowread!, (out, a, inbox_spin(500)), 1)
    run!(c); dispatch!(dev, inbox_fill!, (a, Int32(1)), 2); Mantle.waitidle(dev)   # compiled
    run!(c)
    dispatch!(dev, inbox_fill!, (a, Int32(7)), 2)
    @test Array(out) == [1]
    @test Array(a) == [7, 7]
end

# P:610-614: an eager slow write of a, then run!(C) reading a: C waits for the eager call's token.
@case "INBOX-02" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, inbox_readfirst!, (out, a), 1)
    run!(c); slowwrite!(dev, a, 1; ms = 10); Mantle.waitidle(dev)   # compiled
    slowwrite!(dev, a, 7; ms = 500)
    run!(c)
    @test Array(out) == [7]
end

# P:617-618: a pending store into a, then run!(C): applied in front of C's first segment, in C's submission (no extra one), and C sees it.
@case "INBOX-03" 4 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, inbox_readfirst!, (out, a), 1)
    run!(c); Mantle.waitidle(dev)
    _, plain = counted(() -> run!(c))
    Mantle.waitidle(dev)
    a[1:1] = Int32[5]
    @test length(Mantle.pendingof(a)) == 1
    Mantle.steplog!(dev, false); Mantle.steplog!(dev, true)
    _, withstore = counted(() -> run!(c))
    steps = Mantle.steplog(dev); Mantle.steplog!(dev, false)
    @test submitcount(withstore) == submitcount(plain)
    @test count(s -> s.target === a, steps) == 1
    @test isempty(Mantle.pendingof(a))
    @test Array(out) == [5]
end

# P:622-631: an eager read of a while reader C (second queue) spins 500 ms reading a: read after read, so the eager call waits for nothing.
@case "INBOX-04" 3 :multiqueue begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[3, 3]), MantleArray(dev, Int32[0])
    b = MantleArray(dev, Int32, 2; memory = Mantle.Readback())
    c = Graph(dev; priority = 1); dispatch!(c, inbox_slowread!, (out, a, inbox_spin(500)), 1)
    run!(c); dispatch!(dev, inbox_copy!, (b, a), 2); Mantle.waitidle(dev)   # compiled
    run!(c)
    @test (@elapsed (dispatch!(dev, inbox_copy!, (b, a), 2); Array(b))) < 0.25
    @test Array(b) == [3, 3]
end

# I:1697-1708: one eager segment reads, then writes a (after a spin): it is recorded as a writer, so reader C waits for it.
@case "INBOX-05" 3 begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, inbox_readfirst!, (out, a), 1)
    run!(c); dispatch!(dev, inbox_slowbump!, (MantleArray(dev, Int32[0, 0]), Int32(1)), 1)
    Mantle.waitidle(dev)                                                    # compiled
    dispatch!(dev, inbox_slowbump!, (a, inbox_spin(500)), 1)                # a[1] = 2, after 500 ms
    run!(c)
    @test Array(out) == [2]
end

# P:607-610: a resize moves a into a sparse reserve, whose pages the applying submission binds, and a store lands there; C reads it: C's segment waits for the bind token, else it reads unbound memory. 20 rounds, a fresh array each; then growth within the reserve (pages only).
@case "INBOX-06" 4 :sparse begin
    dev = testdevice()
    n = 1 << 20                     # 4 MiB of Int32: a reserve (core's sparsethreshold is a few sparse pages)
    seen = map(1:20) do r
        a, out = MantleArray(dev, Int32, 1024), MantleArray(dev, Int32[0])
        k = GPURef(dev, Int32(1))
        c = Graph(dev); dispatch!(c, inbox_readat!, (out, a, k), 1)
        run!(c)
        resize!(a, n); a[n:n] = Int32[r]; k[] = Int32(n)
        run!(c)                     # its prefix: BindPages, MoveStorage, CellWrite, the store
        moved = only(Array(out))
        resize!(a, 2n); a[2n:2n] = Int32[-r]; k[] = Int32(2n)
        run!(c)                     # within the reserve: BindPages, CellWrite, the store
        grown = only(Array(out))
        free!(c); free!(a)
        (moved, grown)
    end
    @test seen == [(r, -r) for r in 1:20]
end

# P:619-621: 10^4 eager writes of a between two runs of C: C's next run handles them in less than a tenth of their own host time (inbox dedupe unspecified) and sees the last.
@case "INBOX-07" 3 :long begin
    dev = testdevice()
    a, out = MantleArray(dev, Int32[0, 0]), MantleArray(dev, Int32[0])
    c = Graph(dev); dispatch!(c, inbox_readfirst!, (out, a), 1)
    run!(c); dispatch!(dev, inbox_fill!, (a, Int32(1)), 2); run!(c); Mantle.waitidle(dev)
    eager = @elapsed for v in 1:10^4
        dispatch!(dev, inbox_fill!, (a, Int32(v)), 2)
    end
    many = @elapsed run!(c)
    @test many < eager / 10
    @test Array(out) == [10^4]
end

# A:466: a frame naming 5 000 arrays with one of them written before each run: its host time per run is that of a frame naming 50 (it follows the inbox, not the names).
@case "INBOX-08" 3 begin
    dev = testdevice()
    small, large = inbox_hosttime(dev, 50), inbox_hosttime(dev, 5000)
    @test large < 2small
end

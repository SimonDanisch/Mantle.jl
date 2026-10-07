# Scheduling: a call changes nothing on the device (catalogue-resize-rt.md,
# table A: SC). resize!, stores and copy! only enqueue at the call (Rule 8,
# decisions.md B3.13; a store's one copy into an upload region, B3.18); the
# next submission that touches the array applies them in front of its own
# work, in sequence order (resizing-and-raytracing.jl B.3, B.4).
#
# Doc keys as in the catalogue: PLAN recording-plan.md, RR
# resizing-and-raytracing.jl, API api.md, INT internals.jl, DEC decisions.md.
# The step log (Mantle.steplog): entries with `kind` (a Symbol: :InlineStore,
# :StagedStore, :MoveStorage, :CellWrite, :BindPages, ...), `target` (the root
# array the user holds), `seq` and `token`, in emission order.

using KernelAbstractions: @kernel, @index

@kernel function sc_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function sc_firsts!(out, a, b)       # out = [a[1], b[1]]
    @inbounds out[1] = a[1]
    @inbounds out[2] = b[1]
end

# Stage 1 of a gated graph: dst[i] = src[i], and t[1] = 0 for the host node to read.
@kernel function sc_copyfirst!(dst, src, t)
    i = @index(Global)
    @inbounds dst[i] = src[i]
    if i == 1
        @inbounds t[1] = Int32(0)
    end
end

# Stage 2: dst[i] = src[i] + t[1]; reading t keeps it after the host node.
@kernel function sc_copythen!(dst, src, t)
    i = @index(Global)
    @inbounds dst[i] = src[i] + t[1]
end

# Spins, then copies src[1:4] into dst; the indices depend on the spin.
@kernel function sc_slowcopy4!(dst, src, iters)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    base = acc == Int32(-7) ? Int32(1) : Int32(0)
    for j in Int32(1):Int32(4)
        @inbounds dst[j] = src[j + base]
    end
end

# Spins, then out = [length(a), sum(a)]; the reads depend on the spin.
@kernel function sc_slowsum!(out, a, iters)
    acc = Int32(0)
    for k in Int32(1):iters
        acc = acc * Int32(1103515245) + k
    end
    base = acc == Int32(-7) ? Int32(1) : Int32(0)
    n = Int32(length(a))
    s = Int32(0)
    for j in Int32(1):n
        @inbounds s += a[min(j + base, n)]
    end
    @inbounds out[1] = n
    @inbounds out[2] = s
end

# out = [length(a), Σ i * a[i]] (wrapping Int32).
@kernel function sc_lengthsum!(out, a)
    n = Int32(length(a))
    s = Int32(0)
    for j in Int32(1):n
        @inbounds s += j * a[j]
    end
    @inbounds out[1] = n
    @inbounds out[2] = s
end

sc_spin(ms) = Int32(min(calibratedspin(testdevice(), ms), typemax(Int32)))

"""The (length, checksum) sc_lengthsum! writes for host data `p`."""
sc_checksum(p) = (Int32(length(p)), foldl((s, j) -> s + Int32(j) * p[j], eachindex(p); init = Int32(0)))

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function sc_steps(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    steps = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, steps
end

"""The steps that changed `a`."""
sc_on(steps, a) = [s for s in steps if s.target === a]

# ── A. Scheduling: a call changes nothing on the device (Rule 8) ──

# PLAN:59-71, RR:238-247, DEC:208-216: resize!(a, 10^8) of a fixed 1000-element array only enqueues: length(a) is 10^8; reserved, mapped, tokens unchanged; nothing allocated; pending == [Resize].
@case "SC-01" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 1000)
    m = memory(dev)
    _, d = counted(() -> resize!(a, 10^8))
    @test length(a) == 10^8
    @test (Mantle.reserved(dev), Mantle.mapped(dev)) == (m.reserved, m.mapped)
    @test submitcount(d) == 0 && d.allocations == 0 && d.hostwaits == 0
    @test only(Mantle.pendingof(a)) isa Mantle.Resize
    free!(a)                                          # the last hold drops the Resize: 400 MB never allocated
end

# RR:244-251, DEC:269-277: a[1:1024] = x (4 KiB): one upload region live, device reserved unchanged, no token, pending == [Store]; the region is retired once applied.
@case "SC-02" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 4096)
    a[1:4] = Int32[1, 2, 3, 4]; Array(a)              # an upload block exists
    x = testdata(Int32, 1024)
    m = memory(dev)
    _, d = counted(() -> (a[1:1024] = x))
    @test Mantle.liveuploads(dev) == m.uploads + 1
    @test Mantle.reserved(dev) == m.reserved
    @test submitcount(d) == 0 && d.hostwaits == 0
    @test only(Mantle.pendingof(a)) isa Mantle.Store
    @test Array(a)[1:1024] == x
    @test memory(dev).uploads == m.uploads
end

# RR:18-20, PLAN:69-70: eager K reads a[1:4] after a 500 ms spin; then a[1:4] = y; then g2 reads a: K sees the old values, g2 sees y, applied in g2's one submission.
@case "SC-03" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32.(1:16))
    outk, outg = MantleArray(dev, Int32, 4), MantleArray(dev, Int32, 4)
    g2 = Graph(dev); dispatch!(g2, sc_copy!, (outg, a), 4)
    run!(g2); dispatch!(dev, sc_slowcopy4!, (outk, a, Int32(1)), 1); Mantle.waitidle(dev)   # compiled
    dispatch!(dev, sc_slowcopy4!, (outk, a, sc_spin(500)), 1)                            # K, in flight
    y = Int32[-1, -2, -3, -4]
    a[1:4] = y
    (_, d), steps = sc_steps(() -> counted(() -> run!(g2)))
    @test submitcount(d) == 1
    @test length(sc_on(steps, a)) == 1
    @test Array(outk) == 1:4
    @test Array(outg) == y
end

# PLAN:69-70, RR:544-548: run1 (g1) sums a (n elements) after a spin; thread B resizes a to 2n (a move) and stores; run2 (g2) applies both: run1 saw n and the old data, run2 2n and the new; the old region is not reused before run1 ends (canary).
@case "SC-04" 4 begin
    dev = testdevice(); n = 1024
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, Int32, 2), MantleArray(dev, Int32, 2)
    spin = sc_spin(500)
    g1 = Graph(dev); dispatch!(g1, sc_slowsum!, (out1, a, spin), 1)
    g2 = Graph(dev); dispatch!(g2, sc_slowsum!, (out2, a, spin), 1)
    run!(g1); run!(g2); Mantle.waitidle(dev)
    run!(g1)                                          # run1, in flight
    fetch(Threads.@spawn begin
        resize!(a, 2n)
        a[n+1:2n] = Int32.(n+1:2n)
    end)
    run!(g2)                                          # run2: the move and the store in its prefix
    c = MantleArray(dev, Int32, n)                    # would take the old region if it were released before run1's token
    dispatch!(dev, sc_copy!, (c, MantleArray(dev, fill(Int32(-1), n))), n)
    @test Array(out1) == [n, sum(1:n)]
    @test Array(out2) == [2n, sum(1:2n)]
end

# API:452: thread A runs g 10^4 times, reading (length(a), checksum) after each; thread B alternates copy!(a, p_i) over three patterns: every pair read is one whole p_i.
@case "SC-05" 4 :long begin
    dev = testdevice(); rounds = 10^4
    patterns = [testdata(Int32, k; seed = k) .% Int32(1000) for k in (37, 100, 1000)]
    want = Set(sc_checksum(p) for p in patterns)
    a, out = MantleArray(dev, patterns[1]), MantleArray(dev, Int32, 2)
    g = Graph(dev); dispatch!(g, sc_lengthsum!, (out, a), 1)
    reader = Threads.@spawn [(run!(g); Tuple(Array(out))) for _ in 1:rounds]
    writer = Threads.@spawn for k in 1:rounds
        copy!(a, patterns[mod1(k, 3)])
    end
    seen = withtimeout(() -> (wait(writer); fetch(reader)), 1800)
    @test all(in(want), seen)
end

# RR:705-712, DEC:213-214: g is held between stages (its host node blocks) and launches directly over fixed a; thread B calls resize!(a, 2n), a[1:2] = x, copy!(b, y): each returns at once, with no host wait and no submission.
@case "SC-06" 4 begin
    dev = testdevice(); n = 1024
    a, b = MantleArray(dev, Int32.(1:n)), MantleArray(dev, Int32, n)
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, sc_copyfirst!, (out1, a, t), launchrange(a))
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    dispatch!(g, sc_copythen!, (out2, a, t), launchrange(a))
    running = Threads.@spawn run!(g)
    waitentered(gate)
    took, d = counted(() -> @elapsed (resize!(a, 2n); a[1:2] = Int32[7, 8]; copy!(b, Int32.(1:n))))
    @test d.hostwaits == 0 && submitcount(d) == 0
    @test took < 0.1
    letgo!(gate)
    withtimeout(() -> wait(running), 30)
    @test Array(b) == 1:n
end

# API:80: resize!(a); then a run of g2, which does not name a, and Array(b): a's Resize stays pending and no step on a is in either submission.
@case "SC-07" 4 begin
    dev = testdevice()
    a, b, out = MantleArray(dev, Int32, 1024), MantleArray(dev, Int32.(1:16)), MantleArray(dev, Int32, 16)
    g2 = Graph(dev); dispatch!(g2, sc_copy!, (out, b), 16)
    run!(g2); Mantle.waitidle(dev)
    resize!(a, 4096)
    _, steps = sc_steps(() -> (run!(g2); Array(b)))
    @test isempty(sc_on(steps, a))
    @test only(Mantle.pendingof(a)) isa Mantle.Resize
    @test Array(out) == 1:16
end

# PLAN:66-67, INT:2117-2131: three stores pending on a, then one eager dispatch!(dev, k, (out, a)): one submission, the three stores (inline where the device has an inline limit) in sequence order before the dispatch; out reflects them.
@case "SC-08" 4 begin
    dev = testdevice()
    a, out = MantleArray(dev, zeros(Int32, 16)), MantleArray(dev, Int32, 4)
    dispatch!(dev, sc_copy!, (out, a), 4); Mantle.waitidle(dev)          # compiled
    a[1:1] = Int32[10]; a[2:2] = Int32[20]; a[3:3] = Int32[30]
    (_, d), steps = sc_steps(() -> counted(() -> dispatch!(dev, sc_copy!, (out, a), 4)))
    stores = sc_on(steps, a)
    kind = caps(dev).inlinestore > 0 ? :InlineStore : :StagedStore
    @test submitcount(d) == 1
    @test [s.kind for s in stores] == [kind, kind, kind]
    @test issorted([s.seq for s in stores])
    @test Array(out) == [10, 20, 30, 0]
end

# API:483, INT:1003-1004: window minimized; g names a, which has a pending store; run!(g) returns Skipped and submits nothing, the store stays pending; Array(a) applies it in its own submission.
@case "SC-09" 7 :window begin   # pending decision 4: Skipped() is public
    dev = testdevice()
    win = Window(dev, 320, 240)
    a, out = MantleArray(dev, Int32[1, 2]), MantleArray(dev, Int32, 1)
    g = Graph(dev)
    dispatch!(g, sc_copy!, (out, a), 1)
    render!(g, win => Clear((0f0, 0f0, 0f0, 1f0))) do p end
    run!(g); Mantle.waitidle(dev)
    a[1:1] = Int32[9]
    resize!(win, 0, 0)                                # minimized
    r, d = counted(() -> run!(g))
    @test r isa Mantle.Skipped
    @test submitcount(d) == 0
    @test only(Mantle.pendingof(a)) isa Mantle.Store
    v, d = counted(() -> Array(a))
    @test v == [9, 2]
    @test submitcount(d) == 1
    @test isempty(Mantle.pendingof(a))
end

# PLAN:66-68, RR:291-292,352-353: a[1:1] = x, b[1:1] = y, resize!(a, 64) (a move), then one eager kernel over both: the steps are emitted in seq order, a's Store, b's Store, then a's Resize steps.
@case "SC-10" 4 begin
    dev = testdevice()
    a, b, out = MantleArray(dev, zeros(Int32, 16)), MantleArray(dev, zeros(Int32, 16)), MantleArray(dev, Int32, 2)
    dispatch!(dev, sc_firsts!, (out, a, b), 1); Mantle.waitidle(dev)    # compiled
    a[1:1] = Int32[5]; b[1:1] = Int32[6]; resize!(a, 64)
    _, steps = sc_steps(() -> dispatch!(dev, sc_firsts!, (out, a, b), 1))
    ours = [s for s in steps if s.target === a || s.target === b]
    @test length(ours) >= 3
    @test issorted([s.seq for s in ours])
    @test [s.target === a for s in ours[1:2]] == [true, false]
    @test all(s -> s.target === a && s.kind !== :InlineStore && s.kind !== :StagedStore, ours[3:end])
    @test Array(out) == [5, 6]
    @test length(a) == 64
end

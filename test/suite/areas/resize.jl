# resize! and storage growth (catalogue-resize-rt.md, table B).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md; INT = docs/internals.jl (line numbers of 2026-10-05).
# What this file assumes of the test hooks (helpers.jl):
#   Mantle.pendingof(x)  the Change objects (Store, Resize, Rebind, BuildAccel)
#                        pending on x, in sequence order, with RR B.3's fields.
#   Mantle.steplog(dev)  entries with `kind` (the step's name as a Symbol:
#                        :Becomes, :CellWrite, :MoveStorage, :BindPages,
#                        :UnbindPages, :Restore, :InlineStore, :StagedStore),
#                        `target` (the root resource the user holds; a view's
#                        steps are its parent's), `seq` and `token`.
# A case marked VAL relies on validation layers and GPU-AV being on in the
# suite's device configuration.

using KernelAbstractions: @kernel, @index

@kernel function rz_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function rz_mark!(marks, a)
    i = @index(Global)
    marks[i] = Int32(1)
end

# Copies after reading `gate`, which slowwrite! writes: the copy waits for the spin.
@kernel function rz_copyafter!(dst, src, gate)
    i = @index(Global)
    if gate[1] != Int32(-7)
        dst[i] = src[i]
    end
end

@kernel function rz_readfirst!(out, a)
    out[1] = a[1]
end

@kernel function rz_writelast!(a)
    i = @index(Global)
    a[length(a) - 4 + i] = Float32(i)
end

@kernel function rz_readlast!(out, a)
    i = @index(Global)
    out[i] = a[length(a) - 4 + i]
end

"""The kinds of the changes pending on `x`, in sequence order."""
pendingkinds(x) = [nameof(typeof(c)) for c in Mantle.pendingof(x)]

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""The kinds of the steps in `log` that changed `x`, in emission order."""
stepkinds(log, x) = [s.kind for s in log if s.target === x]

"""How many work items a launch marked in `marks`."""
launchcount(marks) = count(==(Int32(1)), Array(marks))

"""An eager call reading `a[1]` into `s`: a submission that touches `a`, so it applies what is pending on it."""
applypending!(s, a) = dispatch!(location(s, a), rz_readfirst!, (s, a), 1)

"""An element count whose growth by 16x stays below the sparse threshold (4 sparse pages; RR:202)."""
smallcount(dev) = max(16, caps(dev).sparsepage ÷ 64)

"""The Float32 elements of `pages` sparse pages."""
pagecount(dev, pages) = pages * caps(dev).sparsepage ÷ sizeof(Float32)

"""The step a store of `bytes` at byte `offset` becomes (api.md:82): inline within the device type's limit and alignment, else staged."""
storekind(dev, offset, bytes) = (c = caps(dev);
    bytes <= c.inlinestore && offset % c.inlinealign == 0 && bytes % c.inlinealign == 0 ? :InlineStore : :StagedStore)

"""`c` (an immutable caps record) with the fields in `kw` replaced."""
withcaps(c; kw...) = typeof(c)((get(kw, f, getfield(c, f)) for f in fieldnames(typeof(c)))...)

"""A graph that spins about `ms` ms, then copies `n` elements of `src` into `dst`: a run that reads `src` late."""
function slowcopygraph(dev, dst, src, n; ms = 300)
    gate = MantleArray(dev, Int32, 1)
    g = Graph(dev)
    slowwrite!(g, gate, 1; ms)
    dispatch!(g, rz_copyafter!, (dst, src, gate), n)
    return g
end

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

# ── B. resize! and storage growth ──

# RZ-01 api.md:80, RR:190,560: fixed region of 1000; resize!(a, 500); the next submission adopts the region (Becomes, CellWrite), no move; a[1:500] intact.
@case "RZ-01" 4 begin
    dev = testdevice()
    x = testdata(Float32, 1000)
    a = MantleArray(dev, x)
    out = MantleArray(dev, Float32, 1000)
    resize!(a, 500)
    _, log = stepsduring(() -> dispatch!(dev, rz_copy!, (out, a), launchrange(a)), dev)
    @test stepkinds(log, a) == [:Becomes, :CellWrite]
    @test Array(out)[1:500] == x[1:500]
    @test Array(a) == x[1:500]
    foreach(free!, (a, out))
end

# RZ-02 api.md:80, RR:540-548,560,775: a recorded plan launches directly over fixed `a`; resize!(a, length(a)) adopts the storage and recompiles that plan once; later growth recompiles nothing.
@case "RZ-02" 4 begin
    dev = testdevice()
    n = 1000
    x = testdata(Float32, n)
    a = MantleArray(dev, x)
    out, marks = MantleArray(dev, Float32, 4n), MantleArray(dev, Int32, 4n)
    g = Graph(dev)
    dispatch!(g, rz_copy!, (out, a), launchrange(a))
    dispatch!(g, rz_mark!, (marks, a), launchrange(a))
    run!(g)
    resize!(a, n)
    (_, d), log = stepsduring(() -> counted(() -> run!(g), dev), dev)
    @test d.plancompiles == 1
    @test stepkinds(log, a) == [:Becomes, :CellWrite]
    @test Array(out)[1:n] == x
    resize!(a, 2n)
    fill!(marks, Int32(0))
    _, d2 = counted(() -> run!(g), dev)
    @test d2.plancompiles == 0
    @test launchcount(marks) == 2n
    free!(g); foreach(free!, (a, out, marks))
end

# RZ-03 api.md:80, RR:196-198,565: small fixed `a` grows to 3n: one move into a pool region of 2·need, applied by one submission; growth up to 6n writes only the cell, 6n+1 moves again.
@case "RZ-03" 4 begin
    dev = testdevice()
    n = smallcount(dev)
    x = testdata(Float32, n)
    a = MantleArray(dev, x)
    resize!(a, 3n)
    r, log = stepsduring(() -> Array(a), dev)
    @test stepkinds(log, a) == [:MoveStorage, :CellWrite]
    @test allequal(s.token for s in log if s.target === a)
    @test r[1:n] == x
    resize!(a, 6n)
    _, log2 = stepsduring(() -> Array(a), dev)
    @test stepkinds(log2, a) == [:CellWrite]
    resize!(a, 6n + 1)
    r3, log3 = stepsduring(() -> Array(a), dev)
    @test stepkinds(log3, a) == [:MoveStorage, :CellWrite]
    @test r3[1:n] == x
    free!(a)
end

# RZ-04 api.md:80, RR:196,203,561-564: a fixed array of 4 sparse pages grows to 2n: pages bound on the sparse queue before the move into a reserve of 4·need; growth to 4·need moves nothing.
@case "RZ-04" 4 :sparse begin
    dev = testdevice()
    n = pagecount(dev, 4)
    x = testdata(Float32, n)
    a = MantleArray(dev, x)
    resize!(a, 2n)
    r, log = stepsduring(() -> Array(a), dev)
    steps = [s for s in log if s.target === a]
    @test [s.kind for s in steps] == [:BindPages, :MoveStorage, :CellWrite]
    @test steps[1].token != steps[2].token          # the bind is its own sparse-queue submission; the move's waits on it
    @test r[1:n] == x
    resize!(a, 8n)
    r2, log2 = stepsduring(() -> Array(a), dev)
    @test :MoveStorage ∉ stepkinds(log2, a)
    @test r2[1:n] == x
    free!(a)
end

# RZ-05 RR:192: a reserve-backed array shrunk and grown back within its mapped pages: only a cell write, no bind, no move.
@case "RZ-05" 4 :sparse begin
    dev = testdevice()
    n = pagecount(dev, 4)
    x = testdata(Float32, n)
    a = MantleArray(dev, x)
    resize!(a, 2n); Array(a)
    resize!(a, n); Array(a)
    resize!(a, 2n)
    r, log = stepsduring(() -> Array(a), dev)
    @test stepkinds(log, a) == [:CellWrite]
    @test r[1:n] == x
    free!(a)
end

# RZ-06 RR:186-189,192-193: a reserve-backed array grows by one element before each of 10^5 runs: O(log) page binds, no move (same address), no recompile.
@case "RZ-06" 4 :sparse :long begin
    dev = testdevice()
    n0 = pagecount(dev, 4) + 1
    a = MantleArray(dev, Float32, pagecount(dev, 4))
    s = MantleArray(dev, Float32, 1)
    fill!(a, 1f0)
    resize!(a, n0); applypending!(s, a)            # now in a reserve
    marks = MantleArray(dev, Int32, n0 + 10^5)
    g = Graph(dev)
    dispatch!(g, rz_mark!, (marks, a), launchrange(a))
    run!(g)
    (_, d), log = stepsduring(dev) do
        counted(dev) do
            for _ in 1:10^5
                resize!(a, length(a) + 1)
                run!(g)
            end
        end
    end
    kinds = stepkinds(log, a)
    @test count(==(:BindPages), kinds) <= ceil(Int, log2((n0 + 10^5) / n0)) + 1
    @test :MoveStorage ∉ kinds
    @test d.plancompiles == 0
    @test launchcount(marks) == n0 + 10^5
    free!(g); foreach(free!, (a, s, marks))
end

# RZ-07 RR:194: a reserve outgrown: one move into a new reserve (address changes once), contents kept.
@case "RZ-07" 4 :sparse begin
    dev = testdevice()
    n = pagecount(dev, 4)
    x = testdata(Float32, n)
    a = MantleArray(dev, x)
    resize!(a, 2n); Array(a)                        # a reserve of 4·need = 8n elements
    resize!(a, 8n)
    _, log = stepsduring(() -> Array(a), dev)
    @test :MoveStorage ∉ stepkinds(log, a)
    resize!(a, 8n + 1)
    r, log2 = stepsduring(() -> Array(a), dev)
    @test stepkinds(log2, a) == [:BindPages, :MoveStorage, :CellWrite]
    @test r[1:n] == x
    free!(a)
end

# RZ-08 RR:191,198: caps report no sparse residency: every growth past the storage moves into 2·need, growth within it only writes the cell, no page binds, for small and large arrays.
@case "RZ-08" 4 begin
    fd = FaultDevice(testdevice())
    inject!(fd, :caps, Answer(1, withcaps(caps(testdevice()); sparseresidency = false)))
    for n in (64, 2^20)
        x = testdata(Float32, n)
        a = MantleArray(fd, x)
        len, room = n, n                               # requested length, elements the storage holds
        for _ in 1:6
            len = 3len ÷ 2 + 1
            resize!(a, len)
            r, log = stepsduring(() -> Array(a), fd)
            @test stepkinds(log, a) == (len > room ? [:MoveStorage, :CellWrite] : [:CellWrite])
            room = len > room ? 2len : room
            @test r[1:n] == x
        end
        free!(a)
    end
end

# RZ-09 RR:196,202: needs of 4·sparsepage-1 and 4·sparsepage bytes: the first moves into a pool region, the second into a reserve.
@case "RZ-09" 4 :sparse begin
    dev = testdevice()
    page = caps(dev).sparsepage
    a, b = MantleArray(dev, UInt8, 16), MantleArray(dev, UInt8, 16)
    resize!(a, 4page - 1)
    resize!(b, 4page)
    _, loga = stepsduring(() -> Array(a), dev)
    _, logb = stepsduring(() -> Array(b), dev)
    @test stepkinds(loga, a) == [:MoveStorage, :CellWrite]
    @test stepkinds(logb, b) == [:BindPages, :MoveStorage, :CellWrite]
    foreach(free!, (a, b))
end

# RZ-10 RR:203-206: on lavapipe (2 GiB of sparse address space) an array grown past the reserve limit gets storage of at least its need: its last elements are written and read back, no device fault.
@case "RZ-10" 4 :lavapipe :long begin
    dev = testdevice()
    a = MantleArray(dev, Float32[1, 2, 3, 4])
    resize!(a, 2^28 + 2^20)                           # just over 1 GiB
    dispatch!(dev, rz_writelast!, (a,), 4)
    tail, head = MantleArray(dev, Float32, 4), MantleArray(dev, Float32, 4)
    dispatch!(dev, rz_readlast!, (tail, a), 4)
    dispatch!(dev, rz_copy!, (head, a), 4)
    @test Array(tail) == Float32[1, 2, 3, 4]
    @test Array(head) == Float32[1, 2, 3, 4]
    foreach(free!, (a, tail, head))
end

# RZ-11 api.md:80-81, RR:235-237: a reserve with 64 MiB mapped shrinks to one element: nothing is unmapped, and a view taken before reads the old values (VAL: validation clean).
@case "RZ-11" 4 :sparse begin
    dev = testdevice()
    x = testdata(Float32, 10^6)
    a, s = MantleArray(dev, Float32, 16), MantleArray(dev, Float32, 1)
    resize!(a, 2^24)
    a[1:10^6] = x
    v = view(a, 1:10^6)
    applypending!(s, a)
    mapped = memory(dev).mapped
    resize!(a, 1)
    out = MantleArray(dev, Float32, 10^6)
    dispatch!(dev, rz_copy!, (out, v), 10^6)
    @test Array(out) == x
    @test memory(dev).mapped == mapped
    foreach(free!, (a, s, out))
end

# RZ-12 api.md:81, RR:100,235-236: trim!(a) stays pending until the next submission that touches `a`, which unmaps the pages past the size; growing again binds pages and keeps a[1].
@case "RZ-12" 4 :sparse begin
    dev = testdevice()
    a, s = MantleArray(dev, Float32, 16), MantleArray(dev, Float32, 1)
    resize!(a, 2^24)
    a[1:1] = [5f0]
    applypending!(s, a)
    resize!(a, 1); applypending!(s, a)
    mapped = memory(dev).mapped
    trim!(a)
    @test memory(dev).mapped == mapped
    @test !isempty(Mantle.pendingof(a))
    _, log = stepsduring(() -> applypending!(s, a), dev)
    @test :UnbindPages in stepkinds(log, a)
    @test memory(dev).mapped < mapped
    resize!(a, 10^6)
    r, log2 = stepsduring(() -> Array(a), dev)
    @test :BindPages in stepkinds(log2, a)
    @test r[1] == 5f0
    foreach(free!, (a, s))
end

# RZ-13 api.md:81,486: with a small budget, pages past an array's size stay mapped under pressure; an allocation nothing else can make room for throws OutOfDeviceMemory.
@case "RZ-13" 4 :sparse begin
    fd = FaultDevice(testdevice())
    a, s = MantleArray(fd, Float32, 16), MantleArray(fd, Float32, 1)
    resize!(a, 2^24)
    a[1:1] = [7f0]
    applypending!(s, a)
    resize!(a, 1); applypending!(s, a)
    m = memory(fd)
    inject!(fd, :budget, Answer(calls(fd, :budget) + 1, m.reserved + m.mapped + 2^20))
    @test_throws Mantle.OutOfDeviceMemory MantleArray(fd, Float32, 2^24)
    @test memory(fd).mapped == m.mapped
    @test Array(a) == [7f0]
    foreach(free!, (a, s))
end

# RZ-14 api.md:80, PLAN:484-486: resize!(a, 7) of a 4×5 array sets the last dimension: (4, 7), the first 20 elements kept as linear memory.
@case "RZ-14" 4 begin
    dev = testdevice()
    x = testdata(Float32, 20)
    a = MantleArray(dev, reshape(x, 4, 5))
    resize!(a, 7)
    @test size(a) == (4, 7)
    r = Array(a)
    @test size(r) == (4, 7)
    @test vec(r)[1:20] == x
    free!(a)
end

# RZ-15 api.md:80,471, RR:530: resize! of a view or of a reshape throws ArgumentError; nothing pends, the length is unchanged.
@case "RZ-15" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    @test_throws ArgumentError resize!(view(a, 1:10), 20)
    @test_throws ArgumentError resize!(reshape(a, 10, 10), 20)
    @test isempty(Mantle.pendingof(a))
    @test length(a) == 100
    free!(a)
end

# RZ-16 api.md:79, RR:543: resize! after free!(a) throws ArgumentError under the lock; nothing pends, the length is unchanged.
@case "RZ-16" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    free!(a)
    @test_throws ArgumentError resize!(a, 10)
    @test isempty(Mantle.pendingof(a))
    @test length(a) == 100
end

# RZ-17 api.md:206,481, INT:1780: while `a` is lent through withview, resize! and a store throw; after the block both succeed.
@case "RZ-17" 6 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, testdata(Float32, 100))
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test_throws ArgumentError resize!(a, 200)
        @test_throws ArgumentError (a[1:1] = [1f0])
    end
    resize!(a, 200)
    a[1:1] = [1f0]
    @test length(a) == 200
    @test Array(a)[1] == 1f0
    free!(a)
end

# RZ-18 api.md:80, RR:814-815: a graph array placed for n: resizes to n÷2 and n are cell writes with no recompile; n+1 recompiles once at the next run!, never a throw.
@case "RZ-18" 4 begin
    dev = testdevice()
    n = 1024
    g = Graph(dev)
    t = MantleArray(g, Float32, n)
    marks = MantleArray(dev, Int32, 2n)
    dispatch!(g, rz_mark!, (marks, t), launchrange(t))
    resize!(t, n); run!(g)                           # the first resize! makes `t` resizable (a recompile allowed here)
    for (len, compiles) in ((n ÷ 2, 0), (n, 0), (n + 1, 1))
        fill!(marks, Int32(0))
        resize!(t, len)
        (_, d), log = stepsduring(() -> counted(() -> run!(g), dev), dev)
        @test d.plancompiles == compiles
        compiles == 0 && @test :CellWrite in stepkinds(log, t)
        @test launchcount(marks) == len
    end
    free!(g); free!(marks)
end

# RZ-19 api.md:80, RR:810: resize!(a, 0): a launch over launchrange(a) does no work (VAL: validation clean); grown back to n with a store, a run reads exactly the stored data.
@case "RZ-19" 4 begin
    dev = testdevice()
    n = 1000
    a = MantleArray(dev, testdata(Float32, n))
    resize!(a, 0)
    marks = MantleArray(dev, Int32, n)
    fill!(marks, Int32(0))
    dispatch!(dev, rz_mark!, (marks, a), launchrange(a))
    @test launchcount(marks) == 0
    @test isempty(Array(a))
    x = testdata(Float32, n; seed = 2)
    resize!(a, n)
    a[1:n] = x
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, rz_copy!, (out, a), launchrange(a))
    run!(g)
    @test Array(out) == x
    free!(g); foreach(free!, (a, marks, out))
end

# RZ-20 RR:290-300,557 (storageat): a pool-region `a` of n: resize!(a, 2n), a[1:1] = x, resize!(a, 8n), one submission: move, store, move in sequence; a[1] == x, a[2:n] old.
@case "RZ-20" 4 begin
    dev = testdevice()
    n = smallcount(dev)
    x = testdata(Float32, n)
    a, s = MantleArray(dev, x), MantleArray(dev, Float32, 1)
    resize!(a, n); applypending!(s, a)              # adopted in place: storage for n
    resize!(a, 2n)
    a[1:1] = [42f0]
    resize!(a, 8n)
    @test pendingkinds(a) == [:Resize, :Store, :Resize]
    r, log = stepsduring(() -> Array(a), dev)
    @test filter(!=(:CellWrite), stepkinds(log, a)) == [:MoveStorage, storekind(dev, 0, 4), :MoveStorage]
    @test allequal(s.token for s in log if s.target === a)
    @test r[1] == 42f0
    @test r[2:n] == x[2:n]
    foreach(free!, (a, s))
end

# RZ-21 RR:290-296: resize!(a, 10) then resize!(a, 1000) on 100 elements: one Resize to (1000,) keeping 10 elements; a[1:10] old.
@case "RZ-21" 4 begin
    dev = testdevice()
    x = testdata(Float32, 100)
    a = MantleArray(dev, x)
    resize!(a, 10)
    resize!(a, 1000)
    @test pendingkinds(a) == [:Resize]
    c = only(Mantle.pendingof(a))
    @test c.dims == (1000,)
    @test c.keep == 10 * sizeof(Float32)
    r = Array(a)
    @test length(r) == 1000
    @test r[1:10] == x[1:10]
    free!(a)
end

# RZ-22 RR:395-397,569-571: graph A reads `a` and is in flight; an eager resize! and fill! move `a`: the move waits for A, and A's storage is not reused before A's token.
@case "RZ-22" 4 :multiqueue begin
    dev = testdevice()
    n = smallcount(dev)
    x = testdata(Float32, n)
    a, out = MantleArray(dev, x), MantleArray(dev, Float32, n)
    g = slowcopygraph(dev, out, a, n)
    run!(g); Array(out)                             # compiled and recorded
    run!(g)                                         # in flight: the copy of `a` comes after the spin
    resize!(a, 4n)
    fill!(a, -1f0)
    others = [fill!(MantleArray(dev, Float32, n), -2f0) for _ in 1:8]   # would land in A's storage if it were free
    @test Array(out) == x
    @test all(==(-1f0), Array(a))
    free!(g); foreach(free!, others); foreach(free!, (a, out))
end

# RZ-23 api.md:490, RR:195,567-568: `a` evicted under a small budget, then resize!(a, 2n): the next submission restores it into storage of 2n and writes the cell, no separate move.
@case "RZ-23" 4 :separateheap begin
    fd = FaultDevice(testdevice())
    n = 2^24
    bytes = sizeof(Float32) * n
    x = testdata(Float32, n)
    a, s = MantleArray(fd, x), MantleArray(fd, Float32, 1024)
    m = memory(fd)
    inject!(fd, :budget, Answer(calls(fd, :budget) + 1, m.reserved + bytes + bytes ÷ 4))
    b = MantleArray(fd, Float32, 2n)                 # fits only once `a` (idle, nothing pending) is evicted
    free!(b)
    resize!(a, 2n)
    _, log = stepsduring(() -> dispatch!(fd, rz_copy!, (s, a), 1024), fd)
    @test stepkinds(log, a) == [:Restore, :CellWrite]
    @test Array(s) == x[1:1024]
    foreach(free!, (a, s))
end

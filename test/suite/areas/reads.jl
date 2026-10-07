# Host reads: Array(a), copyto!(host, a), devicesize(a) (catalogue-core.md,
# table 19: READ). Contracts: api.md:85-87, 452, 481; recording-plan.md
# "Eager work", host reads (972-980); internals.jl section 11 (2212-2250).
# Line numbers of 2026-10-05.
#
# Device side, lengthof(a) arrives as a one-element integer array
# (`len[1] = n`); a value GPURef arrives as its value.

using KernelAbstractions: @kernel, @index

# out[i] = x[i] + k
@kernel function rd_addvalue!(out, x, k)
    i = @index(Global)
    out[i] = x[i] + k
end

# a[i] = k for i <= 100k and 100k as a's length, after reading gate (written by slowwrite!).
@kernel function rd_fillrun!(a, len, k, gate)
    i = @index(Global)
    n = Int32(100) * k + (gate[1] == Int32(-7) ? Int32(1) : Int32(0))
    if i <= n
        a[i] = k
    end
    if i == 1
        len[1] = n
    end
end

# Spins `iters` steps, then writes `n` as a's length.
@kernel function rd_spinlength!(a, len, n, iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    len[1] = n + (acc == Int32(-7) ? Int32(1) : Int32(0))
end

# a[offset + i] = k
@kernel function rd_fill!(a, k, offset)
    i = @index(Global)
    a[offset + i] = k
end

# a[offset + i] = k, after reading gate (written by slowwrite!).
@kernel function rd_fillafter!(a, k, offset, gate)
    i = @index(Global)
    a[offset + i] = k + (gate[1] == Int32(-7) ? Int32(1) : Int32(0))
end

"""Whether `v` is what one run of READ-04's graph writes: `100k` elements, all `k`."""
onerun(v) = !isempty(v) && length(v) == 100 * Int(v[1]) && all(==(v[1]), v)

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

# ── 19. Host reads (READ) ──

# api.md:85: Array(a) with a pending store is one submission of its own that applies the store; nothing is pending after it.
@case "READ-01" 4 begin
    dev = testdevice()
    x = testdata(Int32, 1024)
    a = MantleArray(dev, x)
    a[1:4] = Int32[1, 2, 3, 4]
    v, diag = counted(() -> Array(a), dev)
    @test v == [Int32[1, 2, 3, 4]; x[5:end]]
    @test submitcount(diag) == 1
    @test isempty(Mantle.pendingof(a))
end

# internals.jl:2217-2228, recording-plan.md:975-978: a graph writes `out` (Readback memory) at the end of a slow run; Array(out) right after run! waits for that run (its values, not the previous run's) and makes no GPU copy.
@case "READ-02" 3 begin
    dev = testdevice()
    x = testdata(Int32, 1024)
    w = MantleArray(dev, x)
    out = MantleArray(dev, Int32, 1024; memory = Mantle.Readback())
    k = GPURef(dev, Int32(1))
    g = Graph(dev)
    t = MantleArray(g, Int32, 1024)
    dispatch!(g, rd_addvalue!, (t, w, k), 1024)
    slowwrite!(g, t, 99; ms = 300)        # t[1] = 99 after about 300 ms
    copyto!(out, t)                       # reads t: after the slow kernel
    run!(g)
    @test Array(out) == [Int32(99); x[2:end] .+ Int32(1)]
    k[] = Int32(2)
    run!(g)
    v, diag = counted(() -> Array(out), dev)
    @test v == [Int32(99); x[2:end] .+ Int32(2)]
    @test submitcount(diag) == 0
    @test diag.hostwaits >= 1
end

# api.md:85, 92: Array and copyto!(host, t) of a graph array throw.
@case "READ-03" 6 begin
    dev = testdevice()
    g = Graph(dev)
    t = MantleArray(g, Int32, 16)
    @test_throws ArgumentError Array(t)
    @test_throws ArgumentError copyto!(zeros(Int32, 16), t)
end

# internals.jl:2213-2215: Array(a) of a kernel-written length while the next run (which writes another length and other data) is in flight: length and data come from one point in the queue.
@case "READ-04" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 0; capacity = 1024)
    gate = MantleArray(dev, zeros(Int32, 1))
    k = GPURef(dev, Int32(1))
    g = Graph(dev)
    slowwrite!(g, gate, 0; ms = 300)
    dispatch!(g, rd_fillrun!, (a, lengthof(a), k, gate), 1024)
    run!(g)
    @test onerun(Array(a))
    k[] = Int32(2)
    run!(g)                               # on the GPU for about 300 ms
    v = Array(a)
    @test onerun(v)
    @test v[1] in (Int32(1), Int32(2))
end

# api.md:87: devicesize(a) copies the cell after the kernel that writes the length has run: it waits, and reads 10, then 20.
@case "READ-05" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 0; capacity = 1024)
    iters = Int32(calibratedspin(dev, 300))
    dispatch!(dev, rd_spinlength!, (a, lengthof(a), Int32(10), iters), 1)
    @test devicesize(a) == 10
    dispatch!(dev, rd_spinlength!, (a, lengthof(a), Int32(20), iters), 1)
    @test devicesize(a) == 20
end

# api.md:85, 481: Array(a) and copyto!(host, a) of an array lent to withview throw; after the block they read.
@case "READ-06" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 16))
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test_throws ArgumentError Array(a)
        @test_throws ArgumentError copyto!(zeros(Int32, 16), a)
    end
    @test Array(a) == zeros(Int32, 16)
end

# api.md:452: host reads racing 50 runs that write a in two halves (a slow kernel between them): every read sees one run's whole result, never a mix.
@case "READ-07" 3 begin
    dev = testdevice()
    n = 2^20
    a = MantleArray(dev, zeros(Int32, n))
    gate = MantleArray(dev, zeros(Int32, 1))
    k = GPURef(dev, Int32(0))
    g = Graph(dev)
    dispatch!(g, rd_fill!, (a, k, 0), n ÷ 2)
    slowwrite!(g, gate, 0; ms = 20)
    dispatch!(g, rd_fillafter!, (a, k, n ÷ 2, gate), n ÷ 2)
    run!(g)
    writer = Threads.@spawn for i in 1:50
        k[] = Int32(i)
        run!(g)
    end
    whole = Bool[]
    while !istaskdone(writer) && length(whole) < 1000
        v = Array(a)
        push!(whole, all(==(v[1]), v))
    end
    withtimeout(() -> fetch(writer), 60)
    @test !isempty(whole)
    @test all(whole)
    @test Array(a) == fill(Int32(50), n)
end

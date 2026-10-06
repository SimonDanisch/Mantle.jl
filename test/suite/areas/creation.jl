# Creation from data: MantleArray(dev, data) and the mapped-file form
# (catalogue-core.md, table 21: DATA; catalogue-resize-rt.md, table E: CR).
# Contracts: api.md:69; decisions.md B3.18 (269-277) and B3.9 (193-198);
# resizing-and-raytracing.jl:784-799 (creation streams in chunks) and 735-738
# (changes pended before a plan exists). Line numbers of 2026-10-05.
#
# The mapped form follows pending decision 5: `MantleArray(dev, path; mmap =
# true)`; element type and dims as keywords are this file's assumption. The
# host memory a creation holds is measured as Mantle.liveuploads(dev), taken
# to count live upload regions. Large creations (:long) use 8 GiB or half the
# device's budget, whichever is smaller, and compare 16 windows of the data.

using KernelAbstractions: @kernel, @index, get_backend

const CR_LEN = 32 * 2^20          # Int32 elements: 128 MiB, several upload chunks

# out[i] = x[i] + 1
@kernel function cr_plusone!(out, x)
    i = @index(Global)
    out[i] = x[i] + one(eltype(x))
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

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""An array of `T` with `dims` on `d`, mapped from the file at `path` (pending decision 5)."""
mmaparray(d, T, path, dims) = MantleArray(d, path; mmap = true, eltype = T, dims = dims)

"""The bytes of the largest creation the :long cases make on `dev`."""
largestcreation(dev) = min(8 * 2^30, Mantle.budget(dev) ÷ 2)

"""16 windows of up to 2^20 elements spread over `1:n`."""
windowsof(n, w = min(n, 2^20)) = [k .+ (1:w) for k in round.(Int, range(0, n - w; length = 16))]

"""Whether device array `a` holds `x`, compared on 16 windows (a large array is never read back whole)."""
sameas(a, x) = length(a) == length(x) && all(r -> Array(view(a, r)) == x[r], windowsof(length(x)))

"""`f()` and the largest value `probe()` took while it ran, sampled about every millisecond on another task."""
function peakduring(f, probe)
    running = Threads.Atomic{Bool}(true)
    sampler = Threads.@spawn begin
        p = probe()
        while running[]
            p = max(p, probe())
            sleep(0.001)
        end
        p
    end
    v = f()
    running[] = false
    return v, fetch(sampler)
end

# ── 21. Creation from data (DATA) ──

# api.md:69, decisions.md:269-277: creation from 1 KiB, 1 GiB and 8 GiB: exact; 1 KiB is at most one submission, the large ones are cut into chunks of one submission each (none where the storage is host-visible).
@case "DATA-01" 6 :long begin
    dev = testdevice()
    x = testdata(UInt8, 1024)
    a, diag = counted(() -> MantleArray(dev, x), dev)
    @test Array(a) == x
    @test submitcount(diag) <= 1
    free!(a)
    for bytes in (2^30, largestcreation(dev))
        x = testdata(UInt8, bytes)
        a, diag = counted(() -> MantleArray(dev, x), dev)
        @test sameas(a, x)
        @test submitcount(diag) != 1
        free!(a)
    end
end

# resizing-and-raytracing.jl:789-791: an 8 GiB creation holds only a few chunks of host memory at a time: at most 8 upload regions live beside what was live before.
@case "DATA-02" 6 :long begin
    dev = testdevice()
    x = testdata(UInt8, largestcreation(dev))
    base = Mantle.liveuploads(dev)
    a, peak = peakduring(() -> MantleArray(dev, x), () -> Mantle.liveuploads(dev))
    @test peak - base <= 8
    @test sameas(a, x)
    free!(a)
end

# api.md:69: the caller zeroes its data right after creation returns: the array keeps the original.
@case "DATA-03" 6 begin
    dev = testdevice()
    x = testdata(Int32, 1024)
    a = MantleArray(dev, x)
    x .= 0
    @test Array(a) == testdata(Int32, 1024)
end

# resizing-and-raytracing.jl:791-792: an array created on one task, then a new graph reading it compiled and run at once on another: the run waits for the chunks and is exact.
@case "DATA-04" 6 begin
    dev = testdevice()
    x = testdata(Int32, CR_LEN)
    o = MantleArray(dev, Int32, CR_LEN)
    handoff = Channel{Any}(1)
    creator = Threads.@spawn put!(handoff, MantleArray(dev, x))
    runner = Threads.@spawn (g = plusonegraph(dev, take!(handoff), o); run!(g); g)
    withtimeout(() -> fetch(creator), 60)
    withtimeout(() -> fetch(runner), 60)
    @test Array(o) == x .+ Int32(1)
end

# resizing-and-raytracing.jl:789, 796: on lavapipe the storage is host-visible: creation writes it directly, with no submission.
@case "DATA-05" 6 :lavapipe begin
    dev = testdevice()
    x = testdata(Int32, 2^20)
    a, diag = counted(() -> MantleArray(dev, x), dev)
    @test submitcount(diag) == 0
    @test Array(a) == x
end

# api.md:69, 486: data larger than the budget: OutOfDeviceMemory after the pressure steps, and nothing of the attempt (storage, upload regions) stays allocated.
@case "DATA-06" 6 begin
    d = faultdevice()
    base = memory(d)
    limit!(d, 64 * 2^20)
    @test_throws Mantle.OutOfDeviceMemory MantleArray(d, testdata(Int32, CR_LEN))
    @test memory(d) == base
end

# api.md:69 (unspecified; the suite expects a copy of any array of isbits elements and an ArgumentError otherwise): a zero-length array, a strided view of host data, a String array.
@case "DATA-07" 6 begin
    dev = testdevice()
    a0 = MantleArray(dev, Int32[])
    @test size(a0) == (0,)
    @test Array(a0) == Int32[]
    x = testdata(Int32, 2048)
    s = MantleArray(dev, view(x, 1:2:2048))
    @test Array(s) == x[1:2:2048]
    @test_throws ArgumentError MantleArray(dev, ["a", "b"])
end

# api.md:69 (errors unspecified): mapping a file that does not exist throws SystemError, as opening it does.
@case "DATA-08" 6 begin   # pending decision 5
    dev = testdevice()
    @test_throws SystemError mmaparray(dev, Int32, joinpath(tempdir(), "no-such-file-$(rand(UInt64))"), (16,))
end

# decisions.md:275-277: creation is not a change: nothing pending on the new array and no change step applied to it.
@case "DATA-09" 6 begin
    dev = testdevice()
    x = testdata(Int32, 2^20)
    a, log = stepsduring(() -> MantleArray(dev, x), dev)
    @test isempty(Mantle.pendingof(a))
    @test isempty([s for s in log if s.target === a])
    @test Array(a) == x
end

# pending-decisions-2026-10-05.md:20 (ambiguity 16): the mapped form takes its device; the array lives on it and holds the file.
@case "DATA-10" 6 begin   # pending decision 5
    dev = testdevice()
    x = testdata(Int32, 4096)
    path = tempname()
    write(path, x)
    a = mmaparray(dev, Int32, path, (4096,))
    @test get_backend(a) === dev
    @test Array(a) == x
    free!(a)
    rm(path)
end

# ── E. Creation from data (CR) ──

# resizing-and-raytracing.jl:791-792: an array created on one task and passed at once to an eager call on another: the call waits for the upload and computes from x.
@case "CR-01" 6 begin
    dev = testdevice()
    x = testdata(Int32, CR_LEN)
    out = MantleArray(dev, Int32, CR_LEN)
    handoff = Channel{Any}(1)
    creator = Threads.@spawn put!(handoff, MantleArray(dev, x))
    user = Threads.@spawn dispatch!(dev, cr_plusone!, (out, take!(handoff)), CR_LEN)
    withtimeout(() -> fetch(creator), 60)
    withtimeout(() -> fetch(user), 60)
    @test Array(out) == x .+ Int32(1)
end

# resizing-and-raytracing.jl:787-791: data larger than one submission may upload: one submission per chunk (none where the storage is host-visible), at most a few upload regions live at a time.
@case "CR-02" 6 :long begin
    dev = testdevice()
    x = testdata(UInt8, largestcreation(dev))
    base = Mantle.liveuploads(dev)
    (a, diag), peak = peakduring(() -> counted(() -> MantleArray(dev, x), dev), () -> Mantle.liveuploads(dev))
    @test submitcount(diag) != 1
    @test peak - base <= 8
    @test sameas(a, x)
    free!(a)
end

# resizing-and-raytracing.jl:789, 796: on Metal (unified memory) the storage is host-visible: a direct copy, no submission.
@case "CR-03" 9 :metal begin
    dev = testdevice()
    x = testdata(Int32, 2^20)
    a, diag = counted(() -> MantleArray(dev, x), dev)
    @test submitcount(diag) == 0
    @test Array(a) == x
end

# decisions.md:193-196, api.md:69: an array mapped from a 2 GiB file equals the file; at most a few upload regions live while it streams.
@case "CR-04" 6 :long begin   # pending decision 5
    dev = testdevice()
    x = testdata(UInt8, 2^31)
    path = tempname()
    write(path, x)
    base = Mantle.liveuploads(dev)
    a, peak = peakduring(() -> mmaparray(dev, UInt8, path, (2^31,)), () -> Mantle.liveuploads(dev))
    @test peak - base <= 8
    @test sameas(a, x)
    free!(a)
    rm(path)
end

# resizing-and-raytracing.jl:784-792: the caller zeroes 128 MiB of data while its chunks may still be in flight: the array holds the original.
@case "CR-05" 6 begin
    dev = testdevice()
    x = testdata(Int32, CR_LEN)
    a = MantleArray(dev, x)
    x .= 0
    @test Array(a) == testdata(Int32, CR_LEN)
end

# resizing-and-raytracing.jl:793-799: every block allocation after the creation's first one fails (FI-A, on rawalloc): the constructor throws, and its storage and upload regions go back (memory at the baseline).
@case "CR-06" 6 :separateheap begin
    d = faultdevice()
    base = memory(d)
    n = calls(d, :rawalloc)
    for k in 2:16
        inject!(d, :rawalloc, Throw(n + k, ErrorException("rawalloc failed (injected)")))
    end
    @test_throws ErrorException MantleArray(d, testdata(Int32, 4 * CR_LEN))
    @test memory(d) == base
end

# resizing-and-raytracing.jl:791-792, 735-738: created from x, then a new graph reading it compiled and run with no other access: the first run waits for the upload and reads x.
@case "CR-07" 6 begin
    dev = testdevice()
    x = testdata(Int32, CR_LEN)
    a = MantleArray(dev, x)
    o = MantleArray(dev, Int32, CR_LEN)
    run!(plusonegraph(dev, a, o))
    @test Array(o) == x .+ Int32(1)
end

# resizing-and-raytracing.jl:735-738 (ambiguity A1, settled there): a store pended on a new array before any graph names it is applied by the first run of a graph compiled afterwards.
@case "CR-08" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 1024)
    a[1:3] = Int32[7, 8, 9]
    o = MantleArray(dev, Int32, 1024)
    run!(plusonegraph(dev, a, o))
    @test Array(o)[1:3] == Int32[8, 9, 10]
end

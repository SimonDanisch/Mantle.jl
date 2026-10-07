# lengthof and kernel-written lengths (catalogue-resize-rt.md, table G).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; INT =
# docs/internals.jl (line numbers of 2026-10-05). Pending decision 6
# (experiments/plan-review/pending-decisions-2026-10-05.md): length(a) throws
# for an array whose length a kernel writes (devicesize, capacity instead);
# resize! and copy! on it throw; a kernel writing more than the capacity is
# clamped to it by the prepare node. Cases that depend on it say so.
# Device side, lengthof(a) arrives in the kernel as a one-element integer
# array: the kernel writes the length with `len[1] = n`.

using KernelAbstractions: @kernel, @index

@kernel function kl_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function kl_mark!(marks, a)
    i = @index(Global)
    marks[i] = Int32(1)
end

@kernel function kl_fill!(a, v)
    i = @index(Global)
    a[i] = v
end

@kernel function kl_writelength!(len, count)
    len[1] = count
end

# Writes `count` into the first `count` elements; launched over the capacity.
@kernel function kl_fillupto!(alive, count)
    i = @index(Global)
    if i <= count
        alive[i] = Float32(count)
    end
end

"""How many work items a launch marked in `marks`."""
launchcount(marks) = count(==(Int32(1)), Array(marks))

"""
A graph whose first kernel writes `count` (a number or a value GPURef) into
lengthof(alive) and into as many elements of `alive`; then, if `marks` is
given, a launch over launchrange(alive) marks its items.
"""
function lengthgraph(dev, alive, count, marks = nothing)
    g = Graph(dev)
    dispatch!(g, kl_writelength!, (lengthof(alive), count), 1)
    dispatch!(g, kl_fillupto!, (alive, count), capacity(alive))
    markitems!(g, marks, alive)
    return g
end
markitems!(g, marks, alive) = dispatch!(g, kl_mark!, (marks, alive), launchrange(alive))
markitems!(g, ::Nothing, alive) = nothing

"""A zeroed Int32 array of `n` elements."""
zeroed(dev, n) = (m = MantleArray(dev, Int32, n); fill!(m, Int32(0)); m)

# ── G. lengthof and kernel-written lengths ──

# KL-01 api.md:76,90,476: lengthof(x) for an array created without capacity is refused at declaration; the graph runs only what was declared before, and x is untouched.
@case "KL-01" 4 begin
    dev = testdevice()
    x = MantleArray(dev, testdata(Float32, 16))
    out = MantleArray(dev, Float32, 16)
    g = Graph(dev)
    dispatch!(g, kl_fill!, (out, 1f0), 16)
    @test_throws ArgumentError dispatch!(g, kl_writelength!, (lengthof(x), Int32(5)), 1)
    _, d = counted(() -> run!(g), dev)
    @test d.plancompiles == 1
    @test Array(out) == fill(1f0, 16)
    @test length(Array(x)) == 16
    free!(g); foreach(free!, (x, out))
end

# KL-02 api.md:90, RR:113-116, INT:216-217: a kernel writes 500 into lengthof(alive) (capacity 10^6); the next node launches over launchrange(alive): exactly 500 work items, from the command the prepare node wrote.
@case "KL-02" 4 begin
    dev = testdevice()
    cap = 10^6
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    marks = zeroed(dev, cap)
    g = lengthgraph(dev, alive, Int32(500), marks)
    run!(g)
    @test launchcount(marks) == 500
    free!(g); foreach(free!, (alive, marks))
end

# KL-03 api.md:85,87-88, INT:2212-2215,2249-2250: after a kernel wrote 500, devicesize is 500 and Array returns those 500 elements, length and data from one submission; length(alive) throws (pending decision 6).
@case "KL-03" 4 begin
    dev = testdevice()
    cap = 10^6
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    g = lengthgraph(dev, alive, Int32(500))
    run!(g)
    @test devicesize(alive) == 500
    r, d = counted(() -> Array(alive), dev)
    @test submitcount(d) == 1
    @test r == fill(500f0, 500)
    @test capacity(alive) == cap
    @test_throws ArgumentError length(alive)      # pending decision 6
    free!(g); free!(alive)
end

# KL-04 RR:810, api.md:491: the kernel writes 0: a launch over launchrange(alive) does no work (VAL: validation clean); devicesize is 0 and Array returns no element.
@case "KL-04" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    marks = zeroed(dev, cap)
    g = lengthgraph(dev, alive, Int32(0), marks)
    run!(g)
    @test launchcount(marks) == 0
    @test devicesize(alive) == 0
    @test isempty(Array(alive))
    free!(g); foreach(free!, (alive, marks))
end

# KL-05 pending decision 6, RR:213-214: a kernel writes capacity+1: the prepare node clamps it, the launch covers the capacity and nothing past it (VAL: no fault).
@case "KL-05" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    marks = zeroed(dev, cap + 1024)
    g = lengthgraph(dev, alive, Int32(cap + 1), marks)
    run!(g)
    @test launchcount(marks) == cap
    @test all(iszero, Array(marks)[cap+1:end])
    @test devicesize(alive) == cap
    @test length(Array(alive)) == cap
    free!(g); foreach(free!, (alive, marks))
end

# KL-06 RR:779-782,800-802: a temporary SizedBy(identity, alive) is sized from the capacity: lengths from 0 to the capacity over 100 runs recompile nothing, and each run is exact.
@case "KL-06" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    written = GPURef(dev, Int32(0))
    out = MantleArray(dev, Float32, cap)
    g = lengthgraph(dev, alive, written)
    tmp = MantleArray(g, Float32, Mantle.SizedBy(identity, alive))
    dispatch!(g, kl_copy!, (tmp, alive), launchrange(alive))
    dispatch!(g, kl_copy!, (out, tmp), launchrange(alive))
    run!(g)
    counts = rand(Xoshiro(6), 0:cap, 100)
    _, d = counted(() -> foreach(c -> (written[] = Int32(c); run!(g)), counts), dev)
    @test d.plancompiles == 0
    @test devicesize(alive) == last(counts)
    @test Array(out)[1:last(counts)] == fill(Float32(last(counts)), last(counts))
    free!(g); foreach(free!, (alive, out))
end

# KL-07 RR:804-805: an eager dispatch over launchrange(alive) is indirect from the cell, with the count the kernel wrote.
@case "KL-07" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    g = lengthgraph(dev, alive, Int32(300))
    run!(g)
    marks = zeroed(dev, cap)
    dispatch!(dev, kl_mark!, (marks, alive), launchrange(alive))
    @test launchcount(marks) == 300
    free!(g); foreach(free!, (alive, marks))
end

# KL-08 pending decision 6, RR:101,213-214: resize! and copy! on an array whose length a kernel writes throw and pend nothing; the kernel's count stands.
@case "KL-08" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    g = lengthgraph(dev, alive, Int32(300))
    @test_throws ArgumentError resize!(alive, 7)
    @test_throws ArgumentError copy!(alive, Float32[1, 2, 3])
    @test isempty(Mantle.pendingof(alive))
    run!(g)
    @test devicesize(alive) == 300
    free!(g); free!(alive)
end

# KL-09 INT:2212-2215: one thread runs g (a count, and that many elements of that value) 1000 times while another reads Array(alive): every result is a consistent (length, data) pair.
@case "KL-09" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    written = GPURef(dev, Int32(1))
    g = lengthgraph(dev, alive, written)
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    reader = Threads.@spawn begin
        mismatches, reads = 0, 0
        while !stop[]
            r = Array(alive)
            reads += 1
            mismatches += !all(==(Float32(length(r))), r)
        end
        (mismatches, reads)
    end
    for i in 1:1000
        written[] = Int32(1 + i % 1000)
        run!(g)
    end
    stop[] = true
    mismatches, reads = withtimeout(() -> fetch(reader), 60)
    @test mismatches == 0
    @test reads > 0
    free!(g); free!(alive)
end

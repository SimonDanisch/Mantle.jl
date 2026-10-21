# lengthof and kernel-written lengths: api.md 2.2 (lengthof, length,
# capacity, `Array(a)`, bounds) and section 7; recording-plan.md, Operations
# ("Lengths decided on the GPU"); resizing-and-raytracing.jl B.7.
# A length a kernel writes needs no capacity: the kernel writes within the
# array's current storage; `capacity = n` only reserves room up front.
# length(a) is the real length: it waits for the kernel that writes it and
# reads the cell back, and throws during the declaration of the graph that
# writes it. Nothing clamps a kernel-written length.
# Device side, lengthof(a) arrives in the kernel as a one-element integer
# array: the kernel writes the length with `len[1] = n`.

function kl_copy!(dst, src)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return nothing   # ndrange is rounded up to whole work-groups
    dst[i] = src[i]
    return nothing
end

function kl_mark!(marks, a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    marks[i] = Int32(1)
    return nothing
end

function kl_fill!(a, v)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    a[i] = v
    return nothing
end

function kl_writelength!(len, count)
    KI.get_global_id().x == 1 || return nothing
    len[1] = count
    return nothing
end

# Writes `count` into the first `count` elements, at most `room` (the
# capacity it is launched over): past the length this launch saw, within the
# storage.
function kl_fillupto!(alive, count, room)
    i = KI.get_global_id().x
    i <= min(count, room) || return nothing
    @inbounds alive[i] = Float32(count)
    return nothing
end

"""How many work items a launch marked in `marks`."""
launchcount(marks) = count(==(Int32(1)), Array(marks))

"""
A graph whose first kernel writes `count` (a number or a value GPURef) into
lengthof(alive) and into as many elements of `alive` (at most its capacity,
which the fill launches over); then, if `marks` is
given, a launch over launchrange(alive) marks its items.
"""
function lengthgraph(dev, alive, count, marks = nothing)
    g = Graph(dev)
    dispatch!(g, kl_writelength!, (lengthof(alive), count), 1)
    dispatch!(g, kl_fillupto!, (alive, count, Int32(capacity(alive))), capacity(alive))
    markitems!(g, marks, alive)
    return g
end
markitems!(g, marks, alive) = dispatch!(g, kl_mark!, (marks, alive), launchrange(alive))
markitems!(g, ::Nothing, alive) = nothing

"""A zeroed Int32 array of `n` elements."""
zeroed(dev, n) = (m = MantleArray(dev, Int32, n); fill!(m, Int32(0)); m)

# ── G. lengthof and kernel-written lengths ──

# KL-01 api.md 2.2 (lengthof) and section 7, recording-plan.md, Operations: lengthof(x) for an array created without capacity is accepted at declaration; the kernel writes the length within x's current storage: after the run length(x) is 5 and Array(x) its first 5 elements.
@case "KL-01" 4 begin
    dev = testdevice()
    data = testdata(Float32, 16)
    x = MantleArray(dev, data)
    out = MantleArray(dev, Float32, 16)
    g = Graph(dev)
    dispatch!(g, kl_fill!, (out, 1f0), 16)
    dispatch!(g, kl_writelength!, (lengthof(x), Int32(5)), 1)
    _, d = counted(() -> run!(g), dev)
    @test d.plancompiles == 1
    @test Array(out) == fill(1f0, 16)
    @test length(x) == 5
    @test Array(x) == data[1:5]
    free!(g); foreach(free!, (x, out))
end

# KL-02 api.md 2.2 (lengthof), recording-plan.md, Operations (the prepare node), resizing-and-raytracing.jl A (Arrays): a kernel writes 500 into lengthof(alive) (capacity 10^6); the next node launches over launchrange(alive): exactly 500 work items, from the command the prepare node wrote.
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

# KL-03 api.md 2.2 (length, `Array(a)`, capacity), resizing-and-raytracing.jl A (Arrays), internals.jl section 2 (size of a kernel-written length): during the declaration of the graph that writes it, length(alive) throws; after a kernel wrote 500, length(alive) is 500 and Array returns those 500 elements, length and data from one submission; the capacity is the one given.
@case "KL-03" 4 begin
    dev = testdevice()
    cap = 10^6
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    g = lengthgraph(dev, alive, Int32(500))
    @test_throws ArgumentError length(alive)      # no run has written it yet
    run!(g)
    @test length(alive) == 500
    r, d = counted(() -> Array(alive), dev)
    @test submitcount(d) == 1
    @test r == fill(500f0, 500)
    @test capacity(alive) == cap
    free!(g); free!(alive)
end

# KL-04 resizing-and-raytracing.jl B.7: the kernel writes 0: a launch over launchrange(alive) does no work (VAL: validation clean); length(alive) is 0 and Array returns no element.
@case "KL-04" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    marks = zeroed(dev, cap)
    g = lengthgraph(dev, alive, Int32(0), marks)
    run!(g)
    @test launchcount(marks) == 0
    @test length(alive) == 0
    @test isempty(Array(alive))
    free!(g); foreach(free!, (alive, marks))
end

# KL-05 api.md 2.2 (bounds: nothing clamps a kernel-written length): a kernel writes capacity+1: length(alive) is capacity+1, and the launch over launchrange(alive) covers capacity+1 items and nothing past them (VAL: no fault).
@case "KL-05" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    marks = zeroed(dev, cap + 1024)
    g = lengthgraph(dev, alive, Int32(cap + 1), marks)
    run!(g)
    @test launchcount(marks) == cap + 1
    @test all(iszero, Array(marks)[cap+2:end])
    @test length(alive) == cap + 1
    free!(g); foreach(free!, (alive, marks))
end

# KL-06 resizing-and-raytracing.jl B.7 (SizedBy over a kernel-written length): a temporary SizedBy(identity, alive) is sized from the capacity: lengths from 0 to the capacity over 100 runs recompile nothing, and each run is exact.
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
    @test length(alive) == last(counts)
    @test Array(out)[1:last(counts)] == fill(Float32(last(counts)), last(counts))
    free!(g); foreach(free!, (alive, out))
end

# KL-07 resizing-and-raytracing.jl B.7 (eager launches over a kernel-written length): an eager dispatch over launchrange(alive) is indirect from the cell, with the count the kernel wrote.
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

# KL-08 api.md 2.2 (resize!, copy!), resizing-and-raytracing.jl B.3, B.5: resize! and copy! on an array whose length a kernel writes are scheduled as on any array (one coalesced Resize, then the Store); the run applies them in front of its kernels, and the kernel's count stands.
@case "KL-08" 4 begin
    dev = testdevice()
    cap = 4096
    alive = MantleArray(dev, Float32, 0; capacity = cap)
    g = lengthgraph(dev, alive, Int32(300))
    resize!(alive, 7)
    copy!(alive, Float32[1, 2, 3])
    @test [nameof(typeof(c)) for c in Mantle.pendingof(alive)] == [:Resize, :Store]
    run!(g)
    @test isempty(Mantle.pendingof(alive))
    @test length(alive) == 300
    free!(g); free!(alive)
end

# KL-09 api.md 2.2 (`Array(a)`: for a kernel-written length the cell is copied in the same submission): one thread runs g (a count, and that many elements of that value) 1000 times while another reads Array(alive): every result is a consistent (length, data) pair.
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

# KL-10: ordinary kernels consume prepared slots, not source Cells. Include
# a length writer/prepare chain so a constant empty audit cannot pass.
@case "KL-10" 4 begin
    dev = testdevice()
    alive = MantleArray(dev, Float32, 0; capacity = 4096)
    marks = zeroed(dev, 4096)
    g = lengthgraph(dev, alive, Int32(300))
    dispatch!(g, kl_mark!, (marks, alive), launchrange(alive))
    run!(g)
    access = Mantle.cellaccesses(g)
    @test any(a -> a.role === :lengthwriter && a.writes > 0, access)
    @test any(a -> a.role === :prepare && a.reads > 0, access)
    @test any(a -> a.role === :ordinary, access)
    @test all(a -> a.role !== :ordinary || (a.reads == 0 && a.writes == 0), access)
    @test launchcount(marks) == 300
    free!(g); free!(alive); free!(marks)
end

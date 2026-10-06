# Consumer patterns of JuliaVision's model calls, written against Mantle alone
# (recording-plan.md "One call, one graph"; api.md 2.2-2.5; examples.jl 11-21
# and 38-41; decisions.md B3.9, B3.13, B3.18).
#
# The models compute with small integers held as Float32 (or Int32), so the
# arithmetic is exact on every device whatever the driver fuses, and results
# are compared with `==` against a host model.
#
# Phase: the latest ledger phase among the contracts a case relies on. Every
# case here uses stores, hostwritten arrays, resize!, creation from data or
# eviction, which are phase 4 (the suite runs nothing before APIPHASE).
#
# What this file assumes beyond helpers.jl:
#   Mantle.steplog(dev) entries have `kind`, the step's name as a Symbol
#   (:Restore for an evicted array brought back), as in resize.jl.

using KernelAbstractions: @kernel, @index

const DNN_N = 4096          # elements of a layer

"""Small integers as `T`: sums and products of them are exact in Float32."""
smallints(T, n; seed = 1) = T.(rand(Xoshiro(seed), Int8(-8):Int8(8), n))

"""The weight value at `i` for `seed`, written on the device or into a file."""
weightvalue(i, seed) = Float32((i * seed) % 251 - 125)

"""Host model of one layer: relu(w[mod1(i, end)] * x[i] + b[1])."""
layermodel(w, b, x) = [max(w[mod1(i, length(w))] * x[i] + b[1], 0f0) for i in eachindex(x)]

"""Host model of `declaremodel!`: relu(2 * (w[mod1(i, end)] * x[i] + b[1]))."""
callmodel(w, b, x) = [max(2f0 * (w[mod1(i, length(w))] * x[i] + b[1]), 0f0) for i in eachindex(x)]

@kernel function dnn_affine!(h, x, w, b)
    i = @index(Global)
    @inbounds h[i] = w[mod1(i, length(w))] * x[i] + b[1]
end

@kernel function dnn_relu!(y, h)
    i = @index(Global)
    @inbounds y[i] = max(h[i], 0f0)
end

@kernel function dnn_gather!(out, w, idx)
    i = @index(Global)
    @inbounds out[i] = w[idx[i]]
end

@kernel function dnn_weightpattern!(w, seed)
    i = @index(Global)
    @inbounds w[i] = weightvalue(i, seed)
end

"""The weight index the `i`-th of `m` samples reads in an array of `n`."""
sampleindex(i, n, m) = (i - 1) * (n ÷ m) + 1

@kernel function dnn_sampleweights!(acc, w)
    i = @index(Global)
    @inbounds acc[i] += w[sampleindex(i, length(w), length(acc))]
end

@kernel function dnn_mask!(mask, x)
    i = @index(Global)
    @inbounds mask[i] = x[i] > 0f0 ? Int32(1) : Int32(0)
end

@kernel function dnn_scale!(y, x, s)
    i = @index(Global)
    @inbounds y[i] = s * x[i]
end

@kernel function dnn_accumulate!(acc, x)
    i = @index(Global)
    @inbounds acc[i] += x[i]
end

@kernel function dnn_nexttoken!(tokens, pos)
    p = pos[1]
    @inbounds tokens[p + 1] = (tokens[p] * Int32(31) + p) % Int32(1000)
end

"""The cache entry the step at position `p` appends in row `i`."""
kvvalue(i, p) = Float32((i + 3p) % 17 - 8)

@kernel function dnn_kvappend!(kv, pos)
    i = @index(Global)
    p = pos[1]
    @inbounds kv[i, p] = kvvalue(i, p)
end

# Reads the whole cache up to the step's position: what attention over the
# cache does, reduced to a sum.
@kernel function dnn_kvattend!(outs, kv, pos)
    p = pos[1]
    s = 0f0
    for j in 1:p, i in 1:size(kv, 1)
        @inbounds s += kv[i, j]
    end
    @inbounds outs[p] = s
end

"""An encoder module: declares its layer where its arrays live (example 15)."""
function dnn_encoder!(h, x, w, b)
    dispatch!(location(h, x, w, b), dnn_affine!, (h, x, w, b), length(h))
    return h
end

"""A decoder module: relu, a broadcast declared where its arrays live."""
dnn_decoder!(y, h) = (y .= max.(h, 0f0); y)

"""
One model call declared into `g`: two modules and the element-wise work
between them, so the call is one graph (plan, "One call, one graph"). Returns
the output, a device array the host reads.
"""
function declaremodel!(g, dev, x, w, b)
    h = MantleArray(g, Float32, length(x))
    y = MantleArray(dev, Float32, length(x))
    dnn_encoder!(h, x, w, b)
    h .= h .* 2f0
    dnn_decoder!(y, h)
    return y
end

"""A model whose `n` weights are written on the device; each run adds 1024 samples of them to `acc`."""
function residentmodel(dev, n, seed)
    w = MantleArray(dev, Float32, n)
    dispatch!(dev, dnn_weightpattern!, (w, Int32(seed)), n)
    acc = MantleArray(dev, Float32, 1024)
    fill!(acc, 0f0)
    g = Graph(dev)
    dispatch!(g, dnn_sampleweights!, (acc, w), length(acc))
    expected = [weightvalue(sampleindex(i, n, 1024), seed) for i in 1:1024]
    return (; g, w, acc, expected)
end

"""The host part of a TRELLIS-style call: the indices the device kept, and the length of the next stage."""
function selectsurvivors!(feats, mask, idx)
    keep = findall(!=(0), mask)
    idx[1:length(keep)] .= keep
    resize!(feats, length(keep))     # a change made during the run: applied in front of the next stage
    return
end

"""Host model of the autoregressive loop."""
function hosttokens(steps)
    t = zeros(Int32, steps + 1)
    t[1] = 1
    for p in 1:steps
        t[p + 1] = (t[p] * Int32(31) + Int32(p)) % Int32(1000)
    end
    return t
end

"""`f()` while another task samples `probe()` every millisecond: f's value and the largest sample."""
function withpeak(f, probe)
    done = Threads.Atomic{Bool}(false)
    peak = Threads.Atomic{Int}(probe())
    sampler = Threads.@spawn while !done[]
        Threads.atomic_max!(peak, probe())
        sleep(0.001)
    end
    v = try
        f()
    finally
        done[] = true
        wait(sampler)
    end
    return v, peak[]
end

"""Resident memory of this process in bytes (VmRSS); 0 where /proc is missing."""
function residentbytes()
    Sys.islinux() || return 0
    line = only(filter(startswith("VmRSS:"), readlines("/proc/self/status")))
    return 1024 * parse(Int, split(line)[2])
end

"""Writes `weightvalue.(1:n, 1)` as raw Float32 to `path`, a chunk at a time."""
function writeweights(path, n; chunk = 1 << 22)
    open(path, "w") do io
        for lo in 1:chunk:n
            write(io, weightvalue.(lo:min(lo + chunk - 1, n), 1))
        end
    end
end

# DNN-01: one graph whose input shape changes (api.md 2.2, resize! on a graph
# array). A shape within the size the graph arrays were placed for is a cell
# write; one beyond it makes the graph recompile at its next run. The first
# resize! of the fixed arrays also recompiles once, since the plan launched
# over them directly (derived from api.md 2.2: "Plans that launched over it
# directly recompile once"); here it is also a shape beyond the placement.
@case "DNN-01" 4 begin
    dev = testdevice()
    hw, hb = smallints(Float32, 512; seed = 1), Float32[3]
    w, b = MantleArray(dev, hw), MantleArray(dev, hb)
    g = Graph(dev)
    x = MantleArray(g, Float32, 64; hostwritten = true)
    h = MantleArray(g, Float32, 64)
    y = MantleArray(dev, Float32, 64)
    dispatch!(g, dnn_affine!, (h, x, w, b), launchrange(h))
    dispatch!(g, dnn_relu!, (y, h), launchrange(y))
    # (input length, plan compiles at its run)
    calls = ((64, 1), (64, 0), (128, 1), (128, 0), (96, 0), (64, 0), (512, 1), (300, 0))
    for (k, (n, compiles)) in enumerate(calls)
        hx = smallints(Float32, n; seed = 10 + k)
        n == length(x) || foreach(a -> resize!(a, n), (x, h, y))
        x[:] = hx
        _, d = counted(() -> run!(g))
        @test d.plancompiles == compiles
        @test Array(y) == layermodel(hw, hb, hx)
    end
    free!(g)
end

# DNN-02: one graph per input shape, cached by the caller (JuliaVision's
# per-shape plans). A new shape compiles one plan and no kernel (the kernels'
# bodies do not depend on the shape); a known shape compiles nothing. Each
# call is one run of one graph: one submission, no host wait in run!, the
# intermediate `h` never copied out (plan, "One call, one graph": each
# runner's per-call assertion).
@case "DNN-02" 4 begin
    dev = testdevice()
    hw, hb = smallints(Float32, 512; seed = 2), Float32[-1]
    w, b = MantleArray(dev, hw), MantleArray(dev, hb)
    calls = Dict{Int,Any}()
    for (k, n) in enumerate((64, 128, 64, 1000, 128, 1000, 64))
        known = haskey(calls, n)
        m = get!(calls, n) do
            g = Graph(dev)
            x = MantleArray(g, Float32, n; hostwritten = true)
            (; g, x, y = declaremodel!(g, dev, x, w, b))
        end
        hx = smallints(Float32, n; seed = 20 + k)
        m.x[:] = hx
        _, d = counted(() -> run!(m.g))
        @test d.plancompiles == (known ? 0 : 1)
        k > 1 && @test d.kernelcompiles == 0
        @test submitcount(d) == 1
        @test d.hostwaits == 0
        @test Array(m.y) == callmodel(hw, hb, hx)
    end
    foreach(m -> free!(m.g), values(calls))
end

# DNN-03: one graph over device arrays whose length changes per call, through
# scheduled changes (B3.13): `copy!` of the input (a resize keeping nothing and
# a store) and `resize!` of the output. Both arrays are resizable before the
# plan exists, so its launches over them are indirect and no shape recompiles,
# also one past the arrays' storage (a move: addresses reach the plan through
# cells).
@case "DNN-03" 4 begin
    dev = testdevice()
    hw, hb = smallints(Float32, 512; seed = 3), Float32[1]
    w, b = MantleArray(dev, hw), MantleArray(dev, hb)
    x, y = MantleArray(dev, Float32, 64), MantleArray(dev, Float32, 64)
    resize!(x, 0)
    resize!(y, 0)
    g = Graph(dev)
    dispatch!(g, dnn_affine!, (y, x, w, b), launchrange(y))
    dispatch!(g, dnn_relu!, (y, y), launchrange(y))
    for (k, n) in enumerate((64, 128, 64, 1000, 5000, 100, 5000))
        hx = smallints(Float32, n; seed = 30 + k)
        copy!(x, hx)
        resize!(y, n)
        @test length(y) == n                     # the host view at once
        @test !isempty(Mantle.pendingof(y))      # applied by the next submission that touches y
        _, d = counted(() -> run!(g))
        @test d.plancompiles == (k == 1 ? 1 : 0)
        @test Array(y) == layermodel(hw, hb, hx)
    end
    free!(g)
end

# DNN-04: weights from host data (B3.18; catalogue CR-01, CR-02, CR-05,
# CR-07). Uploaded at creation in chunks through upload regions (or written
# directly into host-visible storage): the pool never holds a second copy of
# the array, only the array and a few chunks. The data is taken at creation;
# a graph naming the weights right after creation reads them (its first run
# waits for the upload's writes).
@case "DNN-04" 4 begin
    dev = testdevice()
    before = memory(dev)
    n = (1 << 30) ÷ sizeof(Float32)                  # 1 GiB: many chunks on any device
    hw = smallints(Float32, n; seed = 4)
    positions = 1:9973:n
    expected = hw[positions]
    base = Mantle.reserved(dev)
    w, peak = withpeak(() -> Mantle.reserved(dev)) do
        MantleArray(dev, hw)
    end
    @test peak - base <= sizeof(hw) + sizeof(hw) ÷ 2
    fill!(hw, 0f0)                                   # the caller's memory is free to change after the call
    hw = nothing
    idx = MantleArray(dev, Int32.(positions))
    out = MantleArray(dev, Float32, length(positions))
    g = Graph(dev)
    dispatch!(g, dnn_gather!, (out, w, idx), length(out))
    run!(g)
    @test Array(out) == expected
    free!(g)
    foreach(free!, (w, idx, out))
    after = memory(dev)
    @test after.uploads == before.uploads            # every upload region given back
    @test after.cells == before.cells
end

# DNN-05: weights from a mapped file (B3.9, B3.18; catalogue CR-04, EVICT-25).
# Streamed from the file in chunks: the process's resident memory grows by the
# chunks in flight, not by the file. Read-only: a store into it throws.
# pending decision 5: `MantleArray(dev, path, …; mmap = true)` with the device
# as an argument (rule 3); the element type and dims are given as for
# `Mmap.mmap(io, Array{T,N}, dims)`, which the plan does not specify; a store
# throws ArgumentError at the call.
@case "DNN-05" 4 begin
    dev = testdevice()
    n = (1 << 30) ÷ sizeof(Float32)
    path = tempname()
    writeweights(path, n)
    positions = 1:9973:n
    rss = residentbytes()
    w, peak = withpeak(residentbytes) do
        MantleArray(dev, path, Vector{Float32}, (n,); mmap = true)      # pending decision 5
    end
    if Sys.islinux()
        @test peak - rss <= n * sizeof(Float32) ÷ 2
    else
        @test_skip "resident memory is read from /proc"
    end
    idx = MantleArray(dev, Int32.(positions))
    out = MantleArray(dev, Float32, length(positions))
    g = Graph(dev)
    dispatch!(g, dnn_gather!, (out, w, idx), length(out))
    run!(g)
    @test Array(out) == weightvalue.(positions, 1)
    @test_throws ArgumentError w[1:1] = Float32[0]                     # pending decision 5
    free!(g)
    foreach(free!, (w, idx, out))
    rm(path)
end

# DNN-06: two models whose weights do not fit together, run alternately back
# to back with no host wait between them (B3.6 "two DNNs that each use 90% of
# VRAM must both run", B3.9; api.md 6 test_eviction.jl; catalogue EVICT-01).
# The device answers a budget of 1 GiB (a FaultDevice, FI-9) and each model's
# weights take 60% of it, so every run restores its weights and evicts the
# other model's. Results exact; nothing recompiles, since addresses reach the
# plans through cells. A device that ignored the budget would pass without
# evicting: that the budget holds is the arena and eviction areas' concern.
@case "DNN-06" 4 :separateheap begin
    budget = 1 << 30
    dev = FaultDevice(testdevice())
    inject!(dev, :budget, Answer(1, budget))
    n = (6 * budget ÷ 10) ÷ sizeof(Float32)
    m1 = residentmodel(dev, n, 1)
    m2 = residentmodel(dev, n, 2)
    run!(m1.g)
    run!(m2.g)
    rounds = 8
    Mantle.steplog!(dev, true)
    _, d = counted(dev) do
        for _ in 1:rounds
            run!(m1.g)
            run!(m2.g)
        end
    end
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    @test d.plancompiles == 0
    @test d.kernelcompiles == 0
    @test count(s -> s.kind === :Restore, log) >= 2rounds
    @test Array(m1.acc) == (rounds + 1) .* m1.expected
    @test Array(m2.acc) == (rounds + 1) .* m2.expected
    foreach(m -> (free!(m.g); free!(m.w); free!(m.acc)), (m1, m2))
end

# DNN-07: a host node between two stages reads a length the device decided
# and sizes the next stage (TRELLIS: a sparse structure's survivors). Stage 1
# marks the elements to keep; the host function finds their indices and
# resizes the output, a change during the run that the next stage sees (plan,
# Ordering: "A change queued while a host function runs ... is ordered before
# the later stages and seen by them"). The output is resizable before the plan
# exists, so stage 2 launches over it indirectly. One submission per stage;
# the run waits on the host only for the host node's inputs.
@case "DNN-07" 4 begin
    dev = testdevice()
    n = DNN_N
    g = Graph(dev)
    x = MantleArray(g, Float32, n; hostwritten = true)
    mask = MantleArray(g, Int32, n)
    idx = MantleArray(g, Int32, n)
    feats = MantleArray(dev, Float32, n)
    resize!(feats, 0)
    dispatch!(g, dnn_mask!, (mask, x), n)
    dispatch!(g, HostCall((m, ix) -> selectsurvivors!(feats, m, ix)), (mask, idx); writes = (idx,))
    dispatch!(g, dnn_gather!, (feats, x, idx), launchrange(feats))
    for k in 1:8
        hx = smallints(Float32, n; seed = 70 + k)
        x[:] = hx
        _, d = counted(() -> run!(g))
        expected = filter(>(0f0), hx)
        @test length(feats) == length(expected)
        @test Array(feats) == expected
        if k > 1
            @test d.plancompiles == 0
            @test submitcount(d) == 2
            @test d.hostwaits == 1
        end
    end
    free!(g)
    free!(feats)
end

# DNN-08: eager element-wise work between two graph calls on shared device
# arrays (encode, adjust the features eagerly, decode), with no host read in
# between (plan, Ordering: the inbox; examples.jl 7 and 29). Each call's input
# is a store. Nothing waits on the host: stores, runs and eager calls are
# ordered on the GPU, and the decoder's sum counts every round exactly once.
@case "DNN-08" 4 begin
    dev = testdevice()
    n = DNN_N
    xin, feat = MantleArray(dev, Float32, n), MantleArray(dev, Float32, n)
    total = MantleArray(dev, Float32, n)
    fill!(total, 0f0)
    encode = Graph(dev)
    dispatch!(encode, dnn_scale!, (feat, xin, 2f0), n)
    decode = Graph(dev)
    dispatch!(decode, dnn_accumulate!, (total, feat), n)
    expected = zeros(Float32, n)
    function onecall(hx)
        xin[1:n] = hx
        run!(encode)
        feat .+= 1f0
        feat .*= 3f0
        run!(decode)
        expected .+= 3f0 .* (2f0 .* hx .+ 1f0)
    end
    onecall(smallints(Float32, n; seed = 80))       # compiles both plans and the eager kernels
    _, d = counted() do
        for k in 1:16
            onecall(smallints(Float32, n; seed = 80 + k))
        end
    end
    @test d.hostwaits == 0
    @test d.plancompiles == 0
    @test Array(total) == expected
    foreach(free!, (encode, decode))
end

# DNN-09: autoregressive generation with the host deciding between steps: one
# run per token, the step's position a one-element store (inline in the run's
# own submission, examples.jl 24), no host wait in the loop (plan, "One call,
# one graph", item 6).
@case "DNN-09" 4 begin
    dev = testdevice()
    steps = 256
    tokens = MantleArray(dev, vcat(Int32[1], zeros(Int32, steps)))
    pos = MantleArray(dev, Int32[1])
    g = Graph(dev)
    dispatch!(g, dnn_nexttoken!, (tokens, pos), 1)
    run!(g)
    _, d = counted() do
        for p in 2:steps
            pos[1:1] = Int32[p]
            run!(g)
        end
    end
    @test d.hostwaits == 0
    @test d.plancompiles == 0
    @test submitcount(d) == steps - 1
    @test Array(tokens) == hosttokens(steps)
    free!(g)
end

# DNN-10: a KV cache that grows by doubling its sequence dimension
# (examples.jl 39; api.md 2.2 resize!). Each step appends a column and reads
# every column so far. The growth is scheduled and applied by the step's run;
# past the cache's capacity it moves, and the new storage holds the old
# columns. The plan is compiled once and never rebuilt.
@case "DNN-10" 4 begin
    dev = testdevice()
    rows, steps = 64, 200
    kv = MantleArray(dev, Float32, rows, 16; capacity = rows * 64)
    pos = MantleArray(dev, Int32[0])
    outs = MantleArray(dev, Float32, steps)
    g = Graph(dev)
    dispatch!(g, dnn_kvappend!, (kv, pos), rows)
    dispatch!(g, dnn_kvattend!, (outs, kv, pos), 1)
    _, d = counted() do
        for p in 1:steps
            p > size(kv, 2) && resize!(kv, 2 * size(kv, 2))
            pos[1:1] = Int32[p]
            run!(g)
        end
    end
    @test d.plancompiles == 1
    @test d.hostwaits == 0
    @test size(kv) == (rows, 256)
    expected = [kvvalue(i, p) for i in 1:rows, p in 1:steps]
    @test Array(kv)[:, 1:steps] == expected
    @test Array(outs) == cumsum(vec(sum(expected; dims = 1)))
    free!(g)
end

# DNN-11: a model's output read with Array after each call reflects that call
# (api.md 2.2 Array: its own submission, one wait). Two calls before a read:
# the read reflects the later one.
@case "DNN-11" 4 begin
    dev = testdevice()
    n = DNN_N
    hw, hb = smallints(Float32, 512; seed = 11), Float32[2]
    w, b = MantleArray(dev, hw), MantleArray(dev, hb)
    xin = MantleArray(dev, Float32, n)
    g = Graph(dev)
    y = declaremodel!(g, dev, xin, w, b)
    for k in 1:8
        hx = smallints(Float32, n; seed = 110 + k)
        xin[1:n] = hx
        run!(g)
        r, d = counted(() -> Array(y))
        @test r == callmodel(hw, hb, hx)
        @test submitcount(d) == 1
        @test d.hostwaits == 1
    end
    earlier, later = smallints(Float32, n; seed = 120), smallints(Float32, n; seed = 121)
    xin[1:n] = earlier
    run!(g)
    xin[1:n] = later
    run!(g)
    @test Array(y) == callmodel(hw, hb, later)
    free!(g)
end

# Declaration rules (api.md 2.2-2.4; plan, Operations; internals.jl sections
# 2 and 10): where a call runs (location), what a graph array refuses, holds
# taken at declaration, copies out of a graph, temporaries, groups as values.

using LinearAlgebra: mul!

function decl_add!(dst, src, v)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds dst[i] = src[i] + v
    return nothing
end

function decl_inc!(a)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] += one(eltype(a))
    return nothing
end

"""dst[i] = a[i] + b[i], as the element type of dst."""
function decl_sum!(dst, a, b)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = eltype(dst)(a[i]) + eltype(dst)(b[i])
    return nothing
end

# The device side of lengthof (the argument through which a kernel writes an
# array's length into its cell) is not specified; `len[] = n` is the form
# assumed here.
function decl_spawn!(alive, len, src)
    i = KI.get_global_id().x
    i <= length(src) || return nothing
    @inbounds alive[i] = src[i]
    i == 1 && (len[] = length(src))
    return nothing
end

zeroed(dev, n) = (a = MantleArray(dev, Int32, n); fill!(a, Int32(0)); a)

"""A graph with one counter kernel, run once."""
function countergraph(dev)
    counter = zeroed(dev, 1)
    g = Graph(dev)
    dispatch!(g, decl_inc!, (counter,), 1)
    run!(g)
    return g, counter
end

# ── 4. Declaration rules (DECL) ──

# api.md 2.4 (location) and section 7; internals.jl section 2 (join): arguments
# from two graphs: ArgumentError, both graphs unchanged.
@case "DECL-01" 6 begin
    dev = testdevice(); n = 64
    g1, g2 = Graph(dev), Graph(dev)
    x1, x2 = MantleArray(g1, Int32, n), MantleArray(g2, Int32, n)
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    fill!(x1, Int32(1)); fill!(x2, Int32(2))
    @test_throws ArgumentError location(x1, x2)
    @test_throws ArgumentError (x1 .= x2)
    copyto!(out1, x1); copyto!(out2, x2)
    run!(g1); run!(g2)
    @test Array(out1) == fill(1, n)
    @test Array(out2) == fill(2, n)
end

# api.md 2.4 (location); plan, Operations ("Where a call runs"): a graph array
# and a device array of the same device: the call declares into the graph,
# which takes one hold on the device array (it keeps the array after free!(a)
# and gives it back with free!(g)).
@case "DECL-02" 6 begin
    dev = testdevice(); n = 16 << 20
    out = MantleArray(dev, Int32, n)
    baseline = memory()
    a = MantleArray(dev, fill(Int32(3), n))
    g = Graph(dev)
    x = MantleArray(g, Int32, n)
    @test location(x, a) === location(x)
    x .= a .+ Int32(1)
    copyto!(out, x)
    free!(a)
    run!(g)
    @test all(==(4), Array(out))
    free!(g)
    @test memory().reserved <= baseline.reserved
end

# internals.jl section 2 (join): a graph on one device and a GPURef of
# another: "two devices".
@case "DECL-03" 6 :twodevices begin
    dev = testdevice()
    other = first(d for d in SUITE.fixtures[].devices if d !== dev)
    x = MantleArray(Graph(dev), Float32, 16)
    @test_throws ArgumentError location(x, GPURef(other, 1f0))
end

# api.md 2.2 (KA.get_backend on a graph array), 2.5 (per-run host data) and
# section 7: getindex, iterate, Array and collect on a graph array throw;
# show prints metadata without a submission. A store into it throws; per-run host data is a
# device array written by a store, which the graph copies in.
@case "DECL-04" 6 begin
    dev = testdevice(); n = 16
    g = Graph(dev)
    x = MantleArray(g, Float32, n)
    @test_throws Exception x[1]
    @test_throws Exception iterate(x)
    @test_throws Exception Array(x)
    @test_throws Exception collect(x)
    shown, d = counted(() -> sprint(show, MIME"text/plain"(), x), dev)
    @test !isempty(shown)
    @test submitcount(d) == 0
    @test d.hostwaits == 0
    data = testdata(Float32, n)
    @test_throws Exception (x[:] = data)
    h = MantleArray(dev, Float32, n)                # per-run host data: a device array written by stores
    h[:] = data
    copyto!(x, h)
    out = MantleArray(dev, Float32, n)
    copyto!(out, x)
    run!(g)
    @test Array(out) == data
end

# internals.jl section 2 (declare!: usable first) and section 10 (usable):
# declaring with a freed device array throws; no node is added (the next run
# compiles nothing) and no hold (the memory comes back).
@case "DECL-05" 2 begin
    dev = testdevice(); n = 16 << 20
    g, counter = countergraph(dev)
    baseline = memory()
    a = MantleArray(dev, Int32, n)
    free!(a)
    @test_throws ArgumentError dispatch!(g, decl_inc!, (a,), n)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test Array(counter) == [2]
    @test memory().reserved <= baseline.reserved
end

# api.md section 7; plan, Operations ("Lengths decided on the GPU"): a
# device-decided length needs no capacity: lengthof(a) is accepted on a graph
# and eagerly, the kernel writing within a's current storage; with a capacity
# it is accepted too.
@case "DECL-06" 4 begin
    dev = testdevice(); n = 64
    src = MantleArray(dev, Int32.(1:n))
    alive = MantleArray(dev, Int32, n)
    eager = MantleArray(dev, Int32, n)
    g = Graph(dev)
    dispatch!(g, decl_spawn!, (alive, lengthof(alive), src), n)
    dispatch!(dev, decl_spawn!, (eager, lengthof(eager), src), n)
    @test Array(eager) == 1:n
    withcap = MantleArray(dev, Int32, 0; capacity = n)
    dispatch!(g, decl_spawn!, (withcap, lengthof(withcap), src), n)
    run!(g)
    @test Array(alive) == 1:n
    @test Array(withcap) == 1:n
end

# internals.jl section 10 (name!) and section 2 (delete!): one device array as
# a launch argument, a view parent and the when! flag (a one-element view) of
# one node in a group: one hold per graph, counted per node, kept after
# free!(a) and dropped once when delete!(g, h) retires the plan. The
# Broadcasted-leaf role is not covered: a broadcast declares into the graph,
# not a group. The keyword-argument role is not covered: no documented
# operation takes an array keyword.
@case "DECL-07" 2 begin
    dev = testdevice(); n = 4 << 20
    out = MantleArray(dev, Int32, n)
    g, _ = countergraph(dev)
    baseline = memory()
    a = MantleArray(dev, ones(Bool, n))             # a[1] is true: the flag is set
    h = Group(g, "block")
    when!(h, view(a, 1:1), decl_sum!, (out, a, view(a, :)), n)
    free!(a)
    run!(g)
    @test Array(out) == fill(2, n)
    delete!(g, h)
    run!(g)
    @test memory().reserved <= baseline.reserved
end

# api.md 2.4 (selection: in a Group only kernel candidates qualify); plan,
# Operations ("Selection is a call in the method"): mul! where a foreign
# candidate (cuBLAS) is best: in the graph any candidate, in a repeated group a
# kernel candidate (a foreign node there would make the declaration throw).
# Both results are right. How an operation reaches a group is open
# (internals.jl section 2); the case passes the group first, as examples.jl 19
# does.
@case "DECL-08" 10 :vendorview begin
    dev = testdevice(); m = 1024
    hA = reshape(testdata(Float32, m * m), m, m)
    hB = reshape(testdata(Float32, m * m; seed = 2), m, m)
    A, B = MantleArray(dev, hA), MantleArray(dev, hB)
    g = Graph(dev)
    c = [MantleArray(g, Float32, m, m) for _ in 1:2]
    mul!(c[1], A, B)
    loop = Group(g, "loop")
    mul!(loop, c[2], A, B)
    repeat!(loop, 1)
    outs = [MantleArray(dev, Float32, m, m) for _ in 1:2]
    foreach(copyto!, outs, c)
    run!(g)
    reference = hA * hB
    @test all(o -> Array(o) ≈ reference, outs)
end

# api.md 2.4 (copyto! with x on a graph), 2.5 (per-run host data):
# copyto!(out::device, y::graph); runs with new data stored into the device
# array x: the copy kernel is recorded once and Array(out) shows each run's
# result.
@case "DECL-09" 6 begin
    dev = testdevice(); n = 256
    g = Graph(dev)
    x = MantleArray(dev, Float32, n)                # per-run host data: a device array written by stores
    y = MantleArray(g, Float32, n)
    out = MantleArray(dev, Float32, n)
    y .= x .* 2f0
    copyto!(out, y)
    for k in 1:3
        data = testdata(Float32, n; seed = k)
        x[:] = data
        _, d = counted(() -> run!(g))
        k > 1 && @test d.records == 0
        @test Array(out) == data .* 2f0
    end
end

# api.md 2.2 (MantleArray(location(a), T, dims...) do t … end): on a device a
# region retired with the call's token (a canary allocated right after is not
# written by the call's pending work; the memory comes back); on a graph a
# transient.
@case "DECL-10" 6 begin
    dev = testdevice(); n = 1 << 20
    out = MantleArray(dev, Int32, n)
    baseline = memory()
    MantleArray(dev, Int32, n) do t
        fill!(t, Int32(3))
        copyto!(out, t)
    end
    c = canary(dev, n)
    @test all(==(3), Array(out))
    @test intact(c)
    free!(c)
    @test memory().reserved <= baseline.reserved
    g = Graph(dev)
    out2 = MantleArray(dev, Int32, n)
    MantleArray(g, Int32, n) do t
        fill!(t, Int32(4))
        copyto!(out2, t)
    end
    run!(g)
    @test all(==(4), Array(out2))
    @test Mantle.naivebytes(g) >= sizeof(Int32) * n
end

# plan, Operations ("Where a call runs", "Groups are places in the order":
# no mode state on the graph): a group is a value, not a mode. A call whose
# arguments are all device arrays runs now on the device, also after
# Group(g, …) and repeat!; work reaches a group only by passing it. The graph's
# own work is unchanged.
@case "DECL-11" 1 begin
    dev = testdevice(); n = 64
    a = zeroed(dev, n)
    g, counter = countergraph(dev)
    loop = Group(g, "loop")
    repeat!(loop, 3)
    fill!(a, Int32(7))
    @test Array(a) == fill(7, n)
    run!(g)
    @test Array(a) == fill(7, n)
    @test Array(counter) == [2]
end

# A callable's fields are arguments too: captured resources need graph holds
# and ordering even when the explicit args tuple contains only the output.
@case "DECL-12" 6 begin
    dev = testdevice()
    hidden = MantleArray(dev, Int32[3, 5, 7, 9])
    out = MantleArray(dev, Int32, 4)
    kernel = let source = hidden
        function (dst)
            i = KI.get_global_id().x
            i <= length(dst) || return nothing
            dst[i] = source[i] + Int32(1)
            nothing
        end
    end
    g = Graph(dev)
    dispatch!(g, kernel, (out,), 4)
    free!(hidden)
    run!(g)
    @test Array(out) == Int32[4, 6, 8, 10]
    free!(g); free!(out)
end

@case "DECL-13" 6 :twodevices begin
    dev = testdevice()
    other = first(d for d in SUITE.fixtures[].devices if d !== dev)
    foreign = MantleArray(other, Int32[99])
    kernel = let source = foreign
        dst -> (dst[1] = source[1]; nothing)
    end
    out = MantleArray(dev, Int32[0])
    g = Graph(dev)
    @test_throws ArgumentError dispatch!(g, kernel, (out,), 1)
    dispatch!(g, decl_inc!, (out,), 1)
    run!(g)
    @test Array(out) == Int32[1]
    free!(g); free!(out); free!(foreign)
end

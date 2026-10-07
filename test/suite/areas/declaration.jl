# Declaration rules (catalogue-core.md, table 4): where a call runs
# (location), what a graph array refuses, holds taken at declaration, copies
# out of a graph, temporaries.

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: mul!

@kernel function decl_add!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] + v
end

@kernel function decl_inc!(a)
    i = @index(Global)
    @inbounds a[i] += one(eltype(a))
end

# The device side of lengthof (the argument through which a kernel writes an
# array's length into its cell) is not specified; `len[] = n` is the form
# assumed here.
@kernel function decl_spawn!(alive, len, @Const(src))
    i = @index(Global)
    @inbounds alive[i] = src[i]
    i == 1 && (len[] = length(src))
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

# A:127, I:196-202: arguments from two graphs: ArgumentError, both graphs unchanged.
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

# A:127, P:342-345: a graph array and a device array of the same device: the
# call declares into the graph, which takes one hold on the device array (it
# keeps the array after free!(a) and gives it back with free!(g)).
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

# I:198: a graph on one device and a GPURef of another: "two devices".
@case "DECL-03" 6 :lavapipe begin
    dev = testdevice()
    other = first(d for d in devices() if d !== dev)
    x = MantleArray(Graph(dev), Float32, 16)
    @test_throws ArgumentError location(x, GPURef(other, 1f0))
end

# A:92, A:477: getindex, iterate, Array, collect and show on a graph array
# throw (error types unspecified); x[:] = data on a hostwritten one is allowed.
@case "DECL-04" 6 begin
    dev = testdevice(); n = 16
    g = Graph(dev)
    x = MantleArray(g, Float32, n)
    @test_throws Exception x[1]
    @test_throws Exception iterate(x)
    @test_throws Exception Array(x)
    @test_throws Exception collect(x)
    @test_throws Exception sprint(show, MIME"text/plain"(), x)
    h = MantleArray(g, Float32, n; hostwritten = true)
    data = testdata(Float32, n)
    h[:] = data
    out = MantleArray(dev, Float32, n)
    copyto!(out, h)
    run!(g)
    @test Array(out) == data
end

# I:213, I:2060: declaring with a freed device array throws; no node is added
# (the next run compiles nothing) and no hold (the memory comes back).
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

# A:76, A:476: lengthof(a) for an array without a capacity is refused at
# declaration, on a graph and eagerly; with a capacity it is accepted.
@case "DECL-06" 4 begin
    dev = testdevice(); n = 64
    src = MantleArray(dev, Int32.(1:n))
    alive = MantleArray(dev, Int32, 0)
    g = Graph(dev)
    @test_throws ArgumentError dispatch!(g, decl_spawn!, (alive, lengthof(alive), src), n)
    @test_throws ArgumentError dispatch!(dev, decl_spawn!, (alive, lengthof(alive), src), n)
    withcap = MantleArray(dev, Int32, 0; capacity = n)
    dispatch!(g, decl_spawn!, (withcap, lengthof(withcap), src), n)
    run!(g)
    @test Array(withcap) == 1:n
end

# P:1304-1307: one device array as Broadcasted leaf, view parent and when! flag
# (a view) of one node in a block: one hold per graph, counted per node, kept
# after free!(a) and dropped once when delete!(g, h) retires the plan. The
# keyword-argument role of the row is not covered: no documented operation
# takes an array keyword.
@case "DECL-07" 2 begin
    dev = testdevice(); n = 4 << 20
    out = MantleArray(dev, Int32, n)
    g, _ = countergraph(dev)
    baseline = memory()
    a = MantleArray(dev, Int32.(1:n))               # a[1] = 1: the flag is nonzero
    h = insert!(g) do
        x = MantleArray(g, Int32, n)
        when!(g, view(a, 1:1)) do
            x .= a .+ view(a, :)
        end
        copyto!(out, x)
    end
    free!(a)
    run!(g)
    @test Array(out) == 2 .* (1:n)
    delete!(g, h)
    run!(g)
    @test memory().reserved <= baseline.reserved
end

# A:134-136, I:413-426: mul! where a foreign candidate (cuBLAS) is best: outside
# a block any candidate, inside repeat! and when! a kernel candidate (a foreign
# node there would make the declaration throw). All three results are right.
@case "DECL-08" 10 :vendorview begin
    dev = testdevice(); m = 1024
    hA = reshape(testdata(Float32, m * m), m, m)
    hB = reshape(testdata(Float32, m * m; seed = 2), m, m)
    A, B = MantleArray(dev, hA), MantleArray(dev, hB)
    flag = MantleArray(dev, Int32[1])
    g = Graph(dev)
    c = [MantleArray(g, Float32, m, m) for _ in 1:3]
    mul!(c[1], A, B)
    repeat!(g, 1) do i
        mul!(c[2], A, B)
    end
    when!(g, flag) do
        mul!(c[3], A, B)
    end
    outs = [MantleArray(dev, Float32, m, m) for _ in 1:3]
    foreach(copyto!, outs, c)
    run!(g)
    reference = hA * hB
    @test all(o -> Array(o) ≈ reference, outs)
end

# A:132, P:974-976: copyto!(out::device, x::graph); runs with new hostwritten
# data: the copy node is recorded once and Array(out) shows each run's result.
@case "DECL-09" 6 begin
    dev = testdevice(); n = 256
    g = Graph(dev)
    x = MantleArray(g, Float32, n; hostwritten = true)
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

# A:77, I:2141-2142: MantleArray(l, T, n) do t … end. On a device: a region
# retired with the call's token (a canary allocated right after is not written
# by the call's pending work; the memory comes back). On a graph: a transient.
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

# A:127 + A:130: inside when!(g, flag) or a repeat! body, a call whose arguments
# are all device arrays.
# pending decision 3: it throws at declaration (nothing runs eagerly, the
# graph is unchanged).
@case "DECL-11" 1 begin
    dev = testdevice(); n = 64
    a = zeroed(dev, n)
    flag = MantleArray(dev, Int32[1])
    g, counter = countergraph(dev)
    @test_throws ArgumentError when!(g, flag) do
        fill!(a, Int32(7))
    end
    @test_throws ArgumentError repeat!(g, 3) do i
        fill!(a, Int32(7))
    end
    run!(g)
    @test Array(a) == zeros(n)
    @test Array(counter) == [2]
end

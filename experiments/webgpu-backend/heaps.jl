# Arrays in different heaps of one device. Heap 0 is filled first, so the arrays
# made after it land in heap 1, and kernels read one heap and write the other.
import KernelAbstractions as KA
using Test

const MW = Base.get_extension(Mantle, :MantleWebGPUExt)

KA.@kernel function heaps_hist!(h, @Const(x))
    i = KA.@index(Global)
    @inbounds KA.@atomic h[x[i]] += UInt32(1)
end

KA.@kernel function heaps_scale!(o, @Const(a))
    i = KA.@index(Global)
    @inbounds o[i] = 2f0 * a[i]
end

"""Allocate the rest of heap 0, in arrays the caller frees."""
function fillheap0(d)
    fillers = Any[]
    while true
        free = d.allocators[1].free
        isempty(free) && break
        push!(fillers, KA.allocate(d, UInt8, maximum(length, free)))
        @assert MW.heapindex(fillers[end]) == 0
    end
    return fillers
end

@testset "heaps" begin
    @test length(dev.heaps) >= 2
    n = 10_000
    x = Mantle.devicearray(dev, Float32.(1:n))           # heap 0
    fillers = fillheap0(dev)
    y = KA.allocate(dev, Float32, n)                      # heap 1
    @test (MW.heapindex(x), MW.heapindex(y)) == (0, 1)

    y .= x .+ 1f0                                         # reads heap 0, writes heap 1
    @test Array(y) == Float32.(2:n + 1)
    x .= y .* 2f0                                         # and back
    @test Array(x) == Float32.(4:2:2n + 2)

    # Atomics and sub-word stores in heap 1.
    keys = Mantle.devicearray(dev, Int32[mod1(i, 7) for i in 1:n])
    h = KA.zeros(dev, UInt32, 7)
    @test MW.heapindex(h) == 1
    heaps_hist!(dev, 64)(h, keys; ndrange = n)
    @test Array(h) == UInt32[count(==(k), mod1.(1:n, 7)) for k in 1:7]
    b = KA.allocate(dev, UInt8, 37)
    b .= UInt8(5)
    @test Array(b) == fill(UInt8(5), 37)

    # A struct across heaps, and a recorded plan whose transients are in heap 1.
    z = Mantle.devicearray(dev, ComplexF32[i + 2i * im for i in 1:100])
    w = similar(z)
    w .= conj.(z)
    @test Array(w) == conj.(ComplexF32[i + 2i * im for i in 1:100])
    g = Mantle.Graph(dev)
    t = Mantle.Transient.Buffer(g, Float32, n)
    out = Mantle.Buffer(dev, Float32, n)
    Mantle.dispatch!(g, heaps_scale!, (t, x), n; name = "scale")
    Mantle.dispatch!(g, heaps_scale!, (out, t), n; name = "again")
    pl = Mantle.record!(Mantle.Plan(g))
    Mantle.run!(pl)
    @test Array(Mantle.storage(out)) == 4 .* Float32.(4:2:2n + 2)
    @test MW.heapindex(Mantle.storage(t)) == 1
    Mantle.free!(pl); Mantle.free!(out)

    foreach(MW.GPUArrays.unsafe_free!, fillers)
end

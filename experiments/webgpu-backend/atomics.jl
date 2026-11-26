# Atomics on Mantle's WebGPU backend: native WGSL atomics on the heap's atomic view,
# compare-and-swap loops for what WGSL lacks, and atomics on workgroup memory.
isdefined(Main, :dev) || include(joinpath(@__DIR__, "setup.jl"))

KA.@kernel function histogram!(h, x)
    i = KA.@index(Global, Linear)
    @inbounds b = x[i] % UInt32(length(h)) + 1
    KA.@atomic h[b] += UInt32(1)
end

# Float32 add has no WGSL atomic: a compare-and-swap loop.
KA.@kernel function atomicsum!(acc, x)
    i = KA.@index(Global, Linear)
    @inbounds KA.@atomic acc[1] += x[i]
end

# Signed max neither: WGSL's atomicMax on the u32 view would compare unsigned.
KA.@kernel function atomicmax!(m, x)
    i = KA.@index(Global, Linear)
    @inbounds KA.@atomic m[1] max x[i]
end

KA.@kernel function groupcount!(out)
    li = KA.@index(Local, Linear)
    gi = KA.@index(Group, Linear)
    c = KA.@localmem UInt32 (1,)
    if li == 1
        @inbounds c[1] = UInt32(0)
    end
    KA.@synchronize
    KA.@atomic c[1] += UInt32(1)
    KA.@synchronize
    if li == 1
        @inbounds out[gi] = c[1]
    end
end

@testset "WebGPU atomics" begin
    x = rand(UInt32(0):UInt32(1_000_000), 1 << 20)
    dx = copyto!(KA.allocate(dev, UInt32, length(x)), x)
    h = KA.zeros(dev, UInt32, 64)
    histogram!(dev)(h, dx; ndrange = length(x))
    want = zeros(UInt32, 64)
    for v in x
        want[v % 64 + 1] += 1
    end
    @test Array(h) == want

    f = rand(Float32, 4096)
    df = copyto!(KA.allocate(dev, Float32, length(f)), f)
    acc = KA.zeros(dev, Float32, 1)
    atomicsum!(dev)(acc, df; ndrange = length(f))
    @test Array(acc)[1] ≈ sum(f) rtol = 1e-4

    s = rand(Int32(-1000):Int32(1000), 10_000)
    ds = copyto!(KA.allocate(dev, Int32, length(s)), s)
    m = copyto!(KA.allocate(dev, Int32, 1), Int32[typemin(Int32)])
    atomicmax!(dev)(m, ds; ndrange = length(s))
    @test Array(m)[1] == maximum(s)

    out = KA.zeros(dev, UInt32, 100)
    groupcount!(dev, 128)(out; ndrange = 128 * 100)
    @test all(==(UInt32(128)), Array(out))
end

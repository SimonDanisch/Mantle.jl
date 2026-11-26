# The first kernels on Mantle's WebGPU backend: an Electron page as the device,
# KernelAbstractions kernels compiled by Lava for WebGPU and printed as WGSL.
isdefined(Main, :dev) || include(joinpath(@__DIR__, "setup.jl"))

KA.@kernel function vadd!(c, a, b)
    i = KA.@index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end

KA.@kernel function saxpy!(y, α, x)
    i = KA.@index(Global, Linear)
    @inbounds y[i] = α * x[i] + y[i]
end

KA.@kernel function rowsum!(out, A)
    j = KA.@index(Global, Linear)
    acc = zero(eltype(out))
    for i in axes(A, 1)
        @inbounds acc += A[i, j]
    end
    @inbounds out[j] = acc
end

# Workgroup memory, and barriers inside a loop: WGSL's uniformity analysis has to
# accept the loop condition.
KA.@kernel function blocksum!(out, a)
    li = KA.@index(Local, Linear)
    gi = KA.@index(Group, Linear)
    i = KA.@index(Global, Linear)
    shared = KA.@localmem Float32 (256,)
    @inbounds shared[li] = a[i]
    KA.@synchronize
    s = 128
    while s > 0
        if li <= s
            @inbounds shared[li] += shared[li + s]
        end
        KA.@synchronize
        s >>= 1
    end
    if li == 1
        @inbounds out[gi] = shared[1]
    end
end

@testset "WebGPU first kernels" begin
    n = 1_000_003
    a, b = rand(Float32, n), rand(Float32, n)
    da, db = KA.allocate(dev, Float32, n), KA.allocate(dev, Float32, n)
    copyto!(da, a); copyto!(db, b)
    dc = KA.zeros(dev, Float32, n)
    vadd!(dev)(dc, da, db; ndrange = n)
    KA.synchronize(dev)
    @test Array(dc) == a .+ b

    y = rand(Float32, n)
    dy = copyto!(KA.allocate(dev, Float32, n), y)
    saxpy!(dev)(dy, 2f0, da; ndrange = n)
    @test Array(dy) ≈ 2f0 .* a .+ y

    A = rand(Float32, 37, 1000)
    dA = copyto!(KA.allocate(dev, Float32, size(A)...), A)
    dout = KA.zeros(dev, Float32, 1000)
    rowsum!(dev)(dout, dA; ndrange = 1000)
    @test Array(dout) ≈ vec(sum(A; dims = 1))

    x = rand(Float32, 256 * 1000)
    dx = copyto!(KA.allocate(dev, Float32, length(x)), x)
    dsum = KA.zeros(dev, Float32, 1000)
    blocksum!(dev, 256)(dsum, dx; ndrange = length(x))
    @test Array(dsum) ≈ vec(sum(reshape(x, 256, 1000); dims = 1))
end

@testset "WebGPU recorded dispatch timing" begin
    ext = Base.get_extension(Mantle, :MantleWebGPUExt)
    if "timestamp-query" in dev.info["features"]
        a = Mantle.Buffer(dev, ones(Float32, 256))
        b = Mantle.Buffer(dev, zeros(Float32, 256))
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, saxpy!, (b, 2f0, a), 256; group = 256, name = "first")
        Mantle.dispatch!(g, saxpy!, (b, 3f0, a), 256; group = 256, name = "second")
        plan = Mantle.record!(Mantle.Plan(g))
        try
            timing = ext.profiledispatches(() -> Mantle.run!(plan), dev)
            @test [sample.name for sample in timing.samples] == ["first", "second"]
            @test all(s -> s.ns !== nothing && s.ns >= 0, timing.samples)
            @test Array(b) == fill(5f0, 256)
            Mantle.run!(plan)
            @test Array(b) == fill(10f0, 256)
        finally
            Mantle.free!(plan)
            foreach(Mantle.free!, (a, b))
        end
    end
end

KA.@kernel function scalar_update!(a, fail)
    fail && error("scalar update failed")
    @inbounds a[1] += Int32(1)
end

@testset "WebGPU synchronized scalar download" begin
    a = Mantle.Buffer(dev, Int32[0])
    try
        scalar_update!(dev, 1)(Mantle.storage(a), false; ndrange = 1)
        before = Mantle.submissions(dev)
        @test Mantle.readscalar(a) == 1
        @test Mantle.submissions(dev) == before + 1
        scalar_update!(dev, 1)(Mantle.storage(a), true; ndrange = 1)
        @test_throws ErrorException Mantle.readscalar(a)
        scalar_update!(dev, 1)(Mantle.storage(a), false; ndrange = 1)
        @test Mantle.readscalar(a) == 2
    finally
        Mantle.free!(a)
    end
end

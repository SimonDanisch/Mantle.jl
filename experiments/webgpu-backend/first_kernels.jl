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

@testset "WebGPU placement uses contiguous heap space" begin
    ext = Base.get_extension(Mantle, :MantleWebGPUExt)
    pool = Mantle.Pool()
    kind = Mantle.Buffers()
    span = Mantle.blocksize(dev)
    blocks = ext.HeapBlock[]
    try
        for _ in dev.heaps
            push!(blocks, Mantle.rawalloc(dev, kind, Mantle.maxalloc(dev) - span, nothing))
        end
        @test Mantle.arenaroom(pool, dev, kind) == span
        @test ext.heapfree(dev) == length(dev.heaps) * span
        region = Mantle.reserve!(pool, dev, kind, (), span; blocksize = span)
        try
            @test Mantle.arenaroom(pool, dev, kind) == length(region) == span
            @test ext.heapfree(dev) == (length(dev.heaps) - 1) * span
        finally
            Mantle.release!(region)
        end
    finally
        Mantle.trim!(pool, dev)
        foreach(block -> Mantle.rawfree(dev, block), blocks)
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

@testset "Packed WebGPU stores declare logical writes" begin
    a = Mantle.Buffer(dev, Float16[1, 2, 3, 4])
    b = Mantle.Buffer(dev, Float16[4, 3, 2, 1])
    c = Mantle.Buffer(dev, Float16, 4)
    try
        @test Mantle.argument_usage(dev, vadd!, (c, a, b), 4, nothing) ==
              [Mantle.WRITE, Mantle.READ, Mantle.READ]
        @test Mantle.argument_usage(dev, saxpy!, (c, Float16(2), a), 4, nothing) ==
              [Mantle.OPAQUE, Mantle.NOTOUCH, Mantle.READ]
        vadd!(dev, 4)(Mantle.storage(c), Mantle.storage(a), Mantle.storage(b); ndrange = 4)
        @test Array(Mantle.storage(c)) == fill(Float16(5), 4)
    finally
        foreach(Mantle.free!, (a, b, c))
    end
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

@testset "WebGPU long recordings preserve queue order" begin
    a = Mantle.Buffer(dev, Float32[0])
    b = Mantle.Buffer(dev, Float32[0])
    graph = Mantle.Graph(dev)
    for _ in 1:20
        Mantle.dispatch!(graph, saxpy!, (a, 1f0, b), 1; group = 1)
        Mantle.dispatch!(graph, saxpy!, (b, 1f0, a), 1; group = 1)
    end
    plan = Mantle.record!(Mantle.Plan(graph))
    try
        for seed in (1f0, 2f0)
            fill!(Mantle.storage(a), seed)
            fill!(Mantle.storage(b), seed + 1f0)
            x, y = seed, seed + 1f0
            for _ in 1:20
                x += y
                y += x
            end
            Mantle.run!(plan)
            @test Array(a) == [x]
            @test Array(b) == [y]
        end
    finally
        Mantle.free!(plan)
        foreach(Mantle.free!, (a, b))
    end
end

@testset "Broadcast over strided WebGPU views" begin
    values = reshape(Float32.(1:12), 4, 3)
    a = Mantle.Buffer(dev, values)
    try
        data = Mantle.storage(a)
        left = view(data, 1:2, :)
        right = view(data, 3:4, :)
        sumview = left .+ right
        @test KA.get_backend(sumview) === dev
        @test Array(sumview) == values[1:2, :] .+ values[3:4, :]
    finally
        Mantle.free!(a)
    end
end

KA.@kernel function mark_dispatched_groups!(groups)
    i = KA.@index(Group, Linear)
    lane = KA.@index(Local, Linear)
    if lane == 1
        @inbounds groups[i] += Int32(1)
    end
end

@testset "WebGPU counted replay clamps and skips gated dispatches" begin
    count = Mantle.Buffer(dev, Int32[77, 0])
    active = Mantle.Buffer(dev, Int32[1])
    groups = Mantle.Buffer(dev, zeros(Int32, 8))
    graph = Mantle.Graph(dev)
    Mantle.repeat!(graph, 2; while_nonzero = active) do _
        Mantle.dispatch!(graph, mark_dispatched_groups!, (groups,),
                         Mantle.DeviceRange(count; max = 512, at = 2); group = 64)
    end
    plan = Mantle.record!(Mantle.Plan(graph))
    try
        for (enabled, n, launched) in ((1, 129, 3), (0, 129, 0), (1, -1, 0), (1, 2048, 8))
            copyto!(Mantle.storage(active), Int32[enabled])
            copyto!(Mantle.storage(count), Int32[77, n])
            fill!(Mantle.storage(groups), 0)
            Mantle.run!(plan)
            @test Array(groups) == Int32[i <= launched ? 2 : 0 for i in 1:8]
        end
    finally
        Mantle.free!(plan)
        foreach(Mantle.free!, (count, active, groups))
    end
end

KA.@kernel function scalar_update!(a, fail)
    fail && error("scalar update failed")
    @inbounds a[1] += Int32(1)
end

@testset "WebGPU prepares a pipeline without running its kernel" begin
    a = Mantle.Buffer(dev, Int32[7])
    ext = Base.get_extension(Mantle, :MantleWebGPUExt)
    try
        data = Mantle.storage(a)
        before = Mantle.submissions(dev)
        builds = ext.capture_compute_pipelines(dev) do
            scalar_update!(dev, 1)(data, false; ndrange=1)
        end
        only(builds)()
        @test Mantle.submissions(dev) == before
        @test Mantle.readscalar(a) == 7
        scalar_update!(dev, 1)(data, false; ndrange=1)
        @test Mantle.readscalar(a) == 8
    finally
        Mantle.free!(a)
    end
end

@testset "WebGPU accepts a successful pipeline reply without cache metadata" begin
    ext = Base.get_extension(Mantle, :MantleWebGPUExt)
    timings = ext.record_pipeline_timings!(dev)
    try
        reply = Dict("ok" => true, "messages" => Any[])
        ext.check_compute_pipeline(reply, :legacy, "")
        ext.log_compute_pipeline(dev, reply, 0, :legacy, 0, 0.0)
        @test !only(timings).cached
    finally
        delete!(dev.info, :pipeline_timings)
    end
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

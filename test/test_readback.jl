using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))

@kernel function rb_write!(out, value)
    i = @index(Global, Linear)
    @inbounds out[i] = eltype(out)(i + value[1])
end

function rb_plan(dev, ::Type{T}, dims) where {T}
    g = Mantle.Graph(dev)
    a = Mantle.Transient.Buffer(g, T, dims)
    k = Mantle.GPURef(dev, Int32(0))
    Mantle.dispatch!(g, rb_write!, (a, k), length(a))
    return g, a, k
end

@testset "graph readback futures" begin
    dev = Mantle.Device(TESTBACKEND)
    @test Mantle.Readback !== KernelAbstractions.CPU

    @testset "one run including stores and multiple shaped outputs, no fetch submission" begin
        g, a, k = rb_plan(dev, Int32, (19, 27))
        v = Mantle.viewof(a, (7, 9); offset=33)
        n = Mantle.submissions(dev)
        f = Mantle.Readback(a)
        fv = Mantle.Readback(v)
        @test Mantle.submissions(dev) == n
        @test !isready(f)
        @test !isready(fv)
        pl = Mantle.Plan(g; budget=Inf)
        # Poison every destination on the host before submission: a skipped
        # transfer must fail even if the pool happens to hold an earlier result.
        for r in g.readbacks
            fill!(Mantle.hostview(dev, r.buffer.store), typemin(Int32))
        end
        @test_throws ArgumentError Mantle.Plan(g; budget=Inf)
        Mantle.record!(pl)
        Mantle.waitidle(dev)
        k[] = Int32(10)
        n = Mantle.submissions(dev)
        Mantle.run!(pl)
        @test Mantle.submissions(dev) == n + 1
        got = fetch(f)
        @test got isa Matrix{Int32}
        @test got == reshape(Int32.(11:523), 19, 27)
        @test fetch(fv) == reshape(Int32.(44:106), 7, 9)
        @test wait(f) === f
        @test isready(f)
        @test fetch(f) === got
        @test Mantle.submissions(dev) == n + 1
        @test_throws ArgumentError Mantle.Readback(g, Mantle.viewof(a, (1,); offset=0))
        Mantle.free!(pl)
        @test fetch(f) === got
        @test_throws ArgumentError Mantle.Readback(a)
        Mantle.free!(k)
    end

    @testset "replay preserves uncollected old results and baked transfers" begin
        g, a, k = rb_plan(dev, Int32, (513,))
        f1 = Mantle.Readback(a)
        pl = Mantle.Plan(g; budget=Inf)
        Mantle.record!(pl)
        rec = pl.recording
        n = Mantle.submissions(dev)
        Mantle.run!(pl)
        # Request two futures for the next run, without changing its recording.
        f2 = Mantle.Readback(a)
        f2b = Mantle.Readback(a)
        @test !isready(f2)
        k[] = Int32(1000)
        Mantle.run!(pl)
        @test pl.recording === rec
        @test length(g.readbacks) == 1
        @test Mantle.submissions(dev) == n + 2
        # Teardown before either result is collected must not retire staging
        # which the second result still needs.
        Mantle.free!(pl)
        GC.gc(false)
        @test fetch(f2) == Int32.(1001:1513)
        @test fetch(f2b) === fetch(f2)
        @test fetch(f1) == Int32.(1:513)
        @test Mantle.submissions(dev) == n + 2
        Mantle.free!(k)
    end

    @testset "declaration captures before a subsequent write" begin
        g, a, k = rb_plan(dev, Int32, (513,))
        f1 = Mantle.Readback(a)
        k2 = Mantle.GPURef(dev, Int32(100))
        Mantle.dispatch!(g, rb_write!, (a, k2), length(a))
        f2 = Mantle.Readback(a)
        pl = Mantle.Plan(g; budget=Inf)
        Mantle.record!(pl)
        Mantle.run!(pl)
        @test fetch(f1) == Int32.(1:513)
        @test fetch(f2) == Int32.(101:613)
        @test_throws ArgumentError Mantle.Readback(a)
        Mantle.free!(pl)
        Mantle.free!(k); Mantle.free!(k2)
    end

    @testset "waiting task wakes on execution or cancellation" begin
        g, a, k = rb_plan(dev, Int32, (257,))
        f = Mantle.Readback(a)
        pl = Mantle.Plan(g; budget=Inf)
        Mantle.record!(pl)
        t = @async fetch(f)
        yield()
        @test !istaskdone(t)
        n = Mantle.submissions(dev)
        Mantle.run!(pl)
        @test fetch(t) == Int32.(1:257)
        @test Mantle.submissions(dev) == n + 1
        cancelled = Mantle.Readback(a)
        t2 = @async try fetch(cancelled) catch err; err end
        yield()
        Mantle.free!(pl)
        @test fetch(t2) isa ArgumentError
        @test isready(cancelled)
        @test_throws ArgumentError fetch(cancelled)
        Mantle.free!(k)
    end

    @testset "failed run rejects its future" begin
        g, a, k = rb_plan(dev, Int32, (257,))
        f = Mantle.Readback(a)
        pl = Mantle.Plan(g; budget=Inf)
        # A recording backend refuses an unrecorded run. The future must wake.
        if Mantle.recordsplans(dev) && Mantle.recordable(dev, pl)
            @test_throws ArgumentError Mantle.run!(pl)
            @test isready(f)
            @test_throws ArgumentError fetch(f)
        end
        Mantle.free!(pl); Mantle.free!(k)
    end

    @testset "explicit graph for persistent arrays and validation" begin
        g = Mantle.Graph(dev)
        a = Mantle.Buffer(dev, Int32.(1:257); capacity=513)
        f = Mantle.Readback(g, a)
        pl = Mantle.Plan(g; budget=Inf)
        Mantle.record!(pl)
        n = Mantle.submissions(dev)
        Mantle.run!(pl)
        @test fetch(f) == Int32.(1:257)
        @test Mantle.submissions(dev) == n + 1
        # Ordinary Array conversion continues to return an actual array.
        @test Array(a) isa Array{Int32,1}
        Mantle.free!(pl); Mantle.free!(a)
        g2 = Mantle.Graph(dev)
        b = Mantle.Transient.Buffer(g2, Float32, 1)
        @test_throws ArgumentError Mantle.Readback(g, b)
    end

    @testset "short elements and non-word-sized output" begin
        for T in (Int16, UInt8)
            g, a, k = rb_plan(dev, T, (51,))
            f = Mantle.Readback(a)
            fv = Mantle.Readback(Mantle.viewof(a, (33,); offset=1))
            pl = Mantle.Plan(g; budget=Inf)
            Mantle.record!(pl); Mantle.run!(pl)
            @test fetch(f) == T.(1:51)
            @test fetch(fv) == T.(2:34)
            Mantle.free!(pl); Mantle.free!(k)
        end
    end
end

@testset "readback after loop and changed source storage" begin
    dev = Mantle.Device(TESTBACKEND)
    g = Mantle.Graph(dev)
    a = Mantle.Buffer(dev, zeros(Int32, 257))
    Mantle.repeat!(g, 25) do i
        Mantle.dispatch!(g, rb_write!, (a, i), 257)
    end
    f = Mantle.Readback(g, a)
    pl = Mantle.Plan(g; budget=Inf)
    Mantle.record!(pl)
    n = Mantle.submissions(dev)
    Mantle.run!(pl)
    @test fetch(f) == Int32.(26:282)
    @test Mantle.submissions(dev) == n + 1
    # Move an external source while preserving its declared logical size. A
    # native transfer embeds a buffer handle, so this must refresh that binding.
    resize!(a, 1024)
    a[:] = Int32.(1:257)  # a store also restores its logical extent
    resize!(a, 257)
    f2 = Mantle.Readback(g, a)
    n = Mantle.submissions(dev)
    Mantle.run!(pl)
    @test fetch(f2) == Int32.(26:282)
    @test Mantle.submissions(dev) == n + 1
    Mantle.free!(pl); Mantle.free!(a)
end

@testset "readback rejects changed extent and in-loop snapshots" begin
    dev = Mantle.Device(TESTBACKEND)
    g = Mantle.Graph(dev)
    a = Mantle.Buffer(dev, Int32.(1:257))
    f = Mantle.Readback(g, a)
    pl = Mantle.Plan(g; budget=Inf)
    Mantle.record!(pl)
    resize!(a, 10)
    n = Mantle.submissions(dev)
    @test_throws DimensionMismatch Mantle.run!(pl)
    @test_throws DimensionMismatch fetch(f)
    @test Mantle.submissions(dev) == n
    Mantle.free!(pl); Mantle.free!(a)

    g2, b, k = rb_plan(dev, Int32, (257,))
    holder = Ref{Any}()
    Mantle.repeat!(g2, 2) do i
        Mantle.dispatch!(g2, rb_write!, (b, i), length(b))
        holder[] = Mantle.Readback(b)
    end
    @test_throws ArgumentError Mantle.Plan(g2; budget=Inf)
    @test_throws ArgumentError fetch(holder[])
    Mantle.free!(k)
end

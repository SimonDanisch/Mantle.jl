using Test, Mantle, KernelAbstractions

@kernel function partition_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function recorded_read_scalar!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[1]
end

@kernel function partition_double!(a)
    i = @index(Global)
    @inbounds a[i] *= 2f0
end

@testset "recorded dispatches synchronize mapped inputs" begin
    back = Mantle.LavaBackend()
    bq = Mantle.batchqueue(Mantle.Device(back))
    src = Mantle.LavaArray{Int32,1}(undef, (1,); bq, unified=true)
    dst = KernelAbstractions.allocate(back, Int32, 65536)
    g = Mantle.Graph(Mantle.Device(back))
    Mantle.dispatch!(g, recorded_read_scalar!,
                         (dst,
                          src), length(dst); group = 64, name = "mapped input")
    pl = Mantle.record!(Mantle.Plan(g))
    try
        for value in Int32[11, 29, 47]
            copyto!(src, [value])
            Mantle.run!(pl)
            # Assert the ordering contract even if this GPU completes before
            # the host reaches the overwrite. Without dispatch registration,
            # the input's stamp still names its previous upload.
            @test Mantle.stampof(src).token == pl.recording.token
            @test Mantle.stampof(dst).token == pl.recording.token
            # No explicit wait: the mapped host write must wait for the read.
            copyto!(src, Int32[-1])
            @test all(==(value), Array(dst))
        end
        other = Mantle.LavaBackend(Mantle.allocate_batch_queue!(Mantle.vk_context()))
        writergraph = Mantle.Graph(Mantle.Device(other))
        Mantle.dispatch!(writergraph, recorded_read_scalar!,
                             (src,
                              dst), 1; group = 1, name = "other queue writes input")
        writer = Mantle.record!(Mantle.Plan(writergraph))
        try
            for _ in 1:3
                Mantle.run!(writer)
                @test Mantle.stampof(src).channel === Mantle.batchqueue(Mantle.Device(other))
                Mantle.run!(pl)
                @test Mantle.stampof(src).channel === bq
                @test all(==(Int32(47)), Array(dst))
            end
        finally
            Mantle.free!(writer)
        end
    finally
        Mantle.free!(pl)
    end
end

@testset "a plan measures itself and is cut by its budget" begin
    back = Mantle.LavaBackend()
    a = KernelAbstractions.allocate(back, Float32, 128)
    function tendoublings(budget)
        g = Mantle.Graph(Mantle.Device(back))
        # Ten dependent passes: a cut anywhere must keep the order between them.
        for k in 1:10
            Mantle.dispatch!(g, partition_double!, (a,), length(a); name = "double$k")
        end
        Mantle.Plan(g; budget)
    end
    pl = tendoublings(0.1)
    Mantle.record!(pl)
    # Not measured yet: nothing bounds a pass, so each is a submission, timed.
    @test length(pl.recording.parts) == 10
    @test pl.stamped
    fill!(a, 1f0)
    @test all(==(1f0), Array(a))            # recording itself must not execute
    for _ in 1:3
        fill!(a, 1f0)
        Mantle.run!(pl)
        @test all(==(1024f0), Array(a))
    end
    # Measured on the first run and re-recorded on the second: ten tiny passes
    # fit one submission, and the timestamps went with the measurement.
    @test !any(isnan, pl.passcost)
    @test !(pl.recording isa Mantle.RecordingParts)
    @test !pl.stamped
    Mantle.invalidate!(pl)
    fill!(a, 1f0)
    Mantle.run!(pl)
    @test !(pl.recording isa Mantle.RecordingParts)   # re-recorded from the measurement
    @test all(==(1024f0), Array(a))
    Mantle.free!(pl)
    @test pl.recording === nothing

    # A budget no pass fits keeps every pass apart after it is measured.
    pl = tendoublings(0.0)
    Mantle.record!(pl)
    for _ in 1:3
        fill!(a, 1f0)
        Mantle.run!(pl)
        @test all(==(1024f0), Array(a))
    end
    @test !any(isnan, pl.passcost)
    @test length(pl.recording.parts) == 10
    Mantle.free!(pl)

    # `budget = Inf` never cuts, measured or not.
    pl = tendoublings(Inf)
    Mantle.record!(pl)
    @test !(pl.recording isa Mantle.RecordingParts)
    fill!(a, 1f0)
    Mantle.run!(pl)
    @test all(==(1024f0), Array(a))
    Mantle.free!(pl)
end

@testset "partitioned recording follows moved inputs" begin
    dev = Mantle.Device(Mantle.LavaBackend())
    src = Mantle.Buffer(dev, ones(Float32, 128))
    out = Mantle.Buffer(dev, zeros(Float32, 128))
    g = Mantle.Graph(dev)
    for i in 1:3
        Mantle.dispatch!(g, partition_copy!, (out, src), 128; name = "copy$i")
    end
    # `budget = 0.0`: every pass its own piece for good.
    pl = Mantle.record!(Mantle.Plan(g; budget = 0.0))
    for _ in 1:2                       # measure, then the re-recording it asks for
        Mantle.run!(pl)
        @test Array(Mantle.storage(out)) == ones(Float32, 128)
    end
    rec = pl.recording
    @test rec isa Mantle.RecordingParts
    Mantle.resize!(src, 512)
    copyto!(Mantle.storage(src), fill(3f0, 512))
    Mantle.run!(pl)
    @test pl.recording === rec
    @test Mantle.stampof(Mantle.storage(src)).token == last(rec.parts).token
    @test Array(Mantle.storage(out)) == fill(3f0, 128)
    Mantle.free!(pl)
end

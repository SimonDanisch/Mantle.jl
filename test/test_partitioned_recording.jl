# What a recorded plan does with its inputs and with its budget.
#
#   * A host write into mapped (unified) memory that a recorded run reads waits
#     for that read: the run is registered on the input, so the overwrite after
#     `run!` cannot reach the bytes before the device has read them.
#   * A plan on a device that times its passes (`timestamps`) measures itself on
#     its first run and is cut into submissions by its budget; one that cannot
#     time them is never cut. Either way the result is the same.
#   * A partitioned recording follows an input that moves under it.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))

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
    dev = Mantle.Device(TESTBACKEND)
    src = Mantle.devicearray(TESTBACKEND, Int32, 1; unified = true)
    dst = KernelAbstractions.allocate(TESTBACKEND, Int32, 65536)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, recorded_read_scalar!,
                         (dst,
                          src), length(dst); group = 64, name = "mapped input")
    pl = Mantle.record!(Mantle.Plan(g))
    try
        for value in Int32[11, 29, 47]
            copyto!(src, [value])
            Mantle.run!(pl)
            # No explicit wait: the mapped host write must wait for the read.
            # (The Vulkan version also asserted the input's submission stamp, which
            # is what makes this deterministic on a GPU that finishes before the
            # host reaches the overwrite; a stamp is that backend's bookkeeping.)
            copyto!(src, Int32[-1])
            @test all(==(value), Array(dst))
        end
        # Another plan WRITES the input between runs: the next run of `pl` reads
        # what that plan wrote, not a stale copy.
        writergraph = Mantle.Graph(dev)
        Mantle.dispatch!(writergraph, recorded_read_scalar!,
                             (src,
                              dst), 1; group = 1, name = "another plan writes input")
        writer = Mantle.record!(Mantle.Plan(writergraph))
        try
            for _ in 1:3
                Mantle.run!(writer)
                Mantle.run!(pl)
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
    dev = Mantle.Device(TESTBACKEND)
    timed = Mantle.timestamps(dev)
    a = KernelAbstractions.allocate(TESTBACKEND, Float32, 128)
    function tendoublings(budget)
        g = Mantle.Graph(dev)
        # Ten dependent passes: a cut anywhere must keep the order between them.
        for k in 1:10
            Mantle.dispatch!(g, partition_double!, (a,), length(a); name = "double$k")
        end
        Mantle.Plan(g; budget)
    end
    pl = tendoublings(0.1)
    Mantle.record!(pl)
    if timed
        # Not measured yet: nothing bounds a pass, so each is a submission, timed.
        @test length(pl.recording.parts) == 10
        @test pl.stamped
    else
        # Nothing can be measured, so nothing is cut.
        @test !(pl.recording isa Mantle.RecordingParts)
    end
    fill!(a, 1f0)
    @test all(==(1f0), Array(a))            # recording itself must not execute
    for _ in 1:3
        fill!(a, 1f0)
        Mantle.run!(pl)
        @test all(==(1024f0), Array(a))
    end
    # Measured on the first run and re-recorded on the second: ten tiny passes
    # fit one submission, and the timestamps went with the measurement.
    timed && @test !any(isnan, pl.passcost)
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
    if timed
        @test !any(isnan, pl.passcost)
        @test length(pl.recording.parts) == 10
    else
        @test !(pl.recording isa Mantle.RecordingParts)
    end
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
    dev = Mantle.Device(TESTBACKEND)
    src = Mantle.Buffer(dev, ones(Float32, 128))
    out = Mantle.Buffer(dev, zeros(Float32, 128))
    g = Mantle.Graph(dev)
    for i in 1:3
        Mantle.dispatch!(g, partition_copy!, (out, src), 128; name = "copy$i")
    end
    # `budget = 0.0`: every pass its own piece for good, on a device that can time
    # its passes; one piece where it cannot.
    pl = Mantle.record!(Mantle.Plan(g; budget = 0.0))
    for _ in 1:2                       # measure, then the re-recording it asks for
        Mantle.run!(pl)
        @test Array(Mantle.storage(out)) == ones(Float32, 128)
    end
    rec = pl.recording
    Mantle.timestamps(dev) && @test rec isa Mantle.RecordingParts
    Mantle.resize!(src, 512)
    copyto!(Mantle.storage(src), fill(3f0, 512))
    Mantle.run!(pl)
    @test pl.recording === rec         # patched, not re-recorded
    @test Array(Mantle.storage(out)) == fill(3f0, 128)
    Mantle.free!(pl)
end

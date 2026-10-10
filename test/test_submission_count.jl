"""
How many submissions a call hands to the driver, counted with `submissions(dev)`.

Two ways to get this wrong, both of them a queue reaching the driver more than
once per call: force-closing whatever batch is open, submitting it, then
submitting a run's recording behind a hand-rolled wait; or appending the run to an
open batch that goes when something else asks, so the run submits nothing until a
later, unrelated call. A run of a recorded plan is ONE submission — the recording,
and when a store is waiting, the copy that lands it in front of it, in the same
submission — and waiting for it is none. Two passes, so the count is not
accidentally right because there is only one thing to submit.

Where the backends differ, the unit is chosen so both answer the same. A Vulkan
launch submits when it is called; a Metal launch joins an open command buffer
that the next wait commits — so "a launch and the wait for it" is one submission
on both. An upload or a download is one on Vulkan, a staged copy, and none on
Metal, whose buffers the host addresses: at most one.

The Vulkan original also pinned what that backend's queue IS: no field that
could hold an open command buffer, one record of what is in flight, a global
barrier written at the head of every closed buffer, a token per submission.
Those are the Vulkan queue's construction; what a user sees of them is the count
here, and the results of the runs, which `test_recorded_run_semantics.jl` and
`test_uploads_back_to_back.jl` assert as well.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function sc_bump!(a)
    i = @index(Global)
    @inbounds a[i] += 1f0
end

@kernel function sc_addref!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

function sc_twopassplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    for name in ("a", "b")
        Mantle.dispatch!(g, sc_addref!, (out, kref), n; name = name)
    end
    Mantle.Plan(g; budget = Inf)
end

"""The submissions `f` makes on `dev`."""
submits(f, dev) = (s0 = Mantle.submissions(dev); f(); Mantle.submissions(dev) - s0)

@testset "submissions per call" begin
    dev = Mantle.Device(TESTBACKEND)
    be = Mantle.backend(dev)
    n = 256

    @testset "a launch and the wait for it are one submission" begin
        a = KA.allocate(be, Float32, n)
        fill!(a, 0f0)
        k = sc_bump!(be, 64)
        k(a; ndrange = n)
        KA.synchronize(be)                     # compiled, and nothing left open
        @test submits(dev) do
            k(a; ndrange = n)
            KA.synchronize(be)
        end == 1
        @test all(==(2f0), Array(a))
        # Waiting with nothing left to wait for submits nothing.
        @test submits(() -> KA.synchronize(be), dev) == 0
    end

    @testset "an upload and a download are at most one each" begin
        b = Mantle.Buffer(dev, zeros(Float32, n))
        Mantle.waitidle(dev)
        @test submits(() -> Mantle.update!(b, fill(3f0, n)), dev) <= 1
        local got
        @test submits(() -> (got = Array(Mantle.storage(b))), dev) <= 1
        @test got == fill(3f0, n)
    end

    @testset "a run is one submission, and waiting for it none" begin
        out = Mantle.Buffer(dev, zeros(Int32, n))
        kref = Mantle.GPURef(dev, Int32(1))
        # `budget = Inf`: recorded whole, never measured. A plan on the device's
        # budget submits once per pass on its first run and is re-recorded after
        # it (see `partitionranges`), and this counts the steady state.
        pl = Base.invokelatest(sc_twopassplan, dev, out, kref, n)
        Mantle.waitidle(dev)
        # Recording executes nothing. It may initialise what the recording keeps on
        # the device — Metal zero-fills its range and grid buffers with a launch,
        # which the next wait commits — and that is one submission at most.
        if Mantle.recordable(dev, pl)
            @test submits(() -> (Mantle.record!(pl); Mantle.waitidle(dev)), dev) <= 1
        end

        # A store pending each time: its copy goes in front of the recording, in
        # the same submission.
        runs = 10
        @test submits(dev) do
            for i in 1:runs
                kref[] = Int32(i)
                Mantle.run!(pl)
            end
        end == runs
        Mantle.waitfor!(pl)
        @test Array(Mantle.storage(out)) == fill(Int32(2 * sum(1:runs)), n)

        # Nothing pending: the recording alone.
        @test submits(() -> Mantle.run!(pl), dev) == 1
        # …and nothing is submitted to wait for it, or for the device.
        @test submits(() -> (Mantle.waitfor!(pl); Mantle.waitidle(dev)), dev) == 0
        @test Array(Mantle.storage(out)) == fill(Int32(2 * sum(1:runs) + 2 * runs), n)
        Mantle.free!(pl)
    end

    @testset "a token beyond what was submitted is refused, not waited for" begin
        # A token covers work a submission has made. One past the newest names
        # work nothing submitted, which a wait would wait for for ever.
        @test_throws ArgumentError Mantle.waitfor!(dev, Mantle.fence(dev) + 1)
    end
end

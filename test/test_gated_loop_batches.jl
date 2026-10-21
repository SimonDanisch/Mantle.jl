# A discarded tail is not a timing bound for the loop's live iterations.
# The whirlpool's 128-iteration bounce loop lost the Laptop device when a
# nearly empty final iteration made the scheduler batch the whole loop.
using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))

@kernel function batch_loop_drain!(out, live)
    i = @index(Global)
    @inbounds out[i] += Int32(1)
    if i == 1
        @inbounds live[1] -= Int32(1)
    end
end

@testset "discarded loop tails cannot batch all live iterations" begin
    dev = Mantle.Device(TESTBACKEND)
    if Mantle.supportspredicate(dev) && Mantle.recordsplans(dev)
        out = Mantle.Buffer(dev, zeros(Int32, 256))
        live = Mantle.Buffer(dev, Int32[5])
        for budget in (Mantle.submissionbudget(dev), Inf)
            g = Mantle.Graph(dev)
            Mantle.repeat!(g, 128; while_nonzero=live) do _
                Mantle.dispatch!(g, batch_loop_drain!, (out, live), 256; name="drain")
            end
            pl = Mantle.record!(Mantle.Plan(g; budget))
            try
                # Emulate the inexpensive discarded tail deterministically,
                # without depending on GPU clocks or inducing a watchdog reset.
                fill!(pl.passcost, 1e-7)
                Mantle.invalidate!(pl)
                Mantle.record!(pl)
                rec = pl.recording
                @test rec isa Mantle.RecordingParts
                if isfinite(budget)
                    @test length(rec.batches) >= 128
                    body = only(r for r in rec.parts if count(x -> x === r, reduce(vcat, rec.batches)) == 128)
                    @test all(b -> count(x -> x === body, b) <= 1, rec.batches)
                else
                    @test length(rec.batches) == 1
                end
                for start in (5, 11)
                    fill!(Mantle.storage(out), Int32(0))
                    copyto!(Mantle.storage(live), Int32[start])
                    Mantle.run!(pl)
                    Mantle.waitfor!(pl)
                    @test all(==(Int32(start)), Array(Mantle.storage(out)))
                    @test only(Array(Mantle.storage(live))) == 0
                end
            finally
                Mantle.free!(pl)
            end
        end
        Mantle.free!(out)
        Mantle.free!(live)
    end
end

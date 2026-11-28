using Test, Mantle

function check_flush_report(ctx)
    @testset "flush timeout report describes outstanding submissions" begin
        bq = ctx.default_bq
        a = Mantle.LavaArray{Float32}(undef, (1024,); bq)
        b = similar(a)
        copyto!(b, a)
        target = Mantle.newest(bq)
        @test !isempty(Mantle.outstanding(bq))
        report = Mantle.flush_stall_report(bq, target)
        @test occursin("(copy): signals $target", report)
        # A target past every signalled submission is a wait nothing will signal.
        Mantle.flush!(bq, ctx)
        @test occursin("nothing will signal", Mantle.flush_stall_report(bq, target + UInt64(1)))
        Mantle.unsafe_free!(a); Mantle.unsafe_free!(b)
    end
end

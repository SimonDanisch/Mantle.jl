# A wait on a channel refuses a token that no submission on it will signal, as a wait
# on the device does. The channel's own `waitfor!` skipped core's check and blocked
# in the driver for ever.
using Test, Mantle

function check_channel_waits(dev)
    @testset "a channel wait refuses a token nothing submitted" begin
        bq = Mantle.batchqueue(dev)
        past = Mantle.fence(dev) + 1000
        @test_throws ArgumentError Mantle.waitfor!(bq, past)
        @test_throws ArgumentError Mantle.waitfor!(dev, past)
        # A token that was submitted is waited for.
        @test Mantle.waitfor!(bq, Mantle.fence(dev)) === nothing
    end
end

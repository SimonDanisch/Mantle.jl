# Filling an exported image goes through the device's channel like every other
# submission. It recorded a command pool of its own and submitted it straight to the
# queue the channel owns, then waited for the whole queue to go idle: no timeline
# value, no holds, and a queue touched from outside the one place that serialises its
# submissions. Observed as the submission count, which counts the channels'.
using Test, Mantle

function check_external_image(dev)
    if !dev.external_memory_available
        @test_skip dev.external_memory_available
        return nothing
    end
    @testset "an exported image is filled through the channel" begin
        img = Mantle.VulkanExternalImage(dev, 16, 8)
        a = Mantle.devicearray(dev, fill(0x7f, 4 * 16 * 8))
        Mantle.waitidle(dev)
        before = Mantle.submissions(dev)
        copyto!(img, a)
        @test Mantle.submissions(dev) == before + 1
        @test img.layout_initialized
        # Again, now from the layout the first fill left.
        copyto!(img, a)
        @test Mantle.submissions(dev) == before + 2
        @test_throws ArgumentError copyto!(img, Mantle.devicearray(dev, fill(0x00, 16)))
        Mantle.waitidle(dev)
        @test isempty(Mantle.validationmessages!(dev))
    end
end

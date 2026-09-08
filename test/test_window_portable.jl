# The window questions that are the same on every backend.
#
# Split out of `test_window.jl`, which is 2,000 lines and reaches for
# `MantleVulkanExt` in twenty-six places — sixteen of them raw `VK.` enums for
# image layouts, load ops and aspects, which are genuinely that backend's and
# belong under `test/vulkan/`. Splitting the rest is the remainder of phase 2.8;
# until then the big file stays Vulkan-gated and this one runs per backend,
# because the property below is exactly the one that had no test on Metal.
#
# `Window(backend, w, h)` is in the BACKEND_VOCABULARY and was answered by Lava
# alone. The Metal side said it could not, for a reason that turned out to be
# false — ColorTypes IS a Mantle dependency.

using Test
import Mantle
const M = Mantle
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ?
    Main.MANTLE_TEST_BACKEND : M.defaultbackend()

@testset "the portable Window is answered by this backend" begin
    # `Window(backend, w, h)` is in the BACKEND_VOCABULARY and was answered by
    # Lava alone; the Metal side said it could not, for a reason that turned out
    # to be false (ColorTypes IS a Mantle dependency). This is the test that
    # says every backend answers it, rather than one backend's tests passing.
    w = Mantle.Window(TESTBACKEND, 320, 200; title = "portable")
    try
        @test w isa Mantle.Window
        @test size(w) == (320, 200)
        @test eltype(w) == BGRA{N0f8}
        @test isopen(w)
        # The extent it reports IS the extent of the texture it hands out. A
        # window that lies about that breaks every pass that strides a buffer
        # by hand, and on a Retina panel the naive answer is twice too large.
        @test Mantle.target_extent(w) == (320, 200)
        Mantle.beginframe!(w)
        Mantle.acquire_next_image!(w)
        t = Mantle.target_view(w)
        @test (Int(t.width), Int(t.height)) == (320, 200)
        Mantle.present_frame!(Mantle.Device(TESTBACKEND), w)
    finally
        close(w)
    end
end

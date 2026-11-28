# A readback copies out of the colour attachment, and the next draw that loads the
# attachment must find it in the layout a render pass expects. A tolerant driver
# draws correctly either way, so the validation layer is the assertion, and the
# layer has to be loaded when the process starts.

using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "a readback leaves the attachment ready for a draw that loads it" begin
    if !Mantle.supports_graphics(Mantle.Device(TESTBACKEND))
        @test_skip Mantle.supports_graphics(Mantle.Device(TESTBACKEND))
        return
    end
    status, out = infreshprocess(raw"""
        using GeometryBasics, ColorTypes
        function vert()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, 0.5f0, 1.0f0),)
        end
        frag(_inputs) = Vec4f(0.25f0, 0f0, 0f0, 1f0)
        dev = Mantle.reset_device!(Mantle.Device(BACKEND); debug = DebugConfig(validation = true))
        Mantle.validating(dev) || exit(3)
        pip = GraphicsPipeline(; vertex = VertexShader(vert), fragment = FragmentShader(frag),
                               blend = Additive(), cull = NoCull(), depth = DepthOff())
        fb = Mantle.Framebuffer(dev, 8, 8; depth = false, color_format = RGBA{Float32})
        target = OffscreenTarget(fb)
        Mantle.validationmessages!(dev)
        @testset "draw, read back, draw on top, read back" begin
            Mantle.draw!(dev, pip, target, 3; clear_color = (0f0, 0f0, 0f0, 1f0))
            Mantle.flush!(dev)
            @test Mantle.readback_framebuffer(fb)[4, 4].r ≈ 0.25f0 atol = 0.01
            Mantle.draw!(dev, pip, target, 3; clear_color = nothing)
            Mantle.flush!(dev)
            @test Mantle.readback_framebuffer(fb)[4, 4].r ≈ 0.5f0 atol = 0.01
            msgs = Mantle.validationmessages!(dev)
            foreach(println, msgs)
            @test isempty(msgs)
        end
        """; debug = DebugConfig(validation = true))
    if status == 3
        @info "no validation layer for this device here; the layout check needs one"
        @test_skip status == 0
    else
        reportfresh(status, out)
        @test status == 0
    end
end

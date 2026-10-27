using Test, Mantle
using ColorTypes: RGBA

held_draw_vertex(data) = (position=data.points[Int(Mantle.vertex_index())],)
held_draw_fragment(inputs) = (0f0,1f0,0f0,1f0)

function check_draw_argument_lifetime(ctx)
    @testset "manual draw holds nested arrays through submission" begin
        dev = ctx
        points = Mantle.LavaArray([(-1f0,-1f0,0f0,1f0), (3f0,-1f0,0f0,1f0),
                                  (-1f0,3f0,0f0,1f0)]; bq=dev.default_bq)
        args = ((points=points,),)
        description = Mantle.GraphicsPipeline(;
            vertex=Mantle.VertexShader(held_draw_vertex),
            fragment=Mantle.FragmentShader(held_draw_fragment),
            depth=Mantle.DepthOff(), cull=Mantle.NoCull())
        draw = Mantle.compile_draw(dev,description,(RGBA{Float32},),nothing,args,())
        fb = Mantle.Framebuffer(dev,8,8;depth=false,color_format=RGBA{Float32})
        Mantle.flush!(dev.default_bq); Mantle.waitidle(dev); Mantle.drain!(dev.default_bq)
        buffer = points.buf[]
        stamp = Mantle.stampof(points)
        @test (@atomic stamp.holders) == 0
        Mantle.pass!(dev,Mantle.OffscreenTarget(fb);clear=(1f0,0f0,0f0,1f0)) do p
            Mantle.draw!(p,draw,args,3)
            @test (@atomic stamp.holders) > 0
            @test any(x -> x === points,last(dev.default_bq.holds))
            @test any(x -> x === buffer,p.handle.o.sync)
            Mantle.unsafe_free!(points)
            Mantle.drain!(dev.default_bq)
            @test (@atomic buffer.state) != Mantle.BUF_STATE_DEAD
        end
        pixels = Mantle.readback_framebuffer(fb)
        @test all(p -> p.g > 0.99f0 && p.r < 0.01f0,pixels)
        Mantle.waitidle(dev); Mantle.drain!(dev.default_bq)
        @test (@atomic stamp.holders) == 0
        @test (@atomic buffer.state) == Mantle.BUF_STATE_DEAD
        @test isempty(Mantle.validationmessages!(dev))
    end
end

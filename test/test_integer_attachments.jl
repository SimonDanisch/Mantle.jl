module IntegerAttachmentTests
using Test, Mantle
using Mantle: RGBA
const IDs = Mantle.Vec{2,UInt32}
vertex() = Mantle.blit_vertex()
fragment(_) = (Mantle.Vec4f(0.8,0.2,0.1,0.5), IDs(0xff123456,0xef654321))
scalar_fragment(_) = UInt32(0xf1234567)

@testset "exact integer MRT and cropped image copies" begin
    dev=Mantle.todevice(Mantle.defaultbackend())
    g=Mantle.Graph(dev)
    color=Mantle.Transient.Image(g,RGBA{Float32},(8,6))
    ids=Mantle.Transient.Image(g,IDs,(8,6))
    pipe=Mantle.GraphicsPipeline(;vertex=Mantle.VertexShader(vertex),
        fragment=Mantle.FragmentShader(fragment),blend=Mantle.AlphaBlend(),
        cull=Mantle.NoCull(),depth=Mantle.DepthOff())
    Mantle.render!(g,"integer MRT",color=>Mantle.Clear((0f0,0f0,0f0,1f0)),
        ids=>Mantle.Clear((UInt32(0),UInt32(0),UInt32(0),UInt32(0)))) do pass
        Mantle.draw!(pass,pipe,(),3)
    end
    roi=Mantle.Buffer(dev,IDs,6)
    Mantle.copy!(g,"six pixels",roi,ids;region=(3,2,2,3))
    @test_throws ArgumentError Mantle.copy!(g,"out of bounds",roi,ids;region=(7,0,2,3))
    plan=Mantle.record!(Mantle.Plan(g))
    try
        Mantle.run!(plan)
        @test Array(Mantle.storage(roi)) == fill(IDs(0xff123456,0xef654321),6)
        @test length(Mantle.storage(roi))*sizeof(IDs)==48
    finally
        Mantle.free!(plan);Mantle.free!(roi)
    end
    for draw in (false,true)
        g=Mantle.Graph(dev)
        img=Mantle.Transient.Image(g,UInt32,(4,4))
        pipe=Mantle.GraphicsPipeline(;vertex=Mantle.VertexShader(vertex),
            fragment=Mantle.FragmentShader(scalar_fragment),cull=Mantle.NoCull(),depth=Mantle.DepthOff())
        Mantle.render!(g,"integer clear",img=>Mantle.Clear((UInt32(0xf2345678),UInt32(0),UInt32(0),UInt32(0)))) do pass
            draw && Mantle.draw!(pass,pipe,(),3)
        end
        out=Mantle.Buffer(dev,UInt32,16)
        Mantle.copy!(g,"read IDs",out,img)
        plan=Mantle.record!(Mantle.Plan(g))
        try
            Mantle.run!(plan)
            @test Array(Mantle.storage(out))==fill(draw ? UInt32(0xf1234567) : UInt32(0xf2345678),16)
        finally
            Mantle.free!(plan);Mantle.free!(out)
        end
    end
end
end

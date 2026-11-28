module RecordedProfileRegression
using Mantle, Metal, Test, KernelAbstractions
@kernel function recorded_profile_increment!(out)
 i=@index(Global)
 @inbounds out[i]+=1f0
end
@testset "recorded GPU work appears in the integrated profiler" begin
 dev=Mantle.Device(Mantle.MetalAPI())
 out=Mantle.Buffer(dev,zeros(Float32,256))
 g=Mantle.Graph(dev)
 Mantle.dispatch!(g,recorded_profile_increment!,(out,),256;name="profile increment")
 pl=Mantle.record!(Mantle.Plan(g;budget=Inf))
 try
  Mantle.run!(pl);Mantle.waitfor!(pl)
  hook=Metal.MTL.profile_hook[]
  metadata=Metal.MTL.profile_metadata[]
  p=Metal.@profile for _ in 1:2
   Mantle.run!(pl);Mantle.waitfor!(pl)
  end
  @test Array(Mantle.storage(out))==fill(3f0,256)
  slots=findall(==("Mantle recording"),p.device.name)
  @test length(slots)==2
  @test all(i->p.device.stop[i]>p.device.start[i],slots)
  @test Metal.MTL.profile_hook[]===hook
  @test Metal.MTL.profile_metadata[]===metadata
  Mantle.run!(pl);Mantle.waitfor!(pl)
  @test Array(Mantle.storage(out))==fill(4f0,256)
 finally
  Mantle.free!(pl)
 end
end
end

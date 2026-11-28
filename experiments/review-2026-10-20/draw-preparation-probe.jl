# include this file, then before=DrawPreparationProbe.baseline().
# Compare via Base.invokelatest(DrawPreparationProbe.compare, ctx, before).
# Both arms use current shader publication; only draw preparation differs.
module DrawPreparationProbe
using Mantle, ColorTypes, Test, Statistics
function vertex(level::Float32)
 i=Int32(Mantle.vertex_index())-Int32(1)
 return (position=(Float32((i&1)*4-1),Float32(((i>>1)&1)*4-1),0f0,1f0),)
end
fragment(inputs,level::Float32)=(0f0,level,0f0,1f0)
function fixture(ctx)
 dev=Mantle.lavadevice(ctx)
 g=Mantle.Graph(dev)
 img=Mantle.Transient.Image(g,RGBA{Float32},(8,8))
 desc=Mantle.GraphicsPipeline(;vertex=Mantle.VertexShader(vertex),fragment=Mantle.FragmentShader(fragment),depth=Mantle.DepthOff(),cull=Mantle.NoCull())
 Mantle.render!(g,"draw",img=>Mantle.Clear((0f0,0f0,0f0,1f0))) do p
  Mantle.draw!(p,desc,(0.7f0,),3;frag_args=(0.7f0,))
 end
 p=only(filter(p->p.kind===:render,g.passes))
 return Mantle.Compile(g),p,only(p.draws)
end
Base.@noinline function exercise(f,c,p,d,n)
 for _ in 1:n
  f(c,p,d,0)
 end
 return nothing
end
function compare(ctx,before;n=100)
 c,p,d=fixture(ctx)
 after=Mantle.compiledraw
 a=before(c,p,d,0);b=after(c,p,d,0)
 @test a.compiled===b.compiled
 @test a.shader===b.shader
 @test all(getfield(a,f)===getfield(b,f) for f in fieldnames(typeof(a)))
 exercise(before,c,p,d,n);exercise(after,c,p,d,n)
 oldbytes=@allocated exercise(before,c,p,d,n)
 newbytes=@allocated exercise(after,c,p,d,n)
 @test newbytes<=oldbytes
 oldtimes,newtimes=Float64[],Float64[]
 for i in 1:15
  fs=isodd(i) ? (before,after) : (after,before)
  for f in fs
   time=@elapsed exercise(f,c,p,d,n)
   push!(f===before ? oldtimes : newtimes,time)
  end
 end
 @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
 row=(device=ctx.device_name,old_bytes=oldbytes÷n,new_bytes=newbytes÷n,
      old_us=median(oldtimes)*1e6/n,new_us=median(newtimes)*1e6/n)
 println(row)
 return row
end
function baseline()
 root=normpath(joinpath(@__DIR__, "../.."))
 source=read(`git -C $root show 4ae893b:src/vulkan/graph.jl`,String)
 method=match(r"function compiledraw\(c::Compile\{LavaDevice\}.*?\nend"s,source).match
 method=replace(method,"function compiledraw("=>"function probe_before_compiledraw(",
                       "convert_args("=>"probe_before_convert_args(")
 converter="probe_before_convert_args(args::Tuple) = map(a -> a isa LavaArray ? LavaDeviceArray(a) : a,args)\n"
 converter*="probe_before_convert_args(::Tuple{}) = ()\n"
 Base.include_string(Mantle,converter*method)
 return Mantle.probe_before_compiledraw
end

end

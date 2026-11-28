module ASBarrierProbe
using Mantle, Test
using Mantle: VK, VulkanAccelBuildContext, buildcmd

const root = normpath(joinpath(@__DIR__,"../.."))
const source = read(`git -C $root show 3898fc2:src/vulkan/raytracing/acceleration.jl`,String)
const definition,_ = Meta.parse(source,first(findfirst("function asbuildbarrier!",source)))
definition.args[1].args[1] = :old_asbuildbarrier!
Core.eval(@__MODULE__,definition)

function cycles(fn,arg,n)
    for _ in 1:n
        fn(arg)
    end
    nothing
end

function measure(ctx;n=100)
    dev = Mantle.lavadevice(ctx)
    counts = Int[]
    tok = Mantle.oneshot!(dev.bq) do emitter
        build = Mantle.AccelBuildContext(dev.bq,emitter)
        for fn in (old_asbuildbarrier!,Mantle.asbuildbarrier!)
            cycles(fn,build,10)
            push!(counts,minimum(@allocated(cycles(fn,build,n)) for _ in 1:3))
        end
    end
    Mantle.waitfor!(dev.bq,tok); Mantle.drain!(dev.bq)
    @test isempty(Mantle.validationmessages!(dev))
    @test counts[2] == 0
    @test counts[1] > counts[2]
    return (;device=ctx.device_name,barriers=n,old_bytes=counts[1],new_bytes=counts[2])
end
end

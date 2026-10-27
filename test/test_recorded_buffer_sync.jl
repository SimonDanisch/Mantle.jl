using Test
import Mantle

function replay_policy_scale!(out,x,n::Int32,scale::Float32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]=x[i]*scale)
    nothing
end

function replay_policy_sum!(out,a,b,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]=a[i]+b[i])
    nothing
end
function replay_policy_inc!(out,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]+=1f0)
    nothing
end
function replay_policy_pair!(a,b,x,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    if i<=n
        @inbounds begin a[i]=2f0*x[i];b[i]=4f0*x[i];end
    end
    nothing
end
function replay_policy_fillref!(out,value,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]=value[1])
    nothing
end
function replay_policy_accumulate!(out,x,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]+=x[i])
    nothing
end
function replay_policy_zero_alias!(out,inputs,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds out[i]=inputs[1][i]*0f0)
    nothing
end
function replay_native_words!(words,n::Int32)
    i=Int32(Mantle.KI.get_global_id().x)
    i<=n&&(@inbounds words[i]=UInt32(0))
    nothing
end

function test_recorded_buffer_sync(dev)
    M=Mantle
    @testset "Automatic recorded buffer synchronization" begin
        @test M.global_buffer_barriers(M.VulkanAPI())
        @test M.global_buffer_barriers(M.MetalAPI())
        @test !M.global_buffer_barriers(M.WebGPUAPI())
        @test !M.global_buffer_barriers(M.ROCmAPI())
        @test !M.buffer_head_ordered(nothing)
        @test M.buffer_head_ordered(dev)
        let n=1024,x=M.KA.allocate(M.backend(dev),Float32,1024),out=M.Buffer(dev,Float32,1024),p=nothing
            try
                fill!(x,1f0);words=reinterpret(UInt32,x)
                g=M.Graph(dev)
                @test M.resourceid(g,x)==M.resourceid(g,words)
                M.dispatch!(g,replay_native_words!,(words,Int32(n)),n;name="native_vector_write")
                M.dispatch!(g,replay_policy_scale!,(out,x,Int32(n),1f0),n;name="native_scalar_read")
                p=M.record!(M.Plan(g;budget=Inf))
                for _ in 1:3
                    fill!(x,1f0);M.run!(p);M.waitidle(dev)
                    @test all(iszero,Array(out))
                end
            finally
                p===nothing||M.free!(p);M.free!(out);M.KA.unsafe_free!(x)
            end
        end
        # One pass aliases separate read/write argument slots. Its write must
        # remain visible to the next reader, including overlapping views.
        for sliced in (false,true)
            n=1024;x=M.Buffer(dev,ones(Float32,n));before=M.Buffer(dev,Float32,n)
            after=M.Buffer(dev,Float32,n);p=nothing
            try
                g=M.Graph(dev);input=sliced ? M.viewof(x,(n,)) : x
                M.dispatch!(g,replay_policy_scale!,(before,x,Int32(n),1f0),n;name="snapshot")
                M.dispatch!(g,replay_policy_zero_alias!,(x,(input,),Int32(n)),n;name="aliased_write")
                M.dispatch!(g,replay_policy_scale!,(after,x,Int32(n),1f0),n;name="read_after_alias")
                p=M.record!(M.Plan(g;budget=Inf))
                @test !isempty(only(filter(pp->pp.pass.name=="read_after_alias",p.passes)).pre)
                M.run!(p);M.waitidle(dev)
                @test all(==(1f0),Array(before))
                @test all(iszero,Array(after))
            finally
                p===nothing||M.free!(p);foreach(M.free!,(x,before,after))
            end
        end
        for (producer_policy,consumer_policy) in ((M.Overlap(),M.Overlap()),(M.Compact(),M.Overlap()),(M.Overlap(),M.Compact()))
            owned=Any[];plans=Any[];n=65536
            try
                mid=M.Buffer(dev,zeros(Float32,n));out=M.Buffer(dev,zeros(Float32,n));value=M.GPURef(dev,0f0);append!(owned,(mid,out,value))
                g=M.Graph(dev);M.dispatch!(g,replay_policy_fillref!,(mid,value,Int32(n)),n;group=256,name="producer")
                a=M.record!(M.Plan(g;budget=Inf,policy=producer_policy));push!(plans,a)
                g=M.Graph(dev);M.dispatch!(g,replay_policy_accumulate!,(out,mid,Int32(n)),n;group=256,name="consumer")
                b=M.record!(M.Plan(g;budget=Inf,policy=consumer_policy));push!(plans,b)
                for k in 1:40
                    value[]=Float32(k);M.run!(a);M.run!(b)
                end
                M.waitidle(dev);@test all(==(820f0),Array(out))
            finally
                foreach(M.free!,reverse(plans));foreach(M.free!,reverse(owned))
            end
        end
        # A barrier for A also orders B's earlier write because Vulkan emits
        # GLOBAL memory dependencies. B's subsequent read needs no duplicate.
        let owned=Any[],p=nothing,n=1024
            upload(x)=begin b=M.Buffer(dev,x);push!(owned,b);b;end
            try
                x=upload(Float32.(1:n));a=upload(zeros(Float32,n));b=upload(zeros(Float32,n))
                c=upload(zeros(Float32,n));d=upload(zeros(Float32,n));out=upload(zeros(Float32,n))
                g=M.Graph(dev)
                M.dispatch!(g,replay_policy_pair!,(a,b,x,Int32(n)),n;group=256,name="write_pair")
                M.dispatch!(g,replay_policy_scale!,(c,a,Int32(n),3f0),n;group=256,name="read_a")
                M.dispatch!(g,replay_policy_scale!,(d,b,Int32(n),5f0),n;group=256,name="read_b")
                M.dispatch!(g,replay_policy_sum!,(out,c,d,Int32(n)),n;group=256,name="join_pair")
                p=M.record!(M.Plan(g;budget=Inf))
                @test !isempty(only(filter(pp->pp.pass.name=="read_a",p.passes)).pre)
                @test isempty(only(filter(pp->pp.pass.name=="read_b",p.passes)).pre)
                @test !isempty(only(filter(pp->pp.pass.name=="join_pair",p.passes)).pre)
                for _ in 1:32;M.run!(p);end
                M.waitidle(dev);@test Array(out)==26f0.*Float32.(1:n)
            finally
                p===nothing||M.free!(p);foreach(M.free!,reverse(owned))
            end
        end
        for alias in (false,true)
            owned=Any[];plans=Any[];n=1024
            upload(x)=begin b=M.Buffer(dev,x);push!(owned,b);b;end
            try
                src=upload(Float32.(1:n));left=upload(zeros(Float32,n));right=upload(zeros(Float32,n));out=upload(zeros(Float32,n))
                g=M.Graph(dev);t=M.Transient.Buffer(g,Float32,n)
                M.dispatch!(g,replay_policy_scale!,(t,src,Int32(n),2f0),n;group=256,name="producer")
                M.dispatch!(g,replay_policy_scale!,(left,t,Int32(n),3f0),n;group=256,name="reader1")
                M.dispatch!(g,replay_policy_scale!,(right,t,Int32(n),5f0),n;group=256,name="reader2")
                M.dispatch!(g,replay_policy_sum!,(out,left,right,Int32(n)),n;group=256,name="join")
                p=M.Plan(g;alias,budget=Inf);push!(plans,p)
                # No plan/recording guarantee: core's standalone compilation
                # must retain cross-run first-use dependencies.
                base=M.Compile(g;alias)
                @test_throws ArgumentError M.run!(p)
                @test M.ordered_entry(M.Compile(p))
                @test !M.ordered_entry(base)
                @test M.policy(M.Compile(p)) isa M.Overlap
                @test isempty(only(filter(pp->pp.pass.name=="producer",p.passes)).pre)
                @test !isempty(only(filter(pp->pp.pass.name=="reader1",p.passes)).pre)
                @test isempty(only(filter(pp->pp.pass.name=="reader2",p.passes)).pre)
                M.record!(p)
                for k in 1:8
                    src[:]=Float32.(1:n).*Float32(k)
                    M.run!(p)
                end
                M.waitidle(dev)
                @test Array(out)==Float32.(1:n).*128f0
                copyto!(src,Float32.(n:-1:1));M.run!(p);M.waitidle(dev)
                @test Array(out)==Float32.(n:-1:1).*16f0
            finally
                foreach(M.free!,reverse(plans));foreach(M.free!,reverse(owned))
            end
        end
        owned=Any[];plans=Any[]
        try
            a=M.Buffer(dev,zeros(Float32,256));push!(owned,a)
            g=M.Graph(dev)
            for i in 1:2
                M.dispatch!(g,replay_policy_inc!,(a,Int32(256)),256;group=256,name="increment$i")
            end
            p=M.record!(M.Plan(g;budget=Inf));push!(plans,p)
            @test !isempty(only(filter(pp->pp.pass.name=="increment2",p.passes)).pre)
            for _ in 1:40;M.run!(p);end
            M.waitidle(dev);@test all(==(80f0),Array(a))
            fill!(a,0f0);loopgraph=M.Graph(dev)
            M.repeat!(loopgraph,5) do _
                M.dispatch!(loopgraph,replay_policy_inc!,(a,Int32(256)),256;group=256,name="loop_increment")
            end
            loopplan=M.record!(M.Plan(loopgraph;budget=Inf));push!(plans,loopplan)
            @test length(loopplan.loops)==1
            for _ in 1:3;M.run!(loopplan);end
            M.waitidle(dev);@test all(==(15f0),Array(a))
            fill!(a,0f0);out=M.Buffer(dev,zeros(Float32,256));push!(owned,out)
            loopgraph=M.Graph(dev);left=M.Transient.Buffer(loopgraph,Float32,256);right=M.Transient.Buffer(loopgraph,Float32,256)
            M.repeat!(loopgraph,5) do _
                M.dispatch!(loopgraph,replay_policy_inc!,(a,Int32(256)),256;group=256,name="advance")
                M.dispatch!(loopgraph,replay_policy_scale!,(left,a,Int32(256),2f0),256;group=256,name="left")
                M.dispatch!(loopgraph,replay_policy_scale!,(right,a,Int32(256),3f0),256;group=256,name="right")
                M.dispatch!(loopgraph,replay_policy_sum!,(out,left,right,Int32(256)),256;group=256,name="join")
            end
            lp=M.record!(M.Plan(loopgraph;budget=Inf));push!(plans,lp)
            for _ in 1:3;M.run!(lp);end
            M.waitidle(dev);@test all(==(15f0),Array(a));@test all(==(75f0),Array(out))
        finally
            foreach(M.free!,reverse(plans));foreach(M.free!,reverse(owned))
        end
        owned=Any[];p=nothing
        try
            src=M.Buffer(dev,ones(Float32,1024));out=M.Buffer(dev,zeros(Float32,1024));append!(owned,(src,out))
            g=M.Graph(dev);a=M.Transient.Buffer(g,Float32,1024);b=M.Transient.Buffer(g,Float32,1024);c=M.Transient.Buffer(g,Float32,1024)
            for (name,dst,input,scale) in (("a",a,src,2f0),("b",b,a,3f0),("c",c,src,4f0))
                M.dispatch!(g,replay_policy_scale!,(dst,input,Int32(1024),scale),1024;group=256,name)
            end
            M.dispatch!(g,replay_policy_sum!,(out,b,c,Int32(1024)),1024;group=256,name="join")
            p=M.record!(M.Plan(g;alias=true,budget=Inf))
            @test sum(count(t->t.resource==0,pp.pre) for pp in p.passes)>0
            for _ in 1:20;M.run!(p);end
            M.waitidle(dev);@test all(==(10f0),Array(out))
        finally
            p===nothing||M.free!(p);foreach(M.free!,reverse(owned))
        end
    end
end

# The common runner supplies this only when including a file for a selected
# backend. Standalone callers can pass an explicitly chosen device instead.
if isdefined(Main,:MANTLE_TEST_BACKEND)
    let dev=Mantle.Device(Main.MANTLE_TEST_BACKEND)
        if Mantle.buffer_head_ordered(dev)
            test_recorded_buffer_sync(dev)
        else
            @test_skip Mantle.buffer_head_ordered(dev)
        end
    end
end

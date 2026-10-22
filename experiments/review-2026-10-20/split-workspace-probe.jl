# baseline(), then Base.invokelatest(SplitWorkspaceProbe.measure, ctx).
module SplitWorkspaceProbe
using Mantle, GPUArrays, Test, Statistics
using Mantle: LavaBackend, LavaArray, vk_context
include(joinpath(@__DIR__,"../../test/split_workspace_helpers.jl"))

mutable struct LegacyScratch
    gemm_split_scratch::Any
    gemm_split_retired::Vector{Any}
end
LegacyScratch() = LegacyScratch(nothing,Any[])

function baseline()
    root=normpath(joinpath(@__DIR__,"../.."))
    source=read(`git -C $root show 15a44e4:src/vulkan/array/gemm.jl`,String)
    method=match(r"function splitscratch\(backend::LavaBackend,.*?\nend"s,source).match
    method=replace(method,"function splitscratch("=>"function legacy(caches::LegacyScratch, ",
                   "    caches = ctx.caches\n"=>"")
    Base.include_string(@__MODULE__,method)
end

function release(state)
    state.gemm_split_scratch === nothing || Mantle.unsafe_free!(state.gemm_split_scratch)
    foreach(Mantle.unsafe_free!,state.gemm_split_retired)
    state.gemm_split_scratch=nothing;empty!(state.gemm_split_retired)
end

function correctness(ctx,old)
    backend=Mantle.backend(Mantle.lavadevice(ctx))
    state=LegacyScratch()
    provider=old ? (args...)->legacy(state,args...) : nothing
    try
        return with_split_programs(backend;workspace=provider) do program
            @test Base.mightalias(program.workspaces...) == old
            Mantle.runlaunches!(backend,program.launches)
            KernelAbstractions.synchronize(backend)
            first_result=Array(program.outputs[1])
            @test first_result == program.expected[old ? 2 : 1]
            @test Array(program.outputs[2]) == program.expected[2]
            (;device=ctx.device_name,policy=old ? "legacy" : "owned",
               first_value=first(first_result),expected=first(program.expected[1]))
        end
    finally
        release(state)
        Mantle.waitidle(Mantle.lavadevice(ctx));Mantle.drain!(ctx.default_bq)
    end
end

@noinline function prepare(provider,backend)
    array=provider(backend,nothing,256,1,16)
    Mantle.unsafe_free!(array);Mantle.drain!(backend.dispatch_bq)
    return nothing
end
@noinline function batch(provider,backend)
    for _ in 1:100;prepare(provider,backend);end
    return nothing
end

function measure(ctx)
    backend=Mantle.backend(Mantle.lavadevice(ctx))
    state=LegacyScratch()
    old=(args...)->legacy(state,args...)
    owned=Mantle.splitscratch
    try
        for _ in 1:10;prepare(old,backend);prepare(owned,backend);end
        rows=[]
        for (name,provider) in (("legacy",old),("owned",owned))
            bytes=minimum(@allocated(prepare(provider,backend)) for _ in 1:20)
            times=[(@elapsed batch(provider,backend))/100 for _ in 1:10]
            row=(;device=ctx.device_name,policy=name,host_bytes=bytes,median_us=1e6median(times))
            println(row);push!(rows,row)
        end
        return rows
    finally
        release(state)
        Mantle.waitidle(Mantle.lavadevice(ctx));Mantle.drain!(ctx.default_bq)
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
@noinline function completed(provider,backend,C,A,B,M,K,S)
    partials=provider(backend,C,M,1,S)
    try
        launches=Mantle.scalar_gemm_launches(C,A,B,M,1,K,1f0,0f0;partials)
        Mantle.runlaunches!(backend,launches)
        KernelAbstractions.synchronize(backend)
    finally
        Mantle.unsafe_free!(partials);Mantle.drain!(backend.dispatch_bq)
    end
    return nothing
end

function completed_measure(ctx,M,K;rounds=32)
    backend=Mantle.backend(Mantle.lavadevice(ctx))
    state=LegacyScratch()
    old=(args...)->legacy(state,args...)
    A=Mantle.devicearray(backend,ones(Float32,M,K))
    B=Mantle.devicearray(backend,ones(Float32,K))
    C=Mantle.devicearray(backend,zeros(Float32,M))
    S=Mantle.gemv_split(C,A,B,M,K)
    before,after=Float64[],Float64[]
    try
        for _ in 1:10
            completed(old,backend,C,A,B,M,K,S)
            completed(Mantle.splitscratch,backend,C,A,B,M,K,S)
        end
        for i in 1:rounds
            # Alternate first use to reduce ordering bias.
            for isold in (isodd(i),!isodd(i))
                provider=isold ? old : Mantle.splitscratch
                t=@elapsed completed(provider,backend,C,A,B,M,K,S)
                push!(isold ? before : after,t)
            end
        end
        @test Array(C)==fill(Float32(K),M)
        row=(;device=ctx.device_name,M,K,S,before_us=1e6median(before),
              after_us=1e6median(after),paired_ratio=median(after./before))
        println(row)
        return row
    finally
        release(state);foreach(Mantle.unsafe_free!,(A,B,C))
        Mantle.waitidle(Mantle.lavadevice(ctx));Mantle.drain!(ctx.default_bq)
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end

end

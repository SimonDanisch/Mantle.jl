# Capture fixtures on b3f991a in a disposable checkout; keep the Julia session alive.
# Apply the generator change, call Revise.revise(), then compare(fixtures) and benchmark.
# Saved recordings keep the original native programs for interleaved measurements.
module GemmFamilyProbe
using Mantle, Test, Random, Statistics
const TABLES = (:GEMM_STAGED_V2_KERNELS,:GEMM_STAGED_V2N_KERNELS,:GEMM_STAGED_ABL_KERNELS)
const STAGED_TABLES = (:GEMM_STAGED_KERNELS,TABLES...,:GEMM_STAGED_PREFETCH_KERNELS,
                       :GEMM_STAGED_DB_KERNELS,:GEMM_STAGED_V4_KERNELS,:GEMM_STAGED_GATHER_KERNELS)
const FAMILIES = (:scalar,:vec2,:narrow,:ablation,:prefetch,:doublebuf,:vec4,:gather)
schedule(table) = FAMILIES[findfirst(==(table),STAGED_TABLES)]
tilings(table) = isdefined(Mantle,:staged_gemm_tilings) ?
    Mantle.staged_gemm_tilings(schedule(table)) : keys(getproperty(Mantle,table))
kernel(table,cfg) = isdefined(Mantle,:staged_gemm_kernel) ?
    Mantle.staged_gemm_kernel(cfg,schedule(table)) : getproperty(Mantle,table)[cfg]
const ORIGINAL_KEYS = Dict(n=>Set(tilings(n)) for n in STAGED_TABLES)

function plan(ctx,cfg,family,K,M,N,A,B,out;profile=false,ablation=:all)
    args=(out,A,B,nothing,identity,Val(M),Val(N),Val(K))
    family===:GEMM_STAGED_GATHER_KERNELS && (args=(out,A,B,nothing,identity,nothing,Val(M),Val(N),Val(K)))
    family===:GEMM_STAGED_ABL_KERNELS && (args=(args...,Val(ablation)))
    kern=kernel(family,cfg)
    wg=Mantle.gemm_wg(cfg)
    g=Mantle.Graph(Mantle.lavadevice(ctx))
    Mantle.dispatch!(g,kern,args,(M÷Mantle.gemm_bm(cfg))*(N÷Mantle.gemm_bn(cfg))*wg;group=wg)
    return Mantle.record!(Mantle.Plan(g;budget=Inf,profile))
end

function fixture(ctx,cfg,family,K;blocks=2,large=false)
    Mantle.defaultdevice!(Mantle.lavadevice(ctx))
    be=Mantle.backend(Mantle.lavadevice(ctx))
    M=(large ? 6 : blocks)*Mantle.gemm_bm(cfg)
    N=(large ? 32 : blocks)*Mantle.gemm_bn(cfg)
    rng=MersenneTwister(14)
    hA=large ? fill(Float16(1),M,K) : Float16.(rand(rng,Float32,M,K).-0.5f0)
    hB=large ? fill(Float16(1),K,N) : Float16.(rand(rng,Float32,K,N).-0.5f0)
    A=Mantle.devicearray(be,hA); B=Mantle.devicearray(be,hB)
    out=Mantle.Buffer(Mantle.lavadevice(ctx),zeros(Float32,M,N))
    p=plan(ctx,cfg,family,K,M,N,A,B,out;profile=large)
    Mantle.run!(p); Mantle.waitfor!(p)
    result=Array(Mantle.storage(out))
    reference=large ? fill(Float32(K),M,N) : Float32.(hA)*Float32.(hB)
    @assert maximum(abs.(result.-reference)) < 1f-3*max(1f0,maximum(abs,reference))
    return (;ctx,cfg,family,K,M,N,A,B,out,plan=p,result,large)
end

function baseline(contexts;tables=TABLES)
    fixtures=Any[]
    @testset "baseline staged generator families" begin
        for ctx in contexts
            for family in tables, cfg in sort(collect(tilings(family)))
                f=fixture(ctx,cfg,family,2Mantle.gemm_bk(cfg))
                @test size(f.result)==(f.M,f.N)
                push!(fixtures,f)
            end
            cfg=Mantle.GEMM_TILINGS[1]
            for kblocks in (1,3,9), family in tables
                push!(fixtures,fixture(ctx,cfg,family,kblocks*Mantle.gemm_bk(cfg)))
            end
            cfg=(3,2,2,4,32,8)
            for family in tables
                push!(fixtures,fixture(ctx,cfg,family,576;large=true))
            end
            @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
            println((device=ctx.device_name,qualified=count(f->f.ctx===ctx,fixtures)))
        end
    end
    return fixtures
end

function compare(fixtures)
    after=Any[]
    @testset "shared GEMM generator preserves every output byte" begin
        for family in unique(f.family for f in fixtures)
            @test ORIGINAL_KEYS[family]==Set(tilings(family))
        end
        for f in fixtures
            Mantle.defaultdevice!(Mantle.lavadevice(f.ctx))
            p=plan(f.ctx,f.cfg,f.family,f.K,f.M,f.N,f.A,f.B,f.out;profile=f.large)
            Mantle.run!(p); Mantle.waitfor!(p)
            @test Array(Mantle.storage(f.out))==f.result
            push!(after,p)
        end
    end
    return after
end

function duration(p,dev)
    Mantle.run!(p); Mantle.waitfor!(p)
    Mantle.collect!(p.profiler,dev)
    i=only(findall(pp->!isempty(pp.dispatches),p.passes))
    return last(p.profiler.gpu_ns[i])
end

function benchmark(fixtures,after;n=15)
    for (f,new) in zip(fixtures,after)
        f.large || continue
        dev=Mantle.lavadevice(f.ctx)
        for _ in 1:5
            duration(f.plan,dev);duration(new,dev)
        end
        oldtimes,newtimes=Float64[],Float64[]
        for i in 1:n
            if isodd(i)
                push!(oldtimes,duration(f.plan,dev));push!(newtimes,duration(new,dev))
            else
                push!(newtimes,duration(new,dev));push!(oldtimes,duration(f.plan,dev))
            end
        end
        println((device=f.ctx.device_name,family=f.family,size=(f.M,f.N,f.K),
                 before_us=median(oldtimes)/1000,after_us=median(newtimes)/1000,
                 ratio=median(newtimes)/median(oldtimes)))
    end
    nothing
end
function save_shaders(plans, directory)
    mkpath(directory)
    opt=Mantle.Lava.SPIRV_Tools_jll.spirv_opt()
    dis=Mantle.Lava.SPIRV_Tools_jll.spirv_dis()
    for (i,p) in enumerate(plans)
        d=only(only(filter(pp->!isempty(pp.dispatches),p.passes)).dispatches)
        raw=joinpath(directory,"$i.spv")
        canonical=joinpath(directory,"$i.canonical.spv")
        text=joinpath(directory,"$i.txt")
        write(raw,d.launch.compiled.spirv_bytes)
        run(`$opt --strip-debug --compact-ids $raw -o $canonical`)
        run(`$dis --no-color $canonical -o $text`)
    end
end

end

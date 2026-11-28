# Run through the workspace Julia tool: include(this file); DirectHostRun.run().
isdefined(Main, :DirectHostProbe) || include("probe.jl")
Base.include(Main.DirectHostProbe, joinpath(@__DIR__, "diagnostics.jl"))
module DirectHostRun
import Main.DirectHostProbe as P
using Test, Dates, SHA
const MATCHES = ("NAVI31", "RTX 4000", "RAPHAEL", "llvmpipe")
function run()
    rows = NamedTuple[]
    mapped = NamedTuple[]
    path = joinpath(@__DIR__, "results.txt")
    open(path,"w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                println("Direct host transfer qualification ",now()," Julia=",VERSION)
                println("Mantle loaded=",any(m->nameof(m)==:Mantle,values(Base.loaded_modules)))
                @testset "direct-host native qualification" begin
                    for match in MATCHES
                        result = P.probe(match)
                        append!(rows,result)
                        @test all(r->r.ok,filter(r->r.kind in (:private_anon,:cache_read),result))
                    end
                    # Larger extent, after method/driver warmup. No source import fallback.
                    for match in MATCHES
                        result=P.probe(match;bytes=64<<20,kinds=(:private_anon,:cache_read))
                        append!(rows,result)
                        @test all(r->r.ok,result)
                    end
                    for match in MATCHES
                        push!(mapped,P.mapped_destination(match))
                    end
                    P.diagnose("NAVI31")
                    P.diagnose("RTX 4000")
                    GC.gc()
                    @test isempty(P.ERRORS)
                end
                for file in ("probe.jl","diagnostics.jl","run.jl",
                    "../../src/vulkan/runtime/external.jl","../../src/runtime/api.jl",
                    "../../src/vulkan/runtime/memory.jl","../../../Vulkan/generated/linux.jl",
                    "../../../Vulkan/generator/src/wrap/initialization.jl")
                    println("SHA256 ",bytes2hex(sha256(read(joinpath(@__DIR__,file))))," ",file)
                end
            end
        end
    end
    open(joinpath(@__DIR__,"results.tsv"),"w") do io
        println(io,"device\tkind\timport_bytes\tpayload_bytes\tok\tupload_gpu_ms\tupload_plus_readback_wall_ms\timport_ms\terror")
        for r in rows
            error=replace(r.error,'\n'=>' ','\t'=>' ')
            println(io,join((r.device,r.kind,r.bytes,r.bytes-12,r.ok,r.gpu_ms,r.wall_ms,r.import_ms,error),'\t'))
        end
    end
    println("Exact-import observations: ",count(r->r.ok,rows),"/",length(rows)," copied; ",
            count(r->!r.ok,rows)," source-specific failures retained for diagnosis.")
    println("Mapped-final-destination cases: ",length(mapped)," passed. Validation errors: ",length(P.ERRORS))
    println("Full results: ",path)
    (; rows,mapped)
end
end

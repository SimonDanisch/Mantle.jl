module DiagonalProbe
using Mantle, LinearAlgebra, Test, Random
using Mantle: LavaArray, AbstractGPUArray

const root = normpath(joinpath(@__DIR__,"../.."))
const old_source = read(`git -C $root show b06b12c:src/vulkan/array/gemm.jl`,String)
const old_start = first(findfirst("# Disambiguation: Diagonal",old_source))
const old_stop = first(findfirst("\"\"\"Does this device implement the tile",old_source))-1
const old_section = old_source[old_start:old_stop]
for m in eachmatch(r"function LinearAlgebra\.mul!\(C::LavaArray[\s\S]*?\nend",old_section)
    Core.eval(@__MODULE__,Meta.parse(replace(m.match,"LinearAlgebra.mul!"=>"old_mul!";count=1)))
end

function baseline(contexts)
    cases = Any[]
    rng = MersenneTwister(83)
    for ctx in contexts
        dev = Mantle.lavadevice(ctx)
        Mantle.defaultdevice!(dev)
        be = Mantle.backend(dev)
        for T in (Float32,ComplexF32), side in (:left,:right,:both),
            layout in (side === :both ? (:plain,) : (:plain,:transpose,:adjoint)),
            beta in (0,3)
            n,m = 6,4
            hd = rand(rng,T,n)
            D = Diagonal(Mantle.devicearray(be,hd))
            if side === :both
                A,B = D,Diagonal(Mantle.devicearray(be,rand(rng,T,n)))
                dims = (n,n)
            else
                dims = side === :left ? (n,m) : (m,n)
                raw = Mantle.devicearray(be,rand(rng,T,layout === :plain ? dims : reverse(dims)))
                matrix = layout === :plain ? raw : layout === :transpose ? transpose(raw) : adjoint(raw)
                A,B = side === :left ? (D,matrix) : (matrix,D)
            end
            seed = side === :both && beta == 0 ? fill(T(NaN),dims) : rand(rng,T,dims)
            C = Mantle.devicearray(be,seed)
            old_mul!(C,A,B,T(2),T(beta))
            push!(cases,(;ctx,A,B,C,seed,alpha=T(2),beta=T(beta),result=Array(C)))
        end
        @test isempty(Mantle.validationmessages!(dev))
    end
    return cases
end

function compare(cases)
    @testset "diagonal delegation preserves output bytes" begin
        for f in cases
            Mantle.defaultdevice!(Mantle.lavadevice(f.ctx))
            copyto!(f.C,f.seed)
            mul!(f.C,f.A,f.B,f.alpha,f.beta)
            @test reinterpret(UInt32,vec(Array(f.C))) == reinterpret(UInt32,vec(f.result))
        end
        for ctx in unique(f.ctx for f in cases)
            dev = Mantle.lavadevice(ctx)
            Mantle.waitidle(dev); Mantle.drain!(dev.bq)
            @test isempty(Mantle.validationmessages!(dev))
        end
    end
end

function allocations(cases,measure_pair)
    rows = NamedTuple[]
    for f in cases
        eltype(f.C) === Float32 && iszero(f.beta) || continue
        any(x->x isa Union{Transpose,Adjoint},(f.A,f.B)) && continue
        Mantle.defaultdevice!(Mantle.lavadevice(f.ctx))
        side = f.A isa Diagonal ? (f.B isa Diagonal ? "both" : "left") : "right"
        args = (f.C,f.A,f.B,f.alpha,f.beta)
        row = measure_pair(f.ctx,"diagonal-$side","b06b12c",old_mul!,mul!,args,args)
        dev = Mantle.lavadevice(f.ctx)
        old_drained = drained_allocation(old_mul!,args,dev)
        new_drained = drained_allocation(mul!,args,dev)
        @test new_drained <= old_drained
        push!(rows,(;row...,old_drained,new_drained))
    end
    return rows
end

# Drain between individual samples so asynchronous queue capacity is not compared.
function drained_allocation(fn,args,dev;n=30)
    best = typemax(Int)
    for _ in 1:n
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        best = min(best,@allocated fn(args...))
    end
    Mantle.waitidle(dev); Mantle.drain!(dev.bq)
    return best
end
end

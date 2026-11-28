module ArrayKernelProbe
using Test
import Mantle, KernelInterface as KI
using LinearAlgebra: Transpose
using Mantle: AbstractGPUArray, fft_kernel!, fftplan, fftgroup, FFT_SKEW, get_backend,
              workgrouplimit, sharedbudget, kilaunch!, subgroupwidth,
              ArrayLaunch, runlaunches!, dispatchlaunches!, backend, caps, sub_group_reduce_add

const FFT_MIXED_KERNELS = Dict{Tuple{Tuple,Int},Any}()
const root = normpath(joinpath(@__DIR__,"../.."))
const baseline = read(`git -C $root show 9b008fa:src/array/fft.jl`,String)
for signature in ("function fft!(dst", "function fftmixed!(dst", "function fftmixed_kernel(")
    ex,_ = Meta.parse(baseline,first(findfirst(signature,baseline)))
    Core.eval(@__MODULE__,ex)
end

const gemv_baseline = read(`git -C $root show a367c3b:src/array/gemv.jl`,String)
Base.include_string(@__MODULE__,gemv_baseline,"gemv-baseline-a367c3b.jl")

function cycles(fn,args,n)
    for _ in 1:n
        fn(args...)
    end
    nothing
end

# Julia allocation after warming queue capacity and compilation; excludes driver memory.
function measure_pair(ctx,path,revision,oldfn,newfn,oldargs,newargs;launches=100,samples=3)
    dev = Mantle.lavadevice(ctx)
    cycles(oldfn,oldargs,1)
    expected = reinterpret(UInt32,vec(Array(first(oldargs))))
    cycles(newfn,newargs,1)
    @test reinterpret(UInt32,vec(Array(first(newargs)))) == expected
    for (fn,args) in ((oldfn,oldargs),(newfn,newargs))
        cycles(fn,args,launches+20)
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
    end
    counts = map(((oldfn,oldargs),(newfn,newargs))) do (fn,args)
        allocations = Int[]
        for _ in 1:samples
            Mantle.waitidle(dev); Mantle.drain!(dev.bq)
            push!(allocations,@allocated cycles(fn,args,launches))
        end
        minimum(allocations)
    end
    Mantle.waitidle(dev); Mantle.drain!(dev.bq)
    @test isempty(Mantle.validationmessages!(dev))
    return (;device=ctx.device_name,path,baseline=revision,launches,old_bytes=counts[1],new_bytes=counts[2])
end

function measure_fft(ctx;kw...)
    be = Mantle.backend(Mantle.lavadevice(ctx))
    rows = NamedTuple[]
    for (path,N,rs) in (("fft-power2",512,nothing),("fft-mixed",96,(8,4,3)))
        src = KI.allocate(be,ComplexF32,N,8); copyto!(src,randn(ComplexF32,N,8))
        old = similar(src); new = similar(src)
        oldfn,newfn = rs === nothing ? (fft!,Mantle.fft!) : (fftmixed!,Mantle.fftmixed!)
        oldargs = rs === nothing ? (old,src) : (old,src,rs)
        newargs = rs === nothing ? (new,src) : (new,src,rs)
        push!(rows,measure_pair(ctx,path,"9b008fa",oldfn,newfn,oldargs,newargs;kw...))
    end
    rows
end

function measure_gemv(ctx;kw...)
    be = Mantle.backend(Mantle.lavadevice(ctx))
    rows = NamedTuple[]
    for transposed in (false,true)
        W = KI.allocate(be,Float32,1280,1280); copyto!(W,randn(Float32,1280,1280))
        x = KI.allocate(be,Float32,1280); copyto!(x,randn(Float32,1280))
        old = similar(x); new = similar(x)
        operands = transposed ? (x,transpose(W)) : (x,W)
        push!(rows,measure_pair(ctx,transposed ? "gemv-n" : "gemv-k","a367c3b",
                               gemv!,Mantle.gemv!,(old,operands...),(new,operands...);kw...))
    end
    rows
end
end

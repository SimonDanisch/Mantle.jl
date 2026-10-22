module FFTConfigProbe
using Test
import Mantle, KernelInterface as KI
using Mantle: AbstractGPUArray, fft_kernel!, fftplan, fftgroup, FFT_SKEW, get_backend,
              workgrouplimit, sharedbudget, kilaunch!

const FFT_MIXED_KERNELS = Dict{Tuple{Tuple,Int},Any}()
const root = normpath(joinpath(@__DIR__,"../.."))
const baseline = read(`git -C $root show 9b008fa:src/array/fft.jl`,String)
for signature in ("function fft!(dst", "function fftmixed!(dst", "function fftmixed_kernel(")
    ex,_ = Meta.parse(baseline,first(findfirst(signature,baseline)))
    Core.eval(@__MODULE__,ex)
end

function cycles(fn,out,src,rs,n)
    for _ in 1:n
        rs === nothing ? fn(out,src) : fn(out,src,rs)
    end
    nothing
end

"""Paired host allocations, after warming queue capacity and kernel compilation.
Returns measured rows. GPU throughput and native driver allocations are excluded."""
function measure(ctx; launches=100, samples=3)
    dev = Mantle.lavadevice(ctx)
    be = Mantle.backend(dev)
    rows = NamedTuple[]
    for (path,N,rs) in (("power2",512,nothing),("mixed",96,(8,4,3)))
        src = KI.allocate(be,ComplexF32,N,8)
        copyto!(src,randn(ComplexF32,N,8))
        out = similar(src)
        oldfn,newfn = rs === nothing ? (fft!,Mantle.fft!) : (fftmixed!,Mantle.fftmixed!)
        cycles(oldfn,out,src,rs,1)
        expected = reinterpret(UInt32,vec(Array(out)))
        cycles(newfn,out,src,rs,1)
        @test reinterpret(UInt32,vec(Array(out))) == expected
        for fn in (oldfn,newfn)
            cycles(fn,out,src,rs,launches+20)
            Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        end
        counts = map((oldfn,newfn)) do fn
            allocations = Int[]
            for _ in 1:samples
                Mantle.waitidle(dev); Mantle.drain!(dev.bq)
                push!(allocations,@allocated cycles(fn,out,src,rs,launches))
            end
            minimum(allocations)
        end
        push!(rows,(;device=ctx.device_name,path,launches,old_bytes=counts[1],new_bytes=counts[2]))
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
    return rows
end
end

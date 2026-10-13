# `Mantle.argmap`: `map` over an argument tuple of any length, allocating nothing.
#
# Base's `map(f, ::Tuple)` unrolls only up to 31 elements and from 32 on fills a
# `Vector{Any}` and splats it back. RayMakie's overlay stages take 69 arguments,
# and resolving, storing and adapting them once per draw per frame cost a
# 5000-plot GUI 290 MiB a frame. A first fix, `ntuple(i -> f(args[i]), Val(N))`,
# allocated as much: with `f` an adaptor turning arrays into device arrays — what
# `devargs` maps — it measured 78.7 KB per call on a real 69-argument draw.
#
# No device here. The adaptor below does what Lava's does to the types, an array
# into an isbits pointer wrapper and everything else unchanged, which is what
# made the `ntuple` form allocate: measured 42 KB per call with it, 0 with
# `argmap`.

using Test, Mantle
using Mantle: Vec3f, Vec4f

const AM_N = 69

struct AMWrap{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end
struct AMAdaptor
    tag::Int
end
amadapt(::AMAdaptor, a::Array{T,N}) where {T,N} = AMWrap{T,N}(pointer(a), size(a))
amadapt(::AMAdaptor, x) = x

# Ten calls, so one allocation per call shows. A function, not a closure in the
# testset: a testset's body is top-level code, where a boxed result would be the
# test's allocation and not argmap's.
function amdevargs(ad, t, n)
    s = 0
    for _ in 1:n
        s += length(Mantle.argmap(a -> amadapt(ad, a), t))
    end
    return s
end

@testset "argmap: map over a long argument tuple" begin
    kinds = ([Vec3f(0)], UInt32[1], Vec4f[Vec4f(0)], Float32[1], Vec3f(1), Int32(2), 1.5f0)
    t = ntuple(i -> kinds[mod1(i, length(kinds))], AM_N)
    ad = AMAdaptor(1)
    f = a -> amadapt(ad, a)
    @test Mantle.argmap(f, t) == map(f, t)
    @test Mantle.argmap(f, ()) == ()
    @test Mantle.argmap(f, (Int32(3),)) == (Int32(3),)

    # Inferred element by element, so a draw's argument types survive it.
    @test only(Base.return_types(Mantle.argmap, (typeof(f), typeof(t)))) == typeof(map(f, t))

    @test GC.@preserve kinds begin
        amdevargs(ad, t, 1)
        @allocated amdevargs(ad, t, 10)
    end == 0
end

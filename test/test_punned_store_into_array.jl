# A struct built with every field zero is zero on the device.
#
# Hikari fills a BSDF slot no layer uses with a lobe of the slot's type with every field
# zero, out of line, and LLVM writes the zeros as `store i32 0` at each byte offset of
# the returned struct. Where a store landed at the start of a `[1 x [4 x float]]` that
# is not the first member of its struct, Lava stored the integer into the array and
# spirv-val refused the module. Lava's `test_punned_store_into_array.jl` checks the
# emission; this runs it, against the same struct filled on the host.

using Test, Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

struct PunS4
    data::NTuple{4, Float32}
end
struct PunLobe
    albedo::PunS4
    radius::PunS4
    extra::PunS4
    alpha::Float32
    ior::Float32
end
struct PunFresnel
    a::Float32
    b::Float32
    c::Float32
    d::Float32
    e::Float32
    lobe::PunLobe
end
struct PunWeighted
    fresnel::PunFresnel
    weight::PunS4
    sample_weight::Float32
    n::NTuple{3, Float32}
    t::NTuple{3, Float32}
end

@generated function punzero(::Type{T}) where {T}
    isprimitivetype(T) && return :(zero($T))
    T <: Tuple && return Expr(:tuple, (:(punzero($(fieldtype(T, i)))) for i in 1:fieldcount(T))...)
    return Expr(:new, T, (:(punzero($(fieldtype(T, i)))) for i in 1:fieldcount(T))...)
end
@noinline punblank() = punzero(PunWeighted)

function punfilled(x::Float32)
    s = PunS4((x, x, x, x))
    return PunWeighted(PunFresnel(x, x, x, x, x, PunLobe(s, s, s, x, x)), s, x, (x, x, x), (x, x, x))
end

KA.@kernel function blank_or_filled!(out, @Const(xs))
    i = KA.@index(Global)
    x = @inbounds xs[i]
    @inbounds out[i] = x > 0.5f0 ? punblank() : punfilled(x)
end

@testset "a struct built with every field zero is zero on the device" begin
    xs = Float32[0.1, 0.7, 0.3, 0.9, 0.5, 0.6]
    out = Mantle.devicearray(TESTBACKEND, [punfilled(1f0) for _ in xs])
    blank_or_filled!(TESTBACKEND)(out, Mantle.devicearray(TESTBACKEND, xs); ndrange = length(xs))
    @test Array(out) == [x > 0.5f0 ? punzero(PunWeighted) : punfilled(x) for x in xs]
end

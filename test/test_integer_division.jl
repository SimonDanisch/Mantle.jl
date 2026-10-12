# 64-bit integer division on the device agrees with the host.
#
# Lava hands the driver no 64-bit division (`lower_wide_divisions!`): a constant
# divisor becomes a multiply-high sequence, a runtime one divides in 32 bits when
# both magnitudes fit and in 64 bits otherwise. That rewrites every `÷`, `%`,
# `fld` and `mod` on `Int64`/`UInt64` in every kernel, so this checks it against
# the host over the edges of each path: signs, both sides of 2^32, the extremes,
# divisors whose magic constant needs the extra add, and divisors past 2^32.
#
# Why it exists is in `test_shared_index_division.jl` and
# `test_int32_cartesian_miscompile.jl`: NVIDIA miscompiles the code around its
# own 64-bit division.

using Test, Mantle, Random
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

function idiv_const!(q, r, x, ::Val{D}, n::Int32) where {D}
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds q[i] = x[i] ÷ D
        @inbounds r[i] = x[i] % D
    end
    return nothing
end

function idiv_runtime!(q, r, f, m, x, y, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds a, b = x[i], y[i]
        @inbounds q[i] = a ÷ b
        @inbounds r[i] = a % b
        @inbounds f[i] = fld(a, b)
        @inbounds m[i] = mod(a, b)
    end
    return nothing
end

"Dividends: small, both sides of 2^32 and 2^63, the extremes, and random bits."
function dividends(::Type{T}, rng) where {T}
    edges = T[0, 1, 2, 3, 95, 96, 97, 1000, 2^31 - 1, 2^31, 2^32 - 1, 2^32, 2^32 + 1,
              typemax(T), typemax(T) - 1, typemax(T) ÷ 2, typemax(T) ÷ 2 + 1]
    T <: Signed && append!(edges, -edges, [typemin(T), typemin(T) + 1])
    return unique([edges; T.(0:2000); rand(rng, T, 4000); rand(rng, T, 2000) .>> rand(rng, 0:63, 2000)])
end

devarray(v) = Mantle.devicearray(TESTBACKEND, v)

@testset "64-bit division by a constant" begin
    rng = MersenneTwister(11)
    for (T, ds) in ((Int64, Int64[1, -1, 2, -2, 3, -3, 7, -7, 96, -96, 641, 1000, 2^31 - 1, 2^31,
                                 2^32 + 1, 10^12, typemax(Int64), -typemax(Int64), typemin(Int64)]),
                    (UInt64, UInt64[1, 2, 3, 7, 96, 641, 1000, 2^31, 2^32 - 1, 2^32, 2^32 + 1,
                                    UInt64(2)^63 + 1, typemax(UInt64)]))
        x = dividends(T, rng)
        for d in ds
            # `typemin ÷ -1` overflows on the host too.
            xs = d == -1 ? filter(!=(typemin(T)), x) : x
            n = length(xs)
            q, r = devarray(zeros(T, n)), devarray(zeros(T, n))
            KI.@launch TESTBACKEND ndrange = n workgroupsize = 64 idiv_const!(
                q, r, devarray(xs), Val(d), Int32(n))
            @test Array(q) == xs .÷ d
            @test Array(r) == xs .% d
        end
    end
end

@testset "64-bit division by a runtime value" begin
    rng = MersenneTwister(12)
    for T in (Int64, UInt64)
        x = dividends(T, rng)
        n = length(x)
        # Small divisors (the 32-bit path whenever the dividend fits), divisors
        # past 2^32 (the 64-bit path), and random bits; never zero, and never
        # `-1` against `typemin`.
        pool = [T.(1:300); T.([2^32 - 1, 2^32, 2^32 + 7, 10^15]); rand(rng, T, 300)]
        T <: Signed && append!(pool, -pool)
        y = [let b = rand(rng, pool)
                 b == 0 || (T <: Signed && b == -1 && a == typemin(T)) ? one(T) : b
             end for a in x]
        q, r, f, m = (devarray(zeros(T, n)) for _ in 1:4)
        KI.@launch TESTBACKEND ndrange = n workgroupsize = 64 idiv_runtime!(
            q, r, f, m, devarray(x), devarray(y), Int32(n))
        @test Array(q) == x .÷ y
        @test Array(r) == x .% y
        @test Array(f) == fld.(x, y)
        @test Array(m) == mod.(x, y)
    end
end

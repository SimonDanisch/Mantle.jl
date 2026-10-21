# LLVM narrows an integer to whatever width it can prove is enough, unless the module's
# datalayout names the native widths. A loop over the six neighbours of a cell, picking
# each with a chain of comparisons on the loop counter, became a switch on an `i63`.
# SPIR-V has no 63-bit integer, and the emitter stopped with "Unsupported integer width:
# 63". Lava now declares SPIR-V's widths (`n8:16:32:64`, as LLVM's own SPIR-V target
# does) once the module is linked. Found in the FLIP whirlpool's level-set
# redistancing, whose sweep this is.

using Test, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Point3f, Vec3f
include(joinpath(@__DIR__, "testbackend.jl"))

# The six neighbours of a cell, `m` in 1:6.
@inline function sixth(i, j, k, m)
    m == 1 && return (i + 1, j, k)
    m == 2 && return (i - 1, j, k)
    m == 3 && return (i, j + 1, k)
    m == 4 && return (i, j - 1, k)
    m == 5 && return (i, j, k + 1)
    return (i, j, k - 1)
end

"The distance at which |∇φ| = 1 from the smallest neighbour distances along each axis (Godunov upwind)."
@inline function upwind(a, b, c, h)
    a, b = min(a, b), max(a, b); b, c = min(b, c), max(b, c); a, b = min(a, b), max(a, b)
    d = a + h
    if d > b
        d = 0.5f0 * (a + b + sqrt(max(2f0 * h * h - (a - b)^2, 0f0)))
        if d > c
            t = a + b + c
            d = (t + sqrt(max(t * t - 3f0 * (a * a + b * b + c * c - h * h), 0f0))) / 3f0
        end
    end
    return d
end

"A box of cells `dx` wide from `lo`, solid outside a cylinder of `radius` and under `floor`."
struct Tank
    lo::Point3f
    dx::Float32
    radius::Float32
    floor::Float32
end
@inline solid(t::Tank, i, j, k) = (p = t.lo + Vec3f(i - 0.5f0, j - 0.5f0, k - 0.5f0) * t.dx;
                                   p[1] * p[1] + p[2] * p[2] > t.radius * t.radius || p[3] < t.floor)
@inline inside3(A, i, j, k) = 1 <= i <= size(A, 1) && 1 <= j <= size(A, 2) && 1 <= k <= size(A, 3)

# One Jacobi sweep of redistancing: a solid cell takes the lowest value of the water next
# to it; a water cell next to the surface (a neighbour of the other sign) keeps its
# value, the others take the upwind distance from their neighbours.
function sweep(φ, t::Tank, i, j, k, φmax)
    c = @inbounds φ[i, j, k]
    if solid(t, i, j, k)
        lowest = c
        found = false
        for m in 1:6
            a, b, cc = sixth(i, j, k, m)
            if inside3(φ, a, b, cc) && !solid(t, a, b, cc)
                f = @inbounds φ[a, b, cc]
                lowest = found ? min(lowest, f) : f
                found = true
            end
        end
        return found ? lowest : clamp(c, -φmax, φmax)
    end
    inside = c < 0f0
    frozen = false
    ax = φmax; ay = φmax; az = φmax
    for m in 1:6
        a, b, cc = sixth(i, j, k, m)
        if inside3(φ, a, b, cc) && !solid(t, a, b, cc)
            f = @inbounds φ[a, b, cc]
            (f < 0f0) != inside && (frozen = true)
            d = abs(f)
            m <= 2 ? (ax = min(ax, d)) : (m <= 4 ? (ay = min(ay, d)) : (az = min(az, d)))
        end
    end
    frozen && return c
    d = min(upwind(ax, ay, az, t.dx), φmax)
    return inside ? -d : d
end

@kernel inbounds=true function sweep_kernel!(out, @Const(φ), t::Tank, φmax::Float32)
    i, j, k = @index(Global, NTuple)
    out[i, j, k] = sweep(φ, t, i, j, k, φmax)
end

@testset "a switch over six neighbours keeps to native integer widths" begin
    n = (28, 28, 16)
    t = Tank(Point3f(-3.5, -3.5, -2.25), 0.25f0, 3f0, -2f0)
    # A tilted water surface, its distance scrambled away from the surface, as redistancing finds it.
    φ = Float32[(0.1f0 * i + 0.05f0 * j + k - 9.5f0) * t.dx for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    φ .+= Float32[abs(x) > 2t.dx ? 0.7f0 * x : 0f0 for x in φ]
    want = Float32[sweep(φ, t, i, j, k, 3f0) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    dφ = Mantle.devicearray(TESTBACKEND, φ)
    out = KA.zeros(TESTBACKEND, Float32, n...)
    sweep_kernel!(TESTBACKEND, (8, 8, 4))(out, dφ, t, 3f0; ndrange = n)
    KA.synchronize(TESTBACKEND)
    @test Array(out) ≈ want
end

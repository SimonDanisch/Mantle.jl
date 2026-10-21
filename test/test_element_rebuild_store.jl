# A kernel that loads an array element and stores a new one built partly from it,
# `v[i] = Vec4f(s * a[1], s * a[2], a[3], a[4])`. LLVM writes the store as four float
# stores at byte offsets `16i` and `16i | 4`, `16i | 8`, `16i | 12`: a multiple of the
# element size with a constant inside the element, joined by a disjoint `or`. The SPIR-V
# emitter divided such a byte offset by the element size and so dropped the constant:
# every store landed on the first float and the element came back `(a[4], a[2], a[3], a[4])`.
# Found as positions that a swirl kernel moved in one array and not in the other.

using Test, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index
using GeometryBasics: Vec4f, Point3f
include(joinpath(@__DIR__, "testbackend.jl"))

@kernel function rebuild_xy!(v, s::Float32)
    i = @index(Global)
    @inbounds a = v[i]
    @inbounds v[i] = Vec4f(s * a[1], s * a[2], a[3], a[4])
end

@kernel function rebuild_z!(v, s::Float32)
    i = @index(Global)
    @inbounds a = v[i]
    @inbounds v[i] = Vec4f(a[1], a[2], s * a[3], a[4])
end

@kernel function rebuild_point_y!(p, s::Float32)
    i = @index(Global)
    @inbounds a = p[i]
    @inbounds p[i] = Point3f(a[1], s * a[2], a[3])
end

@testset "an element rebuilt from itself stores every field" begin
    v = Mantle.devicearray(TESTBACKEND, [Vec4f(1, 2, 3, 4), Vec4f(5, 6, 7, 8)])
    rebuild_xy!(TESTBACKEND, 64)(v, 10f0; ndrange = 2)
    KA.synchronize(TESTBACKEND)
    @test Array(v) == [Vec4f(10, 20, 3, 4), Vec4f(50, 60, 7, 8)]

    v = Mantle.devicearray(TESTBACKEND, [Vec4f(1, 2, 3, 4), Vec4f(5, 6, 7, 8)])
    rebuild_z!(TESTBACKEND, 64)(v, 10f0; ndrange = 2)
    KA.synchronize(TESTBACKEND)
    @test Array(v) == [Vec4f(1, 2, 30, 4), Vec4f(5, 6, 70, 8)]

    # 12-byte elements: the constants inside the element are not powers of two.
    p = Mantle.devicearray(TESTBACKEND, [Point3f(1, 2, 3), Point3f(4, 5, 6), Point3f(7, 8, 9)])
    rebuild_point_y!(TESTBACKEND, 64)(p, 10f0; ndrange = 3)
    KA.synchronize(TESTBACKEND)
    @test Array(p) == [Point3f(1, 20, 3), Point3f(4, 50, 6), Point3f(7, 80, 9)]
end

# Reductions over device arrays, including arrays of structs.
#
# Struct element types in `mapreduce` are a frequent source of regressions: the
# whole pipeline is involved — struct layout, how the array reaches the kernel,
# shared memory, subgroup reductions — and the shapes below are the ones the
# renderers reduce over (an `NTuple` field as a spectral sample carries, several
# scalar fields, mixed `Float32`/`Int32` fields as a work item has them).
#
# From `test/vulkan/test_ci.jl`, the old fast-CI driver, which was never
# registered in `runtests.jl`. Its other sections are covered elsewhere (see the
# portable-tests report); these two were not. The struct types were declared
# inside a `@testset` there, which is not top level, so that section could not
# have run at all.

using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))

struct SpectrumVal
    wavelengths::NTuple{4, Float32}
end

struct Vec3f32
    x::Float32
    y::Float32
    z::Float32
end

struct WorkItem
    origin_x::Float32
    origin_y::Float32
    origin_z::Float32
    pixel_index::Int32
    weight::Float32
end

@testset "reductions" begin
    x = Mantle.devicearray(TESTBACKEND, rand(Float32, 1024))
    @test sum(x) ≈ sum(Array(x)) rtol = 1e-4
    @test minimum(x) ≈ minimum(Array(x))
    @test maximum(x) ≈ maximum(Array(x))

    # 2D reduction
    m = Mantle.devicearray(TESTBACKEND, rand(Float32, 32, 64))
    @test Array(sum(m; dims = 1)) ≈ sum(Array(m); dims = 1) rtol = 1e-4
    @test Array(sum(m; dims = 2)) ≈ sum(Array(m); dims = 2) rtol = 1e-4

    # Large reduction (multi-pass)
    big = Mantle.devicearray(TESTBACKEND, rand(Float32, 100_000))
    @test sum(big) ≈ sum(Array(big)) rtol = 1e-3
end

@testset "mapreduce with struct element types" begin
    # NTuple field (common in spectral rendering): sum of the first wavelength.
    data = [SpectrumVal(ntuple(j -> rand(Float32), 4)) for _ in 1:512]
    ga = Mantle.devicearray(TESTBACKEND, data)
    @test mapreduce(x -> x.wavelengths[1], +, ga) ≈
          mapreduce(x -> x.wavelengths[1], +, data) rtol = 1e-3

    # Multi-field struct: sum of magnitudes squared.
    vecs = [Vec3f32(rand(Float32), rand(Float32), rand(Float32)) for _ in 1:1024]
    gv = Mantle.devicearray(TESTBACKEND, vecs)
    @test mapreduce(v -> v.x^2 + v.y^2 + v.z^2, +, gv) ≈
          mapreduce(v -> v.x^2 + v.y^2 + v.z^2, +, vecs) rtol = 1e-3

    # Struct with mixed types (common in Hikari's work items).
    items = [WorkItem(rand(Float32), rand(Float32), rand(Float32), Int32(i), rand(Float32))
             for i in 1:256]
    gi = Mantle.devicearray(TESTBACKEND, items)
    @test mapreduce(w -> w.weight, +, gi) ≈ mapreduce(w -> w.weight, +, items) rtol = 1e-3
    # Minimum with a struct accessor.
    @test mapreduce(w -> w.weight, min, gi) ≈ mapreduce(w -> w.weight, min, items)

    # 2D reduction with structs.
    hmat = reshape([Vec3f32(rand(Float32), rand(Float32), rand(Float32)) for _ in 1:128], 8, 16)
    mat = Mantle.devicearray(TESTBACKEND, hmat)
    @test Array(mapreduce(v -> v.x, +, mat; dims = 1)) ≈
          mapreduce(v -> v.x, +, hmat; dims = 1) rtol = 1e-3

    # ComplexF32 reduction (uses struct-like layout internally).
    cx = Mantle.devicearray(TESTBACKEND, rand(ComplexF32, 512))
    @test sum(cx) ≈ sum(Array(cx)) rtol = 1e-3
    @test mapreduce(abs2, +, cx) ≈ mapreduce(abs2, +, Array(cx)) rtol = 1e-3
end

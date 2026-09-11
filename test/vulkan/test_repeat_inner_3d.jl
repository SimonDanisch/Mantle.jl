# Regression: NVIDIA miscompiles the standalone `dims[1]*dims[2]` stride product
# in a Cartesian->linear load offset, dropping the I[1] term.
#
# `linear_index(dims, I::CartesianIndex{3})` used to expand column-major as
#   I[1] + dims[1]*(I[2]-1) + dims[1]*dims[2]*(I[3]-1)
# which materialises a `dims[1]*dims[2]` product (and shares dims[1] across two
# terms). NVIDIA's shader compiler evaluates that wrong when the result feeds a
# PhysicalStorageBuffer load offset: the I[1] term is dropped, so every read lands
# on source row 1. The SPIR-V is valid (spirv-val clean) — a pure driver miscompile,
# same complex-integer family as the loop-unswitch and shared-PSB-access-chain bugs.
#
# Surfaced through GPUArrays `repeat_inner_dst_kernel!` (xs[CartesianIndex(sdx)])
# whenever `repeat(x; inner)` has a 3-D inner with argmax(inner)==1. ~2/3 of the
# output was wrong.
#
# Fixed in `array/ka_backend.jl` `linear_index` by using Horner form
#   I[1] + dims[1]*((I[2]-1) + dims[2]*(I[3]-1))
# (and a Horner loop for general N), which never forms the nested product.

using Test, Lava, Mantle
using Random

@testset "repeat(inner=) 3-D — NVIDIA stride-product miscompile" begin
    # The originally-failing case: 2-D source, 3-D inner, argmax(inner)==1.
    xmat = reshape(Float32.(1:12), 3, 4)
    g = Mantle.LavaArray(xmat)
    for inner in [(3, 1, 2), (2, 1, 1), (3, 2, 2), (1, 3, 1), (4, 1, 3)]
        @test Array(repeat(g, inner = inner)) == repeat(xmat, inner = inner)
    end

    # Genuine 3-D source arrays (linear_index NTuple{3} path) and 4-D (general
    # Horner-loop path).
    x3 = reshape(Float32.(1:24), 2, 3, 4)
    g3 = Mantle.LavaArray(x3)
    for inner in [(2, 1, 2), (1, 2, 3), (3, 1, 1)]
        @test Array(repeat(g3, inner = inner)) == repeat(x3, inner = inner)
    end

    x4 = reshape(Float32.(1:48), 2, 3, 2, 4)
    g4 = Mantle.LavaArray(x4)
    for inner in [(2, 1, 3, 1), (1, 2, 1, 2)]
        @test Array(repeat(g4, inner = inner)) == repeat(x4, inner = inner)
    end

    # The bug collapsed every output row to source row 1 — guard against silent
    # regression by asserting rows actually differ.
    r = Array(repeat(g, inner = (3, 1, 2)))
    @test r[1, 1, 1] != r[4, 1, 1]
end

# Direct GPU unit test of linear_index{3}: index a real 3-D source with indices
# built via division (the exact shape that dropped the I[1] term).
@testset "linear_index{3} load offset is exact (no dropped term)" begin
    using Lava.KernelAbstractions
    backend = Mantle.defaultbackend()
    bq = Mantle.batchqueue(Mantle.Device())
    xsrc = reshape(Float32.(1:(3 * 4 * 2)), 3, 4, 2)   # 3-D source
    xg = Mantle.LavaArray(xsrc)

    @kernel function ri_probe!(out, @Const(xs), inner::NTuple{3,Int})
        odx = @index(Global, Cartesian)
        di = odx.I
        sdx = ntuple(3) do i
            @inbounds (di[i] - 1) ÷ inner[i] + 1
        end
        @inbounds out[odx] = xs[CartesianIndex(sdx)]
    end

    out = Mantle.LavaArray(zeros(Float32, 9, 4, 2))
    ri_probe!(backend)(out, xg, (3, 1, 1); ndrange = (9, 4, 2))
    Mantle.flush!(bq)
    got = Array(out)
    expref = similar(got)
    for k in 1:2, j in 1:4, i in 1:9
        expref[i, j, k] = xsrc[(i - 1) ÷ 3 + 1, j, k]
    end
    @test got == expref
    @test got[1, 1, 1] != got[4, 1, 1]   # the bug collapsed every row to row 1
end

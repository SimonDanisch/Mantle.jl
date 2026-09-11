# Every path through `GPUArrays._copyto!(::AnyLavaArray, ::Broadcasted)`.
#
# Lava overrides that method to launch over a flat range and, when every operand
# already has the destination's shape, to flatten the expression tree as well
# (see `array/gpuarrays.jl` — worth 6x on elementwise bandwidth). Three kernels
# come out of that decision, and the cases below are one per kernel plus the two
# wrapper shapes that pick the awkward ones.
#
# These exist because narrowing the linear->Cartesian `divrem` to `Int32` — a
# change that is worth 9% of an inference step in a *different* kernel — made
# the view and permuted operands silently return wrong values here (off by 252
# and 18) while the same-shape cases stayed exact. Nothing else in the suite
# caught it.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "broadcast paths" begin
    be = LavaBackend()
    host = reshape(collect(1f0:105f0), 7, 5, 3)
    A = KA.allocate(be, Float32, 7, 5, 3); copyto!(A, host)
    hostb = reshape(collect(1f0:21f0), 7, 1, 3)
    B = KA.allocate(be, Float32, 7, 1, 3); copyto!(B, hostb)

    # flat kernel: every operand the destination's shape
    same = KA.allocate(be, Float32, 7, 5, 3)
    same .= A .* A .+ 1f0
    KA.synchronize(be)
    @test Array(same) ≈ host .^ 2 .+ 1

    # extruded operand: dest dense, one argument broadcast along a dim
    mixed = KA.allocate(be, Float32, 7, 5, 3)
    mixed .= A .+ B .* 2f0
    KA.synchronize(be)
    @test Array(mixed) ≈ host .+ hostb .* 2

    # non-linear operand: a strided view is not linearly indexable
    vout = KA.allocate(be, Float32, 5, 5, 3)
    vout .= view(A, 2:6, :, :) .* 3f0
    KA.synchronize(be)
    @test Array(vout) ≈ host[2:6, :, :] .* 3

    # non-linear *destination*
    dst = KA.allocate(be, Float32, 7, 5, 3); copyto!(dst, host)
    view(dst, 2:6, :, :) .= 7f0
    KA.synchronize(be)
    @test all(Array(dst)[2:6, :, :] .== 7f0)
    @test Array(dst)[1, :, :] == host[1, :, :]

    # permuted operand
    pout = KA.allocate(be, Float32, 5, 7, 3)
    pout .= PermutedDimsArray(A, (2, 1, 3)) .+ 1f0
    KA.synchronize(be)
    @test Array(pout) ≈ permutedims(host, (2, 1, 3)) .+ 1
end

# A Tuple operand must NOT take the flat path.
#
# `flatok` returns true by default for "scalars, refs, functions", and a Tuple
# fell into that default — but a Tuple is a broadcast *container* with its own
# axes, not a scalar. `flat1` reshapes array leaves to vectors of length(dest)
# while leaving the tuple at its own length, so the flattened tree broadcast a
# 30-element operand against a 3-element one:
#
#   DimensionMismatch: a has axes Base.OneTo(30) and b has axes Base.OneTo(3)
#
# This is GPUArrays' own `broadcasting.jl` "Tuple" case, which accounted for 11
# errors in the suite once the device stopped dying earlier in the run.
@testset "broadcast with a Tuple operand" begin
    be = LavaBackend()
    N = 10
    ha = rand(Float32, 3, N)
    hA, hB, hC = rand(Float32, N), rand(Float32, N), rand(Float32, N)
    hout = zeros(Float32, 3, N)
    broadcast!((xx, yy) -> xx + first(yy), hout, ha, (hA, hB, hC))

    out = KA.allocate(be, Float32, 3, N); copyto!(out, zeros(Float32, 3, N))
    arr = KA.allocate(be, Float32, 3, N); copyto!(arr, ha)
    a = KA.allocate(be, Float32, N); copyto!(a, hA)
    b = KA.allocate(be, Float32, N); copyto!(b, hB)
    c = KA.allocate(be, Float32, N); copyto!(c, hC)

    broadcast!((xx, yy) -> xx + first(yy), out, arr, (a, b, c))
    KA.synchronize(be)
    @test Array(out) ≈ hout

    # And the eligibility test itself, so the default can't silently swallow
    # Tuples again.
    @test Mantle.flatok(out, (a, b, c)) === false
end

# A kernel that loops over a constant tuple, `for σ in (Int32(-1), Int32(1))`, inside
# loops over axes and sub-faces. Julia keeps the tuple as a constant global and guards
# it with a comparison of its address against null, which LLVM left as a constant
# expression: `br i1 icmp ne (ptr addrspace(1) addrspacecast (ptr null to ...),
# ptr addrspace(1) @_j_const_1)`. The SPIR-V emitter had no case for an `icmp`
# constant expression and stopped ("Unsupported ConstantExpr opcode: ICmp"). A global's
# address is never null, so the emitter now folds the comparison to a constant. Found in
# the sparse FLIP whirlpool's coarse-cell coefficients, whose loops these are.

using Test, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
include(joinpath(@__DIR__, "testbackend.jl"))

@inline axisunit(a) = (Int32(a == 1), Int32(a == 2), Int32(a == 3))

# For each cell of a coarse grid four times coarser than `fine`: the sum over its six
# sides of the four fine cells across each side, or of the coarse neighbour where the
# fine grid has none (marked by a negative fine value).
function sides(coarse, fine, I, J, K)
    n1, n2, n3 = size(coarse)
    acc = 0f0
    for a in Int32(1):Int32(3)
        e1, e2, e3 = axisunit(a)
        for σ in (Int32(-1), Int32(1))
            In = I + σ * e1; Jn = J + σ * e2; Kn = K + σ * e3
            if Int32(0) <= In < n1 && Int32(0) <= Jn < n2 && Int32(0) <= Kn < n3
                if @inbounds(fine[4In + 1, 4Jn + 1, 4Kn + 1]) >= 0f0
                    for q in Int32(0):Int32(3), p in Int32(0):Int32(3)
                        FI = a == 1 ? (σ > 0 ? 4In : 4I - Int32(1)) : 4I + p
                        FJ = a == 2 ? (σ > 0 ? 4Jn : 4J - Int32(1)) : 4J + (a == 1 ? p : q)
                        FK = a == 3 ? (σ > 0 ? 4Kn : 4K - Int32(1)) : 4K + q
                        acc += @inbounds fine[FI + 1, FJ + 1, FK + 1]
                    end
                else
                    acc += @inbounds coarse[In + 1, Jn + 1, Kn + 1]
                end
            end
        end
    end
    return acc
end

@kernel inbounds=true function sides_kernel!(out, @Const(coarse), @Const(fine))
    i, j, k = @index(Global, NTuple)
    out[i, j, k] = sides(coarse, fine, Int32(i - 1), Int32(j - 1), Int32(k - 1))
end

@testset "a loop over a constant tuple compiles (constant pointer icmp folded)" begin
    nc = (6, 5, 4)
    coarse = Float32[i + 10j + 100k for i in 1:nc[1], j in 1:nc[2], k in 1:nc[3]]
    fine = Float32[(isodd(fld(i - 1, 4) + fld(j - 1, 4)) ? -1f0 : 0.01f0 * (i + j + k)) for i in 1:4nc[1], j in 1:4nc[2], k in 1:4nc[3]]
    want = Float32[sides(coarse, fine, Int32(i - 1), Int32(j - 1), Int32(k - 1)) for i in 1:nc[1], j in 1:nc[2], k in 1:nc[3]]
    out = KA.zeros(TESTBACKEND, Float32, nc...)
    sides_kernel!(TESTBACKEND, (2, 1, 1))(out, Mantle.devicearray(TESTBACKEND, coarse), Mantle.devicearray(TESTBACKEND, fine);
                                           ndrange = nc)
    KA.synchronize(TESTBACKEND)
    @test Array(out) ≈ want
end

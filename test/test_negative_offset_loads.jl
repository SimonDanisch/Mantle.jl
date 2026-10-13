"""
Adjacent sub-dword loads whose address folds a negative constant: the shapes
RADV/ACO (Mesa 26.2) miscompiled until Lava kept the constant out of its reach
(`Lava/src/compiler/spirv/foldedoffset.jl`).

RADV merges `(x[i], x[i+1])` in fp16 into one vector load and folds the
address's constant into it. When that constant is negative, ACO splits the
2-byte-aligned load and the second half reads 4 GiB below the buffer: zeros, or
a lost device. The `-1` of 1-based indexing is that constant whenever the index's
own `+1` is out of the driver's view, and each kernel here hides it differently:

- `2t + 1`, which LLVM writes as `or disjoint`;
- the same in `Int32`, behind the sign extension;
- an index read from memory, forward `(x[i], x[i+1])` and backward
  `(x[i-1], x[i])`;
- a loop counter, whose `+1` is in a phi.

Float32 is the control: its loads are never split. A failure on the others is
the second element of each pair, so every element is compared.
"""

using Test, KernelAbstractions
const KA = KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

"Lane t copies `(x[g], x[g+1])`, `g = 2t + 1 + off`, with `Int` indices."
function pairs64!(out, x, off::Int)
    t = KI.get_global_id().x - 1
    g = 2t + 1 + off
    @inbounds out[t + 1] = (VecElement(x[g]), VecElement(x[g + 1]))
    return
end

"The same with `Int32` indices."
function pairs32!(out, x, off::Int32)
    t = Int32(KI.get_global_id().x - 1)
    g = Int32(2) * t + Int32(1) + off
    @inbounds out[t + 1] = (VecElement(x[g]), VecElement(x[g + Int32(1)]))
    return
end

"`(x[i], x[i+1])` at an index read from memory."
function forward!(out, x, idx)
    t = KI.get_global_id().x
    @inbounds i = idx[t]
    @inbounds out[t] = (VecElement(x[i]), VecElement(x[i + 1]))
    return
end

"`(x[i-1], x[i])` at an index read from memory."
function backward!(out, x, idx)
    t = KI.get_global_id().x
    @inbounds i = idx[t]
    @inbounds out[t] = (VecElement(x[i - 1]), VecElement(x[i]))
    return
end

"`(x[k], x[k+1])` with `k` a 1-based loop counter, one trip per lane."
function looped!(out, x, n::Int)
    t = KI.get_global_id().x
    k = 1
    while k <= n
        @inbounds out[t] = (VecElement(x[k + 2(t - 1)]), VecElement(x[k + 2(t - 1) + 1]))
        k += 2n
    end
    return
end

"Run `kern` over `n` lanes into a fresh pair array and return it on the host."
function runpairs(kern, ::Type{T}, n, args...) where {T}
    out = KA.allocate(TESTBACKEND, NTuple{2,VecElement{T}}, n)
    Mantle.kilaunch!(TESTBACKEND, kern, out, args...; ndrange = n, workgroupsize = 64)
    KA.synchronize(TESTBACKEND)
    return Array(out)
end

pairof(x, i) = (VecElement(x[i]), VecElement(x[i + 1]))

@testset "negative-offset sub-dword loads — $(nameof(typeof(TESTBACKEND)))" begin
    n = 1024
    for T in (Float16, Int16, UInt8, Float32)
        hx = T.(mod.(1:(2n + 9), 97))
        x = Mantle.devicearray(TESTBACKEND, hx)
        for off in (0, 7)
            want = [pairof(hx, 2t + 1 + off) for t in 0:(n - 1)]
            @test runpairs(pairs64!, T, n, x, off) == want
            @test runpairs(pairs32!, T, n, x, Int32(off)) == want
        end
        hidx = [2 + (37t) % (2n) for t in 1:n]
        idx = Mantle.devicearray(TESTBACKEND, hidx)
        @test runpairs(forward!, T, n, x, idx) == [pairof(hx, i) for i in hidx]
        @test runpairs(backward!, T, n, x, idx) == [pairof(hx, i - 1) for i in hidx]
        @test runpairs(looped!, T, n, x, 1) == [pairof(hx, 2t - 1) for t in 1:n]
    end
end

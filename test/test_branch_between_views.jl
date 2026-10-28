# A branch between loads of two views: `flag[i] == 1 ? a[i] : b[i]`, where a
# view is a device array and an offset and `getindex` adds the offset.
#
# LLVM turns the branch into a phi of the two element addresses, stepped back by
# the -1 of 1-based indexing. Lava emitted that step as `OpPtrAccessChain` with
# the constant -1 as its Element, and NVIDIA (RTX 4000 Ada, driver 595) read 0
# through it in every lane; `ifelse` and raw device arrays, which leave no phi
# of pointers, read correctly. A negative constant step is now address
# arithmetic. `Lava/test/spirv/test_psb_negative_step.jl` pins the SPIR-V.

using Test, Mantle, KernelAbstractions
import Adapt
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"Elements `offset + 1 : offset + length` of `data`, indexed from 1."
struct OffsetView{A}
    data::A
    offset::Int32
end
Adapt.@adapt_structure OffsetView
@inline Base.getindex(v::OffsetView, i::Integer) = @inbounds v.data[v.offset + i]

function select_by_branch!(out, flag, a, b, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return
    @inbounds out[i] = flag[i] == Int32(1) ? a[i] : b[i]
    return
end

function select_by_ifelse!(out, flag, a, b, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return
    @inbounds out[i] = ifelse(flag[i] == Int32(1), a[i], b[i])
    return
end

@testset "a branch between loads of two views — $(nameof(typeof(TESTBACKEND)))" begin
    n = 256
    flag = Int32.(isodd.(1:n) .| ((1:n) .% 7 .== 0))
    for T in testeltypes(Int32, Float32, Float64)
        av = T.(1000 .+ (1:n))
        bv = T.(2000 .+ (1:n))
        want = ifelse.(flag .== 1, av, bv)
        a = OffsetView(Mantle.devicearray(TESTBACKEND, vcat(zeros(T, 3), av)), Int32(3))
        b = OffsetView(Mantle.devicearray(TESTBACKEND, vcat(zeros(T, 5), bv)), Int32(5))
        f = Mantle.devicearray(TESTBACKEND, flag)
        for kern in (select_by_branch!, select_by_ifelse!)
            out = KA.allocate(TESTBACKEND, T, n)
            Mantle.kilaunch!(TESTBACKEND, kern, out, f, a, b, Int32(n); ndrange = n, workgroupsize = 64)
            KA.synchronize(TESTBACKEND)
            @test Array(out) == want
        end
    end
end

# A load or store of a VECTOR through a device pointer.
#
# `NTuple{4,VecElement{UInt32}}` is how a kernel asks for one 16-byte access, and
# three ways of spelling it were each wrong before Lava's emitter learned what a
# vector is:
#
#   * `unsafe_load(p, i)`: Julia emits `align 1`, so the emitter decomposes the
#     access, and it sized the decomposition by a type-size helper with no vector
#     case — 4 bytes. One component moved; the other twelve bytes stayed zero.
#   * the same call's `getelementptr <4 x i32>` was stepped by that same size, 4
#     bytes per index instead of 16, so thread `i` read thread `i/4`'s data.
#   * `pointerref(p, i, 16)` worked, but was declared `Aligned 4`: the driver was
#     never told the access could be one 16-byte load.
#
# Found writing a flash-attention kernel that stages K and V with 16-byte copies.
#
# The third was checked in the emitted SPIR-V (`OpLoad ... Aligned 16`). It
# changes no value, only what the driver may assume, so it is the Vulkan
# compiler's business and is not checked here; the copies are.

using Test, Mantle
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

const VPA_V4 = NTuple{4,VecElement{UInt32}}
const VPA_H4 = NTuple{4,VecElement{Float16}}

# `pointer` of a device array is a `Ptr` under one compiler and a `Core.LLVMPtr`
# under another, and the two spell an aligned access differently. These re-point
# and access either, so each kernel below is written once.
@inline vecptr(p::Ptr, ::Type{V}) where {V} = reinterpret(Ptr{V}, p)
@inline vecptr(p::Core.LLVMPtr{T,A}, ::Type{V}) where {T,A,V} = reinterpret(Core.LLVMPtr{V,A}, p)
@inline alignedload(p::Ptr, i, ::Val{N}) where {N} = Core.Intrinsics.pointerref(p, Int(i), N)
@inline alignedload(p::Core.LLVMPtr, i, align::Val) = unsafe_load(p, i, align)
@inline alignedstore!(p::Ptr, v, i, ::Val{N}) where {N} = Core.Intrinsics.pointerset(p, v, Int(i), N)
@inline alignedstore!(p::Core.LLVMPtr, v, i, align::Val) = unsafe_store!(p, v, i, align)

function vpa_indexed_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        p = vecptr(pointer(src), VPA_V4)
        unsafe_store!(vecptr(pointer(dst), VPA_V4), unsafe_load(p, i), i)
    end
    return nothing
end

function vpa_byteaddr_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        off = 16 * (Int(i) - 1)
        p = vecptr(pointer(src) + off, VPA_V4)
        unsafe_store!(vecptr(pointer(dst) + off, VPA_V4), unsafe_load(p))
    end
    return nothing
end

function vpa_aligned_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        v = alignedload(vecptr(pointer(src), VPA_V4), i, Val(16))
        alignedstore!(vecptr(pointer(dst), VPA_V4), v, i, Val(16))
    end
    return nothing
end

# Half components take the 2-byte decomposition, which had the same missing case.
function vpa_half_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        p = vecptr(pointer(src), VPA_H4)
        unsafe_store!(vecptr(pointer(dst), VPA_H4), unsafe_load(p, i), i)
    end
    return nothing
end

# Two halves of a 16-byte load stored swapped: the split is an LLVM
# `shufflevector`, which Lava lowers to `OpVectorShuffle`.
function vpa_split_swap!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        x = alignedload(vecptr(pointer(src), VPA_H4), i, Val(8))
        q = vecptr(pointer(dst), NTuple{2,VecElement{Float16}})
        alignedstore!(q, (x[3], x[4]), 2 * Int(i) - 1, Val(4))
        alignedstore!(q, (x[1], x[2]), 2 * Int(i), Val(4))
    end
    return nothing
end

@testset "vector loads and stores through a device pointer" begin
    backend = TESTBACKEND
    x = Float16.(randn(Float32, 8 * 1024))
    src = Mantle.devicearray(TESTBACKEND, x)
    for (kern, n) in ((vpa_indexed_copy!, 1024), (vpa_byteaddr_copy!, 1024),
                      (vpa_aligned_copy!, 1024), (vpa_half_copy!, 2048))
        dst = Mantle.devicearray(TESTBACKEND, zeros(Float16, length(x)))
        KI.@launch backend ndrange = n workgroupsize = 64 kern(dst, src, Int32(n))
        @test Array(dst) == x
    end

    dst = Mantle.devicearray(TESTBACKEND, zeros(Float16, length(x)))
    KI.@launch backend ndrange = 2048 workgroupsize = 64 vpa_split_swap!(dst, src, Int32(2048))
    @test Array(dst) == vec(reshape(x, 4, :)[[3, 4, 1, 2], :])
end

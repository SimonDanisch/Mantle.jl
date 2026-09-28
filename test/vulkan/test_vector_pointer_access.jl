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

using Test, Lava, Mantle
import KernelInterface as KI

const VPA_V4 = NTuple{4,VecElement{UInt32}}
const VPA_H4 = NTuple{4,VecElement{Float16}}

function vpa_indexed_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        p = reinterpret(Ptr{VPA_V4}, pointer(src))
        unsafe_store!(reinterpret(Ptr{VPA_V4}, pointer(dst)), unsafe_load(p, i), i)
    end
    return nothing
end

function vpa_byteaddr_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        off = 16 * (Int(i) - 1)
        p = reinterpret(Ptr{VPA_V4}, pointer(src) + off)
        unsafe_store!(reinterpret(Ptr{VPA_V4}, pointer(dst) + off), unsafe_load(p))
    end
    return nothing
end

function vpa_aligned_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        v = Core.Intrinsics.pointerref(reinterpret(Ptr{VPA_V4}, pointer(src)), Int(i), 16)
        Core.Intrinsics.pointerset(reinterpret(Ptr{VPA_V4}, pointer(dst)), v, Int(i), 16)
    end
    return nothing
end

# Half components take the 2-byte decomposition, which had the same missing case.
function vpa_half_copy!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        p = reinterpret(Ptr{VPA_H4}, pointer(src))
        unsafe_store!(reinterpret(Ptr{VPA_H4}, pointer(dst)), unsafe_load(p, i), i)
    end
    return nothing
end

# Two halves of a 16-byte load stored swapped: the split is an LLVM
# `shufflevector`, which Lava lowers to `OpVectorShuffle`.
function vpa_split_swap!(dst, src, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        x = Core.Intrinsics.pointerref(reinterpret(Ptr{VPA_H4}, pointer(src)), Int(i), 8)
        q = reinterpret(Ptr{NTuple{2,VecElement{Float16}}}, pointer(dst))
        Core.Intrinsics.pointerset(q, (x[3], x[4]), 2 * Int(i) - 1, 4)
        Core.Intrinsics.pointerset(q, (x[1], x[2]), 2 * Int(i), 4)
    end
    return nothing
end

@testset "vector loads and stores through a device pointer" begin
    backend = LavaBackend()
    x = Float16.(randn(Float32, 8 * 1024))
    src = LavaArray(x)
    for (kern, n) in ((vpa_indexed_copy!, 1024), (vpa_byteaddr_copy!, 1024),
                      (vpa_aligned_copy!, 1024), (vpa_half_copy!, 2048))
        dst = LavaArray(zeros(Float16, length(x)))
        KI.@kernel backend ndrange = n workgroupsize = 64 kern(dst, src, Int32(n))
        @test Array(dst) == x
    end

    dst = LavaArray(zeros(Float16, length(x)))
    KI.@kernel backend ndrange = 2048 workgroupsize = 64 vpa_split_swap!(dst, src, Int32(2048))
    @test Array(dst) == vec(reshape(x, 4, :)[[3, 4, 1, 2], :])

    # And the declaration: a 16-byte-aligned access says so to the driver.
    tt = Tuple{Lava.LavaDeviceArray{Float16,1},Lava.LavaDeviceArray{Float16,1},Int32}
    dis = Lava.disassemble_spirv(Lava.lava_compile_gpu(vpa_aligned_copy!, tt;
                                                       workgroup_size = (64, 1, 1)).spirv_bytes)
    @test occursin(r"OpLoad %v4uint %\w+ Aligned 16", dis)
    @test occursin(r"OpStore %\w+ %\w+ Aligned 16", dis)
end

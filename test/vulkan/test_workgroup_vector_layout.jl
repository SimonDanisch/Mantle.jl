# Workgroup layout sizes and alignments for vector types — a host-only check of
# the SPIR-V emitter (Lava), split out of `test_coopmat_shared.jl`, whose kernels
# are now portable. Proposed to move to Lava's own test suite: it names Lava
# internals and needs no device.
#
# The bug: `wg_compute_type_size`/`wg_compute_type_alignment` had no `VectorType`
# branch at all and fell through to a constant 4. That is right for `<2 x half>`
# by coincidence, so a vec2 `@localmem` worked while a 4-wide one got
# `ArrayStride 4` for an 8-byte element and `spirv-val` refused the module
# ("array with stride 4 not satisfying alignment to 8"). A `<2 x float>` staging
# buffer would have hit it too and nothing in the tree had one. The vec4 kernel in
# `test/test_coopmat_shared.jl` is the device-side check of the fp16 case; this
# covers the float widths no kernel uses yet.

using Test, Lava, LLVM

@testset "workgroup layout sizes and alignments for vectors" begin
    # The rule the fix encodes, checked without a GPU: packed size, and Vulkan's
    # 2x/4x alignment. A vector reported as 4 bytes whatever it is leaves only
    # the 2-wide fp16 case working.
    LLVM.@dispose ctx = LLVM.Context() begin
        h = LLVM.HalfType(); f = LLVM.FloatType()
        for (ty, sz, al) in ((LLVM.VectorType(h, 2),  4,  4),
                             (LLVM.VectorType(h, 4),  8,  8),
                             (LLVM.VectorType(f, 2),  8,  8),
                             (LLVM.VectorType(f, 4), 16, 16))
            @test Lava.wg_compute_type_size(ty) == sz
            @test Lava.wg_compute_type_alignment(ty) == al
        end
    end
end

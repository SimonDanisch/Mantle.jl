# test_workgroup_struct_accesschain.jl
#
# Regression test for a struct array in workgroup (shared) memory: scatter into
# it, synchronize, gather out of it.
#
# Background — this bug has bitten the Vulkan compiler repeatedly.  Recap of the
# failure mode and why an undisciplined fix breaks raytracing:
#
#   * SPIR-V `OpAccessChain` requires struct *member* indices to be
#     `OpConstantInt`.  Array *element* indices may be runtime values.
#     The two index categories share an instruction operand list — the
#     emitter has to decide per-index whether the LLVM operand is a
#     compile-time literal or a runtime value.
#
#   * `ensure_index_i32!(state, val)` is the single helper that converts
#     LLVM index Values to SPIR-V index IDs.  Old behaviour: always call
#     `get_value_id!` and then `OpUConvert` if the LLVM type was wider
#     than i32.  That is correct for runtime array indices but wrong
#     for struct member indices: a literal `i64 1` produced
#     `OpUConvert %uint %ulong_1` — runtime-typed, even though the
#     underlying value is a constant — and `spirv-val` rejects it.
#
#   * Past patch attempts that "lifted the runtime UConvert into a
#     constant" outside `ensure_index_i32!` interacted badly with the
#     RT pipeline, where index conversion happens in different code
#     paths (`OpAccessChain` for ray payload structs, ray query
#     `Get*KHR` returns).  Each fix-and-regress oscillation left one
#     of (compute kernels with @localmem struct arrays, RT pipelines)
#     broken.
#
# This file pinned both constraints. The ray-query half re-ran
# `test_closesthit_via_rayquery.jl` from here; that file runs in the suite on
# its own, so it is not repeated. The compute half is below, checked by value.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# A 32-byte struct: 3xVec3 + Int32 + UInt32 — exactly the shape of
# `ImplicitBVH.BoundingVolume{BBox{Float32},Int32,UInt32}` that AK's
# merge_sort uses with `@localmem`.  Pinning the *struct shape* (not
# the importing package) keeps the test self-contained.
struct BVStruct
    lo::NTuple{3, Float32}
    hi::NTuple{3, Float32}
    leaf_id::Int32
    morton::UInt32
end

# Mirrors the AK merge_sort shared-memory pattern that triggered the
# OpAccessChain regression: scatter into and gather from a per-block
# Workgroup `[N x BVStruct]` buffer with both an index `i + 1` (where
# `i` is `ithread + iblock*N*2` — runtime) and field accesses through
# `gep BVStruct, ptr s_buf, i64 idx, i64 N`.  When `N` is a constant
# (which it always is for struct member access), the SPIR-V emitter
# must produce `OpConstantInt`, not `OpUConvert`.
@kernel function scatter_gather_struct!(@Const(in_arr), out_arr)
    s_buf = @localmem BVStruct (8,)
    i = @index(Local, Linear)
    if i <= 8
        s_buf[i] = in_arr[i]
    end
    @synchronize()
    if i <= 8
        out_arr[i] = s_buf[i]
    end
end

@testset "Workgroup [N x struct] round trip" begin
    # The kernel body involves @localmem of a struct, scatter+gather, and
    # @synchronize — the exact triple AK.merge_sort uses. On Vulkan the
    # original bug failed spirv-val at compile, which the launch reaches.
    #
    # Distinct elements, and an output that starts out different from all of
    # them: the round trip has to move every field of every element.
    bvs = [BVStruct((Float32(k), 0f0, -Float32(k)), (1f0, Float32(2k), 3f0),
                    Int32(k), UInt32(100 + k)) for k in 1:8]
    blank = BVStruct((0f0, 0f0, 0f0), (0f0, 0f0, 0f0), Int32(-1), UInt32(0))
    in_arr = Mantle.devicearray(TESTBACKEND, bvs)
    out_arr = Mantle.devicearray(TESTBACKEND, fill(blank, 8))

    @test_nowarn scatter_gather_struct!(TESTBACKEND, 8)(in_arr, out_arr; ndrange=8)
    KA.synchronize(TESTBACKEND)
    @test Array(out_arr) == bvs  # round-trip through @localmem worked
end

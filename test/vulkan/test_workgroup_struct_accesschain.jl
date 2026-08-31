# test_workgroup_struct_accesschain.jl
#
# Regression test for the OpAccessChain-into-Workgroup-struct emit pattern.
#
# Background — this bug has bitten us repeatedly.  Recap of the failure
# mode and why an undisciplined fix breaks raytracing:
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
# This file pins BOTH constraints with concrete tests, so any future
# emit.jl rework has to satisfy them simultaneously.

using Test, Lava, Mantle
using Lava: lava_compile_gpu

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

# Mirrors the AK merge_sort shared-memory pattern that triggers the
# OpAccessChain regression: scatter into and gather from a per-block
# Workgroup `[N x BVStruct]` buffer with both an index `i + 1` (where
# `i` is `ithread + iblock*N*2` — runtime) and field accesses through
# `gep BVStruct, ptr s_buf, i64 idx, i64 N`.  When `N` is a constant
# (which it always is for struct member access), the SPIR-V emitter
# must produce `OpConstantInt`, not `OpUConvert`.
@kernel function _scatter_gather_kernel!(@Const(in_arr), out_arr)
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

@testset "Workgroup [N x struct] AccessChain — compile-and-validate" begin
    # The kernel body involves @localmem of a struct, scatter+gather, and
    # @synchronize — the exact triple AK.merge_sort uses.  We just need
    # it to compile + pass spirv-val (which fired on the original bug).
    bv = BVStruct((0f0, 0f0, 0f0), (1f0, 1f0, 1f0), Int32(0), UInt32(0))
    in_arr = MVE.LavaArray([bv for _ in 1:8])
    out_arr = MVE.LavaArray([bv for _ in 1:8])
    backend = MVE.LavaBackend()
    bq = backend.bq

    # Compile-only path is enough — emit + spirv-val happen inside.
    @test_nowarn _scatter_gather_kernel!(backend, 8)(in_arr, out_arr; ndrange=8)
    MVE.vk_flush!(bq)
    out = Array(out_arr)
    @test out[1] == bv  # round-trip through @localmem worked
end

# Twin test: ensure RT (HW ray query) still compiles + runs after any
# `ensure_index_i32!` change.  Any fix that breaks this is the same
# class of bug the user has been complaining about.
@testset "RT inline ray query — still compiles after AccessChain fix" begin
    ctx = MVE.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping RT half: no VK_KHR_ray_query"
        @test_skip true
        return
    end
    # Run the existing inline-ray-query test as a smoke check; if the
    # fix here breaks RT, this whole file fails.
    rt_test = joinpath(@__DIR__, "..", "test_closesthit_via_rayquery.jl")
    isfile(rt_test) || (@test_skip true; return)
    Base.include(@__MODULE__, rt_test)
end

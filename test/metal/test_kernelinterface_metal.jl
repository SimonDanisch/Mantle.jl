"""
How the access walk classifies Metal's opaque AIR intrinsics.

A Metal compiler-level table: after inlining, simdgroup-matrix loads and stores,
atomics and the tensor ops are calls to AIR symbols with no ordinary LLVM load or
store for the walker to read, so each symbol's direction is stated
(`intrinsic_usage(::MetalAPI, name)`, `src/metal/ka.jl`). Only the Metal build
defines those methods, so this runs where Metal does. What the walk makes of a
kernel built from a device's own intrinsics is `test/test_access_kernels.jl`, on
every backend.

The rest of what this file used to hold runs on every backend now:
`test/test_kernelinterface.jl` (launch forms, device queries, sub-group
reductions on every lane, shuffle type lists), `test/test_subgroup_shuffle.jl`
(absolute shuffle), `test/test_array_capability_queries.jl` (the array's queries
against `caps`), `test/test_device_caps.jl` (caps cached) and
`test/test_gemv.jl`.
"""

using Test, Mantle

@testset "Metal: opaque AIR access declarations" begin
    @test Mantle.intrinsic_usage(Symbol("air.atomic.global.load.i32")) === Mantle.READ
    @test Mantle.intrinsic_usage(Symbol("air.atomic.local.store.i32")) === Mantle.WRITE
    @test Mantle.intrinsic_usage(Symbol("air.atomic.global.xchg.i32")) === Mantle.ATOMIC
    @test Mantle.intrinsic_usage(
        Symbol("air.init_strided_private_tensor.i32.global")) === Mantle.READ
    @test Mantle.intrinsic_usage(Symbol(
        "__tensorops_impl_matmul2d_op_run_dv_f16_dv_f16_dv_f16")) === Mantle.ATOMIC
end

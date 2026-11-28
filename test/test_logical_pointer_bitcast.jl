# A select between two shared-memory addresses, and width-mismatched loads and
# stores, compute the right answer.
#
# Both pinned one Vulkan compiler bug: Lava emitted an `OpBitcast` on a logical
# pointer, which the Logical addressing model Vulkan mandates forbids. spirv-val
# rejects it outright:
#
#     Instruction may not have a logical pointer operand
#       %99 = OpBitcast %_ptr_Workgroup__arr_float_uint_128 %98
#
# The construct is a clamped ternary over shared memory,
#     l = lid == 1 ? sh[lid] : sh[lid - 1]
# which LLVM lowers to a select between the array base pointer and an element
# pointer. Reconciling those two operand types by bitcasting the element pointer
# UP to the array pointer type is illegal; drilling the aggregate DOWN to its
# first element with OpAccessChain is the legal direction.
#
# ── What this file checks now, and what it no longer can ──────────────────────
#
# It used to capture the modules the compiler emitted (`LAVA_SPIRV_DUMP_DIR`,
# with the Vulkan context's kernel cache cleared so every kernel recompiled)
# and count pointer `OpBitcast`s. That inspection is the Vulkan compiler's
# business, so it is not here. It was also the only thing that ever caught
# the bug: the driver ACCEPTS the invalid module and returns the correct answer,
# and `test_select_width_mismatch.jl` and `test_shared_memory_stress.jl` both
# passed for as long as the bug existed — only a validation-layer device
# (`DebugConfig(validation = true)`), which is not the default, objected. A Tier 1
# `compile_and_disasm` of an equivalent hand-written function does not reproduce
# it either: the illegal bitcast depends on the exact lowering KernelAbstractions'
# `@kernel` wrapper produces.
#
# What is left is the kernels and their answers, which every backend has to get
# right whatever it emits.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "a select between two shared-memory addresses" begin
    M = 128

    @kernel function lpb_clamped_ternary!(out)
        lid = @index(Local)
        sh = @localmem Float32 (128,)
        @inbounds sh[lid] = Float32(lid)
        @synchronize()
        @inbounds begin
            l = lid == 1   ? sh[lid] : sh[lid - 1]
            r = lid == 128 ? sh[lid] : sh[lid + 1]
            out[lid] = (l + sh[lid] + r) / 3f0
        end
    end

    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, M))
    lpb_clamped_ternary!(TESTBACKEND)(out; ndrange = M, workgroupsize = M)
    KA.synchronize(TESTBACKEND)

    v = Float32.(1:M)
    ref = [ (i == 1 ? v[i] : v[i-1]) + v[i] + (i == M ? v[i] : v[i+1]) for i in 1:M ] ./ 3f0
    @test Array(out) ≈ ref
end

# ── The second family: width mismatches on load and store ────────────────────
#
# The `OpSelect` case above was an ADDRESSING mismatch — array-of-T versus T at
# one address — which `OpAccessChain` expresses, so drilling fixed it. These are
# WIDTH mismatches: a 32-bit value through a pointer whose pointee is 64-bit.
# Logical addressing cannot name a differently-typed pointer to the same address
# at all, so the reconciliation has to happen on the VALUE side, and Lava does it
# there: `emit_reconciled_scalar_load!` for loads, and the equal/narrow/wide split
# in `emit_store!` for stores.
#
# The reproducer is an ordinary sliced `setindex!` over GPUArrays' element types.
# The `Complex` types are REQUIRED — the eight real types alone are all clean,
# which is why nothing hand-written found this. Before the fix it emitted two
# invalid `gpu_getindex_kernel` modules while every one of the twelve types still
# produced the CORRECT answer, so the values below did not catch it on Vulkan;
# they are what every backend has to get right.
@testset "sliced load and store over every GPUArrays element type" begin
    types = testeltypes(Int16, Int32, Int64, Float16, Float32, Float64,
                        ComplexF16, ComplexF32, ComplexF64,
                        Complex{Int16}, Complex{Int32}, Complex{Int64})

    for T in types
        xc = zeros(T, (2, 3, 4))
        yc = rand(T, (2, 3))
        x = Mantle.devicearray(TESTBACKEND, copy(xc))
        y = Mantle.devicearray(TESTBACKEND, copy(yc))

        x[:, :, 2] = y                 # the store path
        xc[:, :, 2] = yc
        @test Array(x) == xc

        z = x[:, :, 2]                 # the load path, same packed slots
        @test Array(z) == yc
    end
end

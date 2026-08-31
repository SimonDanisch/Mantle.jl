# A logical pointer may never be OpBitcast.
#
# Vulkan mandates the Logical addressing model, under which `OpBitcast` may
# neither produce nor consume a pointer. spirv-val rejects it outright:
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
# ── Why this test is shaped the way it is ─────────────────────────────────────
#
# Two false starts, both worth stating so nobody repeats them:
#
# 1. A pure runtime assertion does not work. The driver ACCEPTS the invalid
#    module and returns the correct answer. `test_select_width_mismatch.jl` and
#    `test_shared_memory_stress.jl` both passed for as long as the bug existed;
#    only a validation-layer device (`DebugConfig(validation = true)`), which is
#    not the default, ever objected.
#
# 2. A Tier 1 `compile_and_disasm` check on an equivalent hand-written function
#    does not work either — it emits a VALID module. The illegal bitcast depends
#    on the exact lowering KernelAbstractions' `@kernel` wrapper produces, not on
#    the ternary alone. A Tier 1 version of this test passed with the fix
#    reverted, i.e. it asserted nothing.
#
# So: launch the real KA kernel, capture what the compiler actually emitted via
# LAVA_SPIRV_DUMP_DIR, and assert on that. Verified to fail with the fix
# disabled.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

"""
Count `OpBitcast` instructions whose RESULT type is a pointer, per dumped module.

Walks the binary rather than disassembler text so it does not need `spirv-dis`,
and keys on the result type having been declared by `OpTypePointer` rather than
on any name, so a renamed type cannot make it stop noticing.
"""
function lpb_ptr_bitcasts(dir)
    findings = Tuple{String,Int}[]
    for f in filter(f -> endswith(f, ".spv"), readdir(dir; join = true))
        words = reinterpret(UInt32, read(f))
        ptr_types = Set{UInt32}(); bad = 0; i = 6
        while i <= length(words)
            wc = words[i] >> 16; op = words[i] & 0xFFFF
            wc == 0 && break
            op == 32  && push!(ptr_types, words[i+1])            # OpTypePointer
            op == 124 && words[i+1] in ptr_types && (bad += 1)    # OpBitcast -> pointer
            i += Int(wc)
        end
        bad > 0 && push!(findings, (basename(f), bad))
    end
    findings
end

@testset "no OpBitcast on a logical pointer" begin
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

    dumpdir = mktempdir()
    prev = get(ENV, "LAVA_SPIRV_DUMP_DIR", nothing)
    ENV["LAVA_SPIRV_DUMP_DIR"] = dumpdir
    try
        out = MVE.LavaArray(zeros(Float32, M))
        lpb_clamped_ternary!(MVE.LavaBackend())(out; ndrange = M, workgroupsize = M)
        KA.synchronize(MVE.LavaBackend())

        # The answer must still be right — the fix reconciles the select's
        # operands, it does not change what the kernel computes.
        v = Float32.(1:M)
        ref = [ (i == 1 ? v[i] : v[i-1]) + v[i] + (i == M ? v[i] : v[i+1]) for i in 1:M ] ./ 3f0
        @test Array(out) ≈ ref

        spvs = filter(f -> endswith(f, ".spv"), readdir(dumpdir; join = true))
        @test !isempty(spvs)          # a dump we never wrote would assert nothing

        for f in spvs
            words = reinterpret(UInt32, read(f))
            # Walk the instruction stream: OpBitcast = 124. Its result type is the
            # first operand, so flag any whose result type id was declared by
            # OpTypePointer = 32. Done on the binary rather than on disassembler
            # text so it does not depend on spirv-dis being installed.
            ptr_types = Set{UInt32}()
            bad = 0
            i = 6                      # first word past the 5-word header
            while i <= length(words)
                wc = words[i] >> 16
                op = words[i] & 0xFFFF
                wc == 0 && break
                op == 32 && push!(ptr_types, words[i+1])          # OpTypePointer result id
                op == 124 && words[i+1] in ptr_types && (bad += 1) # OpBitcast to a pointer
                i += Int(wc)
            end
            @test bad == 0
        end
    finally
        prev === nothing ? delete!(ENV, "LAVA_SPIRV_DUMP_DIR") : (ENV["LAVA_SPIRV_DUMP_DIR"] = prev)
        rm(dumpdir; recursive = true, force = true)
    end
end

# ── The second family: width mismatches on load and store ────────────────────
#
# The `OpSelect` case above was an ADDRESSING mismatch — array-of-T versus T at
# one address — which `OpAccessChain` expresses, so drilling fixed it. These are
# WIDTH mismatches: a 32-bit value through a pointer whose pointee is 64-bit.
# Logical addressing cannot name a differently-typed pointer to the same address
# at all, so the reconciliation has to happen on the VALUE side, and it does:
# `emit_reconciled_scalar_load!` for loads, and the equal/narrow/wide split in
# `emit_store!` for stores.
#
# The reproducer is an ordinary sliced `setindex!` over GPUArrays' element types.
# The `Complex` types are REQUIRED — the eight real types alone are all clean,
# which is why nothing hand-written found this. Before the fix it emitted two
# invalid `gpu_getindex_kernel` modules while every one of the twelve types still
# produced the CORRECT answer, so a value assertion alone asserts nothing here;
# the module has to be inspected.
@testset "no pointer OpBitcast from a width-mismatched load or store" begin
    types = (Int16, Int32, Int64, Float16, Float32, Float64,
             ComplexF16, ComplexF32, ComplexF64,
             Complex{Int16}, Complex{Int32}, Complex{Int64})

    dumpdir = mktempdir()
    prev = get(ENV, "LAVA_SPIRV_DUMP_DIR", nothing)
    ENV["LAVA_SPIRV_DUMP_DIR"] = dumpdir
    try
        for T in types
            xc = zeros(T, (2, 3, 4))
            yc = rand(T, (2, 3))
            x = MVE.LavaArray(copy(xc))
            y = MVE.LavaArray(copy(yc))

            x[:, :, 2] = y                 # the store path
            xc[:, :, 2] = yc
            @test Array(x) == xc

            z = x[:, :, 2]                 # the load path, same packed slots
            @test Array(z) == yc
        end

        spvs = filter(f -> endswith(f, ".spv"), readdir(dumpdir))
        @test !isempty(spvs)               # a dump we never wrote would assert nothing
        @test isempty(lpb_ptr_bitcasts(dumpdir))
    finally
        prev === nothing ? delete!(ENV, "LAVA_SPIRV_DUMP_DIR") : (ENV["LAVA_SPIRV_DUMP_DIR"] = prev)
        rm(dumpdir; recursive = true, force = true)
    end
end

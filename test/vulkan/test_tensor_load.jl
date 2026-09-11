using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMt = Lava.AcceleratedMatrix

# `OpCooperativeMatrixLoadTensorNV` end to end: a layout is created, given the
# tensor's extent, sliced to the block this workgroup owns, and the matrix is
# filled straight from memory. No shared-memory staging anywhere.
#
# Compiling and validating is NOT the test — the emitter produced valid SPIR-V
# twice while the instruction was still wrong (once with the object operand a
# plain integer, once missing the trailing tensor-operands word). This runs it
# and compares against the source array.
#
# Three things this pins that the extension's operand list does not state, each
# of which cost a spirv-val round trip:
#
#  * `%object` is the matrix's EXISTING value, so it must come from a real
#    cooperative-matrix op — `coopmat_zero` here. Passing a fabricated handle
#    gives "Object <id> type does not match Result Type".
#  * MemoryOperands must be `Aligned` with a literal, not `None`, because the
#    pointer is a `PhysicalStorageBuffer` address
#    (VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708). glslang's own
#    reference emits `None` — its shader loads from a descriptor binding, so
#    copying it verbatim breaks only on the buffer-device-address path.
#  * `TensorAddressingOperands` is then required, not optional; omitting it
#    aborts the decoder with "expected more operands after 8 words".
const TL_N = 64          # tensor extent, and the array is TL_N x TL_N
const TL_M = 16          # matrix tile

@kernel cpu = false function tensorload_kernel!(out, @Const(src))
    l  = Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT))
    l2 = Lava.tensor_setdim(l, (Int32(TL_N), Int32(TL_N)))
    l3 = Lava.tensor_slice(l2, (Int32(0), Int32(0)), (Int32(TL_M), Int32(TL_M)))
    z  = Lava.coopmat_zero(AMt{Float32,TL_M,TL_M,Lava.Accumulator})
    m  = Lava.tensor_load(z, UInt64(pointer(src)), l3)
    Mantle.copyto!(pointer(out), 1, TL_M, m)
end

@kernel cpu = false function tensorphi_kernel!(out, @Const(src), flag::Int32)
    base = Lava.tensor_setdim(
               Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
               (Int32(TL_N), Int32(TL_N)))
    l = flag == 1 ? Lava.tensor_slice(base, (Int32(0), Int32(0)),      (Int32(TL_M), Int32(TL_M))) :
                    Lava.tensor_slice(base, (Int32(0), Int32(TL_M)),   (Int32(TL_M), Int32(TL_M)))
    z = Lava.coopmat_zero(AMt{Float32,TL_M,TL_M,Lava.Accumulator})
    Mantle.copyto!(pointer(out), 1, TL_M, Lava.tensor_load(z, UInt64(pointer(src)), l))
end

@testset "OpCooperativeMatrixLoadTensorNV loads what the layout describes" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        # A cooperative matrix is subgroup-scoped; the launch is exactly one
        # subgroup wide, asked rather than assumed (32 on Ada, 64 on RDNA 3.5).
        WG = Mantle.device_subgroup_size(ctx)
        src = KA.allocate(back, Float32, TL_N, TL_N)
        copyto!(src, Float32.(reshape(1:(TL_N * TL_N), TL_N, TL_N)))
        out = KA.allocate(back, Float32, TL_M, TL_M)
        fill!(out, -1.0f0)

        tensorload_kernel!(back, (Int(WG),))(out, src; ndrange = (Int(WG),))
        KA.synchronize(back)

        o = Array(out)
        s = Array(src)
        @test all(isfinite, o)
        @test !any(==(-1.0f0), o)          # every element was written
        # ORIENTATION, measured and now pinned. The tensor's LAST dimension is
        # the fastest-varying one, so a `(TL_N, TL_N)` layout over a
        # column-major Julia array yields the TRANSPOSE of the leading block.
        #
        # This is the convention a GEMM port has to match, and reading it
        # backwards is the easiest way here to get a plausible wrong answer —
        # every element finite, every element written, all of it in the wrong
        # place. Hence an equality, not an `||` over both orientations: a test
        # that accepts either cannot catch the mistake it exists for.
        blk = s[1:TL_M, 1:TL_M]
        @test o == permutedims(blk)
        @test o != blk                      # and they are genuinely different
        @test o[1, 2] == s[2, 1]            # spelled out on one element
    end
end

# A tensor layout is an opaque handle carried as an `i32`, exactly like a
# cooperative matrix, so anything deriving a SPIR-V type from the LLVM type gets
# `%uint`. A RUNTIME branch selecting between two layouts is the shape that
# exposes it — a GEMM picking a layout per branch would — and before
# `state.tensor_value_types` existed this failed validation with
#
#     OpPhi's result type '%uint' does not match incoming value type
#
# Two separate fixes were needed and the second is the interesting one. Typing
# the phi result is not sufficient: the kernel body sits under the ndrange guard,
# so the guard edge carries a plain `i32 0` and the phi is still ill-typed. The
# cooperative-matrix path substitutes `OpConstantNull` there; a layout has NO
# null value ("OpConstantNull Result Type cannot have a null value") and needs
# `OpUndef` instead, which is correct because the edge is only taken where the
# value is dead.
@testset "a phi over tensor layouts is typed, not %uint" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        WG = Mantle.device_subgroup_size(ctx)
        src = KA.allocate(back, Float32, TL_N, TL_N)
        s = Float32.(reshape(1:(TL_N * TL_N), TL_N, TL_N))
        copyto!(src, s)
        out = KA.allocate(back, Float32, TL_M, TL_M)
        for f in Int32.((1, 2))
            fill!(out, NaN32)
            tensorphi_kernel!(back, (Int(WG),))(out, src, f; ndrange = (Int(WG),))
            KA.synchronize(back)
            o = Array(out); off = f == 1 ? 0 : TL_M
            @test all(o[i, j] == s[off + j, i] for i in 1:TL_M, j in 1:TL_M)
        end
    end
end

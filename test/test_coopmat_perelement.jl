"""
`OpCooperativeMatrixPerElementOpNV` and component-wise `OpFMul`.

Two ways to transform a cooperative matrix without reaching for a component.

`coopmat_perelement` (`VK_NV_cooperative_matrix2`) hands a callback the element's
`(row, col)`, which `coopmat_getcomp` cannot know — the split across the subgroup
is the implementation's business. `coopmat_mul` is plain KHR: `OpFMul` on two
matrices of the same type is defined component-wise, and combined with a
**stride-0** load it applies a per-row factor in one instruction.

Three things here fail differently and are asserted separately.

**That `(row, col)` are the right way round.** The callback multiplies by the row
and adds the column, so a transposed mapping is not symmetric under the test and
cannot pass by accident.

**That the callback's signature survives.** SPIR-V fixes it at
`(u32, u32, T, extras...)` while LLVM is free to rewrite the signature of a
function it can see every caller of. Dead-argument elimination removes an unused
`col`; interprocedural constant propagation removes a `@localmem` pointer that it
proved constant. Both were observed, and `KI.coopmat_keepparam` exists because
of them — so the callback here deliberately ignores `col` and takes a shared
pointer, which is the combination that broke.

That the callback is left inlinable — a `DontInline` callback measured 8.5x — is a
property of the compiled module, so Lava's suite asserts it on the disassembly
(`Lava/test/spirv/test_coopmat_perelement_inline.jl`).
"""

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

const KA = KernelAbstractions
const AMpe = KI.AcceleratedMatrix
const TILEpe = Mantle.GEMM_TILE

# Depends on both indices, and is not symmetric under swapping them.
rowcolmap(row::UInt32, col::UInt32, e::Float32) = e * Float32(row + 1) + Float32(col)

# Ignores `col` (dead-argument elimination) and reads a `@localmem` through a
# pointer LLVM knows is constant (interprocedural constant propagation).
rowscale(row::UInt32, col::UInt32, e::Float32,
         cs::Core.LLVMPtr{Float32,3}, base::Int32) =
    e * unsafe_load(cs, Int(base + row) + 1)

@kernel cpu=false function pe_rowcol!(C, @Const(A))
    @inbounds begin
        m = AMpe{Float32,TILEpe,TILEpe,KI.Accumulator}(pointer(A), 1, TILEpe)
        Mantle.copyto!(pointer(C), 1, TILEpe, KI.coopmat_perelement(rowcolmap, m))
    end
end

@kernel cpu=false function pe_extras!(C, @Const(A), base::Int32)
    @inbounds begin
        cs = @localmem Float32 (2 * TILEpe,)
        tid = @index(Local, Linear)
        tid <= 2 * TILEpe &&
            (cs[tid] = tid <= TILEpe ? Float32(tid) : Float32(100 + tid - TILEpe))
        @synchronize
        m = AMpe{Float32,TILEpe,TILEpe,KI.Accumulator}(pointer(A), 1, TILEpe)
        Mantle.copyto!(pointer(C), 1, TILEpe,
                     KI.coopmat_perelement(rowscale, m, cs.ptr, base))
    end
end

# The same per-row rescale with no per-element instruction at all: a stride-0
# load broadcasts the 16 factors across all 16 columns, and one `OpFMul` applies
# them. Portable — this is the KHR extension.
@kernel cpu=false function fmul_rowscale!(C, @Const(A), base::Int32)
    @inbounds begin
        cs = @localmem Float32 (2 * TILEpe,)
        tid = @index(Local, Linear)
        tid <= 2 * TILEpe &&
            (cs[tid] = tid <= TILEpe ? Float32(tid) : Float32(100 + tid - TILEpe))
        @synchronize
        m = AMpe{Float32,TILEpe,TILEpe,KI.Accumulator}(pointer(A), 1, TILEpe)
        s = AMpe{Float32,TILEpe,TILEpe,KI.Accumulator}(cs, 1 + base, 0, Val(false))
        Mantle.copyto!(pointer(C), 1, TILEpe, KI.coopmat_mul(m, s))
    end
end

@testset "cooperative-matrix per-element and component-wise ops" begin
    back = TESTBACKEND
    # A cooperative matrix is subgroup-scoped, and a workgroup smaller than one
    # subgroup has undefined behaviour — so the launches below are exactly one
    # subgroup wide. This must be asked, not assumed: it is 32 on Ada and 64 on
    # RDNA 3.5, and the hardcoded 32 this replaced was half a subgroup there.
    WGpe = Mantle.caps(TESTBACKEND).coopmatsubgroup
    A = KA.allocate(back, Float32, TILEpe, TILEpe)
    copyto!(A, Float32.(reshape(1:TILEpe^2, TILEpe, TILEpe)))
    a = Array(A)

    # The component-wise multiply on its own (`fmul_rowscale!`, KHR) is portable
    # and is checked on every backend in `test/test_coopmat_add.jl`; here it is
    # only the reference the per-element callback is compared against below.

    if !KI.supports_coopmat_perelement(TESTBACKEND)
        @info "no VK_NV_cooperative_matrix2 per-element operations; skipping"
    else
        @testset "the callback sees the element's own row and column" begin
            C = KA.zeros(back, Float32, TILEpe, TILEpe)
            pe_rowcol!(back, (WGpe,))(C, A; ndrange = (WGpe,))
            KA.synchronize(back)
            # SPIR-V row/column are 0-based and map to the Julia array's row and
            # column index minus one.
            want = [a[i, j] * Float32(i) + Float32(j - 1) for i in 1:TILEpe, j in 1:TILEpe]
            @test Array(C) == want
        end

        @testset "extras reach the callback, and its signature survives" begin
            for base in Int32.((0, TILEpe))
                C = KA.zeros(back, Float32, TILEpe, TILEpe)
                pe_extras!(back, (WGpe,))(C, A, base; ndrange = (WGpe,))
                KA.synchronize(back)
                add = base == 0 ? 0.0f0 : 100.0f0
                want = [a[i, j] * (Float32(i) + add) for i in 1:TILEpe, j in 1:TILEpe]
                @test Array(C) == want
            end
        end

        @testset "the two rescales agree exactly" begin
            C1 = KA.zeros(back, Float32, TILEpe, TILEpe)
            C2 = KA.zeros(back, Float32, TILEpe, TILEpe)
            pe_extras!(back, (WGpe,))(C1, A, Int32(0); ndrange = (WGpe,))
            fmul_rowscale!(back, (WGpe,))(C2, A, Int32(0); ndrange = (WGpe,))
            KA.synchronize(back)
            @test Array(C1) == Array(C2)
        end
    end
end

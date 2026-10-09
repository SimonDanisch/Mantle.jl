# `OpTensorLayoutSetClampValueNV` — what a Constant-clamped load puts out of range.
#
# The default fill is zero and zero is not always the identity you want: filling
# `V`'s padding columns with ONE is what makes a row sum fall out of a matrix
# product in `attn_flash_cm2!`, deleting a reduction per key block.
#
# The trap this pins is that the operand is a 32-bit integer for a matrix of any
# component type, and nothing in the signature says how the bits map to an
# element. Both readings are run: BITS (the operand is the element's bit pattern)
# and NUMERIC (the operand is converted). They disagree by 45 orders of
# magnitude, so a single run cannot be misread — and `tensor_clampbits`, which is
# what callers use, is only correct under one of them.
#
# The unset arm is the negative control: it must fill with ZERO. Without it a
# clamp value that the driver ignored entirely would look identical to one that
# worked, on any test whose expected fill happened to be zero.
using Test, Mantle, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "tensor clamp value" begin
    if !KI.supports_tensor_addressing(TESTBACKEND)
        @info "no coopmat2 tensor addressing on this device — clamp value not exercised"
    else
        E, EA = 16, 10          # the matrix wants 16 rows; the array HAS 10
        @kernel cpu = false unsafe_indices = true function cvload!(out, @Const(A),
                                                                   v::Int32,
                                                                   ::Val{SET}) where {SET}
            l = KI.tensor_layout(Val(2), Val(KI.TENSOR_CLAMP_CONSTANT))
            l = KI.tensor_setdim(l, (Int32(16), Int32(10)))
            l = KI.tensor_setstride(l, (Int32(10), Int32(1)))
            l = SET ? KI.tensor_setclampvalue(l, v) : l
            l = KI.tensor_slice(l, (Int32(0), Int32(0)), (Int32(16), Int32(16)))
            m = KI.tensor_load(
                    KI.coopmat_zero(KI.AcceleratedMatrix{Float32,16,16,KI.Accumulator}),
                    UInt64(pointer(A)), l)
            Mantle.copyto!(pointer(out), 1, 16, m)
        end

        back = TESTBACKEND
        WG = Mantle.caps(back).coopmatsubgroup   # one subgroup, at the coopmat width
        a = Float32.(reshape(1:(EA * E), EA, E))
        A = KA.allocate(back, Float32, EA, E); copyto!(A, a)
        load(v, set) = begin
            o = KA.allocate(back, Float32, E, E); fill!(o, Float32(NaN))
            cvload!(back, WG)(o, A, Int32(v), Val(set); ndrange = WG)
            KA.synchronize(back)
            Array(o)
        end
        # Columns 1:EA are inside the tensor; EA+1:E are the fill.
        real(g) = g[:, 1:EA]
        fill_(g) = g[:, (EA + 1):E]

        unset = load(0, false)
        @test real(unset) ≈ a'                       # the load itself is right
        @test all(==(0.0f0), fill_(unset))           # control: the default IS zero

        bits = load(KI.tensor_clampbits(1.0f0, Float32), true)
        @test real(bits) ≈ a'                        # in-range elements untouched
        @test all(==(1.0f0), fill_(bits))            # BITS reading: fills with one

        # The other reading, run so the first cannot be a coincidence: passing a
        # numeric 1 fills with the smallest subnormal, not with 1.0.
        numeric = load(1, true)
        @test all(==(reinterpret(Float32, UInt32(1))), fill_(numeric))
        @test !any(==(1.0f0), fill_(numeric))

        # A second value, so "it fills with one" is not a special case of some
        # fixed constant the driver happens to use.
        two = load(KI.tensor_clampbits(2.5f0, Float32), true)
        @test all(==(2.5f0), fill_(two))

        # And the fp16 encoding `attn_flash_cm2!` actually passes.
        @test KI.tensor_clampbits(1.0f0, Float16) == Int32(0x3c00)
        @test KI.tensor_clampbits(1.0f0, Float32) == reinterpret(Int32, 0x3f800000)
    end
end

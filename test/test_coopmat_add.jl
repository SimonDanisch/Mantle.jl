"""
`OpFAdd` and `OpFMul` on two cooperative matrices, component-wise —
`coopmat_add` and `coopmat_mul`.

`SPV_KHR_cooperative_matrix` says the ordinary arithmetic instructions act
component-wise on a cooperative matrix. This is that claim for `OpFAdd` and
`OpFMul`, tested rather than assumed, because "the spec says component-wise" is
exactly the kind of statement that has been wrong here before — the stride-0
broadcast load next door is undefined in the specification and works, and a shape
match that ignored the component type answered yes on hardware that only
implements the integer forms.

What the add is for: a GEMM accumulator can start from `bias + residual` instead
of zero. The bias is a stride-0 load (one vector broadcast across the tile) and
the residual is a normal load at the destination's leading dimension; summing
them needs exactly this instruction. Both operands stay wherever the
implementation keeps them — no component is ever named, so unlike
`coopmat_getcomp` there is no `Function`-storage round trip.

What the multiply is for: a per-row rescale with no per-element instruction at
all. A stride-0 load broadcasts the row factors across every column, and one
`OpFMul` applies them — the KHR counterpart of `VK_NV_cooperative_matrix2`'s
per-element callback, which `test/vulkan/test_coopmat_perelement.jl` checks
against it.

Portable: KHR, not `VK_NV_cooperative_matrix2`, so RDNA3's WMMA path gets it too,
and both are `KernelInterface` operations, so the kernels below are written at
the device's own tile rather than at 16.
"""

using Test, KernelAbstractions
using KernelInterface: AcceleratedMatrix, Accumulator, coopmat_add, coopmat_mul
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel cpu=false unsafe_indices=true function addtiles!(out, @Const(x), @Const(y),
                                                         ::Val{T}) where {T}
    a = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(x), 1, T)
    b = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(y), 1, T)
    copyto!(pointer(out), 1, T, coopmat_add(a, b))
end

# The shape this exists for: a stride-0 row broadcast plus a full tile, which is
# `bias .+ residual` before a single muladd has run.
@kernel cpu=false unsafe_indices=true function biasplusres!(out, @Const(bias), @Const(res),
                                                            ::Val{T}) where {T}
    b = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(bias), 1, 0)
    r = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(res), 1, T)
    copyto!(pointer(out), 1, T, coopmat_add(b, r))
end

# The per-row rescale: the `T` factors are read with stride 0 out of `@localmem`,
# at `base`, so this is also the shared-memory form of the broadcast load. The
# table holds `1..T` and then `101..100+T`, so `base = T` selects the second half
# and a load that ignored the offset would come out 100 short on every row.
@kernel cpu=false function fmul_rowscale!(C, @Const(A), base::Int32,
                                          ::Val{T}, ::Val{W}) where {T,W}
    @inbounds begin
        cs = @localmem Float32 (2 * T,)
        tid = @index(Local, Linear)
        for j in tid:W:(2 * T)
            cs[j] = j <= T ? Float32(j) : Float32(100 + j - T)
        end
        @synchronize
        m = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(A), 1, T)
        s = AcceleratedMatrix{Float32,T,T,Accumulator}(cs, 1 + base, 0, Val(false))
        copyto!(pointer(C), 1, T, coopmat_mul(m, s))
    end
end

@testset "coopmat_add: OpFAdd is component-wise" begin
    backend = TESTBACKEND
    c = Mantle.caps(backend)
    if !c.coopmat
        @info "no cooperative matrices on this device; skipping"
    else
        # The device's tile, and one launch of exactly one subgroup at the width a
        # cooperative-matrix kernel runs at.
        T, W = c.tile, c.coopmatsubgroup
        xh = reshape(Float32.(1:(T * T)), T, T)
        yh = reshape(Float32.((T * T):-1:1), T, T)
        x = Mantle.devicearray(TESTBACKEND, xh); y = Mantle.devicearray(TESTBACKEND, yh)
        out = KA.allocate(backend, Float32, T, T); fill!(out, 0.0f0)

        addtiles!(backend, W)(out, x, y, Val(T); ndrange = W)
        KA.synchronize(backend)
        @test Array(out) == xh .+ yh

        # Stride 0 on one operand: every column reads the same vector. This is
        # the bias half, and it is the part the specification does not define.
        bh = Float32.(1:T)
        bias = Mantle.devicearray(TESTBACKEND, bh)
        res = Mantle.devicearray(TESTBACKEND, yh)
        fill!(out, 0.0f0)
        biasplusres!(backend, W)(out, bias, res, Val(T); ndrange = W)
        KA.synchronize(backend)
        @test Array(out) == bh .+ yh        # bh broadcast down every column

        # It is an add, not a fused anything: adding zero is the identity, and
        # adding twice is doubling. Cheap, and it catches an emitter that wired
        # OpFMul to both names.
        zed = KA.allocate(backend, Float32, T, T); fill!(zed, 0.0f0)
        fill!(out, 0.0f0)
        addtiles!(backend, W)(out, x, zed, Val(T); ndrange = W)
        KA.synchronize(backend)
        @test Array(out) == xh
        fill!(out, 0.0f0)
        addtiles!(backend, W)(out, x, x, Val(T); ndrange = W)
        KA.synchronize(backend)
        @test Array(out) == 2 .* xh
    end
end

@testset "coopmat_mul: OpFMul with a stride-0 factor matrix" begin
    backend = TESTBACKEND
    c = Mantle.caps(backend)
    if !c.coopmat
        @info "no cooperative matrices on this device; skipping"
    else
        T, W = c.tile, c.coopmatsubgroup
        A = Mantle.devicearray(TESTBACKEND, Float32.(reshape(1:(T * T), T, T)))
        a = Array(A)
        for base in Int32.((0, T))
            C = KA.zeros(backend, Float32, T, T)
            fmul_rowscale!(backend, (W,))(C, A, base, Val(T), Val(W); ndrange = (W,))
            KA.synchronize(backend)
            add = base == 0 ? 0.0f0 : 100.0f0
            want = [a[i, j] * (Float32(i) + add) for i in 1:T, j in 1:T]
            @test Array(C) == want
        end
    end
end

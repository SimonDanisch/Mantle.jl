"""
`OpFAdd` on two cooperative matrices, component-wise — the counterpart of
`coopmat_mul`.

`SPV_KHR_cooperative_matrix` says the ordinary arithmetic instructions act
component-wise on a cooperative matrix, and `coopmat_mul` already relies on that
for `OpFMul`. This is the same claim for `OpFAdd`, tested rather than assumed,
because "the spec says component-wise" is exactly the kind of statement that has
been wrong here before — the stride-0 broadcast load next door is undefined in
the specification and works, and a shape match that ignored the component type
answered yes on hardware that only implements the integer forms.

What it is for: a GEMM accumulator can start from `bias + residual` instead of
zero. The bias is a stride-0 load (one vector broadcast across the tile) and the
residual is a normal load at the destination's leading dimension; summing them
needs exactly this instruction. Both operands stay wherever the implementation
keeps them — no component is ever named, so unlike `coopmat_getcomp` there is no
`Function`-storage round trip.

Portable: KHR, not `VK_NV_cooperative_matrix2`, so RDNA3's WMMA path gets it too.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const T = MVE.GEMM_TILE

@kernel cpu=false unsafe_indices=true function addtiles!(out, @Const(x), @Const(y))
    a = Lava.AcceleratedMatrix{Float32,T,T,Lava.Accumulator}(pointer(x), 1, T)
    b = Lava.AcceleratedMatrix{Float32,T,T,Lava.Accumulator}(pointer(y), 1, T)
    copyto!(pointer(out), 1, T, Lava.coopmat_add(a, b))
end

# The shape this exists for: a stride-0 row broadcast plus a full tile, which is
# `bias .+ residual` before a single muladd has run.
@kernel cpu=false unsafe_indices=true function biasplusres!(out, @Const(bias), @Const(res))
    b = Lava.AcceleratedMatrix{Float32,T,T,Lava.Accumulator}(pointer(bias), 1, 0)
    r = Lava.AcceleratedMatrix{Float32,T,T,Lava.Accumulator}(pointer(res), 1, T)
    copyto!(pointer(out), 1, T, Lava.coopmat_add(b, r))
end

@testset "coopmat_add: OpFAdd is component-wise" begin
    backend = LavaBackend()
    ctx = MVE.vk_context()
    if !MVE.coopmat_shape(ctx, Float16, T, T, T)
        @info "no cooperative matrices on this device; skipping" ctx.device_name
    else
        xh = reshape(Float32.(1:(T * T)), T, T)
        yh = reshape(Float32.((T * T):-1:1), T, T)
        x = MVE.LavaArray(xh); y = MVE.LavaArray(yh)
        out = KA.allocate(backend, Float32, T, T); fill!(out, 0.0f0)

        addtiles!(backend, 32)(out, x, y; ndrange = 32)
        KA.synchronize(backend)
        @test Array(out) == xh .+ yh

        # Stride 0 on one operand: every column reads the same vector. This is
        # the bias half, and it is the part the specification does not define.
        bh = Float32.(1:T)
        bias = MVE.LavaArray(bh)
        res = MVE.LavaArray(yh)
        fill!(out, 0.0f0)
        biasplusres!(backend, 32)(out, bias, res; ndrange = 32)
        KA.synchronize(backend)
        @test Array(out) == bh .+ yh        # bh broadcast down every column

        # It is an add, not a fused anything: adding zero is the identity, and
        # adding twice is doubling. Cheap, and it catches an emitter that wired
        # OpFMul to both names.
        zed = KA.allocate(backend, Float32, T, T); fill!(zed, 0.0f0)
        fill!(out, 0.0f0)
        addtiles!(backend, 32)(out, x, zed; ndrange = 32)
        KA.synchronize(backend)
        @test Array(out) == xh
        fill!(out, 0.0f0)
        addtiles!(backend, 32)(out, x, x; ndrange = 32)
        KA.synchronize(backend)
        @test Array(out) == 2 .* xh
    end
end

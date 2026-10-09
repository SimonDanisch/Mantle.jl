"""
A cooperative matrix that reaches a phi through a CYCLE still carries the matrix
type.

The emitter carries a matrix as an `i32` handle in LLVM, so anything deriving a
SPIR-V type from the LLVM type has to be told that the value is really a matrix;
`defer_phi!` does that by looking at a phi's incoming values. One pass over them
cannot see a cycle. A loop-carried accumulator becomes a header phi whose latch
edge is another phi that has not been reached yet, and whose entry edge is the
`i32 0` LLVM uses for a null handle — so the header types itself `%uint`, and
the module ends up with

    OpPhi's result type '%uint' does not match incoming value type

which `spirv-val` rejects, before any driver sees it.

Three things have to coincide, which is why it went unnoticed: a loop carrying a
matrix, a **barrier** inside it — the barrier splits the body and the
structurizer adds the latch phi — and a **conditional** matrix value, whose two
handles LLVM if-converts into a phi of its own. That is a flash-attention key
loop reading its K tile from one of two addresses, and it is how this was found.

The pattern is the test's identity: nothing here is about a vendor, and the same
assertion runs everywhere the device has cooperative matrices at all.
"""

using Test, KernelAbstractions
using KernelInterface: AcceleratedMatrix, Accumulator, coopmat_add
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# `pick` is a runtime argument so the branch cannot be folded, and `scratch` is
# written under a barrier so the loop body is split. `acc` is carried across
# both, which is what closes the cycle. The tile is the device's, passed as
# `Val(T)`; `scratch` only has to exist, so its size does not follow it.
@kernel cpu=false unsafe_indices=true function coopmatphicycle!(out, @Const(x), @Const(y),
                                                                n::Int32, pick::Int32,
                                                                ::Val{T}) where {T}
    tid = Int32(@index(Local, Linear)) - Int32(1)
    scratch = @localmem Float32 (16,)
    acc = zero(AcceleratedMatrix{Float32,T,T,Accumulator})
    for i in Int32(0):(n - Int32(1))
        tid < Int32(16) && (scratch[1 + tid] = Float32(i))
        @synchronize
        m = (i & pick) == Int32(0) ?
            AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(x), 1, T) :
            AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(y), 1, T)
        acc = coopmat_add(acc, m)
        @synchronize
    end
    # `scratch` has to be read or the barriers are dead and the split goes away.
    tid == 0 && (out[1] += scratch[1] * 0.0f0)
    copyto!(pointer(out), 1, T, acc)
end

@testset "a cooperative matrix survives a phi cycle" begin
    backend = TESTBACKEND
    c = Mantle.caps(backend)
    if !c.coopmat
        @info "no cooperative matrices on this device; skipping"
    else
        # The device's tile, and one launch of exactly one subgroup at the width a
        # cooperative-matrix kernel runs at.
        T, W = c.tile, c.coopmatsubgroup
        xh = reshape(Float32.(1:(T * T)), T, T)
        yh = reshape(fill(0.5f0, T * T), T, T)
        x = Mantle.devicearray(TESTBACKEND, xh); y = Mantle.devicearray(TESTBACKEND, yh)
        out = KA.allocate(backend, Float32, T, T); fill!(out, 0.0f0)

        n, pick = Int32(4), Int32(1)
        # Compiling at all is most of the claim: an ill-typed `OpPhi` fails
        # `spirv-val` inside this call on the SPIR-V backend, so reaching the
        # assertion below means the module validated.
        coopmatphicycle!(backend, W)(out, x, y, n, pick, Val(T); ndrange = W)
        KA.synchronize(backend)
        # `i & 1`: two even iterations take `x`, two odd take `y`.
        @test Array(out) ≈ 2 .* xh .+ 2 .* yh

        # And the branch is really live — swapping which side it takes changes
        # the answer, so the test cannot pass by the conditional being folded.
        fill!(out, 0.0f0)
        coopmatphicycle!(backend, W)(out, x, y, n, Int32(0), Val(T); ndrange = W)
        KA.synchronize(backend)
        @test Array(out) ≈ 4 .* xh
    end
end

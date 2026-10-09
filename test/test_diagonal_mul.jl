# A Diagonal operand must not be ambiguous with the dense GEMM.
#
# `LinearAlgebra.mul!(C::LavaArray{T,2}, ::AbstractVecOrMat, ::AbstractVecOrMat, α, β)` in
# gemm.jl and GPUArrays' `LinearAlgebra.mul!(::AbstractGPUVecOrMat,
# ::Diagonal{<:Any,<:AbstractGPUArray}, …)` are both applicable when the left
# operand is a Diagonal, and neither is more specific. That is a MethodError,
# not a wrong answer:
#
#   LinearAlgebra.mul!(::LavaArray{Float32,2}, ::Diagonal{Float32,LavaArray{Float32,1}},
#        ::LavaArray{Float32,2}, ::Float32, ::Float32) is ambiguous
#
# Lava's method only became applicable when the GEMM landed, so this covers the
# disambiguation rather than the multiply itself. GPUArrays' behaviour is the one
# to keep: a diagonal operand is a scaling, and routing it through the dense GEMM
# would materialise the zeros.

using Test, LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "Diagonal mul! disambiguation" begin
    n, m = 6, 4
    hd = rand(Float32, n); hB = rand(Float32, n, m); hC = rand(Float32, n, m)
    α, β = 2.0f0, 3.0f0

    D = Diagonal(Mantle.devicearray(TESTBACKEND, copy(hd)))
    B = Mantle.devicearray(TESTBACKEND, copy(hB))
    C = Mantle.devicearray(TESTBACKEND, copy(hC))
    LinearAlgebra.mul!(C, D, B, α, β)
    @test Array(C) ≈ α .* Diagonal(hd) * hB .+ β .* hC

    # β = 0 must ignore C's existing contents rather than scale them.
    C0 = Mantle.devicearray(TESTBACKEND, copy(hC))
    LinearAlgebra.mul!(C0, D, B, 1.0f0, 0.0f0)
    @test Array(C0) ≈ Diagonal(hd) * hB

    # The dense path must be unaffected by the added method.
    A = Mantle.devicearray(TESTBACKEND, rand(Float32, n, n))
    C2 = Mantle.devicearray(TESTBACKEND, zeros(Float32, n, m))
    LinearAlgebra.mul!(C2, A, B, 1.0f0, 0.0f0)
    @test Array(C2) ≈ Array(A) * hB

    # ...and the MIRROR case, matrix * Diagonal. GPUArrays has a second method for
    # Diagonal on the right which is ambiguous against the dense GEMM in exactly
    # the same way. Covering only the left case leaves the right one broken,
    # which GPUArrays' own linalg/diagonal testset catches.
    #
    # Non-square on purpose: D scales COLUMNS here, so a square shape would hide a
    # rows/columns mix-up.
    hE = rand(Float32, m, n); hdn = rand(Float32, n)
    Dn = Diagonal(Mantle.devicearray(TESTBACKEND, copy(hdn)))
    E  = Mantle.devicearray(TESTBACKEND, copy(hE))
    hC3 = rand(Float32, m, n)
    C3 = Mantle.devicearray(TESTBACKEND, copy(hC3))
    LinearAlgebra.mul!(C3, E, Dn, α, β)
    @test Array(C3) ≈ α .* hE * Diagonal(hdn) .+ β .* hC3

    C4 = Mantle.devicearray(TESTBACKEND, copy(hC3))
    LinearAlgebra.mul!(C4, E, Dn, 1.0f0, 0.0f0)
    @test Array(C4) ≈ hE * Diagonal(hdn)

    # ComplexF32 too: it is the other eltype GPUArrays exercises, and a conjugation
    # mistake would only show here.
    hEc = rand(ComplexF32, m, n); hdc = rand(ComplexF32, n)
    Ec = Mantle.devicearray(TESTBACKEND, copy(hEc)); Dc = Diagonal(Mantle.devicearray(TESTBACKEND, copy(hdc)))
    C5 = Mantle.devicearray(TESTBACKEND, zeros(ComplexF32, m, n))
    LinearAlgebra.mul!(C5, Ec, Dc, one(ComplexF32), zero(ComplexF32))
    @test Array(C5) ≈ hEc * Diagonal(hdc)

    # And Diagonal * Diagonal, which neither method above takes and which was
    # ambiguous the same way. A NaN-filled `C` with β = 0 is the case that shows
    # whether the off-diagonal is written or merely scaled.
    for T in (Float32, ComplexF32)
        ha, hb, hC6 = rand(T, n), rand(T, n), rand(T, n, n)
        Da, Db = Diagonal(Mantle.devicearray(TESTBACKEND, copy(ha))), Diagonal(Mantle.devicearray(TESTBACKEND, copy(hb)))
        C6 = Mantle.devicearray(TESTBACKEND, copy(hC6))
        LinearAlgebra.mul!(C6, Da, Db, T(2), T(3))
        @test Array(C6) ≈ 2 .* Diagonal(ha) * Diagonal(hb) .+ 3 .* hC6
        C7 = Mantle.devicearray(TESTBACKEND, fill(T(NaN), n, n))
        LinearAlgebra.mul!(C7, Da, Db, one(T), zero(T))
        @test Array(C7) ≈ Diagonal(ha) * Diagonal(hb)
    end
end

# Diagonal * matrix must not be ambiguous with the dense GEMM.
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

using Test, Lava, LinearAlgebra

@testset "Diagonal mul! disambiguation" begin
    n, m = 6, 4
    hd = rand(Float32, n); hB = rand(Float32, n, m); hC = rand(Float32, n, m)
    α, β = 2.0f0, 3.0f0

    D = Diagonal(MVE.LavaArray(copy(hd)))
    B = MVE.LavaArray(copy(hB))
    C = MVE.LavaArray(copy(hC))
    LinearAlgebra.mul!(C, D, B, α, β)
    @test Array(C) ≈ α .* Diagonal(hd) * hB .+ β .* hC

    # β = 0 must ignore C's existing contents rather than scale them.
    C0 = MVE.LavaArray(copy(hC))
    LinearAlgebra.mul!(C0, D, B, 1.0f0, 0.0f0)
    @test Array(C0) ≈ Diagonal(hd) * hB

    # The dense path must be unaffected by the added method.
    A = MVE.LavaArray(rand(Float32, n, n))
    C2 = MVE.LavaArray(zeros(Float32, n, m))
    LinearAlgebra.mul!(C2, A, B, 1.0f0, 0.0f0)
    @test Array(C2) ≈ Array(A) * hB

    # ...and the MIRROR case, matrix * Diagonal. GPUArrays has a second method for
    # Diagonal on the right which is ambiguous against the dense GEMM in exactly
    # the same way. Only the left case was covered here originally, so the right
    # one shipped broken and GPUArrays' own linalg/diagonal testset caught it.
    #
    # Non-square on purpose: D scales COLUMNS here, so a square shape would hide a
    # rows/columns mix-up.
    hE = rand(Float32, m, n); hdn = rand(Float32, n)
    Dn = Diagonal(MVE.LavaArray(copy(hdn)))
    E  = MVE.LavaArray(copy(hE))
    hC3 = rand(Float32, m, n)
    C3 = MVE.LavaArray(copy(hC3))
    LinearAlgebra.mul!(C3, E, Dn, α, β)
    @test Array(C3) ≈ α .* hE * Diagonal(hdn) .+ β .* hC3

    C4 = MVE.LavaArray(copy(hC3))
    LinearAlgebra.mul!(C4, E, Dn, 1.0f0, 0.0f0)
    @test Array(C4) ≈ hE * Diagonal(hdn)

    # ComplexF32 too: it is the other eltype GPUArrays exercises, and a conjugation
    # mistake would only show here.
    hEc = rand(ComplexF32, m, n); hdc = rand(ComplexF32, n)
    Ec = MVE.LavaArray(copy(hEc)); Dc = Diagonal(MVE.LavaArray(copy(hdc)))
    C5 = MVE.LavaArray(zeros(ComplexF32, m, n))
    LinearAlgebra.mul!(C5, Ec, Dc, one(ComplexF32), zero(ComplexF32))
    @test Array(C5) ≈ hEc * Diagonal(hdc)
end

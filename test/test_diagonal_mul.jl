# Diagonal operands select scaling, including dense-GEMM dispatch intersections.

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

    C0 = Mantle.devicearray(TESTBACKEND, copy(hC))
    LinearAlgebra.mul!(C0, D, B, 1.0f0, 0.0f0)
    @test Array(C0) ≈ Diagonal(hd) * hB

    A = Mantle.devicearray(TESTBACKEND, rand(Float32, n, n))
    C2 = Mantle.devicearray(TESTBACKEND, zeros(Float32, n, m))
    LinearAlgebra.mul!(C2, A, B, 1.0f0, 0.0f0)
    @test Array(C2) ≈ Array(A) * hB

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

    hEc = rand(ComplexF32, m, n); hdc = rand(ComplexF32, n)
    Ec = Mantle.devicearray(TESTBACKEND, copy(hEc)); Dc = Diagonal(Mantle.devicearray(TESTBACKEND, copy(hdc)))
    C5 = Mantle.devicearray(TESTBACKEND, zeros(ComplexF32, m, n))
    LinearAlgebra.mul!(C5, Ec, Dc, one(ComplexF32), zero(ComplexF32))
    @test Array(C5) ≈ hEc * Diagonal(hdc)

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

@testset "wrapped diagonal products preserve shape and conjugation" begin
    for T in (Float32,ComplexF32), wrap in (identity,transpose,adjoint), side in (:left,:right)
        raw = rand(T,6,4)
        hmat = wrap(raw)
        mat = wrap(Mantle.devicearray(TESTBACKEND,raw))
        hd = rand(T,size(mat,side === :left ? 1 : 2))
        D = Diagonal(Mantle.devicearray(TESTBACKEND,hd))
        A,B = side === :left ? (D,mat) : (mat,D)
        hA,hB = side === :left ? (Diagonal(hd),hmat) : (hmat,Diagonal(hd))
        seed = rand(T,size(mat))
        C = Mantle.devicearray(TESTBACKEND,seed)
        @test mul!(C,A,B,T(2),T(3)) === C
        @test Array(C) ≈ T(2).*(hA*hB) .+ T(3).*seed
        bad = Mantle.devicearray(TESTBACKEND,zeros(T,size(mat,1)+1,size(mat,2)))
        @test_throws DimensionMismatch mul!(bad,A,B,T(2),T(3))
    end
end

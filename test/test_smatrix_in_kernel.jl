# StaticArrays math inside a kernel, against exact answers and invariants:
# products of in-kernel `SMatrix`es, columns read at runtime indices, a Jacobi
# eigensolver on an `MMatrix`, and zeroing loops.
#
# Lava dropped a step off a pointer to an element of a private array: an
# `SMatrix` the compiler keeps in memory is read through a pointer to one
# element and `getelementptr double, ptr %elem, i64 k`, and the `k` was lost, so
# every row read row 1. StaticArrays' `mul_loop` (any product from 5x5 up) reads
# both operands that way, so `Q * Diagonal(λ) * Q'` of a 6x6 orthogonal `Q` gave
# diag [20, 20, 0, 0, 0, 0] for `λ = 20`; `h[r, k]` over `r` with a runtime `k`
# returned `h[1, k]` for every `r`, from 3x3 up; the 3x3 eigensolver's columns,
# written at indices chosen by a comparison, came out wrong; and Jacobi rotations
# on an `MMatrix` from 6x6 up did not converge (its two zero stores at runtime
# indices were blamed).
# spirv-val accepted all of it. `Lava/test/spirv/test_function_array_steps.jl`
# pins the SPIR-V.
#
# A zeroing loop whose start is a loop variable, which LLVM makes a `memset` of a
# runtime length, did not compile at all; `Lava/test/test_dynamic_memset.jl`
# pins the lowering.
#
# The matrices are chosen so that every product is exact in floating point (the
# orthogonal `Q` has entries 0, ±1/2 and ±1, the others small integers), and the
# expected values are computed in rationals.

using Test, Mantle, KernelAbstractions, LinearAlgebra
using StaticArrays: SMatrix, SVector, MMatrix
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"`Q diag(λ) Qᵀ`, `(Q diag(λ)) Qᵀ`, `Q Qᵀ Q` and `A B` of N×N matrices read from `inp`."
function smatrix_products!(out, inp, ::Val{N}) where {N}
    KI.get_global_id().x == 1 || return
    T = eltype(inp)
    L = N * N
    Q = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[j]), Val(L)))
    λ = SVector{N,T}(ntuple(j -> @inbounds(inp[L + j]), Val(N)))
    A = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[L + N + j]), Val(L)))
    B = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[2L + N + j]), Val(L)))
    P1 = Q * Diagonal(λ) * Q'
    P2 = (Q * Diagonal(λ)) * Q'
    P3 = Q * Q' * Q
    P4 = A * B
    for j in 1:L
        @inbounds out[j] = P1[j]
        @inbounds out[L + j] = P2[j]
        @inbounds out[2L + j] = P3[j]
        @inbounds out[3L + j] = P4[j]
    end
    return
end

"Column `k` of an N×N matrix as `ntuple(r -> h[r, k])`, `k` the lane."
function smatrix_columns!(out, inp, ::Val{N}) where {N}
    k = KI.get_global_id().x
    k <= N || return
    T = eltype(inp)
    h = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[j]), Val(N * N)))
    @inbounds out[k] = SVector{N,T}(ntuple(r -> h[r, k], Val(N)))
    return
end

"`(A B)(A B)ᵀ`: a product of two in-kernel temporaries."
function smatrix_temporaries!(out, inp, ::Val{N}) where {N}
    KI.get_global_id().x == 1 || return
    T = eltype(inp)
    L = N * N
    A = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[j]), Val(L)))
    B = SMatrix{N,N,T}(ntuple(j -> @inbounds(inp[L + j]), Val(L)))
    AB = A * B
    P = AB * AB'
    for j in 1:L
        @inbounds out[j] = P[j]
    end
    return
end

"""
The eigenvector matrix of the closed-form 3x3 eigensolver: columns `a` and `b`
written at `k` and `l`, swapped when the gaps `λ` gives say so (indices chosen
by a comparison), and the middle column as their cross product.
"""
function columns_by_comparison!(out, λs, cols)
    t = KI.get_global_id().x
    t <= length(λs) || return
    T = eltype(eltype(λs))
    @inbounds λ, a, b = λs[t], cols[2t - 1], cols[2t]
    k, l = 1, 3
    if λ[3] - λ[2] > λ[2] - λ[1]
        k, l = l, k
    end
    E = MMatrix{3,3,T}(undef)
    for r in 1:3
        E[r, k] = a[r]
        E[r, l] = b[r]
    end
    c = cross(SVector(E[1, 3], E[2, 3], E[3, 3]), SVector(E[1, 1], E[2, 1], E[3, 1]))
    for r in 1:3
        E[r, 2] = c[r]
    end
    @inbounds out[t] = SMatrix{3,3,T}(E)
    return
end

"The position `(p, q)`, `p < q`, of the off-diagonal entry of largest magnitude."
@inline function largest_offdiagonal(M::MMatrix{N,N,T}) where {N,T}
    best = -one(T)
    p = q = Int32(1)
    for i in Int32(1):Int32(N), j in i+Int32(1):Int32(N)
        if abs(M[i, j]) > best
            best = abs(M[i, j])
            p, q = i, j
        end
    end
    return p, q
end

"One Jacobi rotation of `M` zeroing `M[p, q]` and `M[q, p]` (two stores at runtime indices), accumulated into `E`."
@inline function jacobi_rotate!(M::MMatrix{N,N,T}, E::MMatrix{N,N,T}, p::Int32, q::Int32) where {N,T}
    tau = (M[q, q] - M[p, p]) / (2 * M[p, q])
    t = tau >= 0 ? 1 / (tau + sqrt(1 + tau * tau)) : -1 / (-tau + sqrt(1 + tau * tau))
    c = 1 / sqrt(1 + t * t)
    s = t * c
    for i in Int32(1):Int32(N)
        if i != p && i != q
            Mip, Miq = M[i, p], M[i, q]
            M[i, p] = c * Mip - s * Miq
            M[p, i] = M[i, p]
            M[i, q] = s * Mip + c * Miq
            M[q, i] = M[i, q]
        end
        Eip, Eiq = E[i, p], E[i, q]
        E[i, p] = c * Eip - s * Eiq
        E[i, q] = s * Eip + c * Eiq
    end
    Mpp, Mqq, Mpq = M[p, p], M[q, q], M[p, q]
    M[p, p] = c * c * Mpp + s * s * Mqq - 2 * c * s * Mpq
    M[q, q] = s * s * Mpp + c * c * Mqq + 2 * c * s * Mpq
    M[p, q] = 0
    M[q, p] = 0
    return
end

"Eigenvalues and eigenvectors of symmetric matrices by cyclic Jacobi rotations."
function jacobi_eigen!(vals, vecs, mats, ::Val{N}) where {N}
    t = KI.get_global_id().x
    t <= length(mats) || return
    T = eltype(eltype(mats))
    @inbounds M = MMatrix{N,N,T}(mats[t])
    E = MMatrix{N,N,T}(I)
    k = 0
    while k < 1000
        p, q = largest_offdiagonal(M)
        abs(M[p, q]) > T(1e-13) || break
        jacobi_rotate!(M, E, p, q)
        k += 1
    end
    @inbounds vals[t] = SVector{N,T}(ntuple(i -> M[i, i], Val(N)))
    @inbounds vecs[t] = SMatrix{N,N,T}(E)
    return
end

"""
`I`, made the way a Householder sequence makes it: ones on the diagonal, zeros
above, then zeros below column by column from a start that is the loop variable.
LLVM makes the last of these loops a `memset` of a runtime length.
"""
function identity_by_zeroing!(out, inp, ::Val{N}) where {N}
    KI.get_global_id().x == 1 || return
    T = eltype(inp)
    M = MMatrix{N,N,T}(ntuple(j -> @inbounds(inp[j]), Val(N * N)))
    n = Int32(N)
    for j in Int32(1):n
        for i in Int32(1):j-Int32(1)
            M[i, j] = zero(T)
        end
        M[j, j] = one(T)
    end
    for k in n-Int32(1):Int32(-1):Int32(1)
        for r in k+Int32(1):n
            M[r, k] = zero(T)
        end
    end
    for j in 1:N*N
        @inbounds out[j] = M[j]
    end
    return
end

"""
An exactly orthogonal N×N matrix: `I - v vᵀ / 2` with `v` four ones and then
zeros from 4x4 up, a signed cyclic permutation below.
"""
function orthogonal(N)
    N < 4 && return [j == mod1(i + 1, N) ? (iseven(i) ? -1 // 1 : 1 // 1) : 0 // 1 for i in 1:N, j in 1:N]
    v = [i <= 4 ? 1 // 1 : 0 // 1 for i in 1:N]
    return Matrix{Rational{Int}}(I, N, N) - v * v' / 2
end

@testset "SMatrix math in a kernel — $(nameof(typeof(TESTBACKEND)))" begin
    @testset "$T N=$N" for T in testeltypes(Float32, Float64), N in (3, 4, 5, 6, 8, 9, 12)
        L = N * N
        Q = orthogonal(N)
        λ = [k % 3 == 0 ? 20 // 1 : k // 1 for k in 1:N]
        A = [((3i + 5j) % 7 - 3) // 1 for i in 1:N, j in 1:N]
        B = [((2i + 7j) % 5 - 2) // 1 for i in 1:N, j in 1:N]
        inp = Mantle.devicearray(TESTBACKEND, T.(vcat(vec(Q), λ, vec(A), vec(B))))
        out = Mantle.devicearray(TESTBACKEND, zeros(T, 4L))
        Mantle.kilaunch!(TESTBACKEND, smatrix_products!, out, inp, Val(N); ndrange = 1, workgroupsize = 1)
        KA.synchronize(TESTBACKEND)
        got = reshape(Array(out), N, N, 4)
        E = Q * Diagonal(λ) * Q'
        @test got[:, :, 1] == T.(E)
        @test got[:, :, 2] == T.(E)
        @test got[:, :, 3] == T.(Q)
        @test got[:, :, 4] == T.(A * B)

        # The case that was reported: 20·I recomposed from its eigenvectors.
        inp20 = Mantle.devicearray(TESTBACKEND, T.(vcat(vec(Q), fill(20, N), vec(A), vec(B))))
        Mantle.kilaunch!(TESTBACKEND, smatrix_products!, out, inp20, Val(N); ndrange = 1, workgroupsize = 1)
        KA.synchronize(TESTBACKEND)
        @test reshape(Array(out)[1:L], N, N) == T(20) * I

        H = T.(reshape(1:L, N, N))
        cols = KA.allocate(TESTBACKEND, SVector{N,T}, N)
        Mantle.kilaunch!(TESTBACKEND, smatrix_columns!, cols, Mantle.devicearray(TESTBACKEND, vec(H)), Val(N);
                         ndrange = N, workgroupsize = N)
        KA.synchronize(TESTBACKEND)
        @test reduce(hcat, Array(cols)) == H

        AB = A * B
        tmps = Mantle.devicearray(TESTBACKEND, zeros(T, L))
        Mantle.kilaunch!(TESTBACKEND, smatrix_temporaries!, tmps,
                         Mantle.devicearray(TESTBACKEND, T.(vcat(vec(A), vec(B)))), Val(N); ndrange = 1, workgroupsize = 1)
        KA.synchronize(TESTBACKEND)
        @test reshape(Array(tmps), N, N) == T.(AB * AB')

        # Every entry starts as 7, so a zero store that is lost shows.
        id = Mantle.devicearray(TESTBACKEND, zeros(T, L))
        Mantle.kilaunch!(TESTBACKEND, identity_by_zeroing!, id, Mantle.devicearray(TESTBACKEND, fill(T(7), L)), Val(N);
                         ndrange = 1, workgroupsize = 1)
        KA.synchronize(TESTBACKEND)
        @test reshape(Array(id), N, N) == I
    end
end

@testset "columns at indices chosen by a comparison — $(nameof(typeof(TESTBACKEND)))" begin
    for T in testeltypes(Float32, Float64)
        λs, cols, want = SVector{3,T}[], SVector{3,T}[], SMatrix{3,3,T,9}[]
        for (s, λ) in enumerate(((1, 2, 9), (1, 8, 9), (0, 5, 6), (-3, -2, 4), (2, 3, 4)))
            a = SVector{3,T}(s, 2s, -1)
            b = SVector{3,T}(-2, s, 3)
            k, l = λ[3] - λ[2] > λ[2] - λ[1] ? (3, 1) : (1, 3)
            E = zeros(T, 3, 3)
            E[:, k] = a
            E[:, l] = b
            E[:, 2] = cross(E[:, 3], E[:, 1])
            push!(λs, SVector{3,T}(λ))
            append!(cols, (a, b))
            push!(want, SMatrix{3,3,T}(E))
        end
        out = KA.allocate(TESTBACKEND, SMatrix{3,3,T,9}, length(λs))
        Mantle.kilaunch!(TESTBACKEND, columns_by_comparison!, out, Mantle.devicearray(TESTBACKEND, λs),
                         Mantle.devicearray(TESTBACKEND, cols); ndrange = length(λs), workgroupsize = 32)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == want
    end
end

@testset "Jacobi eigensolver on MMatrix — $(nameof(typeof(TESTBACKEND)))" begin
    for T in testeltypes(Float64), N in (3, 4, 6, 9, 12)
        mats = [SMatrix{N,N,T}(begin
                    A = T[(3i + 5j + 7s) % 11 - 5 for i in 1:N, j in 1:N]
                    A + A'
                end) for s in 1:8]
        vals = KA.allocate(TESTBACKEND, SVector{N,T}, length(mats))
        vecs = KA.allocate(TESTBACKEND, SMatrix{N,N,T,N * N}, length(mats))
        Mantle.kilaunch!(TESTBACKEND, jacobi_eigen!, vals, vecs, Mantle.devicearray(TESTBACKEND, mats), Val(N);
                         ndrange = length(mats), workgroupsize = 8)
        KA.synchronize(TESTBACKEND)
        for (M, λ, V) in zip(mats, Array(vals), Array(vecs))
            @test isapprox(V' * V, I; atol = 1e-10)
            @test isapprox(M * V, V * Diagonal(λ); atol = 1e-9 * opnorm(M))
            @test sum(λ) ≈ tr(M) atol = 1e-9 * opnorm(M)
        end
    end
end

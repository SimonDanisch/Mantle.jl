# `mul!` for LavaArray: the entry point into core's GEMM (`src/array/gemm.jl`).
#
# Only what names this backend's array is here. The kernels, their tilings and the
# choice between them are core's; `mul!` is Base's function, so the method that
# routes a `LavaArray` into them belongs to the backend that owns the array type,
# and `gemmstrides` for it describes the one array whose density this tree vouches
# for. The split-K scratch is cached on the context because Lava's pool can
# reclaim a dropped buffer before a queued dispatch has retired; see
# `splitscratch` in core for the portable form.

gemmstrides(A::LavaArray{T,2}) where {T} = (A, 1, 1, size(A, 1))
gemmstrides(A::LavaArray{T,1}) where {T} = (A, 1, 1, length(A))

# A reshape is free only when the thing underneath is dense; `LavaArray` is the
# only parent for which that holds unconditionally.
gemmstrides(A::Base.ReshapedArray{T,2,<:LavaArray}) where {T} =
    (parent(A), 1, 1, size(A, 1))

"""
    mul!(C, A, B, α, β) -> C

`C = A*B*α + C*β` on the device.

Dispatch is on the *destination* rather than on the operands, because an operand
can be wrapped to any depth — an exported ATen graph routinely hands us a reshape
of a permuted array — and no union of wrapper types catches all of those. If `C`
lives on the device then the multiply happens on the device, whatever the
operands look like; `gemmstrides` unwraps them and only genuinely non-strided
ones are copied.

Uses cooperative matrices when the operands are dense fp16 into an fp32 result,
the extents suit the tile, and the device implements it. Failing that it uses the
staged scalar GEMM, and failing *that* the per-element one; see `staged_gemm_ok`
for which, and the header of this file for what the three are. That choice is an
implementation detail: the result is the same either way, to fp32 reassociation.

The `T<:Number` bound is required, not decoration. These kernels accumulate
in the destination's element type — `muladd(T(A[…]), T(B[…]), acc)` — and scale
by `T(α)`, both of which assume `T` can be constructed from the operand and
scalar types. An element type that is merely `+`- and `*`-able does not satisfy
that: a matrix of a two-field struct entered as `mul!(C, A, B, true, false)` and
threw `Duo{Float16}(true)`, and removing only the scalar conversion just moved
the failure into the kernel at `T(B[…])`. Outside the bound,
`LinearAlgebra.mul!` reaches `GPUArrays.generic_matmatmul!`, which multiplies
with the types' own `*` and promotes the accumulator, which is the correct thing
for those and is already a GPU kernel. Widening this signature takes that away.
"""
function LinearAlgebra.mul!(C::LavaArray{T,2}, A::AbstractVecOrMat,
                            B::AbstractVecOrMat, α::Number, β::Number) where {T<:Number}
    M, K = size(A, 1), size(A, 2)
    N = size(B, 2)
    size(B, 1) == K && size(C) == (M, N) ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    # `KA.get_backend(C)`, NOT `LavaBackend()`. An unpinned backend resolves
    # its queue through `vk_context()`, so on a second device this dispatches on
    # whichever context happens to be global — the work lands on the wrong GPU
    # and the buffer's own device never sees it. `get_backend` derives the
    # context from the array's buffer, which has always carried it.
    backend = KernelAbstractions.get_backend(C)

    # …and `vk_context(C)` for the same reason, which the line above got right and
    # this one did not. A no-argument `coopmat_gemm_available()` asks whichever
    # context is global, so on a second device it answered for the FIRST one:
    # lavapipe (8-lane subgroups, no pinning to 32) was told it had tensor cores
    # because the NVIDIA card does, took the cooperative-matrix path, and returned
    # the right answer only because 32 lanes of redundant subgroups happen to
    # agree. The two-device probe found it; a capability query is exactly as
    # device-specific as the handle caches are.
    if T === Float32 && eltype(A) === Float16 && eltype(B) === Float16 &&
       isone(α) && iszero(β) && A isa LavaArray && B isa LavaArray &&
       K % GEMM_TILE == 0 && M % GEMM_TILE == 0 && N % GEMM_TILE == 0 &&
       coopmatgemm(C)
        return coopmat_gemm!(C, A, B, M, N, K)
    end

    gemmlaunch!(C, A, B, M, N, K, T(α), T(β))
end

function splitscratch(backend::LavaBackend, C, M::Int, N::Int, splitk::Int)
    n = M * N * splitk
    ctx = vk_context(backend)
    caches = ctx.caches
    buf = caches.gemm_split_scratch
    if buf === nothing || length(buf::LavaArray{Float32,1})::Int < n
        # Retain, don't drop: dispatches already recorded point into the old
        # buffer and its finalizer would pull it out from under them. Growth is
        # geometric and stops once the largest product has been seen.
        buf === nothing || push!(caches.gemm_split_retired, buf)
        buf = LavaArray{Float32}(undef, n + n ÷ 2; bq = ctx.default_bq)
        caches.gemm_split_scratch = buf
    end
    GPUArrays.derive(Float32, buf::LavaArray{Float32,1}, (M, N, splitk), 0)
end

# A matrix-vector product is just the N == 1 case. `C` carries strides like the
# operands do, so its rank never reaches the kernel and one kernel serves both.
function LinearAlgebra.mul!(C::LavaArray{T,1}, A::AbstractVecOrMat,
                            B::AbstractVector, α::Number, β::Number) where {T<:Number}
    M, K = size(A, 1), size(A, 2)
    length(B) == K && length(C) == M ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    gemmlaunch!(C, A, B, M, 1, K, T(α), T(β))
end

# Disambiguation: Diagonal * matrix.
#
# `mul!(C::LavaArray{T,2}, ::AbstractVecOrMat, ::AbstractVecOrMat, α, β)` above and
# GPUArrays' `mul!(::AbstractGPUVecOrMat, ::Diagonal{<:Any,<:AbstractGPUArray}, …)`
# are both applicable to `mul!(::LavaArray{Float32,2}, ::Diagonal{Float32,LavaArray{Float32,1}}, ::LavaArray{Float32,2}, α, β)`
# and neither is more specific, so the call is an ambiguity error rather than a
# dispatch to either. Lava's method became applicable when the GEMM landed.
#
# GPUArrays' is the one that should win: a diagonal operand is a scaling, and
# routing it through the dense GEMM would materialise the zeros and do O(n)
# times the work. This mirrors its implementation rather than `invoke`-ing it,
# because the signature to invoke through is unwieldy and would silently rot if
# GPUArrays retyped it.
#
# The `T<:Number` bound has to match the general `mul!` above. These two are
# disambiguated only by being narrower in the operand positions, so if the
# general one is narrower in `C`'s element type and these are not, neither method
# wins and every diagonal multiply raises an ambiguity instead of running.
# `test_diagonal_mul.jl` is what catches that.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            α::Number, β::Number) where {T<:Number, S}
    dd = D.diag
    d = length(dd)
    m, n = size(B, 1), size(B, 2)
    m′, n′ = size(C, 1), size(C, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) ||
        throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    @. C = α * dd * B + β * C
    C
end

# Disambiguation: matrix * Diagonal — the mirror of the case above.
#
# Same collision, other operand: GPUArrays has a second method
# `mul!(::AbstractGPUVecOrMat, ::Union{AbstractGPUArray,Adjoint,Transpose}, ::Diagonal{<:Any,<:AbstractGPUArray}, α, β)`
# which is equally ambiguous against Lava's dense GEMM, so both sides need a
# method. GPUArrays' own linalg/diagonal testset covers each, for Float32 and
# ComplexF32.
#
# `D` scales COLUMNS here (A[:,j] * d[j]), so the broadcast transposes `dd`.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            A::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number, β::Number) where {T<:Number, S}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(C, 1), size(C, 2)
    n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
    (m, n) == (m′, n′) ||
        throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    # `ddT` MUST be hoisted out of the `@.`: inside it, `transpose(dd)` becomes
    # `transpose.(dd)`, which broadcasts transpose over each scalar (a no-op) and
    # leaves a length-n column, scaling ROWS instead of columns. GPUArrays hoists
    # it for the same reason.
    ddT = transpose(dd)
    @. C = α * A * ddT + β * C
    C
end

# Disambiguation: Diagonal * Diagonal, the third of the set. Both operands are
# Diagonals, which neither method above takes, and GPUArrays'
# `mul!(::AbstractGPUArray, ::Diagonal, ::Diagonal, α, β)` is as ambiguous
# against the dense GEMM as the other two were. Its body, for the same reason
# theirs is mirrored: a product of two diagonals touches only the diagonal.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number, β::Number) where {T<:Number}
    dc = view(C, LinearAlgebra.diagind(C))
    da, db = A.diag, B.diag
    d = length(dc)
    length(da) == d || throw(DimensionMismatch("right hand side has $(length(da)) rows but output is $d by $d"))
    length(db) == d || throw(DimensionMismatch("left hand side has $(length(db)) rows but output is $d by $d"))
    # `C` may be uninitialised, so a zero `β` fills rather than scales: `0 * NaN`
    # is `NaN`.
    iszero(β) ? fill!(C, zero(T)) : rmul!(C, β)
    @. dc += α * da * db
    return C
end

"""Does this device implement the tile `mul!` wants?"""
# Cooperative-matrix operations are subgroup-scoped, so a device whose subgroup
# is not 32 lanes gets a different number of subgroups per workgroup than this
# kernel assumes, and `lane ÷ 32` stops naming a real subgroup. The arithmetic
# still comes out right for the lanes that run — on a wave64 device exactly half
# the output tile is written, bit-exact, and the other half stays zero, which is
# a silently wrong answer rather than a failure.
#
# Two ways to have a 32-lane subgroup: the device is natively wave32, or it lets
# the pipeline pin its subgroup size, which `get_compute_pipeline` does for any
# coopmat module. Failing both, `mul!` falls through to `gemmlaunch!`, which is
# correct at any width — slower, but not wrong. Making the kernel itself
# wave-size agnostic would mean retuning GEMM_WORKGROUP and the block factors
# together, and those were measured at 32.
function coopmat_gemm_available(ctx::VkContext)
    coopmat_shape(ctx, Float16, GEMM_TILE, GEMM_TILE, GEMM_TILE) &&
        (device_subgroup_size(ctx) == GEMM_SUBGROUP ||
         can_require_subgroup_size(ctx, GEMM_SUBGROUP))
end

# LavaArray multiplication routes to core GEMM and its per-program workspace.

"""
    mul!(C, A, B, α, β) -> C

Compute `C = A*B*α + C*β` on C's device. Operands without a `gemmstrides`
projection are densified first. Dense fp16 `LavaArray` inputs into fp32 with
`α == 1`, `β == 0` and extents divisible by `GEMM_TILE` use cooperative
matrices; the rest use the scalar kernel. Non-`Number` eltypes use GPUArrays.
"""
function LinearAlgebra.mul!(C::LavaArray{T,2}, A::AbstractVecOrMat,
                            B::AbstractVecOrMat, α::Number, β::Number) where {T<:Number}
    M, K = size(A, 1), size(A, 2)
    N = size(B, 2)
    size(B, 1) == K && size(C) == (M, N) ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    # Both queue and capability selection follow the output allocation.
    backend = KernelAbstractions.get_backend(C)

    if T === Float32 && eltype(A) === Float16 && eltype(B) === Float16 &&
       isone(α) && iszero(β) && A isa LavaArray && B isa LavaArray &&
       K % GEMM_TILE == 0 && M % GEMM_TILE == 0 && N % GEMM_TILE == 0 &&
       coopmatgemm(C)
        return coopmat_gemm!(C, A, B, M, N, K)
    end

    gemmlaunch!(C, A, B, M, N, K, T(α), T(β))
end


# Matrix-vector product: the N == 1 case of the same strided kernel.
function LinearAlgebra.mul!(C::LavaArray{T,1}, A::AbstractVecOrMat,
                            B::AbstractVector, α::Number, β::Number) where {T<:Number}
    M, K = size(A, 1), size(A, 2)
    length(B) == K && length(C) == M ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    gemmlaunch!(C, A, B, M, 1, K, T(α), T(β))
end

# Disambiguate against the dense methods above; GPUArrays handles Diagonal
# operands, including scaling and dimension checks.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            α::Number, β::Number) where {T<:Number, S}
    return GPUArrays.generic_diagonal_mul!(C, D, B, α, β)
end

function LinearAlgebra.mul!(C::LavaArray{T,2},
                            A::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number, β::Number) where {T<:Number, S}
    return GPUArrays.generic_diagonal_mul!(C, A, D, α, β)
end

function LinearAlgebra.mul!(C::LavaArray{T,2},
                            A::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number, β::Number) where {T<:Number}
    return GPUArrays.generic_diagonal_mul!(C, A, B, α, β)
end

"""
    coopmat_gemm_available(ctx) -> Bool

Whether `ctx` runs the cooperative-matrix GEMM: the Float16 `GEMM_TILE` shape
and a 32-lane subgroup, native or pinned by `get_compute_pipeline`. The kernel
indexes subgroups as `lane ÷ 32`; on an unpinned wave64 device it writes half
the output tile and leaves the rest zero without error. Otherwise `mul!` uses
the scalar `gemmlaunch!`, which is correct at any subgroup width.
"""
function coopmat_gemm_available(ctx::VkContext)
    coopmat_shape(ctx, Float16, GEMM_TILE, GEMM_TILE, GEMM_TILE) &&
        (device_subgroup_size(ctx) == GEMM_SUBGROUP ||
         can_require_subgroup_size(ctx, GEMM_SUBGROUP))
end

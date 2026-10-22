# LavaArray mul! routes to core GEMM; split-K scratch stays device-owned until retirement.

"""
    mul!(C, A, B, α, β) -> C

Compute C = A*B*α + C*β on C's device. Stride projection supports wrapped
operands; non-strided inputs are densified. Eligible dense fp16 inputs into
fp32 output use cooperative matrices; other numeric shapes use core scalar
kernels. Non-Number elements retain GPUArrays' generic arithmetic.
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

# Resolve the dense-GEMM/Diagonal intersection; GPUArrays owns scaling,
# dimension checks and diagonal/off-diagonal initialization.
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

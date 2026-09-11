# Regression: the same NVIDIA stride-product miscompile as
# test_repeat_inner_3d.jl, reached through *multi-index* `a[i, j, k, …]` rather
# than through `a[::CartesianIndex]`.
#
# `linear_index` was Horner-factored to dodge the miscompile, but only the
# CartesianIndex and linear overrides routed through it. A multi-index read had
# no `@lava_device_override`, so it fell back to Base's generic
# `_to_linear_index` → `_sub2ind`, which emits exactly the naive expansion
#   i1 + d1*(i2-1) + d1*d2*(i3-1) + d1*d2*d3*(i4-1)
# that the driver evaluates wrong when it feeds a PhysicalStorageBuffer offset.
# The SPIR-V is valid; the reads land at bogus addresses and return garbage.
#
# It only misfires when the indices are *computed*: `a[1,2,2,1]` constant-folds
# and is correct, while the same indices arriving from a delinearised loop index
# are not — which is why it hid until GPUArrays' `vectorized_getindex`
# (host/indexing.jl:84, `getindex_generated`) hit it. That silently truncated
# every strided slice: on a (3,2,2,1) array, `a[:, :, 2:2, :]` returned
# [7,8,9,0,0,0] instead of [7,8,9,10,11,12].
#
# Fixed in `array/ka_backend.jl` by adding multi-index getindex/setindex!
# overrides that route through the Horner `linear_index`.

using Test, Lava, Mantle
using KernelAbstractions
const KA = KernelAbstractions

@testset "multi-index a[i,j,k,…] — NVIDIA stride-product miscompile" begin
    @testset "strided slices materialise correctly" begin
        h4 = reshape(Float32.(1:12), 3, 2, 2, 1)
        g4 = Mantle.LavaArray(h4)
        @test Array(g4[:, :, 2:2, :]) == h4[:, :, 2:2, :]
        @test Array(g4[:, :, 2, :]) == h4[:, :, 2, :]
        @test Array(g4[:, :, 2, 1]) == h4[:, :, 2, 1]
        @test Array(g4[:, :, 1:1, :]) == h4[:, :, 1:1, :]
        @test Array(copy(view(g4, :, :, 2:2, :))) == h4[:, :, 2:2, :]

        h3 = reshape(Float32.(1:24), 2, 3, 4)
        g3 = Mantle.LavaArray(h3)
        @test Array(g3[:, :, 2]) == h3[:, :, 2]
        @test Array(g3[:, 2:2, :]) == h3[:, 2:2, :]
        @test Array(g3[:, :, 2:3]) == h3[:, :, 2:3]

        h5 = reshape(Float32.(1:24), 2, 2, 3, 2, 1)
        g5 = Mantle.LavaArray(h5)
        @test Array(g5[:, :, :, 1, :]) == h5[:, :, :, 1, :]
        @test Array(g5[:, :, 2, :, :]) == h5[:, :, 2, :, :]
    end

    # The direct form: indices computed inside the kernel, which is what the
    # driver miscompiled. Literal indices constant-fold and never reproduced it.
    @testset "computed multi-index reads" begin
        @kernel function readcomputed!(out, src, idims)
            i = @index(Global, Linear)
            is = @inbounds CartesianIndices(idims)[i]
            @inbounds out[i] = src[is[1], is[2], 2, 1]
        end
        h = reshape(Float32.(1:12), 3, 2, 2, 1)
        g = Mantle.LavaArray(h)
        out = KA.allocate(Mantle.defaultbackend(), Float32, 3, 2, 1, 1)
        fill!(out, -1.0f0)
        readcomputed!(Mantle.defaultbackend())(out, g, (3, 2, 1, 1); ndrange=size(out))
        KA.synchronize(Mantle.defaultbackend())
        @test vec(Array(out)) == vec(h[:, :, 2:2, :])
    end

    @testset "computed multi-index writes" begin
        @kernel function writecomputed!(dst, idims)
            i = @index(Global, Linear)
            is = @inbounds CartesianIndices(idims)[i]
            @inbounds dst[is[1], is[2], is[3], is[4]] = Float32(i)
        end
        dims = (3, 2, 2, 1)
        dst = KA.allocate(Mantle.defaultbackend(), Float32, dims...)
        fill!(dst, -1.0f0)
        writecomputed!(Mantle.defaultbackend())(dst, dims; ndrange=dims)
        KA.synchronize(Mantle.defaultbackend())
        @test vec(Array(dst)) == Float32.(1:prod(dims))
    end
end

# Trailing singleton indices: MORE indices than the array has dimensions.
#
# Julia allows `v[i, 1]` on a Vector and `A[i, j, 1]` on a Matrix, and generic
# kernels lean on it instead of specialising on ndims. GPUArrays'
# `gpu_kron_kernel!` indexes `a[i, j]` whatever `a` is, so `kron(vec(x), y)`
# hands it a 1-D array — and `linear_index(::NTuple{N}, ::CartesianIndex{M})`
# only had methods for M == N, so the kernel failed to compile:
#
#   Cannot compile gpu_kron_kernel!(…, ::LavaDeviceArray{T,1}, ::LavaDeviceArray{T,2})
#   → getindex @ ka_backend.jl → Problem: method lookup failure
#
# Nothing about it was type-specific; every eltype failed identically.
@testset "trailing singleton indices (more indices than dims)" begin
    be = Mantle.defaultbackend()

    @testset "kron(vec, matrix) — $T" for T in (Int16, Float32, ComplexF32)
        ha = rand(T, 16, 32)
        hb = rand(T, 64, 8)
        a = Mantle.LavaArray(ha)
        b = Mantle.LavaArray(hb)
        for op in (identity, transpose, adjoint)
            got = Array(kron(vec(a), op(b)))
            ref = kron(vec(ha), op(hb))
            @test size(got) == size(ref)
            # Float kron reassociates on the GPU, so compare approximately for
            # the float eltypes and exactly for the integer one.
            @test T <: Integer ? got == ref : got ≈ ref
        end
    end

    @testset "reading a 1-D device array with two indices" begin
        n = 8
        src = Mantle.LavaArray(Float32.(1:n))
        dst = Mantle.LavaArray(zeros(Float32, n))
        @kernel function trailing_one!(dst, @Const(src))
            i = @index(Global, Linear)
            @inbounds dst[i] = src[i, 1]      # trailing singleton
        end
        trailing_one!(be)(dst, src; ndrange = n)
        KA.synchronize(be)
        @test Array(dst) == Float32.(1:n)
    end
end

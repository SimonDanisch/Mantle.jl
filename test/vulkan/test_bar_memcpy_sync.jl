using Test, Lava, Mantle
using KernelAbstractions
using GPUArrays: @allowscalar
import AcceleratedKernels as AK

# Regression: `copy_buffer!` BAR fast-path ([runtime/memory.jl:678-697]) used to
# call `wait_for_write(managed)` and memcpy without flushing the active batch.
# Since a buffer's stamp is only written at submit time, any
# kernel that was recorded but not yet submitted was invisible to the wait —
# the memcpy would read pre-kernel memory (which RADV zeros on alloc).
#
# The symptom: AK.reduce / sum / min / max / extrema on small non-Float32
# LavaArrays (n ≲ 512, scratch ≤ 64 bytes → BAR path) silently returned 0.
# That cascaded into Raycore's `build_blas` boundscheck (`extrema(perm) == (0,0)`
# → `BoundsError` on tiny triangle meshes), and into any user reduction.
#
# The fix flushes the default_bq's active batch before the BAR memcpy.

@testset "BAR-memory copy_buffer! sync (regression)" begin

@testset "integer/Float64 full reduction returns correct result for all n" begin
    # Each n covers the BAR path for small scratch sizes (blocks*2 elements,
    # scratch = 16 bytes at n=2, growing). The staging path (n ≳ 512) is the
    # baseline that already worked.
    for T in (Int32, Int64, UInt32, UInt64, Float64)
        for n in (2, 3, 4, 8, 16, 63, 64, 65, 127, 128, 129, 256, 500, 1000)
            a = LavaArray(collect(T, 1:n))
            expected = T(n) * T(n + 1) ÷ T(2)
            T <: AbstractFloat && (expected = (n * (n + 1)) / 2)
            @test sum(a) == expected
            @test minimum(a) == T(1)
            @test maximum(a) == T(n)
        end
    end
end

@testset "Float32 full reduction (fast path) unaffected" begin
    for n in (2, 3, 64, 500, 1000)
        a = LavaArray(collect(Float32, 1:n))
        @test sum(a) == Float32(n * (n + 1) / 2)
    end
end

@testset "extrema returns correct range (used by getindex boundscheck)" begin
    for n in (2, 4, 16, 100)
        a = LavaArray(collect(Int64, 1:n))
        @test extrema(a) == (1, n)
    end
end

@testset "LavaArray{T}[perm::LavaArray{Int}] — tiny sizes" begin
    # This is the exact pattern `build_blas` uses after `AK.sortperm`.
    for n in (2, 3, 4, 8)
        a    = LavaArray(UInt32[10i for i in 1:n])
        perm = LavaArray(collect(n:-1:1))
        @test Array(a[perm]) == UInt32[10i for i in n:-1:1]
    end
end

@testset "scalar readback immediately after kernel write (no explicit sync)" begin
    # Direct repro of the underlying bug: kernel writes to a BAR buffer, user
    # reads the result without calling KernelAbstractions.synchronize. Before
    # the fix, the read would return 0 (the pre-kernel value of the
    # zero-initialised BAR region).
    @kernel function write_one_kernel!(dst, val)
        i = @index(Global, Linear)
        dst[i] = val
    end
    for T in (Int64, UInt32, Float64)
        dst = KernelAbstractions.allocate(LavaBackend(), T, 2)
        fill!(dst, zero(T))
        k = write_one_kernel!(LavaBackend())
        k(dst, T(42); ndrange=2)
        # NOTE: deliberately no explicit synchronize — copy_buffer! must see
        # the pending write and flush.
        @test @allowscalar(dst[1]) == T(42)
        @test @allowscalar(dst[2]) == T(42)
    end
end

@testset "AK.reduce on 2-element Int64 LavaArray" begin
    # Minimal end-to-end check that mirrors the Raycore call site.
    a = LavaArray(Int64[1, 2])
    @test AK.reduce(+, a; init=0) == 3
    @test AK.mapreduce(identity, +, a; init=0) == 3
end

end  # @testset

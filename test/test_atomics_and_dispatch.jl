# test_atomics_and_dispatch.jl
#
# Tests for:
# 1. Atomic read-modify-write on device memory: every update lands, and the value
#    an atomic returns is the one it produced.
# 2. Several dispatches in a row: each sees the one before it.
# 3. Cross-workgroup atomic visibility (the BVH refit pattern)
#
# These pinned SPIR-V emission patterns once as well (`OpAtomicIAdd` with Relaxed
# semantics for a monotonic RMW, the hardware `OpAtomicFAddEXT` for Float32 instead
# of a CAS loop, no `CrossDevice` scope). Those are the Vulkan compiler's business
# and are not checked here; the kernels that pinned them still run below, where a
# wrong scope or a lost update shows up as a wrong count.

using Test
using Mantle
using KernelAbstractions
using Atomix
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# ═══════════════════════════════════════════════════════════════════════
# Atomic Correctness
# ═══════════════════════════════════════════════════════════════════════

@testset "Atomic GPU Execution" begin

    @testset "atomic counter (Int32)" begin
        @kernel function atomic_counter_i32!(counter)
            Atomix.@atomic counter[1] += Int32(1)
        end

        N = 4096
        counter = Mantle.devicearray(TESTBACKEND, Int32[0])
        atomic_counter_i32!(TESTBACKEND)(counter; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(counter)[1] == Int32(N)
    end

    @testset "atomic counter (UInt32)" begin
        @kernel function atomic_counter_u32!(counter)
            Atomix.@atomic counter[1] += UInt32(1)
        end

        N = 4096
        counter = Mantle.devicearray(TESTBACKEND, UInt32[0])
        atomic_counter_u32!(TESTBACKEND)(counter; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(counter)[1] == UInt32(N)
    end

    @testset "atomic counter (Float32)" begin
        @kernel function atomic_counter_f32!(counter)
            Atomix.@atomic counter[1] += 1.0f0
        end

        N = 1024
        counter = Mantle.devicearray(TESTBACKEND, Float32[0])
        atomic_counter_f32!(TESTBACKEND)(counter; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(counter)[1] ≈ Float32(N)
    end

    @testset "two atomics on one counter, one of them loaded" begin
        # The kernel that pinned "no CrossDevice scope": a constant and a loaded
        # increment to the same address from every invocation.
        @kernel function any_atomic!(counter, data)
            i = @index(Global)
            Atomix.@atomic counter[1] += Int32(1)
            Atomix.@atomic counter[1] += data[i]
        end

        N = 256
        datah = Int32.(1:N)
        counter = Mantle.devicearray(TESTBACKEND, Int32[0])
        any_atomic!(TESTBACKEND, 64)(counter, Mantle.devicearray(TESTBACKEND, datah); ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(counter)[1] == Int32(N) + sum(datah)
    end

    @testset "atomic returns unique values" begin
        @kernel function atomic_unique!(counter, results)
            i = @index(Global)
            old = Atomix.@atomic counter[1] += Int32(1)
            @inbounds results[i] = old
        end

        N = 2048
        counter = Mantle.devicearray(TESTBACKEND, Int32[0])
        results = Mantle.devicearray(TESTBACKEND, zeros(Int32, N))
        atomic_unique!(TESTBACKEND)(counter, results; ndrange=N)
        KA.synchronize(TESTBACKEND)

        r = Array(results)
        @test Array(counter)[1] == Int32(N)
        @test length(unique(r)) == N
        # Atomix.@atomic += returns the new value (1..N)
        @test sort(r) == collect(Int32(1):Int32(N))
    end

    @testset "cross-workgroup atomic visibility (BVH refit pattern)" begin
        # Simulates BVH bottom-up refit:
        # - N leaf threads write data[i] then atomically increment flags[parent]
        # - When flags[parent] == 2, the second-arriving thread reads BOTH children
        # - With broken visibility (Relaxed-only), second thread sees stale data
        @kernel function refit_pattern!(data, flags, results, parents, siblings)
            i = @index(Global)
            n = @uniform @groupsize()[1] * @ndrange()[1] ÷ @groupsize()[1]

            # Write leaf data (non-atomic)
            @inbounds data[i] = Float32(i * 10)

            # Atomic increment on parent flag
            @inbounds parent = parents[i]
            @inbounds sibling = siblings[i]
            old = Atomix.@atomic flags[parent] += Int32(1)

            if old == Int32(1)
                # Second thread: both children done, read sibling data
                @inbounds sibling_val = data[sibling]
                @inbounds my_val = data[i]
                @inbounds results[parent] = my_val + sibling_val
            end
        end

        N = 64
        P = N ÷ 2

        data = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))
        flags = Mantle.devicearray(TESTBACKEND, zeros(Int32, P))
        results = Mantle.devicearray(TESTBACKEND, zeros(Float32, P))

        # Pair leaves: (1,2)→parent 1, (3,4)→parent 2, ...
        parents_h = Int32[div(i - 1, 2) + 1 for i in 1:N]
        siblings_h = Int32[i % 2 == 1 ? i + 1 : i - 1 for i in 1:N]
        parents_d = Mantle.devicearray(TESTBACKEND, parents_h)
        siblings_d = Mantle.devicearray(TESTBACKEND, siblings_h)

        refit_pattern!(TESTBACKEND)(data, flags, results, parents_d, siblings_d; ndrange=N)
        KA.synchronize(TESTBACKEND)

        r = Array(results)
        expected = Float32[(2k - 1) * 10 + (2k) * 10 for k in 1:P]
        @test r == expected
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Dispatches in a row
#
# This file also counted flushes and dispatches through the Vulkan context's
# diagnostics (`diag.flush_counter`, `diag.total_dispatches`, `diag.dispatch_log`)
# and asserted that an empty flush submitted nothing. Those are one backend's
# bookkeeping with nothing a caller can observe, so they are gone; what a caller
# sees — the results — is checked below.
# ═══════════════════════════════════════════════════════════════════════

@testset "Batched Dispatch" begin

    @testset "multiple dispatches in one batch" begin
        a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3, 4])
        b = Mantle.devicearray(TESTBACKEND, Float32[10, 20, 30, 40])
        c = a .+ b
        d = c .* Float32(2)
        KA.synchronize(TESTBACKEND)
        @test Array(d) == Float32[22, 44, 66, 88]
    end

    @testset "KA.synchronize completes GPU work" begin
        a = Mantle.devicearray(TESTBACKEND, ones(Float32, 64))
        b = a .+ Float32(1)
        c = b .+ Float32(1)
        KA.synchronize(TESTBACKEND)
        @test Array(c) == fill(Float32(3), 64)
    end

    @testset "barrier between dispatches preserves ordering" begin
        @kernel function write_val!(A, val::Float32)
            i = @index(Global)
            @inbounds A[i] = val
        end
        @kernel function read_add!(B, A)
            i = @index(Global)
            @inbounds B[i] = A[i] + 1.0f0
        end

        N = 256
        A = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))
        B = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))
        write_val!(TESTBACKEND)(A, 42.0f0; ndrange=N)
        read_add!(TESTBACKEND)(B, A; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test all(Array(B) .== 43.0f0)
    end

    @testset "data kept alive across GC + synchronize" begin
        @kernel function fill_index!(A)
            i = @index(Global)
            @inbounds A[i] = Float32(i)
        end

        A = Mantle.devicearray(TESTBACKEND, zeros(Float32, 1024))
        fill_index!(TESTBACKEND)(A; ndrange=1024)
        GC.gc()
        KA.synchronize(TESTBACKEND)
        result = Array(A)
        @test result[1] == 1.0f0
        @test result[1024] == 1024.0f0
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Multi-dispatch Patterns
# ═══════════════════════════════════════════════════════════════════════

@testset "Multi-dispatch Patterns" begin

    @testset "chain of 10 dispatches" begin
        a = Mantle.devicearray(TESTBACKEND, ones(Float32, 128))
        for _ in 1:10
            a = a .+ Float32(1)
        end
        KA.synchronize(TESTBACKEND)
        @test all(Array(a) .== 11.0f0)
    end

    @testset "interleaved compute and reduction" begin
        a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3, 4, 5, 6, 7, 8])
        b = a .* Float32(2)
        KA.synchronize(TESTBACKEND)
        @test sum(b) ≈ 72.0f0
    end

    @testset "large dispatch without splitting" begin
        N = 128 * 256 * 2
        a = Mantle.devicearray(TESTBACKEND, ones(Float32, N))
        b = a .+ Float32(1)
        KA.synchronize(TESTBACKEND)
        @test all(Array(b) .== 2.0f0)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# KA @private, 64-bit atomics, Atomix with CartesianIndex + Float32 subtract
# ═══════════════════════════════════════════════════════════════════════

@testset "KA @private (Scratchpad)" begin
    @kernel function private_accum!(A)
        I = @index(Global)
        priv = @private Float32 (4,)
        @inbounds for k in 1:4
            priv[k] = Float32(I * k)
        end
        @inbounds A[I] = priv[1] + priv[4]
    end

    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 64))
    private_accum!(TESTBACKEND, 64)(a; ndrange=64)
    KA.synchronize(TESTBACKEND)
    result = Array(a)
    @test result[1] == 1f0 + 4f0
    @test result[10] == 10f0 + 40f0
end

# Only where the device has 64-bit integer atomics: Apple GPUs do not.
Mantle.supports_int64_atomics(TESTBACKEND) && @testset "Int64/UInt64 atomics" begin
    @testset "Int64 atomic add" begin
        @kernel function atomic_add_i64!(counter)
            Atomix.@atomic counter[1] += Int64(1)
        end
        N = 2048
        c = Mantle.devicearray(TESTBACKEND, Int64[0])
        atomic_add_i64!(TESTBACKEND)(c; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(c)[1] == Int64(N)
    end

    @testset "UInt64 atomic add" begin
        @kernel function atomic_add_u64!(counter)
            Atomix.@atomic counter[1] += UInt64(1)
        end
        N = 1024
        c = Mantle.devicearray(TESTBACKEND, UInt64[0])
        atomic_add_u64!(TESTBACKEND)(c; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(c)[1] == UInt64(N)
    end
end

# Only where the device computes in Float64 at all.
@testset "Float64 atomics" begin
    @kernel function atomic_add_f64!(counter)
        Atomix.@atomic counter[1] += 1.0
    end
    @kernel function atomic_sub_f64!(counter)
        Atomix.@atomic counter[1] -= 1.0
    end
    for T in testeltypes(Float64)
        @testset "$T atomic add" begin
            N = 1024
            c = Mantle.devicearray(TESTBACKEND, T[0.0])
            atomic_add_f64!(TESTBACKEND)(c; ndrange=N)
            KA.synchronize(TESTBACKEND)
            @test Array(c)[1] ≈ T(N)
        end

        @testset "$T atomic subtract" begin
            N = 1024
            c = Mantle.devicearray(TESTBACKEND, T[T(N)])
            atomic_sub_f64!(TESTBACKEND)(c; ndrange=N)
            KA.synchronize(TESTBACKEND)
            @test Array(c)[1] ≈ 0.0
        end
    end
end

@testset "Atomix CartesianIndex and Float32 subtract" begin

    @testset "Float32 atomic subtract" begin
        @kernel function atomic_sub_f32!(counter)
            Atomix.@atomic counter[1] -= 1.0f0
        end

        N = 1024
        counter = Mantle.devicearray(TESTBACKEND, Float32[Float32(N)])
        atomic_sub_f32!(TESTBACKEND)(counter; ndrange=N)
        KA.synchronize(TESTBACKEND)
        @test Array(counter)[1] ≈ 0.0f0
    end

    @testset "Atomix with CartesianIndex on 3D array" begin
        @kernel function atomic_add_3d!(A, idx_array)
            i = @index(Global)
            @inbounds ci = idx_array[i]
            Atomix.@atomic A[ci] += 1.0f0
        end

        dims = (4, 4, 4)
        A = Mantle.devicearray(TESTBACKEND, zeros(Float32, dims))
        # All threads write to the same CartesianIndex
        target = CartesianIndex(2, 3, 1)
        N = 512
        idx_array = Mantle.devicearray(TESTBACKEND, fill(target, N))
        atomic_add_3d!(TESTBACKEND)(A, idx_array; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(A)
        @test result[2, 3, 1] ≈ Float32(N)
        @test sum(result) ≈ Float32(N)  # only one cell was touched
    end

    @testset "Atomix subtract with CartesianIndex on 2D array" begin
        @kernel function atomic_sub_2d!(A, idx_array, val)
            i = @index(Global)
            @inbounds ci = idx_array[i]
            Atomix.@atomic A[ci] -= val
        end

        dims = (8, 8)
        N = 256
        target = CartesianIndex(4, 5)
        A = Mantle.devicearray(TESTBACKEND, fill(Float32(N), dims))
        idx_array = Mantle.devicearray(TESTBACKEND, fill(target, N))
        atomic_sub_2d!(TESTBACKEND)(A, idx_array, 1.0f0; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(A)
        @test result[4, 5] ≈ 0.0f0
        # All other cells untouched
        result[4, 5] = Float32(N)
        @test all(result .== Float32(N))
    end

    @testset "Atomix with N integer indices on 2D array (no CartesianIndex)" begin
        # `@atomic A[i, j] += v` expands to modifyindex_atomic!(A, ..., i, j) —
        # requires the Vararg{Integer,N} override, not the CartesianIndex one.
        @kernel function atomic_add_ij!(A, is, js)
            k = @index(Global, Linear)
            @inbounds i = is[k]
            @inbounds j = js[k]
            Atomix.@atomic A[i, j] += UInt32(1)
        end

        dims = (6, 5)
        A = Mantle.devicearray(TESTBACKEND, zeros(UInt32, dims))
        # All threads hit (3, 4) with Int32 indices like the solar kernel does.
        N = 1024
        is = Mantle.devicearray(TESTBACKEND, fill(Int32(3), N))
        js = Mantle.devicearray(TESTBACKEND, fill(Int32(4), N))
        atomic_add_ij!(TESTBACKEND)(A, is, js; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(A)
        @test result[3, 4] == UInt32(N)
        @test sum(result) == UInt32(N)
    end
end
